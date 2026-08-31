"""
02_streaming_pipeline.py
----------------------------------------------------------------------------
THE CORE CONSTRAINT: we are NOT allowed to do model.fit(full_history) once.
Data arrives in daily batches, as it would from a live POS / e-commerce feed.
The pipeline must update its state and models incrementally as each new
day's batch lands, and produce a forecast + anomaly flag for that day using
only data seen so far (no peeking at the future).

TWO MODELS ARE MAINTAINED SIDE BY SIDE, on purpose, to show the trade-off:

  1. ONLINE MODEL (SGDRegressor, true incremental learning)
     - Updated with .partial_fit() on every single new daily batch.
     - Never retrains from scratch. Memory footprint is constant.
     - This is the model that actually satisfies "incrementally-updating"
       in the strict sense.

  2. PERIODIC-RETRAIN MODEL (Prophet, industry-standard forecasting)
     - Prophet does not support partial_fit / online learning.
     - So it is retrained periodically on a bounded rolling window, not every
       batch. This controls runtime and adapts to recent demand drift.
     - This mirrors how most real forecasting teams actually handle
       "streaming" constraints in production: cheap online updates for
       every batch, expensive full retrains on a schedule.

ANOMALY DETECTION, similarly, uses two complementary online-friendly methods:
  - Rolling z-score per category (fully online: running mean/std, O(1)
    update per batch, no retraining ever needed)
  - Isolation Forest, refit periodically (like Prophet above) on the
    accumulated multivariate feature set (units, revenue, price, day-of-week)
    to catch anomalies that a univariate z-score would miss
----------------------------------------------------------------------------
"""

import numpy as np
import pandas as pd
from pathlib import Path
import os
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import IsolationForest
import logging

try:
    from prophet import Prophet
except ImportError:  # pragma: no cover - graceful fallback if Prophet is absent
    Prophet = None
if os.getenv("RETAIL_DISABLE_PROPHET", "").strip().lower() in {"1", "true", "yes"}:
    Prophet = None

logging.getLogger("cmdstanpy").disabled = True
logging.getLogger("prophet").setLevel(logging.WARNING)

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
OUT_DIR = ROOT_DIR / "outputs"

CLEAN_DATA_CANDIDATES = [
    DATA_DIR / "retail_sales_cleaned.csv",
    DATA_DIR / "retail_sales_cleaned.xlsx",
]

OUT_FORECAST = OUT_DIR / "daily_forecast.csv"
OUT_ANOMALY = OUT_DIR / "anomaly_flags.csv"
OUT_MODEL_LOG = OUT_DIR / "model_log.csv"

MIN_DAYS_BEFORE_FORECASTING = 21   # need a minimum history before predicting
REFIT_EVERY_N_DAYS = int(os.getenv("RETAIL_REFIT_EVERY_N_DAYS", "28"))
MAX_REFIT_HISTORY_DAYS = int(os.getenv("RETAIL_REFIT_HISTORY_DAYS", "365"))
Z_SCORE_THRESHOLD = 2.5            # ~ same sensitivity as a 99% CI flag
IF_CONTAMINATION = 0.05            # assume ~5% of days are anomalous

if REFIT_EVERY_N_DAYS <= 0 or MAX_REFIT_HISTORY_DAYS < MIN_DAYS_BEFORE_FORECASTING:
    raise ValueError(
        "RETAIL_REFIT_EVERY_N_DAYS must be positive and "
        "RETAIL_REFIT_HISTORY_DAYS must cover the forecast warm-up period."
    )


def save_csv_atomic(df: pd.DataFrame, output_path: Path) -> None:
    """Replace a CSV only after its complete successor is safely written."""
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        df.to_csv(temporary_path, index=False)
        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def load_clean_data() -> pd.DataFrame:
    """Load and identify the cleaned dataset used by this run."""
    for path in CLEAN_DATA_CANDIDATES:
        if path.exists():
            try:
                if path.suffix.lower() in {".xlsx", ".xls"}:
                    data = pd.read_excel(path)
                else:
                    data = pd.read_csv(path, low_memory=False)
                print(f"Loaded cleaned data: {path} ({len(data):,} rows)")
                return data
            except Exception as exc:
                print(f"WARNING: could not read {path}: {exc}")
                continue
    expected = ", ".join(str(p) for p in CLEAN_DATA_CANDIDATES)
    raise FileNotFoundError(
        f"Could not find a cleaned dataset. Expected one of: {expected}"
    )


def require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise KeyError(
            "The cleaned dataset is missing required column(s): "
            + ", ".join(missing)
        )

# ---------------------------------------------------------------------
# Build the daily-aggregated series per category. In a real streaming
# system this aggregation itself would happen inside each day's batch
# job; here we build it once, then FEED it to the pipeline one day at a
# time to simulate the batch arrival.
# ---------------------------------------------------------------------
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = load_clean_data()
require_columns(df, ["transaction_date", "category", "quantity"])
df = df.copy()
df["transaction_date"] = pd.to_datetime(df["transaction_date"], errors="coerce")
df["category"] = df["category"].astype("string").str.strip()
df["category"] = df["category"].mask(df["category"].eq(""))
df = df.dropna(subset=["transaction_date", "category"])

for col in [
    "quantity", "revenue", "total_spent", "selling_price",
    "price_per_unit", "cost_price",
]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.loc[df["quantity"].notna() & (df["quantity"] >= 0)].copy()
if df.empty:
    raise ValueError("No valid dated sales rows remain in the cleaned dataset.")

if "transaction_id" not in df.columns:
    df["transaction_id"] = np.arange(len(df))

revenue_source = "total_spent" if "total_spent" in df.columns else None
price_source = next(
    (c for c in ["selling_price", "price_per_unit", "cost_price"] if c in df.columns),
    None,
)
if revenue_source is None and price_source is None:
    raise KeyError(
        "The cleaned dataset needs either 'total_spent' or one of "
        "'selling_price' / 'price_per_unit' for the streaming pipeline."
    )

if revenue_source is None:
    df["revenue_value"] = df[price_source] * df["quantity"]
    revenue_source = "revenue_value"

# A missing monetary value must not silently become zero during groupby.sum(),
# because that manufactures valid-looking revenue and corrupts anomaly inputs.
valid_revenue = df[revenue_source].notna() & np.isfinite(df[revenue_source])
invalid_revenue_rows = int((~valid_revenue).sum())
if invalid_revenue_rows:
    print(f"WARNING: dropping {invalid_revenue_rows:,} row(s) with invalid revenue")
    df = df.loc[valid_revenue].copy()
if df.empty:
    raise ValueError("No rows with valid revenue remain in the cleaned dataset.")

# Isolation Forest needs a meaningful price feature. If the source has only
# line revenue, derive its effective unit price instead of using quantity as
# the old fallback did.
if price_source is not None:
    df["unit_price_value"] = df[price_source]
else:
    df["unit_price_value"] = np.where(
        df["quantity"] > 0,
        df[revenue_source] / df["quantity"],
        np.nan,
    )

daily = (
    df.groupby([pd.Grouper(key="transaction_date", freq="D"), "category"])
    .agg(
        units_sold=("quantity", "sum"),
        revenue=(revenue_source, "sum"),
        avg_price=("unit_price_value", "mean"),
        n_transactions=("quantity", "size"),
    )
    .reset_index()
    .rename(columns={"transaction_date": "date"})
)

# Fill gaps inside each category's observed active period. Extending every
# category across the global range would invent zero demand before a product
# line launched or after it was discontinued.
all_categories = daily["category"].dropna().unique()
calendar_frames = []
for category, category_daily in daily.groupby("category", sort=False):
    category_daily = category_daily.set_index("date").drop(columns="category")
    category_dates = pd.date_range(
        category_daily.index.min(), category_daily.index.max(), freq="D"
    )
    category_daily = category_daily.reindex(category_dates, fill_value=0)
    category_daily.index.name = "date"
    category_daily["category"] = category
    calendar_frames.append(category_daily.reset_index())
daily = pd.concat(calendar_frames, ignore_index=True)
if daily.empty:
    raise ValueError("Daily aggregation produced no category/date rows.")
daily["avg_price"] = pd.to_numeric(daily["avg_price"], errors="coerce")
derived_avg_price = np.where(
    daily["units_sold"] > 0,
    daily["revenue"] / daily["units_sold"],
    0.0,
)
daily["avg_price"] = daily["avg_price"].fillna(
    pd.Series(derived_avg_price, index=daily.index)
)
if not np.isfinite(daily[["units_sold", "revenue", "avg_price"]].to_numpy()).all():
    raise ValueError("Daily model features contain non-finite numeric values.")
daily["day_of_week"] = daily["date"].dt.dayofweek
daily["is_weekend"] = (daily["day_of_week"] >= 5).astype(int)
daily = daily.sort_values(["category", "date"]).reset_index(drop=True)

print(f"Simulating streaming arrival for {daily['date'].nunique()} calendar days "
      f"across {len(all_categories)} category active periods = {len(daily)} rows")
if Prophet is None:
    print("Prophet is not installed; Prophet forecasts will be skipped.")

# ---------------------------------------------------------------------
# Per-category incremental state
# ---------------------------------------------------------------------
class CategoryState:
    """Holds everything that must persist across daily batches for one category."""
    def __init__(self, name):
        self.name = name
        self.history = []              # list of dicts, the expanding window
        self.online_model = SGDRegressor(max_iter=1, learning_rate="invscaling",
                                          eta0=0.01, warm_start=True, random_state=42)
        self.prophet_model = None
        self.if_model = None
        self.last_refit_day = -999
        # running stats for O(1) z-score anomaly detection
        self.running_mean = 0.0
        self.running_m2 = 0.0          # for Welford's online variance
        self.running_n = 0

    def update_running_stats(self, value):
        """Welford's online algorithm: update mean/variance with a single new point."""
        self.running_n += 1
        delta = value - self.running_mean
        self.running_mean += delta / self.running_n
        delta2 = value - self.running_mean
        self.running_m2 += delta * delta2

    def running_std(self):
        if self.running_n < 2:
            return 0.0
        return np.sqrt(self.running_m2 / (self.running_n - 1))

    def make_features(self, day_index, day_of_week, is_weekend):
        # Use a fixed transform. Incrementally changing a StandardScaler would
        # move the coordinate system underneath coefficients learned on prior
        # days and make the online model internally inconsistent.
        angle = 2 * np.pi * day_of_week / 7
        return np.array([[
            day_index / 365.25,
            np.sin(angle),
            np.cos(angle),
            is_weekend,
        ]], dtype=float)


states = {cat: CategoryState(cat) for cat in all_categories}

forecast_rows = []
anomaly_rows = []
model_log_rows = []

day_counter = {cat: 0 for cat in all_categories}

# ---------------------------------------------------------------------
# MAIN STREAMING LOOP — one iteration = one day's batch arriving
# ---------------------------------------------------------------------
for date, day_df in daily.groupby("date"):
    for _, row in day_df.iterrows():
        cat = row["category"]
        state = states[cat]
        d_idx = day_counter[cat]
        actual_units = row["units_sold"]

        # ---- (a) FORECAST using models trained on data seen so far ONLY ----
        online_pred, prophet_pred = np.nan, np.nan
        if d_idx >= MIN_DAYS_BEFORE_FORECASTING:
            feats = state.make_features(d_idx, row["day_of_week"], row["is_weekend"])
            online_pred = max(0, state.online_model.predict(feats)[0])

            if Prophet is not None and state.prophet_model is not None:
                future = pd.DataFrame({"ds": [date]})
                prophet_pred = max(0, state.prophet_model.predict(future)["yhat"].iloc[0])

        forecast_rows.append({
            "date": date, "category": cat, "actual_units": actual_units,
            "online_model_forecast": round(online_pred, 2) if not np.isnan(online_pred) else np.nan,
            "prophet_forecast": round(prophet_pred, 2) if not np.isnan(prophet_pred) else np.nan,
            "revenue": row["revenue"], "n_transactions": row["n_transactions"],
        })

        # ---- (b) ANOMALY CHECK using stats/models built on PAST days only ----
        z_score = np.nan
        z_flag = False
        if_flag = False
        if state.running_n >= 10:
            std = state.running_std()
            if std > 0:
                z_score = (actual_units - state.running_mean) / std
                z_flag = abs(z_score) > Z_SCORE_THRESHOLD

        if state.if_model is not None:
            if_feats = np.array([[actual_units, row["revenue"], row["avg_price"],
                                   row["day_of_week"]]])
            if_pred = state.if_model.predict(if_feats)[0]     # -1 = anomaly, 1 = normal
            if_flag = if_pred == -1

        anomaly_type = []
        if z_flag:
            anomaly_type.append(
                "low_demand_zscore" if z_score < 0 else "demand_spike_zscore"
            )
        if if_flag:
            anomaly_type.append("multivariate_isolation_forest")

        anomaly_rows.append({
            "date": date, "category": cat, "actual_units": actual_units,
            "z_score": round(z_score, 2) if not np.isnan(z_score) else np.nan,
            "z_score_flag": z_flag, "isolation_forest_flag": if_flag,
            "any_anomaly": z_flag or if_flag,
            "anomaly_type": ";".join(anomaly_type) if anomaly_type else "",
        })

        # ---- (c) UPDATE STATE with this day's now-known actuals ----
        state.update_running_stats(actual_units)
        state.history.append({
            "ds": date, "y": actual_units, "day_idx": d_idx,
            "day_of_week": row["day_of_week"], "is_weekend": row["is_weekend"],
            "revenue": row["revenue"], "avg_price": row["avg_price"],
        })

        # (c1) true incremental update — happens on EVERY batch
        feats = state.make_features(d_idx, row["day_of_week"], row["is_weekend"])
        state.online_model.partial_fit(feats, [actual_units])

        # (c2) periodic full retrain — happens every REFIT_EVERY_N_DAYS
        did_refit = False
        if (d_idx >= MIN_DAYS_BEFORE_FORECASTING and
                (d_idx - state.last_refit_day) >= REFIT_EVERY_N_DAYS):
            # A bounded rolling window controls training time and adapts more
            # quickly to current demand than repeatedly fitting all history.
            hist_df = pd.DataFrame(state.history[-MAX_REFIT_HISTORY_DAYS:])

            # Prophet retrain on the bounded recent window
            if Prophet is not None:
                m = Prophet(daily_seasonality=False, weekly_seasonality=True,
                            yearly_seasonality=False, interval_width=0.9)
                m.fit(hist_df[["ds", "y"]])
                state.prophet_model = m

            # Isolation Forest retrain on the same bounded recent window
            if_train = hist_df[["y", "revenue", "avg_price", "day_of_week"]].rename(
                columns={"y": "units_sold"})
            if len(if_train) >= 15:
                iso = IsolationForest(contamination=IF_CONTAMINATION, random_state=42)
                iso.fit(if_train.values)
                state.if_model = iso

            state.last_refit_day = d_idx
            did_refit = True

        model_log_rows.append({
            "date": date, "category": cat, "day_index": d_idx,
            "history_size": len(state.history), "refit_triggered": did_refit,
        })

        day_counter[cat] += 1

forecast_df = pd.DataFrame(forecast_rows)
anomaly_df = pd.DataFrame(anomaly_rows)
model_log_df = pd.DataFrame(model_log_rows)

save_csv_atomic(forecast_df, OUT_FORECAST)
save_csv_atomic(anomaly_df, OUT_ANOMALY)
save_csv_atomic(model_log_df, OUT_MODEL_LOG)

print(f"\nDone. {forecast_df['online_model_forecast'].notna().sum()} online forecasts, "
      f"{forecast_df['prophet_forecast'].notna().sum()} Prophet forecasts produced.")
print(f"Flagged {anomaly_df['any_anomaly'].sum()} anomalous category-days "
      f"out of {len(anomaly_df)} ({anomaly_df['any_anomaly'].mean():.1%})")
print(f"\nSaved:\n  {OUT_FORECAST}\n  {OUT_ANOMALY}\n  {OUT_MODEL_LOG}")
