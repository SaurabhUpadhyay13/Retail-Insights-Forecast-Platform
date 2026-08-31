"""
03_export_for_powerbi.py
----------------------------------------------------------------------------
Reshapes pipeline outputs into a small, clean star-schema that Power BI
likes: one fact table + one date dimension + one rollup KPI table.
Power BI's own "Refresh" (manual, scheduled, or via Power BI Gateway if
the source is on-prem) is what makes this "refreshed daily" in production
-- see README for exact click-path.
----------------------------------------------------------------------------
"""

import pandas as pd
import numpy as np
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT_DIR / "outputs"

OUT_DIR.mkdir(parents=True, exist_ok=True)

forecast_path = OUT_DIR / "daily_forecast.csv"
anomaly_path = OUT_DIR / "anomaly_flags.csv"


def save_csv_atomic(df: pd.DataFrame, output_path: Path) -> None:
    """Replace a CSV only after its complete successor is safely written."""
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        df.to_csv(temporary_path, index=False)
        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing pipeline output: {path}. Run 02_streaming_pipeline.py first."
        )


def require_columns(df: pd.DataFrame, columns: list[str], source: Path) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise KeyError(f"{source.name} is missing columns: {', '.join(missing)}")


require_file(forecast_path)
require_file(anomaly_path)

forecast = pd.read_csv(forecast_path, parse_dates=["date"])
anomaly = pd.read_csv(anomaly_path, parse_dates=["date"])
require_columns(
    forecast,
    [
        "date", "category", "actual_units", "revenue",
        "online_model_forecast", "prophet_forecast",
    ],
    forecast_path,
)
require_columns(
    anomaly,
    [
        "date", "category", "z_score", "z_score_flag",
        "isolation_forest_flag", "any_anomaly", "anomaly_type",
    ],
    anomaly_path,
)
if forecast.empty or anomaly.empty:
    raise ValueError("Forecast and anomaly outputs must both contain rows.")
for frame, source in [(forecast, forecast_path), (anomaly, anomaly_path)]:
    if frame["date"].isna().any() or frame["category"].isna().any():
        raise ValueError(f"{source.name} contains null date/category keys.")
    if frame.duplicated(["date", "category"]).any():
        raise ValueError(f"{source.name} contains duplicate date/category keys.")

forecast_keys = pd.MultiIndex.from_frame(forecast[["date", "category"]])
anomaly_keys = pd.MultiIndex.from_frame(anomaly[["date", "category"]])
forecast_only = forecast_keys.difference(anomaly_keys)
anomaly_only = anomaly_keys.difference(forecast_keys)
if len(forecast_only) or len(anomaly_only):
    raise ValueError(
        "Forecast and anomaly outputs have different date/category keys "
        f"({len(forecast_only)} forecast-only, {len(anomaly_only)} anomaly-only)."
    )

# ---------------------------------------------------------------------
# FACT TABLE: one row per (date, category) with forecast + anomaly cols
# merged, so Power BI only needs to import ONE table for the main visuals
# ---------------------------------------------------------------------
fact = forecast.merge(
    anomaly[["date", "category", "z_score", "z_score_flag",
             "isolation_forest_flag", "any_anomaly", "anomaly_type"]],
    on=["date", "category"], how="inner", validate="one_to_one"
)
fact = fact.sort_values(["category", "date"]).reset_index(drop=True)
# A weekly seasonal baseline is both a production fallback and a necessary
# benchmark. The current backtest shows it beating the fitted models, so it is
# the honest champion until monitored performance says otherwise.
fact["seasonal_naive_forecast"] = fact.groupby("category")["actual_units"].shift(7)
fact["production_forecast"] = (
    fact["seasonal_naive_forecast"]
    .fillna(fact["prophet_forecast"])
    .fillna(fact["online_model_forecast"])
)
fact["production_model"] = np.select(
    [
        fact["seasonal_naive_forecast"].notna(),
        fact["prophet_forecast"].notna(),
        fact["online_model_forecast"].notna(),
    ],
    ["seasonal_naive_7d", "prophet", "online_sgd"],
    default="warmup_unavailable",
)
fact["forecast_error_online"] = (fact["actual_units"] - fact["online_model_forecast"])
fact["forecast_error_prophet"] = (fact["actual_units"] - fact["prophet_forecast"])
fact["forecast_error_seasonal_naive"] = (
    fact["actual_units"] - fact["seasonal_naive_forecast"]
)
fact["forecast_error_production"] = (
    fact["actual_units"] - fact["production_forecast"]
)
fact["forecast_error_pct_prophet"] = np.where(
    fact["actual_units"] > 0,
    (fact["forecast_error_prophet"] / fact["actual_units"]) * 100,
    np.nan
)
save_csv_atomic(fact, OUT_DIR / "fact_forecast_anomaly.csv")

# ---------------------------------------------------------------------
# MODEL SCORECARD: proves whether complex models beat a simple baseline.
# WAPE is stable in the presence of zero-demand days where MAPE is undefined.
# ---------------------------------------------------------------------
model_columns = {
    "online_sgd": "online_model_forecast",
    "prophet": "prophet_forecast",
    "seasonal_naive_7d": "seasonal_naive_forecast",
    "production_selection": "production_forecast",
}
performance_rows = []
for model_name, forecast_column in model_columns.items():
    for scope, category, group in [
        ("overall", "ALL", fact),
        *[("category", str(name), frame) for name, frame in fact.groupby("category")],
    ]:
        evaluated = group.loc[
            group[forecast_column].notna(), ["actual_units", forecast_column]
        ]
        if evaluated.empty:
            continue
        errors = evaluated["actual_units"] - evaluated[forecast_column]
        absolute_errors = errors.abs()
        actual_total = evaluated["actual_units"].abs().sum()
        performance_rows.append(
            {
                "scope": scope,
                "category": category,
                "model": model_name,
                "observations": len(evaluated),
                "mae": round(float(absolute_errors.mean()), 4),
                "rmse": round(float(np.sqrt(np.mean(np.square(errors)))), 4),
                "wape_pct": round(
                    float(absolute_errors.sum() / actual_total * 100), 4
                ) if actual_total else np.nan,
            }
        )
model_performance = pd.DataFrame(performance_rows)
save_csv_atomic(model_performance, OUT_DIR / "model_performance.csv")

# ---------------------------------------------------------------------
# DATE DIMENSION: standard Power BI date table for slicers / time intel
# ---------------------------------------------------------------------
date_dim = pd.DataFrame({"date": pd.date_range(fact["date"].min(), fact["date"].max())})
date_dim["year"] = date_dim["date"].dt.year
date_dim["month"] = date_dim["date"].dt.month
date_dim["month_name"] = date_dim["date"].dt.strftime("%b")
date_dim["week_of_year"] = date_dim["date"].dt.isocalendar().week
date_dim["day_name"] = date_dim["date"].dt.strftime("%a")
date_dim["is_weekend"] = date_dim["date"].dt.dayofweek >= 5
save_csv_atomic(date_dim, OUT_DIR / "dim_date.csv")

# ---------------------------------------------------------------------
# KPI ROLLUP: one row per category, headline numbers for card visuals
# ---------------------------------------------------------------------
kpi = (
    fact.groupby("category")
    .agg(
        total_units=("actual_units", "sum"),
        total_revenue=("revenue", "sum"),
        avg_daily_units=("actual_units", "mean"),
        anomaly_days=("any_anomaly", "sum"),
        total_days=("actual_units", "count"),
        avg_prophet_mae=("forecast_error_prophet", lambda s: s.abs().mean()),
        avg_production_mae=("forecast_error_production", lambda s: s.abs().mean()),
    )
    .reset_index()
)
kpi["anomaly_rate_pct"] = (kpi["anomaly_days"] / kpi["total_days"] * 100).round(1)
save_csv_atomic(kpi, OUT_DIR / "kpi_summary_by_category.csv")

print("Power BI-ready files written:")
for f in [
    "fact_forecast_anomaly.csv", "dim_date.csv",
    "kpi_summary_by_category.csv", "model_performance.csv",
]:
    n = len(pd.read_csv(OUT_DIR / f))
    print(f"  {f}  ({n:,} rows)")
