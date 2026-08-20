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
    if frame.duplicated(["date", "category"]).any():
        raise ValueError(f"{source.name} contains duplicate date/category keys.")

# ---------------------------------------------------------------------
# FACT TABLE: one row per (date, category) with forecast + anomaly cols
# merged, so Power BI only needs to import ONE table for the main visuals
# ---------------------------------------------------------------------
fact = forecast.merge(
    anomaly[["date", "category", "z_score", "z_score_flag",
             "isolation_forest_flag", "any_anomaly", "anomaly_type"]],
    on=["date", "category"], how="left", validate="one_to_one"
)
fact["forecast_error_online"] = (fact["actual_units"] - fact["online_model_forecast"])
fact["forecast_error_prophet"] = (fact["actual_units"] - fact["prophet_forecast"])
fact["forecast_error_pct_prophet"] = np.where(
    fact["actual_units"] > 0,
    (fact["forecast_error_prophet"] / fact["actual_units"]) * 100,
    np.nan
)
fact.to_csv(OUT_DIR / "fact_forecast_anomaly.csv", index=False)

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
date_dim.to_csv(OUT_DIR / "dim_date.csv", index=False)

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
    )
    .reset_index()
)
kpi["anomaly_rate_pct"] = (kpi["anomaly_days"] / kpi["total_days"] * 100).round(1)
kpi.to_csv(OUT_DIR / "kpi_summary_by_category.csv", index=False)

print("Power BI-ready files written:")
for f in ["fact_forecast_anomaly.csv", "dim_date.csv", "kpi_summary_by_category.csv"]:
    n = len(pd.read_csv(OUT_DIR / f))
    print(f"  {f}  ({n:,} rows)")
