"""
04_daily_refresh_job.py
----------------------------------------------------------------------------
This is what you would actually schedule (cron / Windows Task Scheduler /
Azure Data Factory / GitHub Actions) to run once per day in production:

    0 6 * * *  python3 04_daily_refresh_job.py

It does NOT re-run the whole streaming simulation. In production there is
no "simulation" -- there's just today's new batch of POS/e-commerce data
landing in a folder or database. This script:
    1. Loads yesterday's saved incremental model state (in this demo,
       re-derives it quickly from history; in production you'd pickle
       state.online_model / state.prophet_model / state.if_model per
       category and load them here instead of retraining)
    2. Appends the new day's transactions
    3. Produces one new day's forecast + anomaly row
    4. Appends it to the fact table Power BI reads
    5. Power BI's scheduled refresh (configured separately, see README)
       then picks up the updated CSV/table automatically

For this portfolio project, run 02_streaming_pipeline.py once to build the
full backtest history + demonstrate the incremental-learning mechanics end
to end. Use this script as evidence you understand how it plugs into a
real daily-batch production schedule.
----------------------------------------------------------------------------
"""
import pandas as pd
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
FACT_PATH = ROOT_DIR / "outputs" / "fact_forecast_anomaly.csv"

def run_daily_job(new_batch_date: str):
    """
    Pseudocode-level illustration of the production daily job.
    In production, replace the placeholder steps with:
      - pull new_batch_date's transactions from the POS/e-commerce DB
      - clean with the same logic as 01_data_cleaning.py
      - load pickled per-category model state from blob storage / disk
      - state.online_model.partial_fit(...) on just this one day
      - every 7th day: refit Prophet + IsolationForest, re-pickle state
      - append the resulting forecast + anomaly row to the fact table
      - Power BI's scheduled refresh (or DirectQuery) reflects it same day
    """
    if not FACT_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {FACT_PATH}. Run 03_export_for_powerbi.py first."
        )

    fact = pd.read_csv(FACT_PATH, parse_dates=["date"])
    print(f"Fact table currently has {len(fact):,} rows through "
          f"{fact['date'].max().date()}.")
    print(f"Next scheduled batch would process: {new_batch_date}")
    print("(See 02_streaming_pipeline.py for the actual incremental-update "
          "logic this job would call per category.)")

if __name__ == "__main__":
    run_daily_job(new_batch_date="2024-08-28")
