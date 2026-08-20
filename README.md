# Retail Demand Forecasting + Anomaly Alert Dashboard

A portfolio project that simulates a real retail/e-commerce pipeline:
raw sales data -> cleaning -> an **incrementally-updating** forecasting model
(not a one-shot `.fit()`) -> daily anomaly flags -> a Power BI dashboard.

Run the whole project from one entry point:

```bash
python main.py
```

## Why this project maps to hiring needs

- Demand forecasting is the backbone of retail inventory planning.
- Anomaly detection catches stockouts, fraud spikes, or POS system failures
  before they cost money.
- The streaming-batch constraint mirrors how real systems work: data lands
  daily, and models have to update incrementally rather than being retrained
  from scratch every time.

---

## 1. Dataset

Put your raw sales files in `data/`. The cleaner accepts multiple source
schemas as long as the columns can be standardized by
`scripts/column_standardizer.py`.

Example source names the standardizer understands:

- `Order_ID` / `Invoice ID` / `Transaction ID`
- `Product_Line` / `Category`
- `Units_Sold` / `Quantity`
- `MRP` / `Price Per Unit`
- `Order_Date` / `Transaction Date`
- `Sales_Channel` / `Payment Method`

The cleaned outputs are written to:

- `data/retail_sales_cleaned.csv`
- `data/retail_sales_cleaned.xlsx`

---

## 2. Tools used

| Tool | Purpose |
|---|---|
| `pandas` / `numpy` | data cleaning, aggregation, feature engineering |
| `scikit-learn` - `SGDRegressor` | true online/incremental forecasting model (`partial_fit`) |
| `scikit-learn` - `IsolationForest` | multivariate anomaly detection, refit periodically |
| `Prophet` | forecasting model retrained on an expanding window on a schedule |
| Power BI Desktop | dashboard: trend charts, anomaly table, KPI cards |

---

## 3. Project Log

This is a running development journal for portfolio and case-study use.
We will keep adding dated entries here as the project evolves.

### 2026-08-18

- Started a structured project log in the README so the work can later be
  turned into a portfolio case study.
- Updated the data-cleaning pipeline to make missing-value handling and
  outlier handling more auditable.
- Added explicit `*_was_imputed` flags for filled fields so every imputed
  value can be traced later.
- Switched numeric imputation to category-level medians with a global median
  fallback.
- Changed `total_spent` to be recomputed only when missing instead of being
  overwritten blindly.
- Updated outlier handling to cap values with the IQR method per category
  instead of deleting transactions.
- Added logging for missing-value percentages, critical row drops, and
  capped values so the cleaning decisions stay visible.

### 2026-08-19

- Clarified that `total_spent` is now rebuilt for every eligible row using the
  available price field and `quantity`, rather than only backfilling missing
  values.
- Kept the category-level IQR pass after recomputing totals so the final
  cleaned data still gets winsorized for extreme values.
- Confirmed the terminal logs now include the recomputation count and the
  outlier-capping counts during cleaning runs.

### How to keep updating this log

- Add one dated entry per work session.
- Record what changed, why it changed, and any tradeoffs.
- Keep entries short but specific enough that they can later become resume
  bullets or case-study talking points.

---

## 4. Pipeline

```text
main.py
scripts/
  01_data_cleaning.py        # standardize, clean, impute, validate
  02_streaming_pipeline.py   # daily batch simulation + incremental models
  03_export_for_powerbi.py   # reshape into Power BI-friendly tables
  04_daily_refresh_job.py    # example daily refresh entry point
```

### One-command run

```bash
pip install -r requirements.txt
python main.py
```

`main.py` runs the three production-build stages in order: clean data,
generate forecasts/anomaly flags, then rebuild the Power BI tables. The
`04_daily_refresh_job.py` file is a separate production handoff example and
is not part of this historical rebuild command. The automated run writes the
cleaned CSV used by forecasting and skips the redundant million-row Excel
export. Running `01_data_cleaning.py` directly still writes both formats.

### Manual run

If you want to run each stage yourself:

```bash
python scripts/01_data_cleaning.py
python scripts/02_streaming_pipeline.py
python scripts/03_export_for_powerbi.py
```

To inspect the separate daily-refresh handoff example, run:

```bash
python scripts/04_daily_refresh_job.py
```

### Step 1 - Data cleaning (`01_data_cleaning.py`)

- Drops exact duplicate rows
- Normalizes text casing and whitespace
- Coerces numeric columns stored as strings
- Prints missing-value percentages before cleaning starts
- Drops rows missing critical `category` or `transaction_date` values
- Drops physically impossible rows like negative quantity or price values
- Imputes missing numeric values using category medians, with global
  fallback if needed
- Adds audit columns like `price_per_unit_was_imputed`
- Fills categorical fields like `payment_method` and `location` with
  explicit `Unknown` values
- Recomputes `total_spent` for every eligible row using the available price
  field and `quantity`, then flags rows that were rebuilt
- Winsorizes outliers per category using the IQR method instead of deleting
  transactions
- Parses and sorts by `transaction_date`
- Saves both CSV and Excel versions of the cleaned dataset

### Cleaning logs

When you run `python scripts/01_data_cleaning.py` or `python main.py`, the
terminal should show:

- Missing-value percentages before cleaning
- Rows removed for missing critical fields
- Rows removed for physically impossible numeric values
- IQR capping counts for numeric columns
- A recomputation count for `total_spent`
- Final row counts and file-save locations

### Step 2 - Streaming/incremental core (`02_streaming_pipeline.py`)

This script does not call `model.fit(full_history)` once. Instead it loops
day by day and for each day, for each category:

1. Forecasts using only data seen before that day
2. Checks for anomalies using only past data
3. Updates the models with that day's actuals

Two forecasting models run side by side:

| Model | How it updates | Why |
|---|---|---|
| `SGDRegressor` | `.partial_fit()` on every daily batch | True incremental learning with constant memory |
| `Prophet` | Full refit every 7 days on the expanding window | Prophet has no online-learning API |

Anomaly detection also uses two approaches:

- Rolling z-score per category using Welford's algorithm
- Isolation Forest refit every 7 days on the accumulated feature set

Outputs:

- `outputs/daily_forecast.csv`
- `outputs/anomaly_flags.csv`
- `outputs/model_log.csv`

If `prophet` is not installed, the script still runs the online model and
anomaly detection path, but Prophet forecasts are skipped.

### Step 3 - Power BI export (`03_export_for_powerbi.py`)

Creates a small star schema for Power BI:

- `outputs/fact_forecast_anomaly.csv`
- `outputs/dim_date.csv`
- `outputs/kpi_summary_by_category.csv`

### Step 4 - Daily refresh (`04_daily_refresh_job.py`)

This is a lightweight example of the job that would run once per day in
production. It points to the exported fact table and illustrates where the
daily batch refresh would plug in.

---

## 5. Power BI dashboard

1. Open Power BI Desktop.
2. Import the CSV files from `outputs/`.
3. Link `dim_date[date]` to `fact_forecast_anomaly[date]`.
4. Build visuals:
   - Line chart: `actual_units` and `prophet_forecast`
   - Alert table: rows where `any_anomaly = TRUE`
   - KPI cards: from `kpi_summary_by_category`
   - Slicers: `category`, `date`, `any_anomaly`
5. Use conditional formatting on `z_score` to make spikes stand out.

---

