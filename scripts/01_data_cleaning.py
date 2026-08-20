"""
01_data_cleaning.py
----------------------------------------------------------------------------
Reads every raw sales file in DATA_FOLDER (each file is allowed to use
different column names/casing -- e.g. "Order ID" vs "Invoice_Id" vs
"Transaction ID"), standardizes each file's columns to one canonical
schema using column_standardizer.py, cleans each file, concatenates
everything into one combined dataframe, and writes a single cleaned
Excel file.

WHY PER-FILE CLEANING, THEN COMBINE (not: combine first, clean once):
Different source files can have wildly different missingness patterns and
even different price fields (one file might have price_per_unit, another
only cost_price/selling_price). Imputing per-file (e.g. filling a missing
price with that FILE's category median) is more honest than pooling
everything first -- pooling first would let one file's price distribution
leak into another file's imputation.

FOLDER LAYOUT EXPECTED:
    DATA_FOLDER/
        store_a_sales.csv
        store_b_sales.csv
        store_c_sales.xlsx
        ... any number of .csv / .xlsx / .xls files

Requires: pandas, numpy, openpyxl (for writing .xlsx)
    pip install openpyxl
----------------------------------------------------------------------------
"""

import glob
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from column_standardizer import COLUMN_ALIASES, standardize_columns

# ---------------------------------------------------------------------
# CONFIG -- adjust these two paths for your machine
# ---------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_FOLDER = ROOT_DIR / "data"
OUTPUT_XLSX = DATA_FOLDER / "retail_sales_cleaned.xlsx"
OUTPUT_CSV = DATA_FOLDER / "retail_sales_cleaned.csv"
AUDIT_LOG = DATA_FOLDER / "data_cleaning_audit_log.txt"

SUPPORTED_EXTENSIONS = (".csv", ".xlsx", ".xls")

# Fields the cleaning logic below knows how to handle if present.
# Any file missing some of these just skips that step -- nothing breaks.
TEXT_COLUMNS = ["category", "item", "product_name", "brand",
                 "payment_method", "location", "gender", "channel",
                 "city", "state", "region"]
NUMERIC_COLUMNS = ["price_per_unit", "cost_price", "selling_price",
                    "quantity", "total_spent", "discount_applied"]
PRICE_COLUMNS = ["price_per_unit", "cost_price", "selling_price"]
IMPUTE_COLUMNS = ["quantity", *PRICE_COLUMNS]
CATEGORICAL_FILL_COLUMNS = [
    "payment_method", "location", "gender", "channel", "brand"
]
DATE_PART_COLUMNS = [
    "transaction_year", "transaction_month", "transaction_day",
    "transaction_day_name",
]
ALWAYS_OUTPUT_COLUMNS = [
    "transaction_id", "customer_id", "gender",
    "country", "region", "state", "city",
]
PER_FILE_MISSING_THRESHOLD = 0.25
OVERALL_MISSING_THRESHOLD = 0.40
EXCEL_MAX_DATA_ROWS = 1_048_575  # Excel limit minus the header row.


def load_raw_file(filepath: str) -> pd.DataFrame:
    """Read a single raw file (csv or excel) into a dataframe."""
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".csv":
        return pd.read_csv(filepath)
    elif ext in (".xlsx", ".xls"):
        return pd.read_excel(filepath)
    raise ValueError(f"Unsupported file type: {filepath}")


def to_numeric_currency(series: pd.Series) -> pd.Series:
    """Strip currency symbols/commas etc, then coerce to numeric."""
    cleaned = series.astype("string").str.replace(",", "", regex=False)
    number = cleaned.str.extract(
        r"([-+]?(?:\d+(?:\.\d*)?|\.\d+))", expand=False
    )
    return pd.to_numeric(number, errors="coerce")


def normalize_discount(series: pd.Series) -> pd.Series:
    """Convert numeric or yes/no discount values to non-negative numbers."""
    normalized = series.astype(str).str.strip().str.lower()
    flags = normalized.map({
        "true": 1.0, "false": 0.0, "yes": 1.0, "no": 0.0,
        "y": 1.0, "n": 0.0, "t": 1.0, "f": 0.0,
    })
    numeric = to_numeric_currency(series).abs()
    return numeric.fillna(flags)


def parse_transaction_dates(series: pd.Series) -> pd.Series:
    """Parse mixed separators while preserving explicit year-first dates."""
    values = series.astype("string").str.strip()
    values = values.replace({"": pd.NA, "nan": pd.NA, "none": pd.NA})
    year_first = values.str.match(r"^\d{4}[^0-9]+\d{1,2}[^0-9]+\d{1,2}", na=False)
    parsed = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")

    def parse_subset(subset: pd.Series, dayfirst: bool) -> pd.Series:
        try:
            return pd.to_datetime(
                subset,
                format="mixed",
                dayfirst=dayfirst,
                yearfirst=not dayfirst,
                errors="coerce",
            )
        except ValueError:  # pandas < 2.0 does not support format="mixed"
            return pd.to_datetime(
                subset,
                dayfirst=dayfirst,
                yearfirst=not dayfirst,
                errors="coerce",
            )

    parsed.loc[year_first] = parse_subset(values.loc[year_first], dayfirst=False)
    parsed.loc[~year_first] = parse_subset(values.loc[~year_first], dayfirst=True)
    return parsed


def normalize_gender(series: pd.Series) -> pd.Series:
    """Standardize common gender and audience labels."""
    values = series.astype("string").str.strip().str.lower()
    values = values.str.replace(r"[^a-z]+", "", regex=True)
    gender_map = {
        "m": "Male", "male": "Male", "man": "Male", "men": "Male",
        "f": "Female", "female": "Female", "woman": "Female",
        "women": "Female",
        "kid": "Kids", "kids": "Kids", "child": "Kids",
        "children": "Kids", "boy": "Kids", "boys": "Kids",
        "girl": "Kids", "girls": "Kids", "boysgirls": "Kids",
        "unisex": "Unisex", "mixed": "Unisex", "menwomen": "Unisex",
    }
    normalized = values.map(gender_map)
    return normalized.fillna(series.astype("string").str.strip().str.title())


def normalize_country(series: pd.Series) -> pd.Series:
    """Preserve country codes in uppercase and title-case country names."""
    values = series.astype("string").str.strip()
    values = values.mask(values.eq(""))
    is_country_code = values.str.fullmatch(r"[A-Za-z]{2,3}", na=False)
    return values.str.title().mask(is_country_code, values.str.upper())


def normalize_channel(series: pd.Series) -> pd.Series:
    """Normalize casing, spacing, and separators in channel labels."""
    values = series.astype("string").str.strip()
    values = values.str.replace(r"[^A-Za-z0-9]+", " ", regex=True)
    values = values.str.replace(r"\s+", " ", regex=True).str.strip()
    return values.mask(values.eq("")).str.title()


def print_missing_value_percentages(df: pd.DataFrame) -> None:
    """Print missingness before any rule-specific cleaning is applied."""
    print("\nMissing value percentages before cleaning:")
    for col in df.columns:
        print(f"  {col}: {df[col].isna().mean():.1%}")


def add_imputation_flag(df: pd.DataFrame, column: str, mask: pd.Series) -> None:
    """Mark rows where a field was filled by a cleaning rule."""
    flag_column = f"{column}_was_imputed"
    if flag_column not in df.columns:
        df[flag_column] = False
    df.loc[mask.fillna(False), flag_column] = True


def log_audit(message: str) -> None:
    """Write a cleaning audit line to the terminal and append it to disk."""
    print(message)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with AUDIT_LOG.open("a", encoding="utf-8") as handle:
        handle.write(f"[{timestamp}] {message}\n")


def drop_physically_impossible_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows with impossible numeric values before any capping logic.
    Negative or zero prices/quantities are treated as invalid records.
    """
    impossible_mask = pd.Series(False, index=df.index)
    for col in PRICE_COLUMNS:
        if col in df.columns:
            impossible_mask |= df[col].notna() & (df[col] <= 0)
    if "quantity" in df.columns:
        impossible_mask |= df["quantity"].notna() & (df["quantity"] <= 0)

    removed = int(impossible_mask.sum())
    if removed:
        print(f"  Removed {removed:,} row(s) with physically impossible values")
    return df.loc[~impossible_mask].copy()


def fill_categorical_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill selected categorical columns with the explicit 'Unknown' label."""
    for col in CATEGORICAL_FILL_COLUMNS:
        if col in df.columns:
            missing_mask = df[col].isna()
            if missing_mask.any():
                df[col] = df[col].fillna("Unknown")
                add_imputation_flag(df, col, missing_mask)
    return df


def impute_numeric_by_category(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute numeric fields with the category median, then global median fallback.
    Rows are expected to have a non-missing category before this runs.
    """
    if "category" not in df.columns:
        return df

    for col in IMPUTE_COLUMNS:
        if col not in df.columns:
            continue

        missing_mask = df[col].isna()
        if not missing_mask.any():
            continue

        category_medians = df.groupby("category")[col].transform("median")
        global_median = df[col].median()
        fill_values = category_medians.fillna(global_median)
        impute_mask = missing_mask & fill_values.notna()
        df.loc[impute_mask, col] = fill_values[impute_mask]
        add_imputation_flag(df, col, impute_mask)
    return df


def recompute_total_spent_from_formula(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rebuild total_spent from the available unit-price field and quantity.

    We prefer selling_price / price_per_unit, but fall back to cost_price when
    that's the only usable price field in the file. Any row with a valid price
    and quantity gets overwritten so stale or negative source totals do not
    survive the cleaning step.
    """
    if "quantity" not in df.columns:
        return df

    price_field = next(
        (
            c
            for c in ["selling_price", "price_per_unit", "cost_price"]
            if c in df.columns
        ),
        None,
    )
    if price_field is None:
        return df

    if "total_spent" not in df.columns:
        df["total_spent"] = np.nan

    price = pd.to_numeric(df[price_field], errors="coerce")
    quantity = pd.to_numeric(df["quantity"], errors="coerce")

    discount_multiplier = 1.0
    if "discount_applied" in df.columns:
        discount = pd.to_numeric(df["discount_applied"], errors="coerce").fillna(0)
        discount_multiplier = np.where(discount > 0, 0.9, 1.0)

    recomputed = price * quantity * discount_multiplier
    recompute_mask = price.notna() & quantity.notna() & recomputed.notna()
    if not recompute_mask.any():
        return df

    df.loc[recompute_mask, "total_spent"] = recomputed[recompute_mask]
    if "total_spent_was_imputed" not in df.columns:
        df["total_spent_was_imputed"] = False
    df.loc[recompute_mask, "total_spent_was_imputed"] = True
    print(
        f"  Recomputed total_spent for {int(recompute_mask.sum()):,} row(s) "
        f"using {price_field} x quantity"
    )
    return df


def cap_outliers_by_category_iqr(df: pd.DataFrame, column: str) -> tuple[pd.DataFrame, int]:
    """Winsorize values outside the category-level IQR bounds."""
    if column not in df.columns or "category" not in df.columns:
        return df, 0

    # Work in a float buffer so nullable integer/decimal columns do not fight
    # with pandas during clipping and assignment.
    working = pd.to_numeric(df[column], errors="coerce").astype("float64")
    capped_count = 0
    for category_value, group_index in df.groupby("category").groups.items():
        values = working.loc[group_index]
        valid = values.dropna()
        if valid.empty:
            continue

        q1 = valid.quantile(0.25)
        q3 = valid.quantile(0.75)
        if pd.isna(q1) or pd.isna(q3):
            continue

        iqr = q3 - q1
        if pd.isna(iqr) or iqr == 0:
            continue

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        clipped = values.clip(lower=lower_bound, upper=upper_bound)
        if column == "quantity":
            clipped = clipped.round().astype("Int64")
        changed_mask = values.notna() & (clipped != values)
        capped_count += int(changed_mask.sum())
        if changed_mask.any():
            working.loc[group_index] = clipped

    print(f"  Capped {capped_count:,} value(s) in {column}")
    if column == "quantity":
        df[column] = working.round().astype("Int64")
    else:
        df[column] = working
    return df, capped_count


def resolve_customer_gender_conflicts(df: pd.DataFrame) -> pd.DataFrame:
    """Detect and resolve conflicting customer_id gender histories."""
    if "customer_id" not in df.columns or "gender" not in df.columns:
        return df

    df = df.copy()
    df["gender_conflict_flag"] = False
    df["gender_resolved_by_mode"] = False

    customer_groups = df.groupby("customer_id", dropna=False)
    total_unique_customers = int(customer_groups.ngroups)

    conflict_customers = []
    for customer_id, group in customer_groups:
        genders = group["gender"].dropna().astype(str).str.strip()
        genders = genders[genders.ne("")]
        if genders.nunique() > 1:
            conflict_customers.append(customer_id)

    conflict_customer_count = len(conflict_customers)
    conflict_pct = (
        conflict_customer_count / total_unique_customers * 100
        if total_unique_customers
        else 0.0
    )
    log_audit(
        "Gender conflict scan: "
        f"{conflict_customer_count:,} conflicted customers out of "
        f"{total_unique_customers:,} unique customers "
        f"({conflict_pct:.2f}%)."
    )

    if not conflict_customers:
        return df

    conflict_mask = df["customer_id"].isin(conflict_customers)
    df.loc[conflict_mask, "gender_conflict_flag"] = True

    if conflict_pct <= 5.0:
        rows_removed = int(conflict_mask.sum())
        df = df.loc[~conflict_mask].copy()
        remaining_customers = df["customer_id"].nunique(dropna=False)
        log_audit(
            "Gender conflict action: dropped all rows for conflicted "
            "customers because conflict_pct <= 5%.\n"
            f"  Customers removed: {conflict_customer_count:,}\n"
            f"  Rows removed: {rows_removed:,}\n"
            f"  Remaining unique customers: {remaining_customers:,}"
        )
        return df

    mode_lookup = {}
    for customer_id, group in df.loc[conflict_mask].groupby("customer_id", dropna=False):
        genders = group["gender"].dropna().astype(str).str.strip()
        genders = genders[genders.ne("")]
        if genders.empty:
            continue
        mode_values = genders.mode()
        mode_lookup[customer_id] = mode_values.iloc[0] if not mode_values.empty else genders.iloc[0]

    if mode_lookup:
        resolved_mask = df["customer_id"].isin(mode_lookup)
        df.loc[resolved_mask, "gender"] = df.loc[resolved_mask, "customer_id"].map(mode_lookup)
        df.loc[resolved_mask, "gender_resolved_by_mode"] = True
        log_audit(
            "Gender conflict action: resolved conflicted customers by mode "
            "because conflict_pct > 5%.\n"
            f"  Customers resolved: {len(mode_lookup):,}\n"
            f"  Rows resolved: {int(resolved_mask.sum()):,}"
        )
    else:
        log_audit(
            "Gender conflict action: no valid non-null gender mode found for "
            "conflicted customers, so no rows were changed."
        )

    return df


def resolve_customer_address_conflicts(df: pd.DataFrame) -> pd.DataFrame:
    """Resolve high-variance customer addresses using the latest transaction."""
    address_cols = [col for col in ["state", "city", "region"] if col in df.columns]
    if "customer_id" not in df.columns or "transaction_date" not in df.columns or not address_cols:
        return df

    df = df.copy()
    df["address_conflict_flag"] = False
    df["address_resolved_by_most_recent"] = False

    conflict_customers = []
    breakdown_3 = 0
    breakdown_4_plus = 0

    for customer_id, group in df.groupby("customer_id", dropna=False):
        max_distinct = 0
        has_conflict = False
        for col in address_cols:
            values = group[col].dropna().astype(str).str.strip()
            values = values[values.ne("")]
            distinct_count = values.nunique()
            max_distinct = max(max_distinct, distinct_count)
            if distinct_count > 2:
                has_conflict = True
        if has_conflict:
            conflict_customers.append(customer_id)
            if max_distinct == 3:
                breakdown_3 += 1
            elif max_distinct >= 4:
                breakdown_4_plus += 1

    conflict_count = len(conflict_customers)
    log_audit(
        "Address conflict scan: "
        f"{conflict_count:,} flagged customers with >2 distinct values in "
        "state/city/region.\n"
        f"  Breakdown: {breakdown_3:,} customers with 3 distinct values, "
        f"{breakdown_4_plus:,} customers with 4+ distinct values"
    )

    if not conflict_customers:
        return df

    conflict_mask = df["customer_id"].isin(conflict_customers)
    df.loc[conflict_mask, "address_conflict_flag"] = True

    latest_rows = (
        df.loc[conflict_mask]
        .sort_values(["customer_id", "transaction_date"])
        .groupby("customer_id", dropna=False)
        .tail(1)
        .set_index("customer_id")
    )

    for col in address_cols:
        if col not in latest_rows.columns:
            continue
        value_map = latest_rows[col].to_dict()
        resolved_mask = df["customer_id"].isin(value_map)
        df.loc[resolved_mask, col] = df.loc[resolved_mask, "customer_id"].map(value_map)
        df.loc[resolved_mask, "address_resolved_by_most_recent"] = True

    log_audit(
        "Address conflict action: resolved state/city/region from each "
        "customer's most recent transaction_date.\n"
        f"  Customers resolved: {conflict_count:,}\n"
        f"  Rows updated: {int(conflict_mask.sum()):,}"
    )

    return df


def drop_duplicate_transaction_ids(
    df: pd.DataFrame, scope: str
) -> pd.DataFrame:
    """Drop complete rows for repeated non-empty transaction IDs."""
    if "transaction_id" not in df.columns:
        before = len(df)
        result = df.drop_duplicates()
        print(f"  Removed {before - len(result):,} exact duplicate row(s) {scope}")
        return result

    ids = df["transaction_id"].astype("string").str.strip()
    ids = ids.mask(ids.eq(""))
    duplicate_mask = ids.notna() & ids.duplicated(keep="first")
    result = df.loc[~duplicate_mask].copy()
    result["transaction_id"] = ids.loc[~duplicate_mask]
    print(
        f"  Removed {int(duplicate_mask.sum()):,} duplicate transaction row(s) "
        f"{scope}"
    )
    return result


def save_excel_in_sheets(df: pd.DataFrame, output_path: Path) -> int:
    """Stream large results across as many Excel sheets as required."""
    from openpyxl import Workbook

    sheet_count = max(1, (len(df) + EXCEL_MAX_DATA_ROWS - 1) // EXCEL_MAX_DATA_ROWS)
    workbook = Workbook(write_only=True)
    for sheet_index, start in enumerate(
        range(0, max(len(df), 1), EXCEL_MAX_DATA_ROWS), start=1
    ):
        worksheet = workbook.create_sheet(title=f"Cleaned_Data_{sheet_index}")
        worksheet.append(list(df.columns))
        sheet = df.iloc[start:start + EXCEL_MAX_DATA_ROWS]
        for row in sheet.itertuples(index=False, name=None):
            excel_row = []
            for value in row:
                if pd.isna(value):
                    excel_row.append(None)
                elif isinstance(value, pd.Timestamp):
                    excel_row.append(value.to_pydatetime())
                elif isinstance(value, np.generic):
                    excel_row.append(value.item())
                else:
                    excel_row.append(value)
            worksheet.append(excel_row)
    workbook.save(output_path)
    return sheet_count


def clean_file(df: pd.DataFrame, source_file: str) -> pd.DataFrame:
    """
    Standardize one file's columns, then apply the cleaning rules --
    written against the CANONICAL (lowercase, snake_case) column names,
    so the same logic works no matter what the source file originally
    called its columns.
    """
    df = standardize_columns(df, verbose=False)
    n_start = len(df)
    df = drop_duplicate_transaction_ids(df, "within this file")

    print_missing_value_percentages(df)

    # ---- 2. Normalize text columns (only ones actually present) ----
    for col in TEXT_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip().str.title()
            df.loc[df[col].isin(["Nan", "None", "", "<Na>", "<NA>"]), col] = np.nan

    if "gender" in df.columns:
        df["gender"] = normalize_gender(df["gender"])
    if "country" in df.columns:
        df["country"] = normalize_country(df["country"])
    if "channel" in df.columns:
        df["channel"] = normalize_channel(df["channel"])

    # ---- 3. Coerce numeric columns that may be stored as strings ----
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            if col == "discount_applied":
                df[col] = normalize_discount(df[col])
            else:
                df[col] = to_numeric_currency(df[col])

    # Parse dates early so critical missing values can be dropped first.
    if "transaction_date" in df.columns:
        df["transaction_date"] = parse_transaction_dates(df["transaction_date"])

    # Critical fields: no fallback exists, so drop missing rows immediately.
    required = [c for c in ["category", "transaction_date"] if c in df.columns]
    if required:
        missing_required = df[required].isna().any(axis=1)
        removed_required = int(missing_required.sum())
        if removed_required:
            print(
                f"  Removed {removed_required:,} row(s) missing critical "
                f"field(s): {', '.join(required)}"
            )
        df = df.loc[~missing_required].copy()

    # Physically impossible numeric values are deleted before any IQR logic.
    df = drop_physically_impossible_rows(df)

    # ---- 5. Fill categorical missing values with explicit unknown labels ----
    df = fill_categorical_missing_values(df)

    # ---- 6. Impute numeric values by category median, then global median ----
    df = impute_numeric_by_category(df)

    # ---- 7. Parse date parts after invalid dates have been removed ----
    if "transaction_date" in df.columns:
        df["transaction_year"] = df["transaction_date"].dt.year.astype("Int64")
        df["transaction_month"] = df["transaction_date"].dt.month.astype("Int64")
        df["transaction_day"] = df["transaction_date"].dt.day.astype("Int64")
        df["transaction_day_name"] = df["transaction_date"].dt.day_name()

    # ---- 8. Winsorize outliers per category, then rebuild derived totals ----
    for col in ["price_per_unit", "cost_price", "selling_price", "quantity"]:
        df, _ = cap_outliers_by_category_iqr(df, col)

    df = recompute_total_spent_from_formula(df)
    df, _ = cap_outliers_by_category_iqr(df, "total_spent")

    if "total_spent" in df.columns:
        df["total_spent"] = df["total_spent"].round(2)

    print(f"    {n_start:,} -> {len(df):,} rows after per-file cleaning")
    return df


def main():
    files = sorted(
        f for f in glob.glob(os.path.join(DATA_FOLDER, "*"))
        if f.lower().endswith(SUPPORTED_EXTENSIONS)
        and not os.path.basename(f).startswith("~$")
        and not os.path.basename(f).lower().startswith("retail_sales_cleaned")
        and os.path.abspath(f) not in {
            os.path.abspath(OUTPUT_XLSX),
            os.path.abspath(OUTPUT_CSV),
        }
    )
    if not files:
        raise FileNotFoundError(f"No CSV/Excel files found in {DATA_FOLDER}")

    print(f"Found {len(files)} raw file(s) in {DATA_FOLDER}:")
    for f in files:
        print(f"  - {os.path.basename(f)}")
    log_audit(
        f"Started data cleaning run for {len(files)} raw file(s) from {DATA_FOLDER}"
    )

    cleaned_frames = []
    available_columns_by_file = {}
    for filepath in files:
        print(f"\nProcessing {os.path.basename(filepath)} ...")
        raw = load_raw_file(filepath)
        print(f"  Loaded {len(raw):,} raw rows, {len(raw.columns)} columns")
        cleaned = clean_file(raw, filepath)
        available_columns_by_file[os.path.basename(filepath)] = {
            col for col in cleaned.columns if col in COLUMN_ALIASES
        }
        cleaned_frames.append(cleaned)

    canonical_columns = list(COLUMN_ALIASES)
    print("\nStandardized column availability by file:")
    for filename, available in available_columns_by_file.items():
        missing = [col for col in canonical_columns if col not in available]
        print(f"  {filename}")
        print(f"    Missing: {', '.join(missing) if missing else 'None'}")

    # pd.concat appends every cleaned file vertically, one below another.
    combined = pd.concat(cleaned_frames, ignore_index=True, sort=False)
    print(
        f"Appended {len(cleaned_frames)} files: "
        f"{sum(len(frame) for frame in cleaned_frames):,} cleaned input rows"
    )

    # Remove repeated transaction IDs after appending, including IDs that
    # occur in different source files.
    before = len(combined)
    combined = drop_duplicate_transaction_ids(combined, "across all files")
    print(f"  Rows after cross-file deduplication: {before:,} -> {len(combined):,}")

    combined = resolve_customer_gender_conflicts(combined)
    combined = resolve_customer_address_conflicts(combined)

    print("\nCanonical column missingness:")
    excluded_columns = []
    unmatched_or_empty_columns = []
    output_columns = []
    for col in canonical_columns:
        per_file_rates = {
            filename: (
                float(frame[col].isna().mean()) if col in frame.columns else 1.0
            )
            for filename, frame in zip(available_columns_by_file, cleaned_frames)
        }
        overall_rate = (
            float(combined[col].isna().mean()) if col in combined.columns else 1.0
        )
        exceeds_file_limit = any(
            rate >= PER_FILE_MISSING_THRESHOLD for rate in per_file_rates.values()
        )
        exceeds_overall_limit = overall_rate >= OVERALL_MISSING_THRESHOLD
        has_no_data = overall_rate == 1.0
        is_forced = col in ALWAYS_OUTPUT_COLUMNS
        should_exclude = (
            has_no_data
            or (exceeds_file_limit and exceeds_overall_limit and not is_forced)
        )

        rates = ", ".join(
            f"{filename}={rate:.1%}" for filename, rate in per_file_rates.items()
        )
        if has_no_data:
            status = "NO DATA / NOT MATCHED - EXCLUDED"
            unmatched_or_empty_columns.append(col)
        elif is_forced:
            status = "FORCED OUTPUT"
        else:
            status = "NOT MATCHED - EXCLUDED" if should_exclude else "OUTPUT"
        print(f"  {col}: {rates}, overall={overall_rate:.1%} -> {status}")

        if should_exclude:
            excluded_columns.append(col)
        else:
            output_columns.append(col)

    for col in ALWAYS_OUTPUT_COLUMNS:
        if col in output_columns and col not in combined.columns:
            combined[col] = pd.NA

    output_columns += [
        col for col in DATE_PART_COLUMNS if col in combined.columns
    ]
    output_columns += [
        col for col in combined.columns if col.endswith("_was_imputed")
    ]
    output_columns += [
        col for col in combined.columns
        if col.endswith("_flag") or col.endswith("_resolved_by_mode")
        or col.endswith("_resolved_by_most_recent")
    ]
    combined = combined.loc[:, output_columns]
    print(
        "\nColumns excluded by missingness rules: "
        + (", ".join(excluded_columns) if excluded_columns else "None")
    )
    print(
        "Columns with no data or no match in any file: "
        + (
            ", ".join(unmatched_or_empty_columns)
            if unmatched_or_empty_columns else "None"
        )
    )

    if "transaction_date" in combined.columns:
        combined = combined.sort_values("transaction_date").reset_index(drop=True)

    print(f"\nFinal combined cleaned dataset: {len(combined):,} rows, "
          f"{len(combined.columns)} columns")
    na_counts = combined.isna().sum()
    print(f"Missing values remaining:\n{na_counts[na_counts > 0]}")

    combined.to_csv(OUTPUT_CSV, index=False)
    if "--csv-only" in sys.argv:
        print(
            "Skipped Excel export (--csv-only); the automated forecast "
            "pipeline consumes the cleaned CSV."
        )
        print(f"Saved combined cleaned data -> {OUTPUT_CSV}")
        log_audit(
            f"Completed data cleaning run. Final dataset rows: {len(combined):,}, "
            f"columns: {len(combined.columns)}"
        )
        return

    excel_output = OUTPUT_XLSX
    sheet_count = None
    excel_save_error = None
    excel_candidates = [
        OUTPUT_XLSX,
        OUTPUT_XLSX.with_name(f"{OUTPUT_XLSX.stem}_updated{OUTPUT_XLSX.suffix}"),
    ]
    for candidate in excel_candidates:
        try:
            sheet_count = save_excel_in_sheets(combined, candidate)
            excel_output = candidate
            excel_save_error = None
            break
        except PermissionError as exc:
            excel_save_error = exc
            if candidate == OUTPUT_XLSX:
                print(
                    f"\nWARNING: {OUTPUT_XLSX.name} is open in Excel; "
                    f"trying {excel_candidates[1].name} instead."
                )
        except Exception as exc:
            excel_save_error = exc
            if candidate == OUTPUT_XLSX:
                print(
                    f"\nWARNING: saving {OUTPUT_XLSX.name} failed with "
                    f"{type(exc).__name__}: {exc}. Trying an alternate name."
                )
            else:
                print(
                    f"\nWARNING: alternate Excel save also failed with "
                    f"{type(exc).__name__}: {exc}. Continuing with CSV only."
                )
    if sheet_count is not None:
        print(f"\nSaved combined cleaned data -> {excel_output}")
        print(f"  Excel workbook uses {sheet_count} sheet(s)")
    elif excel_save_error is not None:
        print(
            f"\nWARNING: Excel workbook was not saved. "
            f"CSV output is still available at {OUTPUT_CSV}."
        )
    print(f"Saved combined cleaned data -> {OUTPUT_CSV}")
    log_audit(
        f"Completed data cleaning run. Final dataset rows: {len(combined):,}, "
        f"columns: {len(combined.columns)}"
    )


if __name__ == "__main__":
    main()
