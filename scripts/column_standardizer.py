"""
column_standardizer.py
----------------------------------------------------------------------------
Dynamic column-name standardization for retail sales files.

Different files call the same thing different names:
    "Transaction_id" / "Order_id" / "Invoice ID" / "Receipt No"  -> all mean
    the same underlying field. This module maps ANY of those variations to
    one canonical name, so every downstream script can rely on a fixed
    schema regardless of what the source file called its columns.

HOW IT WORKS (two-stage matching, in priority order):
    1. EXACT match against a known alias list, after normalizing both
       sides (lowercase, camelCase-split, punctuation -> underscore).
       This is O(1) per column and handles every variation you've
       already seen.
    2. FUZZY match fallback (difflib, stdlib -> no extra dependency)
       against the same alias list, for variations you HAVEN'T seen
       before (typos, unusual abbreviations, etc). This is what makes
       it "dynamic" rather than a hardcoded if/else per column.

USAGE:
    from column_standardizer import standardize_columns

    df = pd.read_csv("some_other_retail_file.csv")
    df = standardize_columns(df)
    # df now has canonical snake_case columns: transaction_id, category,
    # item, price_per_unit, quantity, total_spent, payment_method,
    # location, transaction_date, discount_applied, customer_id
----------------------------------------------------------------------------
"""

import re
from difflib import get_close_matches

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# 1. MASTER SCHEMA — canonical_name -> every alias you've seen so far.
#    Add new variations here as you encounter them in new files; you
#    do NOT need to touch any matching logic below.
# ---------------------------------------------------------------------
COLUMN_ALIASES = {
    "transaction_id": [
        "transaction_id", "transaction id", "order_id", "order id",
        "invoice_id", "invoice id", "txn_id", "txn id", "receipt_id",
        "receipt no", "receipt number", "bill_no", "bill number", "sale_id",
        "orderid", "invoiceid",
    ],
    "customer_id": [
        "customer_id", "customer id", "cust_id", "client_id", "user_id",
        "member_id", "shopper_id", "memberid", "customerid",
    ],
    "category": [
        "category", "product_category", "product category", "dept",
        "department", "prod_cat", "item_category", "product_line",
        "product line",
    ],
    "product_name": [
        "product_name", "product name", "prod_name", "productname",
        "product_title", "item_title", "Product",
    ],
    "cost_price": [
        "cost_price", "cost price", "costprice", "buying_price", "MRP",
        "buying price", "price_local", "purchase_price", "purchase price", "cp",
    ],
    "selling_price": [
        "selling_price", "selling price", "selliing price", "price_per_unit", "price per unit", "unit_price", "unit price",
        "selliing_price", "sale_price","sale_price_local", "sale price", "sales_price", "sp",
        "sale_price_local", "saleprice", "sellingprice",
    ],
    "quantity": [
        "quantity", "qty", "units", "units_sold", "no_of_units",
        "item_qty", "order_qty", "quantity_sold", "units sold",
    ],
    "total_spent": [
        "total_spent", "total spent", "total_amount", "total amount",
        "amount", "total", "sales_amount", "revenue", "grand_total",
        "net_amount", "profit", "Total_Sales", "Total Sales",
    ],
    "payment_method": [
        "payment_method", "payment method", "payment_type",
        "payment mode", "pay_method", "mode_of_payment",
    ],
    "country": ["country", "country_name", "country_code", "nation", 
                "nation_name", "iso_country", "country_iso_code", "geo_country", "region_country"
    ],
    "region": ["region", "region_name", "sales_region", "geographic_region",
               "market_region"],
    "state": ["state", "state_name", "province", "province_name", "state_code", 
              "state_iso_code", "admin_region", "territory"
    ],
    "city": ["city", "city_name", "town", "town_name", "municipality", 
             "locality", "urban_area", "metro_city", "city_code"
    ],
    "transaction_date": [
        "transaction_date", "transaction date", "order_date",
        "order date", "date", "sale_date", "purchase_date", "invoice_date",
        "invoice date",
    ],
    "discount_applied": [
        "discount_applied", "discount applied", "discount",
        "has_discount", "discount_flag", "promo_applied", "discount_pct",
        "discount pct", "discount_rate",
    ],
    "gender": [
        "gender", "customer_gender", "customer gender", "sex",
        "gender_segment", "gender_category",
    ],
    "channel": [
        "channel", "sales_channel", "sales channel", "order_channel",
        "order channel", "platform", "purchase_channel", "sales_method",
        "sales method",
    ],
}


def _normalize(name: str) -> str:
    """
    Collapse any column-name style down to a comparable form:
    "Order-ID", "orderID", "order_id", "Order Id" all become "order_id".
    """
    name = str(name).strip()
    name = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name)   # split camelCase
    name = name.lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)                # punctuation/space -> _
    return name.strip("_")


# Flat lookup built once at import time: normalized_alias -> canonical_name
_ALIAS_LOOKUP = {
    _normalize(alias): canonical
    for canonical, aliases in COLUMN_ALIASES.items()
    for alias in aliases
}
_NORMALIZED_ALIASES = list(_ALIAS_LOOKUP.keys())


def standardize_columns(df: pd.DataFrame, fuzzy_cutoff: float = 0.82,
                         verbose: bool = True) -> pd.DataFrame:
    """
    Rename df's columns to the canonical schema defined in COLUMN_ALIASES.

    Parameters
    ----------
    df : the raw dataframe, straight from any source file
    fuzzy_cutoff : similarity threshold (0-1) for the fallback fuzzy match.
        Higher = stricter (fewer false-positive renames). 0.82 is a safe
        default; lower it if a file uses very abbreviated/unusual names.
    verbose : print a mapping report (recommended while onboarding new files)

    Returns
    -------
    df with columns renamed in place where a confident match was found.
    Columns that couldn't be matched are left untouched (never silently
    dropped) so you can inspect and add them as new aliases.
    """
    n_original = len(df.columns)
    rename_map = {}
    # Exact canonical column names take precedence over aliases regardless of
    # source-column order (for example, prefer "gender" to "gender_segment").
    matched_canonicals = {
        _normalize(col): col
        for col in df.columns
        if _normalize(col) in COLUMN_ALIASES
    }
    unmatched = []

    for col in df.columns:
        norm_col = _normalize(col)

        if norm_col in COLUMN_ALIASES:
            rename_map[col] = norm_col
            continue

        # Stage 1: exact match (fast path, handles everything seen before)
        canonical = _ALIAS_LOOKUP.get(norm_col)

        # Stage 2: fuzzy fallback for unseen variations
        if canonical is None:
            close = get_close_matches(norm_col, _NORMALIZED_ALIASES,
                                       n=1, cutoff=fuzzy_cutoff)
            if close:
                canonical = _ALIAS_LOOKUP[close[0]]

        if canonical is None:
            unmatched.append(col)
            continue

        if canonical in matched_canonicals:
            # Two source columns both mapped to the same canonical field —
            # flag it loudly rather than silently overwriting one.
            print(f"  WARNING: both '{matched_canonicals[canonical]}' and "
                  f"'{col}' matched '{canonical}'. Keeping "
                  f"'{matched_canonicals[canonical]}', leaving '{col}' as-is. "
                  f"Rename manually or tighten fuzzy_cutoff.")
            unmatched.append(col)
            continue

        rename_map[col] = canonical
        matched_canonicals[canonical] = col

    df = df.rename(columns=rename_map)

    if verbose:
        print(f"Standardized {len(rename_map)}/{n_original} columns:")
        for orig, std in rename_map.items():
            marker = "==" if orig == std else "->"
            print(f"  '{orig}' {marker} '{std}'")
        if unmatched:
            print(f"Left as-is (no confident match): {unmatched}")

    return df


if __name__ == "__main__":
    # Quick self-test with a deliberately "different" file schema to prove
    # the dynamic matching works beyond the one column you hardcoded.
    sample = pd.DataFrame({
        "Invoice ID": ["A1", "A2"],
        "Prod_Cat": ["Electronics", "Groceries"],
        "product name": ["Fast Charger", "Whole Milk"],
        "Item": ["Charger", "Milk"],
        "Brand": ["Anker", "Almarai"],
        "UnitPrice": [29.99, 3.5],
        "Cost Price": [18.0, 2.0],
        "Selliing Price": [29.99, 3.5],   # matches the user's own typo
        "Qty": [2, 5],
        "Grand Total": [59.98, 17.5],
        "Payment Mode": ["Cash", "Card"],
        "Store": ["Mall", "Downtown"],
        "Invoice_Date": ["2024-01-01", "2024-01-02"],
        "Customer Gender": ["F", "M"],
        "Sales Channel": ["Online", "In-Store"],
        "Some Random Column": [1, 2],   # should stay unmatched
    })
    standardize_columns(sample)

    # Sanity check: make sure no single alias was accidentally assigned to
    # two different canonical fields (would silently mis-route a column)
    seen = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            norm = _normalize(alias)
            if norm in seen and seen[norm] != canonical:
                print(f"  ALIAS COLLISION: '{alias}' maps to both "
                      f"'{seen[norm]}' and '{canonical}'")
            seen[norm] = canonical
    else:
        print("\nNo alias collisions found across COLUMN_ALIASES.")
