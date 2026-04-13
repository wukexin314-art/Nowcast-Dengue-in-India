"""
update_monthly.py — Merge freshly fetched data into monthly_data.csv.

Overwrite / update rules (applied column by column):
  Case 1 — source fetch fully failed (new_df is None):
    → skip overwrite; all existing values preserved

  Case 2 — source fetch succeeded (new_df is a DataFrame):
    for each row where new value IS NOT NULL → overwrite existing
    for each row where new value IS NULL     → preserve existing
    months in existing data absent from fetch → preserve existing

  Google Trends special case (is_google_trends=True):
    → on failure (new_df is None): NEVER modify existing data
    → on success: full-window fetch result overwrites all stored months
      for that keyword (because the index may re-normalize across runs)

The CSV key is the 'date' column (YYYY-MM string).
All date values are normalized via config.normalize_monthly_date().
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

import config

logger = logging.getLogger(__name__)


def add_wiki_normalized_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute wikipedia_total_dengue_views_normalized and add/update it in df.

    Formula:
        normalized_t = raw_t × (weighted_t / raw_t) × WIKI_NORM_GLOBAL_SCALE

    where the month-level ratio comes from WIKI_NORM_REFERENCE_CSV.
    Missing months in the reference are filled with ffill then bfill.
    The original wikipedia_total_dengue_views column is never modified.
    """
    ref_path = config.WIKI_NORM_REFERENCE_CSV
    if not ref_path.exists():
        logger.warning(
            "add_wiki_normalized_column | reference file not found: %s — skipping", ref_path
        )
        if "wikipedia_total_dengue_views_normalized" not in df.columns:
            df["wikipedia_total_dengue_views_normalized"] = None
        return df

    ref = pd.read_csv(ref_path)
    ref = ref.drop_duplicates(subset="month", keep="first").copy()
    ref["wiki_weight_ratio"] = (
        pd.to_numeric(ref["weighted_total_pageviews"], errors="coerce") /
        pd.to_numeric(ref["raw_total_pageviews"],      errors="coerce")
    )

    # Build ratio series covering all months present in either source
    data_months = df["date"].dropna().unique().tolist()
    all_months = sorted(set(ref["month"].tolist()) | set(data_months))
    ratio_series = (
        ref.set_index("month")["wiki_weight_ratio"]
        .reindex(all_months)
        .ffill()
        .bfill()
    )

    raw_wiki = pd.to_numeric(df["wikipedia_total_dengue_views"], errors="coerce")
    ratio_vals = df["date"].map(ratio_series.to_dict())
    normalized = raw_wiki * ratio_vals * config.WIKI_NORM_GLOBAL_SCALE

    df = df.copy()
    df["wikipedia_total_dengue_views_normalized"] = normalized.where(
        normalized.notna(), other=None
    ).astype(object)
    # Round to 4 decimal places for readability
    df["wikipedia_total_dengue_views_normalized"] = df[
        "wikipedia_total_dengue_views_normalized"
    ].apply(lambda x: round(float(x), 4) if pd.notna(x) and x is not None else None)

    n_filled = normalized.notna().sum()
    logger.info(
        "add_wiki_normalized_column | computed normalized wiki for %d/%d rows",
        n_filled, len(df),
    )
    return df


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def load_monthly_data() -> pd.DataFrame:
    """
    Load monthly_data.csv, creating an empty frame if it does not exist.

    Returns a DataFrame with config.MONTHLY_COLUMNS, indexed by 'date'.
    """
    if config.MONTHLY_DATA_CSV.exists():
        df = pd.read_csv(config.MONTHLY_DATA_CSV, dtype=str)
        logger.info("update_monthly | Loaded %d rows from %s", len(df), config.MONTHLY_DATA_CSV)
    else:
        df = pd.DataFrame(columns=config.MONTHLY_COLUMNS)
        logger.info("update_monthly | monthly_data.csv not found — starting fresh")

    # Ensure all expected columns exist
    for col in config.MONTHLY_COLUMNS:
        if col not in df.columns:
            df[col] = None

    df = df[config.MONTHLY_COLUMNS].copy()
    df["date"] = df["date"].apply(_safe_normalize_date)
    df = df.dropna(subset=["date"])
    df = df.sort_values("date").drop_duplicates(subset=["date"]).reset_index(drop=True)
    return df


def save_monthly_data(df: pd.DataFrame) -> None:
    """Write the monthly DataFrame to monthly_data.csv."""
    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    df_out = df.sort_values("date").reset_index(drop=True)
    df_out.to_csv(config.MONTHLY_DATA_CSV, index=False)
    logger.info("update_monthly | Saved %d rows to %s", len(df_out), config.MONTHLY_DATA_CSV)


def apply_update(
    existing: pd.DataFrame,
    new_df: Optional[pd.DataFrame],
    value_col: str,
    source_col: str,
    is_google_trends: bool = False,
) -> tuple[pd.DataFrame, int, int]:
    """
    Apply a fetched source update to the existing monthly DataFrame.

    Args:
        existing:         Current monthly_data DataFrame (date + all columns).
        new_df:           Fetched result with columns (date, value) or None.
        value_col:        Target column name in existing (e.g. 'who_cases_monthly').
        source_col:       Source column name in new_df (typically 'value').
        is_google_trends: If True, on success overwrite ALL stored months
                          for this column (full-window replacement).

    Returns:
        (updated_df, rows_added, rows_updated)
    """
    if new_df is None:
        logger.info(
            "update_monthly | %s: fetch failed — preserving all existing values",
            value_col,
        )
        return existing, 0, 0

    # Normalize dates in incoming data
    new_df = new_df.copy()
    new_df["date"] = new_df["date"].apply(_safe_normalize_date)
    new_df = new_df.dropna(subset=["date"]).copy()
    new_df = new_df[new_df["date"] >= config.START_DATE]

    if new_df.empty:
        logger.warning(
            "update_monthly | %s: fetched DataFrame is empty after date filtering",
            value_col,
        )
        return existing, 0, 0

    # Merge on date (outer join to handle new dates not yet in existing)
    merged = pd.merge(
        existing[["date"]].copy(),
        new_df[["date", source_col]].rename(columns={source_col: "_new"}),
        on="date",
        how="outer",
    )
    # Also add dates from new_df not in existing
    all_dates = sorted(set(existing["date"].tolist()) | set(new_df["date"].tolist()))
    result = existing.copy()

    rows_added   = 0
    rows_updated = 0

    for date in all_dates:
        new_val_series = new_df.loc[new_df["date"] == date, source_col]
        new_val = new_val_series.iloc[0] if not new_val_series.empty else None

        in_existing = date in result["date"].values
        existing_val = (
            result.loc[result["date"] == date, value_col].iloc[0]
            if in_existing else None
        )

        if not in_existing:
            # New date — add row
            new_row = {col: None for col in config.MONTHLY_COLUMNS}
            new_row["date"] = date
            if _is_non_null(new_val):
                new_row[value_col] = str(new_val)
                rows_added += 1
            result = pd.concat([result, pd.DataFrame([new_row])], ignore_index=True)
        else:
            # Existing date — apply overwrite rules
            if is_google_trends:
                # Full-window replacement: always overwrite with new value (even if null)
                # but only update the column, not add a new null row
                if _is_non_null(new_val):
                    result.loc[result["date"] == date, value_col] = str(new_val)
                    if _is_non_null(existing_val):
                        rows_updated += 1
                    else:
                        rows_added += 1
                else:
                    # Google Trends returned null for this month — clear it
                    result.loc[result["date"] == date, value_col] = None
            else:
                # Standard rule: only overwrite with non-null new values
                if _is_non_null(new_val):
                    result.loc[result["date"] == date, value_col] = str(new_val)
                    if _is_non_null(existing_val):
                        rows_updated += 1
                    else:
                        rows_added += 1
                # else: preserve existing value (null new_val → do nothing)

    result = result.sort_values("date").reset_index(drop=True)
    logger.info(
        "update_monthly | %s: +%d new, ~%d updated",
        value_col, rows_added, rows_updated,
    )
    return result, rows_added, rows_updated


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_normalize_date(raw) -> Optional[str]:
    """Normalize a date value; return None if it cannot be parsed."""
    if pd.isna(raw) or raw is None:
        return None
    try:
        return config.normalize_monthly_date(str(raw))
    except ValueError:
        return None


def _is_non_null(val) -> bool:
    """Return True if val is not None and not NaN."""
    if val is None:
        return False
    try:
        return not pd.isna(val)
    except Exception:
        return val is not None
