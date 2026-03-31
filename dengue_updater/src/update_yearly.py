"""
update_yearly.py — Merge freshly fetched data into yearly_data.csv.

Overwrite / update rules (identical semantics to update_monthly.py):
  Case 1 — source fetch failed (new_df is None):
    → skip overwrite; all existing values preserved

  Case 2 — source fetch succeeded:
    for each row where new value IS NOT NULL → overwrite existing
    for each row where new value IS NULL     → preserve existing

The CSV key is the 'year' column (YYYY string).
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def load_yearly_data() -> pd.DataFrame:
    """
    Load yearly_data.csv, creating an empty frame if it does not exist.

    Returns a DataFrame with config.YEARLY_COLUMNS.
    """
    if config.YEARLY_DATA_CSV.exists():
        df = pd.read_csv(config.YEARLY_DATA_CSV, dtype=str)
        logger.info("update_yearly | Loaded %d rows from %s", len(df), config.YEARLY_DATA_CSV)
    else:
        df = pd.DataFrame(columns=config.YEARLY_COLUMNS)
        logger.info("update_yearly | yearly_data.csv not found — starting fresh")

    for col in config.YEARLY_COLUMNS:
        if col not in df.columns:
            df[col] = None

    df = df[config.YEARLY_COLUMNS].copy()
    df["year"] = df["year"].apply(_safe_normalize_year)
    df = df.dropna(subset=["year"])
    df = df.sort_values("year").drop_duplicates(subset=["year"]).reset_index(drop=True)
    return df


def save_yearly_data(df: pd.DataFrame) -> None:
    """Write the yearly DataFrame to yearly_data.csv."""
    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    df_out = df.sort_values("year").reset_index(drop=True)
    df_out.to_csv(config.YEARLY_DATA_CSV, index=False)
    logger.info("update_yearly | Saved %d rows to %s", len(df_out), config.YEARLY_DATA_CSV)


def apply_yearly_update(
    existing: pd.DataFrame,
    new_df: Optional[pd.DataFrame],
    value_col: str,
    source_col: str,
) -> tuple[pd.DataFrame, int, int]:
    """
    Apply a fetched yearly source update to the existing yearly DataFrame.

    Args:
        existing:   Current yearly_data DataFrame.
        new_df:     Fetched result with columns (year, value) or None.
        value_col:  Target column name in existing.
        source_col: Source column name in new_df (typically 'open_dengue_national_yearly').

    Returns:
        (updated_df, rows_added, rows_updated)
    """
    if new_df is None:
        logger.info(
            "update_yearly | %s: fetch failed — preserving all existing values",
            value_col,
        )
        return existing, 0, 0

    new_df = new_df.copy()
    new_df["year"] = new_df["year"].apply(_safe_normalize_year)
    new_df = new_df.dropna(subset=["year"]).copy()
    new_df = new_df[new_df["year"] >= config.START_YEAR]

    if new_df.empty:
        logger.warning(
            "update_yearly | %s: fetched DataFrame is empty after year filtering",
            value_col,
        )
        return existing, 0, 0

    all_years = sorted(set(existing["year"].tolist()) | set(new_df["year"].tolist()))
    result = existing.copy()
    rows_added   = 0
    rows_updated = 0

    for year in all_years:
        new_val_series = new_df.loc[new_df["year"] == year, source_col]
        new_val = new_val_series.iloc[0] if not new_val_series.empty else None

        in_existing = year in result["year"].values
        existing_val = (
            result.loc[result["year"] == year, value_col].iloc[0]
            if in_existing else None
        )

        if not in_existing:
            new_row = {col: None for col in config.YEARLY_COLUMNS}
            new_row["year"] = year
            if _is_non_null(new_val):
                new_row[value_col] = new_val
                rows_added += 1
            result = pd.concat([result, pd.DataFrame([new_row])], ignore_index=True)
        else:
            if _is_non_null(new_val):
                result.loc[result["year"] == year, value_col] = new_val
                if _is_non_null(existing_val):
                    rows_updated += 1
                else:
                    rows_added += 1
            # else: preserve existing value

    result = result.sort_values("year").reset_index(drop=True)
    logger.info(
        "update_yearly | %s: +%d new, ~%d updated",
        value_col, rows_added, rows_updated,
    )
    return result, rows_added, rows_updated


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_normalize_year(raw) -> Optional[str]:
    if pd.isna(raw) or raw is None:
        return None
    try:
        return config.normalize_year(str(raw))
    except ValueError:
        return None


def _is_non_null(val) -> bool:
    if val is None:
        return False
    try:
        return not pd.isna(val)
    except Exception:
        return val is not None
