"""
migrate_legacy.py — ONE-TIME migration from legacy data files to the new schema.

Reads three legacy files from the parent repo root:
    master_data.csv              (long-format: resolution, date, value, source)
    total_dengue_views.csv       (Month, Total_Views)
    monthly_mosquito_aggregate.csv (timestamp, Hindi, Tamil, ..., TOTAL_MONTHLY_VIEWS)

Writes to:
    data/processed/monthly_data.csv
    data/processed/yearly_data.csv

⚠ MOSQUITO DEFINITION CHANGE:
    The legacy TOTAL_MONTHLY_VIEWS column was computed from 6 languages:
    Hindi, Tamil, Marathi, Malayalam, Kannada, Gujarati.

    The new system uses 5 languages (Gujarati dropped).
    Migrated values use the old 6-language definition and will differ from
    future-fetched values once the new fetcher overwrites them.
    This difference is a DEFINITION CHANGE, not a data revision.

Usage:
    python src/migrate_legacy.py [--dry-run]
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from typing import Optional

import pandas as pd

import config
import update_monthly as um
import update_yearly as uy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def run_migration(dry_run: bool = False) -> None:
    """
    Execute the full one-time migration.

    Args:
        dry_run: If True, print what would be written but do not write files.
    """
    config.ensure_dirs()
    _check_legacy_files()

    logger.info("=== Starting legacy migration ===")
    logger.info("Legacy files location: %s", config.PROJECT_ROOT.parent)

    monthly_df = _build_monthly_data()
    yearly_df  = _build_yearly_data()

    if dry_run:
        logger.info("[DRY RUN] Would write %d rows to monthly_data.csv", len(monthly_df))
        logger.info("[DRY RUN] Would write %d rows to yearly_data.csv", len(yearly_df))
        print("\n=== Monthly data (first 5 rows) ===")
        print(monthly_df.head().to_string())
        print("\n=== Yearly data ===")
        print(yearly_df.to_string())
        return

    um.save_monthly_data(monthly_df)
    uy.save_yearly_data(yearly_df)

    # Log the migration event to update_log.csv
    _log_migration(monthly_df, yearly_df)

    logger.info("=== Migration complete ===")
    logger.info("  monthly_data.csv: %d rows", len(monthly_df))
    logger.info("  yearly_data.csv:  %d rows", len(yearly_df))
    logger.info(
        "  NOTE: wikipedia_mosquito_views_total was migrated with the legacy 6-language "
        "definition (includes Gujarati). The new fetcher uses 5 languages (no Gujarati). "
        "Values will be updated to the new definition on first successful fetch."
    )


# ---------------------------------------------------------------------------
# Monthly data construction
# ---------------------------------------------------------------------------

def _build_monthly_data() -> pd.DataFrame:
    """
    Build monthly_data.csv from legacy sources.

    Source mapping:
      master_data.csv [source=WHO]                       → who_cases_monthly
      master_data.csv [source=Google_Trends_Dengue_fever]→ google_trends_dengue_fever
      master_data.csv [source=Google_Trends_Dengue_vaccine]→ google_trends_dengue_vaccine
      total_dengue_views.csv [Total_Views]               → wikipedia_total_dengue_views
      monthly_mosquito_aggregate.csv [TOTAL_MONTHLY_VIEWS]→ wikipedia_mosquito_views_total
    """
    master = _load_master_monthly()

    frames: dict[str, pd.DataFrame] = {}

    # WHO
    who_df = _extract_master_source(master, "WHO", "who_cases_monthly")
    if who_df is not None:
        frames["who_cases_monthly"] = who_df
        logger.info("migrate | WHO: %d rows (≥%s)", len(who_df), config.START_DATE)

    # Google Trends — dengue fever
    gt_fever = _extract_master_source(master, "Google_Trends_Dengue_fever", "google_trends_dengue_fever")
    if gt_fever is not None:
        frames["google_trends_dengue_fever"] = gt_fever
        logger.info("migrate | Google_Trends_fever: %d rows", len(gt_fever))

    # Google Trends — dengue vaccine
    gt_vacc = _extract_master_source(master, "Google_Trends_Dengue_vaccine", "google_trends_dengue_vaccine")
    if gt_vacc is not None:
        frames["google_trends_dengue_vaccine"] = gt_vacc
        logger.info("migrate | Google_Trends_vaccine: %d rows", len(gt_vacc))

    # Wikipedia dengue views
    dengue_views = _load_total_dengue_views()
    if dengue_views is not None:
        frames["wikipedia_total_dengue_views"] = dengue_views
        logger.info("migrate | Wikipedia dengue views: %d rows", len(dengue_views))

    # Wikipedia mosquito views (6-language legacy definition)
    mosquito_views = _load_mosquito_aggregate()
    if mosquito_views is not None:
        frames["wikipedia_mosquito_views_total"] = mosquito_views
        logger.info("migrate | Wikipedia mosquito views (legacy 6-lang): %d rows", len(mosquito_views))

    # Build a date spine covering all dates from all sources
    all_dates = set()
    for df in frames.values():
        all_dates.update(df["date"].tolist())
    all_dates = sorted(d for d in all_dates if d >= config.START_DATE)

    result = pd.DataFrame({"date": all_dates})
    for col, df in frames.items():
        result = pd.merge(result, df.rename(columns={"value": col}), on="date", how="left")

    # Ensure all expected columns exist
    for col in config.MONTHLY_COLUMNS:
        if col not in result.columns:
            result[col] = None

    result = result[config.MONTHLY_COLUMNS].copy()
    result = result.sort_values("date").reset_index(drop=True)
    return result


def _extract_master_source(
    master: pd.DataFrame,
    source_name: str,
    col_name: str,
) -> Optional[pd.DataFrame]:
    """
    Extract a single source from the long-format master DataFrame.

    Returns DataFrame(date, value) for dates >= START_DATE, or None.
    """
    subset = master[master["source"] == source_name].copy()
    if subset.empty:
        logger.warning("migrate | Source '%s' not found in master_data.csv", source_name)
        return None

    subset["date"] = subset["date"].apply(_safe_norm_date)
    subset = subset.dropna(subset=["date"])
    subset = subset[subset["date"] >= config.START_DATE]

    if subset.empty:
        logger.warning(
            "migrate | Source '%s': no rows with date >= %s", source_name, config.START_DATE
        )
        return None

    result = pd.DataFrame({
        "date":  subset["date"].values,
        "value": pd.to_numeric(subset["value"], errors="coerce").values,
    })
    return result.sort_values("date").drop_duplicates(subset=["date"]).reset_index(drop=True)


def _load_total_dengue_views() -> Optional[pd.DataFrame]:
    """Load total_dengue_views.csv → (date, value)."""
    if not config.LEGACY_DENGUE_VIEWS_CSV.exists():
        logger.warning("migrate | %s not found — skipping wikipedia dengue views", config.LEGACY_DENGUE_VIEWS_CSV)
        return None

    df = pd.read_csv(config.LEGACY_DENGUE_VIEWS_CSV)
    # Columns: Month, Total_Views
    if "Month" not in df.columns or "Total_Views" not in df.columns:
        logger.warning(
            "migrate | total_dengue_views.csv missing expected columns. Found: %s",
            list(df.columns),
        )
        return None

    df["date"] = df["Month"].apply(_safe_norm_date)
    df = df.dropna(subset=["date"])
    df = df[df["date"] >= config.START_DATE]
    result = pd.DataFrame({
        "date":  df["date"].values,
        "value": pd.to_numeric(df["Total_Views"], errors="coerce").values,
    })
    return result.sort_values("date").drop_duplicates(subset=["date"]).reset_index(drop=True)


def _load_mosquito_aggregate() -> Optional[pd.DataFrame]:
    """
    Load monthly_mosquito_aggregate.csv → (date, value).

    Uses TOTAL_MONTHLY_VIEWS which includes Gujarati (legacy 6-language definition).
    """
    if not config.LEGACY_MOSQUITO_CSV.exists():
        logger.warning("migrate | %s not found — skipping mosquito views", config.LEGACY_MOSQUITO_CSV)
        return None

    df = pd.read_csv(config.LEGACY_MOSQUITO_CSV)
    # Columns: timestamp, Hindi, Tamil, Marathi, Malayalam, Kannada, Gujarati, TOTAL_MONTHLY_VIEWS
    if "timestamp" not in df.columns or "TOTAL_MONTHLY_VIEWS" not in df.columns:
        logger.warning(
            "migrate | monthly_mosquito_aggregate.csv missing expected columns. Found: %s",
            list(df.columns),
        )
        return None

    df["date"] = df["timestamp"].apply(_safe_norm_date)
    df = df.dropna(subset=["date"])
    df = df[df["date"] >= config.START_DATE]
    result = pd.DataFrame({
        "date":  df["date"].values,
        "value": pd.to_numeric(df["TOTAL_MONTHLY_VIEWS"], errors="coerce").values,
    })
    return result.sort_values("date").drop_duplicates(subset=["date"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Yearly data construction
# ---------------------------------------------------------------------------

def _build_yearly_data() -> pd.DataFrame:
    """
    Build yearly_data.csv from the legacy master_data.csv.

    Source mapping:
      master_data.csv [source=OpenDengue_National_Yearly] → open_dengue_national_yearly
    """
    master_all = _load_master_all()
    yearly_src = master_all[
        (master_all["resolution"].str.lower() == "yearly") &
        (master_all["source"] == "OpenDengue_National_Yearly")
    ].copy()

    if yearly_src.empty:
        logger.warning(
            "migrate | OpenDengue_National_Yearly not found in master_data.csv. "
            "yearly_data.csv will have null values — will be filled on first fetch."
        )
        # Return a skeleton yearly frame for START_YEAR onward
        from datetime import datetime as dt
        years = [str(y) for y in range(int(config.START_YEAR), dt.now().year + 1)]
        return pd.DataFrame({
            "year": years,
            "open_dengue_national_yearly": [None] * len(years),
        })

    yearly_src["year"] = yearly_src["date"].apply(_safe_norm_year)
    yearly_src = yearly_src.dropna(subset=["year"])
    yearly_src = yearly_src[yearly_src["year"] >= config.START_YEAR]

    result = pd.DataFrame({
        "year": yearly_src["year"].values,
        "open_dengue_national_yearly": pd.to_numeric(yearly_src["value"], errors="coerce").values,
    })
    result = result.sort_values("year").drop_duplicates(subset=["year"]).reset_index(drop=True)
    logger.info("migrate | OpenDengue yearly: %d rows (≥%s)", len(result), config.START_YEAR)
    return result


# ---------------------------------------------------------------------------
# File loading
# ---------------------------------------------------------------------------

def _load_master_monthly() -> pd.DataFrame:
    """Load master_data.csv and return only Monthly-resolution rows."""
    return _load_master_all(resolution_filter="monthly")


def _load_master_all(resolution_filter: Optional[str] = None) -> pd.DataFrame:
    """Load master_data.csv (all rows or filtered by resolution)."""
    if not config.LEGACY_MASTER_CSV.exists():
        logger.error("migrate | Legacy master_data.csv not found at %s", config.LEGACY_MASTER_CSV)
        sys.exit(1)

    df = pd.read_csv(config.LEGACY_MASTER_CSV, dtype=str)
    required = {"resolution", "date", "value", "source"}
    if not required.issubset(set(df.columns)):
        logger.error(
            "migrate | master_data.csv missing columns. Expected %s, got %s",
            required, set(df.columns),
        )
        sys.exit(1)

    if resolution_filter:
        df = df[df["resolution"].str.lower() == resolution_filter.lower()].copy()

    return df


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _log_migration(monthly_df: pd.DataFrame, yearly_df: pd.DataFrame) -> None:
    """Append a migration event to update_log.csv."""
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    run_time = datetime.now().isoformat()
    log_rows = [
        {
            "run_time": run_time,
            "source_name": "migration_monthly",
            "status": "SUCCESS",
            "fetch_mode": "legacy_migration",
            "rows_fetched": len(monthly_df),
            "rows_added": len(monthly_df),
            "rows_updated": 0,
            "expected_languages": "",
            "successful_languages": "",
            "error_message": (
                "DEFINITION CHANGE: wikipedia_mosquito_views_total migrated with "
                "legacy 6-language definition (includes Gujarati). New fetcher uses 5 languages."
            ),
        },
        {
            "run_time": run_time,
            "source_name": "migration_yearly",
            "status": "SUCCESS",
            "fetch_mode": "legacy_migration",
            "rows_fetched": len(yearly_df),
            "rows_added": len(yearly_df),
            "rows_updated": 0,
            "expected_languages": "",
            "successful_languages": "",
            "error_message": "",
        },
    ]

    log_df = pd.DataFrame(log_rows)
    if config.UPDATE_LOG_CSV.exists():
        log_df.to_csv(config.UPDATE_LOG_CSV, mode="a", header=False, index=False)
    else:
        log_df.to_csv(config.UPDATE_LOG_CSV, index=False)
    logger.info("migrate | Log written to %s", config.UPDATE_LOG_CSV)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_norm_date(raw) -> Optional[str]:
    if pd.isna(raw) or raw is None:
        return None
    try:
        return config.normalize_monthly_date(str(raw))
    except ValueError:
        return None


def _safe_norm_year(raw) -> Optional[str]:
    if pd.isna(raw) or raw is None:
        return None
    try:
        return config.normalize_year(str(raw))
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _check_legacy_files() -> None:
    """Warn about missing legacy files (non-fatal for individual missing files)."""
    for path in [
        config.LEGACY_MASTER_CSV,
        config.LEGACY_DENGUE_VIEWS_CSV,
        config.LEGACY_MOSQUITO_CSV,
    ]:
        if not path.exists():
            logger.warning("migrate | Legacy file not found: %s", path)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Migrate legacy dengue data files to new schema")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be written without writing any files",
    )
    args = parser.parse_args()

    run_migration(dry_run=args.dry_run)
