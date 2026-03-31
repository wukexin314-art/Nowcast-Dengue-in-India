"""
fetch_open_dengue.py — Fetch the aggregated yearly India dengue series from OpenDengue.

OpenDengue publishes a versioned CSV on GitHub containing national-level
(Admin0) case totals per country per year.  This module downloads that CSV,
filters to India, keeps only yearly-resolution rows, and returns the
result for years >= config.START_YEAR.

Column produced:
    open_dengue_national_yearly  — aggregated yearly India dengue case total

The column name is preserved as-is for schema compatibility with existing
analysis notebooks.  It now refers to the aggregated yearly India series
provided by OpenDengue, regardless of how OpenDengue constructs it internally.

Returns:
    Optional[pd.DataFrame] with columns (year, open_dengue_national_yearly).
    Returns None on any unrecoverable error; existing stored values are NOT
    overwritten in that case.
"""

from __future__ import annotations

import io
import logging
import zipfile
from datetime import datetime
from typing import Optional

import pandas as pd
import requests

import config

logger = logging.getLogger(__name__)

# Known column name patterns in OpenDengue Admin0 CSV
_COUNTRY_COLS    = ["adm_0_name", "country", "countryname", "Country"]
_YEAR_COLS       = ["year", "Year", "calendar_year", "report_year"]
_CASES_COLS      = ["dengue_total", "cases", "total_cases", "dengue_cases", "Count"]
_RESOLUTION_COLS = ["adm_level", "adm0_level", "spatial_scale", "resolution"]

_INDIA_NAMES = {"india", "ind"}


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def fetch_open_dengue() -> Optional[pd.DataFrame]:
    """
    Download the OpenDengue aggregated yearly India series and return
    a clean DataFrame.

    Returns:
        DataFrame(year str YYYY, open_dengue_national_yearly float)
        or None on failure.
    """
    df_raw, source_url = _download_opendengue_csv()
    if df_raw is None:
        return None

    df = _extract_india_yearly(df_raw, source_url)
    if df is None or df.empty:
        logger.error(
            "fetch_open_dengue | No India yearly rows found after filtering. "
            "Existing stored values will be preserved."
        )
        return None

    # Save raw download
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_path = config.RAW_OPENDENGUE / f"national_yearly_v1_3_{ts}.csv"
    df_raw.to_csv(raw_path, index=False)
    logger.info("fetch_open_dengue | Raw CSV saved to %s", raw_path)

    # Log missing years
    all_years = set(str(y) for y in range(int(config.START_YEAR), datetime.now().year + 1))
    found_years = set(df["year"].tolist())
    missing = sorted(all_years - found_years)
    if missing:
        logger.info(
            "fetch_open_dengue | Missing years (will remain null): %s",
            ", ".join(missing),
        )

    logger.info(
        "fetch_open_dengue | Fetched %d yearly India rows (years: %s → %s), source: %s",
        len(df), df["year"].min(), df["year"].max(), source_url,
    )
    return df


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def _download_opendengue_csv() -> tuple[Optional[pd.DataFrame], str]:
    """
    Try config.OPENDENGUE_GITHUB_CSV first; if that fails, try to find a
    download link on config.OPENDENGUE_DATA_PAGE.

    Returns (DataFrame, source_url) or (None, '').
    """
    # Primary: GitHub raw CSV
    df = _get_csv(config.OPENDENGUE_GITHUB_CSV)
    if df is not None:
        return df, config.OPENDENGUE_GITHUB_CSV

    logger.warning(
        "fetch_open_dengue | GitHub URL failed: %s. Trying data page fallback.",
        config.OPENDENGUE_GITHUB_CSV,
    )

    # Fallback: scrape the OpenDengue data page for a download link
    fallback_url = _find_download_url_on_page(config.OPENDENGUE_DATA_PAGE)
    if fallback_url:
        df = _get_csv(fallback_url)
        if df is not None:
            return df, fallback_url

    logger.error(
        "fetch_open_dengue | All download attempts failed. "
        "Existing stored values will be preserved."
    )
    return None, ""


def _get_csv(url: str) -> Optional[pd.DataFrame]:
    """Download a CSV (or ZIP containing a single CSV) from url and return as DataFrame."""
    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        content = resp.content
        # Detect ZIP by magic bytes (PK\x03\x04) or URL suffix
        if content[:4] == b"PK\x03\x04" or url.lower().endswith(".zip"):
            zf = zipfile.ZipFile(io.BytesIO(content))
            csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
            if not csv_names:
                logger.warning("fetch_open_dengue | ZIP at %s contains no CSV files", url)
                return None
            df = pd.read_csv(zf.open(csv_names[0]))
            logger.info(
                "fetch_open_dengue | Downloaded %d rows from ZIP %s (file: %s)",
                len(df), url, csv_names[0],
            )
        else:
            df = pd.read_csv(io.BytesIO(content))
            logger.info("fetch_open_dengue | Downloaded %d rows from %s", len(df), url)
        return df
    except Exception as exc:
        logger.warning("fetch_open_dengue | Failed to download %s: %s", url, exc)
        return None


def _find_download_url_on_page(page_url: str) -> Optional[str]:
    """
    Scrape the OpenDengue data page and look for a CSV download link.
    Returns the first matching URL or None.
    """
    try:
        resp = requests.get(page_url, timeout=20)
        resp.raise_for_status()
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(resp.text, "html.parser")
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if ".csv" in href.lower() and (
                "national" in href.lower() or "admin0" in href.lower() or "adm0" in href.lower()
            ):
                if href.startswith("http"):
                    return href
                return f"https://opendengue.org{href}"
        logger.warning("fetch_open_dengue | No matching CSV link found on %s", page_url)
        return None
    except Exception as exc:
        logger.warning("fetch_open_dengue | Could not scrape data page %s: %s", page_url, exc)
        return None


# ---------------------------------------------------------------------------
# Extraction and validation
# ---------------------------------------------------------------------------

def _extract_india_yearly(df: pd.DataFrame, source_url: str) -> Optional[pd.DataFrame]:
    """
    Given the raw OpenDengue DataFrame, extract India yearly rows and return
    a clean two-column DataFrame.

    Schema validation:
    - At least one of the expected country, year, and cases columns must exist.
    - Rows must have year >= START_YEAR.
    - Cases must be numeric.
    """
    # Normalize column names for matching
    col_map = {c.lower().strip(): c for c in df.columns}

    # --- Country column ---
    country_col = _find_col(col_map, _COUNTRY_COLS)
    if country_col is None:
        logger.error(
            "fetch_open_dengue | Schema error: cannot find country column in %s. "
            "Columns found: %s",
            source_url, list(df.columns),
        )
        return None

    # --- Year column ---
    year_col = _find_col(col_map, _YEAR_COLS)
    if year_col is None:
        logger.error(
            "fetch_open_dengue | Schema error: cannot find year column in %s. "
            "Columns found: %s",
            source_url, list(df.columns),
        )
        return None

    # --- Cases column ---
    cases_col = _find_col(col_map, _CASES_COLS)
    if cases_col is None:
        logger.error(
            "fetch_open_dengue | Schema error: cannot find cases column in %s. "
            "Columns found: %s",
            source_url, list(df.columns),
        )
        return None

    # --- Optional resolution/admin-level column (for informational logging) ---
    res_col = _find_col(col_map, _RESOLUTION_COLS)
    if res_col:
        unique_res = df[res_col].dropna().unique()
        logger.info(
            "fetch_open_dengue | Resolution values in source: %s",
            list(unique_res),
        )

    # Filter to India
    india_mask = df[country_col].astype(str).str.strip().str.lower().isin(_INDIA_NAMES)
    if not india_mask.any():
        india_mask = df[country_col].astype(str).str.lower().str.contains("india", na=False)
    if not india_mask.any():
        logger.error(
            "fetch_open_dengue | No India rows found in column '%s'. "
            "Unique values (first 10): %s",
            country_col, list(df[country_col].dropna().unique()[:10]),
        )
        return None

    df_india = df[india_mask].copy()

    # Filter to national (Admin0) rows only
    if "S_res" in df_india.columns:
        before = len(df_india)
        df_india = df_india[df_india["S_res"] == "Admin0"]
        logger.info("fetch_open_dengue | S_res filter: %d → %d rows", before, len(df_india))
        if df_india.empty:
            logger.error("fetch_open_dengue | No India Admin0 rows after S_res filter")
            return None
    else:
        logger.warning("fetch_open_dengue | S_res column not found; skipping Admin0 filter")

    # Filter to yearly-resolution rows only (exclude partial-year monthly records)
    if "T_res" in df_india.columns:
        monthly_excluded = df_india[df_india["T_res"] != "Year"]
        if not monthly_excluded.empty:
            logger.info(
                "fetch_open_dengue | Excluding %d non-Year T_res rows "
                "(partial-year monthly data, years: %s)",
                len(monthly_excluded),
                sorted(monthly_excluded["Year"].astype(str).unique()),
            )
        df_india = df_india[df_india["T_res"] == "Year"]
        if df_india.empty:
            logger.error("fetch_open_dengue | No India rows with T_res==Year after filter")
            return None
    else:
        logger.warning("fetch_open_dengue | T_res column not found; skipping temporal resolution filter")

    # Normalize year
    try:
        df_india["year"] = df_india[year_col].apply(config.normalize_year)
    except Exception as exc:
        logger.error("fetch_open_dengue | Year normalization failed: %s", exc)
        return None

    # Filter year >= START_YEAR
    df_india = df_india[df_india["year"] >= config.START_YEAR]
    if df_india.empty:
        logger.warning(
            "fetch_open_dengue | No India rows with year >= %s", config.START_YEAR
        )
        return None

    # Cases column
    df_india["open_dengue_national_yearly"] = pd.to_numeric(
        df_india[cases_col], errors="coerce"
    )

    result = (
        df_india[["year", "open_dengue_national_yearly"]]
        .sort_values("year")
        .drop_duplicates(subset=["year"])
        .reset_index(drop=True)
    )

    return result


def _find_col(col_map: dict[str, str], candidates: list[str]) -> Optional[str]:
    """Return the first matching original column name from the normalized map."""
    for cand in candidates:
        if cand.lower() in col_map:
            return col_map[cand.lower()]
    return None
