"""
config.py — Central configuration for dengue_updater.

All paths, constants, and shared utilities live here.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------

# Root of dengue_updater/ (parent of src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_RAW        = PROJECT_ROOT / "data" / "raw"
DATA_INTERIM    = PROJECT_ROOT / "data" / "interim"
DATA_PROCESSED  = PROJECT_ROOT / "data" / "processed"
LOGS_DIR        = PROJECT_ROOT / "logs"
CONFIG_DIR      = PROJECT_ROOT / "config"

# Sub-directories under raw/
RAW_WHO         = DATA_RAW / "who"
RAW_GOOGLE      = DATA_RAW / "google_trends"
RAW_OPENDENGUE  = DATA_RAW / "open_dengue"
RAW_WIKIPEDIA   = DATA_RAW / "wikipedia"

# Output files
MONTHLY_DATA_CSV    = DATA_PROCESSED / "monthly_data.csv"
YEARLY_DATA_CSV     = DATA_PROCESSED / "yearly_data.csv"
MASTER_XLSX         = DATA_PROCESSED / "master_data.xlsx"
UPDATE_LOG_CSV      = LOGS_DIR / "update_log.csv"

# Wiki config
WIKI_TITLE_MAP_YAML = CONFIG_DIR / "wiki_title_map.yaml"

# Legacy input files (relative to the parent repo root)
LEGACY_MASTER_CSV       = PROJECT_ROOT.parent / "master_data.csv"
LEGACY_DENGUE_VIEWS_CSV = PROJECT_ROOT.parent / "total_dengue_views.csv"
LEGACY_MOSQUITO_CSV     = PROJECT_ROOT.parent / "monthly_mosquito_aggregate.csv"

# Wiki normalization reference file (old-style weighted pageviews, used to
# compute the month-level ratio: weighted_total / raw_total)
WIKI_NORM_REFERENCE_CSV = PROJECT_ROOT.parent / "wikipedia_raw_vs_weighted_pageviews(in).csv"
WIKI_NORM_GLOBAL_SCALE  = 0.918  # additional global correction factor

# ---------------------------------------------------------------------------
# Date range
# ---------------------------------------------------------------------------

START_DATE = "2021-01"  # inclusive lower bound for all monthly data (YYYY-MM)
START_YEAR = "2021"     # inclusive lower bound for yearly data (YYYY)

# ---------------------------------------------------------------------------
# Source toggles (can be overridden via --skip-* CLI flags)
# ---------------------------------------------------------------------------

SOURCES_ENABLED: dict[str, bool] = {
    "who":           True,
    "google_trends": True,
    "open_dengue":   True,
    "wikipedia":     True,
}

# ---------------------------------------------------------------------------
# WHO fetch configuration
# ---------------------------------------------------------------------------

# Layer-1: try a direct API/download endpoint first.
# Set this to the URL if a stable endpoint is discovered by inspecting the
# WHO Shiny dashboard's network traffic (Inspect → Network → XHR).
# Leave as None to skip layer-1 and fall through to layer-2.
WHO_DIRECT_URL: str | None = None

# Layer-2: direct download link on the WHO page (if a CSV/Excel button exists).
# Set after manually inspecting https://worldhealthorg.shinyapps.io/dengue_global/
WHO_DOWNLOAD_URL: str | None = None

# Layer-3 (HTML): the Shiny dashboard URL used for HTML parsing fallback.
WHO_DASHBOARD_URL = "https://worldhealthorg.shinyapps.io/dengue_global/"

# ---------------------------------------------------------------------------
# OpenDengue configuration
# ---------------------------------------------------------------------------

# Primary: GitHub raw CSV (stable, version-controlled)
OPENDENGUE_GITHUB_CSV = (
    "https://raw.githubusercontent.com/OpenDengue/master-repo/"
    "main/data/releases/V1.3/National_extract_V1_3.zip"
)

# Fallback: OpenDengue data page (inspect for alternative download links)
OPENDENGUE_DATA_PAGE = "https://opendengue.org/data.html"

# ---------------------------------------------------------------------------
# Google Trends configuration
# ---------------------------------------------------------------------------

GOOGLE_TRENDS_GEO      = "IN"          # India
GOOGLE_TRENDS_KEYWORDS = ["dengue fever", "dengue vaccine"]
GOOGLE_TRENDS_SLEEP_MIN = 1.0          # seconds between keyword requests
GOOGLE_TRENDS_SLEEP_MAX = 3.0
GOOGLE_TRENDS_BACKOFF_INITIAL = 30     # seconds for first 429 retry
GOOGLE_TRENDS_MAX_RETRIES = 5

# ---------------------------------------------------------------------------
# Wikimedia configuration
# ---------------------------------------------------------------------------

WIKIMEDIA_BASE_URL = "https://wikimedia.org/api/rest_v1"
WIKIMEDIA_SLEEP    = 0.5               # seconds between API calls
WIKIMEDIA_MAX_RETRIES = 4

# ---------------------------------------------------------------------------
# Schema: column names for monthly_data.csv and yearly_data.csv
# ---------------------------------------------------------------------------

MONTHLY_COLUMNS = [
    "date",
    "who_cases_monthly",
    "google_trends_dengue_fever",
    "google_trends_dengue_vaccine",
    "wikipedia_total_dengue_views",
    "wikipedia_total_dengue_views_normalized",
    "wikipedia_mosquito_views_total",
]

YEARLY_COLUMNS = [
    "year",
    "open_dengue_national_yearly",
]

# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def normalize_monthly_date(raw: str) -> str:
    """
    Normalize a date string to YYYY-MM format.

    Accepts: 'YYYY-MM', 'YYYY-MM-DD', 'YYYY-MM-01', 'YYYYMM', 'YYYY/MM', etc.
    Raises ValueError if the string cannot be parsed.
    """
    raw = str(raw).strip()
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y-%m", "%Y/%m", "%Y%m"):
        try:
            dt = datetime.strptime(raw[:len(fmt.replace("%Y", "0000").replace("%m", "00").replace("%d", "00"))], fmt)
            return dt.strftime("%Y-%m")
        except ValueError:
            continue
    # last-ditch: try pandas
    try:
        import pandas as pd
        dt = pd.to_datetime(raw)
        return dt.strftime("%Y-%m")
    except Exception:
        raise ValueError(f"Cannot parse monthly date: {raw!r}")


def normalize_year(raw: str | int) -> str:
    """Normalize a year value to a 4-digit YYYY string."""
    year = str(raw).strip()
    if len(year) == 4 and year.isdigit():
        return year
    try:
        import pandas as pd
        return str(pd.to_datetime(year).year)
    except Exception:
        raise ValueError(f"Cannot parse year: {raw!r}")


def ensure_dirs() -> None:
    """Create all required directories if they do not exist."""
    for d in [
        DATA_RAW, DATA_INTERIM, DATA_PROCESSED, LOGS_DIR,
        RAW_WHO, RAW_GOOGLE, RAW_OPENDENGUE, RAW_WIKIPEDIA,
    ]:
        d.mkdir(parents=True, exist_ok=True)
