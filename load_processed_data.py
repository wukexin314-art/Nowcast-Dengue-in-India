"""
Adapter: reads dengue_updater/data/processed/ and exposes interfaces
compatible with the legacy manual data files.

Functions
---------
load_monthly_raw()   -> pd.DataFrame   wide-format monthly_data.csv as-is
load_yearly_raw()    -> pd.DataFrame   yearly_data.csv as-is
build_master_df()    -> pd.DataFrame   long-format [resolution, date, value, source]
                        mirrors what master_data.csv looked like
get_wiki_dengue_df() -> pd.DataFrame   [Month, Total_Views]
                        mirrors total_dengue_views.csv
get_mosquito_df()    -> pd.DataFrame   [timestamp, TOTAL_MONTHLY_VIEWS]
                        mirrors monthly_mosquito_aggregate.csv

Notes on definition changes
----------------------------
- OpenDengue_State_Aggregated: no longer available in new data; build_master_df()
  only emits OpenDengue_National_Yearly for yearly rows. Code that uses
  yearly_proxy_sources_priority will naturally fall back to "OpenDengue_National_Yearly"
  since it iterates the list and picks the first match.
- Mosquito aggregate: old file used 6 languages (Hindi, Tamil, Marathi, Malayalam,
  Kannada, Gujarati). New data uses 5 languages (Gujarati dropped). Values for
  overlapping months will differ slightly; the column name and interface are
  preserved so downstream code requires no changes.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd

# Locate processed directory relative to this file (project root → dengue_updater/data/processed/)
_THIS_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = _THIS_DIR / "dengue_updater" / "data" / "processed"


def _require_file(p: Path) -> None:
    if not p.exists():
        raise FileNotFoundError(
            f"Required processed data file not found: {p}\n"
            "Run the updater first: python dengue_updater/src/main.py"
        )


# ---------------------------------------------------------------------------
# Low-level loaders (return new format as-is)
# ---------------------------------------------------------------------------

def load_monthly_raw() -> pd.DataFrame:
    """Return monthly_data.csv in its native wide format."""
    p = PROCESSED_DIR / "monthly_data.csv"
    _require_file(p)
    return pd.read_csv(p)


def load_yearly_raw() -> pd.DataFrame:
    """Return yearly_data.csv in its native format."""
    p = PROCESSED_DIR / "yearly_data.csv"
    _require_file(p)
    return pd.read_csv(p)


# ---------------------------------------------------------------------------
# Compatibility adapters (return legacy-compatible formats)
# ---------------------------------------------------------------------------

def build_master_df() -> pd.DataFrame:
    """
    Build a DataFrame that matches the old master_data.csv long format:
      columns: [resolution, date, value, source]

    Monthly column mapping (new → old source name):
      who_cases_monthly             → "WHO"
      google_trends_dengue_fever    → "Google_Trends_Dengue_fever"
      google_trends_dengue_vaccine  → "Google_Trends_Dengue_vaccine"

    Yearly rows:
      open_dengue_national_yearly   → source="OpenDengue_National_Yearly"

    Dates are kept in YYYY-MM (monthly) and YYYY (yearly) string format,
    exactly as they appeared in the old file.
    """
    monthly = load_monthly_raw()
    yearly = load_yearly_raw()

    monthly_col_map = {
        "who_cases_monthly": "WHO",
        "google_trends_dengue_fever": "Google_Trends_Dengue_fever",
        "google_trends_dengue_vaccine": "Google_Trends_Dengue_vaccine",
    }

    rows: list[dict] = []

    # Monthly rows
    for _, row in monthly.iterrows():
        date_val = str(row["date"])  # already YYYY-MM
        for col, source in monthly_col_map.items():
            if col in monthly.columns:
                rows.append(
                    {
                        "resolution": "Monthly",
                        "date": date_val,
                        "value": row[col],
                        "source": source,
                    }
                )

    # Yearly rows
    for _, row in yearly.iterrows():
        raw_year = row.get("year")
        if pd.isna(raw_year):
            continue
        year_str = str(int(float(raw_year)))  # "2021", "2022", …
        val = row.get("open_dengue_national_yearly")
        rows.append(
            {
                "resolution": "Yearly",
                "date": year_str,
                "value": val,
                "source": "OpenDengue_National_Yearly",
            }
        )

    df = pd.DataFrame(rows, columns=["resolution", "date", "value", "source"])
    return df


def get_wiki_dengue_df() -> pd.DataFrame:
    """
    Return a DataFrame with columns [Month, Total_Views],
    matching the old total_dengue_views.csv interface.

    Source: wikipedia_total_dengue_views_normalized column in monthly_data.csv
    (falls back to wikipedia_total_dengue_views if the normalized column is absent).
    """
    monthly = load_monthly_raw()
    col = "wikipedia_total_dengue_views_normalized"
    if col not in monthly.columns:
        col = "wikipedia_total_dengue_views"
    if col not in monthly.columns:
        raise ValueError(
            f"Column 'wikipedia_total_dengue_views_normalized' (and fallback "
            f"'wikipedia_total_dengue_views') not found in monthly_data.csv. "
            f"Available columns: {list(monthly.columns)}"
        )
    out = monthly[["date", col]].copy()
    out = out.rename(columns={"date": "Month", col: "Total_Views"})
    out = out[out["Total_Views"].notna()].reset_index(drop=True)
    return out


def get_mosquito_df() -> pd.DataFrame:
    """
    Return a DataFrame with columns [timestamp, TOTAL_MONTHLY_VIEWS],
    matching the old monthly_mosquito_aggregate.csv interface.

    Source: wikipedia_mosquito_views_total column in monthly_data.csv.

    NOTE: The old file aggregated 6 languages (including Gujarati); the new
    data uses 5 languages (Gujarati dropped). Values will differ slightly for
    overlapping months. The column name and interface are identical.
    """
    monthly = load_monthly_raw()
    col = "wikipedia_mosquito_views_total"
    if col not in monthly.columns:
        raise ValueError(
            f"Column '{col}' not found in monthly_data.csv. "
            f"Available columns: {list(monthly.columns)}"
        )
    out = monthly[["date", col]].copy()
    out = out.rename(columns={"date": "timestamp", col: "TOTAL_MONTHLY_VIEWS"})
    out = out[out["TOTAL_MONTHLY_VIEWS"].notna()].reset_index(drop=True)
    return out
