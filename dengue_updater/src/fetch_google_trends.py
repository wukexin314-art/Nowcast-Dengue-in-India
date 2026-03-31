"""
fetch_google_trends.py — Fetch Google Trends monthly data for India.

Fetches both 'dengue fever' and 'dengue vaccine' keywords for India,
covering the full window from START_DATE to today.  Always re-fetches
the full historical window so that the relative index values remain
consistent (pytrends re-normalizes based on the request window).

Failure semantics (STRICT):
    A failed fetch is a run-level failure, NOT real missing data.
    On failure, existing persisted monthly_data.csv values are NEVER
    touched.  Failure is recorded in logs and run summary only.

Returns:
    dict[str, pd.DataFrame | None]
    Keys: 'dengue_fever', 'dengue_vaccine'
    Values: DataFrame(date, value) or None on failure for that keyword.
"""

from __future__ import annotations

import logging
import random
import time
from datetime import datetime
from typing import Optional

import pandas as pd

import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def fetch_google_trends() -> dict[str, Optional[pd.DataFrame]]:
    """
    Fetch Google Trends monthly index for dengue fever and dengue vaccine
    in India, covering the full history from config.START_DATE to today.

    Returns:
        dict with keys 'dengue_fever' and 'dengue_vaccine'.
        Each value is a DataFrame(date YYYY-MM, value float) or None on failure.
    """
    try:
        from pytrends.request import TrendReq
    except ImportError:
        logger.error(
            "fetch_google_trends | 'pytrends' not installed.  Run: pip install pytrends"
        )
        return {"dengue_fever": None, "dengue_vaccine": None}

    results: dict[str, Optional[pd.DataFrame]] = {}

    for keyword in config.GOOGLE_TRENDS_KEYWORDS:
        col_key = keyword.replace(" ", "_")
        df = _fetch_single_keyword(TrendReq, keyword)
        results[col_key] = df

        if df is not None:
            # Save raw output
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            raw_path = config.RAW_GOOGLE / f"{col_key}_{ts}.csv"
            df.to_csv(raw_path, index=False)
            logger.info(
                "fetch_google_trends | '%s': %d rows saved to %s",
                keyword, len(df), raw_path,
            )

        # Rate limiting between keywords
        sleep_secs = random.uniform(
            config.GOOGLE_TRENDS_SLEEP_MIN,
            config.GOOGLE_TRENDS_SLEEP_MAX,
        )
        logger.debug("fetch_google_trends | Sleeping %.1fs before next keyword", sleep_secs)
        time.sleep(sleep_secs)

    return results


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fetch_single_keyword(
    TrendReq,
    keyword: str,
) -> Optional[pd.DataFrame]:
    """
    Fetch a single keyword with exponential backoff retry.

    Args:
        TrendReq: pytrends TrendReq class (passed in to avoid repeated import).
        keyword:  Search keyword string.

    Returns:
        DataFrame(date, value) or None on unrecoverable failure.
    """
    start_str = f"{config.START_DATE.replace('-', '')[:6]}"  # YYYYMM
    # pytrends timeframe format: 'YYYY-MM-DD YYYY-MM-DD'
    start_date = f"{config.START_DATE}-01"
    end_date   = datetime.now().strftime("%Y-%m-%d")
    timeframe  = f"{start_date} {end_date}"

    last_exc: Optional[Exception] = None

    for attempt in range(1, config.GOOGLE_TRENDS_MAX_RETRIES + 1):
        try:
            pytrends = TrendReq(
                hl="en-US",
                tz=330,                   # IST offset
                timeout=(10, 25),
                retries=0,                # we handle retries ourselves
            )
            pytrends.build_payload(
                kw_list=[keyword],
                cat=0,
                timeframe=timeframe,
                geo=config.GOOGLE_TRENDS_GEO,
                gprop="",
            )
            raw = pytrends.interest_over_time()

            if raw is None or raw.empty:
                logger.warning(
                    "fetch_google_trends | '%s': Empty result from pytrends (attempt %d/%d)",
                    keyword, attempt, config.GOOGLE_TRENDS_MAX_RETRIES,
                )
                return None

            # Drop isPartial=True rows
            if "isPartial" in raw.columns:
                n_before = len(raw)
                raw = raw[~raw["isPartial"]].copy()
                n_dropped = n_before - len(raw)
                if n_dropped:
                    logger.info(
                        "fetch_google_trends | '%s': Dropped %d isPartial rows",
                        keyword, n_dropped,
                    )

            if raw.empty:
                logger.warning(
                    "fetch_google_trends | '%s': No rows remain after dropping isPartial",
                    keyword,
                )
                return None

            # Build clean DataFrame
            raw = raw.reset_index()
            raw["date"] = pd.to_datetime(raw["date"]).dt.strftime("%Y-%m")
            df = pd.DataFrame({
                "date":  raw["date"],
                "value": pd.to_numeric(raw[keyword], errors="coerce"),
            })
            df = df.dropna(subset=["date", "value"])
            df = df[df["date"] >= config.START_DATE]
            df = df.sort_values("date").drop_duplicates(subset=["date"]).reset_index(drop=True)

            logger.info(
                "fetch_google_trends | '%s': Fetched %d rows (attempt %d)",
                keyword, len(df), attempt,
            )
            return df

        except Exception as exc:
            last_exc = exc
            exc_str = str(exc).lower()

            # Detect rate-limit / quota errors
            if any(token in exc_str for token in ("429", "too many", "quota", "rate")):
                backoff = config.GOOGLE_TRENDS_BACKOFF_INITIAL * (2 ** (attempt - 1))
                logger.warning(
                    "fetch_google_trends | '%s': Rate limited (attempt %d/%d). "
                    "Sleeping %ds before retry.",
                    keyword, attempt, config.GOOGLE_TRENDS_MAX_RETRIES, backoff,
                )
                time.sleep(backoff)
            else:
                logger.warning(
                    "fetch_google_trends | '%s': Error on attempt %d/%d: %s",
                    keyword, attempt, config.GOOGLE_TRENDS_MAX_RETRIES, exc,
                )
                if attempt < config.GOOGLE_TRENDS_MAX_RETRIES:
                    time.sleep(random.uniform(2, 5))

    logger.error(
        "fetch_google_trends | '%s': All %d attempts failed. Last error: %s\n"
        "Existing stored values in monthly_data.csv will NOT be modified.",
        keyword, config.GOOGLE_TRENDS_MAX_RETRIES, last_exc,
    )
    return None
