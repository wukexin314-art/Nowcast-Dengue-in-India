"""
fetch_wikipedia.py — Fetch Wikipedia monthly pageviews for dengue and mosquito.

Uses the Wikimedia REST pageviews API:
    GET /metrics/pageviews/per-article/{project}/all-access/all-agents/{article}/monthly/{start}/{end}

For each topic (dengue / mosquito), fetches pageviews per language, saves
per-language intermediate files, then sums across successful languages to
produce the monthly aggregate column.

Returns:
    dict with keys:
      'dengue'   → FetchResult(df, expected, successful, failed_languages)
      'mosquito' → FetchResult(df, expected, successful, failed_languages)

    df columns: date (YYYY-MM), value (float sum of successful languages)
    df is None if ALL languages failed.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import pandas as pd
import requests
import yaml

import config
import verify_titles as vt

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class WikiFetchResult:
    """Result for one topic (dengue or mosquito)."""
    df: Optional[pd.DataFrame]          # (date, value) aggregated; None if all languages failed
    expected_languages: int             # total configured languages
    successful_languages: int           # languages that returned data
    failed_languages: list[str] = field(default_factory=list)
    partial: bool = False               # True if some but not all languages succeeded


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def fetch_wikipedia() -> dict[str, WikiFetchResult]:
    """
    Fetch monthly Wikipedia pageviews for dengue and mosquito topics.

    Steps:
    1. Load title maps from wiki_title_map.yaml
    2. Validate titles (skip invalid ones; log failures)
    3. Fetch per-language monthly pageviews from Wikimedia API
    4. Save per-language intermediate files
    5. Sum across successful languages per month

    Returns:
        dict with keys 'dengue' and 'mosquito', values WikiFetchResult.
    """
    title_map = _load_title_map()
    validity  = vt.verify_titles(title_map)
    valid_map = vt.filter_valid_titles(title_map, validity)

    results: dict[str, WikiFetchResult] = {}

    for topic in ("dengue", "mosquito"):
        lang_map = valid_map.get(topic, {})
        full_map = title_map.get(topic, {})
        expected = len(full_map)
        failed_languages: list[str] = [
            lang for lang in full_map if not validity.get(topic, {}).get(lang, False)
        ]

        if not lang_map:
            logger.error(
                "fetch_wikipedia | topic=%s: ALL %d language titles failed validation. "
                "Existing stored values will be preserved.",
                topic, expected,
            )
            results[topic] = WikiFetchResult(
                df=None,
                expected_languages=expected,
                successful_languages=0,
                failed_languages=list(full_map.keys()),
                partial=False,
            )
            continue

        per_lang_frames, fetch_failed = _fetch_all_languages(topic, lang_map)
        failed_languages += fetch_failed

        successful = len(per_lang_frames)
        partial = (successful > 0) and (successful < expected)

        if not per_lang_frames:
            logger.error(
                "fetch_wikipedia | topic=%s: ALL language fetches failed. "
                "Existing stored values will be preserved.",
                topic,
            )
            results[topic] = WikiFetchResult(
                df=None,
                expected_languages=expected,
                successful_languages=0,
                failed_languages=failed_languages,
                partial=False,
            )
            continue

        if partial:
            logger.warning(
                "fetch_wikipedia | topic=%s: Only %d/%d languages succeeded. "
                "Aggregate will be understated. Failed: %s",
                topic, successful, expected, failed_languages,
            )

        agg_df = _aggregate_languages(per_lang_frames)

        results[topic] = WikiFetchResult(
            df=agg_df,
            expected_languages=expected,
            successful_languages=successful,
            failed_languages=failed_languages,
            partial=partial,
        )
        logger.info(
            "fetch_wikipedia | topic=%s: %d rows aggregated from %d/%d languages",
            topic, len(agg_df), successful, expected,
        )

    return results


# ---------------------------------------------------------------------------
# Per-language fetching
# ---------------------------------------------------------------------------

def _fetch_all_languages(
    topic: str,
    lang_map: dict[str, str],
) -> tuple[list[pd.DataFrame], list[str]]:
    """
    Fetch pageviews for each language in lang_map.

    Returns:
        (list of per-language DataFrames with date/value columns, list of failed lang codes)
    """
    per_lang_frames: list[pd.DataFrame] = []
    failed_langs: list[str] = []

    for lang, title in lang_map.items():
        logger.info("fetch_wikipedia | topic=%s lang=%s: Fetching '%s'", topic, lang, title)
        df = _fetch_single_language(lang, title)

        if df is not None and not df.empty:
            # Save per-language intermediate file
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            interim_path = config.DATA_INTERIM / f"wiki_{topic}_{lang}_{ts}.csv"
            df.to_csv(interim_path, index=False)
            logger.info(
                "fetch_wikipedia | topic=%s lang=%s: %d rows saved to %s",
                topic, lang, len(df), interim_path,
            )
            per_lang_frames.append(df.rename(columns={"value": lang}))
        else:
            logger.warning(
                "fetch_wikipedia | topic=%s lang=%s: Fetch failed or empty — will be excluded from aggregate",
                topic, lang,
            )
            failed_langs.append(lang)

        time.sleep(config.WIKIMEDIA_SLEEP)

    return per_lang_frames, failed_langs


def _fetch_single_language(lang: str, title: str) -> Optional[pd.DataFrame]:
    """
    Fetch monthly pageviews for one language/title pair from Wikimedia API.

    Covers config.START_DATE (inclusive) through the current month.

    Returns:
        DataFrame(date YYYY-MM, value float) or None on failure.
    """
    start_ym = config.START_DATE.replace("-", "") + "01"    # e.g. 20210101
    end_ym   = datetime.now().strftime("%Y%m") + "01"

    encoded = requests.utils.quote(title, safe="")
    project = f"{lang}.wikipedia"
    url = (
        f"{config.WIKIMEDIA_BASE_URL}/metrics/pageviews/per-article/"
        f"{project}/all-access/all-agents/{encoded}/monthly/{start_ym}/{end_ym}"
    )

    for attempt in range(1, config.WIKIMEDIA_MAX_RETRIES + 1):
        try:
            resp = requests.get(url, timeout=20, headers={"User-Agent": "dengue-updater/1.0"})

            if resp.status_code == 404:
                logger.warning(
                    "fetch_wikipedia | lang=%s title=%r: HTTP 404 — page may not exist. Skipping.",
                    lang, title,
                )
                return None

            if resp.status_code in (429, 503):
                backoff = 10 * (2 ** (attempt - 1))
                logger.warning(
                    "fetch_wikipedia | lang=%s: HTTP %d — sleeping %ds (attempt %d/%d)",
                    lang, resp.status_code, backoff, attempt, config.WIKIMEDIA_MAX_RETRIES,
                )
                time.sleep(backoff)
                continue

            resp.raise_for_status()
            data = resp.json()

            rows = _parse_wikimedia_response(data)
            if not rows:
                logger.warning(
                    "fetch_wikipedia | lang=%s title=%r: No items in response",
                    lang, title,
                )
                return None

            df = pd.DataFrame(rows, columns=["date", "value"])
            df = df[df["date"] >= config.START_DATE]
            df = df.sort_values("date").drop_duplicates(subset=["date"]).reset_index(drop=True)
            return df

        except requests.RequestException as exc:
            logger.warning(
                "fetch_wikipedia | lang=%s attempt %d/%d failed: %s",
                lang, attempt, config.WIKIMEDIA_MAX_RETRIES, exc,
            )
            if attempt < config.WIKIMEDIA_MAX_RETRIES:
                time.sleep(config.WIKIMEDIA_SLEEP * 2)

    logger.error(
        "fetch_wikipedia | lang=%s title=%r: All %d attempts failed",
        lang, title, config.WIKIMEDIA_MAX_RETRIES,
    )
    return None


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def _parse_wikimedia_response(data: dict) -> list[tuple[str, float]]:
    """
    Parse the Wikimedia pageviews API response.

    Expected format:
        {"items": [{"timestamp": "2021010100", "views": 1234}, ...]}

    Returns list of (date_str YYYY-MM, views float).
    """
    items = data.get("items", [])
    rows: list[tuple[str, float]] = []
    for item in items:
        ts = str(item.get("timestamp", ""))
        if len(ts) >= 8:
            date = f"{ts[:4]}-{ts[4:6]}"
            views = float(item.get("views", 0))
            rows.append((date, views))
    return rows


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _aggregate_languages(per_lang_frames: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Merge per-language DataFrames on date and sum views.

    Each frame has columns: date, {lang_code}.
    Missing months in a language are treated as 0 for the sum.
    """
    if not per_lang_frames:
        return pd.DataFrame(columns=["date", "value"])

    merged = per_lang_frames[0].copy()
    for df in per_lang_frames[1:]:
        merged = pd.merge(merged, df, on="date", how="outer")

    # Sum across all language columns (fill missing with 0)
    lang_cols = [c for c in merged.columns if c != "date"]
    merged["value"] = merged[lang_cols].fillna(0).sum(axis=1)

    result = merged[["date", "value"]].copy()
    result = result.sort_values("date").reset_index(drop=True)
    return result


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _load_title_map() -> dict[str, dict[str, str]]:
    """Load wiki_title_map.yaml and return its contents."""
    with open(config.WIKI_TITLE_MAP_YAML, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError("wiki_title_map.yaml must be a YAML mapping")
    return data
