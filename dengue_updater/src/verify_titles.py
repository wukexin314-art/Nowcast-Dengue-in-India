"""
verify_titles.py — Validate Wikipedia page titles before bulk fetching.

For each (language, title) pair in wiki_title_map.yaml, this module
checks that the Wikimedia pageviews API returns HTTP 200 for at least
one month.  Invalid titles are logged and skipped; they do NOT abort
the run.

Usage (standalone):
    python src/verify_titles.py
"""

from __future__ import annotations

import logging
import time
from typing import Any

import requests
import yaml

import config

logger = logging.getLogger(__name__)


def _load_title_map() -> dict[str, dict[str, str]]:
    """Load and return the wiki_title_map.yaml contents."""
    with open(config.WIKI_TITLE_MAP_YAML, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError("wiki_title_map.yaml must be a YAML mapping")
    return data


def _build_pageviews_url(lang: str, title: str, yyyymm: str = "202101") -> str:
    """
    Build a Wikimedia monthly pageviews URL for a single article.

    Args:
        lang:   BCP-47 language code, e.g. 'hi'
        title:  Wikipedia article title (underscores for spaces)
        yyyymm: A month to probe (YYYYMM format), default 2021-01

    Returns:
        Full URL string.
    """
    encoded = requests.utils.quote(title, safe="")
    project = f"{lang}.wikipedia"
    # Wikimedia monthly API requires at least one full month between start and end.
    # Use the probe month as start and the following month as end.
    year, month = int(yyyymm[:4]), int(yyyymm[4:])
    next_month = month + 1 if month < 12 else 1
    next_year  = year if month < 12 else year + 1
    start = f"{yyyymm}01"
    end   = f"{next_year}{next_month:02d}01"
    return (
        f"{config.WIKIMEDIA_BASE_URL}/metrics/pageviews/per-article/"
        f"{project}/all-access/all-agents/{encoded}/monthly/{start}/{end}"
    )


def verify_titles(title_map: dict[str, dict[str, str]] | None = None) -> dict[str, dict[str, bool]]:
    """
    Validate every (language, title) pair by making a probe API call.

    Args:
        title_map: Optional pre-loaded title map; loads from YAML if None.

    Returns:
        Nested dict: {topic: {lang_code: True/False}} where True = valid.
    """
    if title_map is None:
        title_map = _load_title_map()

    results: dict[str, dict[str, bool]] = {}

    for topic, lang_map in title_map.items():
        results[topic] = {}
        for lang, title in lang_map.items():
            url = _build_pageviews_url(lang, title)
            try:
                resp = requests.get(url, timeout=10, headers={"User-Agent": "dengue-updater/1.0"})
                ok = resp.status_code == 200
                results[topic][lang] = ok
                status_str = "OK" if ok else f"HTTP {resp.status_code}"
                logger.info("verify_titles | %s / %s (%s): %s", topic, lang, title, status_str)
                if not ok:
                    logger.warning(
                        "verify_titles | INVALID title — topic=%s lang=%s title=%r → HTTP %d",
                        topic, lang, title, resp.status_code,
                    )
            except requests.RequestException as exc:
                results[topic][lang] = False
                logger.warning(
                    "verify_titles | ERROR — topic=%s lang=%s title=%r: %s",
                    topic, lang, title, exc,
                )
            time.sleep(config.WIKIMEDIA_SLEEP)

    return results


def filter_valid_titles(
    title_map: dict[str, dict[str, str]],
    validity: dict[str, dict[str, bool]],
) -> dict[str, dict[str, str]]:
    """
    Return a copy of title_map containing only valid (lang, title) pairs.

    Args:
        title_map: Full title map loaded from YAML.
        validity:  Output of verify_titles().

    Returns:
        Filtered title map with only valid entries.
    """
    filtered: dict[str, dict[str, str]] = {}
    for topic, lang_map in title_map.items():
        filtered[topic] = {
            lang: title
            for lang, title in lang_map.items()
            if validity.get(topic, {}).get(lang, False)
        }
    return filtered


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    title_map = _load_title_map()
    validity  = verify_titles(title_map)

    print("\n=== Title Validation Results ===")
    all_ok = True
    for topic, lang_map in validity.items():
        for lang, ok in lang_map.items():
            mark = "✓" if ok else "✗"
            title = title_map[topic][lang]
            print(f"  [{mark}] {topic}/{lang}: {title}")
            if not ok:
                all_ok = False
    if all_ok:
        print("\nAll titles are valid.")
    else:
        print("\nWARNING: Some titles are invalid — they will be skipped during fetching.")
