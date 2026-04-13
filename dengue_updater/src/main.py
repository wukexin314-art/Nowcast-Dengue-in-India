"""
main.py — Orchestrator for the India dengue data updater.

Usage:
    python src/main.py [--skip-who] [--skip-google-trends]
                       [--skip-open-dengue] [--skip-wikipedia]
                       [--dry-run]

Workflow:
    1. Fetch data from each enabled source
    2. Merge into monthly_data.csv and yearly_data.csv (overwrite rules applied)
    3. Export master_data.xlsx
    4. Append per-source rows to update_log.csv
    5. Write timestamped run_summary_{YYYYMMDD_HHMMSS}.json

Failure behavior:
    - Source-level failures do not abort other sources (loose failure mode)
    - Failed sources never overwrite existing stored values
    - Overall run status is PARTIAL if any source fails, SUCCESS if all pass
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd

# Ensure src/ is on the path when called as `python src/main.py`
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
import export_outputs
import fetch_google_trends as fgt
import fetch_open_dengue as fod
import fetch_who as fw
import fetch_wikipedia as fwiki
import update_monthly as um
import update_yearly as uy

logger = logging.getLogger(__name__)

# Status constants
STATUS_SUCCESS  = "SUCCESS"
STATUS_PARTIAL  = "PARTIAL"
STATUS_FAILED   = "FAILED"
STATUS_SKIPPED  = "SKIPPED"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args   = _parse_args()
    run_ts = datetime.now()
    ts_str = run_ts.strftime("%Y%m%d_%H%M%S")

    _setup_logging(run_ts)
    config.ensure_dirs()

    logger.info("=" * 60)
    logger.info("Dengue Updater  run_id=%s", ts_str)
    logger.info("=" * 60)

    # Apply CLI skip flags
    enabled = dict(config.SOURCES_ENABLED)
    if args.skip_who:            enabled["who"]           = False
    if args.skip_google_trends:  enabled["google_trends"] = False
    if args.skip_open_dengue:    enabled["open_dengue"]   = False
    if args.skip_wikipedia:      enabled["wikipedia"]     = False

    log_entries: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Load existing data
    # ------------------------------------------------------------------
    monthly_df = um.load_monthly_data()
    yearly_df  = uy.load_yearly_data()

    # ------------------------------------------------------------------
    # WHO
    # ------------------------------------------------------------------
    who_status = STATUS_SKIPPED
    if enabled["who"]:
        logger.info("--- WHO fetch ---")
        t0 = time.monotonic()
        who_df, who_mode = fw.fetch_who()
        elapsed = time.monotonic() - t0

        if who_df is not None:
            monthly_df, added, updated = um.apply_update(
                monthly_df, who_df, "who_cases_monthly", "who_cases_monthly"
            )
            who_status = STATUS_SUCCESS
            log_entries.append(_log_entry(
                run_ts, "WHO", STATUS_SUCCESS, len(who_df), added, updated,
                fetch_mode=who_mode,
            ))
            logger.info("WHO: SUCCESS (%d rows fetched, +%d new, ~%d updated) in %.1fs",
                        len(who_df), added, updated, elapsed)
        else:
            who_status = STATUS_FAILED
            log_entries.append(_log_entry(
                run_ts, "WHO", STATUS_FAILED, 0, 0, 0,
                fetch_mode=who_mode,
                error=f"All layers failed (mode={who_mode})",
            ))
            logger.warning("WHO: FAILED — existing values preserved")

    # ------------------------------------------------------------------
    # Google Trends
    # ------------------------------------------------------------------
    gt_status = STATUS_SKIPPED
    if enabled["google_trends"]:
        logger.info("--- Google Trends fetch ---")
        t0 = time.monotonic()
        gt_results = fgt.fetch_google_trends()
        elapsed = time.monotonic() - t0

        gt_any_failed = False
        for kw_key, col_name in [
            ("dengue_fever",   "google_trends_dengue_fever"),
            ("dengue_vaccine", "google_trends_dengue_vaccine"),
        ]:
            kw_df = gt_results.get(kw_key)
            if kw_df is not None:
                monthly_df, added, updated = um.apply_update(
                    monthly_df, kw_df, col_name, "value", is_google_trends=True
                )
                log_entries.append(_log_entry(
                    run_ts, f"google_trends_{kw_key}", STATUS_SUCCESS,
                    len(kw_df), added, updated,
                ))
                logger.info("Google Trends '%s': SUCCESS (%d rows)", kw_key, len(kw_df))
            else:
                gt_any_failed = True
                log_entries.append(_log_entry(
                    run_ts, f"google_trends_{kw_key}", STATUS_FAILED, 0, 0, 0,
                    error="Fetch failed — existing stored values unchanged",
                ))
                logger.warning("Google Trends '%s': FAILED — existing values preserved", kw_key)

        gt_status = STATUS_FAILED if gt_any_failed else STATUS_SUCCESS
        logger.info("Google Trends total time: %.1fs", elapsed)

    # ------------------------------------------------------------------
    # OpenDengue
    # ------------------------------------------------------------------
    od_status = STATUS_SKIPPED
    if enabled["open_dengue"]:
        logger.info("--- OpenDengue fetch ---")
        t0 = time.monotonic()
        od_df = fod.fetch_open_dengue()
        elapsed = time.monotonic() - t0

        if od_df is not None:
            yearly_df, added, updated = uy.apply_yearly_update(
                yearly_df, od_df, "open_dengue_national_yearly", "open_dengue_national_yearly"
            )
            od_status = STATUS_SUCCESS
            log_entries.append(_log_entry(
                run_ts, "open_dengue", STATUS_SUCCESS, len(od_df), added, updated,
            ))
            logger.info("OpenDengue: SUCCESS (%d rows fetched, +%d new, ~%d updated) in %.1fs",
                        len(od_df), added, updated, elapsed)
        else:
            od_status = STATUS_FAILED
            log_entries.append(_log_entry(
                run_ts, "open_dengue", STATUS_FAILED, 0, 0, 0,
                error="Download or extraction failed — existing values preserved",
            ))
            logger.warning("OpenDengue: FAILED — existing values preserved")

    # ------------------------------------------------------------------
    # Wikipedia
    # ------------------------------------------------------------------
    wiki_dengue_status   = STATUS_SKIPPED
    wiki_mosquito_status = STATUS_SKIPPED
    wiki_dengue_coverage: dict[str, Any]   = {}
    wiki_mosquito_coverage: dict[str, Any] = {}

    if enabled["wikipedia"]:
        logger.info("--- Wikipedia fetch ---")
        t0 = time.monotonic()
        wiki_results = fwiki.fetch_wikipedia()
        elapsed = time.monotonic() - t0

        # Dengue views
        dengue_res = wiki_results.get("dengue")
        if dengue_res:
            wiki_dengue_coverage = {
                "expected":          dengue_res.expected_languages,
                "successful":        dengue_res.successful_languages,
                "failed_languages":  dengue_res.failed_languages,
            }
            if dengue_res.df is not None:
                monthly_df, added, updated = um.apply_update(
                    monthly_df, dengue_res.df, "wikipedia_total_dengue_views", "value"
                )
                wiki_dengue_status = STATUS_PARTIAL if dengue_res.partial else STATUS_SUCCESS
                log_entries.append(_log_entry(
                    run_ts, "wiki_dengue", wiki_dengue_status,
                    len(dengue_res.df), added, updated,
                    expected_langs=dengue_res.expected_languages,
                    successful_langs=dengue_res.successful_languages,
                    error=", ".join(dengue_res.failed_languages) if dengue_res.failed_languages else "",
                ))
                logger.info(
                    "Wikipedia dengue: %s (%d/%d languages, %d rows)",
                    wiki_dengue_status, dengue_res.successful_languages,
                    dengue_res.expected_languages, len(dengue_res.df),
                )
            else:
                wiki_dengue_status = STATUS_FAILED
                log_entries.append(_log_entry(
                    run_ts, "wiki_dengue", STATUS_FAILED, 0, 0, 0,
                    expected_langs=dengue_res.expected_languages,
                    successful_langs=0,
                    error="All language fetches failed — existing values preserved",
                ))
                logger.warning("Wikipedia dengue: FAILED — existing values preserved")

        # Mosquito views
        mosquito_res = wiki_results.get("mosquito")
        if mosquito_res:
            wiki_mosquito_coverage = {
                "expected":          mosquito_res.expected_languages,
                "successful":        mosquito_res.successful_languages,
                "failed_languages":  mosquito_res.failed_languages,
            }
            if mosquito_res.df is not None:
                monthly_df, added, updated = um.apply_update(
                    monthly_df, mosquito_res.df, "wikipedia_mosquito_views_total", "value"
                )
                wiki_mosquito_status = STATUS_PARTIAL if mosquito_res.partial else STATUS_SUCCESS
                log_entries.append(_log_entry(
                    run_ts, "wiki_mosquito", wiki_mosquito_status,
                    len(mosquito_res.df), added, updated,
                    expected_langs=mosquito_res.expected_languages,
                    successful_langs=mosquito_res.successful_languages,
                    error=", ".join(mosquito_res.failed_languages) if mosquito_res.failed_languages else "",
                ))
                logger.info(
                    "Wikipedia mosquito: %s (%d/%d languages, %d rows)",
                    wiki_mosquito_status, mosquito_res.successful_languages,
                    mosquito_res.expected_languages, len(mosquito_res.df),
                )
            else:
                wiki_mosquito_status = STATUS_FAILED
                log_entries.append(_log_entry(
                    run_ts, "wiki_mosquito", STATUS_FAILED, 0, 0, 0,
                    expected_langs=mosquito_res.expected_languages,
                    successful_langs=0,
                    error="All language fetches failed — existing values preserved",
                ))
                logger.warning("Wikipedia mosquito: FAILED — existing values preserved")

        logger.info("Wikipedia total time: %.1fs", elapsed)

    # ------------------------------------------------------------------
    # Compute wikipedia_total_dengue_views_normalized before saving
    # ------------------------------------------------------------------
    monthly_df = um.add_wiki_normalized_column(monthly_df)

    # ------------------------------------------------------------------
    # Persist processed data
    # ------------------------------------------------------------------
    if not args.dry_run:
        um.save_monthly_data(monthly_df)
        uy.save_yearly_data(yearly_df)
        export_outputs.export_master_xlsx()
    else:
        logger.info("[DRY RUN] Would save %d monthly rows and %d yearly rows",
                    len(monthly_df), len(yearly_df))

    # ------------------------------------------------------------------
    # Append to update_log.csv
    # ------------------------------------------------------------------
    _write_log_entries(log_entries)

    # ------------------------------------------------------------------
    # Write run_summary_{ts}.json
    # ------------------------------------------------------------------
    all_statuses = [
        s for s in [
            who_status, gt_status, od_status,
            wiki_dengue_status, wiki_mosquito_status,
        ]
        if s != STATUS_SKIPPED
    ]

    if all(s == STATUS_SUCCESS for s in all_statuses):
        overall = STATUS_SUCCESS
    elif all(s == STATUS_FAILED for s in all_statuses):
        overall = STATUS_FAILED
    elif any(s in (STATUS_FAILED, STATUS_PARTIAL) for s in all_statuses):
        overall = STATUS_PARTIAL
    else:
        overall = STATUS_SUCCESS

    successful_sources = [name for name, st in [
        ("WHO",            who_status),
        ("google_trends",  gt_status),
        ("open_dengue",    od_status),
        ("wiki_dengue",    wiki_dengue_status),
        ("wiki_mosquito",  wiki_mosquito_status),
    ] if st == STATUS_SUCCESS]

    failed_sources = [name for name, st in [
        ("WHO",            who_status),
        ("google_trends",  gt_status),
        ("open_dengue",    od_status),
        ("wiki_dengue",    wiki_dengue_status),
        ("wiki_mosquito",  wiki_mosquito_status),
    ] if st == STATUS_FAILED]

    partial_sources = [name for name, st in [
        ("wiki_dengue",   wiki_dengue_status),
        ("wiki_mosquito", wiki_mosquito_status),
    ] if st == STATUS_PARTIAL]

    notes_parts = []
    if failed_sources:
        notes_parts.append(f"Failed sources: {', '.join(failed_sources)}")
    if partial_sources:
        notes_parts.append(f"Partial sources: {', '.join(partial_sources)}")
    if "google_trends" in failed_sources:
        notes_parts.append(
            "Google Trends failure: existing stored values in monthly_data.csv unchanged; will retry next run"
        )

    summary = {
        "run_time":           run_ts.isoformat(),
        "overall_status":     overall,
        "successful_sources": successful_sources,
        "failed_sources":     failed_sources,
        "partial_sources":    partial_sources,
        "wiki_dengue_coverage":   wiki_dengue_coverage,
        "wiki_mosquito_coverage": wiki_mosquito_coverage,
        "output_files": {
            "monthly_data": str(config.MONTHLY_DATA_CSV),
            "yearly_data":  str(config.YEARLY_DATA_CSV),
            "master_xlsx":  str(config.MASTER_XLSX),
        },
        "notes": "; ".join(notes_parts) if notes_parts else "",
    }

    summary_path = config.LOGS_DIR / f"run_summary_{ts_str}.json"
    if not args.dry_run:
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Run summary written to %s", summary_path)
    else:
        logger.info("[DRY RUN] Run summary:\n%s", json.dumps(summary, indent=2))

    logger.info("=" * 60)
    logger.info("Overall status: %s", overall)
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def _log_entry(
    run_ts: datetime,
    source_name: str,
    status: str,
    rows_fetched: int,
    rows_added: int,
    rows_updated: int,
    fetch_mode: str = "",
    expected_langs: int | str = "",
    successful_langs: int | str = "",
    error: str = "",
) -> dict[str, Any]:
    return {
        "run_time":            run_ts.isoformat(),
        "source_name":         source_name,
        "status":              status,
        "fetch_mode":          fetch_mode,
        "rows_fetched":        rows_fetched,
        "rows_added":          rows_added,
        "rows_updated":        rows_updated,
        "expected_languages":  expected_langs,
        "successful_languages": successful_langs,
        "error_message":       error,
    }


def _write_log_entries(entries: list[dict[str, Any]]) -> None:
    if not entries:
        return
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_df = pd.DataFrame(entries)
    if config.UPDATE_LOG_CSV.exists():
        log_df.to_csv(config.UPDATE_LOG_CSV, mode="a", header=False, index=False)
    else:
        log_df.to_csv(config.UPDATE_LOG_CSV, index=False)
    logger.info("Log entries appended to %s", config.UPDATE_LOG_CSV)


def _setup_logging(run_ts: datetime) -> None:
    """Configure logging to both console and a timestamped log file."""
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_file = config.LOGS_DIR / f"run_{run_ts.strftime('%Y%m%d_%H%M%S')}.log"

    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, encoding="utf-8"),
    ]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Update India dengue dataset from multiple sources",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/main.py
  python src/main.py --skip-google-trends
  python src/main.py --skip-who --skip-open-dengue
  python src/main.py --dry-run
        """,
    )
    parser.add_argument("--skip-who",            action="store_true", help="Skip WHO fetch")
    parser.add_argument("--skip-google-trends",  action="store_true", help="Skip Google Trends fetch")
    parser.add_argument("--skip-open-dengue",    action="store_true", help="Skip OpenDengue fetch")
    parser.add_argument("--skip-wikipedia",      action="store_true", help="Skip Wikipedia fetch")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch data and log results but do not write output files",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
