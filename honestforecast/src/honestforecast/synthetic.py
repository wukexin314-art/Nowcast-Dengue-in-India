"""Run the Milestone 1 synthetic known-answer vertical slice."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from datetime import date
from pathlib import Path
from typing import Dict, Iterable, List, Mapping

from .audit import (
    AuditError,
    assert_common_grid,
    assert_no_future_availability,
    assert_no_future_vintage,
)
from .comparison import compare_feature_family
from .evidence import render_evidence_card
from .features import fit_fold_transform
from .metrics import MetricError, QUANTILE_LEVELS, quantile_column, summarize_predictions
from .screening import screen_candidates


def _add_months(value: date, months: int) -> date:
    month_index = value.year * 12 + value.month - 1 + months
    return date(month_index // 12, month_index % 12 + 1, 1)


def _actual(month_index: int) -> float:
    seasonal = 20.0 * math.sin(2.0 * math.pi * month_index / 12.0)
    return round(100.0 + month_index + seasonal, 6)


def _row(origin_index: int, horizon: int) -> Dict[str, object]:
    target_index = origin_index + horizon
    actual = _actual(target_index)
    point = actual
    row: Dict[str, object] = {
        "origin": _add_months(date(2021, 1, 1), origin_index).isoformat(),
        "target_date": _add_months(date(2021, 1, 1), target_index).isoformat(),
        "horizon": horizon,
        "actual": actual,
        "point": point,
    }
    offsets = {0.025: -20.0, 0.1: -10.0, 0.25: -5.0, 0.5: 0.0, 0.75: 5.0, 0.9: 10.0, 0.975: 20.0}
    for level in QUANTILE_LEVELS:
        row[quantile_column(level)] = point + offsets[level]
    return row


def clean_predictions() -> List[Dict[str, object]]:
    return [_row(origin, horizon) for origin in range(24, 32) for horizon in (1, 2)]


def _shifted_predictions(errors: List[float]) -> List[Dict[str, object]]:
    rows = [_row(origin, 1) for origin in range(24, 24 + len(errors))]
    offsets = {0.025: -3.0, 0.1: -2.0, 0.25: -1.0, 0.5: 0.0, 0.75: 1.0, 0.9: 2.0, 0.975: 3.0}
    for row, error in zip(rows, errors):
        row["point"] = float(row["actual"]) + error
        for level in QUANTILE_LEVELS:
            row[quantile_column(level)] = float(row["point"]) + offsets[level]
    return rows


def null_feature_experiment(repetitions: int = 40) -> Dict[str, object]:
    """Compare naive post-hoc lag searching with registered/FDR screening."""
    naive_promotions = 0
    honest_promotions = 0
    warning_detected = False
    for seed in range(repetitions):
        generator = random.Random(9173 + seed)
        feature = [generator.gauss(0, 1) for _ in range(48)]
        target = [generator.gauss(0, 1) for _ in range(48)]
        candidates = {}
        for lag in range(12):
            rows = []
            for index in range(lag, 48):
                rows.append(
                    {
                        "date": _add_months(date(2020, 1, 1), index).isoformat(),
                        "feature": feature[index - lag],
                        "target": target[index],
                    }
                )
            candidates[f"null_lag_{lag}"] = rows
        screened = screen_candidates(
            candidates, development_end="2023-12-01", registered_variants=3
        )
        warning_detected = warning_detected or bool(screened["multiplicity_warning"])
        if any(result["circular_pvalue"] <= 0.05 for result in screened["results"]):
            naive_promotions += 1
        if any(result["eligible"] for result in screened["results"]):
            honest_promotions += 1
    return {
        "repetitions": repetitions,
        "naive_promotions": naive_promotions,
        "honest_promotions": honest_promotions,
        "naive_false_promotion_rate": naive_promotions / repetitions,
        "honest_false_promotion_rate": honest_promotions / repetitions,
        "multiplicity_warning_detected": warning_detected,
    }


def _expect_failure(name: str, error_type, function) -> str:
    try:
        function()
    except error_type:
        return "detected"
    raise AssertionError(f"{name} was not detected")


def run_scenarios() -> Dict[str, str]:
    clean = clean_predictions()
    summarize_predictions(clean)
    results = {"S0_clean": "passed"}
    results["S1_future_availability"] = _expect_failure(
        "future availability",
        AuditError,
        lambda: assert_no_future_availability(
            [
                {
                    "feature_id": "leaked_proxy",
                    "origin": "2023-01-01",
                    "available_at": "2023-02-01",
                }
            ]
        ),
    )
    results["S5_unequal_origins"] = _expect_failure(
        "unequal origins",
        AuditError,
        lambda: assert_common_grid(clean, clean[2:]),
    )
    crossing = [dict(row) for row in clean]
    crossing[0][quantile_column(0.025)] = crossing[0][quantile_column(0.1)] + 1.0
    results["S6_crossing_quantiles"] = _expect_failure(
        "crossing quantiles",
        MetricError,
        lambda: summarize_predictions(crossing),
    )
    results["S2_latest_revision_as_old_vintage"] = _expect_failure(
        "latest revised history substituted for an old vintage",
        AuditError,
        lambda: assert_no_future_vintage(
            [
                {
                    "feature_id": "revised_proxy",
                    "origin": "2023-01-01",
                    "vintage_at": "2024-01-01",
                }
            ]
        ),
    )
    null_experiment = null_feature_experiment()
    if not null_experiment["multiplicity_warning_detected"]:
        raise AssertionError("S3 multiplicity warning was not detected")
    if null_experiment["honest_false_promotion_rate"] > null_experiment["naive_false_promotion_rate"]:
        raise AssertionError("S3 honest false-promotion rate exceeded naive rate")
    results["S3_post_hoc_null_lags"] = "detected"
    concentrated = compare_feature_family(
        "outbreak_only",
        _shifted_predictions([20.0] + [1.0] * 7),
        _shifted_predictions([0.0] + [2.0] * 7),
        bootstrap_repetitions=200,
    )
    if "event_concentration" not in concentrated.warnings:
        raise AssertionError("S8 event concentration was not detected")
    results["S8_single_outbreak_gain"] = "detected"
    return results


def _write_csv(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    materialized = list(rows)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(materialized[0]))
        writer.writeheader()
        writer.writerows(materialized)


def run(output_dir: Path) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = clean_predictions()
    metrics = summarize_predictions(predictions)
    scenarios = run_scenarios()
    null_experiment = null_feature_experiment()
    feature_rows = [
        {
            "date": _add_months(date(2020, 1, 1), index).isoformat(),
            "value": None if index == 5 else float(index),
        }
        for index in range(24)
    ]
    _, lineage = fit_fold_transform(
        feature_rows,
        train_end="2021-05-01",
        output_feature_id="proxy_log1p",
        source_feature_id="proxy",
        family="synthetic_proxy",
        source_version="1",
        availability_rule="available_at <= origin",
        transform="log1p",
    )
    receipt = {
        "study_id": "synthetic-delayed-surveillance",
        "experiment_id": "milestone-2-known-answer",
        "metrics_recomputed_from_predictions": True,
        "scenario_results": scenarios,
        "null_feature_experiment": null_experiment,
        "lineage": lineage.to_dict(),
    }

    _write_csv(output_dir / "predictions.csv", predictions)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "audit_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "feature_lineage.json").write_text(
        json.dumps(lineage.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "evidence_card.md").write_text(
        render_evidence_card(
            receipt["study_id"], receipt["experiment_id"], metrics, scenarios
        ),
        encoding="utf-8",
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("build/synthetic"))
    args = parser.parse_args()
    receipt = run(args.output_dir)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
