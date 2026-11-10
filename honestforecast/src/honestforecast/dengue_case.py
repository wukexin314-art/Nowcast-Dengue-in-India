"""Run the exploratory India dengue repository-vintage pilot."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
from datetime import date
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

from .comparison import FamilyDecision, compare_feature_family
from .metrics import QUANTILE_LEVELS, quantile_column, summarize_predictions
from .vintages import GitSnapshot, collect_git_snapshots


class DengueCaseError(ValueError):
    """Raised when the public dengue pilot cannot be reproduced."""


def _rows(snapshot: GitSnapshot) -> Dict[str, Dict[str, str]]:
    reader = csv.DictReader(io.StringIO(snapshot.content.decode("utf-8-sig")))
    return {str(row["date"]): row for row in reader}


def _add_months(period: str, months: int) -> str:
    year, month = map(int, period.split("-"))
    index = year * 12 + month - 1 + months
    return f"{index // 12:04d}-{index % 12 + 1:02d}"


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (position - lower) * (ordered[upper] - ordered[lower])


def _solve(matrix: List[List[float]], vector: List[float]) -> List[float]:
    augmented = [row[:] + [value] for row, value in zip(matrix, vector)]
    size = len(vector)
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-12:
            raise DengueCaseError("ridge system is singular")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        augmented[column] = [value / divisor for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            augmented[row] = [
                value - factor * pivot_value
                for value, pivot_value in zip(augmented[row], augmented[column])
            ]
    return [augmented[row][-1] for row in range(size)]


def ridge_predict(
    train_x: Sequence[Sequence[float]],
    train_y: Sequence[float],
    score_x: Sequence[float],
    ridge: float = 1.0,
) -> Tuple[float, List[float]]:
    """Fold-local standardized ridge prediction and training residuals."""
    if len(train_x) != len(train_y) or len(train_x) < 3:
        raise DengueCaseError("ridge requires matching training rows of length at least three")
    width = len(score_x)
    if any(len(row) != width for row in train_x):
        raise DengueCaseError("ridge feature widths differ")
    means = [mean(row[column] for row in train_x) for column in range(width)]
    scales = [pstdev(row[column] for row in train_x) or 1.0 for column in range(width)]
    standardized = [
        [1.0] + [(value - means[column]) / scales[column] for column, value in enumerate(row)]
        for row in train_x
    ]
    score = [1.0] + [
        (value - means[column]) / scales[column] for column, value in enumerate(score_x)
    ]
    size = width + 1
    gram = [
        [sum(row[left] * row[right] for row in standardized) for right in range(size)]
        for left in range(size)
    ]
    for index in range(1, size):
        gram[index][index] += ridge
    rhs = [sum(row[index] * target for row, target in zip(standardized, train_y)) for index in range(size)]
    coefficients = _solve(gram, rhs)
    fitted = [sum(value * coefficient for value, coefficient in zip(row, coefficients)) for row in standardized]
    prediction = sum(value * coefficient for value, coefficient in zip(score, coefficients))
    return prediction, [target - fit for target, fit in zip(train_y, fitted)]


def _value(rows: Mapping[str, Mapping[str, str]], period: str, field: str) -> float:
    try:
        text = rows[period][field]
        if text == "":
            raise KeyError
        return float(text)
    except (KeyError, TypeError, ValueError) as exc:
        raise DengueCaseError(f"missing {field} for {period}") from exc


def _features(
    rows: Mapping[str, Mapping[str, str]], period: str, family: str
) -> List[float]:
    month = int(period.split("-")[1])
    values = [
        math.log1p(_value(rows, _add_months(period, -12), "who_cases_monthly")),
        math.sin(2 * math.pi * (month - 1) / 12),
        math.cos(2 * math.pi * (month - 1) / 12),
    ]
    if family == "digital":
        values.extend(
            [
                _value(rows, period, "google_trends_dengue_fever"),
                _value(rows, period, "google_trends_dengue_vaccine"),
                math.log1p(_value(rows, period, "wikipedia_total_dengue_views")),
                math.log1p(_value(rows, period, "wikipedia_mosquito_views_total")),
            ]
        )
    return values


def _model_prediction(
    rows: Mapping[str, Mapping[str, str]], target: str, family: str
) -> Tuple[float, List[float], int]:
    train_x, train_y = [], []
    for period in sorted(rows):
        if period >= target or not rows[period].get("who_cases_monthly"):
            continue
        try:
            features = _features(rows, period, family)
        except DengueCaseError:
            continue
        train_x.append(features)
        train_y.append(math.log1p(float(rows[period]["who_cases_monthly"])))
    score_x = _features(rows, target, family)
    prediction, residuals = ridge_predict(train_x, train_y, score_x)
    return prediction, residuals, len(train_y)


def _seasonal_prediction(
    rows: Mapping[str, Mapping[str, str]], target: str
) -> Tuple[float, List[float], int]:
    prediction = math.log1p(_value(rows, _add_months(target, -12), "who_cases_monthly"))
    residuals = []
    for period in sorted(rows):
        if period >= target or not rows[period].get("who_cases_monthly"):
            continue
        try:
            previous = _value(rows, _add_months(period, -12), "who_cases_monthly")
        except DengueCaseError:
            continue
        residuals.append(math.log1p(float(rows[period]["who_cases_monthly"])) - math.log1p(previous))
    return prediction, residuals, len(residuals)


def _prediction_row(
    origin: str,
    target: str,
    actual: float,
    prediction_log: float,
    residuals: Sequence[float],
    n_train: int,
) -> Dict[str, object]:
    if len(residuals) < 6:
        raise DengueCaseError("at least six residuals are required for intervals")
    row: Dict[str, object] = {
        "origin": origin,
        "target_date": f"{target}-01",
        "horizon": 1,
        "actual": actual,
        "point": max(0.0, math.expm1(prediction_log)),
        "n_train": n_train,
    }
    residual_center = _quantile(residuals, 0.5)
    for level in QUANTILE_LEVELS:
        row[quantile_column(level)] = max(
            0.0,
            math.expm1(
                prediction_log + _quantile(residuals, level) - residual_center
            ),
        )
    return row


def _eligible_origins(
    snapshots: Sequence[GitSnapshot], latest_rows: Mapping[str, Mapping[str, str]]
) -> List[Tuple[GitSnapshot, str]]:
    selected: Dict[str, GitSnapshot] = {}
    for snapshot in snapshots:
        if not snapshot.subject.startswith("auto: update dengue data"):
            continue
        rows = _rows(snapshot)
        google_periods = [
            period for period, row in rows.items() if row.get("google_trends_dengue_fever")
        ]
        if not google_periods:
            continue
        target = max(google_periods)
        if rows[target].get("who_cases_monthly") or not latest_rows.get(target, {}).get("who_cases_monthly"):
            continue
        selected.setdefault(target, snapshot)
    return [(snapshot, target) for target, snapshot in sorted(selected.items())]


def run_dengue_pilot(repo: Path, data_path: str, output_dir: Path) -> Dict[str, object]:
    snapshots = collect_git_snapshots(repo, data_path)
    latest_rows = _rows(snapshots[-1])
    origins = _eligible_origins(snapshots, latest_rows)
    if len(origins) < 6:
        raise DengueCaseError("fewer than six scored repository-vintage origins")
    methods: Dict[str, List[Dict[str, object]]] = {
        "seasonal_naive": [],
        "ar": [],
        "ar_digital_asof": [],
        "ar_digital_latest_revised": [],
    }
    for snapshot, target in origins:
        asof_rows = _rows(snapshot)
        actual = float(latest_rows[target]["who_cases_monthly"])
        origin = snapshot.committed_at[:10]
        base_pred, base_residuals, base_n = _seasonal_prediction(asof_rows, target)
        ar_pred, ar_residuals, ar_n = _model_prediction(asof_rows, target, "ar")
        digital_pred, digital_residuals, digital_n = _model_prediction(asof_rows, target, "digital")
        latest_feature_rows = {period: dict(row) for period, row in latest_rows.items()}
        for period, row in latest_feature_rows.items():
            if period in asof_rows:
                row["who_cases_monthly"] = asof_rows[period].get("who_cases_monthly", "")
        latest_pred, latest_residuals, latest_n = _model_prediction(
            latest_feature_rows, target, "digital"
        )
        methods["seasonal_naive"].append(
            _prediction_row(origin, target, actual, base_pred, base_residuals, base_n)
        )
        methods["ar"].append(
            _prediction_row(origin, target, actual, ar_pred, ar_residuals, ar_n)
        )
        methods["ar_digital_asof"].append(
            _prediction_row(origin, target, actual, digital_pred, digital_residuals, digital_n)
        )
        methods["ar_digital_latest_revised"].append(
            _prediction_row(origin, target, actual, latest_pred, latest_residuals, latest_n)
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for method, rows in methods.items():
        with (output_dir / f"predictions_{method}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(rows[0]), lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(rows)
    metrics = {method: summarize_predictions(rows) for method, rows in methods.items()}
    decisions = {
        "target_history_vs_seasonal": compare_feature_family(
            "target_history", methods["seasonal_naive"], methods["ar"], bootstrap_repetitions=1000
        ),
        "digital_vs_target_history": compare_feature_family(
            "digital_proxies", methods["ar"], methods["ar_digital_asof"], bootstrap_repetitions=1000
        ),
    }
    revision_gap = {
        "wis_latest_minus_asof": metrics["ar_digital_latest_revised"]["wis"]
        - metrics["ar_digital_asof"]["wis"],
        "mae_latest_minus_asof": metrics["ar_digital_latest_revised"]["mae"]
        - metrics["ar_digital_asof"]["mae"],
    }
    receipt = {
        "study_id": "india-dengue-repository-vintage-pilot",
        "status": "exploratory",
        "evidence_level": "E4 plus repository-state historical pilot",
        "scored_origins": len(origins),
        "targets": [target for _, target in origins],
        "metrics": metrics,
        "decisions": {name: decision.to_dict() for name, decision in decisions.items()},
        "retrospective_revision_gap": revision_gap,
        "limitations": [
            "repository commit time is not source-native publication time",
            "only seven scored origins are available",
            "prediction intervals use in-sample residual quantiles",
            "no feature family is eligible for confirmation-level promotion",
        ],
    }
    (output_dir / "pilot_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "REPORT.md").write_text(_report(receipt), encoding="utf-8")
    return receipt


def _report(receipt: Mapping[str, object]) -> str:
    metrics = receipt["metrics"]
    decisions = receipt["decisions"]
    gap = receipt["retrospective_revision_gap"]
    lines = [
        "# India dengue repository-vintage pilot",
        "",
        f"Status: **exploratory**; scored origins: {receipt['scored_origins']}.",
        "",
        "| Method | MAE | WIS | 80% coverage |",
        "| --- | ---: | ---: | ---: |",
    ]
    for method in ["seasonal_naive", "ar", "ar_digital_asof", "ar_digital_latest_revised"]:
        row = metrics[method]
        lines.append(
            f"| {method} | {row['mae']:.1f} | {row['wis']:.1f} | {row['coverage_80']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Registered decisions",
            "",
            f"- Target history versus seasonal naive: `{decisions['target_history_vs_seasonal']['state']}`.",
            f"- Digital proxies versus target history: `{decisions['digital_vs_target_history']['state']}`.",
            "",
            "## Revision sensitivity",
            "",
            f"- Latest-revised minus as-of WIS: {gap['wis_latest_minus_asof']:.1f}.",
            f"- Latest-revised minus as-of MAE: {gap['mae_latest_minus_asof']:.1f}.",
            "",
            "Negative values mean a retrospective latest-revised feature history looks better; "
            "positive values mean it looks worse. Either direction shows that feature revisions change the result.",
            "",
            "## Limitations",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in receipt["limitations"])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument(
        "--data-path", default="dengue_updater/data/processed/monthly_data.csv"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            run_dengue_pilot(args.repo, args.data_path, args.output_dir),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
