"""Paired feature-family ablation and append-only promotion decisions."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

from .audit import assert_common_grid, case_key
from .metrics import quantile_column, validate_prediction_rows, weighted_interval_score


@dataclass(frozen=True)
class FamilyDecision:
    family: str
    state: str
    n_cases: int
    n_origins: int
    mean_wis_difference: float
    mean_mae_difference: float
    improved_origin_proportion: float
    bootstrap_95_lower: float
    bootstrap_95_upper: float
    drop_best_origin_wis_difference: float
    coverage_difference: float
    warnings: Tuple[str, ...]

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def _coverage(rows: Sequence[Mapping[str, object]], lower: float, upper: float) -> float:
    return mean(
        float(row[quantile_column(lower)])
        <= float(row["actual"])
        <= float(row[quantile_column(upper)])
        for row in rows
    )


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    index = round((len(ordered) - 1) * probability)
    return ordered[index]


def _bootstrap_origin_means(
    origin_differences: Mapping[str, float], repetitions: int, seed: int
) -> Tuple[float, float]:
    origins = sorted(origin_differences)
    generator = random.Random(seed)
    samples = [
        mean(origin_differences[generator.choice(origins)] for _ in origins)
        for _ in range(repetitions)
    ]
    return _percentile(samples, 0.025), _percentile(samples, 0.975)


def compare_feature_family(
    family: str,
    baseline_rows: Iterable[Mapping[str, object]],
    candidate_rows: Iterable[Mapping[str, object]],
    *,
    confirmation_frozen: bool = False,
    all_audit_gates_passed: bool = False,
    coverage_interval: Tuple[float, float] = (0.1, 0.9),
    bootstrap_repetitions: int = 1000,
    seed: int = 20261109,
) -> FamilyDecision:
    baseline = validate_prediction_rows(baseline_rows)
    candidate = validate_prediction_rows(candidate_rows)
    assert_common_grid(baseline, candidate)
    baseline_by_key = {case_key(row): row for row in baseline}
    candidate_by_key = {case_key(row): row for row in candidate}

    wis_differences: List[float] = []
    mae_differences: List[float] = []
    by_origin: Dict[str, List[float]] = {}
    for key in sorted(baseline_by_key):
        base, cand = baseline_by_key[key], candidate_by_key[key]
        if float(base["actual"]) != float(cand["actual"]):
            raise ValueError(f"actual values differ for case {key}")
        wis_difference = weighted_interval_score(cand) - weighted_interval_score(base)
        mae_difference = abs(float(cand["actual"]) - float(cand["point"])) - abs(
            float(base["actual"]) - float(base["point"])
        )
        wis_differences.append(wis_difference)
        mae_differences.append(mae_difference)
        by_origin.setdefault(key[0], []).append(wis_difference)

    origin_differences = {origin: mean(values) for origin, values in by_origin.items()}
    improved_proportion = mean(value < 0 for value in origin_differences.values())
    lower, upper = _bootstrap_origin_means(
        origin_differences, bootstrap_repetitions, seed
    )
    best_origin = min(origin_differences, key=origin_differences.get)
    without_best = [value for origin, value in origin_differences.items() if origin != best_origin]
    drop_best = mean(without_best) if without_best else float("inf")
    coverage_difference = _coverage(candidate, *coverage_interval) - _coverage(
        baseline, *coverage_interval
    )
    mean_wis = mean(wis_differences)
    warnings = []
    if drop_best >= 0:
        warnings.append("event_concentration")
    if coverage_difference < -0.05:
        warnings.append("coverage_degradation")
    if improved_proportion < 0.6:
        warnings.append("fewer_than_60_percent_origins_improved")
    if upper >= 0:
        warnings.append("uncertainty_includes_no_improvement")
    horizon_differences: Dict[int, List[float]] = {}
    for key, difference in zip(sorted(baseline_by_key), wis_differences):
        horizon_differences.setdefault(key[2], []).append(difference)
    if mean_wis < 0 and any(mean(values) >= 0 for values in horizon_differences.values()):
        warnings.append("material_horizon_loss")

    robust = not warnings and mean_wis < 0
    if robust and confirmation_frozen and all_audit_gates_passed:
        state = "promoted"
    elif mean_wis < 0:
        state = "challenger"
    else:
        state = "rejected"
    return FamilyDecision(
        family=family,
        state=state,
        n_cases=len(baseline),
        n_origins=len(origin_differences),
        mean_wis_difference=mean_wis,
        mean_mae_difference=mean(mae_differences),
        improved_origin_proportion=improved_proportion,
        bootstrap_95_lower=lower,
        bootstrap_95_upper=upper,
        drop_best_origin_wis_difference=drop_best,
        coverage_difference=coverage_difference,
        warnings=tuple(warnings),
    )


def append_decision(path: Path, decision: FamilyDecision) -> None:
    """Append a decision without rewriting previous feature-family outcomes."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(decision.to_dict(), sort_keys=True) + "\n")
