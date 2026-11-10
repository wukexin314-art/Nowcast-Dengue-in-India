"""Point and hub-style probabilistic forecast validation and scoring."""

from __future__ import annotations

import math
from statistics import mean
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

from .audit import case_keys


class MetricError(ValueError):
    """Raised when predictions cannot support the requested metric."""


QUANTILE_LEVELS: Tuple[float, ...] = (0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975)
INTERVALS: Tuple[Tuple[float, float], ...] = ((0.025, 0.975), (0.1, 0.9), (0.25, 0.75))


def quantile_column(level: float) -> str:
    return f"q{level:g}"


def _number(row: Mapping[str, object], key: str) -> float:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise MetricError(f"{key} must be numeric") from exc
    if not math.isfinite(value):
        raise MetricError(f"{key} must be finite")
    return value


def validate_prediction_rows(rows: Iterable[Mapping[str, object]]) -> List[Mapping[str, object]]:
    materialized = list(rows)
    case_keys(materialized)
    for row in materialized:
        _number(row, "actual")
        _number(row, "point")
        quantiles = [_number(row, quantile_column(level)) for level in QUANTILE_LEVELS]
        if any(left > right for left, right in zip(quantiles, quantiles[1:])):
            raise MetricError(
                "forecast quantiles cross for "
                f"origin={row.get('origin')}, target_date={row.get('target_date')}"
            )
    return materialized


def _interval_score(actual: float, lower: float, upper: float, alpha: float) -> float:
    score = upper - lower
    if actual < lower:
        score += (2.0 / alpha) * (lower - actual)
    elif actual > upper:
        score += (2.0 / alpha) * (actual - upper)
    return score


def weighted_interval_score(row: Mapping[str, object]) -> float:
    actual = _number(row, "actual")
    median = _number(row, quantile_column(0.5))
    total = 0.5 * abs(actual - median)
    for lower_level, upper_level in INTERVALS:
        alpha = 2.0 * lower_level
        lower = _number(row, quantile_column(lower_level))
        upper = _number(row, quantile_column(upper_level))
        total += (alpha / 2.0) * _interval_score(actual, lower, upper, alpha)
    return total / (len(INTERVALS) + 0.5)


def _coverage(rows: Sequence[Mapping[str, object]], lower: float, upper: float) -> float:
    inside = []
    for row in rows:
        actual = _number(row, "actual")
        inside.append(
            _number(row, quantile_column(lower))
            <= actual
            <= _number(row, quantile_column(upper))
        )
    return mean(inside)


def _width(rows: Sequence[Mapping[str, object]], lower: float, upper: float) -> float:
    return mean(
        _number(row, quantile_column(upper)) - _number(row, quantile_column(lower))
        for row in rows
    )


def summarize_predictions(rows: Iterable[Mapping[str, object]]) -> Dict[str, float]:
    valid = validate_prediction_rows(rows)
    return {
        "n_cases": len(valid),
        "mae": mean(abs(_number(row, "actual") - _number(row, "point")) for row in valid),
        "wis": mean(weighted_interval_score(row) for row in valid),
        "coverage_50": _coverage(valid, 0.25, 0.75),
        "coverage_80": _coverage(valid, 0.1, 0.9),
        "coverage_95": _coverage(valid, 0.025, 0.975),
        "width_50": _width(valid, 0.25, 0.75),
        "width_80": _width(valid, 0.1, 0.9),
        "width_95": _width(valid, 0.025, 0.975),
    }
