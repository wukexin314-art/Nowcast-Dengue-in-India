"""Development-only feature screening with time-preserving null checks."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from datetime import date
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


class ScreeningError(ValueError):
    """Raised when screening inputs are invalid."""


def _rank(values: Sequence[float]) -> List[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    index = 0
    while index < len(order):
        end = index + 1
        while end < len(order) and values[order[end]] == values[order[index]]:
            end += 1
        average_rank = (index + end - 1) / 2.0 + 1.0
        for position in order[index:end]:
            ranks[position] = average_rank
        index = end
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) != len(y) or len(x) < 3:
        raise ScreeningError("Spearman correlation requires equal lengths of at least three")
    x_rank, y_rank = _rank(x), _rank(y)
    x_mean = sum(x_rank) / len(x_rank)
    y_mean = sum(y_rank) / len(y_rank)
    numerator = sum((a - x_mean) * (b - y_mean) for a, b in zip(x_rank, y_rank))
    denominator = math.sqrt(
        sum((a - x_mean) ** 2 for a in x_rank) * sum((b - y_mean) ** 2 for b in y_rank)
    )
    return 0.0 if denominator == 0 else numerator / denominator


def circular_shift_pvalue(x: Sequence[float], y: Sequence[float]) -> float:
    observed = abs(spearman(x, y))
    null = [abs(spearman(x, y[shift:] + y[:shift])) for shift in range(1, len(y))]
    return (1 + sum(value >= observed for value in null)) / (len(null) + 1)


@dataclass(frozen=True)
class ScreenResult:
    feature_id: str
    n_total: int
    n_usable: int
    coverage: float
    spearman: float
    early_spearman: float
    later_spearman: float
    circular_pvalue: float
    adjusted_pvalue: float
    sign_stable: bool
    eligible: bool
    reasons: Tuple[str, ...]

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def _screen(feature_id: str, rows: Iterable[Mapping[str, object]], min_coverage: float) -> ScreenResult:
    materialized = sorted(rows, key=lambda row: str(row.get("date", "")))
    pairs: List[Tuple[float, float]] = []
    for row in materialized:
        if row.get("feature") not in {None, ""} and row.get("target") not in {None, ""}:
            try:
                feature, target = float(row["feature"]), float(row["target"])
            except (TypeError, ValueError) as exc:
                raise ScreeningError("feature and target must be numeric or missing") from exc
            if not math.isfinite(feature) or not math.isfinite(target):
                raise ScreeningError("feature and target must be finite")
            pairs.append((feature, target))
    if len(pairs) < 6:
        raise ScreeningError("screening requires at least six usable observations")
    x, y = [pair[0] for pair in pairs], [pair[1] for pair in pairs]
    split = len(pairs) // 2
    overall = spearman(x, y)
    early = spearman(x[:split], y[:split])
    later = spearman(x[split:], y[split:])
    stable = overall == 0 or (early * later >= 0 and (early == 0 or early * overall > 0))
    coverage = len(pairs) / len(materialized) if materialized else 0.0
    reasons = []
    if coverage < min_coverage:
        reasons.append("coverage_below_threshold")
    if not stable:
        reasons.append("direction_reversal")
    return ScreenResult(
        feature_id=feature_id,
        n_total=len(materialized),
        n_usable=len(pairs),
        coverage=coverage,
        spearman=overall,
        early_spearman=early,
        later_spearman=later,
        circular_pvalue=circular_shift_pvalue(x, y),
        adjusted_pvalue=1.0,
        sign_stable=stable,
        eligible=not reasons,
        reasons=tuple(reasons),
    )


def _bh_adjust(pvalues: Sequence[float]) -> List[float]:
    order = sorted(range(len(pvalues)), key=pvalues.__getitem__)
    adjusted = [1.0] * len(pvalues)
    running = 1.0
    for rank_index in range(len(order) - 1, -1, -1):
        original_index = order[rank_index]
        rank = rank_index + 1
        running = min(running, pvalues[original_index] * len(pvalues) / rank)
        adjusted[original_index] = min(1.0, running)
    return adjusted


def screen_candidates(
    candidates: Mapping[str, Iterable[Mapping[str, object]]],
    *,
    development_end: str,
    min_coverage: float = 0.8,
    alpha: float = 0.05,
    registered_variants: Optional[int] = None,
) -> Dict[str, object]:
    """Screen registered candidates and control displayed p-values with BH-FDR."""
    if not candidates:
        raise ScreeningError("no candidates")
    try:
        cutoff = date.fromisoformat(development_end)
    except ValueError as exc:
        raise ScreeningError("development_end must be an ISO date") from exc
    materialized_candidates = {
        feature_id: list(rows) for feature_id, rows in candidates.items()
    }
    for feature_id, materialized in materialized_candidates.items():
        for row in materialized:
            try:
                row_date = date.fromisoformat(str(row.get("date", "")))
            except ValueError as exc:
                raise ScreeningError(f"invalid date for {feature_id}: {row.get('date')}") from exc
            if row_date > cutoff:
                raise ScreeningError(
                    f"{feature_id} includes {row_date} after development_end {cutoff}"
                )
    results = [
        _screen(feature_id, rows, min_coverage)
        for feature_id, rows in materialized_candidates.items()
    ]
    adjusted = _bh_adjust([result.circular_pvalue for result in results])
    final: List[ScreenResult] = []
    for result, adjusted_pvalue in zip(results, adjusted):
        reasons = list(result.reasons)
        if adjusted_pvalue > alpha:
            reasons.append("time_preserving_null_not_rejected")
        final.append(
            ScreenResult(
                **{
                    **result.to_dict(),
                    "adjusted_pvalue": adjusted_pvalue,
                    "eligible": not reasons,
                    "reasons": tuple(reasons),
                }
            )
        )
    warning = registered_variants is not None and len(candidates) > registered_variants
    return {
        "multiplicity_warning": warning,
        "registered_variants": registered_variants,
        "tested_variants": len(candidates),
        "results": [result.to_dict() for result in final],
    }
