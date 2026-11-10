"""Forecast-safe feature construction with explicit as-of lineage."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from datetime import date
from statistics import median
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .audit import AuditError, assert_no_future_availability


class FeatureError(ValueError):
    """Raised when a feature cannot be constructed without ambiguity."""


REGISTERED_TRANSFORMS = {
    "identity",
    "log1p",
    "signed_log1p",
    "lag_1",
    "lag_2",
    "lag_12",
    "trailing_mean_2",
    "trailing_mean_3",
    "momentum_1",
    "fourier_sin_1",
    "fourier_cos_1",
    "fourier_sin_2",
    "fourier_cos_2",
}


@dataclass(frozen=True)
class FeatureLineage:
    output_feature_id: str
    source_feature_id: str
    family: str
    source_version: str
    availability_rule: str
    transform: str
    fit_start: str
    fit_end: str
    parameters: Mapping[str, object]

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def _date(value: object, field: str) -> date:
    try:
        return date.fromisoformat(str(value))
    except ValueError as exc:
        raise FeatureError(f"{field} must be an ISO date: {value}") from exc


def _number(value: object, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise FeatureError(f"{field} must be numeric") from exc
    if not math.isfinite(result):
        raise FeatureError(f"{field} must be finite")
    return result


def as_of_join(
    requests: Iterable[Mapping[str, object]],
    observations: Iterable[Mapping[str, object]],
) -> List[Dict[str, object]]:
    """Select the latest revision that was available at each requested origin."""
    indexed: Dict[Tuple[str, str], List[Mapping[str, object]]] = {}
    revision_keys = set()
    for row in observations:
        feature_id = str(row.get("feature_id", "")).strip()
        observation_date = str(row.get("observation_date", "")).strip()
        if not feature_id or not observation_date:
            raise FeatureError("observations require feature_id and observation_date")
        _date(observation_date, "observation_date")
        _date(row.get("available_at"), "available_at")
        _number(row.get("value"), "value")
        revision_key = (feature_id, observation_date, str(row["available_at"]))
        if revision_key in revision_keys:
            raise FeatureError(f"duplicate feature revision: {revision_key}")
        revision_keys.add(revision_key)
        indexed.setdefault((feature_id, observation_date), []).append(row)

    output: List[Dict[str, object]] = []
    for request in requests:
        feature_id = str(request.get("feature_id", "")).strip()
        observation_date = str(request.get("observation_date", "")).strip()
        origin_text = str(request.get("origin", "")).strip()
        if not feature_id or not observation_date or not origin_text:
            raise FeatureError("requests require feature_id, observation_date, and origin")
        origin = _date(origin_text, "origin")
        candidates = indexed.get((feature_id, observation_date), [])
        usable = [row for row in candidates if _date(row["available_at"], "available_at") <= origin]
        selected = max(usable, key=lambda row: _date(row["available_at"], "available_at"), default=None)
        joined = dict(request)
        if selected is None:
            joined.update(
                value=None,
                available_at=None,
                source_version=None,
                vintage_status="unavailable_at_origin",
            )
        else:
            later_revision = any(
                _date(row["available_at"], "available_at") > origin for row in candidates
            )
            joined.update(
                value=_number(selected["value"], "value"),
                available_at=str(selected["available_at"]),
                source_version=str(selected.get("source_version", "unversioned")),
                vintage_status=(
                    "as_of_value_with_later_revision" if later_revision else "as_of_value"
                ),
            )
        output.append(joined)

    available = [row for row in output if row["available_at"] is not None]
    if available:
        assert_no_future_availability(available)
    return output


def _add_months(value: date, months: int) -> date:
    month_index = value.year * 12 + value.month - 1 + months
    return date(month_index // 12, month_index % 12 + 1, 1)


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (position - lower) * (ordered[upper] - ordered[lower])


def _derived_value(
    transform: str,
    when: date,
    values: Mapping[date, Optional[float]],
) -> Optional[float]:
    value = values.get(when)
    if transform == "identity":
        return value
    if transform == "log1p":
        if value is None:
            return None
        if value < 0:
            raise FeatureError("log1p requires nonnegative values")
        return math.log1p(value)
    if transform == "signed_log1p":
        return None if value is None else math.copysign(math.log1p(abs(value)), value)
    if transform.startswith("lag_"):
        lag = int(transform.split("_")[1])
        return values.get(_add_months(when, -lag))
    if transform.startswith("trailing_mean_"):
        width = int(transform.rsplit("_", 1)[1])
        window = [values.get(_add_months(when, -offset)) for offset in range(width)]
        return None if any(item is None for item in window) else sum(window) / width
    if transform == "momentum_1":
        previous = values.get(_add_months(when, -1))
        return None if value is None or previous is None else value - previous
    if transform.startswith("fourier_"):
        parts = transform.split("_")
        if len(parts) != 3 or parts[1] not in {"sin", "cos"}:
            raise FeatureError(f"unsupported transform: {transform}")
        harmonic = int(parts[2])
        angle = 2.0 * math.pi * harmonic * (when.month - 1) / 12.0
        return math.sin(angle) if parts[1] == "sin" else math.cos(angle)
    raise FeatureError(f"unsupported transform: {transform}")


def fit_fold_transform(
    rows: Iterable[Mapping[str, object]],
    *,
    train_end: str,
    output_feature_id: str,
    source_feature_id: str,
    family: str,
    source_version: str,
    availability_rule: str,
    transform: str = "identity",
    impute: bool = True,
    clip_quantiles: Optional[Tuple[float, float]] = (0.05, 0.95),
) -> Tuple[List[Dict[str, object]], FeatureLineage]:
    """Fit imputation and clipping on the training fold, then apply to all rows."""
    materialized = [dict(row) for row in rows]
    if not materialized:
        raise FeatureError("no feature rows")
    if transform not in REGISTERED_TRANSFORMS:
        raise FeatureError(f"transform is not registered: {transform}")
    lineage_fields = {
        "output_feature_id": output_feature_id,
        "source_feature_id": source_feature_id,
        "family": family,
        "source_version": source_version,
        "availability_rule": availability_rule,
    }
    missing_lineage = [key for key, value in lineage_fields.items() if not str(value).strip()]
    if missing_lineage:
        raise FeatureError(f"lineage fields are required: {missing_lineage}")
    availability_rows = [
        row for row in materialized if row.get("origin") and row.get("available_at")
    ]
    if availability_rows:
        try:
            assert_no_future_availability(availability_rows)
        except AuditError as exc:
            raise FeatureError(str(exc)) from exc
    dates = [_date(row.get("date"), "date") for row in materialized]
    if len(dates) != len(set(dates)):
        raise FeatureError("feature dates must be unique")
    train_end_date = _date(train_end, "train_end")
    raw_values = {
        when: None if row.get("value") in {None, ""} else _number(row["value"], "value")
        for when, row in zip(dates, materialized)
    }
    derived = {when: _derived_value(transform, when, raw_values) for when in dates}
    training = [value for when, value in derived.items() if when <= train_end_date and value is not None]
    if not training:
        raise FeatureError("training fold has no usable transformed values")
    if clip_quantiles is not None:
        lower_q, upper_q = clip_quantiles
        if not 0 <= lower_q < upper_q <= 1:
            raise FeatureError("clip quantiles must satisfy 0 <= lower < upper <= 1")
        clip_lower, clip_upper = _quantile(training, lower_q), _quantile(training, upper_q)
    else:
        clip_lower = clip_upper = None
    impute_value = median(training) if impute else None

    output: List[Dict[str, object]] = []
    for row, when in zip(materialized, dates):
        value = derived[when]
        was_imputed = value is None and impute
        if was_imputed:
            value = impute_value
        if value is not None and clip_lower is not None and clip_upper is not None:
            value = min(max(value, clip_lower), clip_upper)
        output.append(
            {
                **row,
                "feature_id": output_feature_id,
                "value": value,
                "was_imputed": was_imputed,
                "fit_end": train_end,
            }
        )

    fit_dates = [when for when in dates if when <= train_end_date]
    lineage = FeatureLineage(
        output_feature_id=output_feature_id,
        source_feature_id=source_feature_id,
        family=family,
        source_version=source_version,
        availability_rule=availability_rule,
        transform=transform,
        fit_start=min(fit_dates).isoformat(),
        fit_end=train_end_date.isoformat(),
        parameters={
            "impute": impute,
            "impute_value": impute_value,
            "clip_quantiles": clip_quantiles,
            "clip_lower": clip_lower,
            "clip_upper": clip_upper,
        },
    )
    return output, lineage
