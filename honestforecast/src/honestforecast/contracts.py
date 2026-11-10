"""Small trust-boundary schemas for a forecasting study."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Tuple


class ContractError(ValueError):
    """Raised when a study or feature contract is incomplete or inconsistent."""


def _required_text(data: Mapping[str, object], key: str) -> str:
    value = str(data.get(key, "")).strip()
    if not value:
        raise ContractError(f"{key} is required")
    return value


@dataclass(frozen=True)
class ForecastContract:
    study_id: str
    target_name: str
    frequency: str
    horizons: Tuple[int, ...]
    primary_metric: str = "WIS"
    development_only_selection: bool = True

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> "ForecastContract":
        raw_horizons = data.get("horizons")
        if not isinstance(raw_horizons, (list, tuple)) or not raw_horizons:
            raise ContractError("horizons must be a non-empty list")
        try:
            horizons = tuple(int(value) for value in raw_horizons)
        except (TypeError, ValueError) as exc:
            raise ContractError("horizons must contain integers") from exc
        if any(value <= 0 for value in horizons) or len(set(horizons)) != len(horizons):
            raise ContractError("horizons must be unique positive integers")

        development_only = data.get("development_only_selection", True)
        if not isinstance(development_only, bool):
            raise ContractError("development_only_selection must be boolean")

        return cls(
            study_id=_required_text(data, "study_id"),
            target_name=_required_text(data, "target_name"),
            frequency=_required_text(data, "frequency"),
            horizons=horizons,
            primary_metric=_required_text(data, "primary_metric"),
            development_only_selection=development_only,
        )


_AVAILABILITY_CLASSES = {
    "known_ahead",
    "observed_with_lag",
    "forecast_required",
    "unverified",
}

_FEATURE_STATES = {
    "proposed",
    "ineligible",
    "screening",
    "candidate",
    "challenger",
    "promoted",
    "rejected",
    "retired",
    "exploratory",
}


@dataclass(frozen=True)
class FeatureSpec:
    feature_id: str
    family: str
    version: str
    availability_class: str
    minimum_reporting_lag: str
    vintage_policy: str
    state: str

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> "FeatureSpec":
        availability = _required_text(data, "availability_class")
        if availability not in _AVAILABILITY_CLASSES:
            raise ContractError(f"unsupported availability_class: {availability}")
        state = _required_text(data, "state")
        if state not in _FEATURE_STATES:
            raise ContractError(f"unsupported feature state: {state}")
        return cls(
            feature_id=_required_text(data, "feature_id"),
            family=_required_text(data, "family"),
            version=_required_text(data, "version"),
            availability_class=availability,
            minimum_reporting_lag=_required_text(data, "minimum_reporting_lag"),
            vintage_policy=_required_text(data, "vintage_policy"),
            state=state,
        )


def load_feature_manifest(path: Path) -> Tuple[FeatureSpec, ...]:
    """Load and validate a CSV manifest, including stable ID/version uniqueness."""
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        specs = tuple(FeatureSpec.from_mapping(row) for row in csv.DictReader(handle))
    if not specs:
        raise ContractError("feature manifest is empty")
    identities = [(spec.feature_id, spec.version) for spec in specs]
    if len(identities) != len(set(identities)):
        raise ContractError("feature_id/version pairs must be unique")
    return specs


def feature_identities(specs: Iterable[FeatureSpec]) -> Tuple[str, ...]:
    return tuple(f"{spec.feature_id}@{spec.version}" for spec in specs)
