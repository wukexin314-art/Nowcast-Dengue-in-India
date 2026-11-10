"""Integrity checks that run before performance claims."""

from __future__ import annotations

from datetime import date
from typing import Iterable, Mapping, Set, Tuple


class AuditError(ValueError):
    """Raised when evidence is invalid or not directly comparable."""


CaseKey = Tuple[str, str, int]


def _parse_date(value: object, field: str) -> date:
    try:
        return date.fromisoformat(str(value))
    except ValueError as exc:
        raise AuditError(f"{field} must be an ISO date: {value}") from exc


def case_key(row: Mapping[str, object]) -> CaseKey:
    try:
        horizon = int(row["horizon"])
        origin = str(row["origin"])
        target_date = str(row["target_date"])
    except (KeyError, TypeError, ValueError) as exc:
        raise AuditError("rows require origin, target_date, and integer horizon") from exc
    _parse_date(origin, "origin")
    _parse_date(target_date, "target_date")
    if horizon <= 0:
        raise AuditError("horizon must be positive")
    return origin, target_date, horizon


def case_keys(rows: Iterable[Mapping[str, object]]) -> Set[CaseKey]:
    keys = [case_key(row) for row in rows]
    if len(keys) != len(set(keys)):
        raise AuditError("duplicate origin/target_date/horizon keys")
    if not keys:
        raise AuditError("no prediction rows")
    return set(keys)


def assert_common_grid(
    baseline_rows: Iterable[Mapping[str, object]],
    candidate_rows: Iterable[Mapping[str, object]],
) -> Set[CaseKey]:
    """Require exact case equality before paired comparison."""
    baseline = case_keys(baseline_rows)
    candidate = case_keys(candidate_rows)
    if baseline != candidate:
        missing = sorted(baseline - candidate)[:3]
        extra = sorted(candidate - baseline)[:3]
        raise AuditError(
            "evaluation grids differ: "
            f"baseline={len(baseline)}, candidate={len(candidate)}, "
            f"missing_from_candidate={missing}, extra_in_candidate={extra}"
        )
    return baseline


def assert_no_future_availability(rows: Iterable[Mapping[str, object]]) -> None:
    """Require every feature value to be available no later than its origin."""
    found = False
    for row in rows:
        found = True
        origin = _parse_date(row.get("origin"), "origin")
        available_at = _parse_date(row.get("available_at"), "available_at")
        if available_at > origin:
            feature = row.get("feature_id", "unknown")
            raise AuditError(
                f"future availability for {feature}: {available_at} > origin {origin}"
            )
    if not found:
        raise AuditError("no feature-availability rows")


def assert_no_future_vintage(rows: Iterable[Mapping[str, object]]) -> None:
    """Reject a historical as-of claim backed by a snapshot retrieved later."""
    found = False
    for row in rows:
        found = True
        origin = _parse_date(row.get("origin"), "origin")
        vintage_at = _parse_date(row.get("vintage_at"), "vintage_at")
        if vintage_at > origin:
            feature = row.get("feature_id", "unknown")
            raise AuditError(
                f"vintage mismatch for {feature}: snapshot {vintage_at} > origin {origin}"
            )
    if not found:
        raise AuditError("no vintage rows")
