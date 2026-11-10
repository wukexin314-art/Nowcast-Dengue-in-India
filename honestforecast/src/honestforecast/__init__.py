"""Audit-friendly forecasting research primitives."""

from .audit import (
    AuditError,
    assert_common_grid,
    assert_no_future_availability,
    assert_no_future_vintage,
)
from .comparison import FamilyDecision, append_decision, compare_feature_family
from .contracts import ContractError, FeatureSpec, ForecastContract, load_feature_manifest
from .features import FeatureError, FeatureLineage, as_of_join, fit_fold_transform
from .metrics import MetricError, summarize_predictions, validate_prediction_rows
from .screening import ScreeningError, screen_candidates
from .vintages import GitSnapshot, SnapshotAudit, VintageError, audit_snapshots

__all__ = [
    "AuditError",
    "ContractError",
    "FeatureSpec",
    "FeatureError",
    "FeatureLineage",
    "FamilyDecision",
    "ForecastContract",
    "GitSnapshot",
    "MetricError",
    "ScreeningError",
    "SnapshotAudit",
    "VintageError",
    "append_decision",
    "as_of_join",
    "assert_common_grid",
    "assert_no_future_availability",
    "assert_no_future_vintage",
    "audit_snapshots",
    "compare_feature_family",
    "fit_fold_transform",
    "load_feature_manifest",
    "summarize_predictions",
    "validate_prediction_rows",
    "screen_candidates",
]

__version__ = "0.3.0"
