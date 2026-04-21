"""
types.py — TypedDict definitions for API response shapes.

These are documentation aids and work with IDE type checkers.
All client methods return plain dicts; cast to these if you want strict typing.
"""
from __future__ import annotations

from typing import Any

try:
    from typing import TypedDict
except ImportError:  # Python 3.7 compat (not required by this package)
    from typing_extensions import TypedDict  # type: ignore[assignment]


class HealthResponse(TypedDict):
    status: str
    store_loaded: bool
    n_predictions: int
    n_splits: int
    split_month_min: str | None
    split_month_max: str | None
    built_at: str | None


class SplitsResponse(TypedDict):
    count: int
    items: list[str]


class PredictionRecord(TypedDict):
    model_name: str
    model_version: str
    split_month: str
    target_month: str
    horizon: int
    point_pred: float | None
    ci_low: float | None
    ci_high: float | None
    y_true: float | None
    is_observed: bool
    benchmark_t12: float | None
    model_squared_error: float | None
    benchmark_squared_error: float | None
    feature_set: str
    data_version: str
    source_file: str
    created_at: str


class PredictionsResponse(TypedDict):
    count: int
    items: list[PredictionRecord]


class EvaluationRecord(TypedDict):
    model_name: str
    model_version: str
    split_month: str
    test_end_month: str | None
    test_horizon_months: int | None
    RMSE_train: float | None
    MSE_train: float | None
    MAPE_train_pct: float | None
    RMSE_test: float | None
    MSE_test: float | None
    MAPE_test_pct: float | None
    RMSE_test_seasonal_naive: float | None
    MAPE_test_seasonal_naive_pct: float | None
    epochs_used: int | None


class EvaluationResponse(TypedDict):
    count: int
    items: list[EvaluationRecord]


# Generic response envelope returned by all list endpoints
ListResponse = dict[str, Any]
