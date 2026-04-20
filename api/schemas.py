"""
schemas.py — Pydantic response models for the Dengue Nowcast History API.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Shared
# ---------------------------------------------------------------------------

class ListResponse(BaseModel):
    count: int
    items: list[Any]


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str
    store_loaded: bool
    n_predictions: int
    n_splits: int
    split_month_min: str | None
    split_month_max: str | None
    built_at: str | None


# ---------------------------------------------------------------------------
# /splits
# ---------------------------------------------------------------------------

class SplitsResponse(BaseModel):
    count: int
    items: list[str]


# ---------------------------------------------------------------------------
# /predictions
# ---------------------------------------------------------------------------

class PredictionRecord(BaseModel):
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


class PredictionsResponse(BaseModel):
    count: int
    items: list[PredictionRecord]


# ---------------------------------------------------------------------------
# /evaluation
# ---------------------------------------------------------------------------

class EvaluationRecord(BaseModel):
    model_name: str
    model_version: str
    split_month: str
    test_end_month: str | None
    test_horizon_months: int | None
    n_train_months: int | None
    n_test_months: int | None
    n_train_who: int | None
    n_test_who: int | None
    RMSE_train: float | None
    MSE_train: float | None
    MAPE_train_pct: float | None
    RMSE_test: float | None
    MSE_test: float | None
    MAPE_test_pct: float | None
    RMSE_test_seasonal_naive: float | None
    MSE_test_seasonal_naive: float | None
    MAPE_test_seasonal_naive_pct: float | None
    epochs_used: int | None
    created_at: str | None


class EvaluationResponse(BaseModel):
    count: int
    items: list[EvaluationRecord]
