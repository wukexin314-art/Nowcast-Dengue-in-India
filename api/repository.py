"""
repository.py — Load Parquet stores and apply filters for API queries.

All public functions return plain Python lists/dicts; serialisation is left
to FastAPI / Pydantic.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR  = Path(__file__).resolve().parent / "data"
PRED_FILE = DATA_DIR / "historical_predictions.parquet"
EVAL_FILE = DATA_DIR / "evaluation.parquet"
META_FILE = DATA_DIR / "metadata.json"


# ---------------------------------------------------------------------------
# Lazy-loaded DataFrames (loaded once per process)
# ---------------------------------------------------------------------------
_pred_df: pd.DataFrame | None = None
_eval_df: pd.DataFrame | None = None
_metadata: dict[str, Any] | None = None


def _load() -> None:
    global _pred_df, _eval_df, _metadata
    if _pred_df is None:
        if not PRED_FILE.exists():
            raise FileNotFoundError(
                f"Prediction store not found: {PRED_FILE}\n"
                "Run: python api/build_prediction_store.py"
            )
        _pred_df = pd.read_parquet(PRED_FILE)

    if _eval_df is None:
        if not EVAL_FILE.exists():
            raise FileNotFoundError(
                f"Evaluation store not found: {EVAL_FILE}\n"
                "Run: python api/build_prediction_store.py"
            )
        _eval_df = pd.read_parquet(EVAL_FILE)

    if _metadata is None:
        if META_FILE.exists():
            _metadata = json.loads(META_FILE.read_text(encoding="utf-8"))
        else:
            _metadata = {}


def _safe_float(v: Any) -> float | None:
    """Convert to float; return None for NaN / inf / non-numeric."""
    if v is None:
        return None
    try:
        f = float(v)
        return None if (math.isnan(f) or math.isinf(f)) else f
    except (TypeError, ValueError):
        return None


def _safe_int(v: Any) -> int | None:
    if v is None:
        return None
    try:
        f = float(v)
        return None if (math.isnan(f) or math.isinf(f)) else int(f)
    except (TypeError, ValueError):
        return None


def _row_to_pred(row: pd.Series) -> dict[str, Any]:
    return {
        "model_name":              str(row.get("model_name", "")),
        "model_version":           str(row.get("model_version", "")),
        "split_month":             str(row["split_month"]),
        "target_month":            str(row["target_month"]),
        "horizon":                 int(row["horizon"]),
        "point_pred":              _safe_float(row.get("point_pred")),
        "ci_low":                  _safe_float(row.get("ci_low")),
        "ci_high":                 _safe_float(row.get("ci_high")),
        "y_true":                  _safe_float(row.get("y_true")),
        "is_observed":             bool(row.get("is_observed", False)),
        "benchmark_t12":           _safe_float(row.get("benchmark_t12")),
        "model_squared_error":     _safe_float(row.get("model_squared_error")),
        "benchmark_squared_error": _safe_float(row.get("benchmark_squared_error")),
        "feature_set":             str(row.get("feature_set", "")),
        "data_version":            str(row.get("data_version", "")),
        "source_file":             str(row.get("source_file", "")),
        "created_at":              str(row.get("created_at", "")),
    }


def _row_to_eval(row: pd.Series) -> dict[str, Any]:
    return {
        "model_name":                    str(row.get("model_name", "")),
        "model_version":                 str(row.get("model_version", "")),
        "split_month":                   str(row["split_month"]),
        "test_end_month":                str(row["test_end_month"]) if pd.notna(row.get("test_end_month")) else None,
        "test_horizon_months":           _safe_int(row.get("test_horizon_months")),
        "n_train_months":                _safe_int(row.get("n_train_months")),
        "n_test_months":                 _safe_int(row.get("n_test_months")),
        "n_train_who":                   _safe_int(row.get("n_train_who")),
        "n_test_who":                    _safe_int(row.get("n_test_who")),
        "RMSE_train":                    _safe_float(row.get("RMSE_train")),
        "MSE_train":                     _safe_float(row.get("MSE_train")),
        "MAPE_train_pct":                _safe_float(row.get("MAPE_train_pct")),
        "RMSE_test":                     _safe_float(row.get("RMSE_test")),
        "MSE_test":                      _safe_float(row.get("MSE_test")),
        "MAPE_test_pct":                 _safe_float(row.get("MAPE_test_pct")),
        "RMSE_test_seasonal_naive":      _safe_float(row.get("RMSE_test_seasonal_naive")),
        "MSE_test_seasonal_naive":       _safe_float(row.get("MSE_test_seasonal_naive")),
        "MAPE_test_seasonal_naive_pct":  _safe_float(row.get("MAPE_test_seasonal_naive_pct")),
        "epochs_used":                   _safe_int(row.get("epochs_used")),
        "created_at":                    str(row.get("created_at", "")) if pd.notna(row.get("created_at")) else None,
    }


# ---------------------------------------------------------------------------
# Public API functions
# ---------------------------------------------------------------------------

def get_metadata() -> dict[str, Any]:
    _load()
    return _metadata or {}


def get_splits() -> list[str]:
    _load()
    return sorted(_pred_df["split_month"].unique().tolist())


def get_predictions(
    split_month: str | None = None,
    target_month: str | None = None,
    horizon: int | None = None,
    limit: int = 200,
    offset: int = 0,
) -> tuple[int, list[dict]]:
    _load()
    df = _pred_df.copy()

    if split_month:
        df = df[df["split_month"] == split_month]
    if target_month:
        df = df[df["target_month"] == target_month]
    if horizon is not None:
        df = df[df["horizon"] == horizon]

    df = df.sort_values(["split_month", "target_month", "horizon"])
    total = len(df)
    page = df.iloc[offset : offset + limit]
    return total, [_row_to_pred(row) for _, row in page.iterrows()]


def get_predictions_by_target(target_month: str) -> tuple[int, list[dict]]:
    _load()
    df = _pred_df[_pred_df["target_month"] == target_month].sort_values(
        ["split_month", "horizon"]
    )
    return len(df), [_row_to_pred(row) for _, row in df.iterrows()]


def get_evaluation(
    split_month: str | None = None,
    horizon: int | None = None,
) -> tuple[int, list[dict]]:
    _load()
    df = _eval_df.copy()

    if split_month:
        df = df[df["split_month"] == split_month]
    # horizon maps to test_horizon_months in the evaluation table
    if horizon is not None and "test_horizon_months" in df.columns:
        df = df[df["test_horizon_months"] == horizon]

    df = df.sort_values("split_month")
    return len(df), [_row_to_eval(row) for _, row in df.iterrows()]
