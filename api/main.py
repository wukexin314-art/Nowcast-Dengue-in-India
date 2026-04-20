"""
main.py — FastAPI application for the Dengue Nowcast History API.

Start from project root:
    uvicorn api.main:app --reload --port 8001

Interactive docs:
    http://localhost:8001/docs
"""
from __future__ import annotations

from typing import Annotated

from fastapi import FastAPI, HTTPException, Query

from . import repository
from . schemas import (
    EvaluationResponse,
    HealthResponse,
    PredictionsResponse,
    SplitsResponse,
)

app = FastAPI(
    title="Dengue Nowcast History API",
    description=(
        "Read-only API for querying historical India dengue nowcast predictions "
        "produced by the log-ARX rolling-split pipeline. "
        "Model: **nowcast / log_arx_v1**."
    ),
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["Status"])
def health() -> HealthResponse:
    """Return service status and basic store statistics."""
    try:
        meta = repository.get_metadata()
        return HealthResponse(
            status="ok",
            store_loaded=True,
            n_predictions=meta.get("n_predictions", 0),
            n_splits=meta.get("n_splits", 0),
            split_month_min=meta.get("split_month_min"),
            split_month_max=meta.get("split_month_max"),
            built_at=meta.get("built_at"),
        )
    except FileNotFoundError as e:
        return HealthResponse(
            status="error — store not built",
            store_loaded=False,
            n_predictions=0,
            n_splits=0,
            split_month_min=None,
            split_month_max=None,
            built_at=None,
        )


# ---------------------------------------------------------------------------
# GET /splits
# ---------------------------------------------------------------------------

@app.get("/splits", response_model=SplitsResponse, tags=["Splits"])
def list_splits() -> SplitsResponse:
    """Return all available split_month values in chronological order."""
    try:
        splits = repository.get_splits()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    return SplitsResponse(count=len(splits), items=splits)


# ---------------------------------------------------------------------------
# GET /predictions
# ---------------------------------------------------------------------------

@app.get("/predictions", response_model=PredictionsResponse, tags=["Predictions"])
def list_predictions(
    split_month:  Annotated[str | None, Query(description="Filter by split month (YYYY-MM)")] = None,
    target_month: Annotated[str | None, Query(description="Filter by target month (YYYY-MM)")] = None,
    horizon:      Annotated[int | None, Query(description="Filter by forecast horizon (1, 2, …)")] = None,
    limit:        Annotated[int,        Query(ge=1, le=1000, description="Max rows to return")] = 200,
    offset:       Annotated[int,        Query(ge=0,          description="Row offset for pagination")] = 0,
) -> PredictionsResponse:
    """
    Return historical prediction records, with optional filtering by
    split_month, target_month, and/or horizon.

    - Omitting a filter means "no restriction on that field".
    - Results are sorted by split_month → target_month → horizon.
    - Use limit/offset for pagination.
    """
    try:
        total, items = repository.get_predictions(
            split_month=split_month,
            target_month=target_month,
            horizon=horizon,
            limit=limit,
            offset=offset,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    return PredictionsResponse(count=total, items=items)


# ---------------------------------------------------------------------------
# GET /predictions/by-target
# ---------------------------------------------------------------------------

@app.get("/predictions/by-target", response_model=PredictionsResponse, tags=["Predictions"])
def predictions_by_target(
    target_month: Annotated[str, Query(description="Target month to query (YYYY-MM)")],
) -> PredictionsResponse:
    """
    Return all historical predictions for a specific target month,
    across every split month in which that target was predicted.

    Useful for seeing how the nowcast evolved as more data arrived.
    Results are sorted by split_month → horizon.
    """
    try:
        total, items = repository.get_predictions_by_target(target_month=target_month)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    return PredictionsResponse(count=total, items=items)


# ---------------------------------------------------------------------------
# GET /evaluation
# ---------------------------------------------------------------------------

@app.get("/evaluation", response_model=EvaluationResponse, tags=["Evaluation"])
def list_evaluation(
    split_month: Annotated[str | None, Query(description="Filter by split month (YYYY-MM)")] = None,
    horizon:     Annotated[int | None, Query(description="Filter by test horizon months (e.g. 2)")] = None,
) -> EvaluationResponse:
    """
    Return rolling-split evaluation metrics (RMSE, MAPE, etc.).

    - `horizon` maps to the `test_horizon_months` column in the evaluation table
      (i.e. how many months ahead were evaluated in that split's test window).
    - Column name changes from source CSV:
        MAPE_train_%  → MAPE_train_pct
        MAPE_test_%   → MAPE_test_pct
        MAPE_test_seasonal_naive_% → MAPE_test_seasonal_naive_pct
    """
    try:
        total, items = repository.get_evaluation(
            split_month=split_month,
            horizon=horizon,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    return EvaluationResponse(count=total, items=items)
