# Dengue Nowcast History API

Read-only FastAPI service for querying historical India dengue nowcast
predictions from the rolling-split pipeline.

**Model identifier (fixed):**
- `model_name` = `nowcast`
- `model_version` = `log_arx_v1`

---

## Quick Start

### 1. Install dependencies

```bash
pip install fastapi "uvicorn[standard]" pyarrow pydantic
# or from project root:
pip install -r requirements.txt
```

### 2. Build the prediction store

Run **once** (or after updating the rolling-split CSVs):

```bash
# from project root
python api/build_prediction_store.py
```

This reads three CSV files and writes three store files:

```
api/data/historical_predictions.parquet   (26 rows, 13 splits × 2 horizons)
api/data/evaluation.parquet               (13 rows, one per split)
api/data/metadata.json
```

### 3. Start the API

```bash
# from project root
uvicorn api.main:app --reload --port 8001
```

Interactive docs: <http://localhost:8001/docs>

---

## Endpoints

### `GET /health`

Returns service status and store statistics.

```bash
curl http://localhost:8001/health
```

### `GET /splits`

Returns all available `split_month` values (YYYY-MM), sorted chronologically.

```bash
curl http://localhost:8001/splits
```

### `GET /predictions`

Query historical prediction records. All parameters are optional.

| Parameter | Type | Description |
|---|---|---|
| `split_month` | string | Filter by split month (e.g. `2025-06`) |
| `target_month` | string | Filter by target/predicted month |
| `horizon` | int | Filter by forecast horizon (1 = one month ahead, 2 = two months ahead) |
| `limit` | int | Max rows to return (default 200, max 1000) |
| `offset` | int | Row offset for pagination (default 0) |

```bash
# All predictions for split month 2025-06
curl "http://localhost:8001/predictions?split_month=2025-06"

# All horizon-1 predictions
curl "http://localhost:8001/predictions?horizon=1"

# Predictions for a specific target month
curl "http://localhost:8001/predictions?target_month=2026-03"

# Paginated
curl "http://localhost:8001/predictions?limit=10&offset=0"
```

Response shape:
```json
{
  "count": 26,
  "items": [
    {
      "model_name": "nowcast",
      "model_version": "log_arx_v1",
      "split_month": "2025-06",
      "target_month": "2025-07",
      "horizon": 1,
      "point_pred": 14678.74,
      "ci_low": 6730.94,
      "ci_high": 32009.81,
      "y_true": 10302.0,
      "is_observed": true,
      ...
    }
  ]
}
```

### `GET /predictions/by-target`

Returns the **prediction history** for one target month across all split months
that produced a forecast for it. Useful for seeing how the nowcast evolved as
more data became available.

| Parameter | Type | Description |
|---|---|---|
| `target_month` | string (required) | Target month to query (YYYY-MM) |

```bash
# All splits that predicted 2026-03
curl "http://localhost:8001/predictions/by-target?target_month=2026-03"
```

### `GET /evaluation`

Returns rolling-split evaluation metrics.

| Parameter | Type | Description |
|---|---|---|
| `split_month` | string | Filter to one specific split |
| `horizon` | int | Filter by `test_horizon_months` (e.g. `2` for 2-month test window) |

```bash
# All evaluation rows
curl "http://localhost:8001/evaluation"

# One split
curl "http://localhost:8001/evaluation?split_month=2025-09"

# Only 2-month test window splits
curl "http://localhost:8001/evaluation?horizon=2"
```

---

## Field Definitions

### `historical_predictions.parquet`

| Field | Type | Description |
|---|---|---|
| `model_name` | str | Fixed: `nowcast` |
| `model_version` | str | Fixed: `log_arx_v1` |
| `split_month` | str (YYYY-MM) | The rolling split identifier month |
| `target_month` | str (YYYY-MM) | The month being predicted |
| `horizon` | int | Months between `split_month` and `target_month` (1 = one month ahead) |
| `point_pred` | float | Model point prediction (WHO cases, original scale) |
| `ci_low` | float | Lower bound of 95% prediction interval |
| `ci_high` | float | Upper bound of 95% prediction interval |
| `y_true` | float \| null | Actual WHO monthly cases (null if not yet observed) |
| `is_observed` | bool | `true` if `y_true` is non-null |
| `benchmark_t12` | float | Seasonal naive benchmark (same month, prior year) |
| `model_squared_error` | float \| null | `(point_pred - y_true)²` |
| `benchmark_squared_error` | float \| null | `(benchmark_t12 - y_true)²` |
| `feature_set` | str | Placeholder (empty string in V1) |
| `data_version` | str | Placeholder (empty string in V1) |
| `source_file` | str | Source CSV filename |
| `created_at` | str (ISO) | Timestamp when the store was built |

### `evaluation.parquet`

| Field | API name | Source CSV column |
|---|---|---|
| `split_month` | same | `split_month` |
| `test_end_month` | same | `test_end_month` |
| `test_horizon_months` | same | `test_horizon_months` |
| `RMSE_train` | same | `RMSE_train` |
| `MAPE_train_pct` | **renamed** | `MAPE_train_%` |
| `RMSE_test` | same | `RMSE_test` |
| `MAPE_test_pct` | **renamed** | `MAPE_test_%` |
| `RMSE_test_seasonal_naive` | same | `RMSE_test_seasonal_naive` |
| `MAPE_test_seasonal_naive_pct` | **renamed** | `MAPE_test_seasonal_naive_%` |

`%` characters are removed from column names for API / JSON compatibility.
The `horizon` query parameter in `GET /evaluation` maps to `test_horizon_months`.

---

## Source File → Field Mapping

### `rolling_predictions_long.csv` → `historical_predictions`

| Source column | Target field | Notes |
|---|---|---|
| `split_month` | `split_month` | direct |
| `target_month` | `target_month` | direct |
| `horizon_step` | `horizon` | renamed |
| `x_pred` | `point_pred` | renamed |
| `y_who` | `y_true` | renamed; null for future months |
| `benchmark_t12` | `benchmark_t12` | direct |
| `model_squared_error` | `model_squared_error` | direct |
| `benchmark_squared_error` | `benchmark_squared_error` | direct |

### `rolling_prediction_intervals_2month_test_window_long.csv` → `ci_low` / `ci_high`

Filter applied before merge:
- `is_test_window == 1` AND `pred_lo_95_test_window_only` is non-null

| Source column | Target field |
|---|---|
| `split_month` | `split_month` (merge key) |
| `date` | `target_month` (merge key) |
| `pred_lo_95_test_window_only` | `ci_low` |
| `pred_hi_95_test_window_only` | `ci_high` |

Merge type: left join on `(split_month, target_month)`.
In the current data all 26 rows have CI values (test_window filter matches all rolling-split predictions).

### `rolling_split_metrics.csv` → `evaluation`

Column rename only (% suffix → _pct suffix). All other columns kept as-is.

---

## Known Observations / Ambiguities

1. **`split_month 2026-02` RMSE_test is NaN** — this is the latest/operational split;
   no held-out test WHO values exist yet, so test metrics are undefined. This is correct.

2. **`horizon` vs `test_horizon_months`** — `horizon` in the predictions table is
   the per-row forecast step (1 or 2). `test_horizon_months` in the evaluation table
   is how many months were in the test window for that split (always 2 in current data).
   The `GET /evaluation?horizon=` parameter filters on `test_horizon_months`.

3. **`feature_set` and `data_version` are empty strings** — these cannot be reliably
   extracted from the current output files. They are reserved for future use when the
   notebook stores feature-set metadata in the summary JSON.

4. **`horizon_specific_predictions_summary.csv`** — not used in V1 because it duplicates
   `rolling_predictions_long.csv` content (same columns, same rows after inspection).
   Available as a cross-check if needed.

5. **CI coverage** — all 26 prediction rows currently have CI bounds. This is because
   the CI file contains test-window entries for all rolling-split predictions. If future
   runs include out-of-window months, `ci_low`/`ci_high` will be `null` for those rows.

---

## Directory Structure

```
api/
  main.py                    — FastAPI app, route handlers
  schemas.py                 — Pydantic response models
  repository.py              — Parquet loading + filter logic
  build_prediction_store.py  — Build/rebuild the Parquet stores
  data/
    historical_predictions.parquet
    evaluation.parquet
    metadata.json
  README_api.md              — This file
```
