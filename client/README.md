# dengue-nowcast-client

Python client library for the **India Dengue Nowcast History API** — a read-only
service that exposes historical rolling-split predictions, confidence intervals,
and evaluation metrics produced by the log-ARX nowcasting pipeline.

- **API base URL**: `https://dengue-nowcast-api.onrender.com`
- **Interactive docs**: `https://dengue-nowcast-api.onrender.com/docs`
- **Model**: `nowcast / log_arx_v1`

---

## Installation

```bash
pip install dengue-nowcast-client
```

Or install directly from this repository:

```bash
pip install ./client          # from the repo root
pip install -e ./client       # editable install for development
```

---

## Quick Start

```python
from dengue_nowcast_client import DengueNowcastClient

client = DengueNowcastClient()

print(client.health())
print(client.splits())
print(client.predictions(limit=5))
print(client.predictions_by_target("2026-01"))
print(client.evaluation(horizon=2))
```

---

## Methods

All methods return a plain Python `dict`.

### `health()`

```python
h = client.health()
# {'status': 'ok', 'store_loaded': True, 'n_predictions': 26,
#  'n_splits': 13, 'split_month_min': '2025-02', 'split_month_max': '2026-02',
#  'built_at': '2026-04-21T...'}
```

### `splits()`

Returns all available `split_month` values in chronological order.

```python
s = client.splits()
# {'count': 13, 'items': ['2025-02', '2025-03', ..., '2026-02']}
```

### `predictions(...)`

Query historical prediction records.  All parameters are optional.

```python
# All predictions for one split
p = client.predictions(split_month="2025-06")

# All horizon-1 predictions
p = client.predictions(horizon=1)

# Specific target month with pagination
p = client.predictions(target_month="2025-09", limit=10, offset=0)

# Combined filters
p = client.predictions(split_month="2025-06", horizon=1)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `split_month` | `str \| None` | `None` | Filter to one rolling split (YYYY-MM) |
| `target_month` | `str \| None` | `None` | Filter to one target month (YYYY-MM) |
| `horizon` | `int \| None` | `None` | Forecast step: `1` = 1 month ahead, `2` = 2 months ahead |
| `limit` | `int` | `200` | Max rows to return (1–1000) |
| `offset` | `int` | `0` | Row offset for pagination |

Response shape:
```python
{
  "count": 2,
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
      "is_observed": True,
      "benchmark_t12": 5597.0,
      ...
    }
  ]
}
```

### `predictions_by_target(target_month)`

Returns the full **prediction history** for one target month across all splits
that produced a forecast for it.  Useful for seeing how the nowcast evolved.

```python
bt = client.predictions_by_target("2026-03")
# shows predictions from splits 2026-01 (h=2) and 2026-02 (h=1)
for row in bt["items"]:
    print(row["split_month"], row["horizon"], row["point_pred"])
```

### `evaluation(...)`

Rolling-split evaluation metrics (RMSE, MAPE, benchmark comparison).

```python
# All splits
ev = client.evaluation()

# One split
ev = client.evaluation(split_month="2025-09")

# Filter by test-window size
ev = client.evaluation(horizon=2)
```

> **Column name note**: `MAPE_train_%`, `MAPE_test_%`, and
> `MAPE_test_seasonal_naive_%` from the source CSV are returned as
> `MAPE_train_pct`, `MAPE_test_pct`, and `MAPE_test_seasonal_naive_pct`
> (the `%` is removed for JSON/Python compatibility).

---

## Custom base URL (local development)

```python
# Point at a locally running instance
client = DengueNowcastClient(base_url="http://localhost:8001")

# Custom timeout (seconds)
client = DengueNowcastClient(timeout=60)
```

---

## Context manager

```python
with DengueNowcastClient() as client:
    data = client.splits()
# Session is automatically closed on exit
```

---

## Error handling

```python
from dengue_nowcast_client import (
    DengueNowcastClient,
    DengueNowcastError,        # base class
    DengueNowcastHTTPError,    # non-2xx HTTP response
    DengueNowcastConnectionError,  # network failure / timeout
    DengueNowcastResponseError,    # JSON parse failure
)

client = DengueNowcastClient()

try:
    result = client.predictions(split_month="9999-99")
except DengueNowcastHTTPError as e:
    print(f"HTTP {e.status_code}: {e.detail}")
except DengueNowcastConnectionError as e:
    print(f"Network error: {e}")
except DengueNowcastError as e:
    print(f"Other error: {e}")
```

| Exception | When raised |
|---|---|
| `DengueNowcastHTTPError` | API returned 4xx/5xx. Has `.status_code` and `.detail` attributes. |
| `DengueNowcastConnectionError` | Timeout, DNS failure, connection refused. |
| `DengueNowcastResponseError` | Response body is not valid JSON. |
| `DengueNowcastError` | Base class; catch this to handle any client error. |

---

## Minimal script example

```python
#!/usr/bin/env python3
"""Print the latest nowcast predictions."""
from dengue_nowcast_client import DengueNowcastClient

with DengueNowcastClient() as client:
    sp = client.splits()
    latest_split = sp["items"][-1]

    preds = client.predictions(split_month=latest_split)
    print(f"Latest split: {latest_split}")
    for row in preds["items"]:
        print(
            f"  {row['target_month']} (h={row['horizon']}): "
            f"{row['point_pred']:,.0f} cases  "
            f"[{row['ci_low']:,.0f} – {row['ci_high']:,.0f}]"
        )
```

---

## Publishing to PyPI

```bash
# 1. Install build tools
pip install build twine

# 2. Build distribution (run from the client/ directory)
cd client
python -m build
# Creates dist/dengue_nowcast_client-0.1.0-py3-none-any.whl
#         dist/dengue_nowcast_client-0.1.0.tar.gz

# 3. Upload to PyPI (requires PyPI account + token)
twine upload dist/*

# 4. After publishing, others can install with:
pip install dengue-nowcast-client
```

To upload to TestPyPI first: `twine upload --repository testpypi dist/*`

---

## Running tests

```bash
# From the repo root
pip install -e client/                  # install in editable mode
pip install pytest                      # install pytest
pytest client/tests/ -v                 # run tests
pytest client/tests/ -v --cov=dengue_nowcast_client  # with coverage
```

---

## Project layout

```
client/
├── pyproject.toml
├── README.md
├── requirements-dev.txt
├── .gitignore
├── src/
│   └── dengue_nowcast_client/
│       ├── __init__.py       # exports DengueNowcastClient + exceptions
│       ├── client.py         # DengueNowcastClient implementation
│       ├── exceptions.py     # custom exception hierarchy
│       └── types.py          # TypedDict shapes for IDE support
├── tests/
│   └── test_client.py
└── examples/
    └── basic_usage.py
```
