# Nowcast Dengue in India

An end-to-end pipeline for **nowcasting monthly dengue activity in India** using a
combination of official WHO surveillance data and real-time digital proxy signals
(Google Trends, Wikipedia pageviews).

---

## Quick Access

| Resource | Link |
|---|---|
| **Live dashboard** | [nowcast-dengue-in-india.streamlit.app](https://nowcast-dengue-in-india.streamlit.app/#dengue-nowcast-in-india) |
| **API** | [dengue-nowcast-api.onrender.com](https://dengue-nowcast-api.onrender.com) |
| **API interactive docs** | [dengue-nowcast-api.onrender.com/docs](https://dengue-nowcast-api.onrender.com/docs) |
| **Python client** | `pip install "git+https://github.com/wukexin314-art/Nowcast-Dengue-in-India.git@main#subdirectory=client"` |

---

## Overview

Official dengue case data in India suffers from significant reporting delays.
Monthly WHO surveillance figures for a given month may not be available or finalized
until several weeks later, limiting timely situational awareness.

This project addresses that lag by training a short-term nowcasting model that
combines:
- **WHO monthly case counts** (available from January 2024, used as the primary
  supervision signal)
- **Google Trends** (relative search interest for *dengue fever* and *dengue vaccine*
  in India, available with near-zero delay)
- **Wikipedia pageviews** (monthly views of dengue and mosquito articles across
  Indian-language editions, also updated in near real-time)
- **OpenDengue national yearly totals** (used as an annual calibration constraint
  in signal construction)

The result is a two-step framework that first constructs a reliable monthly
dengue activity signal back to 2021, then trains an ARX-type nowcasting model
that can project estimates forward into months for which WHO data has not yet
been reported.

Rolling time-series cross-validation across 13 splits (2025-02 through 2026-02)
shows that the model broadly tracks the major seasonal cycle of dengue in India.
Short-term prediction errors are reasonable for a research-stage system of this
kind; systematic uncertainty is captured through 95% prediction intervals derived
from Hessian-based confidence estimates.

---

## Current Outputs

- **Live dashboard** — visualises WHO observed activity alongside the current
  2-month nowcast with confidence intervals; refreshes automatically as new data
  arrives.
- **Historical API** — read-only FastAPI service exposing all rolling-split
  predictions, confidence intervals, and evaluation metrics.
- **Python client** — lightweight wrapper (`dengue_nowcast_client`) that lets
  researchers and collaborators query the API from Python in one line.
- **Automated data refresh** — GitHub Actions pipeline updates the underlying
  datasets on the 5th and 20th of each month.

---

## Modeling Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│  1. DATA COLLECTION (dengue_updater/)                        │
│     WHO, Google Trends, Wikipedia, OpenDengue               │
│     → dengue_updater/data/processed/monthly_data.csv        │
│     → dengue_updater/data/processed/yearly_data.csv         │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│  2. STEP 1 — Signal calibration  (preprocessing.py)         │
│     Joint-loss PyTorch model: WHO monthly +                  │
│     OpenDengue yearly constraints + L2 regularisation        │
│     → preprocessing_outputs/preprocessed_monthly.csv        │
│       (smooth monthly dengue activity signal from 2021-01)   │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│  3. STEP 2 — ARX nowcasting  (updated_nowcast_log.ipynb)    │
│     Autoregressive-with-exogenous model:                     │
│     log1p target, Google Trends + Wikipedia + mosquito +     │
│     lags (1,2,12) + Fourier seasonality                      │
│     Rolling time-series CV (13 splits, 2-month horizon)      │
│     Recursive multi-step nowcast + Hessian-based CI          │
│     → outputs_updated_nowcast_log/rolling_splits/            │
│     → outputs_updated_nowcast_log/latest_nowcast_summary.json│
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│  4. SERVING                                                  │
│     dashboard/app.py   — Streamlit visualisation             │
│     api/main.py        — FastAPI historical query service    │
│     client/            — Python wrapper package              │
└─────────────────────────────────────────────────────────────┘
```

### Step 1 — Monthly Signal Calibration (`preprocessing.py`)

**Problem**: WHO India data only starts in January 2024; the model needs a monthly
activity signal back to 2021 to build enough lag history for Step 2.

**Approach**: A PyTorch linear model jointly optimised against three loss terms:

| Term | Weight | Purpose |
|---|---|---|
| WHO monthly MSE | λ = 5.0 | Primary: closely track observed monthly case counts |
| OpenDengue yearly constraint | λ = 0.01 | Monthly predictions must sum to national annual totals |
| L2 regularisation | λ = 1e-3 | Prevent overfitting; smooth the signal |

Features: Google Trends (z-scored), Wikipedia dengue pageviews (log1p z-scored),
11 month dummies.

**Output**: `preprocessing_outputs/preprocessed_monthly.csv` — a smooth monthly
dengue activity signal from 2021-01 that respects both monthly WHO observations
and yearly aggregate totals.

---

### Step 2 — ARX Nowcasting (`updated_nowcast_log.ipynb`)

**Model**: Autoregressive with exogenous inputs (ARX), log1p-transformed target,
trained via joint PyTorch loss (WHO monthly + yearly constraint + L2 + lag
regularisation).

**Features used**:

| Feature | Transformation | Source |
|---|---|---|
| Google Trends *dengue fever* | ÷ 100 | `monthly_data.csv` |
| Google Trends *dengue vaccine* | ÷ 100 | `monthly_data.csv` |
| Wikipedia dengue pageviews | log1p + normalised | `monthly_data.csv` |
| Wikipedia mosquito pageviews | log1p | `monthly_data.csv` |
| Lagged activity (lag 1, 2, 12) | standardised | Step 1 signal / WHO observed |
| Fourier seasonality (K=2) | sin/cos, period=12 | Constructed |

**Training boundary**: months ≤ `who_observed_through` (last real WHO report).

**Nowcast**: months from `who_observed_through + 1` through the latest month any
external proxy signal is available.  For nowcast horizons > 1 step, predictions
are generated **recursively** — each predicted value is fed back as a lag feature
for the next month.

**Uncertainty**: 95% prediction intervals computed via the delta method in
log1p-space using the Hessian of the training loss (Hessian-based Var(β)).

---

### Rolling Evaluation

To assess model performance without data leakage, a **rolling time-series
cross-validation** is run over 13 splits (2025-02 through 2026-02), each with a
2-month test window:

```
split 2025-02 → train: ≤ 2025-02 │ test: 2025-03, 2025-04
split 2025-03 → train: ≤ 2025-03 │ test: 2025-04, 2025-05
...
split 2026-02 → train: ≤ 2026-02 │ test: future (no actuals yet)
```

Metrics reported: RMSE, MAPE, and comparison against a seasonal-naive benchmark
(same month in the prior year).

---

## Models and Notebooks

### Currently deployed / primary

| File | Role |
|---|---|
| `preprocessing.py` | Step 1 signal calibration (PyTorch, joint loss). **Run this first.** |
| `updated_nowcast_log.ipynb` | **Primary Step 2 notebook**: log1p ARX, rolling CV, operational nowcast, all serving outputs. This is what the dashboard and API read. |

### Supporting / experimental

| File | Role |
|---|---|
| `step2_nowcast_mosquito_log.ipynb` | Variant of the Step 2 ARX notebook with the same log1p setup; used for comparison and earlier experiments. |
| `step2_nowcast_mosquito.ipynb` | Earlier Step 2 variant operating on the original case scale (no log transform). |
| `updated_nowcast.ipynb` | Step 2 notebook without log1p; predecessor of `updated_nowcast_log.ipynb`. |
| `lambda_tuning.ipynb` | Hyperparameter tuning only (single fixed split, 3-parameter grid for λ_year / λ_reg / λ_lag_reg). Does not produce serving outputs. |
| `load_processed_data.py` | Adapter layer translating `dengue_updater/data/processed/` into the data-loading interface expected by all notebooks. |

---

## Key Files and Directory Guide

### Data pipeline

| Path | What it is | When to look |
|---|---|---|
| `dengue_updater/` | Automated fetcher for all 5 data sources (WHO, Google Trends × 2, OpenDengue, Wikipedia × 2). Runs on GitHub Actions. | If data is stale or a source is failing. |
| `dengue_updater/src/main.py` | Entry point for manual data refresh. | `python dengue_updater/src/main.py` |
| `dengue_updater/data/processed/monthly_data.csv` | Wide-format monthly table (6 columns, from 2021-01). This is the primary input to all modeling steps. | Whenever you need to inspect current input data. |
| `dengue_updater/data/processed/yearly_data.csv` | Annual OpenDengue national totals; used as yearly constraint in Step 1. | Step 1 calibration / debugging. |
| `dengue_updater/logs/update_log.csv` | Append-only log of every fetch run. | Monitoring / troubleshooting data freshness. |

### Modeling

| Path | What it is | When to look |
|---|---|---|
| `preprocessing.py` | Step 1 signal calibration script. Run before any Step 2 notebook. | Changing feature set, loss weights, or retraining. |
| `preprocessing_outputs/` | Step 1 outputs: `preprocessed_monthly.csv`, training plots, `params_step1.json`. | Inspecting the calibrated signal or Step 1 performance. |
| `outputs_step1_wiki/` | Earlier Step 1 outputs (named before the current `preprocessing_outputs/` convention). | Historical reference. |
| `updated_nowcast_log.ipynb` | Primary Step 2 notebook; produces all serving outputs. | Step 2 retraining, debugging, or updating the nowcast. |
| `outputs_updated_nowcast_log/` | All Step 2 outputs: rolling-split CSVs, nowcast summary JSON, evaluation metrics, plots. | Inspecting predictions or feeding the dashboard/API. |
| `outputs_lambda_tuning/` | Lambda tuning results: `lambda_tuning_results.csv`. | If retuning hyperparameters. |

### Serving

| Path | What it is | When to look |
|---|---|---|
| `dashboard/app.py` | Streamlit dashboard entry point. | Local dashboard development or debugging. |
| `dashboard/utils.py` | Data loading logic for the dashboard (priority: `latest_nowcast_summary.json` → live CSVs → stale JSON cache). | Debugging data display issues. |
| `api/main.py` | FastAPI application; serves historical rolling-split predictions. | API development or deployment. |
| `api/repository.py` | Parquet loading and filter logic for the API. | Adding queries or changing filter behaviour. |
| `api/build_prediction_store.py` | Builds `api/data/*.parquet` from rolling-split CSVs. Run after updating the notebook outputs. | Refreshing the API data store. |
| `client/` | Self-contained Python client package (`dengue_nowcast_client`). | Python integration / distribution. |

### Documentation

| Path | What it is |
|---|---|
| `PROJECT_HANDOFF.md` | Comprehensive design-decision log; covers data sources, model details, known issues, changelog. |
| `dengue_updater/HANDOFF.md` | Detailed notes on the data updater: per-source rules, failure modes, schema definitions. |
| `api/README_api.md` | API field definitions, endpoint reference, source-to-field mapping table. |
| `client/README.md` | Client package docs: installation, all method signatures, error handling, PyPI publishing steps. |

---

## Dashboard

**URL**: [nowcast-dengue-in-india.streamlit.app](https://nowcast-dengue-in-india.streamlit.app/#dengue-nowcast-in-india)

The dashboard provides:
- **WHO observed activity** — actual monthly reported cases since 2024-01
- **Current nowcast** — model predictions for months after the latest WHO report, with 95% confidence intervals
- **Nowcast window** — automatically computed as the range between the last WHO-reported month and the latest month for which external proxy signals are available
- **Data status panel** — availability flags for each data source
- **Rolling backtest chart** — 1-step-ahead predictions across historical splits, giving a visual sense of model tracking ability

---

## API

**Base URL**: `https://dengue-nowcast-api.onrender.com`  
**Interactive docs**: `https://dengue-nowcast-api.onrender.com/docs`

> Note: the root path (`/`) returns a 404. Use `/health` or `/docs` as your
> starting point.

### Endpoints

| Endpoint | Description |
|---|---|
| `GET /health` | Service status and prediction-store statistics |
| `GET /splits` | All available `split_month` values (YYYY-MM), sorted chronologically |
| `GET /predictions` | Historical predictions; filter by `split_month`, `target_month`, `horizon`, with pagination |
| `GET /predictions/by-target` | All splits' forecasts for one target month — shows how the nowcast evolved |
| `GET /evaluation` | Rolling-split metrics (RMSE, MAPE, seasonal naive benchmark) |

### Quick examples

```bash
# Service health
curl https://dengue-nowcast-api.onrender.com/health

# All available splits
curl https://dengue-nowcast-api.onrender.com/splits

# Predictions for split month 2025-09
curl "https://dengue-nowcast-api.onrender.com/predictions?split_month=2025-09"

# How did forecasts for 2026-03 evolve across splits?
curl "https://dengue-nowcast-api.onrender.com/predictions/by-target?target_month=2026-03"

# Evaluation metrics for all splits
curl https://dengue-nowcast-api.onrender.com/evaluation
```

---

## Python Client

For researchers or collaborators who prefer working in Python directly.

### Install

```bash
# From GitHub (no PyPI release required)
pip install "git+https://github.com/wukexin314-art/Nowcast-Dengue-in-India.git@main#subdirectory=client"

# Or from a local clone of this repo
pip install -e ./client
```

### Usage

```python
from dengue_nowcast_client import DengueNowcastClient

client = DengueNowcastClient()  # connects to production API by default

# Service status
print(client.health())

# All split months
print(client.splits())

# Predictions for the latest split, first 5 rows
preds = client.predictions(limit=5)
for row in preds["items"]:
    print(row["split_month"], row["target_month"], row["horizon"], row["point_pred"])

# Forecast history for one target month
history = client.predictions_by_target("2026-03")
for row in history["items"]:
    print(row["split_month"], row["horizon"], row["point_pred"])

# Evaluation metrics
ev = client.evaluation(horizon=2)
for row in ev["items"]:
    print(row["split_month"], row["RMSE_test"], row["MAPE_test_pct"])
```

Point at a local API instance:

```python
client = DengueNowcastClient(base_url="http://localhost:8001")
```

For full documentation — all method signatures, error handling, and PyPI
publishing steps — see [`client/README.md`](client/README.md).

---

## How to Navigate the Repo

| Goal | Where to start |
|---|---|
| See the live nowcast | Dashboard link above |
| Query historical predictions programmatically | API → `/docs`, or Python client |
| Understand the modeling approach | `preprocessing.py` + `updated_nowcast_log.ipynb` |
| Inspect or update the input data | `dengue_updater/` |
| Rebuild the API data store | `python api/build_prediction_store.py` |
| Understand design decisions | `PROJECT_HANDOFF.md` |
| Develop the client package | `client/src/dengue_nowcast_client/` |

---

## Local Development

### Requirements

```bash
pip install -r requirements.txt
```

### Dashboard (local)

```bash
streamlit run dashboard/app.py
# → http://localhost:8501
```

### API (local)

```bash
# 1. Build the Parquet store from rolling-split outputs
python api/build_prediction_store.py

# 2. Start the server
uvicorn api.main:app --reload --port 8001
# → http://localhost:8001/docs
```

### Data updater (local)

```bash
# Full refresh of all data sources
python dengue_updater/src/main.py

# Dry run (fetches but does not write)
python dengue_updater/src/main.py --dry-run
```

### Running Step 1 + Step 2

```bash
# Step 1: rebuild the monthly calibration signal
python preprocessing.py

# Step 2: open and run updated_nowcast_log.ipynb from a clean kernel
jupyter notebook updated_nowcast_log.ipynb
```

### Client package (local install + tests)

```bash
pip install -e ./client          # editable install
pytest client/tests/ -v          # 16 unit tests, no live network calls
```

### Automated data refresh

A GitHub Actions workflow (`.github/workflows/dengue_update.yml`) runs on the
**5th and 20th of each month at 03:00 UTC**.  It fetches all five data sources and
auto-commits changes with the message `auto: update dengue data YYYY-MM-DD`.
It can also be triggered manually from the Actions tab.

---

## Notes and Limitations

- **Research-stage system.** This is an active research and development project.
  Outputs should be interpreted as short-term situational awareness estimates,
  not as a replacement for official WHO surveillance data or clinical guidance.

- **WHO data starts January 2024.** India's monthly dengue case counts have only
  been available in the WHO Shiny system from January 2024.  The 2021–2023 period
  uses the Step 1 calibrated signal as a proxy for training purposes.

- **Google Trends re-normalisation.** `pytrends` re-fetches the full historical
  window each run and re-normalises the relative index; small shifts in historical
  values between runs are expected and inherent to how Google Trends works.

- **Wikipedia language coverage.** Dengue pageviews are summed across 7 Indian-
  language Wikipedia editions; mosquito pageviews across 5.  The language sets
  differ from legacy files used in earlier experiments; coefficients are not
  directly comparable across versions.

- **Uncertainty intervals.** The 95% prediction intervals are derived from a
  normal approximation using the Hessian of the training loss.  For multi-step
  nowcasts (horizon > 1), lags are substituted with previously predicted values,
  which means interval width underestimates true propagated uncertainty.

---

## Contact and Collaboration

This project is maintained by **Cathie Wu** (wukexin314@gmail.com).  
Feedback, questions, and collaboration enquiries are welcome.
