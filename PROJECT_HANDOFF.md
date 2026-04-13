# PROJECT HANDOFF — Nowcast Dengue in India

> **Purpose**: Core reference document for quickly restoring project context after conversation compression.  
> **Last updated**: 2026-04-12  
> **Detailed migration record**: see `MIGRATION_NOTES.md`

---

## 1. Project Overview

### Research Goal

Official dengue case data in India (WHO monthly reports) suffers from significant reporting delays. The goal of this project is:

> **Use real-time digital proxy signals (Google Trends, Wikipedia pageviews) to nowcast monthly dengue activity levels in India.**

### Two-Step Modeling Framework

#### Step 1: Monthly Signal Construction (`preprocessing.py`)

**Problem**: WHO monthly data only begins from 2024-01; the only long-running official reference is OpenDengue's annual totals.  
**Approach**: A PyTorch-optimized linear model jointly fit to three types of constraints:

1. **WHO monthly observations** (available from 2024; used when present)
2. **OpenDengue annual total constraints** (model's monthly predictions must sum to OpenDengue yearly totals)
3. **L2 regularization**

Features: `intercept + g_fever_z + g_vaccine_z + wiki_views_z (log1p then z-scored) + month dummies (m_2 … m_12)`  
Continuous features are z-score standardized over the full modeling window before entering the model. Raw values are kept in the output CSV for inspection.  
Output: a smooth monthly dengue activity signal from 2021-01, saved to `outputs_step1_wiki/preprocessed_monthly.csv`.

#### Step 2: Nowcasting Model (three notebooks)

Uses Step 1's signal as the training target (or to impute missing WHO values) to train an ARX model:

- Features: `Google Trends + Wikipedia + mosquito Wikipedia + lagged terms (lag 1, 2, 12) + Fourier seasonality terms`
- Validation: rolling time-series CV (rolling train/test split), evaluating 1-step and 2-step ahead RMSE/MAPE
- Three notebooks correspond to three model specification variants (see Section 4)

---

## 2. Data Sources

### WHO Monthly Dengue Data

| Item | Detail |
|---|---|
| Meaning | Official monthly reported dengue cases in India |
| Granularity | Monthly |
| Coverage | From 2024-01 (**no monthly data for 2021–2023** — a limitation of the WHO database itself) |
| Current column name | `who_cases_monthly` (monthly_data.csv) / `source="WHO"` (long format) |
| Fetch method | `dengue_updater/src/fetch_who.py`, three-layer fallback: Shiny WebSocket → direct download → HTML parsing |
| Known limitation | Source is the WHO Shiny dashboard, which has an unstable structure and may break if WHO redesigns the page; existing values are preserved on failure |

### Google Trends

| Item | Detail |
|---|---|
| Meaning | Monthly search volume index (0–100) for "dengue fever" / "dengue vaccine" in India (geo=IN) |
| Granularity | Monthly |
| Coverage | New data from 2021-01 (updater's `START_DATE`); old `master_data.csv` from 2010-11 |
| Current column names | `google_trends_dengue_fever`, `google_trends_dengue_vaccine` |
| Fetch method | `fetch_google_trends.py` using `pytrends`; full-window refetch every run (because pytrends re-normalizes the index each time) |
| Known limitation | Prone to 429 errors; exponential backoff implemented. Normalization means historical values may shift slightly across runs |
| Model treatment | Step 1: used as-is (0–100); Step 2: divided by 100 to normalize to [0, 1] |

### OpenDengue (Yearly)

| Item | Detail |
|---|---|
| Meaning | National-level annual dengue case totals for India, from the OpenDengue database |
| Granularity | Yearly |
| Coverage | 2021–2024 (current range in yearly_data.csv) |
| Current column name | `open_dengue_national_yearly` (yearly_data.csv) / `source="OpenDengue_National_Yearly"` (long format) |
| Fetch method | `fetch_open_dengue.py`, downloads V1.3 CSV from OpenDengue GitHub, filters India + Admin0 + T_res=Year |
| Role in model | **Step 1 annual constraint**: model's monthly predictions must sum to OpenDengue yearly totals |
| Known limitation | Old data had `OpenDengue_State_Aggregated` (state-level aggregate); new data only has national. Handled in the adapter layer |

### Wikipedia Dengue Pageviews

| Item | Detail |
|---|---|
| Meaning | **Sum** of monthly Wikipedia pageviews for dengue articles across multiple Indian-language editions |
| Language set | **7 languages**: Hindi (hi), Kannada (kn), Malayalam (ml), Marathi (mr), Tamil (ta), Telugu (te), Bengali (bn) |
| Granularity | Monthly |
| Coverage | From 2021-01 |
| Current column name | `wikipedia_total_dengue_views` |
| Fetch method | `fetch_wikipedia.py`, calls the Wikimedia REST API, validates article titles before fetching, sums successful languages and marks as `partial` if some fail |
| Model treatment | Applied `log1p` transform, used as exogenous regressor (`wiki_views`) |

> ⚠️ **Important definition note**:  
> `wikipedia_total_dengue_views` is a **multi-language sum across Indian-language Wikipedia editions (not English Wikipedia)**. It is not the pageview count from the en.wikipedia.org dengue article.  
> The old `total_dengue_views.csv` used a different language set or aggregation method, resulting in values approximately **7× smaller** than the new data. The two are not directly comparable.

### Wikipedia Mosquito Pageviews

| Item | Detail |
|---|---|
| Meaning | Sum of monthly Wikipedia pageviews for mosquito articles across multiple Indian-language editions |
| Language set | **5 languages**: Hindi (hi), Tamil (ta), Marathi (mr), Malayalam (ml), Kannada (kn) (excludes Bengali, Telugu) |
| Current column name | `wikipedia_mosquito_views_total` |
| Model treatment | Applied `log1p` transform, used as exogenous regressor (`mosq_total_monthly_views`); only used in mosquito-related notebooks |

> ⚠️ **Definition change**: Old `monthly_mosquito_aggregate.csv` used 6 languages (including Gujarati); new data uses 5 (Gujarati removed). Values differ by ~1.3×.

---

## 3. Data Pipeline / Updater

### Module Layout

```
dengue_updater/
├── src/
│   ├── main.py              # main entry point
│   ├── config.py            # paths, constants, utility functions
│   ├── fetch_who.py         # WHO data fetcher
│   ├── fetch_google_trends.py
│   ├── fetch_open_dengue.py
│   ├── fetch_wikipedia.py
│   ├── update_monthly.py    # merge logic (monthly)
│   ├── update_yearly.py     # merge logic (yearly)
│   ├── export_outputs.py    # generates master_data.xlsx
│   └── verify_titles.py     # validates Wikipedia article titles
├── config/
│   └── wiki_title_map.yaml  # Wikipedia article names per language
├── data/
│   ├── processed/           # ← modeling code reads directly from here
│   │   ├── monthly_data.csv
│   │   ├── yearly_data.csv
│   │   └── master_data.xlsx
│   ├── raw/                 # raw fetched files (gitignored)
│   └── interim/             # intermediate files (gitignored)
└── logs/
    └── update_log.csv       # append-only per-source run log
```

### How to Run

```bash
# Full update (all sources)
python dengue_updater/src/main.py

# Skip specific sources
python dengue_updater/src/main.py --skip-who --skip-google-trends

# Dry run (fetch but do not write files)
python dengue_updater/src/main.py --dry-run
```

### Automation

The GitHub Actions workflow (`.github/workflows/dengue_update.yml`) runs automatically on the **5th and 20th of each month at 03:00 UTC**. When data changes, it auto-commits with the message `auto: update dengue data YYYY-MM-DD`. Can also be triggered manually from the GitHub Actions UI.

### Output Files

| File | Description | Committed to git? |
|---|---|---|
| `data/processed/monthly_data.csv` | Primary monthly data, wide format, 6 columns, from 2021-01 | ✅ |
| `data/processed/yearly_data.csv` | Annual totals, 2021–2024 | ✅ |
| `data/processed/master_data.xlsx` | Excel version of both tables | ✅ |
| `logs/update_log.csv` | Append-only fetch status log per run | ✅ |
| `logs/run_summary_*.json` | Detailed summary for each full run | ❌ gitignored |
| `data/raw/` | Raw fetched files | ❌ gitignored |

### Merge Logic

- New data fetched successfully → overwrites the corresponding cells (including existing non-null values)
- Fetch fails → **existing values are preserved**, no overwrite
- Google Trends exception: full-window refetch and complete replacement each run (due to normalization behavior)

### Known Status

- WHO fetch uses 3-layer fallback; layers 1/2 (direct URLs) are currently `None`, so layer 3 (HTML parsing) is actually used — relatively fragile
- If some Wikipedia languages time out, the fetch returns `PARTIAL` status and continues; does not abort the full update

---

## 4. Modeling Pipeline

### Step 1: `preprocessing.py`

**Entry**: `python preprocessing.py`  
**Output directory**: `outputs_step1_wiki/`

| Input | Description |
|---|---|
| `who_cases_monthly` | Monthly target (sparse; only available from 2024) |
| `google_trends_dengue_fever` / `_vaccine` | Primary exogenous features |
| `wikipedia_total_dengue_views` (log1p) | Optional exogenous feature, enabled when `use_wiki=True` |
| `open_dengue_national_yearly` | Annual constraint term (via `build_yearly_proxy`) |

**Training**: PyTorch Adam with three-term loss:

```
L = λ_who  × MSE(WHO observed months)
  + λ_year × MSE(yearly sums vs OpenDengue)
  + λ_reg  × ||β||²
```

Current defaults: `λ_who=5.0, λ_year=0.3, λ_reg=1e-3, epochs=5000, lr=1e-2`  
(Changed from 1.0/1.0/1e-3 to prioritize WHO monthly fit over yearly constraint.)

**Learned feature coefficients (most recent run, with z-scored features)**:

```
intercept:   9.53    g_fever_z:  12.33    g_vaccine_z: -2.20
wiki_views_z: 0.54   m_9:        13.22    m_10:        19.17
m_11:        16.68   m_12:       10.53
```

(Peak season in months 9–12, consistent with India's dengue epidemiology)

### Step 2: Three Notebooks

| Notebook | Key features | Output directory |
|---|---|---|
| `step2_nowcast_mosquito.ipynb` | Includes mosquito Wikipedia feature; original case scale | `outputs_step2_mosquito/` |
| `step2_nowcast_mosquito_log.ipynb` | Same, but target is log1p-transformed; `target_scale=1.0` | `outputs_step2_mosquito_log/` |
| `step2_nowcast_loss_function_...lambda_tuning.ipynb` | **Lambda tuning only** — single fixed split, selects global `lambda_year`/`lambda_reg`/`lambda_lag_reg` | `outputs_lambda_tuning/` |

**Common Step 2 design**:

- Target: `WHO` (observed months); missing months imputed with Step 1 predictions as `x_tilde` (for lagged inputs)
- Lag terms: `lag 1, 2, 12` (includes same-month-last-year lag)
- Seasonality: Fourier terms (sin/cos, K=2), preferred over month dummies
- Feature standardization: computed from training-set statistics
- Early stopping: patience = 1000 epochs
- Evaluation window: `step2_nowcast_mosquito_log.ipynb` uses **auto-computed** rolling splits (range derived from last valid WHO month — see below); the lambda tuning notebook does **not** run rolling eval
- Annual constraints: same as Step 1, applied only to complete years within the training window

#### Rolling Split Auto-Range (`step2_nowcast_mosquito_log.ipynb`)

`rolling_start` / `rolling_end` are **no longer hardcoded** in the Config. They are computed at runtime by `compute_rolling_split_range()` (Cell 3):

| Parameter | Rule |
|---|---|
| `rolling_start` | Last valid WHO month **− 12 months** |
| `rolling_end` | Last valid WHO month **− 2 months** |

**Why −12 / −2**: the last 2 months before the latest WHO report are the most uncertain (recent revisions); starting 12 months back gives enough splits for stable CV. The function reads the actual wide DataFrame so it updates automatically whenever WHO data is refreshed.

```python
def compute_rolling_split_range(wide, target_source='WHO', back_start=12, back_end=2):
    # → returns (rolling_start: str, rolling_end: str, last_who_month: Timestamp,
    #             rolling_split_months: list[str])
```

Called in **Cell 6** immediately after building the wide DataFrame. `rolling_start`, `rolling_end`, `last_who_month`, and `rolling_split_months` are all module-level variables in scope for Cells 8 and 10. Cells 8 and 10 reference `rolling_end` (not a hardcoded string) for legend positioning.

As of **2026-04-05** (last valid WHO month = **2026-02**):
- `last_who_month` = **2026-02**
- `rolling_start` = **2025-02**
- `rolling_end` = **2025-12**
- Splits: `['2025-02', '2025-03', '2025-04', '2025-05', '2025-06', '2025-07', '2025-08', '2025-09', '2025-10', '2025-11', '2025-12']` (11 splits)

### Step 2 Lambda Tuning (`step2_nowcast_loss_function_...lambda_tuning.ipynb`)

**Purpose**: Select one global set of three lambdas before running rolling evaluation in other notebooks.

**Baseline alignment**: fully aligned with **`step2_nowcast_mosquito_log.ipynb`** — Cells 1, 3, 4 copied verbatim; `step2_nowcast_mosquito_log.ipynb` itself was **not modified**.

| Setting | Value / Detail |
|---|---|
| Tuning cutoff | `tune_split_month = "2025-01"` (all data ≤ 2025-01 is the tuning window) |
| Validation months | Last 3 WHO-observed months inside the window (auto-detected) |
| Training inner | Tuning window minus validation months |
| Grid | `lambda_year` × `lambda_reg` × `lambda_lag_reg` (3 params, 7×5×5 = 175 combos default) |
| Selection metric | Minimum validation RMSE on WHO-observed months |
| Output | `outputs_lambda_tuning/lambda_tuning_results.csv` + printed best triplet |

**Data processing aligned with `step2_nowcast_mosquito_log.ipynb`**:

| Aspect | Treatment (same as mosquito_log) |
|---|---|
| Target | Raw WHO cases; `log1p` applied **inside** `train_step2_joint_loss` |
| `x_tilde` lags | `log1p(x_tilde_raw)` — consistent with log1p-transformed target |
| Year constraint | `n_months × log1p(od_total / n_months)` — in log space |
| `make_predictions` | `expm1(X @ beta)` back to WHO units |
| `target_scale` | `1.0` (log1p transform; no additional scaling) |
| `clip_nonnegative` | `False` (expm1 guarantees > −1) |
| Mosquito feature | `use_mosquito=True`, `log1p` transform |
| Standardization | Per training-inner window (no leakage to validation) |

The notebook contains no rolling evaluation or plotting — it is a pure tuning tool.

---

## 5. Data Reading / Compatibility

### Current Data Reading Architecture

```
dengue_updater/data/processed/
    monthly_data.csv  (wide, 6 cols)
    yearly_data.csv
          ↓
    load_processed_data.py   ← unified adapter layer (project root)
          ↓
    preprocessing.py          |  step2 notebooks
    load_master_csv()         |  load_master_csv()
    load_wiki_monthly()       |  load_wiki_monthly()
                              |  load_mosquito_monthly()
          ↓
    downstream modeling logic  (unchanged)
```

### Adapter Layer `load_processed_data.py`

| Function | Return format | Replaces legacy file |
|---|---|---|
| `build_master_df()` | long format: `[resolution, date, value, source]` | `master_data.csv` |
| `get_wiki_dengue_df()` | `[Month, Total_Views]` | `total_dengue_views.csv` |
| `get_mosquito_df()` | `[timestamp, TOTAL_MONTHLY_VIEWS]` | `monthly_mosquito_aggregate.csv` |
| `load_monthly_raw()` | wide format; returns monthly_data.csv as-is | — |
| `load_yearly_raw()` | returns yearly_data.csv as-is | — |

Paths are anchored via `Path(__file__).resolve().parent`, independent of working directory.

### Data Loading (Current — No Fallback)

All data loading now goes directly through the adapter layer with no file-existence checks or fallback logic:

```python
# preprocessing.py
load_master_csv()        → build_master_df()         (always)
load_wiki_monthly()      → get_wiki_dengue_df()       (always)

# notebooks
load_master_csv()        → build_master_df()
load_wiki_monthly(cfg)   → get_wiki_dengue_df()
load_mosquito_monthly(cfg) → get_mosquito_df()
```

The legacy `master_data.csv`, `total_dengue_views.csv`, and `monthly_mosquito_aggregate.csv` files are no longer read by any code. They can be deleted.

### Status of Legacy Files

The following three files still exist in the project root but are **no longer read by any code**:

- `master_data.csv` (15 KB, manually maintained, data current as of ~2024-01)
- `total_dengue_views.csv` (1.7 KB, manually maintained)
- `monthly_mosquito_aggregate.csv` (2.4 KB, manually maintained)

---

## 6. Current Preprocessing Logic

### Step 1 (`preprocessing.py`)

| Feature | Treatment |
|---|---|
| Google Trends | Forward/backward fill for missing values; then **z-score standardized** (`g_fever_z`, `g_vaccine_z`) |
| Wikipedia dengue pageviews | `log1p` transform, then **z-score standardized** (`wiki_views_z`); no imputation |
| WHO monthly data | Missing values are NOT imputed; used only as a loss mask |
| Month seasonality | 11 month dummies (drop_first; January is the reference month) |
| Target scaling | Divided by `target_scale=1000` during training; multiplied back at output |
| Non-negativity | `clip_nonnegative=True`; negative predictions are clipped to 0 |
| Annual constraint | Applied only to years where **all 12 months are present** in the training window |

### Step 2 (Notebooks)

| Feature | Treatment |
|---|---|
| Google Trends | Divided by 100, normalized to [0, 1]; ffill/bfill applied |
| Wikipedia dengue | `log1p` transform; **no imputation** (data assumed complete) |
| Wikipedia mosquito | `log1p` transform (mosquito notebooks only); **no imputation** |
| Lag terms | `x_tilde` (WHO observations / Step 1 estimates blended) / `target_scale` scaled |
| Feature standardization | Standardized using **training-set** mean and std; intercept excluded |
| Missing values | Google Trends only: interpolate + ffill/bfill; lag term NaNs handled at the matrix level |
| Seasonality | Fourier terms sin/cos (K=2, period=12); preferred over month dummies |

---

## 7. Results / Outputs

### Step 1 Outputs (`outputs_step1_wiki/`)

| File | Contents |
|---|---|
| `predictions_step1_monthly.csv` | Smoothed monthly dengue signal — **used as input by Step 2** |
| `params_step1.json` | Model config + learned coefficients β |
| `loss_step1.csv` | Loss value per epoch |
| `yearly_comparison_step1.csv` | Predicted annual totals vs OpenDengue comparison table |
| `who_vs_pred_step1.png` | WHO observations vs model predictions plot |
| `yearly_vs_opendengue_step1.png` | Yearly aggregate comparison plot |
| `loss_curve_step1.png` | Training loss curve |

### Step 2 Outputs — Rolling Evaluation Notebooks (`outputs_step2_*/rolling_splits/`)

These notebooks (`step2_nowcast_mosquito.ipynb`, `step2_nowcast_mosquito_log.ipynb`) still produce retrospective rolling-CV evaluation outputs:

| File | Contents |
|---|---|
| `rolling_predictions_long.csv` | 1-step and 2-step predictions for each split |
| `rolling_split_metrics.csv` | RMSE/MAPE for each split |
| `horizon_specific_predictions_summary.csv` | Predictions summarized by horizon (1-step / 2-step) |
| `rolling_split_coefficients_long.csv` | Coefficients per split (with confidence intervals) |
| `rolling_prediction_intervals_*.csv` | Prediction interval (CI) data |
| `pred_ci_2month_test_window_split_YYYY-MM.png` | Prediction + CI plot for each split |
| `who_vs_pred_step2_split_YYYY-MM.png` | WHO vs prediction plot for each split |
| `first/second_month_ahead_*.png` | 1-step / 2-step ahead values + error plots |

### Step 2 Outputs — Operational Nowcast Notebook (`outputs_updated_nowcast_log/`)

`updated_nowcast_log.ipynb` no longer produces rolling evaluation outputs. **All files in `outputs_updated_nowcast_log/` represent the current operational nowcast** (trained on all available WHO data, predicting forward into the nowcast window). The filenames are kept for dashboard backward compatibility — their contents have been replaced.

**Parent directory (`outputs_updated_nowcast_log/`):**

| File | Contents |
|---|---|
| `latest_nowcast_full_series.csv` | All months from 2021-01 onward with columns: `date, segment, who_observed, step1_est, nowcast, lo_95, hi_95, pred_se_log` |
| `latest_nowcast_summary.json` | Compact summary: three boundaries, per-month predictions + CI, `nowcast_predictions` list |

**Rolling-splits subdirectory (`outputs_updated_nowcast_log/rolling_splits/`):**

> ⚠️ These filenames match the old rolling evaluation schema for dashboard backward compatibility. Their contents now reflect the **current operational nowcast** (not a rolling CV evaluation).

| File | Nominal name | Actual contents |
|---|---|---|
| `rolling_predictions_long.csv` | Rolling predictions | One entry per nowcast horizon step; `split_month` = `who_observed_through` |
| `rolling_prediction_intervals_2month_test_window_long.csv` | Rolling CI | Per-month CI for nowcast window; `is_test_window=1` for nowcast months |
| `rolling_split_metrics.csv` | Rolling metrics | One row: training-set RMSE/MAPE only; `RMSE_test=NaN` (no held-out test set) |
| `horizon_specific_predictions_summary.csv` | Horizon summary | One row per nowcast horizon step |
| `first_month_ahead_values.png` | H=1 plot | Full nowcast visualization (WHO observed + nowcast + CI) |
| `second_month_ahead_values.png` | H=2 plot | Same plot with H=2 month highlighted |

**Files deleted on each re-run (cleanup):**
- `pred_ci_*_test_window_split_*.png` (per-split CI plots)
- `who_vs_pred_step2_split_*.png` (per-split WHO vs prediction plots)
- `rolling_split_coefficients_*.csv` (coefficient tables)

---

## 8. Known Issues / Caveats

### 8.1 Wikipedia Value Magnitude Difference (High Risk)

`Total_Views` in the old `total_dengue_views.csv` is approximately 1/7 of `wikipedia_total_dengue_views` in the new data (~7× larger in new data). The root cause has not been fully verified — likely different language sets or aggregation methods. **After switching to new data, wiki feature coefficients are not directly comparable to historical experiments.**

### 8.2 Reduced Historical Range

New data starts from 2021-01 (`dengue_updater`'s `START_DATE`). Old `master_data.csv` Google Trends started from 2010-11; OpenDengue yearly from 1991. If a model requires 2010–2020 historical features, `START_DATE` must be changed in `dengue_updater/src/config.py` and the updater re-run.

### 8.3 WHO Data Only Available from 2024-01

This is a limitation of the WHO database itself — India's monthly dengue data has only been reported in the WHO Shiny system from 2024-01 onward. NaN values for 2021–2023 are expected behavior, not a fetch failure.

### 8.4 WHO Fetch Layer Is Fragile

`fetch_who.py` layers 1/2 (direct URLs) are currently `None`, so layer 3 (HTML parsing) is actually used. WHO dashboard redesigns may break the fetch; existing values are preserved in that case.

### 8.5 OpenDengue_State_Aggregated Dropped

New data does not include `OpenDengue_State_Aggregated`. The first entry in `yearly_proxy_sources_priority = ('OpenDengue_State_Aggregated', 'OpenDengue_National_Yearly')` will never match and automatically falls back to the second. Current behavior is correct, but the Config comment may be misleading.

### 8.6 Mosquito Pageview Language Definition Change

Old file used 6 languages; new data uses 5 (Gujarati removed), yet new values are ~1.3× higher (indicating changes in other languages as well). Mosquito feature coefficients from historical experiments are not directly comparable to new runs.

### 8.7 Legacy Files No Longer Read

After the 2026-04-05 code simplification, there is no longer any fallback logic — all loading goes directly through `load_processed_data.py`. The legacy CSVs (`master_data.csv`, `total_dengue_views.csv`, `monthly_mosquito_aggregate.csv`) are not read by any code and can be safely deleted.

### 8.8 Wikipedia Old-Style Normalization (`wikipedia_total_dengue_views_normalized`)

The raw `wikipedia_total_dengue_views` in the new pipeline is approximately 7× larger than the old `total_dengue_views.csv` values. The difference arises from a multiplicative month-level re-weighting in the old data (not z-score or min-max).

A corrected column `wikipedia_total_dengue_views_normalized` is now computed and stored directly in `monthly_data.csv` by `update_monthly.add_wiki_normalized_column()`:

```
normalized_t = raw_t × (weighted_t / raw_t)_from_reference × 0.918
```

where the month-level ratio comes from `wikipedia_raw_vs_weighted_pageviews(in).csv`.  
`updated_nowcast_log.ipynb` is configured to use this normalized column by default (`wiki_column = 'wikipedia_total_dengue_views_normalized'` in `Step2Config`). The raw column is preserved alongside it. `load_processed_data.get_wiki_dengue_df()` also returns the normalized column by default.

### 8.9 `load_monthly_features` Index-Alignment Bug (FIXED 2026-04-11)

**Root cause**: The original `load_monthly_features()` function in both `updated_nowcast_log.ipynb` and `updated_nowcast.ipynb` used the following anti-pattern:

```python
# BUGGY — do not use
df['date'] = parse_monthly_date_any(df['date'])
df = df.dropna(subset=['date']).sort_values('date').copy()
wide = pd.DataFrame(index=pd.DatetimeIndex(df['date']))   # DatetimeIndex
wide['Google_Trends_Dengue_fever'] = pd.to_numeric(df['google_trends_dengue_fever'], errors='coerce')
# ↑ df still has RangeIndex → pandas aligns by label → ALL values become NaN
```

After `sort_values`, `df` retains its original `RangeIndex`. When assigning `df[col]` into `wide` (which has a `DatetimeIndex`), pandas cannot align the integer labels to the datetime labels, so every assigned column is entirely `NaN`.

**Impact**: The model silently received `NaN` for all Google Trends, Wikipedia, and mosquito features. Step 2 effectively degenerated to a model using only lag terms, Fourier seasonality, and the intercept — no external signals at all. This is the primary reason the new-pipeline rolling-split results (especially during the growth period) were substantially worse than the old pipeline.

**Fix**: Replaced the buggy block with a `set_index` approach in both notebooks:

```python
# FIXED — safe pattern
df = df.set_index('date')          # DatetimeIndex on df itself
cols = ['google_trends_dengue_fever', 'google_trends_dengue_vaccine',
        'wikipedia_total_dengue_views', 'wikipedia_mosquito_views_total']
if 'wikipedia_total_dengue_views_normalized' in df.columns:
    cols.append('wikipedia_total_dengue_views_normalized')
wide = df[cols].apply(pd.to_numeric, errors='coerce').copy()
wide = wide.rename(columns={
    'google_trends_dengue_fever':  'Google_Trends_Dengue_fever',
    'google_trends_dengue_vaccine': 'Google_Trends_Dengue_vaccine',
})
return wide.sort_index()
```

**Validation (2026-04-11)**:
- Old (buggy) pattern: `Google_Trends_Dengue_fever` non-null count = **0** of 64
- New (fixed) pattern: `Google_Trends_Dengue_fever` non-null count = **63** of 64, all other feature columns similarly non-null

**Files changed**: `updated_nowcast_log.ipynb` (cell 3), `updated_nowcast.ipynb` (cell 3)

> ⚠️ **After this fix, `updated_nowcast_log.ipynb` must be re-run from a clean kernel** to obtain valid rolling-split results. All previously cached outputs in `outputs_updated_nowcast_log/rolling_splits/` were produced with the buggy loader and should be discarded.

### 8.10 Three Canonical Data Boundaries (Added 2026-04-11)

To avoid conflating observed WHO data with imputed or nowcasted values, three explicit time boundaries are now defined and propagated through the pipeline:

| Boundary | Definition | Source |
|---|---|---|
| `who_observed_through` | Last month with an actual WHO-reported case count (`who_cases_monthly` non-null) | `wide['WHO'].dropna().index.max()` in notebook; `design_df['y_who'].notna()` in `preprocessing.py` |
| `external_data_through` | Last month where ALL required external features (Google Trends, Wikipedia, mosquito) are non-null — i.e., the farthest the model can nowcast | `min(wide[c].dropna().index.max() for c in required_cols)` |
| `nowcast_available_through` | Equals `external_data_through`; model can generate a prediction up to and including this month | Same as `external_data_through` |

**As of 2026-04-11 (data current):**
- `who_observed_through` = **2026-02** (last WHO monthly report)
- `external_data_through` = **2026-03** (Google Trends + Wikipedia through March 2026)
- `nowcast_available_through` = **2026-03**

**Where they are used:**
- `preprocessing.py`: `compute_data_boundaries(design_df)` function; result saved in `params_step1.json` under `data_boundaries` key
- `updated_nowcast_log.ipynb` Cells 8–9: computed from `wide` DataFrame (Cell 8); used to define training mask, nowcast months, recursive prediction loop, and all file saves (Cell 9)

**Segment labels in `latest_nowcast_full_series.csv`:**
- `observed` — months with actual WHO reported cases (2024-01 through `who_observed_through`)
- `step1_imputed` — training months without WHO data (2021-01 through 2023-12, where Step 1 smooth signal is used as proxy)
- `nowcast` — months after `who_observed_through` through `nowcast_available_through`

**Current nowcast outputs** (`outputs_updated_nowcast_log/`):
- `latest_nowcast_full_series.csv` — all months with segment labels, `who_observed`, `step1_est`, `nowcast`, `lo_95`, `hi_95`, `pred_se_log`
- `latest_nowcast_summary.json` — compact summary: all three boundaries + per-month predictions + CI + `latest_2_months` key for dashboard

**Rolling split outputs in `outputs_updated_nowcast_log/rolling_splits/`** have been replaced with operational nowcast content — see §8.12 and the updated §7 table for details.

### 8.11 Google Trends Normalization Drifts Over Time

`pytrends` re-fetches the full historical window and re-normalizes each time. Historical values may shift slightly between runs — this is not a bug; it is inherent to how Google Trends works.

### 8.12 `updated_nowcast_log.ipynb` Output Strategy Change (2026-04-12)

**Summary of change**: Rolling evaluation has been removed from `updated_nowcast_log.ipynb`. All output files now contain the **current operational nowcast** (trained on all available WHO data, predicting forward using all available external signals).

**What was removed**:
- Rolling split loop (11 splits, 2025-02 → 2025-12)
- Per-split CI plots (`pred_ci_2month_test_window_split_YYYY-MM.png`)
- Per-split WHO vs prediction plots (`who_vs_pred_step2_split_YYYY-MM.png`)
- Coefficient tables (`rolling_split_coefficients_*.csv`) — 4 variants

**What replaced them**:
- A single final model trained on all months ≤ `who_observed_through`
- A recursive multi-step nowcast loop for months (`who_observed_through + 1` → `external_data_through`)
- Hessian-based Var(β) for prediction intervals (delta method in log1p space)
- All old output filenames reused — content replaced with current nowcast

**Filenames retained for dashboard compatibility**:
- `rolling_predictions_long.csv` — now contains one row per nowcast horizon step; `split_month` = `who_observed_through` (not a true rolling split)
- `rolling_prediction_intervals_2month_test_window_long.csv` — always written under this exact name (dashboard hardcodes it); also written under the dynamic-horizon name if different
- `rolling_split_metrics.csv` — one row: training metrics only; `RMSE_test` = NaN
- `horizon_specific_predictions_summary.csv` — one row per nowcast horizon step
- `first_month_ahead_values.png`, `second_month_ahead_values.png` — full nowcast visualization

**Cleanup on re-run** (Cell 8 deletes before writing new files):
- `pred_ci_*_test_window_split_*.png`
- `who_vs_pred_step2_split_*.png`
- `rolling_split_coefficients_*.csv` (all 4 variants)
- `*_month_ahead_squared_error.png`

**Notebook structure** (10 cells, as of 2026-04-12):

| Cell | Type | Purpose |
|---|---|---|
| 0 | markdown | Title |
| 1 | code | Imports |
| 2 | code | `Step2Config` dataclass |
| 3 | code | Helper functions (`build_design_matrix`, `train_step2_joint_loss`, `compute_rolling_split_range`, etc.) |
| 4 | code | Metrics / CI / plotting helpers |
| 5 | markdown | §1 header |
| 6 | code | Data loading, `who_display_start`, rolling range (kept for reference; no longer drives output) |
| 7 | markdown | §2 header — three boundaries, file table |
| 8 | code | Cleanup → boundaries → final model → Hessian CI → recursive nowcast → DataFrames → summary dict |
| 9 | code | Save all output files + nowcast visualization |

**External boundary is cfg-driven** (not a hardcoded column list): Cell 8 builds the required-features list dynamically from `cfg.google_sources`, `cfg.use_wiki`/`cfg.wiki_column`, and `cfg.use_mosquito`, then takes the `min(latest non-null month)` across those columns.

---

## 9. Recommended Entry Points

If picking up the project or navigating quickly, consult files in this order:

| Priority | File | Reason |
|---|---|---|
| ★★★ | `PROJECT_HANDOFF.md` (this file) | Project overview; start here |
| ★★★ | `dengue_updater/data/processed/monthly_data.csv` | The actual data currently used for modeling |
| ★★★ | `load_processed_data.py` | Unified data reading entry point |
| ★★★ | `preprocessing.py` | Complete Step 1 model implementation |
| ★★ | `step2_nowcast_mosquito.ipynb` | Primary Step 2 notebook |
| ★★ | `dengue_updater/src/main.py` | Data pipeline entry point |
| ★★ | `dengue_updater/src/config.py` | Path config and data schema |
| ★ | `dengue_updater/config/wiki_title_map.yaml` | Wikipedia language / article definitions |
| ★ | `MIGRATION_NOTES.md` | Detailed technical record of the data migration |
| ★ | `outputs_step1_wiki/params_step1.json` | Most recent Step 1 training parameters |
| ★ | `outputs_step2_mosquito/rolling_splits/rolling_split_metrics.csv` | Most recent Step 2 evaluation results |

---

## 10. Progress Log / What Has Been Done So Far

| Period | Work done |
|---|---|
| Before 2026-01 | Manually curated `master_data.csv`, `total_dengue_views.csv`, `monthly_mosquito_aggregate.csv` |
| 2026-01–02 | Built `preprocessing.py` (Step 1); iterative development of multiple Step 2 notebooks |
| ~2026-03 | Built `dengue_updater/` automated data pipeline (WHO / Google Trends / OpenDengue / Wikipedia) |
| 2026-03–04 | Configured GitHub Actions automation (triggers on 5th / 20th of each month; auto-commits updated data) |
| 2026-04-05 | **Data reading migration**: added `load_processed_data.py` adapter layer; modified `preprocessing.py` and 3 notebooks to fall back automatically from legacy CSVs to `dengue_updater/data/processed/` |
| 2026-04-05 | Wrote `MIGRATION_NOTES.md` (migration technical details) and this file `PROJECT_HANDOFF.md` (project overview) |
| 2026-04-05 | **Code simplification**: removed file-existence-check fallback pattern from all data loading functions; removed wiki/mosquito imputation (ffill/bfill) since processed data is assumed complete; removed `data_path`/`wiki_path`/`wiki_month_col`/`wiki_value_col` from `Config` and corresponding CLI args; all three notebooks updated in Cell 3 |
| 2026-04-05 | **Lambda tuning notebook simplified**: rewritten to single fixed split (`tune_split_month=2025-01`), validation = last 3 WHO-observed months in window, 3-param grid (`lambda_year`, `lambda_reg`, `lambda_lag_reg`); rolling eval and plotting removed; output is `outputs_lambda_tuning/lambda_tuning_results.csv` |
| 2026-04-05 | **Lambda tuning notebook aligned with `step2_nowcast_mosquito_log.ipynb`**: Cells 1/3/4 copied verbatim; data processing now identical (log1p target, `x_tilde=log1p(raw)`, year constraint in log space, `expm1` back-transform, mosquito features, `target_scale=1.0`); `step2_nowcast_mosquito_log.ipynb` was NOT modified |
| 2026-04-05 | **Rolling split auto-range** (implemented): removed `rolling_start`/`rolling_end` fields from `Step2Config` (Cell 2); added `compute_rolling_split_range()` to Cell 3 of `step2_nowcast_mosquito_log.ipynb`; Cell 6 calls this function to auto-compute the range from the last valid WHO month (rule: −12 months → −2 months); fixed `load_master_csv(cfg.data_path)` → `load_master_csv()` bug in Cell 6; replaced hardcoded `'2025-12'` legend position guards in Cells 8 and 10 with `rolling_end` variable. Returns 4 values: `rolling_start`, `rolling_end`, `last_who_month`, `rolling_split_months`. |
| 2026-04-05 | **Step 1 optimization (Priority 1)**: (1) z-score standardize `g_fever`, `g_vaccine`, `wiki_views` over modeling window before entering model; raw values kept in output CSV; scaler stats saved to `params_step1.json`. (2) Default loss weights changed to `λ_who=5.0, λ_year=0.3` to prioritize WHO monthly fit and reduce yearly-constraint peak flattening. Not yet done: lag features, Fourier seasonality. |

| 2026-04-11 | **Wikipedia old-style normalization**: Confirmed the old `total_dengue_views.csv` was produced by a multiplicative month-level re-weighting (`weighted = raw × ratio_t`), not z-score/min-max. Added `wikipedia_total_dengue_views_normalized` column to `monthly_data.csv` (computed by `update_monthly.add_wiki_normalized_column()` using reference file + global scale 0.918). Added `wiki_column` config param to `updated_nowcast_log.ipynb`; defaults to `_normalized`. Updated `load_processed_data.get_wiki_dengue_df()` to return the normalized column. |
| 2026-04-11 | **Critical loader bug fixed**: `load_monthly_features()` in `updated_nowcast_log.ipynb` and `updated_nowcast.ipynb` used a buggy `pd.DataFrame(DatetimeIndex)` + `df[col]` assignment pattern that caused pandas index-alignment to produce all-NaN feature columns. Fixed in both notebooks by using `df.set_index('date')` before column selection. Validated: all feature columns now have 63–64 non-null values. **Notebooks must be re-run from a clean kernel.** |
| 2026-04-11 | **Data boundaries explicitly defined** (`preprocessing.py` + `updated_nowcast_log.ipynb`): Three canonical timestamps added throughout the pipeline — see §8.10 for definitions. `preprocessing.py` now computes and saves them in `params_step1.json`. `updated_nowcast_log.ipynb` uses them to generate the operational current nowcast. |
| 2026-04-12 | **`updated_nowcast_log.ipynb` restructured — rolling evaluation removed, all outputs replaced with current operational nowcast**: Notebook reduced from ~15 cells to 10. Old cells (rolling split loop, per-split plots, coefficient tables) removed. New Cells 8-9 implement: (1) cleanup of stale rolling output files on re-run, (2) cfg-driven external boundary, (3) final model training on all observed WHO months, (4) Hessian-based CI, (5) recursive multi-step nowcast, (6) save all outputs. All existing filenames in `outputs_updated_nowcast_log/rolling_splits/` retained for dashboard backward compatibility — content replaced with current nowcast. See §8.12 for full details. |

### Open / To Be Confirmed

- [x] Verify the definition difference between `wikipedia_total_dengue_views` and old `total_dengue_views.csv` — confirmed as multiplicative month-level re-weighting (not language set difference alone)
- [ ] **Re-run `updated_nowcast_log.ipynb` from a clean kernel** — loader bug is fixed; all outputs in `outputs_updated_nowcast_log/rolling_splits/` and the new `latest_nowcast_full_series.csv` / `latest_nowcast_summary.json` must be regenerated
- [ ] After clean re-run, compare new rolling-split RMSE vs old pipeline to assess whether remaining gap is from data-source differences or modeling choices
- [ ] Consider deleting the three legacy CSVs from the root directory once fully committed to new data
- [ ] If 2010–2020 historical Google Trends data is needed, update `START_DATE` in `dengue_updater/src/config.py` and re-run the updater
- [ ] Populate `WHO_DIRECT_URL` / `WHO_DOWNLOAD_URL` in `dengue_updater/src/config.py` (see config for instructions)
