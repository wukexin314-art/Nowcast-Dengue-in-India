# dengue_updater — Comprehensive Handoff Note

> Written: 2026-03-30 | Last updated: 2026-04-12
> Purpose: preserve all design decisions, corrections, source-validation results, and open items so work can resume after context compaction.

---

## 1. Project Goal

Build and maintain a production-quality Python data updater tool (`dengue_updater/`) inside the existing `Nowcast Dengue in India/` project. The tool:

- Fetches India dengue surveillance data from 5 sources (WHO, Google Trends ×2, OpenDengue, Wikipedia ×2 topics)
- Maintains `data/processed/monthly_data.csv` and `data/processed/yearly_data.csv`
- Exports `data/processed/master_data.xlsx` with 2 sheets
- Supports loose failure mode (one source failing does not stop others)
- Never overwrites existing non-null values with nulls on failure
- Preserves full history from 2021-01 onward
- Produces append-only `logs/update_log.csv` + timestamped `logs/run_summary_{YYYYMMDD_HHMMSS}.json` per run
- One-time migration from legacy files already implemented in `src/migrate_legacy.py`

---

## 2. Final Output Files and Schema

### monthly_data.csv

Key: `date` (YYYY-MM string)

| Column | Type | Source |
|---|---|---|
| date | str YYYY-MM | — |
| who_cases_monthly | float | WHO Shiny dashboard |
| google_trends_dengue_fever | float | pytrends |
| google_trends_dengue_vaccine | float | pytrends |
| wikipedia_total_dengue_views | float | Wikimedia API (7 langs summed) |
| wikipedia_total_dengue_views_normalized | float | Derived: `raw × ratio_t × 0.918` (see §wiki-normalization) |
| wikipedia_mosquito_views_total | float | Wikimedia API (5 langs summed) |

### yearly_data.csv

Key: `year` (YYYY string)

| Column | Type | Source |
|---|---|---|
| year | str YYYY | — |
| open_dengue_national_yearly | float | OpenDengue GitHub CSV |

### master_data.xlsx

Sheet 1: `monthly_data` (copy of monthly_data.csv)
Sheet 2: `yearly_data` (copy of yearly_data.csv)

### Logging outputs

- `logs/update_log.csv` — append-only, one row per source per run
- `logs/run_summary_{YYYYMMDD_HHMMSS}.json` — one file per run, never overwritten

---

## 3. Per-Source Rules

### 3a. WHO Monthly

**Status: IMPLEMENTED AND WORKING** (as of 2026-03-30)

**Fetch mechanism (Layer 1 — "shiny"):**
1. HTTP GET the dashboard HTML at `https://worldhealthorg.shinyapps.io/dengue_global/` → extract `_w_{32-char hex}` worker prefix via regex
2. Open native WebSocket at `wss://worldhealthorg.shinyapps.io/dengue_global/{worker_prefix}/websocket/`
3. Receive config message → extract `sessionId` and `workerId`
4. Send Shiny `init` message with `.clientdata_output_dl_all_data_hidden: False` and other download-related outputs marked visible
5. Receive values message containing `dl_all_data` download path
6. Send `closeModal` button click (accepts terms of use modal — **mandatory, download 500s without this**)
7. Navigate to download tab: set tabset to `meta`, sub-tab to `dl_data` (tabset IDs extracted dynamically from HTML via regex `data-tabsetid="(\d+)"`)
8. Set `dl_filter_adm_level` to `global`
9. HTTP GET the download URL **while WebSocket is still open** → receives ~739KB Excel file
10. Parse Excel `data` sheet → filter `country == "India"` → normalize dates → return DataFrame

**Excel structure:** Sheet `data` has columns: `date, date_lab, who_region, who_region_long, country, iso3, cases, confirmed_cases, severe_cases, deaths, cfr, prop_sev, ...`
Sheet `metadata` has 192 countries with population and serotype info.

**Critical constraint:** India monthly data in the WHO system starts at **2024-01**, not 2021-01. This is a data source limitation — India only began monthly reporting to WHO from January 2024. Other countries have data back to 2010–2014. This was confirmed by:
- Inspecting the full downloaded Excel (10,906 rows, 192 countries, India = 26 rows from 2024-01 to 2026-02)
- Inspecting the legacy `master_data.csv` in the repo (WHO source also starts 2024-01)
- The dashboard's older visuals for India are based on yearly aggregated data, not monthly

**Fallback layers:**
- Layer 2 (download): `config.WHO_DOWNLOAD_URL` — manually set direct URL (currently None)
- Layer 3 (html): BeautifulSoup parse of dashboard HTML (fragile, currently never finds tables)

**Acceptance criterion:** Returns India monthly new cases from 2024-01 onward. Months 2021-01 through 2023-12 will be NaN for `who_cases_monthly`. This is correct and expected.

**Dependencies:** `websockets>=12.0` (added to requirements.txt)

### 3b. Google Trends

**Status: IMPLEMENTED** (not yet integration-tested in a full run)

- Keywords: `"dengue fever"`, `"dengue vaccine"`
- Geo: `IN` (India)
- Always re-fetches full window `2021-01-01` to today (pytrends renormalizes the relative index per request window)
- Drops `isPartial=True` rows
- Exponential backoff on 429/rate-limit errors, up to 5 retries

**CRITICAL failure rule (corrected during design review):**
- On failure: **NEVER write null into persisted `monthly_data.csv`**
- Failed runs appear ONLY in logs, run summary, and the in-memory diagnostics
- The `apply_update()` function receives `None` and short-circuits, preserving all existing stored values
- On success with `is_google_trends=True`: full-window replacement — overwrites ALL stored months for that keyword column (because the index renormalizes)

**Verified coverage:** 2021-01 to 2026-03 (63 rows each keyword) — matches intended range.

### 3c. OpenDengue Yearly

**Status: IMPLEMENTED BUT URL IS BROKEN — NEEDS FIX**

**Current config URL (404):**
```
https://raw.githubusercontent.com/OpenDengue/master-repo/main/data/processed/Admin0_Cases_1990_2024.csv
```

**Correct URL (200, verified 2026-03-30):**
```
https://raw.githubusercontent.com/OpenDengue/master-repo/main/data/releases/V1.1/National_extract_V1_1.csv
```

**Actual schema of National_extract_V1_1.csv:**
- Columns: `adm_0_name, adm_1_name, adm_2_name, full_name, ISO_A0, FAO_GAUL_code, RNE_iso_code, calendar_start_date, calendar_end_date, Year, dengue_total, S_res, T_res, sourceID`
- 32,223 rows total
- India rows: 39 (years 1991–2022)
- Filter: `adm_0_name == "INDIA"` (uppercase in source)
- Must also filter `S_res == "Admin0"` for national-level only (not state-aggregated)
- Latest year: **2022** (academic release lag)

**The current `fetch_open_dengue.py` parser:**
- `_COUNTRY_COLS` includes `adm_0_name` → will match ✓
- `_YEAR_COLS` includes `Year` → will match ✓
- `_CASES_COLS` includes `dengue_total` → will match ✓
- `_INDIA_NAMES` = `{"india", "ind"}` — but actual value is `"INDIA"` (uppercase). The code does `.str.strip().str.lower().isin(...)` so this **will match** ✓
- **Missing:** No filter for `S_res == "Admin0"`. The India rows have mixed S_res values. Need to add `S_res` filter or accept that all 39 India rows are Admin0-level (they appear to be based on inspection, all have `S_res == "Admin0"`). **Should verify and add filter to be safe.**

**Required fix:** Update `config.OPENDENGUE_GITHUB_CSV` to the correct URL.

**Coverage:** India years 2021 and 2022 only (2 rows). Years 2023+ not yet in OpenDengue V1.1.

### 3d. Wikipedia Dengue Pageviews

**Status: IMPLEMENTED** (not yet integration-tested in a full run)

**Languages (7):** hi (Hindi), kn (Kannada), ml (Malayalam), mr (Marathi), ta (Tamil), te (Telugu), bn (Bengali)

**Page titles (from `config/wiki_title_map.yaml`):**
- hi: `डेंगू_बुख़ार`
- kn: `ಡೆಂಗೇ`
- ml: `ഡെങ്കിപ്പനി`
- mr: `डेंग्यू_ताप`
- ta: `முடக்குக்_காய்ச்சல்`
- te: `డెంగ్యూ_జ్వరం`
- bn: `ডেঙ্গু_জ্বর`

**API:** `https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/{lang}.wikipedia/all-access/all-agents/{title}/monthly/{start}/{end}`

**Aggregation:** Sum across all successful languages per month. Missing languages treated as 0. Partial success logged but still produces a result.

**Verified coverage:** 2021-01 to 2026-03 (63 months for English test; all languages use same API with same date range).

### 3e. Wikipedia Mosquito Pageviews

**Status: IMPLEMENTED** (not yet integration-tested in a full run)

**Languages (5):** hi (Hindi), ta (Tamil), mr (Marathi), ml (Malayalam), kn (Kannada)

**Page titles (from `config/wiki_title_map.yaml`):**
- hi: `मच्छर`
- ta: `கொசு`
- mr: `डास`
- ml: `കൊതുക്`
- kn: `ಸೊಳ್ಳೆ`

**Definition change from legacy:** Legacy used 6 languages (included Gujarati `gu`). New system uses 5 languages (Gujarati dropped). First successful fetch will overwrite migrated values with the new 5-language sum. This is documented in `migrate_legacy.py` and logged.

**Verified coverage:** 2021-01 to 2026-03.

---

## 4. Agreed Overwrite/Update Logic

Implemented in `update_monthly.py:apply_update()` and `update_yearly.py:apply_yearly_update()`.

### Standard sources (WHO, Wikipedia, OpenDengue):
- If fetch **failed** (new_df is None): **preserve all existing values** — no changes
- If fetch **succeeded**:
  - New value is non-null → overwrite existing (even if existing was non-null)
  - New value is null → preserve existing (never overwrite with null)
  - New date not in existing → add new row with the value
  - Existing dates absent from fetch → preserve existing values

### Google Trends special case (`is_google_trends=True`):
- If fetch **failed**: **NEVER modify existing data** — this is a run-level failure, not real missing data
- If fetch **succeeded**: full-window replacement
  - Non-null new values overwrite all existing
  - Null new values clear the existing value (because pytrends renormalizes the full window)
  - This is correct because the Google Trends index is relative and renormalized per request window

---

## 5. Agreed Failure-Handling Logic

- **Loose failure mode:** Each source is fetched independently. One source failing does not abort others.
- **No null-on-failure writes:** A failed fetch NEVER writes nulls or empty values to persisted CSVs.
- **Overall status logic:**
  - All SUCCESS → SUCCESS
  - All FAILED → FAILED
  - Any mix of FAILED/PARTIAL with SUCCESS → PARTIAL
- **Wikipedia partial success:** If some languages succeed but others fail, the aggregate is still produced (with a lower sum) and status is PARTIAL. Failed languages are logged.

---

## 6. Logging and Run-Summary Rules

### update_log.csv (append-only)
Columns: `run_time, source_name, status, fetch_mode, rows_fetched, rows_added, rows_updated, expected_languages, successful_languages, error_message`

One row per source per run. Never overwritten.

### run_summary_{YYYYMMDD_HHMMSS}.json (one per run)
- **Timestamped filename** — never overwrites previous summaries
- Contains: `run_time, overall_status, successful_sources, failed_sources, partial_sources, wiki_dengue_coverage, wiki_mosquito_coverage, output_files, notes`
- Google Trends failure note explicitly mentions "existing stored values unchanged"

### Per-run log file
`logs/run_{YYYYMMDD_HHMMSS}.log` — full console output mirrored to file.

---

## 7. Confirmed Page-Title Mappings

All titles are defined in `config/wiki_title_map.yaml`. Title validation is done at runtime by `verify_titles.py` (probes Wikimedia API for each title before fetching).

**Dengue (7 languages):** hi, kn, ml, mr, ta, te, bn — titles listed in section 3d above.
**Mosquito (5 languages):** hi, ta, mr, ml, kn — titles listed in section 3e above.

---

## 8. Confirmed Date Coverage Requirements

| Source | Intended Start | Actual Earliest Retrievable | Actual Latest Retrievable | Matches? | Notes |
|---|---|---|---|---|---|
| WHO monthly | 2021-01 | **2024-01** | 2026-02 | **No** — source limitation | India started monthly WHO reporting Jan 2024. Months 2021-01 to 2023-12 will be NaN. This is correct. |
| Google Trends "dengue fever" | 2021-01 | 2021-01 | 2026-03 | Yes | |
| Google Trends "dengue vaccine" | 2021-01 | 2021-01 | 2026-03 | Yes | |
| OpenDengue yearly | 2021 | 2021 | **2022** | **Partial** | V1.1 release only goes to 2022. Years 2023+ will be NaN until OpenDengue updates. |
| Wikipedia dengue | 2021-01 | 2021-01 | 2026-03 | Yes | |
| Wikipedia mosquito | 2021-01 | 2021-01 | 2026-03 | Yes | |

---

## 9. Open Questions / Unresolved Checks Still Pending

### Must fix before first production run:

1. **OpenDengue URL is broken (404).** Must update `config.OPENDENGUE_GITHUB_CSV` from `data/processed/Admin0_Cases_1990_2024.csv` to `data/releases/V1.1/National_extract_V1_1.csv`. The parser column matching appears compatible but needs an integration test.

2. **OpenDengue S_res filter.** All 39 India rows in V1.1 appear to have `S_res == "Admin0"`, but this should be explicitly filtered in the parser to prevent state-aggregated rows from leaking in if the dataset changes. The current code does NOT filter on S_res.

3. **Full integration test (`python src/main.py`).** Has never been run end-to-end with all 5 sources. Individual source fetchers have been tested in isolation:
   - WHO: tested and working (returns 26 rows, 2024-01 to 2026-02) ✓
   - Google Trends: tested in isolation (63 rows each keyword) ✓
   - OpenDengue: NOT tested (URL is 404)
   - Wikipedia: NOT integration-tested (individual API calls verified)
   - Migration (`migrate_legacy.py --dry-run`): tested successfully (62 monthly rows, yearly rows empty as expected)

4. **WHO Shiny robustness concerns:**
   - Worker prefix changes per deployment (extracted dynamically from HTML — resilient)
   - Tabset IDs are numeric and may change (extracted dynamically — resilient)
   - `closeModal` button ID could change (hardcoded — moderate risk)
   - The download requires navigating to `meta` → `dl_data` tab and setting `dl_filter_adm_level=global` — these string values are UI-specific and could change

### Nice to have / future:

5. **Legacy migration has not been run for real** (only dry-run tested). Should run once before first production update.

6. **Wikipedia title validation** (`verify_titles.py`) makes 12 API calls at startup. Could cache results.

---

## 10. Current Implementation Status

### Files implemented (all under `dengue_updater/`):

| File | Status | Notes |
|---|---|---|
| `requirements.txt` | Done | requests, pandas, pytrends, openpyxl, pyyaml, websockets |
| `config/wiki_title_map.yaml` | Done | 7 dengue + 5 mosquito language titles |
| `src/config.py` | **Needs fix** | `OPENDENGUE_GITHUB_CSV` URL is 404 |
| `src/verify_titles.py` | Done | |
| `src/fetch_who.py` | Done | 3-layer fallback, Shiny WebSocket download working |
| `src/fetch_google_trends.py` | Done | Full-window fetch, backoff, strict failure semantics |
| `src/fetch_open_dengue.py` | **Needs fix** | URL broken; may need S_res filter |
| `src/fetch_wikipedia.py` | Done | Per-language fetch + aggregate |
| `src/update_monthly.py` | Done | Overwrite rules including Google Trends special case |
| `src/update_yearly.py` | Done | Standard overwrite rules |
| `src/export_outputs.py` | Done | 2-sheet Excel export |
| `src/migrate_legacy.py` | Done | One-time migration with mosquito definition change note |
| `src/main.py` | Done | Orchestrator with --skip-* and --dry-run |

### What's left to code:

1. Fix `config.OPENDENGUE_GITHUB_CSV` URL
2. Optionally add `S_res == "Admin0"` filter in `fetch_open_dengue.py`
3. Run full integration test
4. Run legacy migration (non-dry-run)

---

## 11. Corrections Made During Discussion That Override Earlier Assumptions

These are decisions that were initially proposed differently and then corrected during design review:

1. **WHO date range:** Initially assumed WHO India data goes back to 2021-01. **CORRECTED:** WHO India monthly data starts at 2024-01 only. This is a data source limitation, not a code bug. Confirmed by inspecting the full downloaded Excel.

2. **OpenDengue state-aggregated fallback:** Initially proposed falling back to state-aggregated India data if national direct records were missing. **CORRECTED (Round 1 review):** Must use only national direct records (`S_res == "Admin0"`). No state-aggregated fallback.

3. **Google Trends failure behavior:** Initially proposed that failed Google Trends runs could write nulls to `monthly_data.csv` for missing months. **CORRECTED (Round 2 review):** Failed Google Trends runs must NEVER touch persisted CSV. Failure is a run-level event only (appears in logs/run_summary). Existing stored values are always preserved on failure.

4. **Run summary file naming:** Initially proposed overwriting a single `run_summary.json` each run. **CORRECTED (Round 2 review):** Each run produces a timestamped file `run_summary_{YYYYMMDD_HHMMSS}.json` — never overwrites previous summaries.

5. **OpenDengue column naming:** Initially proposed renaming the column. **CORRECTED (Round 2 review):** Keep `open_dengue_national_yearly` as-is for schema compatibility. Document that it refers to the aggregated yearly India series.

6. **WHO fetch mechanism:** Initially designed with `WHO_DIRECT_URL` (JSON API) as Layer 1. **CORRECTED:** The WHO xmart-api requires Microsoft Azure AD OAuth authentication (always 302-redirects to login.microsoftonline.com). User explicitly rejected authentication-based approaches. Replaced with Shiny WebSocket download as Layer 1, which works without any authentication.

7. **OpenDengue GitHub URL:** Config was set to `Admin0_Cases_1990_2024.csv` which returns 404. **DISCOVERED:** Correct URL is `National_extract_V1_1.csv`. Must be fixed in `config.py`.

8. **WHO Shiny download flow:** Initially attempted HTTP GET on download URL after closing WebSocket → 404. Then tried keeping WebSocket open but skipping terms acceptance → 500. **FINAL WORKING FLOW:** Must (a) accept terms via `closeModal`, (b) navigate to download tab (`meta` → `dl_data`), (c) set `dl_filter_adm_level=global`, (d) download while WebSocket stays open → 200 with valid Excel.

---

## 12. Changes Made 2026-04-11

### 12.1 Wikipedia Old-Style Normalization Column Added

**Problem**: `wikipedia_total_dengue_views` in the new pipeline is ~7× larger than the old `total_dengue_views.csv` values, because the old data applied a month-level multiplicative re-weighting (not z-score/min-max). This caused the wiki feature to enter the Step 2 model at a completely different scale than it was trained on historically.

**Fix**: Added `add_wiki_normalized_column()` to `src/update_monthly.py`. This function:
1. Reads `wikipedia_raw_vs_weighted_pageviews(in).csv` (project root)
2. Computes `ratio_t = weighted_total_pageviews_t / raw_total_pageviews_t` per month
3. Fills missing months via ffill/bfill
4. Writes `wikipedia_total_dengue_views_normalized = raw × ratio_t × 0.918` into `monthly_data.csv`

The original `wikipedia_total_dengue_views` column is preserved unchanged.

`src/main.py` now calls `add_wiki_normalized_column(monthly_df)` before every `save_monthly_data()` call, so the column is always current after any data update run.

**Config constants added to `src/config.py`**:
```python
WIKI_NORM_REFERENCE_CSV = PROJECT_ROOT.parent / "wikipedia_raw_vs_weighted_pageviews(in).csv"
WIKI_NORM_GLOBAL_SCALE  = 0.918
```

**Modeling impact**: `updated_nowcast_log.ipynb` defaults to `wiki_column = 'wikipedia_total_dengue_views_normalized'`. `load_processed_data.get_wiki_dengue_df()` also returns the normalized column by default (falls back to raw if column absent).

### 12.2 Critical `load_monthly_features` Bug Fixed in Step 2 Notebooks

**Bug**: Both `updated_nowcast_log.ipynb` (cell 3) and `updated_nowcast.ipynb` (cell 3) had:

```python
# BUGGY PATTERN — produces all-NaN columns
df = df.sort_values('date').copy()                          # keeps RangeIndex
wide = pd.DataFrame(index=pd.DatetimeIndex(df['date']))    # DatetimeIndex
wide['Google_Trends_Dengue_fever'] = pd.to_numeric(df['google_trends_dengue_fever'], ...)
# ↑ pandas aligns by label: RangeIndex int vs DatetimeIndex → all NaN
```

**Impact**: All external features (Google Trends, Wikipedia, mosquito) were silently `NaN`. Step 2 was running as a lag-only + seasonality model with no external signals. This is the primary cause of degraded performance in the new pipeline.

**Fix applied** (both notebooks, cell 3):
```python
# FIXED PATTERN
df = df.set_index('date')   # DatetimeIndex on df itself before slicing
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

**Validation**: After fix, non-null counts: `Google_Trends_Dengue_fever=63, Google_Trends_Dengue_vaccine=63, wikipedia_total_dengue_views=64, wikipedia_mosquito_views_total=64, wikipedia_total_dengue_views_normalized=64` (out of 64 rows).

**Next required step**: Re-run `updated_nowcast_log.ipynb` from a clean kernel. Outputs currently in `outputs_updated_nowcast_log/rolling_splits/` were produced with the buggy loader and are invalid.

### 12.3 Data Boundaries and Current Latest Nowcast (2026-04-11 / 2026-04-12)

**Problem** (2026-04-11): Rolling split outputs in `outputs_updated_nowcast_log/rolling_splits/` were retrospective CV evaluations — they did not represent the operational "current nowcast". The pipeline lacked an explicit separation between:
1. True WHO observed endpoint
2. External signal endpoint
3. The resulting nowcast output that uses all available information

**Fix — `preprocessing.py`** (2026-04-11):
Added `compute_data_boundaries(design_df)` function that returns three YYYY-MM strings:
- `who_observed_through` — last non-null `y_who` date
- `external_data_through` — last non-null `g_fever` date (Google Trends, the minimum required feature)
- `nowcast_available_through` — same as `external_data_through`

These are now saved in `params_step1.json` under the `data_boundaries` key after every `preprocessing.py` run.

**Fix — `updated_nowcast_log.ipynb`** (2026-04-12 — major restructuring):

Rolling evaluation has been **completely removed** from this notebook. The notebook was restructured from ~15 cells to **10 cells**. All output files in `outputs_updated_nowcast_log/` now contain the **current operational nowcast** only.

**New notebook structure (10 cells)**:

| Cell | Purpose |
|---|---|
| 0 | Markdown title |
| 1 | Imports |
| 2 | `Step2Config` dataclass |
| 3 | Helper functions (`build_design_matrix`, `train_step2_joint_loss`, `compute_rolling_split_range`, etc.) |
| 4 | Metrics / CI / plotting helpers |
| 5 | §1 markdown header |
| 6 | Data loading, `who_display_start`, rolling range (kept for reference; no longer drives output) |
| 7 | §2 markdown header — three boundaries, file table |
| 8 | Cleanup stale files → compute boundaries → train final model → Hessian CI → recursive nowcast → build DataFrames + summary dict |
| 9 | Save all output files + nowcast visualization |

**Cell 8 implements** (in order):
1. **Cleanup**: deletes stale per-split PNGs and coefficient CSVs from previous runs using glob patterns
2. **External boundary**: dynamically built from active cfg features (`cfg.google_sources`, `cfg.use_wiki`/`cfg.wiki_column`, `cfg.use_mosquito`) — takes `min(latest non-null month)` across all active feature columns
3. **Training**: final model on all months ≤ `who_observed_through`
4. **Hessian CI**: `Var_beta = sigma2 × cfac^2 × (Hinv @ XtX @ Hinv)`; CI via delta method in log1p space (`expm1(log_pred ± z × pred_se)`)
5. **Recursive nowcast**: for each nowcast month, substitutes lag features whose source months were already nowcasted (stores `log1p(pred)` in `_predicted_log1p` dict)
6. **DataFrames**: full-series CSV + rolling-compat CSVs + summary dict

**Segment labels in `latest_nowcast_full_series.csv`**:
- `observed` — 2024-01 through `who_observed_through` (actual WHO data)
- `step1_imputed` — 2021-01 through 2023-12 (no WHO; Step 1 smooth signal used for lags)
- `nowcast` — `who_observed_through + 1` through `nowcast_available_through`

**`latest_nowcast_summary.json` structure**:
```json
{
  "who_observed_through": "2026-02",
  "external_data_through": "2026-03",
  "nowcast_available_through": "2026-03",
  "nowcast_window": "2026-03 to 2026-03",
  "n_nowcast_months": 1,
  "nowcast_predictions": [{"month": "2026-03", "predicted": ..., "lo_95": ..., "hi_95": ...}],
  "latest_1_or_2_months": [...],
  "ci_method": "Normal approximation via Hessian (log1p space, expm1 back-transformed)",
  ...
}
```

**Files in `outputs_updated_nowcast_log/rolling_splits/` — retained names, replaced content**:

> These filenames are kept for dashboard backward compatibility. Content is now the current nowcast, not rolling CV evaluation.

| File | Actual contents |
|---|---|
| `rolling_predictions_long.csv` | One row per nowcast horizon; `split_month` = `who_observed_through` |
| `rolling_prediction_intervals_2month_test_window_long.csv` | Per-month CI; `is_test_window=1` for nowcast months |
| `rolling_split_metrics.csv` | One row: training-set RMSE/MAPE; `RMSE_test=NaN` |
| `horizon_specific_predictions_summary.csv` | One row per nowcast horizon step |
| `first_month_ahead_values.png` | Full nowcast visualization |
| `second_month_ahead_values.png` | Same plot, H=2 highlighted |

**Files deleted on each re-run**:
- `pred_ci_*_test_window_split_*.png`
- `who_vs_pred_step2_split_*.png`
- `rolling_split_coefficients_*.csv` (all 4 variants)
- `*_month_ahead_squared_error.png`

**Important caveat**: Even after fixing this bug, new-pipeline results may not match old-pipeline exactly, because:
- Google Trends values re-normalize across runs
- Wikipedia language sets differ (7 langs new vs. different set old)
- Mosquito language sets differ (5 langs new vs. 6 langs old)
- Yearly proxy source changed (National only vs. State-aggregated)
The bug fix is necessary before any fair apples-to-apples comparison can be made.
