# Dengue Nowcast Dashboard

Streamlit dashboard for the India Dengue Nowcast project.

## Quick start

From the **project root**:

```bash
# Option A — run directly (reads output files live, no pre-computation needed)
streamlit run dashboard/app.py

# Option B — pre-compute data cache, then run (faster startup)
python dashboard/prepare_dashboard_data.py
streamlit run dashboard/app.py
```

## Files

| File | Purpose |
|---|---|
| `app.py` | Main Streamlit UI |
| `utils.py` | Data loading and formatting helpers |
| `prepare_dashboard_data.py` | (Optional) pre-computes `data/latest_nowcast.json` |
| `data/latest_nowcast.json` | Pre-computed data cache (created by prepare script) |

## Data sources

| Purpose | Source file |
|---|---|
| **Latest WHO update** (metric card) | Last non-null `who_cases_monthly` in `dengue_updater/data/processed/monthly_data.csv` |
| **Main chart — WHO observed line** | `who_cases_monthly` column from `monthly_data.csv` |
| **Main chart — 2-month nowcast line + CI** | `rolling_predictions_long.csv` + `rolling_prediction_intervals_2month_test_window_long.csv` (latest split only) |
| Rolling performance metrics | `rolling_split_metrics.csv` |
| Backtest chart (in diagnostics expander) | `horizon_specific_predictions_summary.csv` (h=1 across all splits) |
| Data status flags | `monthly_data.csv` column presence |

## Dependencies

```
streamlit
plotly
pandas
```

Install with:

```bash
pip install streamlit plotly pandas
```
