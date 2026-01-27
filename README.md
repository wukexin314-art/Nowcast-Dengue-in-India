# Nowcast Dengue in India

A lightweight, end-to-end pipeline for **nowcasting dengue cases in India** using a combination of:
- **Official surveillance data (WHO)**
- **Digital proxy signals (Google Trends)**
- **Optional yearly proxy totals (OpenDengue)**
  
The goal is to generate **timely monthly dengue estimates** when official reporting may be delayed.

---

## Contents
- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Data](#data)
- [Method Summary](#method-summary)
  - [Step 1: Monthly Signal Construction](#step-1-monthly-signal-construction)
  - [Step 2: Nowcasting Model](#step-2-nowcasting-model)
- [Repository Structure](#repository-structure)
- [Quickstart](#quickstart)
- [Reproducibility & Outputs](#reproducibility--outputs)
- [Evaluation](#evaluation)
- [Notes & Limitations](#notes--limitations)

---

## Project Overview
This repository implements a **two-step dengue nowcasting framework**:

1. **Step 1 (Signal construction / calibration)**  
   Build a **monthly dengue activity signal** by jointly fitting to multiple data sources (monthly + yearly proxies), producing a smooth and stable monthly estimate.

2. **Step 2 (Nowcasting / prediction)**  
   Use Step 1’s monthly signal + exogenous signals (e.g., Google Trends; optionally Wikipedia pageviews) to train a model for **nowcasting** dengue activity with:
   - train/test split experiments
   - rolling time splits (time-series CV)
   - multiple lag specifications
   - RMSE / MAPE evaluation and prediction-vs-observed plots

---

## Key Features
- End-to-end pipeline: data → Step1 monthly signal → Step2 nowcast
- Rolling time splits for realistic evaluation
- Mixed lag / multi-feature specifications (easy to extend)
- Optional inclusion of **Wikipedia pageviews** as an additional exogenous signal
- Outputs include:
  - prediction tables (CSV/XLSX)
  - performance metrics (RMSE/MAPE)
  - prediction vs. observed plots for each split

---

## Data
The main data file is typically stored as:

- `master_data.csv` (or similarly named)

Recommended long format:
- `date`: monthly timestamp (e.g., `YYYY-MM-01`)
- `source`: data source name (e.g., `WHO`, `Google_Trends_Dengue_fever`, `Google_Trends_Dengue_vaccine`, `OpenDengue_*`, `Wiki_*`)
- `value`: numeric value

Common sources used in this project:
- **WHO** (monthly official / aggregated proxy)
- **Google Trends**: dengue-related keywords (e.g., *dengue fever*, *dengue vaccine*)
- **OpenDengue proxies**: yearly or state/national aggregates (depending on availability)
- **Wikipedia pageviews** (optional): dengue-related pageviews (possibly by language)

---

## Method Summary

### Step 1: Monthly Signal Construction
Goal: estimate a latent monthly dengue activity signal \( \hat{y}_t \) that aligns with multiple noisy targets.

Typical ingredients:
- Monthly target proxy (e.g., WHO monthly series)
- Yearly target proxy (e.g., OpenDengue annual totals or other annual aggregates)
- Regularization / smoothing to avoid overfitting
- Joint loss (weighted), optimized with gradient-based methods

Outputs:
- `predictions_step1_monthly.csv` (or similarly named)
- diagnostic plots in `outputs_step1/`

---

### Step 2: Nowcasting Model
Goal: train a model using:
- Step 1 monthly signal (as target or as a key feature, depending on design)
- Exogenous signals (Google Trends; optionally Wikipedia)
- Lagged terms (single / mixed lags)

Validation:
- simple train/test split (e.g., specific month cut)
- rolling splits across multiple months (e.g., from 2025-01 to 2025-08)

Outputs:
- `outputs_step2/` figures
- `.xlsx` summary tables (e.g., `output_step2.xlsx`)
- metrics tables for each split

---

## Repository Structure

| File / Folder | Description |
|---|---|
| `master_data.csv` | Unified dataset (WHO + Google Trends + optional proxies) |
| `model_step1_google_only.py` | Step 1: train Google-only proxy model |
| `predictions_step1_monthly.csv` | Cached Step 1 monthly predictions |
| `step2_nowcast_train_test_split.ipynb` | Step 2 baseline nowcasting (train/test split) |
| `step2_nowcast_loss_function_train_test_split.ipynb` | Step 2 variant with custom loss |
| `outputs_step1/` | Step 1 plots and logs |
| `outputs_step2/` | Step 2 plots, metrics, and exported predictions |

---

## Quickstart
### 1) Environment

Recommended:
- Python 3.9+ (3.10/3.11 also OK)
- Jupyter Notebook / JupyterLab

---

### 2) Run Step 1 (monthly signal)

Run the Step 1 script (example):
```bash
python model_step1_google_only.py
```

Expected outputs:
- predictions_step1_monthly.csv
- figures and logs under outputs_step1/

---

### 3) Run Step 2 (nowcasting)

Open notebooks and run from top to bottom:
- step2_nowcast_train_test_split.ipynb (basic split)
- step2_nowcast_loss_function_train_test_split.ipynb (custom loss version, if applicable)
- step2_nowcast_loss_function_train_test_split_rolling.ipynb (rolling validation)
- step2_nowcast_loss_function_train_test_split_rolling_wiki.ipynb (rolling + wiki features)
- step2_nowcast_mixed_lags.ipynb (mixed lag specs)

Expected outputs:
- prediction vs observed plots for each split
- metric summaries (RMSE/MAPE)
- excel outputs (e.g., output_step2.xlsx)

---

## Reproducibility & Outputs

Typical output locations:
- outputs_step1/: Step 1 training diagnostics & monthly signal plots
- outputs_step2/: Step 2 prediction plots, rolling split figures
- predictions_step1_monthly.csv: intermediate artifact used by Step 2
- output_step2.xlsx: consolidated results (may contain multiple sheets, e.g., no_wiki vs wiki)

Suggested naming conventions:
- Figures: pred_vs_who_split_YYYY-MM.png
- Metrics: metrics_rolling.csv / cv_metrics.csv

---

## Evaluation

Common metrics used:
- RMSE: root mean squared error
- MAPE: mean absolute percentage error

Recommended reporting:
- metrics on test windows only
- for rolling splits: summarize distribution (mean/median) and show per-split plots

---

## Notes & Limitations

- Source alignment matters: monthly vs yearly proxies require careful aggregation / weighting.
- Google Trends and Wikipedia pageviews are noisy and may have structural breaks.
- Model performance can vary significantly by split window; prefer rolling evaluation over one-off splits.
