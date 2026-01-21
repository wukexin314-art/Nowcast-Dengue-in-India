# Nowcast Dengue in India

A lightweight, end-to-end pipeline for **nowcasting dengue cases in India** using a combination of:
- **Official surveillance data (WHO)**
- **Digital proxy signals (Google Trends)**
- **Optional yearly proxy totals (OpenDengue)**
  
The goal is to generate **timely monthly dengue estimates** when official reporting may be delayed.

---

## Repository Structure
.
├── master_data.csv
├── model_step1_google_only.py
├── predictions_step1_monthly.csv
├── step2_nowcast_train_test_split.ipynb
├── step2_nowcast_loss_function_train_test_split.ipynb
├── step2_nowcast_mixed_lags.ipynb
├── outputs_step1/
└── outputs_step2/


### Key Files
- **`master_data.csv`**  
  Unified dataset containing monthly signals (WHO-like dengue series, Google Trends series, and optional yearly proxy totals).

- **`model_step1_google_only.py`**  
  Step 1 modeling script. Produces the monthly proxy prediction and saves plots/results to `outputs_step1/`.

- **`predictions_step1_monthly.csv`**  
  Cached Step 1 output (monthly predictions). Useful if you want to skip re-running Step 1.

- **`step2_nowcast_mixed_lags.ipynb`**
  The original Step 2 code without train/test split.

- **`step2_nowcast_train_test_split.ipynb`**  
  Step 2 nowcasting pipeline with train/test split (baseline workflow).

- **`step2_nowcast_loss_function_train_test_split.ipynb`**  
  Step 2 variant using a customized loss / objective for training.

- **`outputs_step1/`, `outputs_step2/`**  
  Saved figures, evaluation summaries, and exported prediction tables.

---

## Data Sources (High-Level)
This repo is designed around:
- **Official dengue surveillance signals** (monthly/annual)
- **Google Trends** signals as early indicators (e.g., “dengue fever”, “dengue vaccine”, etc.)
- Optional yearly totals/proxies (if available in `master_data.csv`)

> Notes:
> - Google Trends is normalized (0–100) and may shift slightly across downloads/time windows.
> - If you regenerate Trends data, keep the same region/time window settings for reproducibility.
