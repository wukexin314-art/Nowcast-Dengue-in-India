"""
build_prediction_store.py — Build normalised Parquet stores from rolling-split CSVs.

Run from the project root:
    python api/build_prediction_store.py

Reads:
    outputs_updated_nowcast_log/rolling_splits/rolling_predictions_long.csv
    outputs_updated_nowcast_log/rolling_splits/rolling_prediction_intervals_2month_test_window_long.csv
    outputs_updated_nowcast_log/rolling_splits/rolling_split_metrics.csv

Writes:
    api/data/historical_predictions.parquet
    api/data/evaluation.parquet
    api/data/metadata.json
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPLITS_DIR   = PROJECT_ROOT / "outputs_updated_nowcast_log" / "rolling_splits"
OUT_DIR      = Path(__file__).resolve().parent / "data"

PRED_CSV     = SPLITS_DIR / "rolling_predictions_long.csv"
CI_CSV       = SPLITS_DIR / "rolling_prediction_intervals_2month_test_window_long.csv"
METRICS_CSV  = SPLITS_DIR / "rolling_split_metrics.csv"

OUT_PRED     = OUT_DIR / "historical_predictions.parquet"
OUT_EVAL     = OUT_DIR / "evaluation.parquet"
OUT_META     = OUT_DIR / "metadata.json"

MODEL_NAME    = "nowcast"
MODEL_VERSION = "log_arx_v1"


# ---------------------------------------------------------------------------
# Build historical_predictions.parquet
# ---------------------------------------------------------------------------

def build_predictions() -> pd.DataFrame:
    if not PRED_CSV.exists():
        raise FileNotFoundError(f"Required file not found: {PRED_CSV}")

    # ── A: main prediction table ────────────────────────────────────────────
    pred = pd.read_csv(PRED_CSV)

    # Normalise column names to expected schema
    pred = pred.rename(columns={
        "horizon_step": "horizon",
        "x_pred":       "point_pred",
        "y_who":        "y_true",
    })

    # Keep only the columns we need from the main table
    keep = ["split_month", "target_month", "horizon", "point_pred",
            "y_true", "benchmark_t12", "model_squared_error", "benchmark_squared_error"]
    pred = pred[[c for c in keep if c in pred.columns]].copy()

    # ── B: CI table — filter to test-window rows with actual predictions ────
    ci_cols = {
        "split_month":               "split_month",
        "date":                      "target_month",
        "x_pred_test_window_only":   "x_pred_ci_window",   # not used in output, just for filter
        "pred_lo_95_test_window_only": "ci_low",
        "pred_hi_95_test_window_only": "ci_high",
    }
    if CI_CSV.exists():
        ci = pd.read_csv(CI_CSV, usecols=list(ci_cols.keys()))
        ci = ci.rename(columns=ci_cols)
        ci = ci[
            (ci.get("is_test_window", pd.Series(dtype=int)) == 1
             if "is_test_window" in pd.read_csv(CI_CSV, nrows=0).columns
             else ci["ci_low"].notna())
            & ci["ci_low"].notna()
        ]
        # Re-read with is_test_window
        ci_full = pd.read_csv(CI_CSV)
        ci_full = ci_full.rename(columns={"date": "target_month"})
        ci_full = ci_full[
            (ci_full["is_test_window"] == 1) & ci_full["pred_lo_95_test_window_only"].notna()
        ][["split_month", "target_month",
           "pred_lo_95_test_window_only", "pred_hi_95_test_window_only"]].rename(columns={
            "pred_lo_95_test_window_only": "ci_low",
            "pred_hi_95_test_window_only": "ci_high",
        })
    else:
        ci_full = pd.DataFrame(columns=["split_month", "target_month", "ci_low", "ci_high"])

    # ── C: Merge CI onto main predictions ──────────────────────────────────
    df = pred.merge(ci_full, on=["split_month", "target_month"], how="left")

    # ── D: Add fixed / derived fields ──────────────────────────────────────
    df["model_name"]    = MODEL_NAME
    df["model_version"] = MODEL_VERSION
    df["is_observed"]   = df["y_true"].notna()
    df["feature_set"]   = ""          # placeholder — not yet derivable from outputs
    df["data_version"]  = ""          # placeholder
    df["source_file"]   = "rolling_predictions_long.csv"
    df["created_at"]    = datetime.now(timezone.utc).isoformat()

    # ── E: Enforce types ───────────────────────────────────────────────────
    for col in ["point_pred", "ci_low", "ci_high", "y_true",
                "benchmark_t12", "model_squared_error", "benchmark_squared_error"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["horizon"]      = df["horizon"].astype(int)
    df["is_observed"]  = df["is_observed"].astype(bool)

    # ── F: Column order ────────────────────────────────────────────────────
    ordered = [
        "model_name", "model_version",
        "split_month", "target_month", "horizon",
        "point_pred", "ci_low", "ci_high",
        "y_true", "is_observed",
        "benchmark_t12", "model_squared_error", "benchmark_squared_error",
        "feature_set", "data_version", "source_file", "created_at",
    ]
    df = df[[c for c in ordered if c in df.columns]]
    return df


# ---------------------------------------------------------------------------
# Build evaluation.parquet
# ---------------------------------------------------------------------------

def build_evaluation() -> pd.DataFrame:
    if not METRICS_CSV.exists():
        raise FileNotFoundError(f"Required file not found: {METRICS_CSV}")

    df = pd.read_csv(METRICS_CSV)

    # Rename columns with % to API-safe names
    rename = {
        "MAPE_train_%":                 "MAPE_train_pct",
        "MAPE_test_%":                  "MAPE_test_pct",
        "MAPE_test_seasonal_naive_%":   "MAPE_test_seasonal_naive_pct",
    }
    df = df.rename(columns=rename)

    df["model_name"]    = MODEL_NAME
    df["model_version"] = MODEL_VERSION
    df["created_at"]    = datetime.now(timezone.utc).isoformat()

    return df


# ---------------------------------------------------------------------------
# Write metadata
# ---------------------------------------------------------------------------

def write_metadata(pred_df: pd.DataFrame, eval_df: pd.DataFrame) -> None:
    split_months = sorted(pred_df["split_month"].unique().tolist())
    meta = {
        "model_name":         MODEL_NAME,
        "model_version":      MODEL_VERSION,
        "n_predictions":      int(len(pred_df)),
        "n_splits":           int(len(split_months)),
        "split_months":       split_months,
        "split_month_min":    split_months[0] if split_months else None,
        "split_month_max":    split_months[-1] if split_months else None,
        "target_month_min":   pred_df["target_month"].min(),
        "target_month_max":   pred_df["target_month"].max(),
        "horizons_available": sorted(pred_df["horizon"].unique().tolist()),
        "n_evaluation_rows":  int(len(eval_df)),
        "source_files": [
            str(PRED_CSV.name),
            str(CI_CSV.name),
            str(METRICS_CSV.name),
        ],
        "built_at": datetime.now(timezone.utc).isoformat(),
    }
    OUT_META.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Building historical_predictions.parquet …")
    pred_df = build_predictions()
    pred_df.to_parquet(OUT_PRED, index=False)
    print(f"  Wrote {len(pred_df)} rows → {OUT_PRED}")
    print(f"  split_months : {sorted(pred_df['split_month'].unique())}")
    print(f"  horizons     : {sorted(pred_df['horizon'].unique())}")
    ci_count = pred_df["ci_low"].notna().sum()
    print(f"  rows with CI : {ci_count}/{len(pred_df)}")

    print("\nBuilding evaluation.parquet …")
    eval_df = build_evaluation()
    eval_df.to_parquet(OUT_EVAL, index=False)
    print(f"  Wrote {len(eval_df)} rows → {OUT_EVAL}")

    print("\nWriting metadata.json …")
    write_metadata(pred_df, eval_df)
    print(f"  Wrote → {OUT_META}")

    print("\nDone.")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
