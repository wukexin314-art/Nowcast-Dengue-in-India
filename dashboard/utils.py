"""
utils.py — Data loading and preparation helpers for the Dengue Nowcast dashboard.

Priority order:
  1. outputs_updated_nowcast_log/latest_nowcast_summary.json  (notebook output — authoritative)
  2. Directly from outputs_updated_nowcast_log/rolling_splits/ (live build)
  3. dashboard/data/latest_nowcast.json  (stale pre-computed cache — last resort only)

All public functions return plain Python dicts/lists for easy JSON serialisation
and straightforward Streamlit consumption.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs_updated_nowcast_log" / "rolling_splits"
MONTHLY_DATA = PROJECT_ROOT / "dengue_updater" / "data" / "processed" / "monthly_data.csv"
DASHBOARD_JSON = Path(__file__).resolve().parent / "data" / "latest_nowcast.json"
# Canonical nowcast summary produced by updated_nowcast_log.ipynb Cell 12
NOWCAST_SUMMARY = PROJECT_ROOT / "outputs_updated_nowcast_log" / "latest_nowcast_summary.json"
NOWCAST_FULL_SERIES = PROJECT_ROOT / "outputs_updated_nowcast_log" / "latest_nowcast_full_series.csv"


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def load_dashboard_data() -> dict[str, Any]:
    """Return dashboard data dict.

    Priority:
      1. outputs_updated_nowcast_log/latest_nowcast_summary.json  (notebook output — authoritative)
      2. Live build from raw rolling-split output files
      3. dashboard/data/latest_nowcast.json  (stale pre-computed cache — last resort only)
    """
    if NOWCAST_SUMMARY.exists():
        return _build_from_nowcast_summary()
    if (OUTPUTS_DIR / "rolling_predictions_long.csv").exists():
        return _build_live()
    if DASHBOARD_JSON.exists():
        with open(DASHBOARD_JSON, encoding="utf-8") as f:
            return json.load(f)
    return {"error": "No dashboard data source found"}


def _build_from_nowcast_summary() -> dict[str, Any]:
    """Build dashboard data from latest_nowcast_summary.json (notebook output)."""
    with open(NOWCAST_SUMMARY, encoding="utf-8") as f:
        summary = json.load(f)

    result: dict[str, Any] = {}

    # ── last data refresh: mtime of the summary file itself ──────────────────
    result["last_data_refresh"] = datetime.fromtimestamp(
        NOWCAST_SUMMARY.stat().st_mtime
    ).isoformat()

    # ── nowcast metadata ──────────────────────────────────────────────────────
    # latest_split_month = who_observed_through (the rolling/training cutoff)
    result["latest_split_month"]  = summary.get("who_observed_through", "N/A")
    result["nowcast_predictions"] = summary.get("nowcast_predictions", [])
    # NOTE: do NOT propagate the stale "nowcast_window" field from the summary;
    # the dashboard recomputes it dynamically from rolling_end + external_data_through.

    # ── WHO observed series (still read from monthly_data.csv) ───────────────
    result["who_observed"] = _load_who_series()

    # ── external data boundary — prefer summary field, fall back to monthly CSV
    _ext_from_summary = summary.get("external_data_through")
    _ext_computed: str | None = None
    if MONTHLY_DATA.exists():
        mdf = pd.read_csv(MONTHLY_DATA)
        _ext_proxy_cols = [
            "google_trends_dengue_fever",
            "google_trends_dengue_vaccine",
            "wikipedia_total_dengue_views_normalized",
            "wikipedia_total_dengue_views",
            "wikipedia_mosquito_views_total",
        ]
        _ext_latest: list[str] = []
        for _col in _ext_proxy_cols:
            if _col in mdf.columns:
                _nonnull = mdf.dropna(subset=[_col])["date"]
                if not _nonnull.empty:
                    _ext_latest.append(_nonnull.max())
        _ext_computed = max(_ext_latest) if _ext_latest else None
    _ext_through = _ext_from_summary or _ext_computed

    # ── data freshness flags ─────────────────────────────────────────────────
    result["data_flags"] = {
        "who_data_through":          summary.get("who_observed_through"),
        "external_data_through":     _ext_through,
        "google_trends_available":   True,
        "wikipedia_available":       True,
        "mosquito_available":        True,
        "rolling_results_available": (OUTPUTS_DIR / "rolling_predictions_long.csv").exists(),
    }
    if MONTHLY_DATA.exists():
        result["data_flags"].update({
            "google_trends_available": _col_has_data(mdf, "google_trends_dengue_fever"),
            "wikipedia_available":     _col_has_data(mdf, "wikipedia_total_dengue_views"),
            "mosquito_available":      _col_has_data(mdf, "wikipedia_mosquito_views_total"),
        })

    # ── rolling metrics (still from rolling_split_metrics.csv) ───────────────
    result["rolling_metrics"] = _load_rolling_metrics()

    # ── h1 series for diagnostics expander ───────────────────────────────────
    result["nowcast_series_h1"] = _load_h1_series()

    # ── boundary fields at top level for app.py ──────────────────────────────
    result["who_observed_through"]    = summary.get("who_observed_through")
    result["external_data_through"]   = _ext_through
    result["nowcast_available_through"] = summary.get("nowcast_available_through")

    return result


# ---------------------------------------------------------------------------
# Live builder (reads raw output files directly)
# ---------------------------------------------------------------------------

def _build_live() -> dict[str, Any]:
    result: dict[str, Any] = {}

    # --- last data refresh ---------------------------------------------------
    mtime_candidates = [
        OUTPUTS_DIR / "rolling_predictions_long.csv",
        MONTHLY_DATA,
    ]
    mtimes = [p.stat().st_mtime for p in mtime_candidates if p.exists()]
    result["last_data_refresh"] = (
        datetime.fromtimestamp(max(mtimes)).isoformat() if mtimes else None
    )

    # --- rolling predictions -------------------------------------------------
    pred_path = OUTPUTS_DIR / "rolling_predictions_long.csv"
    if not pred_path.exists():
        return {"error": f"Required file not found: {pred_path}"}

    pred_df = pd.read_csv(pred_path)
    latest_split = pred_df["split_month"].max()
    latest_preds = pred_df[pred_df["split_month"] == latest_split].sort_values(
        "horizon_step"
    )

    result["latest_split_month"] = latest_split
    # NOTE: do NOT set "nowcast_window" here; the dashboard recomputes it dynamically.
    predictions = [
        {
            "month": row["target_month"],
            "horizon_step": int(row["horizon_step"]),
            "predicted": round(float(row["x_pred"]), 1),
            "lo_95": None,
            "hi_95": None,
        }
        for _, row in latest_preds.iterrows()
    ]

    # --- CI ------------------------------------------------------------------
    ci_path = OUTPUTS_DIR / "rolling_prediction_intervals_2month_test_window_long.csv"
    if ci_path.exists():
        ci_df = pd.read_csv(ci_path)
        test_rows = ci_df[
            (ci_df["split_month"] == latest_split) & (ci_df["is_test_window"] == 1)
        ]
        ci_map: dict[str, dict] = {}
        for _, row in test_rows.iterrows():
            lo = row.get("pred_lo_95_test_window_only")
            hi = row.get("pred_hi_95_test_window_only")
            ci_map[str(row["date"])] = {
                "lo_95": float(lo) if pd.notna(lo) else None,
                "hi_95": float(hi) if pd.notna(hi) else None,
            }
        for pred in predictions:
            ci = ci_map.get(pred["month"], {})
            pred["lo_95"] = ci.get("lo_95")
            pred["hi_95"] = ci.get("hi_95")

    result["nowcast_predictions"] = predictions

    # --- WHO observed series -------------------------------------------------
    who_series = _load_who_series()
    result["who_observed"] = who_series

    # --- data freshness flags + external boundary ----------------------------
    monthly_df = pd.read_csv(MONTHLY_DATA)
    who_months = [r["date"] for r in who_series]

    _ext_proxy_cols = [
        "google_trends_dengue_fever",
        "google_trends_dengue_vaccine",
        "wikipedia_total_dengue_views_normalized",
        "wikipedia_total_dengue_views",
        "wikipedia_mosquito_views_total",
    ]
    _ext_latest: list[str] = []
    for _col in _ext_proxy_cols:
        if _col in monthly_df.columns:
            _nonnull = monthly_df.dropna(subset=[_col])["date"]
            if not _nonnull.empty:
                _ext_latest.append(_nonnull.max())
    _ext_through = max(_ext_latest) if _ext_latest else None

    result["data_flags"] = {
        "who_data_through":          max(who_months) if who_months else None,
        "external_data_through":     _ext_through,
        "google_trends_available":   _col_has_data(monthly_df, "google_trends_dengue_fever"),
        "wikipedia_available":       _col_has_data(monthly_df, "wikipedia_total_dengue_views"),
        "mosquito_available":        _col_has_data(monthly_df, "wikipedia_mosquito_views_total"),
        "rolling_results_available": pred_path.exists(),
    }
    result["external_data_through"] = _ext_through

    # --- rolling metrics -----------------------------------------------------
    result["rolling_metrics"] = _load_rolling_metrics()

    # --- h1 nowcast series (for diagnostics / rolling backtest chart) --------
    result["nowcast_series_h1"] = _load_h1_series()

    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_who_series() -> list[dict]:
    if not MONTHLY_DATA.exists():
        return []
    df = pd.read_csv(MONTHLY_DATA, usecols=["date", "who_cases_monthly"])
    df["who_cases_monthly"] = pd.to_numeric(df["who_cases_monthly"], errors="coerce")
    df = df.dropna(subset=["who_cases_monthly"]).sort_values("date")
    return [
        {"date": row["date"], "value": float(row["who_cases_monthly"])}
        for _, row in df.iterrows()
    ]


def _load_rolling_metrics() -> dict | None:
    metrics_path = OUTPUTS_DIR / "rolling_split_metrics.csv"
    if not metrics_path.exists():
        return None
    df = pd.read_csv(metrics_path)
    rmse_mean = float(df["RMSE_test"].mean()) if "RMSE_test" in df.columns else None
    mape_mean = float(df["MAPE_test_%"].mean()) if "MAPE_test_%" in df.columns else None
    bench_rmse = (
        float(df["RMSE_test_seasonal_naive"].mean())
        if "RMSE_test_seasonal_naive" in df.columns
        else None
    )
    bench_mape = (
        float(df["MAPE_test_seasonal_naive_%"].mean())
        if "MAPE_test_seasonal_naive_%" in df.columns
        else None
    )
    return {
        "rmse_mean": round(rmse_mean, 1) if rmse_mean is not None else None,
        "mape_mean": round(mape_mean, 2) if mape_mean is not None else None,
        "n_splits": len(df),
        "benchmark_rmse_mean": round(bench_rmse, 1) if bench_rmse is not None else None,
        "benchmark_mape_mean": round(bench_mape, 2) if bench_mape is not None else None,
        "benchmark_label": "Seasonal naive (same month, prior year)",
    }


def _load_h1_series() -> list[dict]:
    path = OUTPUTS_DIR / "horizon_specific_predictions_summary.csv"
    if not path.exists():
        return []
    df = pd.read_csv(path)
    h1 = df[df["horizon_step"] == 1][["target_month", "x_pred"]].copy()
    h1 = h1.sort_values("target_month")
    return [
        {"date": row["target_month"], "predicted": float(row["x_pred"])}
        for _, row in h1.iterrows()
    ]


def _col_has_data(df: pd.DataFrame, col: str) -> bool:
    return bool(col in df.columns and df[col].notna().any())


# ---------------------------------------------------------------------------
# Formatting helpers (used by app.py)
# ---------------------------------------------------------------------------

def fmt_cases(val: float | None) -> str:
    if val is None:
        return "N/A"
    return f"{int(round(val)):,}"


def fmt_date_refresh(iso: str | None) -> str:
    if not iso:
        return "Unknown"
    try:
        dt = datetime.fromisoformat(iso)
        return dt.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return str(iso)
