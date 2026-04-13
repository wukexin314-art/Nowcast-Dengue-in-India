"""
prepare_dashboard_data.py — Pre-compute dashboard data and write to
dashboard/data/latest_nowcast.json for stable, fast frontend consumption.

Usage (from project root):
    python dashboard/prepare_dashboard_data.py

Running this script is optional. If the JSON is absent, dashboard/app.py falls
back to reading the raw output files directly via utils.build_live().
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Make sure we can import utils from this directory
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import _build_live  # noqa: E402  (local import after path setup)

OUT_DIR = Path(__file__).resolve().parent / "data"
OUT_FILE = OUT_DIR / "latest_nowcast.json"


def main() -> None:
    print("Building dashboard data from rolling split outputs...")
    data = _build_live()

    if "error" in data:
        print(f"ERROR: {data['error']}")
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")

    print(f"Saved: {OUT_FILE}")
    print(f"  Latest split month : {data.get('latest_split_month', 'N/A')}")
    print(f"  External data through: {data.get('external_data_through', 'N/A')}")
    for p in data.get("nowcast_predictions", []):
        lo = f"{p['lo_95']:,.0f}" if p.get("lo_95") is not None else "N/A"
        hi = f"{p['hi_95']:,.0f}" if p.get("hi_95") is not None else "N/A"
        print(f"  {p['month']} (h={p['horizon_step']}): {p['predicted']:,.0f}  [{lo} – {hi}]")
    metrics = data.get("rolling_metrics") or {}
    print(f"  Rolling RMSE (mean): {metrics.get('rmse_mean', 'N/A')}")
    print(f"  Rolling MAPE (mean): {metrics.get('mape_mean', 'N/A')}%")
    print(f"  Last data refresh  : {data.get('last_data_refresh', 'N/A')}")


if __name__ == "__main__":
    main()
