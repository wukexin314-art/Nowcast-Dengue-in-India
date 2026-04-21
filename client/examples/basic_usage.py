"""
basic_usage.py — Minimal runnable examples for dengue-nowcast-client.

Usage (from the client/ directory after installing the package):
    python examples/basic_usage.py

Or without installing (editable mode):
    pip install -e client/
    python client/examples/basic_usage.py
"""
from dengue_nowcast_client import (
    DengueNowcastClient,
    DengueNowcastConnectionError,
    DengueNowcastHTTPError,
)

client = DengueNowcastClient()   # default: production API

# ── 1. Health check ──────────────────────────────────────────────────────────
print("=== /health ===")
h = client.health()
print(f"  status        : {h['status']}")
print(f"  store_loaded  : {h['store_loaded']}")
print(f"  n_predictions : {h['n_predictions']}")
print(f"  n_splits      : {h['n_splits']}")
print(f"  split range   : {h['split_month_min']} → {h['split_month_max']}")
print()

# ── 2. Available split months ────────────────────────────────────────────────
print("=== /splits ===")
sp = client.splits()
print(f"  {sp['count']} splits: {sp['items']}")
print()

# ── 3. Predictions for one split ─────────────────────────────────────────────
print("=== /predictions?split_month=2025-06 ===")
p = client.predictions(split_month="2025-06")
print(f"  count: {p['count']}")
for row in p["items"]:
    print(
        f"  {row['split_month']} → {row['target_month']} "
        f"(h={row['horizon']})  pred={row['point_pred']:,.0f}  "
        f"CI=[{row['ci_low']:,.0f}, {row['ci_high']:,.0f}]  "
        f"actual={row['y_true']}"
    )
print()

# ── 4. Prediction history for one target month ───────────────────────────────
print("=== /predictions/by-target?target_month=2026-03 ===")
bt = client.predictions_by_target("2026-03")
print(f"  count: {bt['count']}")
for row in bt["items"]:
    print(
        f"  split={row['split_month']}  h={row['horizon']}  "
        f"pred={row['point_pred']:,.0f}"
    )
print()

# ── 5. Evaluation metrics ────────────────────────────────────────────────────
print("=== /evaluation ===")
ev = client.evaluation()
print(f"  count: {ev['count']}")
for row in ev["items"]:
    rmse = row["RMSE_test"]
    rmse_str = f"{rmse:,.1f}" if rmse is not None else "N/A"
    print(f"  {row['split_month']}  RMSE_test={rmse_str}  MAPE_test={row['MAPE_test_pct']}%")
print()

# ── 6. Error handling example ────────────────────────────────────────────────
print("=== Error handling ===")
try:
    bad_client = DengueNowcastClient(base_url="http://localhost:9999", timeout=3)
    bad_client.health()
except DengueNowcastConnectionError as e:
    print(f"  Connection error (expected): {e}")
except DengueNowcastHTTPError as e:
    print(f"  HTTP error: {e.status_code} — {e.detail}")

# ── 7. Context manager ───────────────────────────────────────────────────────
print()
print("=== Context manager ===")
with DengueNowcastClient() as c:
    result = c.splits()
    print(f"  Got {result['count']} splits inside context manager")
print("  Session closed automatically.")
