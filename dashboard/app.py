"""
app.py — Streamlit dashboard for India Dengue Nowcast.

Run from project root:
    streamlit run dashboard/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Ensure dashboard/ is on the path for utils import
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import fmt_cases, fmt_date_refresh, load_dashboard_data  # noqa: E402

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Dengue Nowcast in India",
    page_icon="🦟",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
@st.cache_data(ttl=300)
def get_data():
    return load_dashboard_data()

data = get_data()

if "error" in data:
    st.error(f"Failed to load nowcast data: {data['error']}")
    st.stop()

# Unpack convenience variables
latest_split     = data.get("latest_split_month", "N/A")
predictions      = data.get("nowcast_predictions", [])
who_obs          = data.get("who_observed", [])
data_flags       = data.get("data_flags", {})
rolling_metrics  = data.get("rolling_metrics") or {}
h1_series        = data.get("nowcast_series_h1", [])
last_refresh_raw = data.get("last_data_refresh")

# ── Compute nowcast window: rolling_end + 1  →  external_data_through ────────
#   rolling_end_str = latest_split_month (last month of rolling evaluation)
#   ext_through_str = latest month any external proxy signal is available
#                     (always in data_flags; also mirrored at top level)
#   nowcast window  = [rolling_end + 1 month, external_data_through]
rolling_end_str = latest_split if latest_split != "N/A" else None
ext_through_str = (
    data_flags.get("external_data_through")   # preferred: set by both builders
    or data.get("external_data_through")       # top-level mirror
)

nowcast_months_list: list[str] = []
nowcast_window_display = "N/A"
_nowcast_start_str: str | None = None
_nowcast_end_str: str | None = None

if rolling_end_str and ext_through_str:
    try:
        _roll_p = pd.Period(rolling_end_str, freq="M")
        _ext_p  = pd.Period(ext_through_str, freq="M")
        if _ext_p > _roll_p:
            _rng = pd.period_range(_roll_p + 1, _ext_p, freq="M")
            nowcast_months_list = [str(p) for p in _rng]
            _nowcast_start_str = nowcast_months_list[0]
            _nowcast_end_str   = nowcast_months_list[-1]
            nowcast_window_display = (
                f"{_nowcast_start_str} to {_nowcast_end_str}"
                if len(nowcast_months_list) >= 2
                else _nowcast_start_str
            )
    except Exception:
        pass

# ---------------------------------------------------------------------------
# ── Section 1: Title
# ---------------------------------------------------------------------------
st.title("Dengue Nowcast in India")
st.markdown(
    "Multi-source nowcasting dashboard using WHO, Google Trends, Wikipedia, "
    "and mosquito proxy signals"
)
st.caption(f"Last data refresh: **{fmt_date_refresh(last_refresh_raw)}**")
st.divider()

# ---------------------------------------------------------------------------
# ── Section 2: Metric cards
# ---------------------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)

with col1:
    latest_who_month = data_flags.get("who_data_through", "N/A")
    st.metric(
        label="Latest WHO update",
        value=latest_who_month,
        help="Last month with an official WHO reported dengue case count (from monthly_data.csv)",
    )

with col2:
    st.metric(
        label="Nowcast window",
        value=nowcast_window_display,
        help="Months after the rolling evaluation end (rolling_end + 1) through the latest available external proxy signal",
    )

with col3:
    if predictions:
        lines = "\n".join(
            f"{p['month']}: **{fmt_cases(p['predicted'])}**" for p in predictions
        )
        st.markdown(
            f"""
            <div style="border:1px solid #e0e0e0; border-radius:8px; padding:12px 16px; background:#f9f9f9;">
                <div style="font-size:0.85rem; color:#555; margin-bottom:4px;">Predicted cases (next 2 months)</div>
                {"<br/>".join(f"<span style='font-size:1rem;'>{p['month']}: <b>{fmt_cases(p['predicted'])}</b></span>" for p in predictions)}
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.metric("Predicted cases (next 2 months)", "N/A")

with col4:
    st.metric(
        label="Last data refresh",
        value=fmt_date_refresh(last_refresh_raw),
        help="Most recent modification time of the rolling split output files",
    )

st.divider()

# ---------------------------------------------------------------------------
# ── Section 3: Main time-series chart
# ---------------------------------------------------------------------------
st.subheader("Observed WHO and 2-Month Nowcast")

who_df = pd.DataFrame(who_obs)

fig = go.Figure()

# WHO observed (solid line)
if not who_df.empty:
    fig.add_trace(
        go.Scatter(
            x=who_df["date"],
            y=who_df["value"],
            mode="lines+markers",
            name="WHO observed",
            line=dict(color="#1f77b4", width=2),
            marker=dict(size=5),
        )
    )

# Latest 2-month nowcast: CI band + dashed prediction line
# Source: rolling_predictions_long.csv + rolling_prediction_intervals_*_long.csv
#         (latest split only — this is the current nowcast, not a rolling backtest)
if predictions:
    nc_dates = [p["month"] for p in predictions]
    nc_vals  = [p["predicted"] for p in predictions]
    lo_vals  = [p.get("lo_95") for p in predictions]
    hi_vals  = [p.get("hi_95") for p in predictions]

    # CI band (rendered first so it sits behind the prediction line)
    if any(v is not None for v in lo_vals):
        lo_clean = [v if v is not None else nc_vals[i] for i, v in enumerate(lo_vals)]
        hi_clean = [v if v is not None else nc_vals[i] for i, v in enumerate(hi_vals)]
        fig.add_trace(
            go.Scatter(
                x=nc_dates + nc_dates[::-1],
                y=hi_clean + lo_clean[::-1],
                fill="toself",
                fillcolor="rgba(255,127,14,0.22)",
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                showlegend=True,
                name="95% CI",
            )
        )

    # Nowcast dashed line connecting the two prediction months
    fig.add_trace(
        go.Scatter(
            x=nc_dates,
            y=nc_vals,
            mode="lines+markers",
            name="Nowcast",
            line=dict(color="#d62728", width=2, dash="dash"),
            marker=dict(size=9, symbol="diamond"),
            hovertemplate="%{x}: %{y:,.0f} cases<extra>Nowcast</extra>",
        )
    )

# Nowcast window shading + origin line
# Origin = rolling_end (last month of rolling evaluation / model training cutoff)
if rolling_end_str:
    try:
        _origin_dt = pd.Timestamp(rolling_end_str + "-01")
    except Exception:
        _origin_dt = None

    if _origin_dt is not None:
        # Vertical dotted line — annotation_text omitted here (Plotly datetime axis bug)
        fig.add_vline(
            x=_origin_dt.timestamp() * 1000,  # ms-epoch required for datetime x-axes
            line_width=1.5,
            line_dash="dot",
            line_color="#888",
        )
        # Label via separate add_annotation to avoid the add_vline+annotation_text bug
        fig.add_annotation(
            x=_origin_dt,
            y=1.0,
            xref="x",
            yref="paper",
            text=f"Rolling end / nowcast origin ({rolling_end_str})",
            showarrow=False,
            xanchor="left",
            yanchor="top",
            font=dict(size=11, color="#888"),
            bgcolor="rgba(255,255,255,0.7)",
        )

# Light shading over the nowcast window
if _nowcast_start_str and _nowcast_end_str:
    try:
        _shade_x0 = pd.Timestamp(_nowcast_start_str + "-01")
        # extend one month past end so the last month is fully covered
        _shade_x1 = (pd.Period(_nowcast_end_str, freq="M") + 1).to_timestamp()
        fig.add_vrect(
            x0=_shade_x0,
            x1=_shade_x1,
            fillcolor="rgba(255,127,14,0.08)",
            line_width=0,
            layer="below",
        )
    except Exception:
        pass

fig.update_layout(
    xaxis_title="Month",
    yaxis_title="Dengue cases",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=40, r=20, t=40, b=40),
    height=420,
    plot_bgcolor="#fff",
    paper_bgcolor="#fff",
    xaxis=dict(showgrid=True, gridcolor="#eee"),
    yaxis=dict(showgrid=True, gridcolor="#eee"),
)

st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# ── Section 4 & 5: Results table + Data status panel
# ---------------------------------------------------------------------------
col_left, col_right = st.columns([3, 2])

with col_left:
    st.subheader("Latest 2-Month Nowcast Results")
    if predictions:
        table_rows = []
        for p in predictions:
            table_rows.append(
                {
                    "Month": p["month"],
                    "Predicted cases": fmt_cases(p["predicted"]),
                    "Lower 95% CI": fmt_cases(p.get("lo_95")),
                    "Upper 95% CI": fmt_cases(p.get("hi_95")),
                }
            )
        st.dataframe(
            pd.DataFrame(table_rows),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No nowcast predictions available.")

with col_right:
    st.subheader("Data Status")

    def _status_row(label: str, ok: bool | None, detail: str = "") -> None:
        icon = "✅" if ok else ("⚠️" if ok is None else "❌")
        suffix = f" — {detail}" if detail else ""
        st.markdown(f"{icon} **{label}**{suffix}")

    who_through = data_flags.get("who_data_through")
    _status_row(
        "WHO data through",
        bool(who_through),
        who_through or "unavailable",
    )
    _status_row("Google Trends available", data_flags.get("google_trends_available"))
    _status_row("Wikipedia available",     data_flags.get("wikipedia_available"))
    _status_row("Mosquito proxy available", data_flags.get("mosquito_available"))
    _status_row("Rolling results available", data_flags.get("rolling_results_available"))

st.divider()

# ---------------------------------------------------------------------------
# ── Section 6: Nowcast performance summary
# ---------------------------------------------------------------------------
st.subheader("Nowcast Performance Summary")

if rolling_metrics:
    c1, c2, c3 = st.columns(3)
    n = rolling_metrics.get("n_splits", "?")

    with c1:
        rmse = rolling_metrics.get("rmse_mean")
        st.metric(
            "Rolling RMSE",
            f"{rmse:,.1f}" if rmse is not None else "N/A",
            help=f"Mean RMSE across {n} rolling CV splits (2-month test window)",
        )

    with c2:
        mape = rolling_metrics.get("mape_mean")
        st.metric(
            "Rolling MAPE",
            f"{mape:.1f}%" if mape is not None else "N/A",
            help=f"Mean MAPE across {n} rolling CV splits",
        )

    with c3:
        bench_rmse  = rolling_metrics.get("benchmark_rmse_mean")
        bench_label = rolling_metrics.get("benchmark_label", "Benchmark")
        bench_str   = f"{bench_rmse:,.1f}" if bench_rmse is not None else "N/A"
        st.metric(
            "Benchmark RMSE",
            bench_str,
            help=f"{bench_label}. Mean RMSE across {n} rolling splits.",
        )

    with st.expander("Rolling backtest detail"):
        # ── 1-step-ahead backtest chart ──────────────────────────────────────
        # Source: horizon_specific_predictions_summary.csv (h=1 across all splits)
        # This is a diagnostic view showing how well 1-step nowcasts tracked
        # actuals across the rolling evaluation window — NOT the current nowcast.
        h1_df = pd.DataFrame(h1_series)
        if not h1_df.empty:
            fig_bt = go.Figure()

            who_bt = pd.DataFrame(who_obs)
            if not who_bt.empty:
                fig_bt.add_trace(
                    go.Scatter(
                        x=who_bt["date"],
                        y=who_bt["value"],
                        mode="lines+markers",
                        name="WHO observed",
                        line=dict(color="#1f77b4", width=2),
                        marker=dict(size=5),
                    )
                )
            fig_bt.add_trace(
                go.Scatter(
                    x=h1_df["date"],
                    y=h1_df["predicted"],
                    mode="lines+markers",
                    name="1-step nowcast (backtest)",
                    line=dict(color="#ff7f0e", width=2, dash="dash"),
                    marker=dict(size=5, symbol="diamond"),
                )
            )
            fig_bt.update_layout(
                xaxis_title="Month",
                yaxis_title="Dengue cases",
                height=300,
                margin=dict(l=40, r=20, t=20, b=40),
                plot_bgcolor="#fff",
                paper_bgcolor="#fff",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                xaxis=dict(showgrid=True, gridcolor="#eee"),
                yaxis=dict(showgrid=True, gridcolor="#eee"),
            )
            st.plotly_chart(fig_bt, use_container_width=True)

        # ── Per-split metrics table ──────────────────────────────────────────
        metrics_path = (
            Path(__file__).resolve().parent.parent
            / "outputs_updated_nowcast_log"
            / "rolling_splits"
            / "rolling_split_metrics.csv"
        )
        if metrics_path.exists():
            mdf = pd.read_csv(metrics_path)
            display_cols = [
                c for c in [
                    "split_month", "RMSE_test", "MAPE_test_%",
                    "RMSE_test_seasonal_naive", "MAPE_test_seasonal_naive_%",
                ]
                if c in mdf.columns
            ]
            st.dataframe(
                mdf[display_cols].round(2),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("rolling_split_metrics.csv not found.")
else:
    st.info("Rolling metrics not available.")

st.divider()

# ---------------------------------------------------------------------------
# ── Section 7: About
# ---------------------------------------------------------------------------
st.subheader("About")

col_a, col_b = st.columns(2)

with col_a:
    st.markdown("#### Data preprocessing")
    st.markdown(
        "We align and clean monthly WHO, Google Trends, Wikipedia, and mosquito "
        "proxy signals to build a consistent input dataset for dengue nowcasting. "
        "WHO case data is available from January 2024; proxy signals extend back to 2021."
    )

with col_b:
    st.markdown("#### Nowcast model")
    st.markdown(
        "The model uses historical dengue patterns and external signals to produce "
        "monthly nowcast estimates together with 95% confidence intervals. "
        "Validation uses a rolling time-series cross-validation scheme with a "
        "2-month test window; reported RMSE and MAPE are averages across rolling splits."
    )
