from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt


# (1) Configuration
@dataclass
class Config:
    data_path: str = "master_data.csv"
    outdir: str = "outputs_step1_wiki"

    # Modeling window (inclusive)
    start_month: str = "2021-01"  # YYYY-MM

    # Source labels in master_data.csv
    who_monthly_source: str = "WHO"
    google_fever_source: str = "Google_Trends_Dengue_fever"
    google_vaccine_source: str = "Google_Trends_Dengue_vaccine"

    # Optional external regressor (Wikipedia monthly pageviews; already aggregated across Indian languages)
    use_wiki: bool = True
    wiki_path: str = "total_dengue_views.csv"
    wiki_month_col: str = "Month"
    wiki_value_col: str = "Total_Views"
    wiki_transform: str = "log1p"  # "log1p" or "none"

    # OpenDengue yearly proxy sources (use whatever exists; will pick per-year by priority)
    yearly_proxy_sources_priority: Tuple[str, ...] = (
        "OpenDengue_State_Aggregated",
        "OpenDengue_National_Yearly",
    )

    # Loss weights
    lambda_who: float = 1.0
    lambda_year: float = 1.0
    lambda_reg: float = 1e-3

    # Optimization
    epochs: int = 5000
    lr: float = 1e-2
    seed: int = 42

    # Scaling to stabilize optimization (cases are large; Google Trends are 0-100)
    target_scale: float = 1000.0  # train on (cases / target_scale), then multiply back

    # If True, clip final predictions at 0 before saving/plotting
    clip_nonnegative: bool = True


# (2) Data loading & shaping
def parse_monthly_date(s: pd.Series) -> pd.DatetimeIndex:
    # monthly dates stored as "YYYY-MM"
    return pd.to_datetime(s.astype(str) + "-01", format="%Y-%m-%d", errors="coerce")


def load_master_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    expected = {"resolution", "date", "value", "source"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"master_data.csv missing columns: {sorted(missing)}")
    return df


def build_monthly_wide(df: pd.DataFrame) -> pd.DataFrame:
    m = df[df["resolution"].eq("Monthly")].copy()
    m["date_dt"] = parse_monthly_date(m["date"])
    m = m.dropna(subset=["date_dt"])
    wide = (
        m.pivot_table(index="date_dt", columns="source", values="value", aggfunc="mean")
        .sort_index()
        .reset_index()
        .rename(columns={"date_dt": "date"})
    )
    return wide


def build_yearly_proxy(df: pd.DataFrame, priority_sources: Tuple[str, ...]) -> pd.DataFrame:
    y = df[df["resolution"].eq("Yearly")].copy()
    y["year"] = pd.to_numeric(y["date"], errors="coerce").astype("Int64")
    y = y.dropna(subset=["year"])
    y["year"] = y["year"].astype(int)

    pivot = (
        y.pivot_table(index="year", columns="source", values="value", aggfunc="mean")
        .sort_index()
    )

    # For each year, choose the first available proxy by priority
    chosen = []
    for year, row in pivot.iterrows():
        val = np.nan
        src = None
        for s in priority_sources:
            if s in row.index and pd.notna(row[s]):
                val = float(row[s])
                src = s
                break
        if pd.notna(val):
            chosen.append((year, val, src))

    return pd.DataFrame(chosen, columns=["year", "od_total", "od_source"])


def load_wiki_monthly(
    path: str,
    month_col: str = "Month",
    value_col: str = "Total_Views",
) -> pd.Series:
    """
    Load monthly Wikipedia pageviews time series.
    Expected columns: Month (YYYY-MM), Total_Views (numeric)
    Returns: pd.Series indexed by month (Timestamp at month start).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Wiki monthly file not found: {path}")

    w = pd.read_csv(p)
    if month_col not in w.columns or value_col not in w.columns:
        raise ValueError(
            f"Wiki file must contain columns '{month_col}' and '{value_col}'. "
            f"Got: {list(w.columns)}"
        )

    w = w.copy()
    w["date"] = parse_monthly_date(w[month_col])
    w = w.dropna(subset=["date"]).sort_values("date")
    s = pd.to_numeric(w[value_col], errors="coerce")
    ser = pd.Series(s.values, index=w["date"]).groupby(level=0).mean()
    return ser


# (3) Feature engineering
def build_design_matrix(
    monthly_wide: pd.DataFrame,
    cfg: Config,
):
    df = monthly_wide.copy()

    start_dt = pd.to_datetime(cfg.start_month + "-01")
    df = df[df["date"] >= start_dt].copy()
    df = df.sort_values("date").reset_index(drop=True)

    # Ensure required columns exist (Google Trends should exist for the window)
    for col in [cfg.google_fever_source, cfg.google_vaccine_source]:
        if col not in df.columns:
            raise ValueError(f"Missing monthly source column '{col}' in monthly-wide table.")

    # Extract raw predictors
    df["g_fever"] = pd.to_numeric(df[cfg.google_fever_source], errors="coerce")
    df["g_vaccine"] = pd.to_numeric(df[cfg.google_vaccine_source], errors="coerce")

    # Light imputation
    for col in ["g_fever", "g_vaccine"]:
        df[col] = df[col].interpolate(limit_direction="both").ffill().bfill()

    # Wiki regressor (monthly)
    if cfg.use_wiki:
        wiki_ser = load_wiki_monthly(cfg.wiki_path, cfg.wiki_month_col, cfg.wiki_value_col)
        df["wiki_raw"] = wiki_ser.reindex(df["date"]).values
        df["wiki_raw"] = pd.to_numeric(df["wiki_raw"], errors="coerce")
        df["wiki_raw"] = df["wiki_raw"].interpolate(limit_direction="both").ffill().bfill()

        if cfg.wiki_transform.lower() == "log1p":
            df["wiki_views"] = np.log1p(df["wiki_raw"].astype(float))
        elif cfg.wiki_transform.lower() == "none":
            df["wiki_views"] = df["wiki_raw"].astype(float)
        else:
            raise ValueError("cfg.wiki_transform must be 'log1p' or 'none'.")
    else:
        df["wiki_raw"] = np.nan
        df["wiki_views"] = np.nan

    # Month-of-year indicators (drop_first to avoid collinearity with intercept)
    df["month"] = df["date"].dt.month.astype(int)
    month_dummies = pd.get_dummies(df["month"], prefix="m", drop_first=True)
    df = pd.concat([df, month_dummies], axis=1)

    # Target (WHO monthly cases) — may be missing for many months
    if cfg.who_monthly_source not in df.columns:
        df[cfg.who_monthly_source] = np.nan
    df["y_who"] = pd.to_numeric(df[cfg.who_monthly_source], errors="coerce")

    # Build X with intercept + Google (+ wiki) + month dummies
    df["intercept"] = 1.0
    feature_cols = ["intercept", "g_fever", "g_vaccine"]
    if cfg.use_wiki:
        feature_cols += ["wiki_views"]
    feature_cols += list(month_dummies.columns)

    X = df[feature_cols].to_numpy(dtype=np.float32)
    y_who = df["y_who"].to_numpy(dtype=np.float32)
    mask_who = np.isfinite(y_who)

    return df, X, y_who, mask_who, feature_cols


def build_year_constraints(
    dates: pd.Series,
    yearly_proxy: pd.DataFrame,
) -> List[Tuple[int, np.ndarray, float, str]]:
    """
    Build constraints for years where:
      - OpenDengue yearly total exists
      - We have all 12 months present in our monthly date index

    Returns list of tuples: (year, idx_array, od_total, od_source)
    """
    df_dates = pd.DataFrame({"date": pd.to_datetime(dates)})
    df_dates["year"] = df_dates["date"].dt.year
    df_dates["month"] = df_dates["date"].dt.month

    constraints = []
    for _, r in yearly_proxy.iterrows():
        y = int(r["year"])
        od_total = float(r["od_total"])
        od_src = str(r["od_source"])

        idx = df_dates.index[df_dates["year"].eq(y)].to_numpy()
        if len(idx) == 0:
            continue

        # require full 12 months to match a full-year total
        months_present = set(df_dates.loc[idx, "month"].tolist())
        if months_present != set(range(1, 13)):
            continue

        constraints.append((y, idx, od_total, od_src))

    return constraints


# (4) Model training
def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_step1(
    X: np.ndarray,
    y_who: np.ndarray,
    mask_who: np.ndarray,
    year_constraints: List[Tuple[int, np.ndarray, float, str]],
    cfg: Config,
):
    device = torch.device("cpu")

    X_t = torch.tensor(X, dtype=torch.float32, device=device)  # [n, p]
    n, p = X_t.shape

    # Scale targets for stable optimization
    y_who_scaled = y_who / cfg.target_scale
    y_who_t = torch.tensor(y_who_scaled, dtype=torch.float32, device=device)
    mask_who_t = torch.tensor(mask_who, dtype=torch.bool, device=device)

    # Year constraints: pre-scale totals and store indices
    year_terms = []
    for (year, idx, od_total, od_src) in year_constraints:
        year_terms.append(
            (year, torch.tensor(idx, dtype=torch.long, device=device), float(od_total / cfg.target_scale), od_src)
        )

    beta = torch.nn.Parameter(torch.zeros(p, dtype=torch.float32, device=device))
    opt = torch.optim.Adam([beta], lr=cfg.lr)

    rows = []
    for epoch in range(1, cfg.epochs + 1):
        opt.zero_grad(set_to_none=True)

        x = X_t @ beta  # [n]

        # WHO monthly loss (mean squared error over available months)
        if mask_who_t.any():
            diff_who = x[mask_who_t] - y_who_t[mask_who_t]
            L_who = (diff_who ** 2).mean()
        else:
            L_who = torch.tensor(0.0, device=device)

        # Yearly proxy loss (mean over years)
        if len(year_terms) > 0:
            diffs = []
            for _, idx_t, od_total_scaled, _ in year_terms:
                year_sum = x.index_select(0, idx_t).sum()
                diffs.append((year_sum - od_total_scaled) ** 2)
            L_year = torch.stack(diffs).mean()
        else:
            L_year = torch.tensor(0.0, device=device)

        # L2 regularization
        L_reg = (beta ** 2).sum()

        L_total = cfg.lambda_who * L_who + cfg.lambda_year * L_year + cfg.lambda_reg * L_reg
        L_total.backward()
        opt.step()

        rows.append(
            {
                "epoch": epoch,
                "L_total": float(L_total.detach().cpu().item()),
                "L_who": float(L_who.detach().cpu().item()),
                "L_year": float(L_year.detach().cpu().item()),
                "L_reg": float(L_reg.detach().cpu().item()),
            }
        )

        # basic stability guard (stop if NaN)
        if not np.isfinite(rows[-1]["L_total"]):
            raise RuntimeError("Training diverged (loss is NaN/Inf). Try lowering lr or increasing target_scale.")

    beta_hat = beta.detach().cpu().numpy()
    loss_df = pd.DataFrame(rows)
    return beta_hat, loss_df


# (5) Outputs
def save_params(outdir: Path, feature_cols: List[str], beta_hat: np.ndarray, cfg: Config) -> None:
    params = {
        "config": asdict(cfg),
        "features": feature_cols,
        "beta": {name: float(val) for name, val in zip(feature_cols, beta_hat)},
        "target_scale": cfg.target_scale,
    }
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "params_step1.json").write_text(json.dumps(params, indent=2), encoding="utf-8")


def make_predictions(design_df: pd.DataFrame, X: np.ndarray, beta_hat: np.ndarray, cfg: Config) -> pd.DataFrame:
    x_scaled = X @ beta_hat
    x = x_scaled * cfg.target_scale
    if cfg.clip_nonnegative:
        x = np.maximum(x, 0.0)

    out = pd.DataFrame(
        {
            "date": design_df["date"].dt.strftime("%Y-%m"),
            "year": design_df["date"].dt.year.astype(int),
            "month": design_df["date"].dt.month.astype(int),
            "x_pred": x.astype(float),
            "g_fever": design_df["g_fever"].astype(float).values,
            "g_vaccine": design_df["g_vaccine"].astype(float).values,
            "wiki_raw": design_df["wiki_raw"].astype(float).values if "wiki_raw" in design_df.columns else np.nan,
            "wiki_views": design_df["wiki_views"].astype(float).values if "wiki_views" in design_df.columns else np.nan,
            "y_who": design_df["y_who"].astype(float).values,
        }
    )
    return out


def plot_loss_curve(loss_df: pd.DataFrame, outdir: Path) -> None:
    fig = plt.figure()
    plt.plot(loss_df["epoch"], loss_df["L_total"], label="Total")
    if (loss_df["L_who"] != 0).any():
        plt.plot(loss_df["epoch"], loss_df["L_who"], label="WHO")
    if (loss_df["L_year"] != 0).any():
        plt.plot(loss_df["epoch"], loss_df["L_year"], label="Yearly")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (log scale)")
    plt.title("Training Loss Curve (Step 1)")
    plt.legend()
    plt.tight_layout()
    fig.savefig(outdir / "loss_curve_step1.png", dpi=200)
    plt.close(fig)


def plot_who_vs_pred(pred_df: pd.DataFrame, outdir: Path) -> None:
    df = pred_df.dropna(subset=["y_who"]).copy()
    if df.empty:
        return

    fig = plt.figure()
    dates = pd.to_datetime(df["date"] + "-01")
    plt.plot(dates, df["x_pred"], label="Predicted")
    plt.plot(dates, df["y_who"], label="WHO observed")
    plt.xlabel("Date")
    plt.ylabel("Monthly dengue cases")
    plt.title("WHO vs Prediction (Step 1)")
    plt.legend()
    plt.tight_layout()
    fig.savefig(outdir / "who_vs_pred_step1.png", dpi=200)
    plt.close(fig)


def plot_yearly_vs_od(
    pred_df: pd.DataFrame,
    yearly_proxy: pd.DataFrame,
    outdir: Path,
) -> None:
    pred_year = pred_df.groupby("year", as_index=False)["x_pred"].sum().rename(columns={"x_pred": "pred_year_total"})
    od = yearly_proxy.copy()
    if od.empty:
        return

    merged = pred_year.merge(od, on="year", how="inner").sort_values("year")

    fig = plt.figure()
    plt.plot(merged["year"], merged["pred_year_total"], marker="o", label="Predicted yearly sum")
    plt.plot(merged["year"], merged["od_total"], marker="o", label="OpenDengue yearly total")
    plt.xlabel("Year")
    plt.ylabel("Total dengue cases (year)")
    plt.title("Yearly Aggregation: Prediction vs OpenDengue (Step 1)")
    plt.legend()
    plt.tight_layout()
    fig.savefig(outdir / "yearly_vs_opendengue_step1.png", dpi=200)
    plt.close(fig)

    merged.to_csv(outdir / "yearly_comparison_step1.csv", index=False)


# (6) Main
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", type=str, default=Config.data_path, help="Path to master_data.csv")
    parser.add_argument("--outdir", type=str, default=Config.outdir, help="Output directory")
    parser.add_argument("--start", type=str, default=Config.start_month, help="Start month YYYY-MM (inclusive)")

    # Wiki
    parser.add_argument("--use_wiki", type=int, default=1, help="1 to include wiki regressor; 0 to disable")
    parser.add_argument("--wiki_path", type=str, default=Config.wiki_path, help="Path to total_dengue_views.csv")
    parser.add_argument("--wiki_transform", type=str, default=Config.wiki_transform, choices=["log1p", "none"])

    parser.add_argument("--lambda_who", type=float, default=Config.lambda_who)
    parser.add_argument("--lambda_year", type=float, default=Config.lambda_year)
    parser.add_argument("--lambda_reg", type=float, default=Config.lambda_reg)

    parser.add_argument("--epochs", type=int, default=Config.epochs)
    parser.add_argument("--lr", type=float, default=Config.lr)
    parser.add_argument("--target_scale", type=float, default=Config.target_scale)
    parser.add_argument("--seed", type=int, default=Config.seed)

    args = parser.parse_args()

    cfg = Config(
        data_path=args.data,
        outdir=args.outdir,
        start_month=args.start,
        use_wiki=bool(args.use_wiki),
        wiki_path=args.wiki_path,
        wiki_transform=args.wiki_transform,
        lambda_who=args.lambda_who,
        lambda_year=args.lambda_year,
        lambda_reg=args.lambda_reg,
        epochs=args.epochs,
        lr=args.lr,
        target_scale=args.target_scale,
        seed=args.seed,
    )

    set_seed(cfg.seed)
    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load
    df = load_master_csv(cfg.data_path)
    monthly_wide = build_monthly_wide(df)
    yearly_proxy = build_yearly_proxy(df, cfg.yearly_proxy_sources_priority)

    # Build design matrix & constraints
    design_df, X, y_who, mask_who, feature_cols = build_design_matrix(monthly_wide, cfg)
    year_constraints = build_year_constraints(design_df["date"], yearly_proxy)

    # Train
    beta_hat, loss_df = train_step1(X, y_who, mask_who, year_constraints, cfg)

    # Save artifacts
    save_params(outdir, feature_cols, beta_hat, cfg)
    loss_df.to_csv(outdir / "loss_step1.csv", index=False)
    plot_loss_curve(loss_df, outdir)

    pred_df = make_predictions(design_df, X, beta_hat, cfg)
    pred_df.to_csv(outdir / "predictions_step1_monthly.csv", index=False)

    plot_who_vs_pred(pred_df, outdir)
    plot_yearly_vs_od(pred_df, yearly_proxy, outdir)

    # quick console summary
    print("Saved outputs to:", outdir.resolve())
    print("Learned coefficients (beta):")
    for name, val in zip(feature_cols, beta_hat):
        print(f"  {name:>15s}: {val:.6f}")


if __name__ == "__main__":
    main()
