# university-of-Birmingham
I building porfolio
# app.py
"""
Streamlit app that follows formulas found in:
- 'sp500_weekly_Return' (returns = P_t / P_{t-1} - 1)
- 'covariance matrix' (VARP on diagonal => population variance/covariance)
- 'Optimal benchmark 30 stock (2)' (portfolio return, var/std, risk-free periodic, Sharpe)

How to run:
    pip install streamlit pandas numpy scipy openpyxl matplotlib
    streamlit run app.py
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import streamlit as st
from scipy.optimize import minimize
import matplotlib.pyplot as plt


FREQ_PERIODS_PER_YEAR = {
    "Daily": 252,
    "Weekly": 52,
    "Annual": 1,
}


@dataclass(frozen=True)
class PortfolioStats:
    expected_return: float
    variance: float
    std: float
    sharpe: float


def _safe_to_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _detect_date_column(df: pd.DataFrame) -> Optional[str]:
    candidates = [c for c in df.columns if str(c).strip().lower() in {"date", "datetime", "time"}]
    if candidates:
        return candidates[0]
    return None


def load_tabular(file_bytes: bytes, filename: str, sheet_name: Optional[str]) -> pd.DataFrame:
    ext = filename.lower().split(".")[-1]
    if ext == "csv":
        return pd.read_csv(io.BytesIO(file_bytes))
    if ext in {"xlsx", "xls"}:
        return pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet_name, engine="openpyxl")
    raise ValueError("Unsupported file type. Upload a CSV or Excel (.xlsx/.xls).")


def clean_and_extract_assets(df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    df = df_raw.copy()

    # Drop fully empty rows/cols
    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")

    date_col = _detect_date_column(df)
    date_series = None

    if date_col is not None:
        parsed = pd.to_datetime(df[date_col], errors="coerce")
        df = df.loc[parsed.notna()].copy()
        date_series = parsed.loc[parsed.notna()].reset_index(drop=True)
        df = df.drop(columns=[date_col]).reset_index(drop=True)

    # Keep only numeric-like asset columns
    df_assets = _safe_to_numeric_df(df)
    df_assets = df_assets.dropna(axis=1, how="all")

    return df_assets, date_series


def compute_simple_returns_from_prices(prices: pd.DataFrame) -> pd.DataFrame:
    # Matches Excel: P_t / P_{t-1} - 1
    rets = prices.pct_change()
    return rets.dropna(how="all")


def population_covariance(returns: pd.DataFrame) -> pd.DataFrame:
    # Matches Excel VARP (ddof=0) behavior implied by 'covariance matrix' diagonal.
    # cov = (X - mean)^T (X - mean) / N
    x = returns.to_numpy(dtype=float)
    x = x[~np.isnan(x).all(axis=1)]
    col_means = np.nanmean(x, axis=0)
    xc = x - col_means
    # Handle NaNs by pairwise masking
    n_assets = xc.shape[1]
    cov = np.zeros((n_assets, n_assets), dtype=float)
    for i in range(n_assets):
        for j in range(n_assets):
            mask = ~np.isnan(xc[:, i]) & ~np.isnan(xc[:, j])
            if mask.sum() == 0:
                cov[i, j] = np.nan
            else:
                cov[i, j] = (xc[mask, i] * xc[mask, j]).sum() / mask.sum()
    return pd.DataFrame(cov, index=returns.columns, columns=returns.columns)


def risk_free_periodic(rf_annual: float, periods_per_year: int) -> float:
    # Matches pattern in Excel: (1 + rf_annual)^(1/52) - 1
    return (1.0 + rf_annual) ** (1.0 / float(periods_per_year)) - 1.0


def portfolio_stats(weights: np.ndarray, mu: np.ndarray, cov: np.ndarray, rf_periodic: float) -> PortfolioStats:
    # Matches Excel:
    # Return: SUM(weight_i * mean_return_i)
    # Var: w^T Cov w
    # Std: sqrt(Var)
    # Sharpe: (Return - rf) / Std
    w = weights.reshape(-1, 1)
    exp_ret = float(weights @ mu)
    var = float((w.T @ cov @ w).item())
    std = float(np.sqrt(max(var, 0.0)))
    sharpe = float((exp_ret - rf_periodic) / std) if std > 0 else float("nan")
    return PortfolioStats(expected_return=exp_ret, variance=var, std=std, sharpe=sharpe)


def optimize_max_sharpe(mu: np.ndarray, cov: np.ndarray, rf_p: float) -> np.ndarray:
    n = len(mu)
    x0 = np.full(n, 1.0 / n, dtype=float)
    bounds = [(0.0, 1.0) for _ in range(n)]
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    def objective(w: np.ndarray) -> float:
        stats = portfolio_stats(w, mu, cov, rf_p)
        if np.isnan(stats.sharpe):
            return 1e9
        return -stats.sharpe

    res = minimize(objective, x0=x0, method="SLSQP", bounds=bounds, constraints=cons, options={"maxiter": 500})
    if not res.success:
        raise RuntimeError(f"Optimization failed (Max Sharpe): {res.message}")
    return res.x


def optimize_min_variance_target_return(mu: np.ndarray, cov: np.ndarray, target_return: float) -> np.ndarray:
    n = len(mu)
    x0 = np.full(n, 1.0 / n, dtype=float)
    bounds = [(0.0, 1.0) for _ in range(n)]
    cons = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        {"type": "eq", "fun": lambda w: float(w @ mu) - float(target_return)},
    ]

    def objective(w: np.ndarray) -> float:
        w = w.reshape(-1, 1)
        return float((w.T @ cov @ w).item())

    res = minimize(objective, x0=x0, method="SLSQP", bounds=bounds, constraints=cons, options={"maxiter": 700})
    if not res.success:
        raise RuntimeError(f"Optimization failed (Min Var @ Target Return): {res.message}")
    return res.x


def compute_efficient_frontier(mu: np.ndarray, cov: np.ndarray, n_points: int) -> Tuple[np.ndarray, np.ndarray]:
    # Mirrors the Excel note: for each specific return -> minimize std by changing weights.
    mu_min, mu_max = float(np.min(mu)), float(np.max(mu))
    targets = np.linspace(mu_min, mu_max, n_points)
    rets: List[float] = []
    risks: List[float] = []
    for t in targets:
        try:
            w = optimize_min_variance_target_return(mu, cov, t)
            s = portfolio_stats(w, mu, cov, rf_periodic=0.0)
            rets.append(s.expected_return)
            risks.append(s.std)
        except Exception:
            # Skip infeasible targets under [0,1] bounds
            continue
    return np.array(rets, dtype=float), np.array(risks, dtype=float)


def main() -> None:
    st.set_page_config(page_title="Optimal Portfolio (Excel-formula based)", layout="wide")
    st.title("Optimal Portfolio Optimizer (matches your attached Excel formulas)")

    with st.sidebar:
        st.header("1) Upload data")
        file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])
        data_kind = st.radio("Input data type", ["Prices", "Returns"], horizontal=True)
        freq = st.selectbox("Data frequency", list(FREQ_PERIODS_PER_YEAR.keys()), index=1)
        periods = FREQ_PERIODS_PER_YEAR[freq]

        st.header("2) Risk-free rate")
        rf_annual = st.number_input(
            "Risk-free (annual), decimal",
            min_value=0.0,
            max_value=1.0,
            value=0.0365,  # matches the Excel sheet default shown
            step=0.0005,
            help="Excel uses this and converts to weekly: (1+rf_annual)^(1/52)-1. We generalize by frequency.",
        )
        rf_p = risk_free_periodic(float(rf_annual), periods)

        st.header("3) Optimization")
        objective = st.selectbox(
            "Objective",
            [
                "Max Sharpe (Optimal portfolio)",
                "Min Variance for Target Return (Solver frontier style)",
            ],
        )

        target_return = None
        if objective == "Min Variance for Target Return (Solver frontier style)":
            target_return = st.number_input(
                f"Target return per {freq.lower()} period",
                value=0.0020 if periods == 52 else 0.0,
                step=0.0001,
                help="This is the same kind of 'specific return' constraint mentioned in your Excel note.",
            )

        st.header("4) Cleaning")
        max_missing_frac = st.slider("Drop asset column if missing fraction >", 0.0, 1.0, 0.2, 0.05)
        fill_missing = st.checkbox("Forward-fill remaining missing values", value=True)

        st.header("5) Optional outputs")
        show_frontier = st.checkbox("Plot efficient frontier (min variance for many target returns)", value=True)
        frontier_points = st.slider("Frontier points", 10, 80, 30, 5)

    if not file:
        st.info("Upload a dataset to begin. Columns should be assets; a 'Date' column is optional.")
        return

    # Load
    sheet_name = None
    if file.name.lower().endswith((".xlsx", ".xls")):
        try:
            # Let user select sheet
            import openpyxl  # noqa: F401

            xl = pd.ExcelFile(file)
            sheets = xl.sheet_names
            sheet_name = st.sidebar.selectbox("Excel sheet", sheets, index=0)
        except Exception:
            sheet_name = None

    df_raw = load_tabular(file.getvalue(), file.name, sheet_name)
    assets_df, date_series = clean_and_extract_assets(df_raw)

    if assets_df.shape[1] < 2:
        st.error("Need at least 2 asset columns with numeric data.")
        return

    # Drop columns with too many missing
    miss_frac = assets_df.isna().mean(axis=0)
    keep_cols = miss_frac[miss_frac <= max_missing_frac].index.tolist()
    assets_df = assets_df[keep_cols].copy()

    if assets_df.shape[1] < 2:
        st.error("After dropping missing-heavy columns, fewer than 2 assets remain.")
        return

    # Fill remaining missing if requested
    if fill_missing:
        assets_df = assets_df.ffill()

    assets_df = assets_df.dropna(axis=0, how="any")
    if assets_df.shape[0] < 3:
        st.error("Not enough rows after cleaning (need at least 3).")
        return

    # Select subset
    selected_assets = st.multiselect("Select assets to include", list(assets_df.columns), default=list(assets_df.columns))
    assets_df = assets_df[selected_assets].copy()

    # Compute returns
    if data_kind == "Prices":
        returns_df = compute_simple_returns_from_prices(assets_df)
    else:
        returns_df = assets_df.copy()

    returns_df = returns_df.dropna(axis=0, how="any")
    if returns_df.shape[0] < 3:
        st.error("Not enough return rows after cleaning (need at least 3).")
        return

    # Core stats (Excel-style)
    mu_s = returns_df.mean(axis=0)  # AVERAGE
    cov_df = population_covariance(returns_df)  # VARP/COVP style
    mu = mu_s.to_numpy(dtype=float)
    cov = cov_df.to_numpy(dtype=float)

    # Optimize
    try:
        if objective == "Max Sharpe (Optimal portfolio)":
            w = optimize_max_sharpe(mu, cov, rf_p)
        else:
            w = optimize_min_variance_target_return(mu, cov, float(target_return))
    except Exception as e:
        st.error(str(e))
        return

    stats = portfolio_stats(w, mu, cov, rf_p)

    # Display
    left, right = st.columns([1.1, 0.9], gap="large")

    with left:
        st.subheader("Weights")
        weights_df = pd.DataFrame(
            {
                "Asset": mu_s.index,
                "Weight": w,
                "Mean Return (per period)": mu,
                "Weight * Mean Return": w * mu,
            }
        ).sort_values("Weight", ascending=False)

        st.dataframe(weights_df, use_container_width=True)

        st.caption(
            "Matches Excel layout: WEIGHT, Return (mean), w*return, and portfolio Return = SUM(w*mean_return)."
        )

    with right:
        st.subheader("Portfolio metrics (per input period)")
        c1, c2 = st.columns(2)
        c1.metric("Expected return", f"{stats.expected_return:.6f}")
        c2.metric("Std (risk)", f"{stats.std:.6f}")
        c1.metric("Variance", f"{stats.variance:.8f}")
        c2.metric("Sharpe (SLOPE)", f"{stats.sharpe:.4f}")
        st.write(f"Risk-free rate per {freq.lower()} period: **{rf_p:.6f}**")

        # Portfolio return series chart (same SUMPRODUCT idea applied per-row)
        st.subheader("Portfolio return series (optional view)")
        port_rets = returns_df.to_numpy(dtype=float) @ w
        cum = (1.0 + pd.Series(port_rets)).cumprod()

        fig = plt.figure()
        plt.plot(cum.values)
        plt.xlabel("Time step")
        plt.ylabel("Cumulative growth (start=1)")
        st.pyplot(fig, clear_figure=True)

    if show_frontier:
        st.subheader("Efficient frontier (Solver-style: min variance for each target return)")
        frontier_rets, frontier_risks = compute_efficient_frontier(mu, cov, n_points=int(frontier_points))

        fig2 = plt.figure()
        if len(frontier_rets) > 0:
            plt.plot(frontier_risks, frontier_rets, marker="o", linestyle="-")
            plt.scatter([stats.std], [stats.expected_return], marker="x")
            plt.xlabel("Risk (Std)")
            plt.ylabel("Return")
        else:
            plt.text(0.5, 0.5, "No feasible frontier points under 0..1 weights.", ha="center", va="center")
            plt.axis("off")
        st.pyplot(fig2, clear_figure=True)

    # Downloads
    st.subheader("Download")
    out_csv = weights_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download weights CSV", data=out_csv, file_name="optimal_portfolio_weights.csv", mime="text/csv")


if __name__ == "__main__":
    main()
