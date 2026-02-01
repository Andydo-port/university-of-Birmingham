import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import plotly.graph_objects as go

# Professional Page Config
st.set_page_config(page_title="OptiPortfolio Expert | Andy Do", layout="wide")
st.title("📈 OptiPortfolio Expert")
st.markdown("### Strategic Asset Allocation & Mean-Variance Optimization")
st.write("Developed by: **Andy Do** (MSc Investment, University of Birmingham)")

# --- SIDEBAR: PARAMETERS ---
st.sidebar.header("⚙️ Optimization Settings")
freq = st.sidebar.selectbox("Data Frequency", ['Daily', 'Weekly', 'Monthly', 'Yearly'])
rf_rate = st.sidebar.number_input("Annual Risk-Free Rate (decimal)", value=0.04, step=0.001)

st.sidebar.subheader("Weight Constraints")
min_w = st.sidebar.slider("Minimum Weight per Asset (%)", 0, 100, 0) / 100
max_w = st.sidebar.slider("Maximum Weight per Asset (%)", 0, 100, 100) / 100

# --- DATA UPLOAD ---
st.header("1. Data Input")
uploaded_file = st.file_uploader("Upload Historical Returns/Prices (CSV or XLSX)", type=["xlsx", "csv"])

if uploaded_file is not None:
    # Load Data
    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
    st.success("Data loaded successfully!")
    
    # Process numeric columns only
    returns = df.select_dtypes(include=[np.number])
    tickers = returns.columns.tolist()
    n_assets = len(tickers)

    # Annualization Multiplier
    freq_map = {'Daily': 252, 'Weekly': 52, 'Monthly': 12, 'Yearly': 1}
    adj = freq_map[freq]
    
    # Calculate Annualized Stats
    ann_rets = returns.mean() * adj
    ann_cov = returns.cov() * adj

    # --- OPTIMIZATION LOGIC ---
    def get_portfolio_metrics(weights):
        p_ret = np.sum(ann_rets * weights)
        p_vol = np.sqrt(np.dot(weights.T, np.dot(ann_cov, weights)))
        return p_ret, p_vol

    def negative_sharpe(weights):
        p_ret, p_vol = get_portfolio_metrics(weights)
        return -(p_ret - rf_rate) / p_vol

    # Constraints: Sum of weights = 1
    cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((min_w, max_w) for _ in range(n_assets))
    init_guess = n_assets * [1. / n_assets]

    # Solver
    res = minimize(negative_sharpe, init_guess, method='SLSQP', bounds=bounds, constraints=cons)
    
    if res.success:
        opt_weights = res.x
        opt_ret, opt_vol = get_portfolio_metrics(opt_weights)
        sharpe = (opt_ret - rf_rate) / opt_vol

        # --- DISPLAY RESULTS ---
        st.header("2. Optimization Output (Annualized)")
        m1, m2, m3 = st.columns(3)
        m1.metric("Expected Return", f"{opt_ret:.2%}")
        m2.metric("Portfolio Volatility", f"{opt_vol:.2%}")
        m3.metric("Sharpe Ratio", f"{sharpe:.4f}")

        # Bar Chart for Weights
        st.subheader("Optimal Portfolio Composition")
        w_series = pd.Series(opt_weights, index=tickers).sort_values(ascending=False)
        st.bar_chart(w_series)

        # Asset Detail Table
        st.subheader("Asset Breakdown")
        detail_df = pd.DataFrame({
            "Asset": tickers,
            "Annualized Return": ann_rets.values,
            "Optimal Weight": opt_weights
        }).sort_values(by="Optimal Weight", ascending=False)
        st.table(detail_df.style.format({"Annualized Return": "{:.2%}", "Optimal Weight": "{:.2%}"}))
    else:
        st.error("Optimization failed. Please check your constraints or data quality.")
else:
    st.info("Awaiting historical data file to begin analysis.")
