import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import minimize

# Professional Page Config
st.set_page_config(page_title="OptiPortfolio Expert | Andy Do", layout="wide")
st.title("📈 OptiPortfolio Expert")
st.markdown("### Strategic Asset Allocation & Individual Asset Constraints")
st.write("Developed by: **Andy Do** (MSc Investment, University of Birmingham)")

# --- SIDEBAR: GLOBAL SETTINGS ---
st.sidebar.header("⚙️ Global Settings")
freq = st.sidebar.selectbox("Data Frequency", ['Daily', 'Weekly', 'Monthly', 'Yearly'])
rf_rate = st.sidebar.number_input("Annual Risk-Free Rate (decimal)", value=0.04, step=0.001)

# --- DATA UPLOAD ---
st.header("1. Data Input")
uploaded_file = st.file_uploader("Upload Historical Returns/Prices (CSV or XLSX)", type=["xlsx", "csv"])

if uploaded_file is not None:
    # Load Data
    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
    st.success("Data loaded successfully!")
    
    all_numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # --- ASSET SELECTION ---
    st.subheader("Asset Selection & Individual Constraints")
    selected_tickers = st.multiselect(
        "Select tickers to include in optimization:", 
        options=all_numeric_cols, 
        default=all_numeric_cols[:5] if len(all_numeric_cols) > 5 else all_numeric_cols
    )

    if len(selected_tickers) < 2:
        st.warning("Please select at least 2 assets to perform optimization.")
    else:
        # --- NEW: INDIVIDUAL WEIGHT CONSTRAINTS ---
        st.write("Set Min/Max weight for each selected asset (0.0 to 1.0):")
        individual_constraints = {}
        cols = st.columns(len(selected_tickers))
        
        for i, ticker in enumerate(selected_tickers):
            with cols[i]:
                st.markdown(f"**{ticker}**")
                min_val = st.number_input(f"Min", key=f"min_{ticker}", min_value=0.0, max_value=1.0, value=0.0, step=0.05)
                max_val = st.number_input(f"Max", key=f"max_{ticker}", min_value=0.0, max_value=1.0, value=1.0, step=0.05)
                individual_constraints[ticker] = (min_val, max_val)

        # Annualization Multiplier
        freq_map = {'Daily': 252, 'Weekly': 52, 'Monthly': 12, 'Yearly': 1}
        adj = freq_map[freq]
        
        returns = df[selected_tickers]
        ann_rets = returns.mean() * adj
        ann_cov = returns.cov() * adj
        n_assets = len(selected_tickers)

        # --- OPTIMIZATION LOGIC ---
        def get_portfolio_metrics(weights):
            p_ret = np.sum(ann_rets * weights)
            p_vol = np.sqrt(np.dot(weights.T, np.dot(ann_cov, weights)))
            return p_ret, p_vol

        def negative_sharpe(weights):
            p_ret, p_vol = get_portfolio_metrics(weights)
            return -(p_ret - rf_rate) / p_vol if p_vol > 0 else 0

        # Constraints: Sum of weights = 1
        cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        
        # Applying Individual Bounds
        bounds = [individual_constraints[t] for t in selected_tickers]
        init_guess = n_assets * [1. / n_assets]

        # Solver
        res = minimize(negative_sharpe, init_guess, method='SLSQP', bounds=bounds, constraints=cons)
        
        if res.success:
            opt_weights = res.x
            opt_ret, opt_vol = get_portfolio_metrics(opt_weights)
            sharpe = (opt_ret - rf_rate) / opt_vol

            # --- DISPLAY RESULTS ---
            st.header("2. Optimization Output")
            m1, m2, m3 = st.columns(3)
            m1.metric("Expected Return", f"{opt_ret:.2%}")
            m2.metric("Portfolio Volatility", f"{opt_vol:.2%}")
            m3.metric("Sharpe Ratio", f"{sharpe:.4f}")

            # Bar Chart for Weights
            st.subheader("Optimal Portfolio Composition")
            w_series = pd.Series(opt_weights, index=selected_tickers).sort_values(ascending=False)
            st.bar_chart(w_series)
            
            # Asset Table
            detail_df = pd.DataFrame({
                "Asset": selected_tickers,
                "Min Constraint": [b[0] for b in bounds],
                "Max Constraint": [b[1] for b in bounds],
                "Optimal Weight": opt_weights
            }).sort_values(by="Optimal Weight", ascending=False)
            st.table(detail_df.style.format({"Min Constraint": "{:.1%}", "Max Constraint": "{:.1%}", "Optimal Weight": "{:.2%}"}))
        else:
            st.error("Optimization failed. Constraints might be mathematically impossible (e.g., sum of Min weights > 100%).")
else:
    st.info("Awaiting historical data file to begin analysis.")
    
