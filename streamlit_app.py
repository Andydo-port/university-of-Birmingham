import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import minimize

# Professional Setup for Andy Do - MSc Investment
st.set_page_config(page_title="OptiPortfolio Pro | Andy Do", layout="wide")
st.title("📈 OptiPortfolio Pro: Multi-Period Optimizer")
st.write(f"Candidate: **Andy Do** | ID: 2926461 | University of Birmingham")

# --- STEP 1: INITIAL SETTINGS ---
st.sidebar.header("1. Frequency & Rate")
# User selects frequency BEFORE data processing
freq_choice = st.sidebar.selectbox(
    "Select Data Frequency", 
    ['Daily', 'Weekly', 'Monthly', 'Yearly'], 
    index=1  # Default to Weekly
)
freq_map = {'Daily': 252, 'Weekly': 52, 'Monthly': 12, 'Yearly': 1}
multiplier = freq_map[freq_choice]

rf_annual = st.sidebar.number_input("Annual Risk-Free Rate (e.g. 0.04)", value=0.04, step=0.001)
# Convert RF to periodic for solver consistency
rf_periodic = rf_annual / multiplier

# --- STEP 2: DATA INPUT ---
st.header("2. Data Upload")
uploaded_file = st.file_uploader("Upload your Prices file (Excel/CSV)", type=["xlsx", "csv"])

if uploaded_file:
    df = pd.read_excel(uploaded_file) if uploaded_file.name.endswith('.xlsx') else pd.read_csv(uploaded_file)
    
    # Extract numeric data (Prices)
    prices = df.select_dtypes(include=[np.number])
    
    # CALCULATION: Arithmetic Periodic Returns
    returns = prices.pct_change().dropna()
    
    st.subheader(f"Asset Universe ({freq_choice})")
    selected = st.multiselect("Select Tickers", returns.columns.tolist(), default=returns.columns.tolist()[:5])

    if len(selected) >= 2:
        # STEP 3: INDIVIDUAL CONSTRAINTS
        st.write("### Individual Weight Constraints")
        ind_cons = {}
        c_cols = st.columns(len(selected))
        for i, t in enumerate(selected):
            with c_cols[i]:
                min_v = st.number_input(f"{t} Min", 0.0, 1.0, 0.0, key=f"min_{t}")
                max_v = st.number_input(f"{t} Max", 0.0, 1.0, 1.0, key=f"max_{t}")
                ind_cons[t] = (min_v, max_v)

        if sum([v[0] for v in ind_cons.values()]) > 1:
            st.error("Total Minimum Weights exceed 100%. Please adjust.")
        else:
            # STEP 4: PERIODIC STATS (Weekly/Daily/etc.)
            mu_periodic = returns[selected].mean()
            cov_periodic = returns[selected].cov()

            def get_port_periodic_stats(w):
                p_ret = np.sum(mu_periodic * w)
                p_vol = np.sqrt(np.dot(w.T, np.dot(cov_periodic, w)))
                return p_ret, p_vol

            # Optimization Goal: Maximize Periodic Sharpe Ratio
            def objective(w):
                r, v = get_port_periodic_stats(w)
                # Sharpe using periodic RF
                return -(r - rf_periodic) / v if v > 0 else 0

            # Constraints: Sum = 1
            constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0})
            bounds = [ind_cons[t] for t in selected]
            init_w = np.array([1.0/len(selected)] * len(selected))

            res = minimize(objective, init_w, method='SLSQP', bounds=bounds, constraints=constraints)

            if res.success:
                opt_w = res.x
                p_ret_per, p_vol_per = get_port_periodic_stats(opt_w)
                
                # STEP 5: DISPLAY PERIODIC & ANNUAL RESULTS
                st.header("3. Optimization Results")
                
                # Metric display
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader(f"Periodic Stats ({freq_choice})")
                    st.metric("Expected Return", f"{p_ret_per:.4%}")
                    st.metric("Volatility (Risk)", f"{p_vol_per:.4%}")
                
                with col2:
                    st.subheader("Annualized Stats")
                    st.metric("Annual Return", f"{(p_ret_per * multiplier):.2%}")
                    st.metric("Annual Risk", f"{(p_vol_per * np.sqrt(multiplier)):.2%}")
                
                st.write(f"**Final Portfolio Sharpe Ratio:** {((p_ret_per * multiplier - rf_annual) / (p_vol_per * np.sqrt(multiplier))):.4f}")
                
                # Weight Visualization
                st.subheader("Optimal Allocations")
                res_df = pd.DataFrame({
                    'Ticker': selected,
                    'Periodic Return': [f"{mu_periodic[t]:.4%}" for t in selected],
                    'Annual Return': [f"{(mu_periodic[t]*multiplier):.2%}" for t in selected],
                    'Optimal Weight': opt_weights := [f"{w:.2%}" for w in opt_w]
                })
                st.table(res_df)
                st.bar_chart(pd.Series(opt_w, index=selected))
            else:
                st.error("Solver could not find a feasible solution.")
