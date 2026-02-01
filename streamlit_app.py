import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import minimize

# Professional Setup
st.set_page_config(page_title="OptiPortfolio Pro | Andy Do", layout="wide")
st.title("📈 OptiPortfolio Pro: Multi-Period Optimizer")
st.write("Developed by: **Andy Do**") 

# --- 1. GLOBAL SETTINGS ---
st.sidebar.header("Global Settings")
freq_choice = st.sidebar.selectbox("Data Frequency", ['Daily', 'Weekly', 'Monthly'], index=1)
freq_map = {'Daily': 252, 'Weekly': 52, 'Monthly': 12}
multiplier = freq_map[freq_choice]

# Risk-free rate input as %
rf_annual_pct = st.sidebar.number_input("Annual Risk-Free Rate (%)", value=4.0, step=0.1)
rf_annual = rf_annual_pct / 100
rf_periodic = rf_annual / multiplier

# --- 2. DATA INPUT ---
uploaded_file = st.file_uploader("Upload Prices file (Excel/CSV)", type=["xlsx", "csv"])

if uploaded_file:
    df = pd.read_excel(uploaded_file) if uploaded_file.name.endswith('.xlsx') else pd.read_csv(uploaded_file)
    prices = df.select_dtypes(include=[np.number])
    
    # CALCULATE ARITHMETIC PERIODIC RETURNS
    returns = prices.pct_change().dropna()
    
    st.subheader(f"Asset Selection & Weight Constraints (% Scale)")
    all_tickers = returns.columns.tolist()
    selected = st.multiselect("Select Tickers", all_tickers, default=all_tickers[:5] if len(all_tickers) > 5 else all_tickers)

    if len(selected) >= 2:
        st.info("Input weights as percentages (e.g., enter 10 for 10%). Total Min % must be <= 100%.")
        
        # --- 3. INDIVIDUAL WEIGHT CONSTRAINTS (IN %) ---
        ind_cons = {}
        c_cols = st.columns(min(len(selected), 5)) 
        for i, t in enumerate(selected):
            with c_cols[i % 5]:
                st.markdown(f"**{t}**")
                min_v_pct = st.number_input(f"Min Weight %", 0.0, 100.0, 0.0, key=f"min_{t}")
                max_v_pct = st.number_input(f"Max Weight %", 0.0, 100.0, 100.0, key=f"max_{t}")
                ind_cons[t] = (min_v_pct / 100, max_v_pct / 100)

        # --- 4. EXECUTION ---
        if st.button("🚀 Run Portfolio Optimization"):
            total_min_pct = sum([v[0] for v in ind_cons.values()])
            
            if total_min_pct > 1.0:
                st.error(f"Impossible Constraints: Total Min Weight is {total_min_pct*100:.1f}%. Must be <= 100%.")
            else:
                mu_periodic = returns[selected].mean()
                cov_periodic = returns[selected].cov()

                def get_port_stats(w):
                    p_ret = np.sum(mu_periodic * w)
                    p_vol = np.sqrt(np.dot(w.T, np.dot(cov_periodic, w)))
                    return p_ret, p_vol

                def objective(w):
                    r, v = get_port_stats(w)
                    return -(r - rf_periodic) / v if v > 0 else 0

                constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0})
                bounds = [ind_cons[t] for t in selected]
                init_w = np.array([1.0/len(selected)] * len(selected))

                with st.spinner('Calculating...'):
                    res = minimize(objective, init_w, method='SLSQP', bounds=bounds, constraints=constraints)

                if res.success:
                    opt_w = res.x
                    p_ret_per, p_vol_per = get_port_stats(opt_w)
                    
                    st.divider()
                    st.header("3. Optimization Results")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"#### Periodic Stats ({freq_choice})")
                        st.metric("Exp. Return", f"{p_ret_per:.4%}")
                        st.metric("Risk (Std Dev)", f"{p_vol_per:.4%}")
                    
                    with col2:
                        st.markdown("#### Annualized Stats")
                        st.metric("Annual Return", f"{(p_ret_per * multiplier):.2%}")
                        st.metric("Annual Volatility", f"{(p_vol_per * np.sqrt(multiplier)):.2%}")

                    ann_sharpe = (p_ret_per * multiplier - rf_annual) / (p_vol_per * np.sqrt(multiplier))
                    st.subheader(f"Portfolio Sharpe Ratio: {ann_sharpe:.4f}")

                    st.subheader("Optimal Allocation Detail")
                    res_df = pd.DataFrame({
                        'Ticker': selected,
                        'Periodic Return': [f"{mu_periodic[t]:.4%}" for t in selected],
                        'Annual Return': [f"{(mu_periodic[t]*multiplier):.2%}" for t in selected],
                        'Optimal Weight (%)': [f"{w*100:.2f}%" for w in opt_w]
                    }).set_index('Ticker')
                    
                    st.table(res_df)
                    st.bar_chart(pd.Series(opt_w * 100, index=selected))
                    st.success(f"Total Portfolio Weight: {np.sum(opt_w)*100:.2f}%")
                else:
                    st.error("Solver failed. Check if constraints are mathematically possible.")
else:
    st.info("Awaiting historical prices to begin optimization.")
