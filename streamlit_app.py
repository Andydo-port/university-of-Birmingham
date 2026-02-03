import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import minimize

st.set_page_config(page_title="OptiPortfolio Pro | Andy Do", layout="wide")
st.title("📈 OptiPortfolio Pro: Comprehensive Optimizer")
st.write("Developed by: **Andy Do**")

# --- 1. GLOBAL SETTINGS ---
st.sidebar.header("Global Settings")
freq_choice = st.sidebar.selectbox("Data Frequency", ['Daily', 'Weekly', 'Monthly'], index=1)
freq_map = {'Daily': 252, 'Weekly': 52, 'Monthly': 12}
multiplier = freq_map[freq_choice]

rf_annual_pct = st.sidebar.number_input("Annual Risk-Free Rate (%)", value=4.0, step=0.1)
rf_annual = rf_annual_pct / 100
rf_periodic = rf_annual / multiplier 

# --- 2. DATA INPUT ---
st.header("1. Data Input")
uploaded_file = st.file_uploader("Upload Historical Prices", type=["xlsx", "csv"])

if uploaded_file:
    df = pd.read_excel(uploaded_file) if uploaded_file.name.endswith('.xlsx') else pd.read_csv(uploaded_file)
    prices = df.select_dtypes(include=[np.number])
    returns = prices.pct_change().dropna()
    
    st.subheader(f"Asset Selection & Combined Constraints")
    all_tickers = returns.columns.tolist()
    selected = st.multiselect("Select Tickers", all_tickers, default=all_tickers[:5])

    if len(selected) >= 2:
        st.info("Set constraints. The Solver will now include 'CASH' (Risk-Free Asset) in the optimization.")
        
        ind_cons = {}
        cols = st.columns(min(len(selected), 5)) 
        for i, t in enumerate(selected):
            with cols[i % 5]:
                st.markdown(f"**{t}**")
                min_v = st.number_input(f"Min %", 0.0, 100.0, 0.0, key=f"min_{t}") / 100
                max_v = st.number_input(f"Max %", 0.0, 100.0, 100.0, key=f"max_{t}") / 100
                ind_cons[t] = (min_v, max_v)

        # --- 3. OPTIMIZATION WITH RISK-FREE ASSET ---
        if st.button("🚀 Calculate Optimal Portfolio (Including Cash)"):
            # 1. Chuẩn bị dữ liệu rủi ro
            mu_stocks = returns[selected].mean().values
            cov_stocks = returns[selected].cov().values
            
            # 2. Thêm CASH vào ma trận (N + 1)
            # Cash có return = rf_periodic và rủi ro = 0
            mu_combined = np.append(mu_stocks, rf_periodic)
            
            # Thêm một hàng và một cột 0 vào ma trận hiệp phương sai cho Cash
            n = len(selected)
            cov_combined = np.zeros((n + 1, n + 1))
            cov_combined[:n, :n] = cov_stocks

            def get_combined_stats(w):
                # w là vector tỷ trọng [stock1, stock2, ..., stockN, CASH]
                r = np.dot(w, mu_combined)
                v = np.sqrt(np.dot(w.T, np.dot(cov_combined, w)))
                return r, v

            def objective(w):
                r, v = get_combined_stats(w)
                # Tối đa hóa Sharpe cho toàn bộ danh mục (bao gồm cả tiền mặt)
                # Chúng ta dùng một epsilon nhỏ để tránh chia cho 0
                return -(r - rf_periodic) / (v + 1e-10)

            # Ràng buộc: Tổng tỷ trọng (Stocks + Cash) = 100%
            constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0})
            
            # Bounds cho stocks và Cash (mặc định Cash từ 0-100%)
            bounds = [ind_cons[t] for t in selected]
            bounds.append((0.0, 1.0)) # Bound cho CASH
            
            init_w = np.array([1.0/(len(selected)+1)] * (len(selected)+1))

            res = minimize(objective, init_w, method='SLSQP', bounds=bounds, constraints=constraints, options={'ftol': 1e-15})

            if res.success:
                opt_weights = res.x
                r_final, v_final = get_combined_stats(opt_weights)
                
                st.divider()
                st.header("2. Final Optimized Allocation")
                
                # Hiển thị Metrics tổng thể
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("Total Annual Return", f"{(r_final * multiplier):.2%}")
                with c2:
                    st.metric("Total Annual Volatility", f"{(v_final * np.sqrt(multiplier)):.2%}")
                with c3:
                    final_sharpe = (r_final * multiplier - rf_annual) / (v_final * np.sqrt(multiplier) + 1e-10)
                    st.metric("Optimized Sharpe Ratio", f"{final_sharpe:.4f}")

                # Bảng kết quả bao gồm cả Cash
                asset_names = selected + ["CASH (Risk-Free)"]
                res_df = pd.DataFrame({
                    'Asset': asset_names,
                    'Type': ['Stock'] * len(selected) + ['Risk-Free'],
                    'Weight (%)': [f"{w*100:.2f}%" for w in opt_weights]
                }).set_index('Asset')
                
                st.subheader("Optimal Weight Distribution")
                st.table(res_df)
                
                # Biểu đồ tỷ trọng
                st.bar_chart(pd.Series(opt_weights * 100, index=asset_names))
                st.success(f"Verified Total Allocation: {np.sum(opt_weights)*100:.2f}%")
            else:
                st.error("Solver could not find an optimal solution with current constraints.")
else:
    st.info("Awaiting historical price data to begin.")
