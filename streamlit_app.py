import streamlit as st
import pandas as pd

st.title('Investment Analysis Platform - University of Birmingham')
st.write('Chào An, đây là dashboard phân tích danh mục đầu tư của bạn.')

import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import plotly.graph_objects as go

# Cấu hình giao diện chuyên nghiệp cho Portfolio tại Birmingham
st.set_page_config(page_title="OptiPortfolio Expert - Andy", layout="wide")
st.title("📈 OptiPortfolio Expert")
st.markdown("### Advanced Portfolio Optimization Platform (MPT)")
st.write("Dự án của Đỗ Thành An - MSc Investment Student ID: 2926461")

# --- SIDEBAR: CÀI ĐẶT THAM SỐ (Settings) ---
st.sidebar.header("⚙️ Settings")
freq = st.sidebar.selectbox("Tần suất dữ liệu (Frequency)", ['Daily', 'Weekly', 'Monthly', 'Yearly'])
rf_rate = st.sidebar.number_input("Lãi suất phi rủi ro (Risk-free Rate)", value=0.04)

# Tính hệ số nhân dựa trên tần suất (giống đoạn code React của bạn)
freq_map = {'Daily': 252, 'Weekly': 52, 'Monthly': 12, 'Yearly': 1}
multiplier = freq_map[freq]

# --- PHẦN LỰA CHỌN SỐ STOCK & CONSTRAINT ---
st.sidebar.subheader("Asset Constraints")
num_stocks = st.sidebar.slider("Số lượng Stock trong danh mục", 2, 30, 10)
min_w = st.sidebar.slider("Weight tối thiểu mỗi mã (%)", 0, 50, 0) / 100
max_w = st.sidebar.slider("Weight tối đa mỗi mã (%)", 0, 100, 40) / 100

# --- XỬ LÝ DỮ LIỆU (Mô phỏng 30 cổ phiếu cho CFA Level II) ---
st.info("💡 Bạn có thể upload file Excel 30 cổ phiếu của mình tại đây trong tương lai.")

# Tạo dữ liệu ngẫu nhiên để demo thuật toán
tickers = [f"Stock {i+1}" for i in range(num_stocks)]
returns_data = np.random.normal(0.01, 0.05, (100, num_stocks))
df_returns = pd.DataFrame(returns_data, columns=tickers)

# Tính toán Mean và Covariance
avg_rets = df_returns.mean() * multiplier
cov_mat = df_returns.cov() * multiplier

# --- THUẬT TOÁN TỐI ƯU (Optimization Engine) ---
def get_stats(w):
    p_ret = np.sum(avg_rets * w)
    p_vol = np.sqrt(np.dot(w.T, np.dot(cov_mat, w)))
    return p_ret, p_vol

# Ràng buộc: Tổng tỷ trọng = 100%
cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
bounds = tuple((min_w, max_w) for _ in range(num_stocks))

# Tìm danh mục có Sharpe Ratio cao nhất (Tangency Portfolio)
def min_func_sharpe(w):
    p_ret, p_vol = get_stats(w)
    return -(p_ret - rf_rate) / p_vol

res = minimize(min_func_sharpe, num_stocks * [1./num_stocks], bounds=bounds, constraints=cons)
opt_w = res.x

# --- HIỂN THỊ KẾT QUẢ (Dashboard) ---
col1, col2, col3 = st.columns(3)
p_ret, p_vol = get_stats(opt_w)
col1.metric("Expected Return", f"{p_ret:.2%}")
col2.metric("Volatility (Risk)", f"{p_vol:.2%}")
col3.metric("Sharpe Ratio", f"{(p_ret - rf_rate) / p_vol:.4f}")

# Vẽ biểu đồ Weights
st.subheader("Optimal Asset Allocation")
st.bar_chart(pd.Series(opt_w, index=tickers))

st.success("Platform đã chạy thành công dựa trên logic từ AI Studio!")
