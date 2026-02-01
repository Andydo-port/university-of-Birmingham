import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import minimize

# Professional Setup for Andy Do - MSc Investment Birmingham
st.set_page_config(page_title="OptiPortfolio Pro | Andy Do", layout="wide")
st.title("📈 OptiPortfolio Pro: Multi-Period Optimizer")
st.write("Student: **Andy Do** | ID: 2926461 | University of Birmingham")

# --- STEP 1: INITIAL SETTINGS ---
st.sidebar.header("1. Global Settings")
# Frequency selection before data processing
freq_choice = st.sidebar.selectbox(
    "Select Data Frequency", 
    ['Daily', 'Weekly', 'Monthly'], 
    index=1  # Default to Weekly
)
freq_map = {'Daily': 252, 'Weekly': 52, 'Monthly': 12}
multiplier = freq_map[freq_choice]

rf_annual = st.sidebar.number_input("Annual Risk-Free Rate (e.g. 0.04)", value=0.04, step=0.001)
# Convert RF to periodic for solver consistency
rf_periodic = rf_annual / multiplier

# --- STEP 2: DATA INPUT ---
st.header("2. Data Input")
uploaded_file = st.file_uploader("Upload your Prices file (Excel/CSV)", type=["xlsx", "csv"])

if uploaded_file:
    # Load data based on file type
    if uploaded_file.name.endswith('.xlsx'):
        df = pd.read_excel(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file)
    
    # Extract numeric columns (Prices)
    prices = df.select_dtypes(include=[np.number])
    
    # CALCULATION: Arithmetic Periodic Returns
    # Using (P_t / P_t-1) - 1
    returns = prices.pct_change().dropna()
    
    st.subheader(f"Asset Selection & Individual Constraints ({freq_choice})")
    all_tickers = returns.columns.tolist()
    selected = st.multiselect("Select Tickers for Portfolio", all_tickers, default=all_tickers[:5] if len(all_tickers) > 5 else all_tickers)

    if len(selected) >= 2:
        # STEP 3: INDIVIDUAL WEIGHT CONSTRAINTS
        st.write("Set specific Min/Max weight boundaries for each asset:")
        ind_cons = {}
        # Create columns dynamically for inputs
        c_cols = st.columns(min(len(selected), 5)) 
        for i, t in enumerate(selected):
            with c_cols[i % 5]:
                st.markdown(f"**{t}**")
                min_v = st.number_input(f"Min %", 0.0, 1.0, 0.0, key=f"min_{t}", step=0.01)
                max_v = st.number_input(f"Max %", 0.0, 1.0, 1.0, key=f"max_{t}", step=0.01)
                ind_cons[t] = (min_v, max_v)

        # Pre-check logic for constraints
        if sum([v[0] for v in ind_cons.values()]) > 1:
            st.error("❌ Total Minimum Weights exceed 100%. Optimization is impossible.")
        else:
            # STEP 4: PERIODIC STATS (Arithmetic)
            mu_periodic = returns[selected].mean()
            cov_periodic = returns[selected].cov
