import streamlit as st
import pandas as pd

st.title('Investment Analysis Platform - University of Birmingham')
st.write('Chào An, đây là dashboard phân tích danh mục đầu tư của bạn.')

# Tạo dữ liệu mẫu cho 30 cổ phiếu
df = pd.DataFrame({'Stock': [f'Stock {i}' for i in range(1, 31)], 'Return': [0.1]*30})
st.bar_chart(df.set_index('Stock'))
