"""
Minimal test to verify pandas import works in Streamlit
"""

import streamlit as st
import pandas as pd
import numpy as np

st.title("Test Pandas Import")

# Test pandas
try:
    data = {'A': [1, 2, 3], 'B': [4, 5, 6]}
    df = pd.DataFrame(data)
    st.success("✅ Pandas import successful!")
    st.dataframe(df)
except Exception as e:
    st.error(f"❌ Pandas error: {e}")

# Test numpy
try:
    arr = np.array([1, 2, 3, 4, 5])
    st.success("✅ Numpy import successful!")
    st.write(f"Numpy array: {arr}")
except Exception as e:
    st.error(f"❌ Numpy error: {e}")