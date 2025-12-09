import streamlit as st
import pandas as pd
import numpy as np
import joblib

# LOAD MODEL MOVING AVERAGE
@st.cache_resource
def load_model():
    model_obj = joblib.load("best_model_bbca.joblib")
    return model_obj

model_obj = load_model()

window_size = model_obj["window_size"]
last_values = model_obj["last_values"]

st.title("📈 Stock Forecasting – Moving Average Model")

st.write(f"""
Aplikasi ini menggunakan **Moving Average** dengan Window Size = **{window_size}**  
Model menghitung prediksi harga berikutnya berdasarkan rata-rata dari {window_size} data terakhir.
""")

# INPUT USER
st.subheader(f"Masukkan {window_size} harga penutupan terakhir:")

user_inputs = []

for i in range(window_size):
    val = st.number_input(f"Harga ke-{i+1}", value=float(last_values[i]))
    user_inputs.append(val)

user_inputs = np.array(user_inputs)

st.write("### Data yang digunakan:")
st.write(user_inputs)

# PREDICT
if st.button("Predict"):
    prediction = np.mean(user_inputs)
    st.success(f"📊 **Prediksi Harga Berikutnya: {prediction:,.4f}**")
