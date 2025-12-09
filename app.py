import streamlit as st
import pandas as pd
import numpy as np
import pickle

# LOAD MODEL
@st.cache_resource
def load_model():
    with open("best_model_BBCA.pkl", "rb") as f:
        model_obj = pickle.load(f)
    return model_obj

model_obj = load_model()
model = model_obj["model"]
preprocessor = model_obj["preprocessor"]
feature_names = model_obj["columns"]

st.title("📈 Stock Forecasting App")

st.write("Aplikasi ini menggunakan model machine learning untuk melakukan prediksi harga saham berikutnya.")

# INPUT USER
st.subheader("Masukkan Data Fitur")

inputs = {}

for col in feature_names:
    val = st.number_input(f"{col}", value=0.0)
    inputs[col] = val

# Convert ke dataframe
input_df = pd.DataFrame([inputs])

st.write("### Data yang akan diprediksi:")
st.dataframe(input_df)

# PREDICT
if st.button("Predict"):
    # Preprocess
    X_processed = preprocessor.transform(input_df)
    
    # Predict
    y_pred = model.predict(X_processed)
    
    st.success(f"📊 **Hasil Forecasting:** {y_pred[0]:,.4f}")

