import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from model import build_and_train_model, predict

st.title("📈 GRU Stock Price Forecasting")
st.write("This app uses a GRU model to predict stock prices.")

# Upload CSV
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_csv("data/sample_data.csv")
    st.info("Using default sample data.")

st.subheader("Raw Data")
st.write(df.tail())

# Model parameters
time_step = st.slider("Time Step (window size)", 5, 60, 10)
epochs = st.slider("Epochs", 10, 200, 50)

# Train model
if st.button("Train Model"):
    with st.spinner("Training GRU model..."):
        model, scaler, X_test, y_test, y_pred = build_and_train_model(df, time_step, epochs)
        
        st.success("Model trained!")
        st.subheader("Prediction vs Actual")
        
        fig, ax = plt.subplots()
        ax.plot(y_test, label="Actual")
        ax.plot(y_pred, label="Predicted")
        ax.legend()
        st.pyplot(fig)