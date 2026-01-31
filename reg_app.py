import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="GLD Price Predictor",
    layout="centered"
)

# -----------------------------
# Title
# -----------------------------
st.title("🏆 Gold Price (GLD) Prediction App")
st.write("Enter market indicators to predict **GLD (Gold ETF Price)**")

# -----------------------------
# Sidebar Inputs
# -----------------------------
st.sidebar.header("Market Inputs")

spx = st.sidebar.number_input("SPX (S&P 500 Index)", value=4500.0)
uso = st.sidebar.number_input("USO (Crude Oil ETF)", value=70.0)
slv = st.sidebar.number_input("SLV (Silver ETF)", value=25.0)
eurusd = st.sidebar.number_input("EUR / USD", value=1.10)

# -----------------------------
# Dummy Training Data
# (Replace with real dataset or saved model)
# -----------------------------
from sklearn.model_selection import train_test_split
gold_data = pd.read_csv('gold_price_data.csv')
gold_data = gold_data.drop(['Date'], axis=1)
X = gold_data.drop(['GLD'], axis=1)
Y = gold_data['GLD']
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=2)


# GLD prices

# Train model
model = LinearRegression()
model.fit(X_train, Y_train)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict GLD Price"):
    input_data = np.array([[spx, uso, slv, eurusd]])
    prediction = model.predict(input_data)

    st.success(f"💰 Predicted GLD Price: **{prediction[0]:.2f} USD**")

    st.subheader("📌 Input Summary")
    input_df = pd.DataFrame({
        "SPX": [spx],
        "USO": [uso],
        "SLV": [slv],
        "EUR/USD": [eurusd]
    })
    st.table(input_df)

# -----------------------------
# Footer
# -----------------------------
st.caption("📊 Model: Linear Regression | For educational purposes")
