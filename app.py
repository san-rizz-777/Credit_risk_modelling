import streamlit as st
import pandas as pd
import joblib

# Load the model and the encoders from joblib
model = joblib.load("./models/xgb_credit_model.pkl")
encoders = {col:joblib.load(f"./encoders/{col}_encoder.pkl") for col in ["Sex", "Saving accounts", "Risk", "Housing", "Checking account"]}

st.title("Credit Risk Prediction App")
st.write("Enter the applicant information to predict if the credit risk is good or bad!!!")

age = st.number_input("Age", min_value=18, max_value=80, value=30)
sex = st.selectbox("Sex", ["Male", "Female"])
job = st.number_input("Job", min_value=0, max_value=100, value=100)