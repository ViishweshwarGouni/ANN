import streamlit as st
import numpy as np
import joblib
import os

# -----------------------------
# Load Model and Scaler
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "ann_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))

# -----------------------------
# UI
# -----------------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊"
)

st.title("📊 Customer Churn Prediction using ANN")

st.write("Enter customer details")

gender = st.selectbox("Gender", ["Female", "Male"])
senior = st.selectbox("Senior Citizen", [0, 1])
partner = st.selectbox("Partner", ["No", "Yes"])
dependents = st.selectbox("Dependents", ["No", "Yes"])

tenure = st.slider("Tenure (Months)", 0, 72, 12)

monthly_charges = st.number_input(
    "Monthly Charges",
    min_value=0.0,
    value=70.0
)

total_charges = st.number_input(
    "Total Charges",
    min_value=0.0,
    value=1000.0
)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Churn"):

    gender = 1 if gender == "Male" else 0
    partner = 1 if partner == "Yes" else 0
    dependents = 1 if dependents == "Yes" else 0

    # 19 features expected by model
    features = np.array([[
        gender,            # gender
        senior,            # SeniorCitizen
        partner,           # Partner
        dependents,        # Dependents
        tenure,            # tenure

        0,                 # PhoneService
        0,                 # MultipleLines
        0,                 # InternetService
        0,                 # OnlineSecurity
        0,                 # OnlineBackup
        0,                 # DeviceProtection
        0,                 # TechSupport
        0,                 # StreamingTV
        0,                 # StreamingMovies
        0,                 # Contract

        monthly_charges,   # MonthlyCharges
        total_charges,     # TotalCharges

        0,                 # PaperlessBilling
        0                  # PaymentMethod
    ]])

    try:
        scaled_data = scaler.transform(features)

        prediction = model.predict(scaled_data)[0]

        if prediction == 1:
            st.error("⚠ Customer is likely to churn")
        else:
            st.success("✅ Customer is likely to stay")

    except Exception as e:
        st.error(f"Error: {e}")