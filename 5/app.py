import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="💰",
    layout="centered"
)

st.title("💰 Loan Approval Predictor")
st.write("Enter the applicant information below to predict loan approval.")

try:
    model = joblib.load("loan_approval_model.joblib")
except Exception as e:
    st.error("Model file could not be loaded.")
    st.write(e)
    st.stop()

no_of_dependents = st.number_input(
    "Number of Dependents",
    min_value=0,
    max_value=10,
    value=2
)

education = st.selectbox(
    "Education",
    ["Graduate", "Not Graduate"]
)

self_employed = st.selectbox(
    "Self Employed",
    ["Yes", "No"]
)

income_annum = st.number_input(
    "Annual Income",
    min_value=0,
    value=500000
)

loan_amount = st.number_input(
    "Loan Amount",
    min_value=0,
    value=2000000
)

loan_term = st.number_input(
    "Loan Term",
    min_value=1,
    max_value=50,
    value=10
)

cibil_score = st.number_input(
    "CIBIL Score",
    min_value=300,
    max_value=900,
    value=700
)

residential_assets_value = st.number_input(
    "Residential Assets Value",
    min_value=0,
    value=5000000
)

commercial_assets_value = st.number_input(
    "Commercial Assets Value",
    min_value=0,
    value=2000000
)

luxury_assets_value = st.number_input(
    "Luxury Assets Value",
    min_value=0,
    value=3000000
)

bank_asset_value = st.number_input(
    "Bank Asset Value",
    min_value=0,
    value=3000000
)

if st.button("Predict Loan Status"):

    input_data = pd.DataFrame({
        "no_of_dependents": [no_of_dependents],
        "education": [education],
        "self_employed": [self_employed],
        "income_annum": [income_annum],
        "loan_amount": [loan_amount],
        "loan_term": [loan_term],
        "cibil_score": [cibil_score],
        "residential_assets_value": [residential_assets_value],
        "commercial_assets_value": [commercial_assets_value],
        "luxury_assets_value": [luxury_assets_value],
        "bank_asset_value": [bank_asset_value]
    })

    try:
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]
        confidence = max(probability) * 100

        if str(prediction).strip().lower() == "approved":
            st.success("✅ Loan Approved")
        else:
            st.error("❌ Loan Rejected")

        st.metric(
            "Prediction Confidence",
            f"{confidence:.2f}%"
        )

    except Exception as e:
        st.error("Prediction failed.")
        st.write(e)