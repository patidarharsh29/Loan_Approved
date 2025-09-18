import streamlit as st
import joblib
import pandas as pd

# Load model, scaler, and expected columns
model  = joblib.load("Model.pkl")
expected_columns = joblib.load("columns.pkl")

st.title("Loan Approval Prediction App ❤️")
st.markdown("Fill applicant details below:")

# User Inputs
loan_id = st.text_input("Loan ID", "LN001")
no_of_dependents = st.number_input("Number of Dependents", 0, 10, 0)
income_annum = st.number_input("Annual Income", 0, 100000000, 500000)
loan_amount = st.number_input("Loan Amount", 0, 100000000, 200000)
loan_term = st.number_input("Loan Term (months)", 0, 480, 120)
cibil_score = st.number_input("CIBIL Score", 0, 900, 700)
residential_assets_value = st.number_input("Residential Assets Value", 0, 100000000, 100000)
commercial_assets_value = st.number_input("Commercial Assets Value", 0, 100000000, 0)
luxury_assets_value = st.number_input("Luxury Assets Value", 0, 100000000, 0)
bank_asset_value = st.number_input("Bank Asset Value", 0, 100000000, 50000)

education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["No", "Yes"])

if st.button("Predict"):
    # Convert categorical to match dummies
    raw_input = {
        "loan_id": sum([ord(c) for c in str(loan_id)]),  # convert string to numeric
        "no_of_dependents": no_of_dependents,
        "income_annum": income_annum,
        "loan_amount": loan_amount,
        "loan_term": loan_term,
        "cibil_score": cibil_score,
        "residential_assets_value": residential_assets_value,
        "commercial_assets_value": commercial_assets_value,
        "luxury_assets_value": luxury_assets_value,
        "bank_asset_value": bank_asset_value,
        "education_ Not Graduate": 1 if education == "Not Graduate" else 0,
        "self_employed_ Yes": 1 if self_employed == "Yes" else 0,
    }

    input_df = pd.DataFrame([raw_input])

    # Ensure all expected columns exist
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[expected_columns]

    prediction = model.predict(input_df)[0]

    if prediction == 0:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")
