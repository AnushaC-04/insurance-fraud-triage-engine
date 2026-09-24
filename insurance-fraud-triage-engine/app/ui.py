import streamlit as st
import requests
import json

st.set_page_config(page_title="Fraud Triage Engine", layout="wide")

st.title("🛡️ Insurance Fraud Triage Engine")
st.markdown("Automated quantitative risk scoring with structured LLM investigation briefs.")

API_URL = "http://localhost:8000/predict"

with st.sidebar:
    st.header("Claim Submission Form")
    total_claim = st.number_input("Total Claim Amount ($)", value=71610.0)
    severity = st.selectbox("Incident Severity", ["Trivial Damage", "Minor Damage", "Major Damage", "Total Loss"], index=2)
    policy_premium = st.number_input("Policy Annual Premium ($)", value=1406.91)
    age = st.number_input("Insured Age", value=48)
    months_as_customer = st.number_input("Months as Customer", value=328)

if st.button("Analyze Claim for Fraud Risk"):
    payload = {
        "months_as_customer": int(months_as_customer),
        "age": int(age),
        "policy_deductable": 1000,
        "policy_annual_premium": float(policy_premium),
        "total_claim_amount": float(total_claim),
        "injury_claim": 6510.0,
        "property_claim": 13020.0,
        "vehicle_claim": 52080.0,
        "incident_severity": severity,
        "incident_type": "Single Vehicle Collision",
        "collision_type": "Side Collision",
        "authorities_contacted": "Police",
        "incident_state": "SC",
        "incident_city": "Columbus",
        "property_damage": "YES",
        "police_report_available": "YES",
        "insured_sex": "FEMALE",
        "insured_education_level": "MD",
        "insured_occupation": "exec-managerial",
        "insured_hobbies": "reading",
        "insured_relationship": "husband",
        "policy_state": "OH",
        "auto_make": "Dodge",
        "auto_model": "RAM",
        "auto_year": 2007,
        "incident_hour_of_the_day": 5,
        "number_of_vehicles_involved": 1,
        "bodily_injuries": 1,
        "witnesses": 2,
        "umbrella_limit": 0,
        "capital_gains": 0.0,
        "capital_loss": 0.0,
        "policy_bind_date": "2014-10-17",
        "incident_date": "2015-01-25"
    }

    try:
        response = requests.post(API_URL, json=payload)
        if response.status_code == 200:
            data = response.json()
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Fraud Probability", f"{data['fraud_probability'] * 100:.2f}%")
            col2.metric("Optimal Threshold", f"{data['optimal_threshold'] * 100:.2f}%")
            col3.metric("Decision", data["decision"])

            st.subheader("Top SHAP Feature Attributions")
            st.json(data["top_shap_features"])

            st.subheader("LLM Audit Brief")
            audit = data["llm_triage_audit"]
            st.write(f"**Risk Category:** {audit['risk_category']}")
            st.write(f"**Action Trigger:** {audit['action_trigger']}")
            st.write("**Supporting Evidence:**")
            for item in audit["supporting_evidence"]:
                st.write(f"- {item}")
            st.info(f"**Audit Notes:** {audit['audit_notes']}")
        else:
            st.error("Failed to fetch prediction from FastAPI backend.")
    except Exception as e:
        st.error(f"Error connecting to server: {e}")