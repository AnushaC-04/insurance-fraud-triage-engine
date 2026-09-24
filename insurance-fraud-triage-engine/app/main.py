import os
import json
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import xgboost as xgb
from src.explain import FraudExplainer
from src.triage import LLMTriageEngine

app = FastAPI(
    title="Insurance Fraud Triage Engine",
    description="Combined ML, Cost-Sensitive Thresholding, and Structured LLM Triage Service",
    version="1.0.0"
)

import os
import sys
import json
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import xgboost as xgb

# 1. Add project root and src directory to sys.path BEFORE importing src modules
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# 2. Import modules after modifying sys.path
from src.explain import FraudExplainer
from src.triage import LLMTriageEngine

# Global Artifacts
MODEL_PATH = "models/xgb_fraud_model.json"
PREPROCESSOR_PATH = "models/preprocessor.pkl"
THRESHOLD_PATH = "models/threshold_config.json"

pipeline = None
model = None
explainer = None
triage_engine = None
optimal_threshold = 0.5

@app.on_event("startup")
def load_artifacts():
    global pipeline, model, explainer, triage_engine, optimal_threshold
    
    if os.path.exists(PREPROCESSOR_PATH) and os.path.exists(MODEL_PATH):
        pipeline = joblib.load(PREPROCESSOR_PATH)
        model = xgb.XGBClassifier()
        model.load_model(MODEL_PATH)
        explainer = FraudExplainer(MODEL_PATH, pipeline)
        triage_engine = LLMTriageEngine()

    if os.path.exists(THRESHOLD_PATH):
        with open(THRESHOLD_PATH, "r") as f:
            config = json.load(f)
            optimal_threshold = config.get("optimal_threshold", 0.5)

class ClaimInput(BaseModel):
    months_as_customer: int = 328
    age: int = 48
    policy_deductable: int = 1000
    policy_annual_premium: float = 1406.91
    total_claim_amount: float = 71610.0
    injury_claim: float = 6510.0
    property_claim: float = 13020.0
    vehicle_claim: float = 52080.0
    incident_severity: str = "Major Damage"
    incident_type: str = "Single Vehicle Collision"
    collision_type: str = "Side Collision"
    authorities_contacted: str = "Police"
    incident_state: str = "SC"
    incident_city: str = "Columbus"
    property_damage: str = "YES"
    police_report_available: str = "YES"
    insured_sex: str = "FEMALE"
    insured_education_level: str = "MD"
    insured_occupation: str = "exec-managerial"
    insured_hobbies: str = "reading"
    insured_relationship: str = "husband"
    policy_state: str = "OH"
    auto_make: str = "Dodge"
    auto_model: str = "RAM"
    auto_year: int = 2007
    incident_hour_of_the_day: int = 5
    number_of_vehicles_involved: int = 1
    bodily_injuries: int = 1
    witnesses: int = 2
    umbrella_limit: int = 0
    capital_gains: float = 0.0
    capital_loss: float = 0.0
    policy_bind_date: str = "2014-10-17"
    incident_date: str = "2015-01-25"

@app.post("/predict")
def predict_fraud(claim: ClaimInput):
    if pipeline is None or model is None:
        raise HTTPException(status_code=500, detail="Model artifacts not loaded.")

    raw_data = claim.dict()
    
    # Safely convert underscores back to hyphens for preprocessor expectations
    if "capital_gains" in raw_data:
        raw_data["capital-gains"] = raw_data.pop("capital_gains")
    if "capital_loss" in raw_data:
        raw_data["capital-loss"] = raw_data.pop("capital_loss")
    
    input_df = pd.DataFrame([raw_data])

    # 1. Transform & Predict
    processed_features = pipeline.transform(input_df)
    probability = float(model.predict_proba(processed_features)[0, 1])

    # 2. Extract Top SHAP Features
    shap_features = explainer.explain_claim(input_df, top_n=5)

    # 3. LLM Audit Generation
    audit_report = triage_engine.generate_audit(
        claim_details=raw_data,
        fraud_probability=probability,
        optimal_threshold=optimal_threshold,
        top_shap_features=shap_features
    )

    return {
        "fraud_probability": round(probability, 4),
        "optimal_threshold": round(optimal_threshold, 4),
        "decision": "ESCALATE_SIU" if probability >= optimal_threshold else "PASS",
        "top_shap_features": shap_features,
        "llm_triage_audit": audit_report.dict()
    }