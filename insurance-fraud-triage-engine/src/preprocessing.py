import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class InsuranceFeatureEngineer(BaseEstimator, TransformerMixin):
    """Custom transformer to engineer domain-specific features and drop leakage."""
    
    def __init__(self, drop_leakage=True):
        self.drop_leakage = drop_leakage
        self.leakage_cols = [
            "policy_number", "policy_bind_date", "incident_date",
            "incident_location", "insured_zip", "_c39"
        ]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        
        # Datetime feature engineering
        if "policy_bind_date" in df.columns and "incident_date" in df.columns:
            bind_dt = pd.to_datetime(df["policy_bind_date"])
            inc_dt = pd.to_datetime(df["incident_date"])
            df["days_to_incident"] = (inc_dt - bind_dt).dt.days
        
        # Financial & Ratio features
        if "total_claim_amount" in df.columns and "policy_annual_premium" in df.columns:
            df["claim_to_premium_ratio"] = (
                df["total_claim_amount"] / (df["policy_annual_premium"] + 1e-5)
            )
            
        if "total_claim_amount" in df.columns and "age" in df.columns:
            df["age_adjusted_claim"] = (
                df["total_claim_amount"] / (df["age"] + 1)
            )

        # Drop leakage columns if present
        if self.drop_leakage:
            cols_to_drop = [c for c in self.leakage_cols if c in df.columns]
            df = df.drop(columns=cols_to_drop)
            
        return df


def build_preprocessing_pipeline():
    """Builds a complete column transformation pipeline."""
    ordinal_cols = ["incident_severity"]
    ordinal_categories = [["Trivial Damage", "Minor Damage", "Major Damage", "Total Loss"]]

    nominal_cols = [
        "policy_state", "insured_sex", "insured_education_level", 
        "insured_occupation", "insured_hobbies", "insured_relationship",
        "incident_type", "collision_type", "authorities_contacted",
        "incident_state", "incident_city", "property_damage",
        "police_report_available", "auto_make", "auto_model"
    ]

    numeric_cols = [
        "months_as_customer", "age", "policy_deductable", "policy_annual_premium",
        "umbrella_limit", "capital-gains", "capital-loss", "total_claim_amount",
        "injury_claim", "property_claim", "vehicle_claim", "incident_hour_of_the_day",
        "number_of_vehicles_involved", "bodily_injuries", "witnesses",
        "auto_year", "days_to_incident", "claim_to_premium_ratio", "age_adjusted_claim"
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("ord", OrdinalEncoder(categories=ordinal_categories, handle_unknown="use_encoded_value", unknown_value=-1), ordinal_cols),
            ("nom", OneHotEncoder(handle_unknown="ignore", sparse_output=False), nominal_cols),
            ("num", StandardScaler(), numeric_cols)
        ],
        remainder="drop"
    )

    pipeline = Pipeline(steps=[
        ("feature_engineering", InsuranceFeatureEngineer(drop_leakage=True)),
        ("transformations", preprocessor)
    ])
    
    return pipeline