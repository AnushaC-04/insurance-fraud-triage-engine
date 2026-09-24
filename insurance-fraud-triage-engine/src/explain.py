import shap
import xgboost as xgb
import pandas as pd
import numpy as np

class FraudExplainer:
    """Wrapper for calculating local SHAP attributions for model predictions."""

    def __init__(self, model_path: str, preprocessor):
        self.model = xgb.XGBClassifier()
        self.model.load_model(model_path)
        self.preprocessor = preprocessor
        self.explainer = shap.TreeExplainer(self.model)

# Open src/explain.py and update line 13:

class FraudExplainer:
    def __init__(self, model_path: str, preprocessor):
        self.model = xgb.XGBClassifier()
        self.model.load_model(model_path)
        self.preprocessor = preprocessor
        
        # Pass the raw booster to bypass SHAP's base_score string parsing bug
        self.explainer = shap.TreeExplainer(self.model.get_booster())

    def get_feature_names(self):
        """Extract feature names after transformation."""
        col_transformer = self.preprocessor.named_steps["transformations"]
        feature_names = []
        for name, trans, cols in col_transformer.transformers_:
            if hasattr(trans, "get_feature_names_out"):
                feature_names.extend(trans.get_feature_names_out(cols))
            else:
                feature_names.extend(cols)
        return feature_names

    def explain_claim(self, raw_claim_df: pd.DataFrame, top_n: int = 5):
        """Processes raw claim and returns top N positive/negative SHAP feature attributions."""
        processed_data = self.preprocessor.transform(raw_claim_df)
        shap_values = self.explainer.shap_values(processed_data)
        
        feature_names = self.get_feature_names()
        claim_shap = shap_values[0]

        # Combine features with attributions
        attributions = [
            {"feature": name, "attribution": float(val)}
            for name, val in zip(feature_names, claim_shap)
        ]

        # Sort by absolute impact
        attributions.sort(key=lambda x: abs(x["attribution"]), reverse=True)
        return attributions[:top_n]