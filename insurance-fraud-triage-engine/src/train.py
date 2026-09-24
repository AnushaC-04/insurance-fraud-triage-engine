import os
import json
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import precision_recall_curve, auc, f1_score
from imblearn.over_sampling import SMOTE
from preprocessing import build_preprocessing_pipeline

# Silence optuna logs
optuna.logging.set_verbosity(optuna.logging.WARNING)

def compute_pr_auc(y_true, y_pred_proba):
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    return auc(recall, precision)

def train_and_optimize(data_path: str):
    df = pd.read_csv(data_path)
    
    # Target encoding
    df["target"] = df["fraud_reported"].apply(lambda x: 1 if x == "Y" else 0)
    X = df.drop(columns=["fraud_reported", "target"])
    y = df["target"]

    # Stratified 70/15/15 Split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42)

    # Fit preprocessing pipeline on train set only
    pipeline = build_preprocessing_pipeline()
    X_train_proc = pipeline.fit_transform(X_train)
    X_val_proc = pipeline.transform(X_val)

    # Calculate class weighting for baseline comparison
    pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400),
            "max_depth": trial.suggest_int("max_depth", 3, 9),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, pos_weight),
            "random_state": 42
        }
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        pr_aucs = []

        for train_idx, val_idx in cv.split(X_train_proc, y_train):
            X_cv_train, y_cv_train = X_train_proc[train_idx], y_train.iloc[train_idx]
            X_cv_val, y_cv_val = X_train_proc[val_idx], y_train.iloc[val_idx]

            model = xgb.XGBClassifier(**params)
            model.fit(X_cv_train, y_cv_train, eval_set=[(X_cv_val, y_cv_val)], verbose=False)
            
            preds = model.predict_proba(X_cv_val)[:, 1]
            pr_aucs.append(compute_pr_auc(y_cv_val, preds))

        return np.mean(pr_aucs)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30)
    
    # Train final model on entire training set
    best_params = study.best_params
    final_model = xgb.XGBClassifier(**best_params)
    final_model.fit(X_train_proc, y_train)

    # Cost-Sensitive Threshold Optimization on Validation Set
    val_preds = final_model.predict_proba(X_val_proc)[:, 1]
    best_threshold = 0.5
    min_loss = float("inf")

    # Financial Matrix: FP = $150 (Investigation Cost), FN = actual total claim amount
    val_claims = X_val["total_claim_amount"].values

    for thresh in np.arange(0.05, 0.95, 0.01):
        pred_labels = (val_preds >= thresh).astype(int)
        
        fp_mask = (pred_labels == 1) & (y_val.values == 0)
        fn_mask = (pred_labels == 0) & (y_val.values == 1)
        
        total_loss = (sum(fp_mask) * 150) + np.sum(val_claims[fn_mask])

        if total_loss < min_loss:
            min_loss = total_loss
            best_threshold = thresh

    # Save artifacts
    os.makedirs("models", exist_ok=True)
    joblib.dump(pipeline, "models/preprocessor.pkl")
    final_model.save_model("models/xgb_fraud_model.json")
    
    with open("models/threshold_config.json", "w") as f:
        json.dump({"optimal_threshold": float(best_threshold), "min_val_loss": float(min_loss)}, f, indent=4)

    print(f"Training Complete. Optimal Threshold: {best_threshold:.4f} | Minimum Loss: ${min_loss:,.2f}")

if __name__ == "__main__":
    train_and_optimize("data/raw_claims.csv")