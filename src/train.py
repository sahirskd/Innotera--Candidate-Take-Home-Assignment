import os
import json
import joblib
import numpy as np
from datetime import datetime
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score, confusion_matrix, precision_recall_curve
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict

from src.pipeline import build_training_data

def train_and_save_model(data_dir: str, artifact_dir: str):
    print(f"Loading data from {data_dir}")
    
    X, y, df_master, t_cutoff = build_training_data(data_dir)
    
    print(f"Cutoff Date: {t_cutoff}")
    print(f"shape: {X.shape}")
    print(f"High Risk class ratio: {y.mean()}")
    
    numeric_cols = [
        "credit_limit_inr", "credit_terms_days", "credit_utilization",
        "total_invoices", "avg_delay_days", "max_delay_days", "on_time_ratio", "overdue_ratio",
        "total_orders", "avg_order_value", "cancellation_rate",
    ]
    categorical_cols = ["region", "channel"]
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', Pipeline([('scaler', StandardScaler())]), numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
    ])
    
    pipeline = Pipeline([
        ('prep', preprocessor),
        ('clf', LogisticRegression(C=0.003931680951235326, penalty='l2', solver="lbfgs", max_iter=1000, class_weight='balanced', random_state=42))
    ])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    
    print("Training model")
    pipeline.fit(X_train, y_train)
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_probas = cross_val_predict(pipeline, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    
    precisions, recalls, thresholds = precision_recall_curve(y_train, cv_probas)
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-9)
    best_idx = f1_scores[:-1].argmax()
    optimal_threshold = float(thresholds[best_idx])
    
    test_probs = pipeline.predict_proba(X_test)[:, 1]
    test_preds = (test_probs >= optimal_threshold).astype(int)
    
    pr_auc = average_precision_score(y_test, test_probs)
    roc_auc = roc_auc_score(y_test, test_probs)
    
    print("Model Evaluation:")
    print(f"best threshold from PR curve: {thresholds[best_idx]}")
    print(f"precision: {precisions[best_idx]}")
    print(f"recall: {recalls[best_idx]}")
    print(f"ROC-AUC: {roc_auc}")
    print(f"PR-AUC:  {pr_auc}")
    print(f"F1 Score:  {f1_scores[best_idx]}")
    print(classification_report(y_test, test_preds))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, test_preds))
    
    os.makedirs(artifact_dir, exist_ok=True)
    model_path = os.path.join(artifact_dir, "model.joblib")
    meta_path = os.path.join(artifact_dir, "metadata.json")
    
    print("Training model on entire dataset")
    pipeline.fit(X, y)
    joblib.dump(pipeline, model_path)
    
    metadata = {
        "model_type": "LogisticRegression_Pipeline",
        "test_roc_auc": float(roc_auc),
        "test_pr_auc": float(pr_auc),
        "decision_threshold": optimal_threshold,
        "features_numeric": numeric_cols,
        "features_categorical": categorical_cols,
        "trained_at": datetime.utcnow().isoformat(),
        "t_cutoff": str(t_cutoff)
    }
    
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=4)
        
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_and_save_model(data_dir="data", artifact_dir="model_artifacts")
