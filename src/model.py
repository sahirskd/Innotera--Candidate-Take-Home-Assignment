import os
import json
import joblib
import pandas as pd
from pydantic import BaseModel
from typing import Optional, List, Dict

class DistributorData(BaseModel):
    distributor_id: str
    distributor_name: Optional[str] = "Unknown"
    region: Optional[str] = "Unknown"
    city: Optional[str] = "Unknown"
    channel: Optional[str] = "Unknown"
    credit_limit_inr: Optional[float] = 0.0
    credit_terms_days: Optional[int] = 0

class OrderHistory(BaseModel):
    order_id: str
    order_date: str
    order_value_inr: float
    order_status: str

class PaymentHistory(BaseModel):
    payment_id: str
    invoice_id: str
    invoice_date: str
    invoice_amount_inr: float
    due_date: str
    payment_date: Optional[str] = None
    amount_paid_inr: float
    payment_status: str

class AssessmentRequest(BaseModel):
    distributor: DistributorData
    orders: Optional[List[OrderHistory]] = []
    payments: Optional[List[PaymentHistory]] = []

class RiskModel:
    def __init__(self, artifact_dir: str = "model_artifacts"):
        self.model_path = os.path.join(artifact_dir, "model.joblib")
        self.meta_path = os.path.join(artifact_dir, "metadata.json")
        self.pipeline = None
        self.metadata = {}
        self._load_artifacts()
        
    def _load_artifacts(self):
        if os.path.exists(self.model_path) and os.path.exists(self.meta_path):
            self.pipeline = joblib.load(self.model_path)
            with open(self.meta_path, 'r') as f:
                self.metadata = json.load(f)
        else:
            print("model_artifacts not found")

    def is_loaded(self):
        return self.pipeline is not None

    def predict(self, features_df: pd.DataFrame) -> dict:
        if not self.is_loaded():
            raise RuntimeError("Model not loaded.")
            
        prob = self.pipeline.predict_proba(features_df)[0, 1]
        threshold = self.metadata.get("decision_threshold", 0.35)
        
        # is_high_risk = prob >= threshold
        
        if prob < 0.2:
            band = "LOW"
            recommendation = "APPROVE"
        elif prob < threshold:
            band = "MEDIUM"
            recommendation = "REVIEW"
        else:
            band = "HIGH"
            recommendation = "REJECT"
            
        return {
            "risk_score": float(prob),
            "risk_band": band,
            "recommendation": recommendation,
            "threshold_used": float(threshold)
        }
