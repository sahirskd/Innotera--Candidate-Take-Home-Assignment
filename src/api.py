from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
import pandas as pd
import logging
from typing import Optional

from src.model import RiskModel, AssessmentRequest
from src.pipeline import process_single_distributor, build_training_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Distributor Credit Risk Service",
    description="API for assessing credit risk of distributors.",
    version="1.0.0"
)

risk_model = RiskModel()

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "model_loaded": risk_model.is_loaded()
    }

@app.post("/assess")
def assess_risk(request: AssessmentRequest):
    """
    Assess the risk of a distributor based on their profile and transaction history.
    """
    if not risk_model.is_loaded():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded or unavailable."
        )
        
    try:
        dist_dict = request.distributor.model_dump()
        orders_list = [o.model_dump() for o in request.orders] if request.orders else []
        payments_list = [p.model_dump() for p in request.payments] if request.payments else []
        
        features_df = process_single_distributor(dist_dict, orders_list, payments_list)
        
        assessment = risk_model.predict(features_df)
        
        return {
            "distributor_id": dist_dict["distributor_id"],
            "assessment": assessment
        }
    except Exception as e:
        logger.error(f"Error processing assessment: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Error processing features: {str(e)}"
        )

@app.get("/portfolio/summary")
def get_portfolio_summary():
    """
    Provides summary of the portfolio's risk distribution
    """
    if not risk_model.is_loaded():
        raise HTTPException(status_code=503, detail="Model unavailable")
        
    try:
        X, y, df_master, cut_off = build_training_data("data")
        
        probs = risk_model.pipeline.predict_proba(X)[:, 1]
        threshold = risk_model.metadata.get("decision_threshold", 0.35)
        
        high_risk = (probs >= threshold).sum()
        medium_risk = ((probs >= 0.2) & (probs < threshold)).sum()
        low_risk = (probs < 0.2).sum()
        
        return {
            "total_distributors": len(df_master),
            "risk_distribution": {
                "LOW": int(low_risk),
                "MEDIUM": int(medium_risk),
                "HIGH": int(high_risk)
            },
        }
    except Exception as e:
        logger.error(f"Error computing portfolio summary: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

class FeedbackRequest(BaseModel):
    distributor_id: str
    decision_taken: str
    actual_default: Optional[bool] = None

@app.post("/feedback")
def record_feedback(feedback: FeedbackRequest):
    """
    To capture the actual outcomes and manual decisions for model monitoring.
    """
    # In production, this would write to a database or message queue
    logger.info(f"Feedback for {feedback.distributor_id}: decision={feedback.decision_taken}, default={feedback.actual_default}")
    
    return {
        "status": "success",
        "message": "Feedback recorded successfully."
    }
