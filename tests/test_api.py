from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "model_loaded" in data

def test_assess_valid_payload():
    payload = {
        "distributor": {
            "distributor_id": "D_TEST",
            "credit_limit_inr": 100000,
            "credit_terms_days": 30
        },
        "orders": [
            {
                "order_id": "O1",
                "order_date": "2024-01-01",
                "order_value_inr": 50000,
                "order_status": "delivered"
            }
        ],
        "payments": [
            {
                "payment_id": "P1",
                "invoice_id": "I1",
                "invoice_date": "2024-01-01",
                "invoice_amount_inr": 50000,
                "due_date": "2024-01-31",
                "payment_date": "2024-01-15",
                "amount_paid_inr": 50000,
                "payment_status": "paid"
            }
        ]
    }
    
    response = client.post("/assess", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["distributor_id"] == "D_TEST"
    assert "assessment" in data
    assert "risk_score" in data["assessment"]
    assert "risk_band" in data["assessment"]

def test_assess_invalid_payload():
    payload = {
        "distributor": {
            "region": "North"
        }
    }
    
    response = client.post("/assess", json=payload)
    assert response.status_code == 422
    
def test_portfolio_summary():
    response = client.get("/portfolio/summary")
    assert response.status_code == 200
    data = response.json()
    assert "total_distributors" in data
    assert "risk_distribution" in data
