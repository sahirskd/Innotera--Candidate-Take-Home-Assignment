# Distributor Credit Risk Service

Risk scoring service for a distributor network.
Takes raw transaction data, trains a classifier, and exposes a REST API for credit risk assessment.

## Getting Started

**Build and Run** (Docker required):
```
docker compose up --build
```

API available at `http://localhost:8000`

Swagger docs at `http://localhost:8000/docs`

## Local Environment Setup

I used `uv` for fast dependency management and installation

```bash
# Create virtual env
uv venv

source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt
```

## Using the notebook

Once the dependencies are install you can explore the data understanding and model experimentation notebook:
`notebook/data-understanding.ipynb`

## Train the Model

To execute the training pipeline, run:

```bash
python -m src.train
```

## Start the API Server

I used Fast API to serve the model predictions:

```bash
PYTHONPATH=. .venv/bin/uvicorn src.api:app --reload
```

After running the server will be started and swagger docs will be available on: `http://localhost:8000/docs`

## Run Automated Tests

To verify that API logic are working correctly, run the test suite using `pytest`:

```bash
PYTHONPATH=. .venv/bin/pytest
```

---

## Methodology & Decisions

### Label Definition (high_risk)
Derived from future payment behavior. A distributor is high_risk if they have any of the following:
- Overdue invoices
- Bad debt (written_off)
- Unpaid ratio (unpaid/invoiced) > 15%
- Average payment delay > 20 days

### Algorithm: Logistic Regression
- **Why:** Outperformed Random Forest and XGBoost in Optuna trials. Logistic Regression generalized better on this noisy dataset without overfitting, handles class imbalance well (via class_weight='balanced'), and is highly interpretable.

### False Negative vs. False Positive Tradeoff
- **False Negative:** Approving a bad distributor (Cost = total cost of goods lost).
- **False Positive:** Rejecting a good distributor (Cost = lost profit margin).

- Approach: The decision threshold of 0.55 was selected by optimizing the F1-score on the Precision-Recall curve.

### Assumptions
- **Time-based Leakage Prevention:** Used a mid cutoff date of 2025-10-01. Features rely on historical data before the cutoff and labels are generated from future data.
- **Missing Data:** Missing order values were handled via imputation or logical dropping.
- **Cold Start:** Distributors without payment history default to a baseline risk assessment using available order features.

## Future Improvements

1. **Synthetic Features:** Can add synthetic features using ADASYN or SMOTE to improve model performance.
2. **MLOps:** Implement model monitoring for data drift and automated retraining pipelines.
3. **CI/CD:** Add GitHub Actions for automated linting and testing.
4. **Model Explainability:** Use SHAP to explain model predictions.
