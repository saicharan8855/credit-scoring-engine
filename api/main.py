# ============================================================
# main.py
# The FastAPI application.
#
# Endpoints:
# GET  /health   — check if the API is running
# POST /predict  — submit applicant data and get credit score
# GET  /docs     — automatic Swagger UI documentation
# ============================================================

import sys
import os

# Add project root to path so we can import from src/
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from api.schemas import ApplicantRequest, PredictionResponse, HealthResponse
from api.predictor import load_all_models, run_prediction
from api.explainer import generate_score_summary


# ============================================================
# Global model store
# Models are loaded once when the app starts
# and stored here so every request can use them
# without reloading from disk each time
# ============================================================

models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load all models when the API starts up.
    This runs once — not on every request.
    Using lifespan instead of on_event because
    on_event is deprecated in newer FastAPI versions.
    """
    print("Starting up Credit Scoring API...")
    models.update(load_all_models())
    print("API is ready to serve requests")
    yield
    print("Shutting down Credit Scoring API...")
    models.clear()


# ============================================================
# FastAPI app initialization
# ============================================================

app = FastAPI(
    title       = "Credit Scoring Engine",
    description = (
        "A bank-grade credit scoring API that takes loan applicant data "
        "and returns a credit score between 300 and 900, risk category, "
        "default probability, scorecard breakdown, and SHAP explanations."
    ),
    version     = "1.0.0",
    lifespan    = lifespan
)

# Allow all origins for development
# In production this should be restricted to specific domains
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"]
)


# ============================================================
# ENDPOINTS
# ============================================================

@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    """
    Check if the API is running and models are loaded.
    Returns a simple status message.
    """
    if not models:
        raise HTTPException(status_code=503, detail="Models not loaded yet")

    return HealthResponse(
        status  = "ok",
        message = "Credit Scoring API is running and all models are loaded"
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(applicant: ApplicantRequest):
    """
    Submit loan applicant data and receive a full credit assessment.

    Returns:
    - credit_score        : 300 to 900
    - risk_category       : Excellent / Good / Fair / Poor / Very Poor
    - default_probability : probability of default between 0 and 1
    - scorecard_breakdown : how each feature contributed to the score
    - shap_explanation    : SHAP values for each feature
    - top_risk_factors    : top 3 features increasing risk
    - top_strengths       : top 3 features decreasing risk
    """

    try:
        # Convert pydantic model to dictionary
        applicant_data = applicant.model_dump()

        # Run the full prediction pipeline
        result = run_prediction(applicant_data, models)

        # Return the response
        return PredictionResponse(**result)

    except Exception as e:
        raise HTTPException(
            status_code = 500,
            detail      = f"Prediction failed: {str(e)}"
        )


@app.post("/predict/explain", tags=["Prediction"])
def predict_with_explanation(applicant: ApplicantRequest):
    """
    Same as /predict but also returns a plain English explanation
    and score percentile. Useful for displaying to applicants
    or loan officers in a dashboard.
    """

    try:
        applicant_data = applicant.model_dump()
        result         = run_prediction(applicant_data, models)
        summary        = generate_score_summary(result)
        return summary

    except Exception as e:
        raise HTTPException(
            status_code = 500,
            detail      = f"Prediction failed: {str(e)}"
        )