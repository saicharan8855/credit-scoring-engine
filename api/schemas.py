# ============================================================
# schemas.py
# Defines the request and response data structures for the API.
#
# Pydantic models are used here. Pydantic automatically:
# - Validates incoming request data
# - Returns clear error messages for invalid inputs
# - Documents the API automatically via FastAPI
# ============================================================

from pydantic import BaseModel, Field
from typing import Optional


# ============================================================
# REQUEST SCHEMA
# This is the data the API expects when someone sends a
# POST request to /predict
# ============================================================

class ApplicantRequest(BaseModel):
    """
    Input data for a single loan applicant.
    These are the raw features before any preprocessing.
    The API will handle all preprocessing internally.
    """

    # Contract and loan details
    NAME_CONTRACT_TYPE  : str   = Field(..., example="Cash loans")
    AMT_CREDIT          : float = Field(..., example=500000.0,  description="Total loan amount")
    AMT_ANNUITY         : float = Field(..., example=25000.0,   description="Monthly annuity amount")
    AMT_INCOME_TOTAL    : float = Field(..., example=180000.0,  description="Total annual income")
    AMT_GOODS_PRICE     : float = Field(..., example=450000.0,  description="Price of goods for loan")

    # Personal details
    CODE_GENDER         : str   = Field(..., example="M",       description="Gender M or F")
    FLAG_OWN_CAR        : str   = Field(..., example="Y",       description="Owns a car Y or N")
    FLAG_OWN_REALTY     : str   = Field(..., example="Y",       description="Owns real estate Y or N")
    CNT_CHILDREN        : int   = Field(..., example=0,         description="Number of children")
    CNT_FAM_MEMBERS     : float = Field(..., example=2.0,       description="Number of family members")

    # Demographic details
    NAME_INCOME_TYPE    : str   = Field(..., example="Working")
    NAME_EDUCATION_TYPE : str   = Field(..., example="Higher education")
    NAME_FAMILY_STATUS  : str   = Field(..., example="Married")
    NAME_HOUSING_TYPE   : str   = Field(..., example="House / apartment")

    # Days features — negative values mean days before application
    DAYS_BIRTH          : int   = Field(..., example=-12000,    description="Days before application (negative)")
    DAYS_EMPLOYED       : int   = Field(..., example=-3000,     description="Days employed (negative), 365243 if unemployed")
    DAYS_REGISTRATION   : float = Field(..., example=-5000.0,   description="Days since registration")
    DAYS_ID_PUBLISH     : int   = Field(..., example=-2000,     description="Days since ID published")
    DAYS_LAST_PHONE_CHANGE : float = Field(..., example=-500.0, description="Days since last phone change")

    # External scores
    EXT_SOURCE_1        : Optional[float] = Field(None, example=0.5)
    EXT_SOURCE_2        : Optional[float] = Field(None, example=0.6)
    EXT_SOURCE_3        : Optional[float] = Field(None, example=0.5)

    # Other features
    OCCUPATION_TYPE         : Optional[str]   = Field(None, example="Laborers")
    ORGANIZATION_TYPE       : Optional[str]   = Field(None, example="Business Entity Type 3")
    REGION_RATING_CLIENT    : int             = Field(..., example=2)
    REGION_RATING_CLIENT_W_CITY : int         = Field(..., example=2)
    REGION_POPULATION_RELATIVE  : float       = Field(..., example=0.035)
    OWN_CAR_AGE             : Optional[float] = Field(None, example=5.0)

    # Property features
    TOTALAREA_MODE          : Optional[float] = Field(None, example=0.05)
    FLOORSMAX_AVG           : Optional[float] = Field(None, example=0.1)
    FLOORSMAX_MODE          : Optional[float] = Field(None, example=0.1)
    FLOORSMAX_MEDI          : Optional[float] = Field(None, example=0.1)
    YEARS_BEGINEXPLUATATION_AVG  : Optional[float] = Field(None, example=0.9)
    YEARS_BEGINEXPLUATATION_MODE : Optional[float] = Field(None, example=0.9)
    YEARS_BEGINEXPLUATATION_MEDI : Optional[float] = Field(None, example=0.9)

    class Config:
        json_schema_extra = {
            "example": {
                "NAME_CONTRACT_TYPE"  : "Cash loans",
                "AMT_CREDIT"          : 500000.0,
                "AMT_ANNUITY"         : 25000.0,
                "AMT_INCOME_TOTAL"    : 180000.0,
                "AMT_GOODS_PRICE"     : 450000.0,
                "CODE_GENDER"         : "M",
                "FLAG_OWN_CAR"        : "Y",
                "FLAG_OWN_REALTY"     : "Y",
                "CNT_CHILDREN"        : 0,
                "CNT_FAM_MEMBERS"     : 2.0,
                "NAME_INCOME_TYPE"    : "Working",
                "NAME_EDUCATION_TYPE" : "Higher education",
                "NAME_FAMILY_STATUS"  : "Married",
                "NAME_HOUSING_TYPE"   : "House / apartment",
                "DAYS_BIRTH"          : -12000,
                "DAYS_EMPLOYED"       : -3000,
                "DAYS_REGISTRATION"   : -5000.0,
                "DAYS_ID_PUBLISH"     : -2000,
                "DAYS_LAST_PHONE_CHANGE" : -500.0,
                "EXT_SOURCE_1"        : 0.5,
                "EXT_SOURCE_2"        : 0.6,
                "EXT_SOURCE_3"        : 0.5,
                "OCCUPATION_TYPE"     : "Laborers",
                "ORGANIZATION_TYPE"   : "Business Entity Type 3",
                "REGION_RATING_CLIENT": 2,
                "REGION_RATING_CLIENT_W_CITY": 2,
                "REGION_POPULATION_RELATIVE" : 0.035,
                "OWN_CAR_AGE"         : 5.0,
                "TOTALAREA_MODE"      : 0.05,
                "FLOORSMAX_AVG"       : 0.1,
                "FLOORSMAX_MODE"      : 0.1,
                "FLOORSMAX_MEDI"      : 0.1,
                "YEARS_BEGINEXPLUATATION_AVG"  : 0.9,
                "YEARS_BEGINEXPLUATATION_MODE" : 0.9,
                "YEARS_BEGINEXPLUATATION_MEDI" : 0.9
            }
        }


# ============================================================
# RESPONSE SCHEMAS
# This is the data the API returns after processing
# ============================================================

class SHAPFeatureExplanation(BaseModel):
    """SHAP explanation for a single feature."""
    feature    : str
    woe_value  : float
    shap_value : float
    impact     : str


class PredictionResponse(BaseModel):
    """
    Full response returned by the /predict endpoint.
    Contains the credit score, risk category, default probability,
    scorecard breakdown, and SHAP explanation.
    """

    # Core prediction
    credit_score      : int   = Field(..., description="Credit score between 300 and 900")
    risk_category     : str   = Field(..., description="Excellent / Good / Fair / Poor / Very Poor")
    default_probability : float = Field(..., description="Probability of default between 0 and 1")

    # Scorecard breakdown
    scorecard_breakdown : list = Field(..., description="Per feature point contribution to the score")

    # SHAP explanation
    shap_explanation  : list  = Field(..., description="SHAP values explaining the prediction")

    # Top reasons
    top_risk_factors  : list  = Field(..., description="Top 3 features increasing default risk")
    top_strengths     : list  = Field(..., description="Top 3 features decreasing default risk")


class HealthResponse(BaseModel):
    """Response for the /health endpoint."""
    status  : str
    message : str