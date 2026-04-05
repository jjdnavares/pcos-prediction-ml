"""
Pydantic models for request/response validation
"""
from pydantic import BaseModel, Field
from typing import Optional, Dict
from enum import Enum

# Risk level enumeration
class RiskLevel(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"

# Decision recommendation
class Decision(str, Enum):
    ROUTINE = "Routine annual screening"
    MONITOR = "3-month monitoring + lifestyle counseling"
    REFER = "Immediate specialist referral"

class PatientInput(BaseModel):
    """
    Patient data for PCOS prediction.

    Fields match the 15 features (13 raw + 2 engineered) used by the trained SVM model.
    Value ranges are based on clinical norms + dataset statistics.
    """

    # Demographics
    age: float = Field(..., ge=10, le=60, description="Patient age in years")
    weight: float = Field(..., ge=20, le=200, description="Weight in kilograms")
    waist: float = Field(..., ge=15, le=60, description="Waist circumference (inches)")
    marriage_status: float = Field(..., ge=0, le=40, description="Marriage status (years)")

    # Metabolic markers
    tsh: float = Field(..., ge=0, le=25, description="TSH level (mIU/L)")

    # Ovarian morphology
    follicle_no_l: int = Field(..., ge=0, le=30, description="Left ovary follicle count")
    follicle_no_r: int = Field(..., ge=0, le=30, description="Right ovary follicle count")

    # Physical symptoms (0=No, 1=Yes)
    skin_darkening: int = Field(..., ge=0, le=1, description="Skin darkening (0=No, 1=Yes)")
    hair_growth: int = Field(..., ge=0, le=1, description="Excessive hair growth (0=No, 1=Yes)")
    weight_gain: int = Field(..., ge=0, le=1, description="Weight gain (0=No, 1=Yes)")
    hair_loss: int = Field(..., ge=0, le=1, description="Hair loss (0=No, 1=Yes)")
    fast_food: int = Field(..., ge=0, le=1, description="Regular fast food consumption (0=No, 1=Yes)")

    # Menstrual cycle
    cycle_regularity: int = Field(..., ge=2, le=5, description="Cycle regularity (2=Regular, 4=Irregular)")
    cycle_length: float = Field(..., ge=1, le=7, description="Cycle length category (1-7 scale, dataset encoding)")

    class Config:
        json_schema_extra = {
            "example": {
                "age": 28,
                "weight": 65.0,
                "waist": 34,
                "marriage_status": 3,
                "tsh": 2.5,
                "follicle_no_l": 13,
                "follicle_no_r": 14,
                "skin_darkening": 1,
                "hair_growth": 1,
                "weight_gain": 1,
                "hair_loss": 0,
                "fast_food": 1,
                "cycle_regularity": 5,
                "cycle_length": 5
            }
        }

class PredictionResponse(BaseModel):
    """API response for PCOS prediction"""

    # Prediction results
    prediction: int = Field(..., description="0 = No PCOS, 1 = PCOS detected")
    probability: float = Field(..., ge=0, le=1, description="Probability of PCOS (0-1)")
    confidence: float = Field(..., ge=0, le=1, description="Model confidence score")

    # Risk assessment
    risk_level: RiskLevel = Field(..., description="Risk stratification: Low/Medium/High")
    decision: Decision = Field(..., description="Clinical decision recommendation")

    # Explainability (optional)
    top_features: Optional[Dict[str, float]] = Field(
        None,
        description="Top contributing features (SHAP-like importance)"
    )

    # Metadata
    model_version: str = Field(..., description="Model version used for prediction")

    class Config:
        json_schema_extra = {
            "example": {
                "prediction": 1,
                "probability": 0.91,
                "confidence": 0.91,
                "risk_level": "High",
                "decision": "Immediate specialist referral",
                "top_features": {
                    "Total_Follicle_Count": 0.18,
                    "Symptom_Burden": 0.15,
                    "Cycle(R/I)": 0.12
                },
                "model_version": "1.0.0"
            }
        }

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    service: str
    version: str
    model_loaded: bool

class ErrorResponse(BaseModel):
    """Error response"""
    error: str
    detail: Optional[str] = None
