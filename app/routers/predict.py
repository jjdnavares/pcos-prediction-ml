"""
PCOS prediction endpoints
"""
from fastapi import APIRouter, Depends, HTTPException
from app.models import PatientInput, PredictionResponse, RiskLevel, Decision
from app.dependencies import get_model, get_scaler, get_feature_names, get_model_config
from app.utils.preprocessing import engineer_features, prepare_model_input
from app.config import settings
import logging
import numpy as np

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/predict", response_model=PredictionResponse)
def predict_pcos(
    patient: PatientInput,
    model=Depends(get_model),
    scaler=Depends(get_scaler),
    feature_names=Depends(get_feature_names),
    config=Depends(get_model_config)
):
    """
    Predict PCOS for a patient.

    **Workflow:**
    1. Validate input (Pydantic handles this)
    2. Engineer derived features (LH/FSH ratio, etc.)
    3. Prepare feature vector in correct order
    4. Scale features
    5. Model prediction
    6. Risk stratification
    7. Clinical decision recommendation

    **Returns:**
    - Prediction (0 or 1)
    - Probability (0.0 to 1.0)
    - Risk level (Low/Medium/High)
    - Clinical decision (Routine/Monitor/Refer)
    """

    try:
        # Step 1: Convert Pydantic model to dict
        patient_data = patient.dict()
        logger.info(f"Received prediction request for patient age {patient.age}")

        # Step 2: Engineer features
        patient_data = engineer_features(patient_data)

        # Step 3: Prepare model input (correct feature order)
        X = prepare_model_input(patient_data, feature_names)

        # Step 4: Scale features (extract scaling params for selected features only,
        # since scaler was fitted on all 49 features but model uses only 15)
        scaler_features = list(scaler.feature_names_in_)
        indices = [scaler_features.index(f) for f in feature_names]
        X_scaled = (X - scaler.mean_[indices]) / scaler.scale_[indices]

        # Step 5: Model prediction
        prediction = int(model.predict(X_scaled)[0])
        probabilities = model.predict_proba(X_scaled)[0]
        probability = float(probabilities[1])  # Probability of PCOS (class 1)
        confidence = float(max(probabilities))  # Confidence = max probability

        # Step 6: Risk stratification
        if probability < settings.LOW_RISK_THRESHOLD:
            risk_level = RiskLevel.LOW
            decision = Decision.ROUTINE
        elif probability < settings.HIGH_RISK_THRESHOLD:
            risk_level = RiskLevel.MEDIUM
            decision = Decision.MONITOR
        else:
            risk_level = RiskLevel.HIGH
            decision = Decision.REFER

        # Step 7: Feature importance (optional - simplified version)
        # For full SHAP values, you'd compute them here
        total_follicles = patient.follicle_no_l + patient.follicle_no_r
        top_features = {
            "Total_Follicle_Count": float(total_follicles / 60),  # Normalized
            "Symptom_Burden": float((patient.skin_darkening + patient.hair_growth + patient.weight_gain + patient.hair_loss) / 4),
            "Cycle(R/I)": float(patient.cycle_regularity / 5)
        }

        logger.info(f"Prediction: {prediction}, Probability: {probability:.3f}, Risk: {risk_level}")

        return PredictionResponse(
            prediction=prediction,
            probability=probability,
            confidence=confidence,
            risk_level=risk_level,
            decision=decision,
            top_features=top_features,
            model_version=config.get("version", settings.APP_VERSION)
        )

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid input: {str(e)}")
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@router.post("/batch-predict")
def batch_predict(
    patients: list[PatientInput],
    model=Depends(get_model),
    scaler=Depends(get_scaler),
    feature_names=Depends(get_feature_names),
    config=Depends(get_model_config)
):
    """
    Batch prediction for multiple patients.

    Useful for screening multiple patients in one API call.
    """

    if len(patients) > 100:
        raise HTTPException(status_code=400, detail="Batch size limited to 100 patients")

    results = []
    for i, patient in enumerate(patients):
        try:
            # Reuse single prediction logic
            result = predict_pcos(patient, model, scaler, feature_names, config)
            results.append({"patient_id": i, "result": result})
        except Exception as e:
            results.append({"patient_id": i, "error": str(e)})

    return {
        "total": len(patients),
        "predictions": results
    }
