import time
import uuid
import pandas as pd
from fastapi import APIRouter, HTTPException
from api.schemas.patient import PatientInput
from api.schemas.response import PredictionResponse
from api.model_loader import model_state

router = APIRouter()

def get_risk_tier(score: float) -> str:
    if score < 0.35:
        return "low"
    elif score <= 0.65:
        return "medium"
    else:
        return "high"

@router.post("/predict", response_model=PredictionResponse)
def predict_readmission(patient: PatientInput):
    """
    Accepts patient information, preprocesses features, runs inference,
    and returns risk score + tier. Stores details in model_state for explanation.
    """
    if not model_state.loaded or model_state.model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded on server.")
        
    start_time = time.time()
    
    # 1. Convert Pydantic model to dict, resolving aliases
    patient_dict = patient.model_dump(by_alias=True)
    
    # 2. Convert to DataFrame
    df = pd.DataFrame([patient_dict])
    
    try:
        # 3. Preprocess
        df_preprocessed = model_state.preprocessor.transform(df)
        
        # 4. Infer
        score = model_state.predict_proba(df_preprocessed)
            
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Inference preprocessing failed: {str(e)}")
        
    inference_ms = (time.time() - start_time) * 1000.0
    patient_id = str(uuid.uuid4())
    
    # 5. Store details in state for subsequent explanation requests
    model_state.predictions_store[patient_id] = {
        "raw_df": df,
        "preprocessed_df": df_preprocessed
    }
    
    return PredictionResponse(
        patient_id=patient_id,
        risk_score=score,
        risk_tier=get_risk_tier(score),
        model_version=model_state.model_version,
        inference_ms=round(inference_ms, 2)
    )
