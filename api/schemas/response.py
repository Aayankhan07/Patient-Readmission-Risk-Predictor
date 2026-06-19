from pydantic import BaseModel
from typing import Dict, List

class PredictionResponse(BaseModel):
    patient_id: str
    risk_score: float
    risk_tier: str
    model_version: str
    inference_ms: float

class ExplainResponse(BaseModel):
    patient_id: str
    shap_values: Dict[str, float]
    top_risk_factors: List[str]
    lime_explanation: str
