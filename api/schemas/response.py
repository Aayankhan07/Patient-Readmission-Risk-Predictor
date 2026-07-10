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
    base_value: float
    lime_html: str


class ModelInfo(BaseModel):
    name: str
    stage: str
    pr_auc: float
    roc_auc: float
    f1: float
    accuracy: float
    is_champion: bool


class ModelsResponse(BaseModel):
    models: List[ModelInfo]
