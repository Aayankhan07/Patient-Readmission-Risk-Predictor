import time
import uuid
import json
import pandas as pd
from fastapi import APIRouter, HTTPException, BackgroundTasks
from api.schemas.patient import PatientInput
from api.schemas.response import PredictionResponse
from api.model_loader import model_state
from src.explainability.shap_explainer import explain_patient_shap
from src.explainability.lime_explainer import explain_patient_lime
from api.utils.audit import log_prediction_audit

router = APIRouter()


def compute_explanation_background(patient_id: str):
    """Asynchronously calculates SHAP and LIME metrics in a background thread."""
    try:
        preprocessed_df = model_state.retrieve_prediction(patient_id)
        if preprocessed_df is None:
            raise ValueError(f"No preprocessed features cached for {patient_id}")

        X_background = model_state.X_background
        if X_background is None:
            raise ValueError("Background dataset not loaded")

        # Compute SHAP
        shap_dict, base_val = explain_patient_shap(
            model_state.model, X_background, preprocessed_df
        )

        # Sort features by absolute SHAP impact
        sorted_feats = sorted(
            shap_dict.items(), key=lambda item: abs(item[1]), reverse=True
        )
        top_risk_factors = [feat for feat, val in sorted_feats[:3]]

        # Compute LIME HTML Explanation
        lime_html = explain_patient_lime(
            model_state.model, X_background, preprocessed_df
        )

        explanation_data = {
            "patient_id": patient_id,
            "shap_values": shap_dict,
            "top_risk_factors": top_risk_factors,
            "base_value": base_val,
            "lime_html": lime_html,
        }

        # Store in Redis/memory and set status to completed
        if model_state.use_redis and model_state.redis_client:
            model_state.redis_client.setex(
                f"explain:{patient_id}", 3600, json.dumps(explanation_data)
            )
            model_state.redis_client.set(f"status:{patient_id}", "completed")
        else:
            model_state.predictions_store[f"explain:{patient_id}"] = explanation_data
            model_state.predictions_store[f"status:{patient_id}"] = "completed"

    except Exception as err:
        print(f"Background XAI processing failed for patient {patient_id}: {err}")
        if model_state.use_redis and model_state.redis_client:
            model_state.redis_client.set(f"status:{patient_id}", f"failed: {str(err)}")
        else:
            model_state.predictions_store[f"status:{patient_id}"] = (
                f"failed: {str(err)}"
            )


def get_risk_tier(score: float) -> str:
    if score < 0.35:
        return "low"
    elif score <= 0.65:
        return "medium"
    else:
        return "high"


@router.post("/predict", response_model=PredictionResponse)
def predict_readmission(patient: PatientInput, background_tasks: BackgroundTasks):
    """
    Accepts patient information, preprocesses features, runs inference,
    and returns risk score + tier. Starts background calculations for explanations.
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
        raise HTTPException(
            status_code=400, detail=f"Inference preprocessing failed: {str(e)}"
        )

    inference_ms = (time.time() - start_time) * 1000.0
    patient_id = str(uuid.uuid4())

    # 5. Store details in state cache
    model_state.store_prediction(patient_id, df_preprocessed)

    # 6. Initialize status and queue background task
    if model_state.use_redis and model_state.redis_client:
        model_state.redis_client.set(f"status:{patient_id}", "pending")
    else:
        model_state.predictions_store[f"status:{patient_id}"] = "pending"

    background_tasks.add_task(compute_explanation_background, patient_id)

    # 7. Log query to SQLite audit DB
    log_prediction_audit(
        patient_id=patient_id,
        risk_score=score,
        risk_tier=get_risk_tier(score),
        model_version=model_state.model_version,
        inference_ms=inference_ms,
    )

    return PredictionResponse(
        patient_id=patient_id,
        risk_score=score,
        risk_tier=get_risk_tier(score),
        model_version=model_state.model_version,
        inference_ms=round(inference_ms, 2),
    )
