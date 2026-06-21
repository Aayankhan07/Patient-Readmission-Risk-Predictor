from fastapi import APIRouter, HTTPException
from api.schemas.response import ExplainResponse
from src.explainability.shap_explainer import explain_patient_shap
from src.explainability.lime_explainer import explain_patient_lime
from api.model_loader import model_state

router = APIRouter()

@router.get("/explain/{patient_id}", response_model=ExplainResponse)
def explain_prediction(patient_id: str):
    """
    Retrieves a cached patient prediction and generates SHAP & LIME explanations.
    """
    if patient_id not in model_state.predictions_store:
        raise HTTPException(status_code=404, detail=f"Patient ID {patient_id} not found in predictions cache.")
        
    store_entry = model_state.predictions_store[patient_id]
    preprocessed_df = store_entry["preprocessed_df"]
    
    # 1. Fetch background data for SHAP & LIME
    X_background = model_state.X_background
    if X_background is None:
        raise HTTPException(status_code=503, detail="Background dataset for explanations is not loaded.")
        
    try:
        # 2. Compute SHAP Values
        shap_dict, base_val = explain_patient_shap(model_state.model, X_background, preprocessed_df)
        
        # Sort features by absolute SHAP impact
        sorted_feats = sorted(shap_dict.items(), key=lambda item: abs(item[1]), reverse=True)
        top_risk_factors = [feat for feat, val in sorted_feats[:3]]
        
        # 3. Compute LIME HTML Explanation
        lime_html = explain_patient_lime(model_state.model, X_background, preprocessed_df)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate explanations: {str(e)}")
        
    return ExplainResponse(
        patient_id=patient_id,
        shap_values=shap_dict,
        top_risk_factors=top_risk_factors,
        base_value=base_val,
        lime_html=lime_html
    )
