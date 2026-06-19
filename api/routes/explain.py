from fastapi import APIRouter, Request, HTTPException
from api.schemas.response import ExplainResponse
from src.explainability.shap_explainer import explain_patient_shap
from src.explainability.lime_explainer import explain_patient_lime

router = APIRouter()

@router.get("/explain/{patient_id}", response_model=ExplainResponse)
def explain_prediction(request: Request, patient_id: str):
    """
    Retrieves a cached patient prediction and generates SHAP & LIME explanations.
    """
    app = request.app
    if patient_id not in app.state.predictions_store:
        raise HTTPException(status_code=404, detail=f"Patient ID {patient_id} not found in predictions cache.")
        
    store_entry = app.state.predictions_store[patient_id]
    preprocessed_df = store_entry["preprocessed_df"]
    
    # 1. Fetch background data for SHAP & LIME
    X_background = app.state.X_background
    if X_background is None:
        raise HTTPException(status_code=503, detail="Background dataset for explanations is not loaded.")
        
    try:
        # 2. Compute SHAP Values
        shap_dict, _ = explain_patient_shap(app.state.model, X_background, preprocessed_df)
        
        # Sort features by absolute SHAP impact
        sorted_feats = sorted(shap_dict.items(), key=lambda item: abs(item[1]), reverse=True)
        top_risk_factors = [feat for feat, val in sorted_feats[:5]]
        
        # 3. Compute LIME HTML Explanation
        lime_html = explain_patient_lime(app.state.model, X_background, preprocessed_df)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate explanations: {str(e)}")
        
    return ExplainResponse(
        patient_id=patient_id,
        shap_values=shap_dict,
        top_risk_factors=top_risk_factors,
        lime_explanation=lime_html
    )
