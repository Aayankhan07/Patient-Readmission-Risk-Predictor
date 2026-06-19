from fastapi import APIRouter, Request

router = APIRouter()

@router.get("/health")
def health_check(request: Request):
    """Checks the health of the serving API and whether the model is loaded."""
    model_loaded = hasattr(request.app.state, "model") and request.app.state.model is not None
    preprocessor_loaded = hasattr(request.app.state, "preprocessor") and request.app.state.preprocessor is not None
    
    status = "ok" if (model_loaded and preprocessor_loaded) else "degraded"
    
    return {
        "status": status,
        "model_loaded": model_loaded,
        "preprocessor_loaded": preprocessor_loaded,
        "version": "1.0.0"
    }

@router.get("/models")
def get_models(request: Request):
    """Returns details of the loaded model in production."""
    model_type = getattr(request.app.state, "model_type", "unknown")
    run_id = getattr(request.app.state, "model_run_id", "unknown")
    
    return {
        "champion_model": {
            "type": model_type,
            "run_id": run_id,
            "stage": "Production"
        }
    }
