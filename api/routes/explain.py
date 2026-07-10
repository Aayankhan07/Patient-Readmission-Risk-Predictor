import json
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from api.model_loader import model_state

router = APIRouter()


@router.get("/explain/{patient_id}")
def explain_prediction(patient_id: str):
    """
    Retrieves the background-calculated explanations.
    Returns HTTP 202 if pending, 200 with data if complete.
    """
    # 1. Retrieve status
    if model_state.use_redis and model_state.redis_client:
        status = model_state.redis_client.get(f"status:{patient_id}")
    else:
        status = model_state.predictions_store.get(f"status:{patient_id}")

    if not status:
        raise HTTPException(
            status_code=404,
            detail=f"Patient ID {patient_id} not found in predictions cache.",
        )

    if status == "pending":
        # 202 Accepted indicates request accepted but processing not complete
        return JSONResponse(
            status_code=202, content={"status": "pending", "patient_id": patient_id}
        )

    if status.startswith("failed"):
        raise HTTPException(
            status_code=500, detail=f"Explanation processing failed: {status}"
        )

    # 2. Retrieve completed payload
    if model_state.use_redis and model_state.redis_client:
        payload_str = model_state.redis_client.get(f"explain:{patient_id}")
        if payload_str:
            return json.loads(payload_str)
    else:
        payload = model_state.predictions_store.get(f"explain:{patient_id}")
        if payload:
            return payload

    raise HTTPException(
        status_code=500,
        detail="Explanation payload not found despite completed status.",
    )
