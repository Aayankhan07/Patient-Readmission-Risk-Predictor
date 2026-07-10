import pytest
from fastapi.testclient import TestClient
from api.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["ok", "degraded"]
    assert "version" in data


def test_models_endpoint(client):
    response = client.get("/models")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert isinstance(data["models"], list)


def test_predict_and_explain_endpoints(client):
    # 1. Test Predict
    patient_payload = {
        "age": "[60-70)",
        "time_in_hospital": 5,
        "num_procedures": 2,
        "num_medications": 14,
        "number_diagnoses": 7,
        "A1Cresult": ">8",
        "insulin": "Steady",
        "diabetesMed": "Yes",
    }

    response = client.post("/predict", json=patient_payload)
    assert response.status_code == 200
    data = response.json()

    assert "patient_id" in data
    assert "risk_score" in data
    assert "risk_tier" in data
    assert "inference_ms" in data

    patient_id = data["patient_id"]

    # 2. Test Explain
    explain_response = client.get(f"/explain/{patient_id}")
    assert explain_response.status_code == 200
    explain_data = explain_response.json()

    assert explain_data["patient_id"] == patient_id
    assert "shap_values" in explain_data
    assert "top_risk_factors" in explain_data
    assert "lime_html" in explain_data
    assert "base_value" in explain_data
    assert isinstance(explain_data["lime_html"], str)


def test_in_memory_fallback_caching(client):
    """
    Explicitly overrides use_redis to False to verify that predictions,
    status checking, and explanation queries work correctly via the in-memory store.
    """
    from api.model_loader import model_state

    # 1. Store original configuration and force fallback
    original_use_redis = model_state.use_redis
    model_state.use_redis = False

    try:
        patient_payload = {
            "age": "[50-60)",
            "time_in_hospital": 3,
            "num_procedures": 1,
            "num_medications": 10,
            "number_diagnoses": 5,
            "A1Cresult": "None",
            "insulin": "No",
            "diabetesMed": "No",
        }

        # 2. Trigger prediction
        response = client.post("/predict", json=patient_payload)
        assert response.status_code == 200
        data = response.json()
        assert "patient_id" in data
        patient_id = data["patient_id"]

        # 3. Trigger explanation check (it should return 202 pending or 200 complete depending on execution speed)
        explain_response = client.get(f"/explain/{patient_id}")
        assert explain_response.status_code in [200, 202]

    finally:
        # Restore original state
        model_state.use_redis = original_use_redis
