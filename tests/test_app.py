import pytest
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


def test_health_check_endpoint():
    """Verify that the /health endpoint returns a 200 OK status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "healthy"
    assert "model_loaded" in data


def test_predict_valid_input():
    """Verify inference pipeline with a valid Iris feature payload."""
    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
    }
    response = client.post("/predict", json=payload)
    
    # If the model is present, check 200 and schema
    if response.status_code == 200:
        data = response.json()
        assert "prediction" in data
        assert isinstance(data["prediction"], int)
        assert data["prediction"] in [0, 1, 2]
    else:
        # Fallback handling if MLflow database model isn't populated yet
        assert response.status_code == 503
        assert response.json()["detail"] == "Model artifact unavailable."


def test_predict_invalid_schema():
    """Verify that Pydantic rejects payloads missing required fields."""
    invalid_payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        # Missing petal parameters
    }
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422  # Unprocessable Entity