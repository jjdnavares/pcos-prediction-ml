"""
API endpoint tests
"""
from fastapi.testclient import TestClient
from app.main import app
import pytest

client = TestClient(app)

def test_health_check():
    """Test health endpoint"""
    response = client.get("/health/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True

def test_readiness():
    """Test readiness endpoint"""
    response = client.get("/health/ready")
    assert response.status_code == 200
    assert response.json()["status"] == "ready"

def test_liveness():
    """Test liveness endpoint"""
    response = client.get("/health/live")
    assert response.status_code == 200
    assert response.json()["status"] == "alive"

def test_predict_valid_input():
    """Test prediction with valid input"""
    payload = {
        "age": 28,
        "weight": 65.0,
        "waist": 34,
        "marriage_status": 3,
        "tsh": 2.5,
        "follicle_no_l": 13,
        "follicle_no_r": 14,
        "skin_darkening": 1,
        "hair_growth": 1,
        "weight_gain": 1,
        "hair_loss": 0,
        "fast_food": 1,
        "cycle_regularity": 4,
        "cycle_length": 5
    }

    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert "prediction" in data
    assert "probability" in data
    assert "risk_level" in data
    assert data["prediction"] in [0, 1]
    assert 0.0 <= data["probability"] <= 1.0

def test_predict_invalid_age():
    """Test prediction with invalid age"""
    payload = {
        "age": 200,  # Invalid
        "weight": 65.0,
        "waist": 34,
        "marriage_status": 3,
        "tsh": 2.5,
        "follicle_no_l": 13,
        "follicle_no_r": 14,
        "skin_darkening": 1,
        "hair_growth": 1,
        "weight_gain": 1,
        "hair_loss": 0,
        "fast_food": 1,
        "cycle_regularity": 4,
        "cycle_length": 5
    }

    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 422  # Validation error

def test_docs_available():
    """Test that API docs are accessible"""
    response = client.get("/docs")
    assert response.status_code == 200
