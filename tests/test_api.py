import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
import os
import sys

# Ensure imports work
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(root_dir, "src"))

from service.main import app
from service.inference import engine

client = TestClient(app)

@pytest.fixture
def mock_engine():
    # Store originals
    orig_model = engine.model
    orig_prep = engine.prep
    orig_explainer = engine.explainer
    orig_model_ver = engine.model_version
    orig_prep_ver = engine.prep_version
    orig_threshold = engine.threshold
    
    # Apply mocks
    engine.model = MagicMock()
    engine.prep = MagicMock()
    engine.explainer = MagicMock()
    engine.model_version = "mock_model_v1"
    engine.prep_version = "mock_prep_v1"
    engine.threshold = 0.35
    
    # Configure mock prediction return values
    engine.predict_single = MagicMock(return_value={
        "risk_score": 0.42,
        "risk_band": "high",
        "top_factors": [
            {"feature": "time_in_hospital", "value": 3.0, "shap_value": 0.12},
            {"feature": "num_medications", "value": 12.0, "shap_value": 0.08},
            {"feature": "has_diabetes_diag", "value": 1.0, "shap_value": 0.05}
        ],
        "model_version": "mock_model_v1",
        "preprocessing_version": "mock_prep_v1"
    })
    
    engine.predict_batch = MagicMock(return_value=[
        {
            "risk_score": 0.42,
            "risk_band": "high",
            "top_factors": [{"feature": "time_in_hospital", "value": 3.0, "shap_value": 0.12}],
            "model_version": "mock_model_v1",
            "preprocessing_version": "mock_prep_v1"
        },
        {
            "risk_score": 0.15,
            "risk_band": "low",
            "top_factors": [{"feature": "time_in_hospital", "value": 1.0, "shap_value": -0.05}],
            "model_version": "mock_model_v1",
            "preprocessing_version": "mock_prep_v1"
        }
    ])
    
    yield engine
    
    # Restore originals
    engine.model = orig_model
    engine.prep = orig_prep
    engine.explainer = orig_explainer
    engine.model_version = orig_model_ver
    engine.prep_version = orig_prep_ver
    engine.threshold = orig_threshold

def test_health_check(mock_engine):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_version"] == "mock_model_v1"
    assert data["preprocessing_version"] == "mock_prep_v1"
    assert data["savings_threshold"] == 0.35

def test_predict_single_endpoint(mock_engine):
    payload = {
        "race": "Caucasian",
        "gender": "Female",
        "age": "[50-60)",
        "time_in_hospital": 3,
        "num_lab_procedures": 40,
        "diag_1": "250.01",
        "insulin": "Steady"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["risk_score"] == 0.42
    assert data["risk_band"] == "high"
    assert len(data["top_factors"]) == 3
    assert data["model_version"] == "mock_model_v1"
    
    # Verify audit log is written
    assert os.path.exists("governance/audit_log.jsonl")

def test_predict_batch_endpoint(mock_engine):
    payload = {
        "patients": [
            {
                "race": "Caucasian",
                "gender": "Female",
                "age": "[50-60)",
                "time_in_hospital": 3
            },
            {
                "race": "AfricanAmerican",
                "gender": "Male",
                "age": "[60-70)",
                "time_in_hospital": 1
            }
        ]
    }
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data["results"]) == 2
    assert data["results"][0]["risk_score"] == 0.42
    assert data["results"][1]["risk_score"] == 0.15
    assert "batch_id" in data
    
    # Verify file saved
    batch_file = f"data/processed/batch_results/{data['batch_id']}.jsonl"
    assert os.path.exists(batch_file)
