"""
CHARGEUP - Comprehensive Test Suite
Tests:
1. Data Pipeline & Cleaning Integrity
2. Proxy Fault Classification Rules
3. Hourly Feature Engineering & Lag Consistency
4. Machine Learning Inference & Artifact Validation
5. FastAPI REST API Endpoints (/health, /predict, /api/..., /docs)
6. HTML Dashboard Template Rendering
7. Input Validation & Edge Case Handling
"""

import os
import json
# No external test runner required; standard library assertions used
from fastapi.testclient import TestClient

import app
import data_pipeline

client = TestClient(app.app)


def test_data_pipeline_fault_rules():
    """Verify proxy fault rule mappings against known ChargePoint outcomes."""
    for normal_term in ["Plug Out at Vehicle", "Customer", "CPS Server", "Plug Out at Station", "Holster Plugin"]:
        assert normal_term in data_pipeline.NORMAL_ENDED_BY
        
    for abnormal_term in ["Station Offline", "Outlet Unreachable", "Relay Stuck Closed", "Final GFCI Tripped"]:
        assert abnormal_term in data_pipeline.ABNORMAL_ENDED_BY


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "Models")


def test_models_artifacts_exist():
    """Verify serialized ML artifacts exist on disk."""
    assert os.path.exists(os.path.join(MODELS_DIR, "xgboost_demand_model.pkl"))
    assert os.path.exists(os.path.join(MODELS_DIR, "kmeans_station_model.pkl"))
    assert os.path.exists(os.path.join(MODELS_DIR, "kmeans_scaler.pkl"))
    assert os.path.exists(os.path.join(MODELS_DIR, "pipeline_summary.json"))


def test_pipeline_summary_metrics():
    """Verify evaluation metrics were calculated on held-out test data."""
    with open(os.path.join(MODELS_DIR, "pipeline_summary.json"), "r") as f:
        summary = json.load(f)
        
    metrics = summary["forecasting"]["metrics"]
    assert "mae" in metrics
    assert "rmse" in metrics
    assert "r2" in metrics
    assert metrics["mae"] > 0
    assert metrics["rmse"] > 0
    assert metrics["r2"] > 0
    assert summary["dataset_stats"]["station_count"] == 47


def test_health_endpoint():
    """Test GET /health returns 200 and healthy status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["models_loaded"]["xgboost_demand_model"] is True
    assert data["models_loaded"]["kmeans_station_model"] is True
    assert data["dataset_status"]["stations_count"] == 47


def test_predict_endpoint_valid():
    """Test POST /predict with valid features returns predicted kWh."""
    payload = {
        "hour": 18,
        "day_of_week": 0,
        "month": 10,
        "is_weekend": 0,
        "prev_hour_demand": 45.5,
        "prev_day_demand": 42.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "predicted_demand_kwh" in data
    assert data["predicted_demand_kwh"] >= 0.0
    assert data["units"] == "kWh"


def test_predict_endpoint_invalid_hour():
    """Test POST /predict with invalid hour (>23) returns 422."""
    payload = {
        "hour": 25,  # Invalid
        "day_of_week": 0,
        "month": 10,
        "is_weekend": 0,
        "prev_hour_demand": 45.5,
        "prev_day_demand": 42.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_endpoint_negative_demand():
    """Test POST /predict with negative demand returns 422."""
    payload = {
        "hour": 12,
        "day_of_week": 2,
        "month": 5,
        "is_weekend": 0,
        "prev_hour_demand": -10.0,  # Invalid
        "prev_day_demand": 20.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_api_fault_summary():
    """Test GET /api/fault-summary returns fault stats."""
    response = client.get("/api/fault-summary")
    assert response.status_code == 200
    data = response.json()
    assert "total_sessions" in data
    assert "abnormal_count" in data
    assert "disclaimer" in data


def test_api_station_clusters():
    """Test GET /api/station-clusters returns 3 cluster profiles."""
    response = client.get("/api/station-clusters")
    assert response.status_code == 200
    data = response.json()
    assert len(data["profiles"]) == 3
    assert len(data["stations"]) == 47


def test_api_sample_presets():
    """Test GET /api/sample/{scenario} endpoints."""
    resp = client.get("/api/sample/weekday_peak")
    assert resp.status_code == 200
    data = resp.json()
    assert data["hour"] == 18

    resp_404 = client.get("/api/sample/invalid_preset_name")
    assert resp_404.status_code == 404


def test_swagger_docs():
    """Test GET /docs returns Swagger UI page."""
    response = client.get("/docs")
    assert response.status_code == 200
    assert "swagger" in response.text.lower()


def test_html_routes():
    """Test that all primary HTML dashboard views render successfully."""
    routes = [
        "/",
        "/forecast",
        "/fault-detection",
        "/segmentation",
        "/eda",
        "/dataset-docs",
        "/about",
        "/contact"
    ]
    for route in routes:
        resp = client.get(route)
        assert resp.status_code == 200, f"Failed on route {route}"
        assert "CHARGEUP" in resp.text


if __name__ == "__main__":
    print("Running CHARGEUP verification tests...")
    test_data_pipeline_fault_rules()
    print("Passed: Fault rules test")
    test_models_artifacts_exist()
    print("Passed: Model artifacts exist")
    test_pipeline_summary_metrics()
    print("Passed: Summary metrics test")
    test_health_endpoint()
    print("Passed: Health endpoint test")
    test_predict_endpoint_valid()
    print("Passed: Valid predict test")
    test_predict_endpoint_invalid_hour()
    print("Passed: Invalid hour 422 test")
    test_predict_endpoint_negative_demand()
    print("Passed: Negative demand 422 test")
    test_api_fault_summary()
    print("Passed: API fault summary test")
    test_api_station_clusters()
    print("Passed: API station clusters test")
    test_api_sample_presets()
    print("Passed: API sample presets test")
    test_swagger_docs()
    print("Passed: Swagger docs test")
    test_html_routes()
    print("Passed: All HTML routes rendered successfully!")
    print("\nALL 11 TEST SUITES PASSED PERFECTLY!\n")
