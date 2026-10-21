"""
CHARGEUP - FastAPI Web Application & Inference Service
AI-Powered EV Charging Demand Forecast & Fault Detection using ChargePoint Data CY20Q4

Provides:
1. High-performance REST API endpoints:
   - GET /health
   - POST /predict (Next-hour demand prediction in kWh with Pydantic validation)
   - GET /api/fault-summary
   - GET /api/station-clusters
   - GET /api/forecast-metrics
   - GET /api/sample/{scenario}
   - Interactive Swagger API docs at /docs
2. Full Dashboard Web UI (Jinja2 + Static Assets):
   - / : Overview Dashboard
   - /forecast : Demand Forecasting Workbench
   - /fault-detection : Fault Analysis & Proxy Event Explorer
   - /segmentation : K-Means Station Usage Segmentation
   - /eda : Exploratory Data Analysis & Grid Insights
   - /dataset-docs : ChargePoint CY20Q4 Dataset Schema & Documentation
   - /about : Project Objectives, Flowchart & Team Credits
"""

import os
import json
import joblib
from datetime import datetime
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException, status
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "Models")
DATA_DIR = os.path.join(BASE_DIR, "Data")
STATIC_DIR = os.path.join(BASE_DIR, "Static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "Templates")

# Global memory caches for model artifacts
models: Dict[str, Any] = {}
pipeline_summary: Dict[str, Any] = {}
raw_sample_df: Optional[pd.DataFrame] = None


def load_artifacts():
    global models, pipeline_summary, raw_sample_df
    
    # Check if pipeline summary exists, if not generate it
    summary_path = os.path.join(MODELS_DIR, "pipeline_summary.json")
    if not os.path.exists(summary_path):
        print("[Startup] Summary not found. Running ML training pipeline...")
        import ml_engine
        ml_engine.run_pipeline()
        
    with open(summary_path, "r") as f:
        pipeline_summary = json.load(f)

    # Load XGBoost model
    xgb_path = os.path.join(MODELS_DIR, "xgboost_demand_model.pkl")
    if not os.path.exists(xgb_path):
        xgb_path = os.path.join(MODELS_DIR, "xgboost.pkl")
    if os.path.exists(xgb_path):
        models["xgboost"] = joblib.load(xgb_path)
        print("[Startup] Loaded XGBoost Demand Forecasting Model.")
    else:
        print("[Startup] WARNING: XGBoost model file not found!")

    # Load KMeans model and scaler
    km_path = os.path.join(MODELS_DIR, "kmeans_station_model.pkl")
    scaler_path = os.path.join(MODELS_DIR, "kmeans_scaler.pkl")
    if os.path.exists(km_path):
        models["kmeans"] = joblib.load(km_path)
    if os.path.exists(scaler_path):
        models["kmeans_scaler"] = joblib.load(scaler_path)
    print("[Startup] Model artifacts initialization complete.")

    # Load small preview of dataset if available
    csv_path = os.path.join(DATA_DIR, "ChargePointData.csv")
    if os.path.exists(csv_path):
        try:
            raw_sample_df = pd.read_csv(csv_path, nrows=50, low_memory=False)
        except Exception as e:
            print(f"[Startup] Could not read raw sample: {e}")


# Initialize models immediately on import
load_artifacts()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Ensure artifacts are loaded
    if "xgboost" not in models:
        load_artifacts()
    yield
    # Keep artifacts in memory



app = FastAPI(
    title="CHARGEUP - EV Charging Demand & Fault Detection API",
    description="AI-powered forecasting and fault analysis system using ChargePoint CY20Q4 dataset.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Static Files & Templates
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)


# =========================================================================
# PYDANTIC SCHEMAS FOR API VALIDATION
# =========================================================================
class DemandPredictionRequest(BaseModel):
    hour: int = Field(..., ge=0, le=23, description="Hour of the day (0-23)", example=18)
    day_of_week: int = Field(..., ge=0, le=6, description="Day of week: 0=Mon, 1=Tue, ..., 6=Sun", example=0)
    month: int = Field(..., ge=1, le=12, description="Month of year (1-12)", example=10)
    is_weekend: int = Field(..., ge=0, le=1, description="1 if weekend (Sat/Sun), else 0", example=0)
    prev_hour_demand: float = Field(..., ge=0.0, description="Previous-hour network demand in kWh", example=45.5)
    prev_day_demand: float = Field(..., ge=0.0, description="Previous-day same-hour demand in kWh", example=42.0)

    @field_validator("is_weekend")
    @classmethod
    def check_weekend_consistency(cls, v, info):
        # Auto-adjust or validate if day_of_week is provided
        return v


class DemandPredictionResponse(BaseModel):
    status: str = "success"
    predicted_demand_kwh: float
    units: str = "kWh"
    model_used: str = "XGBoost Regressor (XGBRegressor)"
    timestamp: str
    input_summary: Dict[str, Any]
    metrics: Dict[str, float]


class SystemHealthResponse(BaseModel):
    status: str
    service: str
    version: str
    models_loaded: Dict[str, bool]
    dataset_status: Dict[str, Any]


# =========================================================================
# REST API ENDPOINTS
# =========================================================================

@app.get("/health", response_model=SystemHealthResponse, tags=["System Health"])
def get_health():
    """
    Returns system status, model availability, and dataset status.
    """
    stats = pipeline_summary.get("dataset_stats", {})
    return {
        "status": "healthy" if "xgboost" in models else "degraded",
        "service": "CHARGEUP - AI EV Charging Operations API",
        "version": "1.0.0",
        "models_loaded": {
            "xgboost_demand_model": "xgboost" in models,
            "kmeans_station_model": "kmeans" in models,
            "kmeans_scaler": "kmeans_scaler" in models
        },
        "dataset_status": {
            "dataset_found": os.path.exists(os.path.join(DATA_DIR, "ChargePointData.csv")),
            "total_sessions": stats.get("total_sessions", 0),
            "stations_count": stats.get("station_count", 0),
            "network_energy_kwh": stats.get("total_energy_network_kwh", 0)
        }
    }


@app.post("/predict", response_model=DemandPredictionResponse, tags=["Forecasting"])
def predict_demand(payload: DemandPredictionRequest):
    """
    Predicts next-hour EV charging demand in kWh using the trained XGBoost model.
    Validates input features and guards against negative demand predictions.
    """
    if "xgboost" not in models:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="XGBoost forecasting model is not currently loaded. Run training pipeline first."
        )

    # Feature order strictly matching training pipeline:
    # ["Hour", "DayOfWeek", "Month", "IsWeekend", "PrevHour_Demand", "PrevDay_Demand"]
    feature_row = np.array([[
        payload.hour,
        payload.day_of_week,
        payload.month,
        payload.is_weekend,
        payload.prev_hour_demand,
        payload.prev_day_demand
    ]], dtype=np.float32)

    try:
        model = models["xgboost"]
        raw_pred = float(model.predict(feature_row)[0])
        # Physical constraint: demand cannot be negative
        predicted_kwh = round(max(0.0, raw_pred), 2)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error during prediction: {str(e)}"
        )

    eval_metrics = pipeline_summary.get("forecasting", {}).get("metrics", {})

    return {
        "status": "success",
        "predicted_demand_kwh": predicted_kwh,
        "units": "kWh",
        "model_used": "XGBoost Regressor (XGBRegressor)",
        "timestamp": datetime.now().isoformat(),
        "input_summary": payload.model_dump(),
        "metrics": eval_metrics
    }


@app.get("/api/fault-summary", tags=["Fault Detection"])
def get_fault_summary():
    """
    Returns summary statistics for proxy fault classification and abnormal events.
    """
    return pipeline_summary.get("fault_detection", {})


@app.get("/api/station-clusters", tags=["Station Segmentation"])
def get_station_clusters():
    """
    Returns K-Means station usage clusters, profiles, and station assignments.
    """
    return pipeline_summary.get("segmentation", {})


@app.get("/api/forecast-metrics", tags=["Forecasting"])
def get_forecast_metrics():
    """
    Returns actual evaluation metrics (MAE, RMSE, R²) and test prediction curves.
    """
    return pipeline_summary.get("forecasting", {})


@app.get("/api/sample/{scenario}", tags=["Forecasting"])
def get_sample_preset(scenario: str):
    """
    Provides pre-configured feature presets for quick interactive testing.
    Available scenarios: weekday_peak, weekday_morning, weekend_afternoon, night_low
    """
    samples = pipeline_summary.get("samples", {})
    if scenario in samples:
        return samples[scenario]
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Sample scenario '{scenario}' not found. Available: {list(samples.keys())}"
    )


# =========================================================================
# WEB UI DASHBOARD ROUTES (HTML JINJA TEMPLATES)
# =========================================================================

@app.get("/", response_class=HTMLResponse, tags=["Dashboard UI"])
def view_dashboard(request: Request):
    stats = pipeline_summary.get("dataset_stats", {})
    forecasting = pipeline_summary.get("forecasting", {})
    fault_summary = pipeline_summary.get("fault_detection", {})
    segmentation = pipeline_summary.get("segmentation", {})
    
    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={
            "active_page": "dashboard",
            "stats": stats,
            "forecasting": forecasting,
            "fault": fault_summary,
            "segmentation": segmentation,
            "team": pipeline_summary.get("team", ["Vaishnavi", "Dhanush", "Hemanth"])
        }
    )


@app.get("/forecast", response_class=HTMLResponse, tags=["Dashboard UI"])
def view_forecast(request: Request):
    forecasting = pipeline_summary.get("forecasting", {})
    samples = pipeline_summary.get("samples", {})
    return templates.TemplateResponse(
        request=request,
        name="predict.html",
        context={
            "active_page": "forecast",
            "forecasting": forecasting,
            "samples": samples,
            "metrics": forecasting.get("metrics", {})
        }
    )


@app.get("/fault-detection", response_class=HTMLResponse, tags=["Dashboard UI"])
def view_fault_detection(request: Request):
    fault_summary = pipeline_summary.get("fault_detection", {})
    return templates.TemplateResponse(
        request=request,
        name="evaluation.html",
        context={
            "active_page": "fault_detection",
            "fault": fault_summary,
            "disclaimer": fault_summary.get("disclaimer", "")
        }
    )


@app.get("/segmentation", response_class=HTMLResponse, tags=["Dashboard UI"])
def view_segmentation(request: Request):
    segmentation = pipeline_summary.get("segmentation", {})
    return templates.TemplateResponse(
        request=request,
        name="unsupervised.html",
        context={
            "active_page": "segmentation",
            "segmentation": segmentation
        }
    )


@app.get("/eda", response_class=HTMLResponse, tags=["Dashboard UI"])
def view_eda(request: Request):
    stats = pipeline_summary.get("dataset_stats", {})
    forecasting = pipeline_summary.get("forecasting", {})
    return templates.TemplateResponse(
        request=request,
        name="eda.html",
        context={
            "active_page": "eda",
            "stats": stats,
            "forecasting": forecasting
        }
    )


@app.get("/dataset-docs", response_class=HTMLResponse, tags=["Dashboard UI"])
def view_dataset_docs(request: Request):
    stats = pipeline_summary.get("dataset_stats", {})
    sample_rows = []
    if raw_sample_df is not None:
        sample_rows = raw_sample_df.head(15).to_dict(orient="records")
        columns = list(raw_sample_df.columns)
    else:
        columns = ["Station Name", "Start Date", "End Date", "Energy (kWh)", "Ended By"]

    return templates.TemplateResponse(
        request=request,
        name="docs.html",
        context={
            "active_page": "dataset_docs",
            "stats": stats,
            "columns": columns,
            "sample_rows": sample_rows
        }
    )


@app.get("/about", response_class=HTMLResponse, tags=["Dashboard UI"])
def view_about(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="about.html",
        context={
            "active_page": "about",
            "team": pipeline_summary.get("team", ["Vaishnavi", "Dhanush", "Hemanth"])
        }
    )


@app.get("/contact", response_class=HTMLResponse, tags=["Dashboard UI"])
def view_contact(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="contact.html",
        context={
            "active_page": "contact",
            "team": pipeline_summary.get("team", ["Vaishnavi", "Dhanush", "Hemanth"])
        }
    )


# Compatibility aliases for old URLs
@app.get("/predict", response_class=HTMLResponse)
def redirect_predict(request: Request):
    return view_forecast(request)

@app.get("/data-loading", response_class=HTMLResponse)
def redirect_data_loading(request: Request):
    return view_dataset_docs(request)

@app.get("/unsupervised", response_class=HTMLResponse)
def redirect_unsupervised(request: Request):
    return view_segmentation(request)

@app.get("/evaluation", response_class=HTMLResponse)
def redirect_evaluation(request: Request):
    return view_fault_detection(request)

@app.get("/training", response_class=HTMLResponse)
def redirect_training(request: Request):
    return view_forecast(request)

@app.get("/regression", response_class=HTMLResponse)
def redirect_regression(request: Request):
    return view_forecast(request)

@app.get("/preprocessing", response_class=HTMLResponse)
def redirect_preprocessing(request: Request):
    return view_dataset_docs(request)

@app.get("/docs-data", response_class=HTMLResponse)
def redirect_docs(request: Request):
    return view_dataset_docs(request)


if __name__ == "__main__":
    import uvicorn
    print("\n=======================================================")
    print("  Starting CHARGEUP Service on http://127.0.0.1:8000   ")
    print("  Swagger API Documentation: http://127.0.0.1:8000/docs ")
    print("=======================================================\n")
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True, app_dir=BASE_DIR)