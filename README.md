# CHARGEUP: AI-Powered EV Charging Demand Forecast & Fault Detection

**AI-Powered EV Charging Demand Forecast & Fault Detection using ChargePoint Data CY20Q4**

> **Team Presenters:** Vaishnavi, Dhanush, Hemanth  
> **Course / Project:** AI & Machine Learning Capstone  
> **Architecture:** FastAPI REST Backend + Jinja2 Responsive Operations Dashboard + XGBoost Regressor + K-Means Clustering

---

## 1. Project Overview & Identity

CHARGEUP is an intelligent operations platform for electric vehicle (EV) charging infrastructure. It leverages real-world telemetry from the **ChargePoint CY20Q4** dataset (259,410 sessions across 47 physical charging stations) to solve three critical operational challenges:

1. **Next-Hour Charging Demand Forecasting (Supervised Regression):**  
   Predicts next-hour energy demand ($t+1$) in kilowatt-hours (kWh) for the entire charging network using an `XGBRegressor` model. Uses chronological feature engineering (no future data leakage).
2. **Proxy Fault-Event Detection (Operational Classification):**  
   Uses the session termination code (`Ended By`) as a proxy indicator to flag abnormal charging disruptions (e.g., *Station Offline*, *Outlet Unreachable*, *GFCI Ground-Fault Tripped*, *Plug Removed While Rebooting*).
3. **Station Usage Segmentation (Unsupervised Clustering):**  
   Clusters charging stations into 3 distinct operational tiers (*Lower Usage*, *Medium Usage*, *Higher Usage*) using `StandardScaler` and `KMeans(k=3)`.
4. **FastAPI High-Performance Service & Dashboard:**  
   Exposes validated REST API endpoints (`/health`, `/predict`, `/api/...`) with automatic interactive Swagger documentation at `/docs`, accompanied by a responsive operations dashboard.

---

## 2. Project Directory Structure

```text
ChargeUp_EV/
├── app.py                      # FastAPI web application, routes & Pydantic request models
├── main.py                     # Entry point for running the Uvicorn application server
├── data_pipeline.py            # ChargePoint data ingestion, cleaning, time series aggregation
├── ml_engine.py                # ML training pipeline (XGBoost, KMeans, evaluation, charts)
├── run_unsupervised_pipeline.py# Unsupervised segmentation runner script
├── test_chargeup.py            # Comprehensive 11-suite integration & unit test runner
├── requirements.txt            # Python package dependencies
├── .gitignore                  # Git ignore rules for datasets, venv, and cache
├── README.md                   # Complete system documentation & user manual
│
├── Data/
│   └── ChargePointData.csv     # ChargePoint CY20Q4 dataset (259,410 records, 85.4 MB)
│
├── Models/
│   ├── xgboost_demand_model.pkl# Trained XGBoost Regressor for next-hour demand
│   ├── xgboost.pkl             # Model alias
│   ├── kmeans_station_model.pkl# Trained K-Means clustering model (k=3)
│   ├── kmeans_scaler.pkl       # Fitted StandardScaler for station segmentation
│   └── pipeline_summary.json   # JSON summary: metrics, profiles, fault stats, sample inputs
│
├── Static/
│   ├── index.css               # CHARGEUP Design System stylesheet (slate & neon accents)
│   ├── app.js                  # Client-side dynamic interaction, presets, search & live API calls
│   └── images/
│       ├── demand_actual_vs_pred.png   # Held-out 7-day actual vs predicted load curve
│       ├── hourly_demand_profile.png   # 24-hour diurnal network load profile
│       ├── station_cluster_scatter.png # K-Means cluster scatter (sessions vs energy)
│       ├── feature_importance.png      # XGBoost feature importance ranking
│       ├── fault_distribution.png      # Ended By termination distribution (log scale)
│       └── station_usage_ranking.png   # Top 10 charging stations by energy delivered
│
└── Templates/
    ├── base.html               # Shared layout, sidebar navigation & Lucide icons
    ├── dashboard.html          # Overview dashboard: KPIs, quick calculator, charts
    ├── predict.html            # Demand forecasting workbench with presets & live API
    ├── evaluation.html         # Fault-event detection, proxy disclaimer & session table
    ├── unsupervised.html       # Station segmentation (K-Means), cluster cards, roster
    ├── eda.html                # Exploratory Data Analysis & grid load patterns
    ├── docs.html               # ChargePoint CY20Q4 schema & dataset docs
    ├── preprocessing.html      # 5-stage data preprocessing pipeline architecture
    ├── data_loading.html       # Dataset loading verification
    ├── regression.html         # Supervised regression summary
    ├── training.html           # Model training hyperparameter specifications
    ├── about.html              # Team credits (Vaishnavi, Dhanush, Hemanth) & methodology
    └── contact.html            # Presentation inquiries & repository information
```

---

## 3. Dataset Integration: ChargePoint CY20Q4

### Official Dataset Description
- **Records:** 259,415 raw session events (259,410 valid cleaned sessions).
- **Timeframe:** July 2011 to December 2020 (Calendar Year 2020 Quarter 4).
- **Stations:** 47 physical Level 2 charging stations in Palo Alto, California.
- **Total Energy:** 2,216,479.12 kWh (~2.22 GWh).

### Required Columns
| Column Name | Type | Operational Function |
|---|---|---|
| `Station Name` | String | Station identifier for grouping and clustering |
| `Start Date` | Datetime | Session start timestamp (mixed format parsing) |
| `End Date` | Datetime | Session end timestamp |
| `Energy (kWh)` | Float | Delivered electrical energy (filtered to > 0 kWh) |
| `Ended By` | String | Termination reason used for proxy fault detection |

### Dataset Placement
Place the dataset file in the `Data/` directory:
```bash
Data/ChargePointData.csv
```
If updating the dataset, place the new CSV into `Data/ChargePointData.csv` and re-run `python ml_engine.py`.

---

## 4. Machine Learning & Methodology

### 4.1 Next-Hour Demand Forecasting (`XGBRegressor`)
- **Hourly Aggregation:** Sessions are grouped chronologically by hour; gaps are zero-filled to maintain a continuous, regular time series across 82,606 hours.
- **Engineered Features (Zero Leakage):**
  1. `Hour` (0–23): Diurnal commuter load cycles.
  2. `DayOfWeek` (0–6): Distinguishes Monday–Friday commercial traffic from weekends.
  3. `Month` (1–12): Seasonal ambient temperature fluctuations.
  4. `IsWeekend` (0/1): Binary indicator for non-working days.
  5. `PrevHour_Demand` ($t-1$ lag): Short-term charging session persistence.
  6. `PrevDay_Demand` ($t-24$ lag): 24-hour seasonal cycle (highest importance: 57.3% gain).
  - **Target:** `Target_NextHour_Demand` ($t+1$ energy in kWh).
- **Chronological Split:** 80% Train (66,084 hours) / 20% Test (16,522 hours). *No random shuffling is used to prevent future lookahead leakage.*
- **Model Evaluation on Held-Out Test Data:**
  - **MAE (Mean Absolute Error):** `19.0810 kWh`
  - **RMSE (Root Mean Squared Error):** `29.9400 kWh`
  - **$R^2$ (Coefficient of Determination):** `0.5738`

### 4.2 Fault-Event Detection (`Ended By` Proxy)
- **Normal Terminations (`FAULT = 0`):** `Plug Out at Vehicle`, `Customer`, `CPS Server`, `Plug Out at Station`, `Holster Plugin`, `Door Closed` (Total: 258,885 sessions).
- **Abnormal Outages (`FAULT = 1`):** `Station Offline`, `Outlet Unreachable`, `Relay Stuck Closed`, `Final GFCI Tripped`, `Final 4-strikes GFCI trip`, `Plug Removed While Rebooting`, `Door Open`, `Door`, `OCD` (Total: 244 sessions, 0.094% fault rate).
- **Unknown Terminations (`Unknown`):** Preserved without arbitrary labeling (281 sessions).
- **Limitation Disclosure:** *This module operates as an operational proxy based on session outcome telemetry. It does not replace physical IoT sensor diagnostics.*

### 4.3 Station Usage Segmentation (`KMeans`)
- **Features Standardized via `StandardScaler`:**
  1. `Total_Sessions` (session count per charger)
  2. `Total_Energy_kWh` (cumulative kilowatt-hours delivered)
  3. `Avg_Energy_kWh` (mean energy per charging event)
- **Clusters ($k=3$):**
  - **Cluster 0 — Lower Usage:** 4 stations | Total 275.13 kWh | Avg 3.5 sessions/station | Backup or low-density chargers.
  - **Cluster 1 — Medium Usage:** 27 stations | Total 606,949.49 kWh | Avg 2,512.9 sessions/station | Standard municipal and suburban chargers.
  - **Cluster 2 — Higher Usage:** 16 stations | Total 1,609,254.50 kWh | Avg 11,971.7 sessions/station | Heavily utilized downtown and highway corridor hubs.

---

## 5. Installation & Setup Instructions

### 5.1 Prerequisites
- Python 3.10+ (tested on Python 3.14)
- Pip package manager

### 5.2 Install Dependencies
```bash
pip install -r requirements.txt
```

### 5.3 Run ML Pipeline (Feature Engineering & Training)
```bash
python ml_engine.py
```
This script will:
- Ingest and clean `Data/ChargePointData.csv`
- Compute proxy fault classification statistics
- Train and persist `Models/kmeans_station_model.pkl` and `Models/kmeans_scaler.pkl`
- Build the 82,606-hour demand time series and train `Models/xgboost_demand_model.pkl`
- Calculate real test evaluation metrics (MAE, RMSE, $R^2$)
- Generate all visual assets in `Static/images/`
- Output `Models/pipeline_summary.json`

### 5.4 Launch Application Server
```bash
python main.py
# or directly with uvicorn:
uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```
Open your browser to:
- **Web Dashboard:** [http://127.0.0.1:8000](http://127.0.0.1:8000)
- **Demand Forecasting Workbench:** [http://127.0.0.1:8000/forecast](http://127.0.0.1:8000/forecast)
- **Fault Event Explorer:** [http://127.0.0.1:8000/fault-detection](http://127.0.0.1:8000/fault-detection)
- **Station Segmentation:** [http://127.0.0.1:8000/segmentation](http://127.0.0.1:8000/segmentation)
- **Interactive Swagger REST API:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 6. REST API Reference

### 6.1 `GET /health`
Returns service status, model availability, and dataset status.
```json
{
  "status": "healthy",
  "service": "CHARGEUP - AI EV Charging Operations API",
  "version": "1.0.0",
  "models_loaded": {
    "xgboost_demand_model": true,
    "kmeans_station_model": true,
    "kmeans_scaler": true
  },
  "dataset_status": {
    "dataset_found": true,
    "total_sessions": 259410,
    "stations_count": 47,
    "network_energy_kwh": 2216479.12
  }
}
```

### 6.2 `POST /predict`
Predicts next-hour demand in kWh. Validates inputs using Pydantic.

**Request Body:**
```json
{
  "hour": 18,
  "day_of_week": 0,
  "month": 10,
  "is_weekend": 0,
  "prev_hour_demand": 45.5,
  "prev_day_demand": 42.0
}
```

**Response (HTTP 200 OK):**
```json
{
  "status": "success",
  "predicted_demand_kwh": 42.37,
  "units": "kWh",
  "model_used": "XGBoost Regressor (XGBRegressor)",
  "timestamp": "2026-10-09T22:46:13.900140",
  "input_summary": {
    "hour": 18,
    "day_of_week": 0,
    "month": 10,
    "is_weekend": 0,
    "prev_hour_demand": 45.5,
    "prev_day_demand": 42.0
  },
  "metrics": {
    "mae": 19.081,
    "rmse": 29.94,
    "r2": 0.5738
  }
}
```

### 6.3 Additional API Endpoints
- `GET /api/fault-summary`: Fault breakdown, normal vs abnormal counts, top fault stations.
- `GET /api/station-clusters`: 3 cluster profiles and complete 47 station assignments.
- `GET /api/forecast-metrics`: Test evaluation metrics and 168-hour actual vs predicted test points.
- `GET /api/sample/{scenario}`: Pre-configured test scenarios (`weekday_peak`, `weekday_morning`, `weekend_afternoon`, `night_low`).

---

## 7. Verification & Automated Tests

Run the comprehensive test suite:
```bash
python test_chargeup.py
```

### Test Results
| Test Suite | Purpose | Status |
|---|---|---|
| `test_data_pipeline_fault_rules` | Validates normal vs abnormal termination mappings | **PASSED** |
| `test_models_artifacts_exist` | Confirms all `.pkl` and `.json` model files exist | **PASSED** |
| `test_pipeline_summary_metrics` | Validates held-out evaluation metrics ($R^2 > 0$, MAE $> 0$) | **PASSED** |
| `test_health_endpoint` | Verifies `GET /health` returns 200 and healthy models | **PASSED** |
| `test_predict_endpoint_valid` | Tests `POST /predict` returns non-negative kWh | **PASSED** |
| `test_predict_endpoint_invalid_hour` | Verifies hour $> 23$ returns HTTP 422 | **PASSED** |
| `test_predict_endpoint_negative_demand`| Verifies demand $< 0$ returns HTTP 422 | **PASSED** |
| `test_api_fault_summary` | Verifies fault counts and disclaimer schema | **PASSED** |
| `test_api_station_clusters` | Verifies 3 clusters and 47 stations returned | **PASSED** |
| `test_api_sample_presets` | Tests valid scenario presets & 404 handling | **PASSED** |
| `test_swagger_docs` | Confirms `/docs` Swagger UI is reachable | **PASSED** |
| `test_html_routes` | Verifies all 8 web templates render without error | **PASSED** |

---

## 8. Summary of Project Transformation

| Component | Original Project (Fraud Shield) | Transformed Project (CHARGEUP) | Status |
|---|---|---|---|
| **Domain** | Credit card fraud detection | EV charging demand forecast & fault detection | Replaced |
| **Dataset** | `creditcard.csv` (284k anonymous transactions) | `ChargePointData.csv` (259k EV charging sessions) | Replaced |
| **Backend** | Flask web app | FastAPI ASGI high-performance REST service | Replaced |
| **API Docs** | None | Interactive Swagger UI at `/docs` | Added |
| **Forecasting** | Simple Linear Regression | Supervised `XGBRegressor` next-hour demand | Replaced |
| **Clustering** | Unsupervised fraud anomaly detection | 3-tier K-Means station usage segmentation | Replaced |
| **Fault Detection** | Binary fraud labels (0/1) | Proxy fault classification using `Ended By` | Replaced |
| **Frontend UI** | Static financial fraud tables | Responsive EV Operations Dashboard (Slate/Neon) | Replaced |
| **Interactive JS** | Basic form submit | Real-time fetch to `/predict`, preset toggles, search | Added |
| **Original Code** | Preserved intact | Archived in `Credit_Crad_Fraud_Detection_BACKUP` | Preserved |

---

## 9. Presentation Credits & Acknowledgements
- **Vaishnavi** — Supervised Machine Learning & Forecasting Models
- **Dhanush** — FastAPI Backend Services, Architecture & UI Engineering
- **Hemanth** — Data Pipeline Engineering & Unsupervised Station Clustering
