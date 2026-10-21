"""
CHARGEUP - Machine Learning Engine
Implements:
1. Supervised Regression with XGBRegressor for Next-Hour EV Demand Forecasting:
   - Chronological train/test split (80/20).
   - Test evaluation: MAE, RMSE, R².
   - Model artifact serialization.
   - Feature importances and prediction time-series output.
2. Unsupervised Station Segmentation with KMeans:
   - Features: Total Sessions, Total Energy (kWh), Average Energy (kWh).
   - StandardScaler.
   - K-Means with 3 clusters labeled Lower Usage, Medium Usage, Higher Usage.
3. High-resolution visual generation for the CHARGEUP dashboard.
4. Comprehensive pipeline summary generation (JSON) for FastAPI & UI.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

import data_pipeline

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "Models")
STATIC_IMG_DIR = os.path.join(BASE_DIR, "Static", "images")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(STATIC_IMG_DIR, exist_ok=True)

# Styling configuration for publication-ready visual assets
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#e2e8f0"
plt.rcParams["axes.linewidth"] = 0.8


def run_pipeline() -> dict:
    print("=========================================================")
    print("        STARTING CHARGEUP ML TRAINING PIPELINE           ")
    print("=========================================================")
    
    # 1. Load Raw Data and Run Preprocessing
    df_raw = data_pipeline.load_and_clean_raw_data()
    
    # 2. Fault Analysis & Proxy Classification
    df_fault, fault_summary = data_pipeline.classify_fault_events(df_raw)
    
    # 3. Station Statistics for Segmentation
    station_stats, station_meta = data_pipeline.extract_station_segmentation_data(df_raw)
    
    # ---------------------------------------------------------
    # K-MEANS STATION SEGMENTATION
    # ---------------------------------------------------------
    print("[MLEngine] Training K-Means Station Clustering Model...")
    cluster_features = ["Total_Sessions", "Total_Energy_kWh", "Avg_Energy_kWh"]
    X_stations = station_stats[cluster_features].values
    
    scaler_km = StandardScaler()
    X_scaled = scaler_km.fit_transform(X_stations)
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    raw_clusters = kmeans.fit_predict(X_scaled)
    station_stats["Raw_Cluster"] = raw_clusters
    
    # Sort cluster IDs by average total energy to give descriptive labels
    cluster_order = (
        station_stats.groupby("Raw_Cluster")["Total_Energy_kWh"]
        .mean()
        .sort_values()
        .index.tolist()
    )
    # Map raw cluster ID to descriptive label
    label_map = {
        cluster_order[0]: "Lower Usage",
        cluster_order[1]: "Medium Usage",
        cluster_order[2]: "Higher Usage",
    }
    cluster_id_map = {
        cluster_order[0]: 0,
        cluster_order[1]: 1,
        cluster_order[2]: 2,
    }
    
    station_stats["Cluster_ID"] = station_stats["Raw_Cluster"].map(cluster_id_map)
    station_stats["Cluster_Label"] = station_stats["Raw_Cluster"].map(label_map)
    
    # Calculate Cluster Summary Stats
    cluster_profiles = []
    for cid in [0, 1, 2]:
        c_df = station_stats[station_stats["Cluster_ID"] == cid]
        cluster_profiles.append({
            "cluster_id": cid,
            "label": c_df["Cluster_Label"].iloc[0] if len(c_df) > 0 else f"Cluster {cid}",
            "station_count": int(len(c_df)),
            "total_energy_kwh": round(float(c_df["Total_Energy_kWh"].sum()), 2),
            "avg_sessions_per_station": round(float(c_df["Total_Sessions"].mean()), 1) if len(c_df) > 0 else 0,
            "avg_energy_per_station_kwh": round(float(c_df["Total_Energy_kWh"].mean()), 2) if len(c_df) > 0 else 0,
            "avg_kwh_per_session": round(float(c_df["Avg_Energy_kWh"].mean()), 2) if len(c_df) > 0 else 0,
        })
    
    # Save KMeans artifacts
    joblib.dump(kmeans, os.path.join(MODELS_DIR, "kmeans_station_model.pkl"))
    joblib.dump(scaler_km, os.path.join(MODELS_DIR, "kmeans_scaler.pkl"))
    print("[MLEngine] Saved KMeans model and scaler.")

    # ---------------------------------------------------------
    # TIME SERIES DEMAND FORECASTING (XGBOOST)
    # ---------------------------------------------------------
    print("[MLEngine] Preparing Hourly Demand Forecasting Time Series...")
    df_ts = data_pipeline.build_hourly_demand_series(df_raw)
    
    feature_cols = [
        "Hour",
        "DayOfWeek",
        "Month",
        "IsWeekend",
        "PrevHour_Demand",
        "PrevDay_Demand"
    ]
    target_col = "Target_NextHour_Demand"
    
    # Chronological Split (80% Train, 20% Test)
    split_idx = int(len(df_ts) * 0.8)
    train_df = df_ts.iloc[:split_idx].copy()
    test_df = df_ts.iloc[split_idx:].copy()
    
    X_train = train_df[feature_cols].values
    y_train = train_df[target_col].values
    X_test = test_df[feature_cols].values
    y_test = test_df[target_col].values
    
    print(f"[MLEngine] Chronological Split: {len(train_df):,} Train rows, {len(test_df):,} Test rows.")
    print("[MLEngine] Training XGBoost Regressor (n_estimators=200, max_depth=6, lr=0.05)...")
    
    xgb_model = XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        n_jobs=-1
    )
    xgb_model.fit(X_train, y_train)
    
    # Held-out Test Predictions
    y_pred_test = xgb_model.predict(X_test)
    # Ensure physical realism: demand cannot be negative
    y_pred_test = np.clip(y_pred_test, 0.0, None)
    
    mae = float(mean_absolute_error(y_test, y_pred_test))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_test)))
    r2 = float(r2_score(y_test, y_pred_test))
    
    print(f"[MLEngine] XGBoost Test Performance: MAE={mae:.4f} kWh, RMSE={rmse:.4f} kWh, R²={r2:.4f}")
    
    # Save XGBoost artifact
    joblib.dump(xgb_model, os.path.join(MODELS_DIR, "xgboost_demand_model.pkl"))
    # Also save as xgboost.pkl for backwards compatibility
    joblib.dump(xgb_model, os.path.join(MODELS_DIR, "xgboost.pkl"))
    
    # Feature Importances
    feat_importances = {
        col: round(float(imp), 4)
        for col, imp in zip(feature_cols, xgb_model.feature_importances_)
    }

    # Extract recent test window for charting (last 168 hours = 7 days)
    chart_window = 168
    test_sub = test_df.iloc[-chart_window:].copy()
    y_sub_actual = y_test[-chart_window:].tolist()
    y_sub_pred = y_pred_test[-chart_window:].tolist()
    timestamps_sub = test_sub["Timestamp"].dt.strftime("%Y-%m-%d %H:%M").tolist()

    forecast_chart_data = {
        "timestamps": timestamps_sub,
        "actual_kwh": [round(float(v), 2) for v in y_sub_actual],
        "predicted_kwh": [round(float(v), 2) for v in y_sub_pred],
    }

    # Hourly Profile (Average Demand by Hour of Day)
    hourly_profile = df_ts.groupby("Hour")["Demand_kWh"].mean().round(2).to_dict()

    # ---------------------------------------------------------
    # GENERATE VISUAL ASSETS
    # ---------------------------------------------------------
    print("[MLEngine] Generating visual charts for dashboard...")
    
    # 1. Actual vs Predicted Curve (Test Period)
    plt.figure(figsize=(12, 5.5))
    plt.plot(range(len(y_sub_actual)), y_sub_actual, label="Actual Demand (kWh)", color="#0ea5e9", linewidth=2.0)
    plt.plot(range(len(y_sub_pred)), y_sub_pred, label="XGBoost Forecast (kWh)", color="#10b981", linestyle="--", linewidth=2.0)
    plt.title("XGBoost Demand Forecasting: Actual vs. Predicted (Held-Out 7-Day Window)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Hour in Test Window (0 to 168)", fontsize=11)
    plt.ylabel("Next-Hour Demand (kWh)", fontsize=11)
    plt.legend(frameon=True, facecolor="#ffffff", edgecolor="#cbd5e1")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_IMG_DIR, "demand_actual_vs_pred.png"), dpi=150)
    plt.close()

    # 2. Hourly Demand Profile Across 24h
    hours = list(range(24))
    avg_demands = [hourly_profile.get(h, 0.0) for h in hours]
    plt.figure(figsize=(10, 5))
    bars = plt.bar(hours, avg_demands, color="#3b82f6", edgecolor="#1d4ed8", alpha=0.85)
    # Highlight peak hour
    peak_h = int(np.argmax(avg_demands))
    bars[peak_h].set_color("#ef4444")
    bars[peak_h].set_edgecolor("#b91c1c")
    plt.title(f"Average Network EV Charging Demand by Hour of Day (Peak: {peak_h}:00)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Hour of Day (0-23)", fontsize=11)
    plt.ylabel("Average Energy Demand (kWh)", fontsize=11)
    plt.xticks(hours)
    plt.grid(True, linestyle="--", alpha=0.5, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_IMG_DIR, "hourly_demand_profile.png"), dpi=150)
    plt.close()

    # 3. K-Means Station Clustering Scatter
    plt.figure(figsize=(10, 6))
    colors = {"Lower Usage": "#3b82f6", "Medium Usage": "#10b981", "Higher Usage": "#f59e0b"}
    for label, group in station_stats.groupby("Cluster_Label"):
        plt.scatter(
            group["Total_Sessions"],
            group["Total_Energy_kWh"],
            s=group["Avg_Energy_kWh"] * 10,
            label=f"{label} ({len(group)} stations)",
            color=colors.get(label, "#64748b"),
            alpha=0.75,
            edgecolors="#1e293b",
            linewidth=1.2
        )
    plt.title("K-Means Station Segmentation (Usage Clusters)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Total Charging Sessions", fontsize=11)
    plt.ylabel("Total Energy Consumed (kWh)", fontsize=11)
    plt.legend(frameon=True)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_IMG_DIR, "station_cluster_scatter.png"), dpi=150)
    plt.close()

    # 4. Feature Importance Plot
    sorted_fi = sorted(feat_importances.items(), key=lambda x: x[1], reverse=True)
    fi_names = [x[0] for x in sorted_fi]
    fi_vals = [x[1] for x in sorted_fi]
    plt.figure(figsize=(9, 4.5))
    plt.barh(fi_names[::-1], fi_vals[::-1], color="#0ea5e9", edgecolor="#0284c7")
    plt.title("XGBoost Regressor: Feature Importance", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Relative Importance Score", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5, axis="x")
    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_IMG_DIR, "feature_importance.png"), dpi=150)
    plt.close()

    # 5. Fault Events Distribution Chart
    fault_top = {k: v for k, v in list(fault_summary["ended_by_breakdown"].items())[:8]}
    plt.figure(figsize=(10, 5))
    bar_colors = ["#10b981" if k in data_pipeline.NORMAL_ENDED_BY else "#ef4444" for k in fault_top.keys()]
    plt.bar(fault_top.keys(), fault_top.values(), color=bar_colors, edgecolor="#334155")
    plt.title("Charging Session Termination Outcomes ('Ended By' Proxy)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Outcome Classification", fontsize=11)
    plt.ylabel("Session Count (Log Scale)", fontsize=11)
    plt.yscale("log")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_IMG_DIR, "fault_distribution.png"), dpi=150)
    plt.close()

    # 6. Top 10 Stations by Energy Consumption
    top_stations = station_stats.sort_values("Total_Energy_kWh", ascending=False).head(10)
    plt.figure(figsize=(10, 5))
    plt.barh(top_stations["Station Name"][::-1], top_stations["Total_Energy_kWh"][::-1], color="#6366f1", edgecolor="#4338ca")
    plt.title("Top 10 Charging Stations by Total Energy (kWh)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Total Energy Consumed (kWh)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5, axis="x")
    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_IMG_DIR, "station_usage_ranking.png"), dpi=150)
    plt.close()

    # ---------------------------------------------------------
    # COMPILE PIPELINE SUMMARY JSON
    # ---------------------------------------------------------
    stations_list = station_stats[[
        "Station Name", "Total_Sessions", "Total_Energy_kWh", "Avg_Energy_kWh", "Abnormal_Faults", "Availability_Pct", "Health_Status", "City", "Cluster_Label", "Cluster_ID"
    ]].to_dict(orient="records")

    # Sample presets for quick testing on the UI
    sample_presets = {
        "weekday_peak": {
            "hour": 18,
            "day_of_week": 0,
            "month": 10,
            "is_weekend": 0,
            "prev_hour_demand": 45.5,
            "prev_day_demand": 42.0,
            "scenario": "Monday Evening Rush Hour (18:00)"
        },
        "weekday_morning": {
            "hour": 8,
            "day_of_week": 2,
            "month": 6,
            "is_weekend": 0,
            "prev_hour_demand": 28.0,
            "prev_day_demand": 26.5,
            "scenario": "Wednesday Morning Commute (08:00)"
        },
        "weekend_afternoon": {
            "hour": 14,
            "day_of_week": 5,
            "month": 8,
            "is_weekend": 1,
            "prev_hour_demand": 35.0,
            "prev_day_demand": 31.2,
            "scenario": "Saturday Afternoon Leisure (14:00)"
        },
        "night_low": {
            "hour": 3,
            "day_of_week": 3,
            "month": 11,
            "is_weekend": 0,
            "prev_hour_demand": 4.2,
            "prev_day_demand": 5.0,
            "scenario": "Late Night Off-Peak (03:00)"
        }
    }

    summary_data = {
        "project_name": "CHARGEUP",
        "full_title": "AI-Powered EV Charging Demand Forecast & Fault Detection using ChargePoint Data CY20Q4",
        "team": ["Vaishnavi", "Dhanush", "Hemanth"],
        "dataset_stats": {
            "total_sessions": len(df_raw),
            "station_count": len(station_stats),
            "total_energy_network_kwh": round(float(station_stats["Total_Energy_kWh"].sum()), 2),
            "date_range_start": df_ts["Timestamp"].min().strftime("%Y-%m-%d"),
            "date_range_end": df_ts["Timestamp"].max().strftime("%Y-%m-%d"),
            "hourly_records_count": len(df_ts),
        },
        "forecasting": {
            "model_type": "XGBRegressor",
            "features": feature_cols,
            "metrics": {
                "mae": round(mae, 4),
                "rmse": round(rmse, 4),
                "r2": round(r2, 4)
            },
            "feature_importance": feat_importances,
            "test_split_ratio": 0.20,
            "chart_data": forecast_chart_data,
            "hourly_profile": hourly_profile
        },
        "fault_detection": fault_summary,
        "segmentation": {
            "algorithm": "KMeans (k=3)",
            "features": cluster_features,
            "profiles": cluster_profiles,
            "stations": stations_list,
            "availability_summary": station_meta
        },
        "samples": sample_presets,
        "feature_cols": feature_cols
    }

    summary_path = os.path.join(MODELS_DIR, "pipeline_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"[MLEngine] Pipeline summary written to: {summary_path}")

    print("=========================================================")
    print("        CHARGEUP ML PIPELINE COMPLETE SUCCESSFULLY       ")
    print("=========================================================")
    return summary_data


if __name__ == "__main__":
    run_pipeline()
