"""
CHARGEUP - Data Pipeline & Feature Engineering
Dataset: ChargePoint Data CY20Q4 (~259k sessions across 47 stations)
Handles:
1. Loading, cleaning, and preprocessing charging sessions.
2. Hourly aggregation of network charging demand in kWh.
3. Feature engineering for next-hour demand forecasting:
   - Hour (0-23)
   - Day of week (0-6)
   - Month (1-12)
   - Weekend indicator (0 or 1)
   - Previous-hour demand (kWh)
   - Previous-day demand (kWh, 24-hr lag)
   - Target: Next-hour demand (kWh)
4. Fault-event proxy labeling based on 'Ended By' outcome.
5. Station-level aggregation for K-Means segmentation:
   - Total sessions
   - Total energy consumed (kWh)
   - Average energy per session (kWh)
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any

# Configurable data path
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data")
DEFAULT_CSV_PATH = os.path.join(DATA_DIR, "ChargePointData.csv")

# Fault classification dictionary
NORMAL_ENDED_BY = {
    "Plug Out at Vehicle",
    "Customer",
    "CPS Server",
    "Plug Out at Station",
    "Holster Plugin",
}

ABNORMAL_ENDED_BY = {
    "Station Offline",
    "Outlet Unreachable",
    "Relay Stuck Closed",
    "Final GFCI Tripped",
    "Final 4-strikes GFCI trip",
    "Plug Removed While Rebooting",
    "Door Open",
    "Door",
    "OCD",
}


def load_and_clean_raw_data(csv_path: str = DEFAULT_CSV_PATH) -> pd.DataFrame:
    """
    Loads raw ChargePoint session data, handles missing values,
    parses timestamps, cleans numeric energy field.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"ChargePoint dataset not found at {csv_path}")

    print(f"[DataPipeline] Loading dataset from: {csv_path}")
    cols_to_read = [
        "Station Name", "Start Date", "End Date", "Energy (kWh)",
        "Ended By", "Total Duration (hh:mm:ss)", "Charging Time (hh:mm:ss)",
        "Port Type", "City", "State/Province"
    ]
    
    # Read CSV
    df = pd.read_csv(csv_path, usecols=lambda c: c in cols_to_read, low_memory=False)
    initial_rows = len(df)
    
    # Clean duplicates
    df = df.drop_duplicates()
    
    # Parse datetimes safely
    df["Start Date"] = pd.to_datetime(df["Start Date"], format="mixed", errors="coerce")
    df["End Date"] = pd.to_datetime(df["End Date"], format="mixed", errors="coerce")
    
    # Clean Energy
    df["Energy (kWh)"] = pd.to_numeric(df["Energy (kWh)"], errors="coerce")
    
    # Filter valid sessions: valid Start Date and positive Energy (> 0 kWh)
    df = df.dropna(subset=["Start Date", "Energy (kWh)"])
    df = df[df["Energy (kWh)"] > 0]
    
    # Clean Ended By
    df["Ended By"] = df["Ended By"].fillna("Unknown").astype(str).str.strip()
    
    print(f"[DataPipeline] Loaded {len(df):,} valid sessions (from {initial_rows:,} raw records).")
    return df


def classify_fault_events(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Applies proxy-based fault classification using the 'Ended By' field.
    Assigns FAULT=0 for normal, FAULT=1 for abnormal, 'Unknown' for unknown.
    """
    print("[DataPipeline] Computing fault classification metrics...")
    
    def get_fault_label(ended_by: str) -> str:
        if ended_by in NORMAL_ENDED_BY:
            return "Normal"
        elif ended_by in ABNORMAL_ENDED_BY:
            return "Abnormal"
        else:
            return "Unknown"

    df_fault = df.copy()
    df_fault["Fault_Status"] = df_fault["Ended By"].apply(get_fault_label)
    df_fault["Fault_Flag"] = df_fault["Fault_Status"].apply(
        lambda s: 1 if s == "Abnormal" else (0 if s == "Normal" else -1)
    )

    counts = df_fault["Fault_Status"].value_counts().to_dict()
    total = len(df_fault)
    abnormal_count = counts.get("Abnormal", 0)
    normal_count = counts.get("Normal", 0)
    unknown_count = counts.get("Unknown", 0)

    # Detailed breakdown of Ended By
    ended_by_breakdown = df_fault["Ended By"].value_counts().head(20).to_dict()

    # Station-level fault counts
    station_faults = (
        df_fault[df_fault["Fault_Flag"] == 1]
        .groupby("Station Name")
        .size()
        .sort_values(ascending=False)
        .head(10)
        .to_dict()
    )

    # Sample abnormal records for UI display
    abnormal_samples = (
        df_fault[df_fault["Fault_Flag"] == 1][
            ["Station Name", "Start Date", "End Date", "Energy (kWh)", "Ended By", "City"]
        ]
        .head(50)
        .assign(
            StartDateStr=lambda x: x["Start Date"].dt.strftime("%Y-%m-%d %H:%M"),
            EndDateStr=lambda x: x["End Date"].dt.strftime("%Y-%m-%d %H:%M")
        )
        .drop(columns=["Start Date", "End Date"])
        .to_dict(orient="records")
    )

    summary = {
        "total_sessions": total,
        "normal_count": normal_count,
        "abnormal_count": abnormal_count,
        "unknown_count": unknown_count,
        "fault_rate_pct": round((abnormal_count / total) * 100, 3) if total > 0 else 0.0,
        "ended_by_breakdown": ended_by_breakdown,
        "top_fault_stations": station_faults,
        "abnormal_samples": abnormal_samples,
        "disclaimer": (
            "Proxy-Based Indicator: 'Ended By' termination code is used as an operational "
            "fault proxy. It does not replace physical hardware sensor telemetry or verified failure diagnosis."
        )
    }

    return df_fault, summary


def build_hourly_demand_series(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates charging session energy into continuous hourly demand.
    Engineers temporal and lag features strictly avoiding target leakage.
    """
    print("[DataPipeline] Aggregating charging sessions into hourly time series...")
    
    # Floor timestamp to hour
    df_ts = df[["Start Date", "Energy (kWh)"]].copy()
    df_ts["Hour_Timestamp"] = df_ts["Start Date"].dt.floor("h")

    # Aggregate total energy demand per hour
    hourly = df_ts.groupby("Hour_Timestamp")["Energy (kWh)"].sum().reset_index()
    hourly.columns = ["Timestamp", "Demand_kWh"]
    hourly = hourly.sort_values("Timestamp")

    # Filter to main active period (e.g. 2018 onwards or full recent continuous span)
    # The dataset spans 2011 to 2020. Using 2018-2020 gives a dense, regular continuous series
    min_date = hourly["Timestamp"].min()
    max_date = hourly["Timestamp"].max()
    print(f"[DataPipeline] Raw hourly bounds: {min_date} to {max_date}")

    # Reindex to continuous hourly grid
    full_idx = pd.date_range(start=min_date, end=max_date, freq="h", name="Timestamp")
    hourly = hourly.set_index("Timestamp").reindex(full_idx, fill_value=0.0).reset_index()

    # Feature Engineering (strictly using data known at prediction time)
    hourly["Hour"] = hourly["Timestamp"].dt.hour
    hourly["DayOfWeek"] = hourly["Timestamp"].dt.dayofweek  # 0=Monday, 6=Sunday
    hourly["Month"] = hourly["Timestamp"].dt.month
    hourly["IsWeekend"] = hourly["DayOfWeek"].apply(lambda d: 1 if d in [5, 6] else 0)

    # Lags: Previous-hour demand and Previous-day demand (24 hours prior)
    hourly["PrevHour_Demand"] = hourly["Demand_kWh"].shift(1)
    hourly["PrevDay_Demand"] = hourly["Demand_kWh"].shift(24)

    # Target: Next-Hour Demand (demand at t+1)
    hourly["Target_NextHour_Demand"] = hourly["Demand_kWh"].shift(-1)

    # Drop edge rows with NaN from lags or lead
    df_features = hourly.dropna().copy().reset_index(drop=True)
    print(f"[DataPipeline] Created {len(df_features):,} hourly demand feature records.")

    return df_features


def extract_station_segmentation_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Aggregates usage and availability statistics per station:
    - Total sessions
    - Total energy (kWh)
    - Average energy per session (kWh)
    - Operational availability percentage & health status
    """
    print("[DataPipeline] Extracting station usage and availability statistics...")
    
    # Check if Fault_Flag exists
    if "Fault_Flag" not in df.columns:
        df_work, _ = classify_fault_events(df)
    else:
        df_work = df

    station_stats = df_work.groupby("Station Name").agg(
        Total_Sessions=("Energy (kWh)", "count"),
        Total_Energy_kWh=("Energy (kWh)", "sum"),
        Avg_Energy_kWh=("Energy (kWh)", "mean"),
        Abnormal_Faults=("Fault_Flag", lambda s: int((s == 1).sum())),
        City=("City", "first")
    ).reset_index()

    # Round figures
    station_stats["Total_Energy_kWh"] = station_stats["Total_Energy_kWh"].round(2)
    station_stats["Avg_Energy_kWh"] = station_stats["Avg_Energy_kWh"].round(2)
    
    # Compute station availability rate
    station_stats["Availability_Pct"] = (
        100.0 * (1.0 - (station_stats["Abnormal_Faults"] / station_stats["Total_Sessions"]))
    ).round(2)

    def get_health_label(faults: int) -> str:
        if faults == 0:
            return "Optimal"
        elif faults < 5:
            return "Monitored"
        else:
            return "Attention Needed"

    station_stats["Health_Status"] = station_stats["Abnormal_Faults"].apply(get_health_label)

    optimal_cnt = int((station_stats["Health_Status"] == "Optimal").sum())
    monitored_cnt = int((station_stats["Health_Status"] == "Monitored").sum())
    attention_cnt = int((station_stats["Health_Status"] == "Attention Needed").sum())
    network_uptime = round(100.0 * (1.0 - (station_stats["Abnormal_Faults"].sum() / station_stats["Total_Sessions"].sum())), 2)

    stats_summary = {
        "station_count": len(station_stats),
        "total_energy_network": round(float(station_stats["Total_Energy_kWh"].sum()), 2),
        "avg_energy_per_station": round(float(station_stats["Total_Energy_kWh"].mean()), 2),
        "avg_sessions_per_station": round(float(station_stats["Total_Sessions"].mean()), 1),
        "network_uptime_pct": network_uptime,
        "optimal_stations_count": optimal_cnt,
        "monitored_stations_count": monitored_cnt,
        "attention_stations_count": attention_cnt,
    }

    return station_stats, stats_summary


if __name__ == "__main__":
    df_raw = load_and_clean_raw_data()
    _, fault_meta = classify_fault_events(df_raw)
    print("Fault summary:", {k: v for k, v in fault_meta.items() if k != "abnormal_samples"})
    df_ts = build_hourly_demand_series(df_raw)
    print("Hourly feature head:\n", df_ts.head())
    st_df, st_meta = extract_station_segmentation_data(df_raw)
    print("Station stats summary:", st_meta)
