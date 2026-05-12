import pandas as pd
import numpy as np
import os
import json

INPUT_LONG_PATH = "data/processed/realtime/live_sensor_readings_long.csv"
OUTPUT_READY_PATH = "data/processed/realtime/live_sensor_model_ready.csv"
FEATURES_PATH = "models/practical_operational_clean_features.json"

def build_live_features():
    print("Building live features...")
    if not os.path.exists(INPUT_LONG_PATH):
        raise FileNotFoundError(f"Input file not found: {INPUT_LONG_PATH}")

    df = pd.read_csv(INPUT_LONG_PATH)
    if df.empty:
        print("Input dataframe is empty.")
        return pd.DataFrame()

    # Sort to get the latest readings
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df.sort_values(by=['station_id', 'parameter', 'timestamp'])

    # Get latest reading per parameter per station
    latest_df = df.groupby(['station_id', 'parameter']).last().reset_index()
    
    # Identify canonical output columns
    expected_params = [
        'dissolved_oxygen', 'bod', 'ph', 'temperature', 'conductivity', 
        'nitrate', 'fecal_coliform', 'total_coliform', 'fecal_streptococci', 
        'turbidity', 'cod', 'total_dissolved_solids'
    ]

    # Pivot into wide format
    pivot_cols = ['value_numeric']
    pivot_df = latest_df.pivot(index='station_id', columns='parameter', values='value_numeric').reset_index()
    
    # Merge station metadata
    metadata_cols = ['station_id', 'station_no', 'station_name', 'state', 'latitude', 'longitude', 'site_no', 'timestamp']
    # Get latest metadata for each station
    meta_df = df.sort_values(by=['station_id', 'timestamp']).groupby('station_id').last().reset_index()
    meta_df = meta_df[[col for col in metadata_cols if col in meta_df.columns]]
    meta_df = meta_df.rename(columns={'timestamp': 'latest_timestamp'})
    
    wide_df = pd.merge(meta_df, pivot_df, on='station_id', how='left')
    
    # Ensure expected param columns exist
    for param in expected_params:
        if param not in wide_df.columns:
            wide_df[param] = np.nan
            
    # Add other model-ready columns (season, tags, etc.)
    # For a real-time system, we can infer season from latest_timestamp
    def get_season(ts):
        if pd.isna(ts): return 'Unknown'
        month = ts.month
        if month in [3, 4, 5]: return 'Summer'
        elif month in [6, 7, 8, 9]: return 'Monsoon'
        elif month in [10, 11]: return 'Post-Monsoon'
        else: return 'Winter'
        
    wide_df['season'] = wide_df['latest_timestamp'].apply(get_season)
    wide_df['station_position_tag'] = 'Unknown'
    wide_df['pollution_context_tag'] = 'Unknown'
    wide_df['river_name'] = np.nan # Hard to infer reliably without a mapper
    
    # Confidence Logic and Parameters
    core_params = ['dissolved_oxygen', 'bod', 'ph']
    
    # Figure out available vs missing
    def calculate_confidence(row):
        available_params = [p for p in expected_params if pd.notna(row.get(p))]
        missing_params = [p for p in expected_params if pd.isna(row.get(p))]
        
        avail_core = [p for p in core_params if pd.notna(row.get(p))]
        miss_core = [p for p in core_params if pd.isna(row.get(p))]
        
        core_count = len(avail_core)
        if core_count == 3:
            conf = 'Full'
        elif core_count == 2:
            conf = 'Medium'
        elif core_count == 1:
            conf = 'Partial'
        else:
            conf = 'Insufficient'
            
        return pd.Series({
            'available_live_parameters': "|".join(available_params) if available_params else "None",
            'missing_model_features': "|".join(missing_params) if missing_params else "None",
            'core_parameter_count': core_count,
            'available_core_parameters': "|".join(avail_core) if avail_core else "None",
            'missing_core_parameters': "|".join(miss_core) if miss_core else "None",
            'live_data_confidence': conf
        })

    confidence_df = wide_df.apply(calculate_confidence, axis=1)
    wide_df = pd.concat([wide_df, confidence_df], axis=1)
    
    os.makedirs(os.path.dirname(OUTPUT_READY_PATH), exist_ok=True)
    wide_df.to_csv(OUTPUT_READY_PATH, index=False)
    print(f"Model ready data saved to {OUTPUT_READY_PATH}")
    
    return wide_df

if __name__ == "__main__":
    build_live_features()
