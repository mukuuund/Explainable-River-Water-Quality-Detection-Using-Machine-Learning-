import pandas as pd
import json
import os
import numpy as np

RAW_DATA_PATH = "data/raw/realtime/cpcb_realtime_raw_response.json"
METADATA_PATH = "reports/realtime/realtime_api_fetch_metadata.json"
OUTPUT_LONG_PATH = "data/processed/realtime/live_sensor_readings_long.csv"

def standardize_sensor_data():
    """
    Cleans and standardizes live sensor records from the CPCB raw JSON.
    Converts to a long-format DataFrame with canonical columns.
    """
    print("Standardizing sensor data...")
    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError(f"Raw data file not found: {RAW_DATA_PATH}")

    with open(RAW_DATA_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if not data:
        print("No data to process.")
        return pd.DataFrame()

    df = pd.DataFrame(data)

    # Field mapping
    column_mapping = {
        'timestamp': 'timestamp',
        'ts_value': 'value',
        'station_id': 'station_id',
        'station_no': 'station_no',
        'station_name': 'station_name',
        'territory_name': 'state',
        'station_latitude': 'latitude',
        'station_longitude': 'longitude',
        'stationparameter_name': 'parameter_raw_name',
        'stationparameter_no': 'parameter_no',
        'stationparameter_longname': 'parameter_longname',
        'ts_unitsymbol': 'unit',
        'site_no': 'site_no',
        'station_status_remark': 'station_status_remark',
        'station_diary_status': 'station_diary_status'
    }

    # Rename columns, keeping only those we need (and avoiding KeyError if some are missing)
    available_cols = {k: v for k, v in column_mapping.items() if k in df.columns}
    df = df.rename(columns=available_cols)

    # Parse timestamps
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

    # Convert value to numeric
    df['value_numeric'] = pd.to_numeric(df['value'], errors='coerce')
    df['is_numeric_value'] = df['value_numeric'].notna()

    # Determine fetch mode
    fetch_mode = "unknown"
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
            fetch_mode = metadata.get("fetch_mode", "unknown")

    df['source'] = "CPCB_RTWQMS"
    df['fetch_mode'] = fetch_mode

    # Canonical parameter mapping
    def map_parameter(raw_name):
        if pd.isna(raw_name):
            return "unknown"
        name_lower = str(raw_name).strip().lower()
        
        # Careful mapping to avoid false positives
        if name_lower in ['ph', 'p_h']:
            return 'ph'
        elif name_lower in ['do', 'dissolved oxygen', 'dissolved_o2']:
            return 'dissolved_oxygen'
        elif name_lower in ['bod', 'biochemical oxygen demand']:
            return 'bod'
        elif name_lower in ['temperature', 'temp']:
            return 'temperature'
        elif name_lower in ['conductivity', 'conductance', 'ec']:
            return 'conductivity'
        elif name_lower in ['nitrate', 'no3']:
            return 'nitrate'
        elif name_lower in ['turbidity']:
            return 'turbidity'
        elif name_lower in ['cod']:
            return 'cod'
        elif name_lower in ['tds', 'total dissolved solids']:
            return 'total_dissolved_solids'
        elif name_lower in ['fecal coliform', 'faecal coliform']:
            return 'fecal_coliform'
        elif name_lower in ['total coliform']:
            return 'total_coliform'
        
        return name_lower # return original lowercase if no standard mapping

    if 'parameter_raw_name' in df.columns:
        df['parameter'] = df['parameter_raw_name'].apply(map_parameter)
    else:
        df['parameter'] = "unknown"

    os.makedirs(os.path.dirname(OUTPUT_LONG_PATH), exist_ok=True)
    df.to_csv(OUTPUT_LONG_PATH, index=False)
    print(f"Standardized {len(df)} records saved to {OUTPUT_LONG_PATH}")

    return df

if __name__ == "__main__":
    standardize_sensor_data()
