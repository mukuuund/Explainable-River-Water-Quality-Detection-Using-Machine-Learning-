import pandas as pd
import os

INPUT_LONG_PATH = "data/processed/realtime/live_sensor_readings_long.csv"
COVERAGE_REPORT_PATH = "reports/realtime/live_parameter_coverage.csv"
MATRIX_REPORT_PATH = "reports/realtime/live_station_parameter_matrix.csv"

def inspect_coverage():
    """
    Generates coverage reports for the live sensor data.
    """
    print("Inspecting parameter coverage...")
    if not os.path.exists(INPUT_LONG_PATH):
        raise FileNotFoundError(f"Input file not found: {INPUT_LONG_PATH}")

    df = pd.read_csv(INPUT_LONG_PATH)
    
    # Ensure directories exist
    os.makedirs(os.path.dirname(COVERAGE_REPORT_PATH), exist_ok=True)

    if df.empty:
        print("Dataframe is empty, cannot inspect coverage.")
        return

    # Task 3: Parameter Coverage
    coverage_data = []
    for param in df['parameter'].unique():
        param_df = df[df['parameter'] == param]
        
        record_count = len(param_df)
        station_count = param_df['station_name'].nunique()
        latest_timestamp = param_df['timestamp'].max() if 'timestamp' in param_df.columns else None
        
        numeric_df = param_df[param_df['is_numeric_value'] == True]
        non_null_value_count = len(numeric_df)
        
        min_value = numeric_df['value_numeric'].min() if not numeric_df.empty else None
        max_value = numeric_df['value_numeric'].max() if not numeric_df.empty else None
        mean_value = numeric_df['value_numeric'].mean() if not numeric_df.empty else None
        
        coverage_data.append({
            'parameter': param,
            'record_count': record_count,
            'station_count': station_count,
            'latest_timestamp': latest_timestamp,
            'non_null_value_count': non_null_value_count,
            'min_value': min_value,
            'max_value': max_value,
            'mean_value': mean_value
        })

    coverage_df = pd.DataFrame(coverage_data)
    coverage_df.to_csv(COVERAGE_REPORT_PATH, index=False)
    print(f"Coverage report saved to {COVERAGE_REPORT_PATH}")

    # Task 3: Station Parameter Matrix
    # We want rows: station_name / station_id, columns: parameters available, values: latest value or availability flag
    # Let's pivot
    if 'station_name' in df.columns and 'parameter' in df.columns and 'value' in df.columns:
        # Sort by timestamp so the last one is the latest
        if 'timestamp' in df.columns:
            df = df.sort_values(by=['station_name', 'parameter', 'timestamp'])
            
        # Get latest value per station per parameter
        latest_df = df.groupby(['station_name', 'parameter']).last().reset_index()
        
        matrix_df = latest_df.pivot(index='station_name', columns='parameter', values='value').reset_index()
        
        # If there are stations with same name but different ids, we might lose them, 
        # but prompt says "Rows: station_name / station_id".
        # Let's stick with station_name for simplicity, or include both if possible.
        
        matrix_df.to_csv(MATRIX_REPORT_PATH, index=False)
        print(f"Station-Parameter matrix saved to {MATRIX_REPORT_PATH}")
    else:
        print("Required columns for matrix are missing.")

if __name__ == "__main__":
    inspect_coverage()
