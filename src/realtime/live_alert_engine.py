import pandas as pd
import os
import uuid
from datetime import datetime

INPUT_PREDICTIONS_PATH = "data/processed/realtime/live_sensor_predictions_enriched.csv"
OUTPUT_ALERTS_PATH = "reports/realtime/live_alerts.csv"

def generate_live_alerts():
    print("Generating live alerts...")
    # fallback to original if enriched not found
    path = INPUT_PREDICTIONS_PATH
    if not os.path.exists(path):
        path = "data/processed/realtime/live_sensor_predictions.csv"
        if not os.path.exists(path):
            raise FileNotFoundError("Input predictions file not found.")

    df = pd.read_csv(path)
    if df.empty:
        print("Dataframe is empty.")
        return pd.DataFrame()

    alerts = []
    current_time = datetime.utcnow()

    for idx, row in df.iterrows():
        station = row.get('station_name', 'Unknown')
        timestamp = row.get('latest_timestamp', pd.NaT)
        
        # Check staleness (if data is older than 24 hours)
        is_stale = False
        try:
            ts = pd.to_datetime(timestamp)
            if pd.notna(ts):
                if (current_time.tz_localize('UTC') if ts.tzinfo else current_time) - ts > pd.Timedelta(hours=24):
                    is_stale = True
                    alerts.append({
                        'live_alert_id': str(uuid.uuid4()),
                        'timestamp': str(ts),
                        'station_name': station,
                        'alert_type': 'Live Stale Data Alert',
                        'severity': 'Warning',
                        'parameter': 'timestamp',
                        'value': str(ts),
                        'reason': 'Live data is older than 24 hours.',
                        'recommended_action': 'Check station connectivity and API status.'
                    })
        except:
            pass

        # pH Alerts
        ph = row.get('ph')
        if pd.notna(ph):
            if ph < 6.5:
                alerts.append({
                    'live_alert_id': str(uuid.uuid4()),
                    'timestamp': str(timestamp),
                    'station_name': station,
                    'alert_type': 'Live pH Acidic Alert',
                    'severity': 'Warning' if ph >= 6.0 else 'High',
                    'parameter': 'ph',
                    'value': ph,
                    'reason': f'pH is {ph}, below safe minimum of 6.5.',
                    'recommended_action': 'Inspect nearby discharge source.'
                })
            elif ph > 8.5:
                alerts.append({
                    'live_alert_id': str(uuid.uuid4()),
                    'timestamp': str(timestamp),
                    'station_name': station,
                    'alert_type': 'Live pH Alkaline Alert',
                    'severity': 'Warning' if ph <= 9.0 else 'High',
                    'parameter': 'ph',
                    'value': ph,
                    'reason': f'pH is {ph}, above safe maximum of 8.5.',
                    'recommended_action': 'Inspect nearby discharge source.'
                })

        # DO Alert
        do = row.get('dissolved_oxygen')
        if pd.notna(do):
            if do < 5:
                alerts.append({
                    'live_alert_id': str(uuid.uuid4()),
                    'timestamp': str(timestamp),
                    'station_name': station,
                    'alert_type': 'Live DO Alert',
                    'severity': 'High' if do >= 3 else 'Severe',
                    'parameter': 'dissolved_oxygen',
                    'value': do,
                    'reason': f'DO is {do}, below required minimum of 5 mg/L.',
                    'recommended_action': 'Escalate for field verification.'
                })

        # BOD Alert
        bod = row.get('bod')
        if pd.notna(bod):
            if bod > 3:
                alerts.append({
                    'live_alert_id': str(uuid.uuid4()),
                    'timestamp': str(timestamp),
                    'station_name': station,
                    'alert_type': 'Live BOD Alert',
                    'severity': 'High' if bod <= 6 else 'Severe',
                    'parameter': 'bod',
                    'value': bod,
                    'reason': f'BOD is {bod}, above required maximum of 3 mg/L.',
                    'recommended_action': 'Escalate for field verification.'
                })
                
        # Missing Core Parameter Alert
        core_count = row.get('core_parameter_count', 0)
        missing_params = row.get('missing_core_parameters', '')
        if core_count < 3 and core_count > 0: # Has some but not all
            alerts.append({
                'live_alert_id': str(uuid.uuid4()),
                'timestamp': str(timestamp),
                'station_name': station,
                'alert_type': 'Live Missing Core Parameter Alert',
                'severity': 'Info',
                'parameter': 'core_parameters',
                'value': missing_params,
                'reason': f'Missing critical regulatory parameters: {missing_params}.',
                'recommended_action': 'Fetch missing DO/BOD live feeds if available.'
            })
            
        # ML Non-Compliance Alert
        ml_label = row.get('live_ml_predicted_compliance_label')
        if ml_label == 'Non-Compliant':
            prob = row.get('live_ml_non_compliance_probability', 0)
            sev = 'High' if prob > 0.8 else 'Warning'
            alerts.append({
                'live_alert_id': str(uuid.uuid4()),
                'timestamp': str(timestamp),
                'station_name': station,
                'alert_type': 'Live ML Non-Compliance Alert',
                'severity': sev,
                'parameter': 'ML Model',
                'value': f"{prob:.2f}" if pd.notna(prob) else 'Unknown',
                'reason': 'Machine learning model predicts non-compliance based on available features.',
                'recommended_action': 'Repeat sampling to confirm compliance.'
            })
            
        # Normal Alert if nothing triggered
        if not alerts or alerts[-1]['station_name'] != station:
            if core_count == 3 and row.get('live_strict_compliance_label') == 'Compliant':
                alerts.append({
                    'live_alert_id': str(uuid.uuid4()),
                    'timestamp': str(timestamp),
                    'station_name': station,
                    'alert_type': 'Normal Status',
                    'severity': 'Normal',
                    'parameter': 'All',
                    'value': 'OK',
                    'reason': 'All parameters within safe limits.',
                    'recommended_action': 'Continue monitoring.'
                })

    alerts_df = pd.DataFrame(alerts)
    os.makedirs(os.path.dirname(OUTPUT_ALERTS_PATH), exist_ok=True)
    
    if not alerts_df.empty:
        alerts_df.to_csv(OUTPUT_ALERTS_PATH, index=False)
        print(f"Generated {len(alerts_df)} alerts. Saved to {OUTPUT_ALERTS_PATH}")
    else:
        # Create empty with schema
        pd.DataFrame(columns=['live_alert_id', 'timestamp', 'station_name', 'alert_type', 
                              'severity', 'parameter', 'value', 'reason', 'recommended_action']).to_csv(OUTPUT_ALERTS_PATH, index=False)
        print(f"No alerts generated. Empty schema saved to {OUTPUT_ALERTS_PATH}")

    return alerts_df

if __name__ == "__main__":
    generate_live_alerts()
