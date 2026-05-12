import pandas as pd
import numpy as np
import uuid
import os
import logging

def map_severity(types):
    if 'Severe Risk Score' in types or 'Persistent Hotspot' in types:
        return 'Severe'
    if 'Regulatory Violation' in types or 'ML Non-Compliance' in types or 'High Risk Score' in types:
        return 'High'
    if 'Increasing Risk' in types:
        return 'Warning'
    if 'Low Confidence Data' in types:
        return 'Info'
    return 'Safe'

def get_recommended_action(severity, types):
    if severity == 'Severe':
        return 'Escalate for immediate field inspection and source tracking.'
    if 'Regulatory Violation' in types:
        return 'Inspect nearby discharge source and issue warning.'
    if 'ML Non-Compliance' in types:
        return 'Schedule prioritized testing to verify ML prediction.'
    if 'Increasing Risk' in types:
        return 'Monitor closely in next sampling cycle.'
    if 'Low Confidence Data' in types:
        return 'Repeat sampling or repair missing sensor data.'
    return 'No action required.'

def generate_alerts():
    logging.info("Running Alert Engine...")
    df = pd.read_csv('data/processed/nwmp_2025_predictions.csv')
    
    try:
        hotspots_df = pd.read_csv('reports/monitoring/hotspot_summary.csv')
        persistent_stations = hotspots_df[hotspots_df['hotspot_status'] == 'Persistent Hotspot']['station_name'].tolist()
    except FileNotFoundError:
        persistent_stations = []
        
    try:
        inc_df = pd.read_csv('reports/monitoring/stations_increasing_risk.csv')
        increasing_stations = inc_df['station_name'].tolist()
    except FileNotFoundError:
        increasing_stations = []
        
    alerts = []
    
    for idx, row in df.iterrows():
        triggers = []
        
        # A. Regulatory Violation
        invalid_labels = ['Insufficient_Data', 'Unknown', 'Missing', 'NaN', 'nan']
        if pd.notna(row['available_compliance_label']) and row['available_compliance_label'] == 'Non-Compliant':
            triggers.append('Regulatory Violation')
            
        # B. ML Non-Compliance
        if row['predicted_non_compliant'] == 1:
            triggers.append('ML Non-Compliance')
            
        # C & D. Risk Score
        risk = row['risk_score'] if pd.notna(row['risk_score']) else 0
        if risk >= 76:
            triggers.append('Severe Risk Score')
        elif risk >= 51:
            triggers.append('High Risk Score')
            
        # E. Persistent Hotspot
        station = row.get('station_name', 'Unknown')
        if station in persistent_stations:
            triggers.append('Persistent Hotspot')
            
        # F. Increasing Risk
        if station in increasing_stations:
            triggers.append('Increasing Risk')
            
        # G. Low Confidence
        conf = row.get('label_confidence', 'Unknown')
        if conf in ['Low', 'Insufficient']:
            triggers.append('Low Confidence Data')
            
        if triggers:
            severity = map_severity(triggers)
            action = get_recommended_action(severity, triggers)
            
            alerts.append({
                'alert_id': str(uuid.uuid4())[:8],
                'station_name': station,
                'river_name': row.get('river_name', 'Unknown'),
                'district': row.get('district', 'Unknown'),
                'state_name': row.get('state_name', 'Unknown'),
                'sampling_date': row.get('sampling_date', 'Unknown'),
                'month': row.get('month', 'Unknown'),
                'alert_types': " | ".join(triggers),
                'severity': severity,
                'predicted_compliance_label': row.get('predicted_compliance_label', 'Unknown'),
                'predicted_non_compliance_probability': row.get('predicted_non_compliance_probability', np.nan),
                'risk_score': risk,
                'risk_category': row.get('risk_category', 'Unknown'),
                'violation_reasons': row.get('violation_reasons', 'None'),
                'label_confidence': conf,
                'explanation': f"Triggered by: {', '.join(triggers)}",
                'recommended_action': action
            })
            
    alerts_df = pd.DataFrame(alerts)
    
    os.makedirs('reports/monitoring', exist_ok=True)
    if not alerts_df.empty:
        alerts_df.to_csv('reports/monitoring/alerts.csv', index=False)
        
        severe_alerts = alerts_df[alerts_df['severity'].isin(['Severe', 'High'])]
        severe_alerts.to_csv('reports/monitoring/severe_alerts.csv', index=False)
        
        # Summary
        summary = alerts_df.groupby('severity').size().reset_index(name='count')
        summary.to_csv('reports/monitoring/alert_summary.csv', index=False)
        
        logging.info(f"Generated {len(alerts_df)} total alerts, including {len(severe_alerts)} severe/high alerts.")
    else:
        logging.info("No alerts triggered.")
        pd.DataFrame().to_csv('reports/monitoring/alerts.csv', index=False)

if __name__ == "__main__":
    generate_alerts()
