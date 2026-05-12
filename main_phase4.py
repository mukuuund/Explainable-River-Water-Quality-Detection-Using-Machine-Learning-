import os
import subprocess
import pandas as pd
import logging
import json

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def run_scripts():
    scripts = [
        'src/monitoring/apply_operational_model.py',
        'src/monitoring/operational_validation.py',
        'src/monitoring/hotspot_detection.py',
        'src/monitoring/monthly_trends.py',
        'src/monitoring/alert_engine.py',
        'src/monitoring/spatial_tagging.py',
        'src/monitoring/generate_figures.py'
    ]
    
    for script in scripts:
        logging.info(f"Running {script}...")
        module_name = script.replace('.py', '').replace('/', '.')
        subprocess.run(["python", "-m", module_name], check=True)

def generate_final_report():
    report_path = 'reports/monitoring/phase4_operational_monitoring_summary.md'
    
    # Load required data
    pred_summary = pd.read_csv('reports/monitoring/nwmp_prediction_summary.csv').iloc[0]
    val_summary = pd.read_csv('reports/monitoring/model_rule_agreement_summary.csv').iloc[0]
    
    try:
        hotspot_df = pd.read_csv('reports/monitoring/hotspot_category_distribution.csv')
        persistent_count = hotspot_df[hotspot_df['hotspot_status'] == 'Persistent Hotspot']['count'].sum() if 'Persistent Hotspot' in hotspot_df['hotspot_status'].values else 0
    except:
        persistent_count = 0
        
    try:
        alert_df = pd.read_csv('reports/monitoring/alert_summary.csv')
        total_alerts = alert_df['count'].sum()
        severe_alerts = alert_df[alert_df['severity'].isin(['Severe', 'High'])]['count'].sum()
    except:
        total_alerts = severe_alerts = 0
        
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Phase 4: Operational Monitoring, Hotspot Detection, and Alert Engine\n\n")
        
        f.write("## Objective\n")
        f.write("Apply the leakage-free Phase 3.6 Extended Clean Model to the July-September 2025 NWMP operational dataset to detect hotspots, monitor monthly trends, and trigger automated decision-support alerts.\n\n")
        
        f.write("## Important Limitations & Interpretation Rules\n")
        f.write("> [!IMPORTANT]\n")
        f.write("> - NWMP July-Sep 2025 is recent operational/demo monitoring data, not a long-term forecasting dataset.\n")
        f.write("> - **Alerts are decision-support indicators**, not official regulatory declarations.\n")
        f.write("> - The Risk Score is transparent and rule-based, not a replacement for laboratory validation.\n")
        f.write("> - Full real-time deployment requires live API or sensor integration.\n")
        f.write("> - Model predictions should be interpreted alongside `label_confidence` and `risk_confidence`.\n")
        f.write("> - **Model-Rule Agreement**: When model predictions agree perfectly with rule-based labels, this represents *operational compliance automation*, not independent ground-truth discovery (since the target was derived from the features).\n\n")
        
        f.write("## Data & Artifacts Used\n")
        f.write("- **Input Data**: `data/processed/nwmp_operational_2025.csv`\n")
        f.write("- **Model**: `models/practical_operational_clean_best_model.pkl`\n")
        f.write("- **Features Used**: `models/practical_operational_clean_features.json`\n\n")
        
        f.write("## Prediction Summary\n")
        f.write(f"- Total Rows Processed: {pred_summary['total_rows']}\n")
        f.write(f"- Predicted Non-Compliant: {pred_summary['predicted_non_compliant']}\n")
        f.write(f"- Predicted Compliant: {pred_summary['predicted_compliant']}\n\n")
        
        f.write("## Rule-vs-Model Agreement Summary\n")
        f.write(f"- Total Valid Rule Rows: {val_summary['rows_with_valid_rule_label']}\n")
        f.write(f"- Agreement Rate: {val_summary['agreement_rate']:.2%}\n")
        f.write(f"- Disagreements: {val_summary['disagreement_count']}\n\n")
        
        f.write("## Hotspot & Trend Findings\n")
        f.write(f"- Persistent Hotspots Detected: {persistent_count}\n")
        f.write(f"- View `reports/monitoring/top_20_hotspots.csv` for the most severe locations.\n\n")
        
        f.write("## Alert Summary\n")
        f.write(f"- Total Alerts Triggered: {total_alerts}\n")
        f.write(f"- Severe/High Alerts: {severe_alerts}\n")
        f.write("- View `reports/monitoring/alerts.csv` for actionable field instructions.\n\n")
        
        f.write("## Next Step Recommendation\n")
        f.write("The operational monitoring pipeline has successfully processed the incoming data. The system is now ready for Phase 5 (Explainability & Insights via SHAP) and subsequent deployment to the Streamlit dashboard.\n")

def main():
    logging.info("Starting Phase 4...")
    run_scripts()
    generate_final_report()
    
    # Extract metrics for terminal summary
    df_preds = pd.read_csv('data/processed/nwmp_2025_predictions.csv')
    total_rows = len(df_preds)
    num_stations = df_preds['station_name'].nunique() if 'station_name' in df_preds.columns else 0
    pred_nc = (df_preds['predicted_non_compliant'] == 1).sum()
    
    val_summary = pd.read_csv('reports/monitoring/model_rule_agreement_summary.csv').iloc[0]
    agreement_rate = val_summary['agreement_rate']
    
    hotspots = pd.read_csv('reports/monitoring/hotspot_summary.csv')
    persistent_count = (hotspots['hotspot_status'] == 'Persistent Hotspot').sum()
    
    top_5 = pd.read_csv('reports/monitoring/top_20_hotspots.csv').head(5)['station_name'].tolist()
    
    try:
        inc_stations = len(pd.read_csv('reports/monitoring/stations_increasing_risk.csv'))
    except:
        inc_stations = 0
        
    try:
        alerts = pd.read_csv('reports/monitoring/alert_summary.csv')
        total_alerts = alerts['count'].sum()
        severe_alerts = alerts[alerts['severity'].isin(['Severe', 'High'])]['count'].sum()
    except:
        total_alerts = severe_alerts = 0
        
    with open('models/practical_operational_clean_features.json', 'r') as f:
        features = json.load(f)
        
    variant = "Unknown"
    if 'dissolved_oxygen' in features and 'temperature' in features:
        variant = "Extended Clean Model"
    elif 'dissolved_oxygen' in features:
        variant = "Core Regulatory Model"
    else:
        variant = "True Auxiliary-Only Model"
        
    print("\n" + "="*50)
    print("PHASE 4: OPERATIONAL MONITORING SUMMARY")
    print("="*50)
    
    print(f"Total NWMP rows processed: {total_rows}")
    print(f"Number of stations: {num_stations}")
    print(f"Model feature variant used: {variant}")
    print(f"Predicted non-compliant records: {pred_nc}")
    print(f"Model-rule agreement rate: {agreement_rate:.2%}")
    print(f"Persistent hotspots detected: {persistent_count}")
    print(f"Increasing-risk stations: {inc_stations}")
    print(f"Total alerts generated: {total_alerts}")
    print(f"High/Severe alerts: {severe_alerts}")
    print("\nTop 5 Hotspot Stations:")
    for s in top_5:
        print(f" - {s}")
        
    print("\nOutput Locations:")
    print("- Predictions: data/processed/nwmp_2025_predictions.csv")
    print("- Summaries & Alerts: reports/monitoring/")
    print("- Visualizations: reports/monitoring/figures/")
    print("- Detailed Report: reports/monitoring/phase4_operational_monitoring_summary.md")
    
    print("\nOperational processing complete: ready for Phase 5")

if __name__ == "__main__":
    main()
