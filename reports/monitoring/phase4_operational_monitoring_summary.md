# Phase 4: Operational Monitoring, Hotspot Detection, and Alert Engine

## Objective
Apply the leakage-free Phase 3.6 Extended Clean Model to the July-September 2025 NWMP operational dataset to detect hotspots, monitor monthly trends, and trigger automated decision-support alerts.

## Important Limitations & Interpretation Rules
> [!IMPORTANT]
> - NWMP July-Sep 2025 is recent operational/demo monitoring data, not a long-term forecasting dataset.
> - **Alerts are decision-support indicators**, not official regulatory declarations.
> - The Risk Score is transparent and rule-based, not a replacement for laboratory validation.
> - Full real-time deployment requires live API or sensor integration.
> - Model predictions should be interpreted alongside `label_confidence` and `risk_confidence`.
> - **Model-Rule Agreement**: When model predictions agree perfectly with rule-based labels, this represents *operational compliance automation*, not independent ground-truth discovery (since the target was derived from the features).

## Data & Artifacts Used
- **Input Data**: `data/processed/nwmp_operational_2025.csv`
- **Model**: `models/practical_operational_clean_best_model.pkl`
- **Features Used**: `models/practical_operational_clean_features.json`

## Prediction Summary
- Total Rows Processed: 666
- Predicted Non-Compliant: 491
- Predicted Compliant: 175

## Rule-vs-Model Agreement Summary
- Total Valid Rule Rows: 645.0
- Agreement Rate: 99.38%
- Disagreements: 4.0

## Hotspot & Trend Findings
- Persistent Hotspots Detected: 134
- View `reports/monitoring/top_20_hotspots.csv` for the most severe locations.

## Alert Summary
- Total Alerts Triggered: 516
- Severe/High Alerts: 495
- View `reports/monitoring/alerts.csv` for actionable field instructions.

## Next Step Recommendation
The operational monitoring pipeline has successfully processed the incoming data. The system is now ready for Phase 5 (Explainability & Insights via SHAP) and subsequent deployment to the Streamlit dashboard.
