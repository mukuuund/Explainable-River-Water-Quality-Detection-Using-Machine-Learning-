# Water Quality AI Dashboard

## Purpose
This dashboard provides an interactive web interface for the Water Quality AI project. It presents explainable compliance monitoring, hotspot detection, alerts, and model interpretation.

## How to Run
Make sure you have Streamlit installed, then run the following command from the project root:
```bash
streamlit run app/streamlit_app.py
```

## Required Input Files
The dashboard dynamically loads data from:
- `data/processed/nwmp_2025_predictions.csv`
- `reports/monitoring/` (hotspots and alerts)
- `reports/explainability/` (global and auxiliary drivers, explainable alerts)
- `reports/expanded_data/` (expanded historical baselines)

## Pages Included
- Project Overview
- Compliance Monitoring
- Hotspot Detection
- Alert Center
- Explainability
- Expanded Historical Baseline
- Methodology & Limitations

## Important Interpretation Notes
- The Extended Clean Model uses DO, BOD, and pH. Compliance labels are rule-derived, making the model an automated regulatory engine.
- Auxiliary importance provides academic insight into secondary indicators.

## Limitations
- Historical context for Maharashtra hotspots is not available in the current expanded baseline.
- Alerts are decision-support indicators, not final regulatory declarations.
