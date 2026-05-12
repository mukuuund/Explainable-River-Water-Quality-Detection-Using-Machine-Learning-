# SHAP and LIME Explainability Implementation Summary

## 1. Operational Model (SHAP)
**Location:** `src/explainability/operational_shap_explainer.py`

### What it does:
We apply SHAP (SHapley Additive exPlanations) to the operational batch predictions derived from the NWMP 2025 data. We load the exact `Pipeline` and features from the main model artifact and apply `shap.KernelExplainer` to dissect the prediction outputs.

### Details:
- Due to the large size of operational datasets, a maximum sample of 300 rows is selected to preserve performance.
- Global metrics (Feature Importance) and local sample explanations are generated and stored in `reports/explainability/`.
- Missing variables or Strings (like "BDL") are properly resolved prior to the explanation process to match the prediction environment.

## 2. Real-Time Model (LIME)
**Location:** `src/realtime/live_lime_explainer.py`

### What it does:
For the Real-Time live monitoring pipeline, we apply LIME (Local Interpretable Model-Agnostic Explanations). Because the live data fluctuates significantly and may frequently omit some parameters, local interpretability fits best.

### Details:
- The script initializes `LimeTabularExplainer` using a recent sample of historical operational data to serve as the background distribution.
- It individually processes each live monitoring station in `live_sensor_predictions.csv` through `predict_proba`.
- Results outline the precise thresholds and variables pushing a station toward 'Non-Compliant' or 'Compliant' states.

## 3. Dashboard Integration
Both SHAP and LIME outputs have been seamlessly integrated into `app/streamlit_app.py` under the "Compliance Monitoring" and "Live Sensor Monitor" tabs respectively.

## Commands:
To regenerate XAI explanations, execute:
```bash
python -m src.explainability.operational_shap_explainer
python -m src.realtime.live_lime_explainer
```
