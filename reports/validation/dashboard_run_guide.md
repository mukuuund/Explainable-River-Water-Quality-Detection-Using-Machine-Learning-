# Dashboard Run Guide

## Running the Pipeline

Do **NOT** run Streamlit in the same pasted multi-command chain as the pipeline steps.
Run the pipeline commands first.

**Correct commands:**
```bash
python main_phase4.py
python main_phase6a_realtime_ml_xai.py
python -m src.validation.model_inference_healthcheck
python -m src.explainability.operational_shap_explainer
python -m src.realtime.live_lime_explainer
python -m src.validation.dashboard_output_sanity_check
```

## Running the Dashboard

Once the pipelines have completed, run Streamlit separately:
```bash
streamlit run app/streamlit_app.py
```

If Streamlit stops immediately or is blocked on port 8501, try:
```bash
streamlit run app/streamlit_app.py --server.port 8502
```

## Troubleshooting
- **Wrong command for SHAP:** `-m src.explainability.operational_shap_explainer`
- **Correct command for SHAP:** `python -m src.explainability.operational_shap_explainer`
