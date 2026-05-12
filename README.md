# Water Quality AI Dashboard

## Description
This project is an Explainable Multi-Modal River Water Quality Intelligence System. It uses machine learning models and rule-based checks to predict water quality compliance (e.g., classifying rivers as Compliant or Non-Compliant). It features real-time sensor monitoring, historical baseline analysis, hotspot detection, an automated alert center, and model explainability (SHAP & LIME).

## Main Features
- **Project Overview**: High-level summary metrics of water quality and model performance.
- **Live Sensor Monitor**: Fetches latest real-time CPCB sensor snapshots and runs on-the-fly compliance checks, ML inference, and generates actionable alerts.
- **Compliance Monitoring**: Explores record-level compliant and non-compliant predictions.
- **Hotspot Detection**: Identifies stations repeatedly exhibiting high risk or non-compliance.
- **Alert Center**: Displays prioritized alerts based on severity and risk scores.
- **Explainability**: Uses SHAP and LIME to explain model drivers and local station predictions.
- **Manual Prediction**: Test water quality compliance interactively by inputting DO, BOD, pH, or optional auxiliary parameters (like Temperature, Conductivity, Turbidity) to evaluate compliance using an auxiliary-only fallback model.

## Required Model and Data Files
The dashboard relies on standard datasets and pre-trained `.pkl` models located in:
- `models/practical_operational_clean_best_model.pkl`
- `models/practical_operational_clean_features.json`
- `models/auxiliary_only_leakage_safe_model.pkl`
- `models/auxiliary_only_leakage_safe_features.json`
- `data/processed/nwmp_2025_predictions.csv`
- Various reports in `reports/monitoring/`, `reports/explainability/`, and `reports/realtime/`.

*(Note: The `models` and `reports` directories must be present and correctly populated for all dashboard features to function without errors.)*

## How to Run Locally
1. Ensure you have Python 3.9+ installed.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit application:
   ```bash
   streamlit run app/streamlit_app.py
   ```
4. Access the dashboard at `http://localhost:8501`.

## How to Deploy on Streamlit Community Cloud
1. Push this repository to GitHub. Ensure the `app/streamlit_app.py`, `models/`, `data/`, `reports/`, and `requirements.txt` are tracked and pushed.
2. Go to [Streamlit Community Cloud](https://share.streamlit.io/).
3. Click **New app**.
4. Select your GitHub repository, branch, and specify the main file path as `app/streamlit_app.py`.
5. Click **Deploy**. Streamlit Cloud will automatically install the packages from `requirements.txt` and launch your dashboard.

## GitHub Upload Steps
To prepare and upload this code to GitHub for the first time, run the following commands in your terminal at the project root:
```bash
git init
git add .
git commit -m "Initial commit for Streamlit Cloud deployment"
git branch -M main
git remote add origin <YOUR_GITHUB_REPOSITORY_URL>
git push -u origin main
```
