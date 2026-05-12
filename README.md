# Explainable River Water Quality Detection Using Machine Learning

## Project Overview

This project is an Explainable River Water Quality Intelligence Dashboard designed to analyze river water-quality records, predict compliance status, detect risky locations, and explain model predictions in an easy-to-understand way.

The system combines water-quality rules, machine learning models, hotspot detection, alert generation, and explainability techniques such as SHAP and LIME. It supports both historical data analysis and live CPCB-based data fetching.

## Live Dashboard

The deployed dashboard is available here:

https://river-quality-ai.streamlit.app/

## Main Features

### 1. Project Overview

Displays important project-level metrics, dataset summaries, model performance indicators, and overall water-quality status.

### 2. Live Sensor Data View

Fetches recent CPCB water-quality data and applies compliance checks, ML prediction, risk scoring, and alert generation.

### 3. Compliance Analysis

Shows compliant and non-compliant water-quality records using important parameters such as pH, Dissolved Oxygen, and BOD.

### 4. Hotspot Detection

Identifies stations or locations that repeatedly show high-risk or non-compliant water-quality behavior.

### 5. Alert Center

Generates priority-based alerts using risk category, compliance status, and repeated hotspot behavior.

### 6. Explainability

Uses SHAP and LIME to explain why a model predicted a record as compliant or non-compliant.

### 7. Manual Prediction

Allows users to manually enter water-quality values and test compliance prediction.

The manual prediction section supports two prediction modes:

- **Core Mode**: Uses pH, DO, and BOD for rule-based and ML prediction.
- **Auxiliary-Only Mode**: Uses optional parameters such as temperature, conductivity, turbidity, and other available features when pH, DO, and BOD are not provided.

## Methodology

The project follows a multi-level pipeline:

```text
Historical / Manual / NWMP / Live Water Quality Data
        ↓
Data Cleaning and Preprocessing
        ↓
Column Standardization and Missing Value Handling
        ↓
Feature Engineering and Data Preparation
        ↓
Machine Learning Model Training and Evaluation
        ↓
Operational Model and Auxiliary-Only Model
        ↓
Hotspot Detection and Alert Generation
        ↓
SHAP / LIME / Parameter-Level Explanation
        ↓
Interactive Streamlit Dashboard
```

## Machine Learning Models

The dashboard uses two main model types:

### Core Operational Model

This model works with important water-quality parameters such as pH, Dissolved Oxygen, and BOD. It is used for the main compliance prediction workflow.

### Auxiliary-Only Model

This model is used when core parameters are not available. It predicts compliance using supporting water-quality features only. This helps the dashboard still provide useful predictions when pH, DO, and BOD are missing.

## Explainability

The project includes explainability to make predictions easier to understand:

- **SHAP** explains global and feature-level model behavior.
- **LIME** explains individual/local predictions.
- Parameter-level explanations help users understand whether values such as pH, DO, or BOD are contributing to compliance or non-compliance.

## Required Files

The dashboard depends on the following important files and folders:

```text
app/streamlit_app.py
requirements.txt
models/
data/
reports/
.streamlit/config.toml
```

Important model and data files include:

```text
models/practical_operational_clean_best_model.pkl
models/practical_operational_clean_features.json
models/auxiliary_only_leakage_safe_model.pkl
models/auxiliary_only_leakage_safe_features.json
data/processed/nwmp_2025_predictions.csv
reports/monitoring/
reports/explainability/
reports/realtime/
```

## Project Structure

```text
.
├── app/
│   └── streamlit_app.py
├── data/
│   └── processed/
├── models/
│   ├── practical_operational_clean_best_model.pkl
│   ├── practical_operational_clean_features.json
│   ├── auxiliary_only_leakage_safe_model.pkl
│   └── auxiliary_only_leakage_safe_features.json
├── reports/
│   ├── monitoring/
│   ├── explainability/
│   └── realtime/
├── .streamlit/
│   └── config.toml
├── requirements.txt
└── README.md
```

## How to Run Locally

### 1. Install Python

Use Python 3.11 or another compatible version used during model training.

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit app

```bash
streamlit run app/streamlit_app.py
```

### 4. Open the dashboard

After running the command, open:

```text
http://localhost:8501
```

## Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Plotly
- Matplotlib
- Joblib
- SHAP
- LIME
- Requests
- BeautifulSoup

## Key Outputs

The dashboard provides:

- Compliance prediction
- ML probability
- Risk score
- Risk category
- Confidence level
- Parameter-level explanation
- Hotspot identification
- Alert generation
- SHAP and LIME explainability outputs

## Limitations

- Live CPCB data availability may depend on external website or API response.
- Model predictions depend on the quality and completeness of input parameters.
- Rule-based compliance cannot be evaluated when pH, DO, and BOD are missing.
- Auxiliary-only predictions are useful when core parameters are unavailable, but they may be less direct than predictions using pH, DO, and BOD.

## Project Title

**Explainable River Water Quality Detection Using Machine Learning**

## Developed For

Minor Project - 2
