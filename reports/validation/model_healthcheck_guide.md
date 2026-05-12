# Model Health Check Guide

## Overview
The `src/validation/model_inference_healthcheck.py` script ensures that your ML models are not just executing, but executing *correctly* by verifying input schemas, predictions, confidence levels, and fallbacks.

## How It Works
It runs an automated verification over two core pipelines:
1. **Operational (NWMP) Pipeline**:
   - Ensures the artifact is loaded successfully alongside the correct feature JSON.
   - Validates that `nwmp_2025_predictions.csv` contains successful predictions, valid probability thresholds, and verifies rule-vs-ML agreement.
2. **Real-Time Pipeline**:
   - Verifies the `live_sensor_predictions.csv` data.
   - Tracks the number of successful ML queries versus queries forced into "fallback" or "Unknown" statuses due to lack of core parameters.

## Usage
To evaluate the models at any time, run:
```bash
python -m src.validation.model_inference_healthcheck
```
The results are consolidated into CSV reports inside `reports/validation/` and surfaced directly in the Streamlit Dashboard under the "Operational Model Health Check" and "Real-Time Model Health Check" modules.
