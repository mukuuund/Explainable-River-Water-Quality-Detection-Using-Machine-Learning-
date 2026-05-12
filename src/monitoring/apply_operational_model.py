import pandas as pd
import numpy as np
import joblib
import json
import os
import logging

def load_feature_list(feature_path):
    with open(feature_path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    if isinstance(obj, list):
        return obj

    if isinstance(obj, dict):
        for key in [
            "features", "feature_names", "selected_features", "model_features",
            "final_features", "extended_clean_features", 
            "practical_operational_clean_features", "best_model_features", "feature_list"
        ]:
            if key in obj and isinstance(obj[key], list):
                return obj[key]
        
        list_candidates = [v for v in obj.values() if isinstance(v, list) and all(isinstance(x, str) for x in v)]
        if list_candidates:
            return max(list_candidates, key=len)

    raise ValueError("Could not resolve feature list from JSON.")

def apply_model():
    logging.info("Loading NWMP 2025 operational data...")
    input_path = 'data/processed/nwmp_operational_2025.csv'
    if not os.path.exists(input_path):
        logging.warning(f"{input_path} not found. Skipping apply_model.")
        return
    df = pd.read_csv(input_path)
    
    model_path = 'models/practical_operational_clean_best_model.pkl'
    features_path = 'models/practical_operational_clean_features.json'
    
    model = joblib.load(model_path)
    features = load_feature_list(features_path)
    logging.info(f"Loaded model expecting {len(features)} features.")
    
    # Track feature readiness
    readiness_data = []
    
    X = pd.DataFrame(index=df.index)
    
    for feat in features:
        present = feat in df.columns
        if not present:
            df[feat] = np.nan
        
        # Original sample value before numeric coercion
        sample_val = df[feat].dropna().iloc[0] if present and not df[feat].dropna().empty else None
        
        # Safe numeric conversion for model input
        # We assume the model pipeline expects all inputs as numeric due to all-NaN categorical columns during training
        # We extract numeric if it's a string, or coerce completely
        s = df[feat].copy()
        cleaned_numeric = False
        if s.dtype == object:
            # simple BDL cleanup just in case standardizer missed it or for safety
            s = s.astype(str).str.replace(r'\(BDL\)', '', regex=True, flags=re.IGNORECASE)
            s = s.replace(r'(?i)bdl', '0', regex=True)
            cleaned_numeric = True
            
        X[feat] = pd.to_numeric(s, errors='coerce')
        
        readiness_data.append({
            'expected_feature': feat,
            'present_in_raw': present,
            'non_null_count': X[feat].notna().sum(),
            'dtype_in_model_input': str(X[feat].dtype),
            'sample_value_raw': str(sample_val),
            'numeric_cleaning_applied': cleaned_numeric
        })
        
    # Save readiness report
    os.makedirs('reports/monitoring', exist_ok=True)
    pd.DataFrame(readiness_data).to_csv('reports/monitoring/operational_feature_readiness_report.csv', index=False)
    
    # Save model input used for river_name for debugging
    if 'river_name' in X.columns:
        df['model_input_river_name_used'] = X['river_name']

    # Generate predictions
    logging.info("Generating predictions...")
    
    df['ml_prediction_status'] = 'Unknown'
    df['ml_prediction_confidence_note'] = ''
    
    try:
        preds = model.predict(X[features])
        try:
            probs = model.predict_proba(X[features])[:, 1]
        except AttributeError:
            probs = np.full(len(preds), np.nan)
            
        df['predicted_non_compliant'] = preds
        df['ml_predicted_compliance_label'] = df['predicted_non_compliant'].map({1: 'Non-Compliant', 0: 'Compliant'})
        df['predicted_compliance_label'] = df['ml_predicted_compliance_label']
        df['ml_non_compliance_probability'] = probs
        df['predicted_non_compliance_probability'] = probs
        df['ml_prediction_status'] = 'Success'
        
    except Exception as e:
        logging.error(f"Prediction failed: {e}")
        df['ml_prediction_status'] = 'Failed'
        df['predicted_non_compliant'] = np.nan
        df['ml_predicted_compliance_label'] = 'Unknown'
        df['predicted_compliance_label'] = 'Unknown'
        df['ml_non_compliance_probability'] = np.nan
        df['predicted_non_compliance_probability'] = np.nan
        df['ml_prediction_confidence_note'] = f"Prediction failed: {e}"

    # Generate confidence notes
    for idx, row in df.iterrows():
        if df.at[idx, 'ml_prediction_status'] != 'Success':
            continue
            
        core_params = ['dissolved_oxygen', 'bod', 'ph']
        avail_core = sum(1 for p in core_params if p in df.columns and pd.notna(row[p]))
        
        if avail_core == 3:
            note = "High confidence: DO, BOD, and pH are available."
        elif avail_core == 2:
            note = "Medium confidence: one core parameter is missing."
        elif avail_core <= 1:
            note = "Low confidence: two or more core parameters are missing."
        else:
            note = "Prediction unavailable: required model input could not be prepared."
            
        df.at[idx, 'ml_prediction_confidence_note'] = note

    # Preserve existing rule-based fields (ensure they exist)
    expected_fields = [
        'available_compliance_label', 'strict_compliance_label', 'rule_based_compliance_label', 
        'risk_score', 'risk_category', 'violation_reasons', 'label_confidence', 'risk_confidence'
    ]
    for field in expected_fields:
        if field not in df.columns:
            if field == 'rule_based_compliance_label' and 'strict_compliance_label' in df.columns:
                df[field] = df['strict_compliance_label']
            else:
                df[field] = 'Unknown'
            
    # Save predictions
    os.makedirs('data/processed', exist_ok=True)
    
    output_path = 'data/processed/nwmp_2025_predictions.csv'
    df.to_csv(output_path, index=False)
    
    summary = {
        'total_rows': len(df),
        'predicted_compliant': (df['ml_predicted_compliance_label'] == 'Compliant').sum() if 'ml_predicted_compliance_label' in df.columns else 0,
        'predicted_non_compliant': (df['ml_predicted_compliance_label'] == 'Non-Compliant').sum() if 'ml_predicted_compliance_label' in df.columns else 0
    }
    
    pd.DataFrame([summary]).to_csv('reports/monitoring/nwmp_prediction_summary.csv', index=False)
    logging.info(f"Predictions saved to {output_path}")

if __name__ == "__main__":
    import re
    apply_model()
