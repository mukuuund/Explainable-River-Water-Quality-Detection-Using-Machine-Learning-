import pandas as pd
import numpy as np
import os
import json
import joblib
import re

INPUT_COMPLIANCE_PATH = "data/processed/realtime/live_sensor_compliance.csv"
OUTPUT_INFERENCE_PATH = "data/processed/realtime/live_sensor_predictions.csv"
MODEL_PATH = "models/practical_operational_clean_best_model.pkl"
FEATURES_PATH = "models/practical_operational_clean_features.json"

def load_feature_list(feature_path):
    with open(feature_path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    if isinstance(obj, list):
        return obj

    if isinstance(obj, dict):
        for key in [
            "features",
            "feature_names",
            "selected_features",
            "model_features",
            "final_features",
            "extended_clean_features",
            "practical_operational_clean_features",
            "best_model_features",
            "feature_list",
        ]:
            if key in obj and isinstance(obj[key], list):
                return obj[key]

        list_candidates = []
        for value in obj.values():
            if isinstance(value, list) and all(isinstance(x, str) for x in value):
                list_candidates.append(value)

        if list_candidates:
            return max(list_candidates, key=len)

    raise ValueError("Could not resolve feature list from JSON.")

def find_estimator(obj):
    if hasattr(obj, "predict"):
        return obj

    if isinstance(obj, dict):
        preferred = ["model", "pipeline", "best_model", "estimator", "clf", "classifier"]
        for key in preferred:
            if key in obj:
                est = find_estimator(obj[key])
                if est is not None:
                    return est
        for value in obj.values():
            est = find_estimator(value)
            if est is not None:
                return est

    if isinstance(obj, (list, tuple)):
        for item in obj:
            est = find_estimator(item)
            if est is not None:
                return est

    return None

def load_model_artifact(model_path):
    obj = joblib.load(model_path)
    est = find_estimator(obj)
    if est is None:
        raise ValueError("No sklearn-compatible estimator with predict() found in model artifact.")
    return est, type(obj).__name__, type(est).__name__

def run_live_inference():
    print("Running live model inference...")
    if not os.path.exists(INPUT_COMPLIANCE_PATH):
        raise FileNotFoundError(f"Input file not found: {INPUT_COMPLIANCE_PATH}")

    df = pd.read_csv(INPUT_COMPLIANCE_PATH)
    if df.empty:
        print("Dataframe is empty.")
        return df

    # Prepare defaults
    df['live_ml_predicted_non_compliant'] = np.nan
    df['live_ml_predicted_compliance_label'] = "Unknown"
    df['live_ml_non_compliance_probability'] = np.nan
    df['model_input_missing_feature_count'] = 0
    df['model_input_missing_feature_list'] = ""
    df['model_input_available_feature_count'] = 0
    df['ml_prediction_confidence_note'] = ""
    df['live_ml_prediction_status'] = "Failed"
    df['live_ml_fallback_used'] = False
    df['live_ml_fallback_reason'] = ""

    if not os.path.exists(MODEL_PATH) or not os.path.exists(FEATURES_PATH):
        msg = "Model or features JSON not found. Skipping ML inference."
        print(msg)
        df['ml_prediction_confidence_note'] = msg
        os.makedirs(os.path.dirname(OUTPUT_INFERENCE_PATH), exist_ok=True)
        df.to_csv(OUTPUT_INFERENCE_PATH, index=False)
        return df

    try:
        expected_features = load_feature_list(FEATURES_PATH)
        os.makedirs("reports/realtime", exist_ok=True)
        with open("reports/realtime/live_feature_artifact_debug.txt", "w") as f:
            f.write(f"Type: {type(expected_features)}\nLength: {len(expected_features)}")
        pd.DataFrame({'features': expected_features}).to_csv("reports/realtime/resolved_live_model_features.csv", index=False)
    except Exception as e:
        msg = f"Failed to load features: {str(e)}"
        print(msg)
        df['ml_prediction_confidence_note'] = msg
        df.to_csv(OUTPUT_INFERENCE_PATH, index=False)
        return df

    try:
        model, obj_type, est_type = load_model_artifact(MODEL_PATH)
        with open("reports/realtime/live_model_artifact_debug.txt", "w") as f:
            f.write(f"Object Type: {obj_type}\nEstimator Type: {est_type}")
    except Exception as e:
        msg = f"Failed to load model: {str(e)}"
        print(msg)
        df['ml_prediction_confidence_note'] = msg
        df.to_csv(OUTPUT_INFERENCE_PATH, index=False)
        return df

    inference_df = pd.DataFrame(index=df.index)
    readiness_data = []
    
    numeric_cols = ['ph', 'dissolved_oxygen', 'bod', 'temperature', 'conductivity', 
                    'nitrate', 'fecal_coliform', 'total_coliform', 'fecal_streptococci', 'turbidity', 'cod', 'total_dissolved_solids']
    def process_row(idx, row):
        missing_feats = []
        avail_feats = []
        for feat in expected_features:
            val = row.get(feat, np.nan)
            
            if pd.isna(val) and feat not in numeric_cols:
                val = "Unknown"
                
            inference_df.loc[idx, feat] = val
            if pd.isna(val) or val == "Unknown":
                missing_feats.append(feat)
            else:
                avail_feats.append(feat)
                
        core_count = row.get('core_parameter_count', 0)
        
        if core_count == 3:
            note = "High confidence: DO, BOD, and pH are available."
        elif core_count == 2:
            note = "Medium confidence: one core parameter is missing."
        elif core_count <= 1:
            note = "Low confidence: two or more core parameters are missing."
        else:
            note = "Insufficient live features for reliable ML prediction."
            
        return pd.Series({
            'model_input_missing_feature_count': len(missing_feats),
            'model_input_missing_feature_list': "|".join(missing_feats) if missing_feats else "None",
            'model_input_available_feature_count': len(avail_feats),
            'ml_prediction_confidence_note': note
        })

    meta_info = df.apply(lambda row: process_row(row.name, row), axis=1)
    df.update(meta_info)

    # Convert ALL expected features to numeric safely, as the model pipeline treats them all as numeric
    for col in expected_features:
        present = col in inference_df.columns
        sample_val = inference_df[col].dropna().iloc[0] if present and not inference_df[col].dropna().empty else None
        
        cleaned_numeric = False
        if present:
            if inference_df[col].dtype == object:
                s = inference_df[col].astype(str)
                s = s.str.replace(r'\(BDL\)', '', regex=True, flags=import_re.IGNORECASE if 'import_re' in locals() else 2) # 2 is re.IGNORECASE
                s = s.replace(r'(?i)bdl', '0', regex=True)
                inference_df[col] = pd.to_numeric(s, errors='coerce')
                cleaned_numeric = True
            else:
                inference_df[col] = pd.to_numeric(inference_df[col], errors='coerce')
                
        readiness_data.append({
            'expected_feature': col,
            'present_in_raw': col in df.columns,
            'non_null_count': inference_df[col].notna().sum() if present else 0,
            'dtype_in_model_input': str(inference_df[col].dtype) if present else "float64",
            'sample_value_raw': str(sample_val),
            'numeric_cleaning_applied': cleaned_numeric
        })
        
    pd.DataFrame(readiness_data).to_csv("reports/realtime/live_feature_readiness_report.csv", index=False)

    X_live = inference_df[expected_features]    
    try:
        preds = model.predict(X_live)
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_live)[:, 1]
        else:
            probs = np.full(len(preds), np.nan)
            
        df['live_ml_predicted_non_compliant'] = preds
        df['live_ml_predicted_compliance_label'] = ["Non-Compliant" if p == 1 else "Compliant" for p in preds]
        df['live_ml_non_compliance_probability'] = probs
        df['live_ml_prediction_status'] = "Success"
        
    except Exception as e:
        msg = f"Model inference failed: {str(e)}"
        print(msg)
        df['ml_prediction_confidence_note'] = msg
        df['live_ml_prediction_status'] = "Failed"
        
        # Fallback
        for idx, row in df.iterrows():
            core_count = row.get('core_parameter_count', 0)
            if core_count == 3:
                strict_label = row.get('live_strict_compliance_label')
                df.at[idx, 'live_ml_predicted_compliance_label'] = strict_label
                df.at[idx, 'live_ml_predicted_non_compliant'] = 1.0 if strict_label == 'Non-Compliant' else 0.0
                df.at[idx, 'live_ml_fallback_used'] = True
                df.at[idx, 'live_ml_fallback_reason'] = "Fallback rule-based estimate used because ML inference failed."
                df.at[idx, 'ml_prediction_confidence_note'] = "Fallback rule-based estimate used because ML inference failed."

    os.makedirs(os.path.dirname(OUTPUT_INFERENCE_PATH), exist_ok=True)
    df.to_csv(OUTPUT_INFERENCE_PATH, index=False)
    print(f"Inference completed. Saved to {OUTPUT_INFERENCE_PATH}")
    return df

if __name__ == "__main__":
    run_live_inference()
