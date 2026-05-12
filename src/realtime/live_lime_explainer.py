import os
import sys
import pandas as pd
import numpy as np
import json
import joblib
import logging
import re
import warnings
from lime import lime_tabular

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
logging.basicConfig(level=logging.INFO)

def load_feature_list(feature_path):
    with open(feature_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        for key in ["features", "feature_names", "selected_features", "model_features", "final_features", "extended_clean_features", "practical_operational_clean_features", "best_model_features", "feature_list"]:
            if key in obj and isinstance(obj[key], list):
                return obj[key]
        list_candidates = [v for v in obj.values() if isinstance(v, list) and all(isinstance(x, str) for x in v)]
        if list_candidates:
            return max(list_candidates, key=len)
    raise ValueError("Could not resolve feature list from JSON.")

def run_live_lime():
    logging.info("Starting Live LIME Explanation")
    
    reports_dir = os.path.join(PROJECT_ROOT, "reports", "realtime")
    os.makedirs(reports_dir, exist_ok=True)
    
    model_path = os.path.join(PROJECT_ROOT, 'models', 'practical_operational_clean_best_model.pkl')
    features_path = os.path.join(PROJECT_ROOT, 'models', 'practical_operational_clean_features.json')
    live_preds_path = os.path.join(PROJECT_ROOT, 'data', 'processed', 'realtime', 'live_sensor_predictions.csv')
    op_preds_path = os.path.join(PROJECT_ROOT, 'data', 'processed', 'nwmp_operational_2025.csv')
    
    if not os.path.exists(model_path) or not os.path.exists(features_path) or not os.path.exists(live_preds_path):
        logging.error("Required model, features, or live predictions missing. Aborting LIME.")
        return
        
    try:
        model = joblib.load(model_path)
        features = load_feature_list(features_path)
        df_live = pd.read_csv(live_preds_path)
    except Exception as e:
        logging.error(f"Failed to load files: {e}")
        return
        
    if 'live_ml_prediction_status' in df_live.columns:
        df_live_succ = df_live[df_live['live_ml_prediction_status'] == 'Success'].copy()
    else:
        df_live_succ = df_live.copy()
        
    if df_live_succ.empty:
        logging.warning("No successful live predictions to explain.")
        return
        
    # Build background data
    # Prefer operational data for background if available
    background_df = None
    if os.path.exists(op_preds_path):
        try:
            op_raw = pd.read_csv(op_preds_path)
            # Take a small sample of operational data
            if len(op_raw) > 500:
                op_raw = op_raw.sample(500, random_state=42)
            background_df = pd.DataFrame(index=op_raw.index)
            for feat in features:
                s = op_raw[feat].copy() if feat in op_raw.columns else pd.Series(np.nan, index=op_raw.index)
                if s.dtype == object:
                    s = s.astype(str).str.replace(r'\(BDL\)', '', regex=True, flags=re.IGNORECASE)
                    s = s.replace(r'(?i)bdl', '0', regex=True)
                background_df[feat] = pd.to_numeric(s, errors='coerce')
        except Exception as e:
            logging.warning(f"Failed to load operational background data: {e}")
            
    if background_df is None or background_df.empty:
        logging.warning("Using live data as background since operational data is unavailable.")
        background_df = pd.DataFrame(index=df_live_succ.index)
        for feat in features:
            s = df_live_succ[feat].copy() if feat in df_live_succ.columns else pd.Series(np.nan, index=df_live_succ.index)
            if s.dtype == object:
                s = s.astype(str).str.replace(r'\(BDL\)', '', regex=True, flags=re.IGNORECASE)
                s = s.replace(r'(?i)bdl', '0', regex=True)
            background_df[feat] = pd.to_numeric(s, errors='coerce')
            
    # Impute background data quickly just for LIME explainer initialization if needed
    # LIME explainer needs a numpy array without NaNs typically
    background_array = background_df.fillna(background_df.median(numeric_only=True)).fillna(0).values
    
    explainer = lime_tabular.LimeTabularExplainer(
        training_data=background_array,
        feature_names=features,
        class_names=['Compliant', 'Non-Compliant'],
        mode='classification'
    )
    
    # We must provide predict_fn that accepts numpy arrays and outputs probabilities
    if not hasattr(model, 'predict_proba'):
        logging.error("Model does not have predict_proba. LIME requires probabilities.")
        return
        
    def predict_fn(x_arr):
        x_df = pd.DataFrame(x_arr, columns=features)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Skipping features without any observed values.*",
                category=UserWarning
            )
            return model.predict_proba(x_df)
        
    explanations = []
    
    for idx, row in df_live.iterrows():
        # Only explain if successful
        if row.get('live_ml_prediction_status', 'Failed') != 'Success':
            explanations.append({
                'station_name': row.get('station_name', 'Unknown'),
                'river_name': row.get('river_name', 'Unknown'),
                'timestamp': row.get('latest_timestamp', row.get('timestamp', 'Unknown')),
                'live_ml_predicted_compliance_label': row.get('live_ml_predicted_compliance_label', 'Unknown'),
                'live_ml_non_compliance_probability': row.get('live_ml_non_compliance_probability', np.nan),
                'lime_prediction_class': 'Unknown',
                'lime_top_positive_features': 'None',
                'lime_top_negative_features': 'None',
                'lime_explanation_text': 'N/A (Prediction Failed)',
                'lime_status': 'Skipped',
                'lime_error': 'Prediction was not successful.'
            })
            continue
            
        try:
            x_instance = []
            for feat in features:
                val = row.get(feat, np.nan)
                if isinstance(val, str):
                    val = re.sub(r'(?i)bdl|\(bdl\)', '0', val)
                try:
                    val = float(val)
                except:
                    val = np.nan
                x_instance.append(val)
                
            x_arr = np.array(x_instance)
            # Impute instance NaNs with background medians for LIME
            x_arr = np.where(np.isnan(x_arr), np.nanmedian(background_array, axis=0), x_arr)
            
            exp = explainer.explain_instance(x_arr, predict_fn, num_features=5)
            
            # exp.as_list() returns features pushing towards class 1 (Non-Compliant)
            exp_list = exp.as_list()
            
            pos_feats = [f"{f[0]} ({f[1]:.3f})" for f in exp_list if f[1] > 0]
            neg_feats = [f"{f[0]} ({f[1]:.3f})" for f in exp_list if f[1] < 0]
            
            pred_class = exp.predict_proba[1] > 0.5
            pred_class_label = 'Non-Compliant' if pred_class else 'Compliant'
            
            # Simple text
            exp_text = ""
            if pos_feats:
                exp_text += f"Values like {', '.join([f[0] for f in exp_list if f[1] > 0])} pushed the reading towards Non-Compliant. "
            if neg_feats:
                exp_text += f"Values like {', '.join([f[0] for f in exp_list if f[1] < 0])} reduced the risk."
            if not exp_text:
                exp_text = "No strong drivers identified."
                
            explanations.append({
                'station_name': row.get('station_name', 'Unknown'),
                'river_name': row.get('river_name', 'Unknown'),
                'timestamp': row.get('latest_timestamp', row.get('timestamp', 'Unknown')),
                'live_ml_predicted_compliance_label': row.get('live_ml_predicted_compliance_label', 'Unknown'),
                'live_ml_non_compliance_probability': row.get('live_ml_non_compliance_probability', np.nan),
                'lime_prediction_class': pred_class_label,
                'lime_top_positive_features': " | ".join(pos_feats) if pos_feats else "None",
                'lime_top_negative_features': " | ".join(neg_feats) if neg_feats else "None",
                'lime_explanation_text': exp_text.strip(),
                'lime_status': 'Success',
                'lime_error': ''
            })
            
        except Exception as e:
            explanations.append({
                'station_name': row.get('station_name', 'Unknown'),
                'river_name': row.get('river_name', 'Unknown'),
                'timestamp': row.get('latest_timestamp', row.get('timestamp', 'Unknown')),
                'live_ml_predicted_compliance_label': row.get('live_ml_predicted_compliance_label', 'Unknown'),
                'live_ml_non_compliance_probability': row.get('live_ml_non_compliance_probability', np.nan),
                'lime_prediction_class': 'Unknown',
                'lime_top_positive_features': 'None',
                'lime_top_negative_features': 'None',
                'lime_explanation_text': 'Error computing LIME',
                'lime_status': 'Failed',
                'lime_error': str(e)
            })
            
    pd.DataFrame(explanations).to_csv(os.path.join(reports_dir, 'live_lime_explanations.csv'), index=False)
    
    with open(os.path.join(reports_dir, 'live_lime_summary.md'), 'w') as f:
        f.write("# Live LIME Explanation Summary\n")
        f.write(f"Computed LIME explanations for {len(df_live)} live station predictions.\n\n")
        f.write("During LIME perturbation, sklearn may warn that some features are all-NaN in the live background. This warning is suppressed only inside repeated LIME prediction calls to keep logs readable. It does not hide prediction failures.\n")
        
    logging.info("Live LIME explainer completed successfully.")

if __name__ == "__main__":
    run_live_lime()
