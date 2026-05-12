import os
import sys
import pandas as pd
import numpy as np
import json
import joblib
import matplotlib.pyplot as plt
import shap
import re
import logging
from sklearn.pipeline import Pipeline

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

def run_operational_shap():
    logging.info("Starting Operational SHAP Explanation")
    
    reports_dir = os.path.join(PROJECT_ROOT, "reports", "explainability")
    figs_dir = os.path.join(reports_dir, "figures")
    os.makedirs(figs_dir, exist_ok=True)
    
    model_path = os.path.join(PROJECT_ROOT, 'models', 'practical_operational_clean_best_model.pkl')
    features_path = os.path.join(PROJECT_ROOT, 'models', 'practical_operational_clean_features.json')
    data_path = os.path.join(PROJECT_ROOT, 'data', 'processed', 'nwmp_2025_predictions.csv')
    
    if not os.path.exists(model_path) or not os.path.exists(features_path) or not os.path.exists(data_path):
        logging.error("Required model, features, or data file is missing. Aborting SHAP.")
        return
        
    try:
        model = joblib.load(model_path)
        features = load_feature_list(features_path)
        df = pd.read_csv(data_path)
    except Exception as e:
        logging.error(f"Failed to load files: {e}")
        return
        
    # Filter only rows where ML prediction was successful
    if 'ml_prediction_status' in df.columns:
        df = df[df['ml_prediction_status'] == 'Success'].copy()
    if df.empty:
        logging.error("No successful predictions to explain.")
        return
        
    # Sample 300 rows max for speed
    if len(df) > 300:
        df = df.sample(300, random_state=42)
        
    X = pd.DataFrame(index=df.index)
    for feat in features:
        s = df[feat].copy() if feat in df.columns else pd.Series(np.nan, index=df.index)
        if s.dtype == object:
            s = s.astype(str).str.replace(r'\(BDL\)', '', regex=True, flags=re.IGNORECASE)
            s = s.replace(r'(?i)bdl', '0', regex=True)
        X[feat] = pd.to_numeric(s, errors='coerce')
        
    logging.info(f"Applying SHAP to {len(X)} operational rows.")
    
    try:
        if isinstance(model, Pipeline):
            preprocessor = model[:-1]
            final_estimator = model[-1]
            
            X_transformed = preprocessor.transform(X)
            if hasattr(X_transformed, "toarray"):
                X_transformed = X_transformed.toarray()
                
            try:
                raw_names = preprocessor.get_feature_names_out()
                shap_feature_names = [n.split('__')[-1] for n in raw_names]
            except Exception:
                shap_feature_names = [f"transformed_feature_{i}" for i in range(X_transformed.shape[1])]
            
            # Save mapping
            mapping_data = []
            for i, name in enumerate(shap_feature_names):
                mapping_data.append({
                    "transformed_index": i,
                    "transformed_feature_name": name,
                    "original_feature": name if name in features else "Unknown",
                    "used_in_shap": True
                })
            pd.DataFrame(mapping_data).to_csv(os.path.join(reports_dir, 'operational_shap_debug_feature_alignment.csv'), index=False)
            
            background = shap.kmeans(X_transformed, 10)
            
            if type(final_estimator).__name__ in ['DecisionTreeClassifier', 'RandomForestClassifier', 'XGBClassifier', 'LGBMClassifier']:
                explainer = shap.TreeExplainer(final_estimator)
                shap_values = explainer.shap_values(X_transformed)
            else:
                if hasattr(final_estimator, 'predict_proba'):
                    predict_fn = lambda x: final_estimator.predict_proba(x)[:, 1]
                else:
                    predict_fn = final_estimator.predict
                explainer = shap.KernelExplainer(predict_fn, background)
                shap_values = explainer.shap_values(X_transformed)
        else:
            X_transformed = X.values
            shap_feature_names = features
            background = shap.kmeans(X_transformed, 10)
            if hasattr(model, 'predict_proba'):
                predict_fn = lambda x: model.predict_proba(pd.DataFrame(x, columns=features))[:, 1]
            else:
                predict_fn = lambda x: model.predict(pd.DataFrame(x, columns=features))
            explainer = shap.KernelExplainer(predict_fn, background)
            shap_values = explainer.shap_values(X_transformed)
            
            mapping_data = [{"transformed_index": i, "transformed_feature_name": name, "original_feature": name, "used_in_shap": True} for i, name in enumerate(features)]
            pd.DataFrame(mapping_data).to_csv(os.path.join(reports_dir, 'operational_shap_debug_feature_alignment.csv'), index=False)

        if isinstance(shap_values, list):
            if len(shap_values) > 1:
                shap_values = shap_values[1]
            else:
                shap_values = shap_values[0]
                
        if len(shap_values.shape) == 3:
            shap_values = shap_values[:, :, 1]
            
        assert shap_values.shape[1] == len(shap_feature_names), f"Shape mismatch: shap_values cols {shap_values.shape[1]} != features {len(shap_feature_names)}"

        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        mean_shap = shap_values.mean(axis=0)
        
        importance_df = pd.DataFrame({
            'feature': shap_feature_names,
            'mean_abs_shap': mean_abs_shap,
            'mean_shap': mean_shap
        })
        importance_df['rank'] = importance_df['mean_abs_shap'].rank(ascending=False, method='min').astype(int)
        importance_df = importance_df.sort_values('mean_abs_shap', ascending=False)
        
        def interpret_shap(row):
            if row['mean_abs_shap'] < 1e-4:
                return "Negligible impact"
            if row['mean_shap'] > 0:
                return "Generally increases non-compliance risk"
            else:
                return "Generally decreases non-compliance risk"
                
        importance_df['interpretation'] = importance_df.apply(interpret_shap, axis=1)
        importance_df.to_csv(os.path.join(reports_dir, 'operational_shap_importance.csv'), index=False)
        
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, pd.DataFrame(X_transformed, columns=shap_feature_names), plot_type="bar", show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(figs_dir, 'operational_shap_bar_plot.png'))
        plt.close()
        
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, pd.DataFrame(X_transformed, columns=shap_feature_names), show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(figs_dir, 'operational_shap_summary_plot.png'))
        plt.close()
        
        sample_exps = []
        for i, idx in enumerate(df.index):
            row_shap = shap_values[i]
            pos_indices = np.argsort(row_shap)[::-1]
            neg_indices = np.argsort(row_shap)
            
            top_pos = [f"{shap_feature_names[j]} ({row_shap[j]:.3f})" for j in pos_indices[:2] if row_shap[j] > 0]
            top_neg = [f"{shap_feature_names[j]} ({row_shap[j]:.3f})" for j in neg_indices[:2] if row_shap[j] < 0]
            
            exp_text = ""
            if top_pos:
                exp_text += f"High values in {', '.join([shap_feature_names[j] for j in pos_indices[:2] if row_shap[j] > 0])} increased the risk. "
            if top_neg:
                exp_text += f"Acceptable levels in {', '.join([shap_feature_names[j] for j in neg_indices[:2] if row_shap[j] < 0])} reduced the risk."
                
            if not exp_text:
                exp_text = "No strong drivers identified."
                
            sample_exps.append({
                'station_name': df.at[idx, 'station_name'] if 'station_name' in df.columns else 'Unknown',
                'river_name': df.at[idx, 'river_name'] if 'river_name' in df.columns else 'Unknown',
                'water_body_type': df.at[idx, 'water_body_type'] if 'water_body_type' in df.columns else 'Unknown',
                'predicted_label': df.at[idx, 'ml_predicted_compliance_label'] if 'ml_predicted_compliance_label' in df.columns else 'Unknown',
                'predicted_non_compliance_probability': df.at[idx, 'predicted_non_compliance_probability'] if 'predicted_non_compliance_probability' in df.columns else np.nan,
                'top_positive_shap_features': " | ".join(top_pos) if top_pos else "None",
                'top_negative_shap_features': " | ".join(top_neg) if top_neg else "None",
                'shap_explanation_text': exp_text.strip()
            })
            
        pd.DataFrame(sample_exps).to_csv(os.path.join(reports_dir, 'operational_shap_sample_explanations.csv'), index=False)
        
        with open(os.path.join(reports_dir, 'operational_shap_summary.md'), 'w') as f:
            f.write("# Operational SHAP Summary\n")
            f.write(f"True SHAP used: Yes (Explainer used based on model type)\n")
            f.write(f"Rows explained: {len(X)}\n")
            f.write(f"Background rows: 10\n")
            f.write(f"Original feature count: {len(features)}\n")
            f.write(f"Transformed feature count: {len(shap_feature_names)}\n")
            f.write(f"Dropped features: {len(features) - len(shap_feature_names)}\n")
            top_features_list = importance_df['feature'].head(5).tolist()
            f.write(f"Top 5 SHAP features: {', '.join(top_features_list)}\n")
            f.write("\nLimitation: Main model uses DO, BOD, and pH, so those may dominate.\n")
        
        logging.info("Operational SHAP explainer completed successfully.")
        
    except Exception as e:
        logging.error(f"SHAP explanation failed: {e}")
        with open(os.path.join(reports_dir, 'operational_shap_failure_report.md'), 'w') as f:
            f.write(f"# Operational SHAP Failed\nError: {e}\n")

if __name__ == "__main__":
    run_operational_shap()
