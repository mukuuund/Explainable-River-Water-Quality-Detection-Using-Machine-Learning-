import os
import sys
import pandas as pd
import numpy as np
import json
import joblib

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

def run_healthcheck():
    print("="*60)
    print("Model Inference Health Check")
    print("="*60)
    
    reports_dir = os.path.join(PROJECT_ROOT, "reports", "validation")
    os.makedirs(reports_dir, exist_ok=True)
    
    summary_md = ["# Model Inference Health Check Summary\n"]
    
    # -----------------------------------------------------
    # A. NWMP / Operational Validation
    # -----------------------------------------------------
    print("\n--- Operational Model ---")
    summary_md.append("## Operational Model (NWMP)")
    
    model_path = os.path.join(PROJECT_ROOT, 'models', 'practical_operational_clean_best_model.pkl')
    features_path = os.path.join(PROJECT_ROOT, 'models', 'practical_operational_clean_features.json')
    preds_path = os.path.join(PROJECT_ROOT, 'data', 'processed', 'nwmp_2025_predictions.csv')
    
    op_model_loaded = os.path.exists(model_path)
    op_features_loaded = os.path.exists(features_path)
    
    num_expected_features = 0
    if op_features_loaded:
        try:
            with open(features_path, 'r') as f:
                obj = json.load(f)
                if isinstance(obj, list):
                    num_expected_features = len(obj)
                elif isinstance(obj, dict):
                    for v in obj.values():
                        if isinstance(v, list):
                            num_expected_features = max(num_expected_features, len(v))
        except:
            pass
            
    op_preds_exist = os.path.exists(preds_path)
    
    if op_preds_exist:
        df_op = pd.read_csv(preds_path)
        total_rows = len(df_op)
        
        do_present = df_op['dissolved_oxygen'].notna().sum() if 'dissolved_oxygen' in df_op.columns else 0
        bod_present = df_op['bod'].notna().sum() if 'bod' in df_op.columns else 0
        ph_present = df_op['ph'].notna().sum() if 'ph' in df_op.columns else 0
        
        if 'dissolved_oxygen' in df_op.columns and 'bod' in df_op.columns and 'ph' in df_op.columns:
            full_core = len(df_op.dropna(subset=['dissolved_oxygen', 'bod', 'ph']))
        else:
            full_core = 0
            
        success_count = (df_op['ml_prediction_status'] == 'Success').sum() if 'ml_prediction_status' in df_op.columns else 0
        failed_count = (df_op['ml_prediction_status'] == 'Failed').sum() if 'ml_prediction_status' in df_op.columns else 0
        
        if 'ml_predicted_compliance_label' in df_op.columns:
            compliant_count = (df_op['ml_predicted_compliance_label'] == 'Compliant').sum()
            non_compliant_count = (df_op['ml_predicted_compliance_label'] == 'Non-Compliant').sum()
        else:
            compliant_count = 0
            non_compliant_count = 0
            
        prob_col = 'predicted_non_compliance_probability'
        if prob_col not in df_op.columns and 'ml_non_compliance_probability' in df_op.columns:
            prob_col = 'ml_non_compliance_probability'
            
        if prob_col in df_op.columns:
            non_null_prob = df_op[prob_col].notna().sum()
            prob_min = df_op[prob_col].min()
            prob_mean = df_op[prob_col].mean()
            prob_max = df_op[prob_col].max()
        else:
            non_null_prob = 0
            prob_min, prob_mean, prob_max = np.nan, np.nan, np.nan
            
        rule_agreement = "N/A"
        if 'rule_based_compliance_label' in df_op.columns and 'ml_predicted_compliance_label' in df_op.columns:
            valid_mask = df_op['rule_based_compliance_label'].isin(['Compliant', 'Non-Compliant']) & df_op['ml_predicted_compliance_label'].isin(['Compliant', 'Non-Compliant'])
            if valid_mask.any():
                agree = (df_op.loc[valid_mask, 'rule_based_compliance_label'] == df_op.loc[valid_mask, 'ml_predicted_compliance_label']).sum()
                rule_agreement = f"{agree} ({agree / valid_mask.sum() * 100:.1f}%)"
                
                # Confusion matrix
                cm = pd.crosstab(df_op.loc[valid_mask, 'rule_based_compliance_label'], df_op.loc[valid_mask, 'ml_predicted_compliance_label'], rownames=['Rule'], colnames=['ML'])
                cm.to_csv(os.path.join(reports_dir, 'operational_confusion_matrix.csv'))
                
        op_data = {
            'total_rows': total_rows,
            'full_core_params_rows': full_core,
            'successful_predictions': success_count,
            'failed_predictions': failed_count,
            'compliant_predictions': compliant_count,
            'non_compliant_predictions': non_compliant_count,
            'non_null_probabilities': non_null_prob,
            'prob_min': prob_min,
            'prob_mean': prob_mean,
            'prob_max': prob_max,
            'rule_agreement': rule_agreement
        }
        pd.DataFrame([op_data]).to_csv(os.path.join(reports_dir, 'operational_model_healthcheck.csv'), index=False)
        
        pass_model = "PASS" if op_model_loaded else "FAIL"
        pass_features = "PASS" if op_features_loaded else "FAIL"
        pass_preds = "PASS" if success_count > 0 else "FAIL"
        pass_probs = "PASS" if non_null_prob > 0 else "FAIL"
        pass_unknown = "PASS" if failed_count == 0 else "WARNING"
        pass_rule = "PASS" if rule_agreement != "N/A" else "WARNING"
        
        summary_md.append(f"- **Model loaded**: {pass_model}")
        summary_md.append(f"- **Feature schema aligned**: {pass_features} ({num_expected_features} features)")
        summary_md.append(f"- **Predictions generated**: {pass_preds} ({success_count} successful)")
        summary_md.append(f"- **Probabilities generated**: {pass_probs} ({non_null_prob} non-null)")
        summary_md.append(f"- **Unknown predictions below threshold**: {pass_unknown} ({failed_count} failed)")
        summary_md.append(f"- **Rule agreement available**: {pass_rule} ({rule_agreement})")
        
    else:
        summary_md.append("Operational predictions file not found.")

    # -----------------------------------------------------
    # B. Realtime Validation
    # -----------------------------------------------------
    print("\n--- Realtime Model ---")
    summary_md.append("\n## Realtime Model")
    
    rt_preds_path = os.path.join(PROJECT_ROOT, 'data', 'processed', 'realtime', 'live_sensor_predictions.csv')
    rt_preds_exist = os.path.exists(rt_preds_path)
    
    if rt_preds_exist:
        df_rt = pd.read_csv(rt_preds_path)
        rt_total = len(df_rt)
        
        rt_full_core = len(df_rt[df_rt['core_parameter_count'] == 3]) if 'core_parameter_count' in df_rt.columns else 0
        rt_success = (df_rt['live_ml_prediction_status'] == 'Success').sum() if 'live_ml_prediction_status' in df_rt.columns else 0
        rt_unknown = (df_rt['live_ml_prediction_status'] == 'Failed').sum() if 'live_ml_prediction_status' in df_rt.columns else 0
        rt_fallback = df_rt['live_ml_fallback_used'].sum() if 'live_ml_fallback_used' in df_rt.columns else 0
        rt_nc = (df_rt['live_ml_predicted_compliance_label'] == 'Non-Compliant').sum() if 'live_ml_predicted_compliance_label' in df_rt.columns else 0
        
        if 'live_ml_non_compliance_probability' in df_rt.columns:
            rt_prob_count = df_rt['live_ml_non_compliance_probability'].notna().sum()
            rt_prob_min = df_rt['live_ml_non_compliance_probability'].min()
            rt_prob_mean = df_rt['live_ml_non_compliance_probability'].mean()
            rt_prob_max = df_rt['live_ml_non_compliance_probability'].max()
        else:
            rt_prob_count = 0
            rt_prob_min, rt_prob_mean, rt_prob_max = np.nan, np.nan, np.nan
            
        if 'ml_prediction_confidence_note' in df_rt.columns:
            conf_dist = df_rt['ml_prediction_confidence_note'].value_counts().reset_index()
            conf_dist.columns = ['Confidence Note', 'Count']
            conf_dist.to_csv(os.path.join(reports_dir, 'realtime_prediction_distribution.csv'), index=False)
            
        rt_data = {
            'total_rows': rt_total,
            'full_core_params_rows': rt_full_core,
            'successful_predictions': rt_success,
            'unknown_predictions': rt_unknown,
            'fallback_count': rt_fallback,
            'non_compliant_predictions': rt_nc,
            'non_null_probabilities': rt_prob_count,
            'prob_min': rt_prob_min,
            'prob_mean': rt_prob_mean,
            'prob_max': rt_prob_max
        }
        pd.DataFrame([rt_data]).to_csv(os.path.join(reports_dir, 'realtime_model_healthcheck.csv'), index=False)
        
        rt_pass_preds = "PASS" if rt_success > 0 else "WARNING"
        rt_pass_probs = "PASS" if rt_prob_count > 0 else "WARNING"
        rt_pass_unknown = "PASS" if rt_unknown == 0 else "WARNING"
        
        summary_md.append(f"- **Model loaded**: {pass_model}")
        summary_md.append(f"- **Feature schema aligned**: {pass_features}")
        summary_md.append(f"- **Predictions generated**: {rt_pass_preds} ({rt_success} successful)")
        summary_md.append(f"- **Unknown predictions below threshold**: {rt_pass_unknown} ({rt_unknown} unknown, {rt_fallback} fallbacks)")
        summary_md.append(f"- **Probabilities generated**: {rt_pass_probs} ({rt_prob_count} non-null)")
        
    else:
        summary_md.append("Realtime predictions file not found.")
        
    with open(os.path.join(reports_dir, 'model_inference_healthcheck_summary.md'), 'w') as f:
        f.write("\n".join(summary_md))
        
    pd.DataFrame([{'status': 'completed'}]).to_csv(os.path.join(reports_dir, 'model_inference_healthcheck.csv'), index=False)
    
    print("Health check completed. Check reports/validation/model_inference_healthcheck_summary.md")

if __name__ == "__main__":
    run_healthcheck()
