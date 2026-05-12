import pandas as pd
import numpy as np
import os
import re
from src.models.train_baseline_models import get_allowed_features

def get_true_auxiliary_features(df):
    blacklist_patterns = [
        'do', 'dissolved', 'oxygen', 'bod', 'biochemical', 'ph', 'p_h',
        'minmax', 'zscore', 'scaled', 'normalized', 'encoded', 'label',
        'risk', 'score', 'violation', 'confidence', 'available', 'missing',
        'class_weight', 'target', 'safe', 'prediction', 'predicted', 'category'
    ]
    
    allowed = [
        'temperature', 'conductivity', 'nitrate', 'fecal_coliform',
        'total_coliform', 'fecal_streptococci', 'turbidity', 'cod',
        'total_dissolved_solids', 'season', 'station_position_tag',
        'pollution_context_tag', 'river_name'
    ]
    
    features = []
    for col in df.columns:
        col_lower = col.lower()
        if any(p in col_lower for p in blacklist_patterns):
            continue
        # Only allow strictly auxiliary explicitly requested or strictly matched
        if any(a in col_lower for a in allowed):
            features.append(col)
            
    return features

def audit_features():
    df_raw = pd.read_csv('data/processed/model_ready_phase1.csv')
    df_exp_b = df_raw[(df_raw['label_confidence'].isin(['High', 'Medium'])) & (df_raw['available_compliance_label'].isin(['Compliant_Based_On_Available_Parameters', 'Compliant', 'Non-Compliant']))].copy()
    
    features_full = get_allowed_features(df_exp_b)
    features_abl = get_allowed_features(df_exp_b, drop_core=True)
    features_aux = get_true_auxiliary_features(df_exp_b)
    
    os.makedirs('reports/model_results', exist_ok=True)
    pd.DataFrame({'Feature': features_full}).to_csv('reports/model_results/final_feature_list_full.csv', index=False)
    pd.DataFrame({'Feature': features_abl}).to_csv('reports/model_results/final_feature_list_ablation.csv', index=False)
    pd.DataFrame({'Feature': features_aux}).to_csv('reports/model_results/final_feature_list_auxiliary_only.csv', index=False)
    
    leakage_patterns = [
        'compliance', 'safe', 'label', 'target', 'violation', 'risk', 'score',
        'category', 'confidence', 'class', 'weight', 'encoded', 'minmax',
        'zscore', 'scaled', 'normalized', 'prediction', 'predicted', 'available', 'missing'
    ]
    core_hidden_patterns = [
        'do', 'dissolved', 'oxygen', 'bod', 'biochemical', 'ph', 'p_h'
    ]
    
    report = []
    
    def check_list(feat_list, name, check_core):
        for f in feat_list:
            f_lower = f.lower()
            leak = any(p in f_lower for p in leakage_patterns)
            core_leak = any(p in f_lower for p in core_hidden_patterns) if check_core else False
            
            if leak or core_leak:
                report.append({
                    'Experiment': name,
                    'Feature': f,
                    'Leakage_Warning': 'General Leakage' if leak else 'Hidden Core Leakage'
                })
                
    check_list(features_full, 'Full Model', check_core=False)
    check_list(features_abl, 'Ablation Model', check_core=True)
    check_list(features_aux, 'True Auxiliary Model', check_core=True)
    
    report_df = pd.DataFrame(report)
    if not report_df.empty:
        report_df.to_csv('reports/model_results/feature_leakage_audit.csv', index=False)
    else:
        pd.DataFrame(columns=['Experiment', 'Feature', 'Leakage_Warning']).to_csv('reports/model_results/feature_leakage_audit.csv', index=False)
        
    return len(report_df) == 0

if __name__ == "__main__":
    passed = audit_features()
    print("Feature Audit Passed" if passed else "Feature Audit Failed")
