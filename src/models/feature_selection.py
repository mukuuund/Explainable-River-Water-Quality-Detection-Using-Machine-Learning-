import json
import os
import pandas as pd

def get_core_regulatory_features():
    return [
        'dissolved_oxygen',
        'bod',
        'ph'
    ]

def get_true_auxiliary_features():
    return [
        'temperature',
        'conductivity',
        'nitrate',
        'fecal_coliform',
        'total_coliform',
        'fecal_streptococci',
        'turbidity',
        'cod',
        'total_dissolved_solids',
        'season',
        'station_position_tag',
        'pollution_context_tag',
        'river_name'
    ]

def get_extended_clean_features():
    return get_core_regulatory_features() + get_true_auxiliary_features()

LEAKAGE_KEYWORDS = [
    'compliance', 'label', 'target', 'prediction', 'predicted', 'probability', 
    'risk_score', 'risk_category', 'violation', 'reason', 'class', 'use_based_class',
    'dissolved_oxygen', 'dissolved_o2', 'oxygen', 'do_', '_do', 'bod', 'b_o_d', 
    'biochemical', 'ph', 'p_h', 'hydrogen'
]

def check_leakage(col_name, variant_name):
    """
    Returns (is_rejected, reason)
    """
    col_lower = col_name.lower()
    
    # Exceptions for canonical variables in Core and Extended
    if variant_name in ['Core Regulatory Model', 'Extended Clean Model']:
        if col_lower in ['dissolved_oxygen', 'bod', 'ph']:
            return False, "Allowed canonical parameter"
            
    for kw in LEAKAGE_KEYWORDS:
        if kw in col_lower:
            return True, f"Contains leakage keyword: '{kw}'"
            
    return False, "Clean"

def audit_and_filter_features(df_cols, variant_name, candidate_features):
    """
    Scans candidate_features against the leakage guard.
    Returns the cleaned list of features.
    """
    cleaned_features = []
    audit_records = []
    
    for f in candidate_features:
        if f not in df_cols:
            audit_records.append({'Variant': variant_name, 'Feature': f, 'Action': 'Rejected', 'Reason': 'Not in dataframe'})
            continue
            
        is_rejected, reason = check_leakage(f, variant_name)
        if is_rejected:
            audit_records.append({'Variant': variant_name, 'Feature': f, 'Action': 'Rejected (Leakage)', 'Reason': reason})
        else:
            audit_records.append({'Variant': variant_name, 'Feature': f, 'Action': 'Accepted', 'Reason': reason})
            cleaned_features.append(f)
            
    return cleaned_features, audit_records

def filter_features(df, allowed_features):
    """Legacy function, kept for backward compatibility."""
    return [col for col in allowed_features if col in df.columns]

def save_feature_list(features, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        json.dump(features, f, indent=4)
        
def save_feature_comparison_csv(features_dict, output_path):
    all_features = set()
    for flist in features_dict.values():
        all_features.update(flist)
        
    records = []
    for f in all_features:
        row = {'Feature': f}
        for model_name, flist in features_dict.items():
            row[model_name] = 'Yes' if f in flist else 'No'
        records.append(row)
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pd.DataFrame(records).to_csv(output_path, index=False)
