import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from src.models.train_baseline_models import get_allowed_features

def audit_duplicates():
    df_raw = pd.read_csv('data/processed/model_ready_phase1.csv')
    df_exp_b = df_raw[(df_raw['label_confidence'].isin(['High', 'Medium'])) & (df_raw['available_compliance_label'].isin(['Compliant_Based_On_Available_Parameters', 'Compliant', 'Non-Compliant']))].copy()
    
    df_exp_b['target'] = (df_exp_b['available_compliance_label'] == 'Non-Compliant').astype(int)
    features = get_allowed_features(df_exp_b)
    
    # Reset index to track rows
    df_exp_b = df_exp_b.reset_index(drop=True)
    df_exp_b['original_index'] = df_exp_b.index
    
    stratify_col = df_exp_b['target'] if df_exp_b['target'].value_counts().min() >= 5 else None
    
    train_df, test_df = train_test_split(df_exp_b, test_size=0.2, random_state=42, stratify=stratify_col)
    
    report = {}
    
    report['Total_Rows'] = len(df_exp_b)
    report['Full_Row_Duplicates'] = df_exp_b.drop(columns=['original_index']).duplicated().sum()
    report['Feature_Only_Duplicates'] = df_exp_b.duplicated(subset=features).sum()
    report['Feature_Target_Duplicates'] = df_exp_b.duplicated(subset=features + ['target']).sum()
    
    # Train/Test contamination
    train_features = train_df[features].drop_duplicates()
    test_features = test_df[features].drop_duplicates()
    
    # Find intersecting feature rows
    # We can merge them to find exact matches
    intersection = pd.merge(train_features, test_features, how='inner')
    report['Train_Test_Contamination_Count'] = len(intersection)
    
    # Check if they are harmful or harmless
    # Harmless: duplicates only within train or test (handled by Feature_Only_Duplicates)
    # Harmful: same feature in both train and test (Train_Test_Contamination_Count)
    # Label Collision: same feature row with different targets
    
    # Label collisions across the whole dataset
    feat_grouped = df_exp_b.groupby(features)['target'].nunique()
    report['Label_Collisions_Count'] = (feat_grouped > 1).sum()
    
    report_df = pd.DataFrame([report])
    os.makedirs('reports/model_results', exist_ok=True)
    report_df.to_csv('reports/model_results/duplicate_audit_report.csv', index=False)
    
    # Audit passes if there's no harmful train/test contamination (or it's extremely small)
    return report['Train_Test_Contamination_Count'] == 0

if __name__ == "__main__":
    passed = audit_duplicates()
    print("Duplicate Audit Passed" if passed else "Duplicate Audit Failed (Train/Test Contamination exists)")
