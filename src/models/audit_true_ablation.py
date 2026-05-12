import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, f1_score
from src.models.train_baseline_models import build_preprocessing_pipeline
from src.models.audit_feature_leakage import get_true_auxiliary_features

def run_evaluation(X, y, train_idx, test_idx, num_cols, cat_cols, split_name):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    if len(y_test.unique()) < 2:
        return None
        
    model = DecisionTreeClassifier(max_depth=5, min_samples_leaf=10, class_weight='balanced', random_state=42)
    preprocessor = build_preprocessing_pipeline(num_cols, cat_cols, is_linear=False)
    clf = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
    
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    try:
        y_prob = clf.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_prob)
    except:
        roc_auc = np.nan
        
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    return {
        'Split_Type': split_name,
        'Model': 'DecisionTree_Restricted',
        'Balanced_Accuracy': bal_acc,
        'ROC_AUC': roc_auc,
        'F1_NonCompliant': f1
    }

def audit_true_ablation():
    df_raw = pd.read_csv('data/processed/model_ready_phase1.csv')
    df_exp_b = df_raw[(df_raw['label_confidence'].isin(['High', 'Medium'])) & (df_raw['available_compliance_label'].isin(['Compliant_Based_On_Available_Parameters', 'Compliant', 'Non-Compliant']))].copy()
    
    df_exp_b['target'] = (df_exp_b['available_compliance_label'] == 'Non-Compliant').astype(int)
    features = get_true_auxiliary_features(df_exp_b)
    
    # Reset index
    df_exp_b = df_exp_b.reset_index(drop=True)
    
    X = df_exp_b[features]
    y = df_exp_b['target']
    
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    results = []
    
    # 1. Random Split
    stratify_col = y if y.value_counts().min() >= 5 else None
    train_idx, test_idx = train_test_split(df_exp_b.index, test_size=0.2, random_state=42, stratify=stratify_col)
    res_random = run_evaluation(X, y, train_idx, test_idx, num_cols, cat_cols, 'Random_Split')
    if res_random:
        results.append(res_random)
        
    # 2. Group Split
    if 'station_name' in df_exp_b.columns and df_exp_b['station_name'].nunique() > 10:
        groups = df_exp_b['station_name']
    elif 'river_name' in df_exp_b.columns and df_exp_b['river_name'].nunique() > 5:
        groups = df_exp_b['river_name']
    else:
        groups = df_exp_b['source_file']
        
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    try:
        train_idx_g, test_idx_g = next(gss.split(X, y, groups))
        res_group = run_evaluation(X, y, train_idx_g, test_idx_g, num_cols, cat_cols, 'Group_Split')
        if res_group:
            results.append(res_group)
    except ValueError:
        pass
        
    report_df = pd.DataFrame(results)
    os.makedirs('reports/model_results', exist_ok=True)
    report_df.to_csv('reports/model_results/true_auxiliary_ablation_metrics.csv', index=False)

if __name__ == "__main__":
    audit_true_ablation()
    print("True Auxiliary Ablation Audit Completed")
