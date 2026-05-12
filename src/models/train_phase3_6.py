import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

from src.models.feature_selection import (
    get_core_regulatory_features,
    get_extended_clean_features,
    get_true_auxiliary_features,
    filter_features,
    save_feature_list,
    save_feature_comparison_csv
)

def build_preprocessing_pipeline(num_cols, cat_cols, is_linear=False):
    num_steps = [('imputer', SimpleImputer(strategy='median'))]
    if is_linear:
        num_steps.append(('scaler', StandardScaler()))
    num_transformer = Pipeline(steps=num_steps)
    
    cat_steps = [
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ]
    cat_transformer = Pipeline(steps=cat_steps)
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_cols),
            ('cat', cat_transformer, cat_cols)
        ]
    )
    return preprocessor

from sklearn.base import clone

def run_evaluation(X, y, train_idx, test_idx, num_cols, cat_cols, model_dict, variant_name, split_name):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    if len(y_test.unique()) < 2:
        return [], None, None
        
    results = []
    best_f1 = -1
    best_model_name = None
    best_pipeline = None
    best_cm = None
    
    for name, model in model_dict.items():
        is_linear = 'LogisticRegression' in name
        preprocessor = build_preprocessing_pipeline(num_cols, cat_cols, is_linear)
        clf = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', clone(model))])
        
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        try:
            y_prob = clf.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, y_prob)
        except:
            roc_auc = np.nan
            
        acc = accuracy_score(y_test, y_pred)
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        results.append({
            'Variant': variant_name,
            'Split_Type': split_name,
            'Model': name,
            'Accuracy': acc,
            'Balanced_Accuracy': bal_acc,
            'Precision_NonCompliant': prec,
            'Recall_NonCompliant': rec,
            'F1_NonCompliant': f1,
            'ROC_AUC': roc_auc
        })
        
        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name
            best_pipeline = clf
            best_cm = confusion_matrix(y_test, y_pred)
            
    return results, best_pipeline, best_cm

def plot_confusion_matrices(cm_dict, output_path):
    fig, axes = plt.subplots(1, len(cm_dict), figsize=(5 * len(cm_dict), 4))
    if len(cm_dict) == 1:
        axes = [axes]
        
    for ax, (title, cm) in zip(axes, cm_dict.items()):
        if cm is not None:
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                        xticklabels=['Compliant (0)', 'Non-Compliant (1)'], 
                        yticklabels=['Compliant (0)', 'Non-Compliant (1)'])
            ax.set_title(title)
            ax.set_ylabel('True Label')
            ax.set_xlabel('Predicted Label')
            
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def train_phase3_6():
    df_raw = pd.read_csv('data/processed/model_ready_phase1.csv')
    df_exp_b = df_raw[(df_raw['label_confidence'].isin(['High', 'Medium'])) & (df_raw['available_compliance_label'].isin(['Compliant_Based_On_Available_Parameters', 'Compliant', 'Non-Compliant']))].copy()
    
    df_exp_b['target'] = (df_exp_b['available_compliance_label'] == 'Non-Compliant').astype(int)
    df_exp_b = df_exp_b.reset_index(drop=True)
    
    variants = {
        'Core Regulatory Model': get_core_regulatory_features(),
        'Extended Clean Model': get_extended_clean_features(),
        'True Auxiliary-Only Model': get_true_auxiliary_features()
    }
    
    models = {
        'DummyClassifier': DummyClassifier(strategy='prior'),
        'LogisticRegression': LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
        'DecisionTree (depth=5)': DecisionTreeClassifier(max_depth=5, min_samples_leaf=10, class_weight='balanced', random_state=42),
        'RandomForest (depth=5)': RandomForestClassifier(max_depth=5, min_samples_leaf=10, class_weight='balanced', random_state=42),
        'GradientBoosting (depth=5)': GradientBoostingClassifier(max_depth=5, random_state=42)
    }
    
    # Check grouping availability
    if 'station_name' in df_exp_b.columns and df_exp_b['station_name'].nunique() > 10:
        groups = df_exp_b['station_name']
    elif 'river_name' in df_exp_b.columns and df_exp_b['river_name'].nunique() > 5:
        groups = df_exp_b['river_name']
    else:
        # Fallback combination if river and station exist but are too sparse alone? Let's just use source_file
        groups = df_exp_b['source_file']
        
    random_metrics = []
    group_metrics = []
    
    # Save feature lists
    actual_features_used = {}
    
    # Track the best pipeline for Extended Clean Model (from Group Split, as requested)
    best_pipeline_extended = None
    best_cm_dict = {}
    
    for variant_name, allowed_feat in variants.items():
        features = filter_features(df_exp_b, allowed_feat)
        actual_features_used[variant_name] = features
        
        X = df_exp_b[features]
        y = df_exp_b['target']
        
        num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Random Split
        stratify_col = y if y.value_counts().min() >= 5 else None
        train_idx_r, test_idx_r = train_test_split(df_exp_b.index, test_size=0.2, random_state=42, stratify=stratify_col)
        
        res_r, best_pipe_r, cm_r = run_evaluation(X, y, train_idx_r, test_idx_r, num_cols, cat_cols, models, variant_name, 'Random_Split')
        random_metrics.extend(res_r)
        
        # Group Split
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        try:
            train_idx_g, test_idx_g = next(gss.split(X, y, groups))
            res_g, best_pipe_g, cm_g = run_evaluation(X, y, train_idx_g, test_idx_g, num_cols, cat_cols, models, variant_name, 'Group_Split')
            group_metrics.extend(res_g)
            
            best_cm_dict[variant_name] = cm_g
            if variant_name == 'Extended Clean Model':
                best_pipeline_extended = best_pipe_g
        except ValueError:
            pass
            
    # Save outputs
    os.makedirs('reports/model_results', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    save_feature_comparison_csv(actual_features_used, 'reports/model_results/phase3_6_feature_sets_used.csv')
    
    df_random = pd.DataFrame(random_metrics)
    df_random.to_csv('reports/model_results/phase3_6_random_split_metrics.csv', index=False)
    
    df_group = pd.DataFrame(group_metrics)
    df_group.to_csv('reports/model_results/phase3_6_group_split_metrics.csv', index=False)
    
    df_all = pd.concat([df_random, df_group], ignore_index=True)
    df_all.to_csv('reports/model_results/phase3_6_clean_model_metrics.csv', index=False)
    
    plot_confusion_matrices(best_cm_dict, 'reports/model_results/phase3_6_confusion_matrices.png')
    
    if best_pipeline_extended:
        joblib.dump(best_pipeline_extended, 'models/practical_operational_clean_best_model.pkl')
        save_feature_list(actual_features_used['Extended Clean Model'], 'models/practical_operational_clean_features.json')
        
    return df_all

if __name__ == "__main__":
    train_phase3_6()
