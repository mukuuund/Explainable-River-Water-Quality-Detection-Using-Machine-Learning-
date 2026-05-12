import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GroupShuffleSplit, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.base import clone

from src.models.feature_selection import (
    get_core_regulatory_features,
    get_extended_clean_features,
    get_true_auxiliary_features,
    audit_and_filter_features,
    save_feature_list,
    save_feature_comparison_csv
)

def build_preprocessing_pipeline(num_cols, cat_cols, is_linear=False, is_tree_based=False):
    transformers = []
    
    if num_cols:
        num_steps = [('imputer', SimpleImputer(strategy='median'))]
        if is_linear:
            num_steps.append(('scaler', StandardScaler()))
        transformers.append(('num', Pipeline(steps=num_steps), num_cols))
        
    if cat_cols:
        cat_steps = [
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]
        transformers.append(('cat', Pipeline(steps=cat_steps), cat_cols))
        
    if not transformers:
        # Fallback if both are empty somehow (shouldn't happen)
        return 'passthrough'
        
    preprocessor = ColumnTransformer(transformers=transformers)
    return preprocessor

def evaluate_pipeline(clf, X_test, y_test, variant_name, split_name, model_name):
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
    
    return {
        'Variant': variant_name,
        'Split_Type': split_name,
        'Model': model_name,
        'Accuracy': acc,
        'Balanced_Accuracy': bal_acc,
        'Precision_NonCompliant': prec,
        'Recall_NonCompliant': rec,
        'F1_NonCompliant': f1,
        'ROC_AUC': roc_auc
    }, confusion_matrix(y_test, y_pred)

def train_phase3_7():
    print("Loading data...")
    df_raw = pd.read_csv('data/processed/model_ready_phase1.csv')
    
    # Filter for valid targets
    if 'available_compliance_label' in df_raw.columns:
        df_exp_b = df_raw[(df_raw['available_compliance_label'].isin(['Compliant_Based_On_Available_Parameters', 'Compliant', 'Non-Compliant']))].copy()
        df_exp_b['target'] = (df_exp_b['available_compliance_label'] == 'Non-Compliant').astype(int)
    else:
        # Fallback to strict compliance label or something if available
        pass # Assuming phase1 data has it.
        
    df_exp_b = df_exp_b.reset_index(drop=True)
    
    variants = {
        'Core Regulatory Model': get_core_regulatory_features(),
        'Extended Clean Model': get_extended_clean_features(),
        'True Auxiliary-Only Model': get_true_auxiliary_features()
    }
    
    # Models to compare
    base_models = {
        'DummyClassifier': DummyClassifier(strategy='prior'),
        'LogisticRegression': LogisticRegression(class_weight='balanced', max_iter=2000, random_state=42),
        'DecisionTree': DecisionTreeClassifier(class_weight='balanced', max_depth=7, random_state=42),
        'RandomForest': RandomForestClassifier(class_weight='balanced', n_estimators=100, max_depth=7, random_state=42),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42),
        'HistGradientBoosting': HistGradientBoostingClassifier(random_state=42, max_iter=100, max_depth=7)
    }
    
    if 'station_name' in df_exp_b.columns and df_exp_b['station_name'].nunique() > 5:
        groups = df_exp_b['station_name']
    elif 'river_name' in df_exp_b.columns and df_exp_b['river_name'].nunique() > 5:
        groups = df_exp_b['river_name']
    else:
        groups = df_exp_b['source_file']
        
    random_metrics = []
    group_metrics = []
    all_audit_records = []
    actual_features_used = {}
    
    best_cm_dict_group = {}
    
    best_pipelines = {
        'Core Regulatory Model': None,
        'Extended Clean Model': None,
        'True Auxiliary-Only Model': None
    }
    
    df_cols = df_exp_b.columns.tolist()
    
    known_cats = ['season', 'river_name', 'station_position_tag', 'pollution_context_tag']
    for c in known_cats:
        if c in df_cols:
            df_exp_b[c] = df_exp_b[c].astype(str).replace(['nan', 'NaN', 'None'], np.nan)
            df_exp_b[c] = df_exp_b[c].fillna('Unknown')
    
    for variant_name, candidate_feat in variants.items():
        print(f"\\nProcessing {variant_name}...")
        
        features, audit_rec = audit_and_filter_features(df_cols, variant_name, candidate_feat)
        all_audit_records.extend(audit_rec)
        actual_features_used[variant_name] = features
        
        X = df_exp_b[features]
        y = df_exp_b['target']
        
        cat_cols = [c for c in features if c in ['season', 'river_name', 'station_position_tag', 'pollution_context_tag']]
        num_cols = [c for c in features if c not in cat_cols]
        
        # 1. Random Split Evaluation
        stratify_col = y if y.value_counts().min() >= 5 else None
        train_idx_r, test_idx_r = train_test_split(df_exp_b.index, test_size=0.2, random_state=42, stratify=stratify_col)
        X_train_r, X_test_r = X.iloc[train_idx_r], X.iloc[test_idx_r]
        y_train_r, y_test_r = y.iloc[train_idx_r], y.iloc[test_idx_r]
        
        # 2. Group Split Evaluation (more realistic)
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        try:
            train_idx_g, test_idx_g = next(gss.split(X, y, groups))
            X_train_g, X_test_g = X.iloc[train_idx_g], X.iloc[test_idx_g]
            y_train_g, y_test_g = y.iloc[train_idx_g], y.iloc[test_idx_g]
        except ValueError:
            # Fallback to random if group fails
            X_train_g, X_test_g, y_train_g, y_test_g = X_train_r, X_test_r, y_train_r, y_test_r
            
        # We will use the group split train set for tuning the Auxiliary model
        best_f1_group = -1
        
        for name, model in base_models.items():
            is_linear = 'LogisticRegression' in name
            preprocessor = build_preprocessing_pipeline(num_cols, cat_cols, is_linear=is_linear)
            
            # --- Hyperparameter Tuning only for Auxiliary-Only model using CV on train set ---
            if variant_name == 'True Auxiliary-Only Model' and name in ['RandomForest', 'HistGradientBoosting', 'GradientBoosting', 'LogisticRegression']:
                print(f"Tuning {name} for Auxiliary-Only model...")
                clf = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', clone(model))])
                
                param_grid = {}
                if name == 'RandomForest':
                    param_grid = {
                        'classifier__n_estimators': [50, 100],
                        'classifier__max_depth': [5, 10, None],
                        'classifier__min_samples_leaf': [1, 5, 10]
                    }
                elif name in ['GradientBoosting', 'HistGradientBoosting']:
                    param_grid = {
                        'classifier__max_depth': [3, 5, 7],
                        'classifier__learning_rate': [0.01, 0.1]
                    }
                elif name == 'LogisticRegression':
                    param_grid = {
                        'classifier__C': [0.1, 1.0, 10.0]
                    }
                
                cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
                grid_search = GridSearchCV(clf, param_grid, cv=cv, scoring='f1', n_jobs=-1)
                
                # Tune on the group split train set
                grid_search.fit(X_train_g, y_train_g)
                best_clf = grid_search.best_estimator_
            else:
                best_clf = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', clone(model))])
                best_clf.fit(X_train_g, y_train_g) # Fit on group train for best_clf selection
                
            # Random split eval
            clf_r = clone(best_clf)
            clf_r.fit(X_train_r, y_train_r)
            res_r, _ = evaluate_pipeline(clf_r, X_test_r, y_test_r, variant_name, 'Random_Split', name)
            random_metrics.append(res_r)
            
            # Group split eval (The most realistic one)
            res_g, cm_g = evaluate_pipeline(best_clf, X_test_g, y_test_g, variant_name, 'Group_Split', name)
            group_metrics.append(res_g)
            
            if res_g['F1_NonCompliant'] > best_f1_group:
                best_f1_group = res_g['F1_NonCompliant']
                best_pipelines[variant_name] = best_clf
                best_cm_dict_group[variant_name] = cm_g
                
    # Save outputs
    print("\\nSaving outputs...")
    os.makedirs('reports/model_results', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Audit report
    pd.DataFrame(all_audit_records).to_csv('reports/model_results/phase3_7_leakage_audit.csv', index=False)
    
    save_feature_comparison_csv(actual_features_used, 'reports/model_results/phase3_7_feature_sets_used.csv')
    
    df_random = pd.DataFrame(random_metrics)
    df_random.to_csv('reports/model_results/phase3_7_random_split_metrics.csv', index=False)
    
    df_group = pd.DataFrame(group_metrics)
    df_group.to_csv('reports/model_results/phase3_7_group_split_metrics.csv', index=False)
    
    df_all = pd.concat([df_random, df_group], ignore_index=True)
    df_all.to_csv('reports/model_results/phase3_7_clean_model_metrics.csv', index=False)
    
    # Plot confusion matrices
    fig, axes = plt.subplots(1, len(best_cm_dict_group), figsize=(5 * len(best_cm_dict_group), 4))
    if len(best_cm_dict_group) == 1:
        axes = [axes]
        
    for ax, (title, cm) in zip(axes, best_cm_dict_group.items()):
        if cm is not None:
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                        xticklabels=['Compliant (0)', 'Non-Compliant (1)'], 
                        yticklabels=['Compliant (0)', 'Non-Compliant (1)'])
            ax.set_title(title)
            ax.set_ylabel('True Label')
            ax.set_xlabel('Predicted Label')
            
    plt.tight_layout()
    plt.savefig('reports/model_results/phase3_7_confusion_matrices.png')
    plt.close()
    
    # Save best models
    file_names = {
        'Core Regulatory Model': ('core_regulatory_model.pkl', 'core_regulatory_features.json'),
        'Extended Clean Model': ('extended_clean_operational_model.pkl', 'extended_clean_operational_features.json'),
        'True Auxiliary-Only Model': ('auxiliary_only_leakage_safe_model.pkl', 'auxiliary_only_leakage_safe_features.json')
    }
    
    for variant, pipe in best_pipelines.items():
        if pipe is not None:
            model_file, feat_file = file_names[variant]
            joblib.dump(pipe, f'models/{model_file}')
            save_feature_list(actual_features_used[variant], f'models/{feat_file}')
            
    # Also save Extended to practical operational for backward compat if needed
    if best_pipelines['Extended Clean Model'] is not None:
        joblib.dump(best_pipelines['Extended Clean Model'], 'models/practical_operational_clean_best_model.pkl')
        save_feature_list(actual_features_used['Extended Clean Model'], 'models/practical_operational_clean_features.json')

    # Summary
    best_summary = df_group.loc[df_group.groupby('Variant')['F1_NonCompliant'].idxmax()]
    best_summary.to_csv('reports/model_results/phase3_7_best_model_summary.csv', index=False)
    
    print("Training complete. Results saved to reports/model_results/ and models/.")
    
if __name__ == "__main__":
    train_phase3_7()
