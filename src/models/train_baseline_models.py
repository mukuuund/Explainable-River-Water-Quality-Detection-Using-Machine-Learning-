import pandas as pd
import numpy as np
import os
import joblib
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def get_allowed_features(df, drop_core=False):
    leakage_keywords = [
        'strict_compliance_label', 'available_compliance_label', 'compliance_label',
        'is_compliant', 'is_safe', 'violation_reasons', 'violation_count',
        'risk_score', 'risk_category', 'label_confidence', 'risk_confidence',
        'available_compliance_parameters', 'missing_required_parameters',
        'class_weight', 'binary_class_weight', 'compliance_label_encoded',
        'source_file', '_minmax', '_zscore', '_encoded'
    ]
    
    features = []
    for col in df.columns:
        if any(keyword in col.lower() for keyword in leakage_keywords):
            continue
        if drop_core and col in ['dissolved_oxygen', 'bod', 'ph']:
            continue
        features.append(col)
    return features

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

def run_experiment(df, experiment_name, target_col, features, output_dir, save_model=True):
    logging.info(f"\\n--- Running {experiment_name} ---")
    
    # Check class distribution
    class_counts = df[target_col].value_counts()
    logging.info(f"Total rows: {len(df)}")
    logging.info(f"Target distribution:\\n{class_counts}")
    
    if len(class_counts) < 2:
        logging.warning(f"Experiment {experiment_name} has less than 2 classes. Skipping training.")
        return None
    
    if class_counts.min() < 5:
        logging.warning(f"Experiment {experiment_name} has a class with < 5 samples. This might fail CV/Split.")
        # Proceed with caution, might need simple train_test_split without stratify
        stratify_col = None
    else:
        stratify_col = df[target_col]
        
    X = df[features]
    y = df[target_col]
    
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify_col)
    
    models = {
        'Dummy': DummyClassifier(strategy='prior'),
        'LogisticRegression': LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
        'DecisionTree': DecisionTreeClassifier(class_weight='balanced', random_state=42),
        'RandomForest': RandomForestClassifier(class_weight='balanced', random_state=42),
        'GradientBoosting': GradientBoostingClassifier(random_state=42) # GB doesn't support class_weight directly
    }
    
    results = []
    best_f1 = -1
    best_model_name = None
    best_pipeline = None
    best_cm = None
    
    for name, model in models.items():
        is_linear = (name == 'LogisticRegression')
        preprocessor = build_preprocessing_pipeline(num_cols, cat_cols, is_linear)
        
        clf = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
        clf.fit(X_train, y_train)
        
        y_pred = clf.predict(X_test)
        
        try:
            y_prob = clf.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, y_prob)
        except:
            roc_auc = np.nan
            
        acc = accuracy_score(y_test, y_pred)
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        # We mapped Non-Compliant = 1, so pos_label=1
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        results.append({
            'Model': name,
            'Accuracy': acc,
            'Balanced_Accuracy': bal_acc,
            'Precision_NonCompliant': prec,
            'Recall_NonCompliant': rec,
            'F1_NonCompliant': f1,
            'ROC_AUC': roc_auc
        })
        
        # Select best model based on F1-Score for Non-Compliant
        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name
            best_pipeline = clf
            best_cm = confusion_matrix(y_test, y_pred)
            
    results_df = pd.DataFrame(results)
    prefix = experiment_name.replace(' ', '_').lower()
    
    # Note on rule-based
    logging.info(f"Note: The ML baseline is evaluated against rule-derived compliance labels. Therefore, high model performance should be interpreted as successful learning of compliance rules, not independent ground truth discovery.")
    
    if save_model and best_pipeline is not None:
        os.makedirs(output_dir, exist_ok=True)
        # Save metrics
        results_df.to_csv(os.path.join(output_dir, f"{prefix}_metrics.csv"), index=False)
        
        # Plot and save CM
        plt.figure(figsize=(6, 4))
        sns.heatmap(best_cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Compliant (0)', 'Non-Compliant (1)'], yticklabels=['Compliant (0)', 'Non-Compliant (1)'])
        plt.title(f"{experiment_name} - {best_model_name} Confusion Matrix")
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{prefix}_confusion_matrix.png"))
        plt.close()
        
        # Save model
        model_dir = output_dir.replace('reports/model_results', 'models')
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(best_pipeline, os.path.join(model_dir, f"{prefix}_best_model.pkl"))
        
        # Save feature importance if tree based
        if best_model_name in ['RandomForest', 'DecisionTree', 'GradientBoosting']:
            try:
                # Extract feature names after onehot encoding
                cat_encoder = best_pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
                cat_feat_names = cat_encoder.get_feature_names_out(cat_cols)
                all_feat_names = num_cols + list(cat_feat_names)
                
                importances = best_pipeline.named_steps['classifier'].feature_importances_
                feat_imp = pd.DataFrame({'Feature': all_feat_names, 'Importance': importances}).sort_values('Importance', ascending=False)
                feat_imp.to_csv(os.path.join(output_dir, f"{prefix}_feature_importance.csv"), index=False)
            except Exception as e:
                logging.warning(f"Could not extract feature importance: {e}")
                
    return results_df, best_model_name

def train_all():
    df_raw = pd.read_csv('data/processed/model_ready_phase1.csv')
    
    # --- Experiment A: Strict Clean Model ---
    df_exp_a = df_raw[(df_raw['label_confidence'] == 'High') & (df_raw['strict_compliance_label'].isin(['Compliant', 'Non-Compliant']))].copy()
    if not df_exp_a.empty:
        df_exp_a['target'] = (df_exp_a['strict_compliance_label'] == 'Non-Compliant').astype(int)
        features_a = get_allowed_features(df_exp_a)
        res_a, best_a = run_experiment(df_exp_a, "Strict Clean", 'target', features_a, 'reports/model_results')
    else:
        logging.warning("Experiment A data is empty.")
        res_a, best_a = None, None
        
    # --- Experiment B: Practical Operational Model ---
    # Include Compliant_Based_On_Available_Parameters and Non-Compliant
    df_exp_b = df_raw[(df_raw['label_confidence'].isin(['High', 'Medium'])) & (df_raw['available_compliance_label'].isin(['Compliant_Based_On_Available_Parameters', 'Compliant', 'Non-Compliant']))].copy()
    if not df_exp_b.empty:
        df_exp_b['target'] = (df_exp_b['available_compliance_label'] == 'Non-Compliant').astype(int)
        features_b = get_allowed_features(df_exp_b)
        res_b, best_b = run_experiment(df_exp_b, "Practical Operational", 'target', features_b, 'reports/model_results')
        
        # --- Ablation Experiment (Exp B without DO, BOD, pH) ---
        features_ablation = get_allowed_features(df_exp_b, drop_core=True)
        res_abl, best_abl = run_experiment(df_exp_b, "Ablation", 'target', features_ablation, 'reports/model_results', save_model=False)
    else:
        logging.warning("Experiment B data is empty.")
        res_b, best_b = None, None
        res_abl, best_abl = None, None
        
    return df_exp_a, res_a, best_a, df_exp_b, res_b, best_b, res_abl, best_abl

if __name__ == "__main__":
    train_all()
