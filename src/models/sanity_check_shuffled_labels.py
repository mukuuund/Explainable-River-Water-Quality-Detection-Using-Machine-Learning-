import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, f1_score
from src.models.train_baseline_models import get_allowed_features, build_preprocessing_pipeline

def sanity_check_shuffled_labels():
    df_raw = pd.read_csv('data/processed/model_ready_phase1.csv')
    df_exp_b = df_raw[(df_raw['label_confidence'].isin(['High', 'Medium'])) & (df_raw['available_compliance_label'].isin(['Compliant_Based_On_Available_Parameters', 'Compliant', 'Non-Compliant']))].copy()
    
    df_exp_b['target'] = (df_exp_b['available_compliance_label'] == 'Non-Compliant').astype(int)
    features = get_allowed_features(df_exp_b)
    
    # Shuffle targets randomly
    np.random.seed(42)
    df_exp_b['shuffled_target'] = np.random.permutation(df_exp_b['target'].values)
    
    X = df_exp_b[features]
    y = df_exp_b['shuffled_target']
    
    stratify_col = y if y.value_counts().min() >= 5 else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify_col)
    
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    preprocessor = build_preprocessing_pipeline(num_cols, cat_cols, is_linear=False)
    clf = Pipeline(steps=[('preprocessor', preprocessor), 
                          ('classifier', DecisionTreeClassifier(class_weight='balanced', random_state=42))])
                          
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    report_df = pd.DataFrame([{
        'Experiment': 'Shuffled Labels (DecisionTree)',
        'Balanced_Accuracy': bal_acc,
        'ROC_AUC': roc_auc,
        'F1_NonCompliant': f1
    }])
    
    os.makedirs('reports/model_results', exist_ok=True)
    report_df.to_csv('reports/model_results/shuffled_label_sanity_metrics.csv', index=False)
    
    # Audit passes if the model fails to learn the shuffled noise (returns near 0.5)
    return bal_acc < 0.6 and roc_auc < 0.6

if __name__ == "__main__":
    passed = sanity_check_shuffled_labels()
    print("Shuffled Label Sanity Passed" if passed else "Shuffled Label Sanity Failed (Model learned shuffled labels!)")
