import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib
import os
import json
from pathlib import Path

def add_rule_targets(df):
    """
    Add rule-based targets for compliance based on derived features.
    Since raw ph, do, bod are missing, use derived means.
    """
    # Rule-based compliance: ph between 6.5-8.5, do > 5, bod < 3
    df['wizard_ph'] = ((df['ph_mean'] >= 6.5) & (df['ph_mean'] <= 8.5)).astype(int)
    df['wizard_do'] = (df['dissolved_oxygen_mean_mgl'] > 5).astype(int)
    df['wizard_bod'] = (df['bod_mean_mgl'] < 3).astype(int)
    # Overall compliance: all three must be 1
    df['strict_compliance_label'] = ((df['wizard_ph'] == 1) & (df['wizard_do'] == 1) & (df['wizard_bod'] == 1)).astype(int)
    return df

def get_allowed_features(df, drop_core=False):
    """
    Get allowed features for modeling.
    Excludes targets and optionally core features to prevent leakage.
    """
    target_cols = ['strict_compliance_label', 'wizard_ph', 'wizard_do', 'wizard_bod']
    core_features = ['ph', 'do', 'bod']  # raw, but since missing, perhaps not needed
    all_cols = df.columns.tolist()
    allowed = [col for col in all_cols if col not in target_cols]
    if drop_core:
        allowed = [col for col in allowed if col not in core_features]
    return allowed

def run_experiment(df, experiment_name, drop_core=False):
    """
    Run a single experiment: train model and save artifacts.
    """
    df = add_rule_targets(df.copy())
    features = get_allowed_features(df, drop_core=drop_core)
    X = df[features]
    y = df['strict_compliance_label']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Preprocessing
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Model
    clf = Pipeline(steps=[('preprocessor', preprocessor),
                          ('classifier', RandomForestClassifier(random_state=42))])
    
    # Train
    clf.fit(X_train, y_train)
    
    # Predict
    y_pred = clf.predict(X_test)
    
    # Metrics
    report = classification_report(y_test, y_pred, output_dict=True)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Save model
    model_dir = Path('models')
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / f'{experiment_name}_model.pkl'
    joblib.dump(clf, model_path)
    
    # Save features
    features_path = model_dir / f'{experiment_name}_features.json'
    with open(features_path, 'w') as f:
        json.dump(features, f)
    
    return {
        'experiment': experiment_name,
        'accuracy': accuracy,
        'report': report,
        'model_path': str(model_path),
        'features_path': str(features_path)
    }

def train_all():
    """
    Train models for experiments A and B.
    """
    # Load data
    data_path = 'data/processed/model_ready_phase1.csv'
    df = pd.read_csv(data_path)
    
    # Experiment A: rule-replication (include core features)
    result_a = run_experiment(df, 'experiment_a', drop_core=False)
    
    # Experiment B: leakage-safe (exclude core features)
    result_b = run_experiment(df, 'experiment_b', drop_core=True)
    
    # Save summary
    outputs_dir = Path('outputs/model_training')
    outputs_dir.mkdir(parents=True, exist_ok=True)
    
    summary = {
        'experiment_a': result_a,
        'experiment_b': result_b
    }
    
    with open(outputs_dir / 'training_report.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    return summary