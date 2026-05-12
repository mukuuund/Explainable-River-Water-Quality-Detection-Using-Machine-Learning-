import pandas as pd
import numpy as np
import os
import sys
import logging
from collections import Counter

# Resolve project root relative to this file
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.features.risk_score import calculate_risk_features


def categorize_station(months_obs, months_pred_nc):
    if months_obs < 2:
        if months_pred_nc >= 1:
            return 'Intermittent Risk'
        return 'Insufficient Monitoring'
        
    if months_pred_nc == months_obs:
        return 'Persistent Hotspot'
    elif months_pred_nc >= 2:
        return 'Recurring Risk'
    elif months_pred_nc == 1:
        return 'Intermittent Risk'
    else:
        return 'Stable / Low Risk'

def get_most_common(series):
    # filter out nans and "Unknown"
    valid = [x for x in series if pd.notna(x) and x != 'Unknown']
    if not valid:
        return 'None'
    return Counter(valid).most_common(1)[0][0]


def derive_risk_category(score):
    """Derive risk category from a numeric risk score."""
    if pd.isna(score):
        return 'Unknown'
    if score <= 25:
        return 'Low Risk'
    elif score <= 50:
        return 'Moderate Risk'
    elif score <= 75:
        return 'High Risk'
    else:
        return 'Severe Risk'


# Severity ranking: higher number = more severe
RISK_SEVERITY_ORDER = {
    'Severe Risk': 4,
    'High Risk': 3,
    'Moderate Risk': 2,
    'Low Risk': 1,
    'Unknown': -1
}
RISK_SEVERITY_REVERSE = {v: k for k, v in RISK_SEVERITY_ORDER.items()}


def get_most_severe_category(series):
    """Return the most severe risk category from a series, by severity ranking (NOT mode)."""
    max_severity = -1
    for val in series:
        if pd.notna(val):
            max_severity = max(max_severity, RISK_SEVERITY_ORDER.get(str(val).strip(), -1))
    return RISK_SEVERITY_REVERSE.get(max_severity, 'Unknown')


def detect_hotspots():
    logging.info("Detecting persistent hotspots...")
    predictions_path = os.path.join(PROJECT_ROOT, 'data', 'processed', 'nwmp_2025_predictions.csv')
    df = pd.read_csv(predictions_path)
    
    if 'station_name' not in df.columns:
        logging.warning("No station_name column found. Cannot detect hotspots.")
        return
    
    # ============================================================
    # PRE-AGGREGATION FIX: Ensure every row has risk_score and risk_category
    # ============================================================
    unknown_before = (df['risk_category'] == 'Unknown').sum() if 'risk_category' in df.columns else len(df)
    missing_risk_before = df['risk_score'].isna().sum() if 'risk_score' in df.columns else len(df)
    
    # If risk_score is missing but DO/BOD/pH exist, calculate using existing logic
    needs_risk = df['risk_score'].isna() if 'risk_score' in df.columns else pd.Series([True] * len(df), index=df.index)
    has_any_core = False
    for col in ['dissolved_oxygen', 'bod', 'ph']:
        if col in df.columns:
            has_any_core = True
            break
    
    if needs_risk.any() and has_any_core:
        rows_needing_risk = df.loc[needs_risk].copy()
        if not rows_needing_risk.empty:
            recalculated = calculate_risk_features(rows_needing_risk)
            # Update only the rows that were missing
            for col in ['risk_score', 'risk_category', 'risk_confidence']:
                if col in recalculated.columns:
                    df.loc[needs_risk, col] = recalculated[col].values
    
    # If risk_category is still missing but risk_score exists, derive it
    if 'risk_score' in df.columns and 'risk_category' in df.columns:
        mask = (df['risk_category'].isna() | (df['risk_category'] == 'Unknown')) & df['risk_score'].notna()
        df.loc[mask, 'risk_category'] = df.loc[mask, 'risk_score'].apply(derive_risk_category)
    
    unknown_after = (df['risk_category'] == 'Unknown').sum() if 'risk_category' in df.columns else len(df)
    missing_risk_after = df['risk_score'].isna().sum() if 'risk_score' in df.columns else len(df)
    
    print(f"[Hotspot Fix] risk_category Unknown: {unknown_before} -> {unknown_after}")
    print(f"[Hotspot Fix] risk_score missing: {missing_risk_before} -> {missing_risk_after}")
        
    # Map valid rule labels
    invalid_labels = ['Insufficient_Data', 'Unknown', 'Missing', 'NaN', 'nan']
    df['rule_binary'] = df['available_compliance_label'].apply(
        lambda x: 1 if pd.notna(x) and x == 'Non-Compliant' else (0 if pd.notna(x) and x not in invalid_labels else np.nan)
    )
    
    # Clean up month
    if 'month' not in df.columns:
        df['month'] = 'Unknown'
        
    hotspots = []
    
    for station, group in df.groupby('station_name'):
        months_obs = group['month'].nunique()
        months_pred_nc = group['predicted_non_compliant'].sum() if 'predicted_non_compliant' in group.columns else 0
        months_rule_nc = group['rule_binary'].sum() if not group['rule_binary'].isna().all() else 0
        
        avg_risk = group['risk_score'].mean() if 'risk_score' in group.columns else np.nan
        max_risk = group['risk_score'].max() if 'risk_score' in group.columns else np.nan
        
        possible_prob_cols = ['predicted_non_compliance_probability', 'ml_non_compliance_probability', 'non_compliance_probability', 'predicted_probability', 'probability_non_compliant']
        prob_col = next((c for c in possible_prob_cols if c in group.columns), None)
        avg_prob = group[prob_col].mean() if prob_col else np.nan
        
        most_common_violation = get_most_common(group['violation_reasons']) if 'violation_reasons' in group.columns else 'Unknown'
        most_severe_risk_category = get_most_severe_category(group['risk_category']) if 'risk_category' in group.columns else 'Unknown'
        
        first_month = group['month'].min()
        last_month = group['month'].max()
        
        status = categorize_station(months_obs, months_pred_nc)
        
        hotspots.append({
            'station_name': station,
            'months_observed': months_obs,
            'months_predicted_non_compliant': months_pred_nc,
            'months_rule_non_compliant': months_rule_nc,
            'average_risk_score': avg_risk,
            'max_risk_score': max_risk,
            'average_predicted_non_compliance_probability': avg_prob,
            'most_common_violation_reason': most_common_violation,
            'most_severe_risk_category': most_severe_risk_category,
            'first_observed_month': first_month,
            'last_observed_month': last_month,
            'hotspot_status': status
        })
        
    hotspots_df = pd.DataFrame(hotspots)
    
    reports_dir = os.path.join(PROJECT_ROOT, 'reports', 'monitoring')
    os.makedirs(reports_dir, exist_ok=True)
    hotspots_df.to_csv(os.path.join(reports_dir, 'hotspot_summary.csv'), index=False)
    
    # Top 20 Hotspots (sort by status priority, then max risk, then months predicted nc)
    status_priority = {
        'Persistent Hotspot': 4,
        'Recurring Risk': 3,
        'Intermittent Risk': 2,
        'Insufficient Monitoring': 1,
        'Stable / Low Risk': 0
    }
    hotspots_df['status_priority'] = hotspots_df['hotspot_status'].map(status_priority)
    top_20 = hotspots_df.sort_values(
        by=['status_priority', 'max_risk_score', 'months_predicted_non_compliant'], 
        ascending=[False, False, False]
    ).head(20).drop(columns=['status_priority'])
    
    top_20.to_csv(os.path.join(reports_dir, 'top_20_hotspots.csv'), index=False)
    
    # Category distribution
    cat_dist = hotspots_df['hotspot_status'].value_counts().reset_index()
    cat_dist.columns = ['hotspot_status', 'count']
    cat_dist.to_csv(os.path.join(reports_dir, 'hotspot_category_distribution.csv'), index=False)
    
    # Risk category distribution in hotspot summary (validation)
    risk_dist = hotspots_df['most_severe_risk_category'].value_counts().reset_index()
    risk_dist.columns = ['most_severe_risk_category', 'count']
    print(f"\n[Hotspot Validation] Risk category distribution in hotspot summary:")
    print(risk_dist.to_string(index=False))
    
    logging.info(f"Hotspot detection complete. Found {len(hotspots_df[hotspots_df['hotspot_status'] == 'Persistent Hotspot'])} persistent hotspots.")

if __name__ == "__main__":
    detect_hotspots()
