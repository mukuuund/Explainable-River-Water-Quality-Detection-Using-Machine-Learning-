import pandas as pd
import os
import logging

def validate_operational_model():
    logging.info("Validating operational model predictions against rules...")
    df = pd.read_csv('data/processed/nwmp_2025_predictions.csv')
    
    total_rows = len(df)
    
    # Filter out invalid rule labels
    invalid_labels = ['Insufficient_Data', 'Unknown', 'Missing', 'NaN', 'nan']
    valid_mask = ~df['available_compliance_label'].isin(invalid_labels) & df['available_compliance_label'].notna()
    
    df_valid = df[valid_mask].copy()
    valid_rows = len(df_valid)
    
    # Map valid rule labels to binary for easier comparison
    df_valid['rule_binary'] = df_valid['available_compliance_label'].map({
        'Non-Compliant': 1,
        'Compliant': 0,
        'Compliant_Based_On_Available_Parameters': 0
    })
    
    model_compliant = (df_valid['predicted_non_compliant'] == 0).sum()
    model_non_compliant = (df_valid['predicted_non_compliant'] == 1).sum()
    
    rule_compliant = (df_valid['rule_binary'] == 0).sum()
    rule_non_compliant = (df_valid['rule_binary'] == 1).sum()
    
    agreement = (df_valid['predicted_non_compliant'] == df_valid['rule_binary']).sum()
    agreement_rate = agreement / valid_rows if valid_rows > 0 else 0
    disagreement_count = valid_rows - agreement
    
    summary = {
        'total_rows': total_rows,
        'rows_with_valid_rule_label': valid_rows,
        'model_predicted_compliant': model_compliant,
        'model_predicted_non_compliant': model_non_compliant,
        'rule_compliant': rule_compliant,
        'rule_non_compliant': rule_non_compliant,
        'agreement_rate': agreement_rate,
        'disagreement_count': disagreement_count
    }
    
    os.makedirs('reports/monitoring', exist_ok=True)
    pd.DataFrame([summary]).to_csv('reports/monitoring/model_rule_agreement_summary.csv', index=False)
    
    disagreements = df_valid[df_valid['predicted_non_compliant'] != df_valid['rule_binary']]
    if not disagreements.empty:
        disagreements.to_csv('reports/monitoring/model_rule_disagreements.csv', index=False)
    else:
        # Create empty file
        pd.DataFrame(columns=df_valid.columns).to_csv('reports/monitoring/model_rule_disagreements.csv', index=False)
        
    logging.info(f"Validation complete. Agreement rate: {agreement_rate:.2%}")

if __name__ == "__main__":
    validate_operational_model()
