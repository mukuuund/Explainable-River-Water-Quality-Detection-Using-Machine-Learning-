import pandas as pd
import os
import logging
from src.models.train_models_corrected import get_allowed_features

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def prepare_validation():
    nwmp_path = 'data/processed/nwmp_operational_2025.csv'
    output_path = 'reports/model_results/nwmp_validation_readiness.csv'
    
    if not os.path.exists(nwmp_path):
        logging.warning(f"File {nwmp_path} not found.")
        return
        
    df_nwmp = pd.read_csv(nwmp_path)
    
    # Identify features that would be used by the leakage-safe operational model
    features = get_allowed_features(df_nwmp, drop_core=True)
    
    report = []
    total_rows = len(df_nwmp)
    for feat in features:
        if feat in df_nwmp.columns:
            missing = df_nwmp[feat].isnull().sum()
            present = total_rows - missing
            coverage = (present / total_rows) * 100 if total_rows > 0 else 0
            report.append({
                'Feature': feat,
                'Available_Count': present,
                'Missing_Count': missing,
                'Coverage_Pct': coverage
            })
        else:
            report.append({
                'Feature': feat,
                'Available_Count': 0,
                'Missing_Count': total_rows,
                'Coverage_Pct': 0
            })
            
    report_df = pd.DataFrame(report)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    report_df.to_csv(output_path, index=False)
    logging.info(f"Saved NWMP validation readiness report to {output_path}")

if __name__ == "__main__":
    prepare_validation()
