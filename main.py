import os
import logging
import pandas as pd

from src.utils import setup_project_structure, copy_raw_data
from src.data.loader import load_data
from src.data.standardizer import standardize_columns
from src.features.compliance import add_compliance_features
from src.features.risk_score import calculate_risk_features
from src.data.profiler import generate_profiles

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    logging.info("1. Setting up project structure...")
    setup_project_structure(base_dir)
    
    logging.info("2. Copying raw data...")
    copy_raw_data(base_dir, source_dir="DATAAAAAAA", target_dir="data/raw")
    
    logging.info("3. Loading data...")
    data_dict = load_data(raw_data_dir=os.path.join(base_dir, "data/raw"), fallback_dir=os.path.join(base_dir, "DATAAAAAAA"))
    
    if not data_dict:
        logging.error("No data loaded. Exiting.")
        return
        
    processed_dfs = []
    
    logging.info("4. Processing individual datasets (Phase 2.5)...")
    for filename, df in data_dict.items():
        logging.info(f"Processing {filename}...")
        
        # Step A: Add source_file first so standardizer doesn't overwrite it
        df['source_file'] = filename
        
        # Step B: Standardize and Coalesce
        df_std = standardize_columns(df)
        
        # Step C: Compliance logic
        df_comp = add_compliance_features(df_std)
        
        # Step D: Risk Scoring
        df_risk = calculate_risk_features(df_comp)
        
        # Save individually
        processed_filename = filename.replace('.csv', '_processed.csv')
        processed_path = os.path.join(base_dir, "data/processed", processed_filename)
        df_risk.to_csv(processed_path, index=False)
        logging.info(f"Saved {processed_filename}")
        
        processed_dfs.append(df_risk)
        
    logging.info("5. Creating combined dataset...")
    # Outer concat
    combined_df = pd.concat(processed_dfs, ignore_index=True, sort=False)
    combined_path = os.path.join(base_dir, "data/processed/all_processed_combined.csv")
    combined_df.to_csv(combined_path, index=False)
    logging.info(f"Saved combined dataset to {combined_path}")
    
    logging.info("6. Generating 3 targeted datasets...")
    # 6a. model_ready_phase1.csv
    model_ready_df = combined_df[combined_df['source_file'] == 'india_water_quality_preprocessed_phase1.csv']
    model_ready_df.to_csv(os.path.join(base_dir, "data/processed/model_ready_phase1.csv"), index=False)
    
    # 6b. nwmp_operational_2025.csv
    nwmp_df = combined_df[combined_df['source_file'].str.contains('NWMP_.*2025', case=False, na=False)]
    nwmp_df.to_csv(os.path.join(base_dir, "data/processed/nwmp_operational_2025.csv"), index=False)
    
    # 6c. historical_baseline_data.csv
    historical_df = combined_df[combined_df['source_file'].str.contains('swq_manual_', case=False, na=False)]
    historical_df.to_csv(os.path.join(base_dir, "data/processed/historical_baseline_data.csv"), index=False)
    
    logging.info("7. Generating profiles and reports...")
    generate_profiles(combined_df, reports_dir=os.path.join(base_dir, "reports/tables"))
    
    logging.info("\n=== FINAL VERIFICATION SUMMARY ===")
    total_rows = len(combined_df)
    print(f"Total rows processed: {total_rows}")
    print(f"Number of source files: {combined_df['source_file'].nunique()}")
    
    all_three_pct = (combined_df['core_parameter_count'] == 3).sum() / total_rows * 100
    at_least_two_pct = (combined_df['core_parameter_count'] >= 2).sum() / total_rows * 100
    print(f"Rows with all DO+BOD+pH available: {all_three_pct:.2f}%")
    print(f"Rows with at least two core parameters: {at_least_two_pct:.2f}%")
    
    print("\n--- Compliance Distribution by Source ---")
    comp_dist = combined_df.groupby(['source_file', 'available_compliance_label']).size().unstack(fill_value=0)
    print(comp_dist)
    
    print("\n--- Label Confidence Distribution ---")
    conf_dist = combined_df['label_confidence'].value_counts()
    print(conf_dist)
    
    print("\n--- Risk Distribution by Source ---")
    risk_dist = combined_df.groupby(['source_file', 'risk_category']).size().unstack(fill_value=0)
    print(risk_dist)
    
    logging.info("Phase 2.5 pipeline completed successfully!")

if __name__ == "__main__":
    main()
