import os
import pandas as pd
import numpy as np
import logging
from src.data.expanded_data_integration import detect_input_folder, build_manifest
from src.data.expanded_canonical_mapper import standardize_columns_and_map
from src.data.expanded_processor import apply_data_quality_safeguards, apply_compliance_labels
from src.monitoring.expanded_reports import generate_coverage_reports, generate_baselines, generate_auxiliary_readiness, get_actual_parameters
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def main():
    logging.info("Starting Phase 4.5A: Expanded Historical Multi-State Data Integration QA")
    
    reports_dir = 'reports/expanded_data'
    figures_dir = os.path.join(reports_dir, 'figures')
    processed_dir = 'data/processed/expanded'
    
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    
    input_folder = detect_input_folder()
    manifest_path = os.path.join(reports_dir, 'new_data_file_manifest.csv')
    manifest_data, stats = build_manifest(input_folder, manifest_path)
    
    if not manifest_data:
        logging.warning("No new files to process.")
        return
        
    df_manifest = pd.DataFrame(manifest_data)
    
    all_processed_dfs = []
    all_audit_dfs = []
    all_unmapped_dfs = []
    all_flags_dfs = []
    
    for _, row in df_manifest.iterrows():
        if row['load_status'] == 'Failed' or row['duplicate_file_flag']:
            continue
            
        filepath = row['absolute_path']
        filename = row['source_file']
        state_infer = row['inferred_state_from_filename']
        param_grp = row['likely_parameter_group']
        encoding = row['detected_encoding']
        
        try:
            df_raw = pd.read_csv(filepath, encoding=encoding)
            
            df_mapped, df_audit, df_unmapped = standardize_columns_and_map(df_raw, filename, state_infer)
            df_mapped['parameter_group'] = param_grp
            
            df_cleaned, df_flags = apply_data_quality_safeguards(df_mapped, filename)
            
            all_processed_dfs.append(df_cleaned)
            if not df_audit.empty: all_audit_dfs.append(df_audit)
            if not df_unmapped.empty: all_unmapped_dfs.append(df_unmapped)
            if not df_flags.empty: all_flags_dfs.append(df_flags)
            
            logging.info(f"Processed: {filename}")
        except Exception as e:
            logging.error(f"Error processing {filename}: {e}")
            stats['failed_files'] += 1
            stats['new_files'] -= 1
            
    if not all_processed_dfs:
        logging.warning("No valid data processed.")
        return
        
    combined_new_df = pd.concat(all_processed_dfs, ignore_index=True)
    
    if all_audit_dfs:
        pd.concat(all_audit_dfs, ignore_index=True).to_csv(os.path.join(reports_dir, 'phase4_5a_current_mapping_diagnosis.csv'), index=False)
    if all_unmapped_dfs:
        pd.concat(all_unmapped_dfs, ignore_index=True).to_csv(os.path.join(reports_dir, 'unmapped_columns_by_source.csv'), index=False)
    if all_flags_dfs:
        pd.concat(all_flags_dfs, ignore_index=True).to_csv(os.path.join(reports_dir, 'data_quality_flags.csv'), index=False)
        
    combined_new_df = apply_compliance_labels(combined_new_df)
    combined_new_df['integration_phase'] = 'phase_4_5'
    
    new_uploads_path = os.path.join(processed_dir, 'new_uploads_canonical_processed.csv')
    combined_new_df.to_csv(new_uploads_path, index=False)
    
    expanded_baseline_df = combined_new_df.copy()
    if os.path.exists('data/processed/historical_baseline_data.csv'):
        old_baseline = pd.read_csv('data/processed/historical_baseline_data.csv')
        old_baseline['data_origin'] = 'existing_historical'
        old_baseline['integration_phase'] = 'phase_1_4'
        expanded_baseline_df = pd.concat([old_baseline, expanded_baseline_df], ignore_index=True)
        
    expanded_path = os.path.join(processed_dir, 'expanded_historical_multistate_baseline.csv')
    expanded_baseline_df.to_csv(expanded_path, index=False)
    
    numeric_cols = expanded_baseline_df.select_dtypes(include=[np.number]).columns.tolist()
    id_vars = [c for c in ['source_file', 'state', 'district', 'river_name', 'station_name', 'sampling_date', 'year', 'month', 'season', 'parameter_group'] if c in expanded_baseline_df.columns]
    val_vars = [c for c in numeric_cols if c not in id_vars and c not in ['latitude', 'longitude']]
    
    long_format = expanded_baseline_df.melt(id_vars=id_vars, value_vars=val_vars, var_name='parameter', value_name='value').dropna(subset=['value'])
    long_format.to_csv(os.path.join(processed_dir, 'expanded_baseline_long_format.csv'), index=False)
    
    generate_coverage_reports(combined_new_df, reports_dir)
    generate_baselines(combined_new_df, reports_dir)
    generate_auxiliary_readiness(combined_new_df, reports_dir)
    
    # Validation criteria outputs
    total_new_rows = len(combined_new_df)
    total_expanded_rows = len(expanded_baseline_df)
    
    do_bod_ph = combined_new_df[['dissolved_oxygen', 'bod', 'ph']].notnull().sum(axis=1) if all(c in combined_new_df.columns for c in ['dissolved_oxygen', 'bod', 'ph']) else pd.Series(0, index=combined_new_df.index)
    all_3_core = (do_bod_ph == 3).sum()
    at_least_2_core = (do_bod_ph >= 2).sum()
    
    aux_params = ['temperature', 'conductivity', 'nitrate', 'fecal_coliform', 'total_coliform', 'turbidity', 'cod', 'total_dissolved_solids']
    avail_aux = [c for c in aux_params if c in combined_new_df.columns]
    at_least_3_aux = combined_new_df[avail_aux].notnull().sum(axis=1) >= 3 if avail_aux else pd.Series(False, index=combined_new_df.index)
    at_least_3_aux_sum = at_least_3_aux.sum()
    
    actual_params = get_actual_parameters(combined_new_df.columns.tolist())
    actual_counts = combined_new_df[actual_params].notnull().sum().sort_values(ascending=False)
    top_10_params = actual_counts.head(10)
    
    print("\n" + "="*50)
    print("PHASE 4.5A: EXPANDED DATA MAPPING QA AND FIX SUMMARY")
    print("="*50)
    print(f"New Rows Integrated: {total_new_rows}")
    print(f"Total Expanded Baseline Rows: {total_expanded_rows}")
    
    print("\nNon-Null Count for Canonical Parameters (Actual WQ parameters):")
    real_wq_sum = 0
    for param, count in actual_counts.items():
        if count > 0:
            print(f"  - {param}: {count}")
            real_wq_sum += count
            
    print(f"\nRows with DO+BOD+pH: {all_3_core}")
    print(f"Rows with >= 2 Core Parameters: {at_least_2_core}")
    print(f"Rows with >= 3 Auxiliary Parameters: {at_least_3_aux_sum}")
    
    print("\nTop 10 Actual Water Quality Parameters by Coverage:")
    for p, c in top_10_params.items():
        print(f"  - {p}: {c}")
        
    print("\nFiles with Strongest Parameter Coverage:")
    if 'source_file' in combined_new_df.columns:
        file_cov = combined_new_df.groupby('source_file')[actual_params].apply(lambda x: x.notnull().sum().sum()).sort_values(ascending=False)
        for f, c in file_cov.head(3).items():
            print(f"  - {f}: {c} valid data points")
            
    print("\nFiles with Weakest Parameter Coverage:")
    if 'source_file' in combined_new_df.columns:
        for f, c in file_cov.tail(3).items():
            print(f"  - {f}: {c} valid data points")
            
    print("\nValidation Result:")
    if real_wq_sum == 0:
        print("Phase 4.5 failed: raw files appear to be in unexpected format or canonical mapping did not detect parameter columns.")
    else:
        print("Phase 4.5 is now valid. Actual parameters have successfully mapped and populated.")
    print("="*50)

if __name__ == "__main__":
    main()
