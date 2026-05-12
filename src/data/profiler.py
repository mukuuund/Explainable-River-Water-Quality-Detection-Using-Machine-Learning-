import pandas as pd
import os
import logging

def generate_profiles(combined_df: pd.DataFrame, reports_dir: str = "reports/tables"):
    os.makedirs(reports_dir, exist_ok=True)
    
    # 1. source_file_summary.csv
    # Columns: source_file, row_count, column_count, date_range, station_count, river_count
    source_summary = []
    for source, group in combined_df.groupby('source_file'):
        row_count = len(group)
        # Original columns vs canonical is tricky, but let's count non-all-NaN columns for the source
        col_count = len(group.dropna(axis=1, how='all').columns)
        
        date_range = "Unknown"
        if 'sampling_date' in group.columns and not group['sampling_date'].isnull().all():
            dates = pd.to_datetime(group['sampling_date'], errors='coerce').dropna()
            if not dates.empty:
                date_range = f"{dates.min().date()} to {dates.max().date()}"
                
        station_count = group['station_name'].nunique() if 'station_name' in group.columns else 0
        river_count = group['river_name'].nunique() if 'river_name' in group.columns else 0
        
        source_summary.append({
            'source_file': source,
            'row_count': row_count,
            'column_count': col_count,
            'date_range': date_range,
            'station_count': station_count,
            'river_count': river_count
        })
    pd.DataFrame(source_summary).to_csv(os.path.join(reports_dir, "source_file_summary.csv"), index=False)
    logging.info("Generated source_file_summary.csv")
    
    # 2. canonical_parameter_coverage_by_source.csv
    coverage_list = []
    for source, group in combined_df.groupby('source_file'):
        total = len(group)
        do_pct = (group['dissolved_oxygen'].notna().sum() / total) * 100 if 'dissolved_oxygen' in group.columns else 0
        bod_pct = (group['bod'].notna().sum() / total) * 100 if 'bod' in group.columns else 0
        ph_pct = (group['ph'].notna().sum() / total) * 100 if 'ph' in group.columns else 0
        
        if 'core_parameter_count' in group.columns:
            all_three = (group['core_parameter_count'] == 3).sum() / total * 100
            at_least_two = (group['core_parameter_count'] >= 2).sum() / total * 100
        else:
            all_three = 0
            at_least_two = 0
            
        coverage_list.append({
            'source_file': source,
            'dissolved_oxygen_available_pct': do_pct,
            'bod_available_pct': bod_pct,
            'ph_available_pct': ph_pct,
            'all_three_core_available_pct': all_three,
            'at_least_two_core_available_pct': at_least_two
        })
    pd.DataFrame(coverage_list).to_csv(os.path.join(reports_dir, "canonical_parameter_coverage_by_source.csv"), index=False)
    logging.info("Generated canonical_parameter_coverage_by_source.csv")
    
    # 3. label_confidence_summary.csv
    conf_list = []
    for source, group in combined_df.groupby('source_file'):
        if 'label_confidence' in group.columns:
            counts = group['label_confidence'].value_counts()
            conf_list.append({
                'source_file': source,
                'high_confidence_count': counts.get('High', 0),
                'medium_confidence_count': counts.get('Medium', 0),
                'low_confidence_count': counts.get('Low', 0),
                'insufficient_count': counts.get('Insufficient', 0)
            })
    if conf_list:
        pd.DataFrame(conf_list).to_csv(os.path.join(reports_dir, "label_confidence_summary.csv"), index=False)
        logging.info("Generated label_confidence_summary.csv")
        
    # 4. compliance_distribution_by_source.csv
    if 'available_compliance_label' in combined_df.columns:
        comp_dist = combined_df.groupby(['source_file', 'available_compliance_label']).size().reset_index(name='count')
        comp_dist.to_csv(os.path.join(reports_dir, "compliance_distribution_by_source.csv"), index=False)
        logging.info("Generated compliance_distribution_by_source.csv")
        
    # 5. risk_distribution_by_source.csv
    if 'risk_category' in combined_df.columns:
        risk_dist = combined_df.groupby(['source_file', 'risk_category']).size().reset_index(name='count')
        risk_dist.to_csv(os.path.join(reports_dir, "risk_distribution_by_source.csv"), index=False)
        logging.info("Generated risk_distribution_by_source.csv")
