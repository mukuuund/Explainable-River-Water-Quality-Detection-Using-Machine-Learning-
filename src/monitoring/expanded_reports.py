import pandas as pd
import numpy as np
import os
import logging
from typing import List

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def get_actual_parameters(df_columns: List[str]) -> List[str]:
    exclude = [
        'source_file', 'parameter_group', 'strict_compliance_label', 
        'available_compliance_label', 'label_confidence', 'risk_confidence', 
        'risk_score', 'risk_category', 'do_compliant', 'bod_compliant', 
        'ph_compliant', 'data_origin', 'integration_phase', 'station_name', 
        'river_name', 'state', 'district', 'date', 'month', 'year', 'season',
        'agency', 'sampling_date', 'latitude', 'longitude', 'sample_id',
        'human_activities'
    ]
    return [c for c in df_columns if c not in exclude]

def generate_coverage_reports(df: pd.DataFrame, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    
    group_cols = ['source_file', 'state', 'parameter_group']
    for c in group_cols:
        if c not in df.columns:
            df[c] = 'Unknown'
            
    summary = df.groupby(group_cols).agg(
        row_count=('source_file', 'size')
    ).reset_index()
    
    params_to_check = ['dissolved_oxygen', 'bod', 'ph', 'temperature', 'conductivity', 'nitrate', 
                       'fecal_coliform', 'total_coliform', 'turbidity', 'cod', 'total_dissolved_solids']
                       
    for p in params_to_check:
        if p in df.columns:
            pct_col = f'{p}_available_pct'
            summary[pct_col] = df.groupby(group_cols)[p].apply(lambda x: x.notnull().mean() * 100).values
            
    if all(c in df.columns for c in ['dissolved_oxygen', 'bod', 'ph']):
        all_three = df[['dissolved_oxygen', 'bod', 'ph']].notnull().sum(axis=1) == 3
        summary['all_three_core_available_pct'] = df.assign(v=all_three).groupby(group_cols)['v'].mean().values * 100
        
        at_least_two = df[['dissolved_oxygen', 'bod', 'ph']].notnull().sum(axis=1) >= 2
        summary['at_least_two_core_available_pct'] = df.assign(v=at_least_two).groupby(group_cols)['v'].mean().values * 100
        
    aux_cols = [c for c in params_to_check if c not in ['dissolved_oxygen', 'bod', 'ph'] and c in df.columns]
    if aux_cols:
        at_least_three_aux = df[aux_cols].notnull().sum(axis=1) >= 3
        summary['at_least_three_auxiliary_available_pct'] = df.assign(v=at_least_three_aux).groupby(group_cols)['v'].mean().values * 100
        
    summary.to_csv(os.path.join(output_dir, 'expanded_parameter_coverage_by_source.csv'), index=False)
    
    state_cov = df.groupby('state').size().reset_index(name='count')
    state_cov.to_csv(os.path.join(output_dir, 'coverage_by_state.csv'), index=False)
    
    if 'parameter_group' in df.columns:
        grp_cov = df.groupby('parameter_group').size().reset_index(name='count')
        grp_cov.to_csv(os.path.join(output_dir, 'coverage_by_parameter_group.csv'), index=False)
    
    # Best Supported
    actual_params = get_actual_parameters(df.columns.tolist())
    best_df = df[actual_params].notnull().sum().reset_index()
    best_df.columns = ['parameter', 'non_null_count']
    best_df = best_df.sort_values(by='non_null_count', ascending=False)
    best_df.to_csv(os.path.join(output_dir, 'phase4_5a_best_supported_parameters_fixed.csv'), index=False)

def generate_baselines(df: pd.DataFrame, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude = ['year', 'month', 'latitude', 'longitude']
    params = [c for c in numeric_cols if c not in exclude]
    
    if 'state' in df.columns and 'river_name' in df.columns:
        top_rivers = df['river_name'].value_counts().head(50).index
        sub_df = df[df['river_name'].isin(top_rivers)]
        
        if not sub_df.empty:
            melted = sub_df.melt(id_vars=['state', 'river_name'], value_vars=params, var_name='parameter', value_name='value').dropna()
            
            if not melted.empty:
                seasonal = melted.groupby(['state', 'river_name', 'parameter'])['value'].agg(
                    ['count', 'mean', 'median', 'std', 'min', 'max']
                ).reset_index()
                seasonal.to_csv(os.path.join(output_dir, 'seasonal_baseline_by_state_river.csv'), index=False)

def generate_auxiliary_readiness(df: pd.DataFrame, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    
    has_target = (df['label_confidence'] == 'High') | (df['label_confidence'] == 'Medium') if 'label_confidence' in df.columns else pd.Series(False, index=df.index)
    
    aux_params = ['temperature', 'conductivity', 'nitrate', 'fecal_coliform', 'total_coliform', 'turbidity', 'cod', 'total_dissolved_solids']
    avail_aux = [c for c in aux_params if c in df.columns]
    
    has_3_aux = df[avail_aux].notnull().sum(axis=1) >= 3 if avail_aux else pd.Series(False, index=df.index)
    
    no_core_but_aux = (~has_target) & has_3_aux
    
    readiness = pd.DataFrame([{
        'total_rows': len(df),
        'rows_with_valid_target': has_target.sum(),
        'rows_with_med_high_confidence': has_target.sum(),
        'rows_with_3_aux': has_3_aux.sum(),
        'rows_no_core_but_aux': no_core_but_aux.sum(),
        'states_represented': df['state'].nunique() if 'state' in df.columns else 0,
        'rivers_represented': df['river_name'].nunique() if 'river_name' in df.columns else 0,
        'station_count': df['station_name'].nunique() if 'station_name' in df.columns else 0,
        'recommended_use': 'baseline only' if has_target.sum() < 1000 else 'auxiliary-only model extension'
    }])
    readiness.to_csv(os.path.join(output_dir, 'expanded_auxiliary_model_readiness.csv'), index=False)
    
    md_content = f"""# Expanded Auxiliary Model Readiness Summary

## Can this new data improve historical baseline analysis?
Yes. The dataset integrates {len(df)} records across various states, significantly expanding the spatial and temporal coverage of the water quality project.

## Can this new data support expanded auxiliary-only model training?
{readiness['recommended_use'].iloc[0].capitalize()}. With {has_target.sum()} records having medium/high confidence targets and {has_3_aux.sum()} records having at least 3 auxiliary parameters, this provides a foundation for future model retraining. However, many records lack the core DO/BOD/pH parameters.

## Should we retrain now or only prepare for future retraining?
**Do not retrain now**. The expanded data represents multi-state physical and chemical parameters that serve as a strong baseline but are not required for the current Phase 5 explainability goals.
"""
    with open(os.path.join(output_dir, 'expanded_auxiliary_model_readiness_summary.md'), 'w') as f:
        f.write(md_content)
