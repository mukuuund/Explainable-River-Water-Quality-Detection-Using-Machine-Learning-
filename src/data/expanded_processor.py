import pandas as pd
import numpy as np
import logging
import re
from typing import Dict, Any, List, Tuple

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def clean_numeric_value(val: Any) -> Tuple[float, str]:
    if pd.isna(val):
        return np.nan, ''
        
    s = str(val).strip().lower()
    
    if s in ['bdl', 'below detection limit', 'na', '-', 'nil', 'nd', 'not detectable', '', 'null']:
        return np.nan, 'Below detection/Missing'
        
    s = s.replace(',', '')
    
    # Check for <x
    if s.startswith('<'):
        match = re.search(r'[-+]?\d*\.?\d+', s)
        if match:
            try:
                v = float(match.group())
                return v / 2.0, f'Converted <{v} to {v/2.0}'
            except:
                return np.nan, 'Parse Error'
                
    match = re.search(r'[-+]?\d*\.?\d+', s)
    if match:
        try:
            return float(match.group()), ''
        except:
            return np.nan, 'Parse Error'
            
    return np.nan, 'Parse Error'

def apply_data_quality_safeguards(df: pd.DataFrame, source_file: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    flags = []
    
    numeric_cols = [
        'dissolved_oxygen', 'bod', 'ph', 'temperature', 'conductivity', 'turbidity',
        'total_dissolved_solids', 'total_solids', 'total_suspended_solids', 'flow',
        'depth', 'fecal_coliform', 'total_coliform', 'fecal_streptococci', 'e_coli',
        'nitrate', 'nitrite', 'ammonia', 'phosphate', 'carbonate', 'bicarbonate',
        'alkalinity', 'chloride', 'sulphate', 'fluoride', 'calcium', 'magnesium',
        'sodium', 'potassium', 'hardness', 'cod', 'sodium_adsorption_ratio',
        'arsenic', 'cadmium', 'chromium', 'copper', 'iron', 'lead', 'manganese',
        'mercury', 'nickel', 'zinc', 'boron'
    ]
    
    for col in numeric_cols:
        if col in df.columns:
            cleaned_vals = []
            for idx, val in df[col].items():
                c_val, flag_msg = clean_numeric_value(val)
                cleaned_vals.append(c_val)
                if flag_msg and flag_msg != 'Below detection/Missing' and flag_msg != 'Parse Error':
                    flags.append({'source_file': source_file, 'row_index': idx, 'column': col, 'value': val, 'flag_type': flag_msg, 'action': 'Kept and coerced'})
            df[col] = cleaned_vals
            
            # Constraints checking
            if col == 'ph':
                invalid_ph = df[(df['ph'] < 0) | (df['ph'] > 14)]
                for idx, row in invalid_ph.iterrows():
                    flags.append({'source_file': source_file, 'row_index': idx, 'column': 'ph', 'value': row['ph'], 'flag_type': 'Extreme pH', 'action': 'Kept but flagged'})
            
            if col in ['bod', 'dissolved_oxygen', 'temperature']:
                invalid_vals = df[df[col] < 0]
                for idx, row in invalid_vals.iterrows():
                    flags.append({'source_file': source_file, 'row_index': idx, 'column': col, 'value': row[col], 'flag_type': f'Negative {col}', 'action': 'Kept but flagged'})
                    
            if col == 'conductivity':
                invalid_vals = df[df[col] > 500000]
                for idx, row in invalid_vals.iterrows():
                    flags.append({'source_file': source_file, 'row_index': idx, 'column': col, 'value': row[col], 'flag_type': 'Extreme conductivity', 'action': 'Kept but flagged'})
                    
    if 'station_name' in df.columns and 'river_name' in df.columns:
        missing_both = df[df['station_name'].isna() & df['river_name'].isna()]
        for idx, row in missing_both.iterrows():
            flags.append({'source_file': source_file, 'row_index': idx, 'column': 'location', 'value': 'NA', 'flag_type': 'Missing station and river', 'action': 'Kept but flagged'})

    flags_df = pd.DataFrame(flags) if flags else pd.DataFrame(columns=['source_file', 'row_index', 'column', 'value', 'flag_type', 'action'])
    return df, flags_df

def apply_compliance_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    if 'dissolved_oxygen' not in df.columns: df['dissolved_oxygen'] = np.nan
    if 'bod' not in df.columns: df['bod'] = np.nan
    if 'ph' not in df.columns: df['ph'] = np.nan
    
    do_ok = df['dissolved_oxygen'] >= 5
    bod_ok = df['bod'] <= 3
    ph_ok = (df['ph'] >= 6.5) & (df['ph'] <= 8.5)
    
    df['do_compliant'] = do_ok
    df['bod_compliant'] = bod_ok
    df['ph_compliant'] = ph_ok
    
    def evaluate_compliance(row):
        is_do_ok = row['do_compliant'] if pd.notnull(row['dissolved_oxygen']) else True
        is_bod_ok = row['bod_compliant'] if pd.notnull(row['bod']) else True
        is_ph_ok = row['ph_compliant'] if pd.notnull(row['ph']) else True
        
        missing_count = sum(pd.isnull(row[['dissolved_oxygen', 'bod', 'ph']]))
        
        if missing_count == 3:
            return 'No Target Data', 'Insufficient Data'
            
        is_compliant = is_do_ok and is_bod_ok and is_ph_ok
        label = 'Compliant' if is_compliant else 'Non-Compliant'
        
        confidence = 'High'
        if missing_count == 1: confidence = 'Medium'
        elif missing_count == 2: confidence = 'Low'
        elif missing_count == 3: confidence = 'Insufficient Data'
        
        return label, confidence

    res = df.apply(evaluate_compliance, axis=1)
    df['available_compliance_label'] = [x[0] for x in res]
    df['label_confidence'] = [x[1] for x in res]
    
    df['strict_compliance_label'] = np.where(
        df['label_confidence'] == 'High',
        df['available_compliance_label'],
        'Unknown'
    )
    
    return df
