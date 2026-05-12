import pandas as pd
import numpy as np

def clean_numeric(val):
    if pd.isna(val):
        return np.nan
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        import re
        match = re.search(r'-?\d+\.?\d*', val)
        if match:
            return float(match.group())
    return np.nan

def add_compliance_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    req_cols = ['dissolved_oxygen', 'bod', 'ph']
    for col in req_cols:
        if col not in df.columns:
            df[col] = np.nan
        else:
            df[col] = df[col].apply(clean_numeric)
            
    def evaluate_row(row):
        do = row['dissolved_oxygen']
        bod = row['bod']
        ph = row['ph']
        
        available = []
        missing = []
        violations = []
        
        if pd.notna(do):
            available.append('DO')
            if do < 5:
                violations.append(f'DO ({do} < 5)')
        else:
            missing.append('DO')
            
        if pd.notna(bod):
            available.append('BOD')
            if bod > 3:
                violations.append(f'BOD ({bod} > 3)')
        else:
            missing.append('BOD')
            
        if pd.notna(ph):
            available.append('pH')
            if ph < 6.5 or ph > 8.5:
                violations.append(f'pH ({ph} not in 6.5-8.5)')
        else:
            missing.append('pH')
            
        core_parameter_count = len(available)
        
        if core_parameter_count == 3:
            label_confidence = 'High'
        elif core_parameter_count == 2:
            label_confidence = 'Medium'
        elif core_parameter_count == 1:
            label_confidence = 'Low'
        else:
            label_confidence = 'Insufficient'
            
        violation_count = len(violations)
        
        # A. available_compliance_label
        if not available:
            available_label = 'Insufficient_Data'
        elif violation_count > 0:
            available_label = 'Non-Compliant'
        else:
            available_label = 'Compliant_Based_On_Available_Parameters'
            
        # B. strict_compliance_label
        if core_parameter_count == 3:
            if violation_count > 0:
                strict_label = 'Non-Compliant'
            else:
                strict_label = 'Compliant'
        else:
            strict_label = 'Insufficient_Data'
            
        return pd.Series({
            'available_compliance_label': available_label,
            'strict_compliance_label': strict_label,
            'core_parameter_count': core_parameter_count,
            'available_compliance_parameters': ', '.join(available) if available else 'None',
            'missing_required_parameters': ', '.join(missing) if missing else 'None',
            'label_confidence': label_confidence,
            'violation_reasons': '; '.join(violations) if violations else 'None',
            'violation_count': violation_count
        })
        
    features = df.apply(evaluate_row, axis=1)
    for col in features.columns:
        df[col] = features[col]
    
    return df
