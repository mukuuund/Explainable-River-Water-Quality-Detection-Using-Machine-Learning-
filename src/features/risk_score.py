import pandas as pd
import numpy as np

def calculate_risk_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    def get_row_risk(row):
        do = row.get('dissolved_oxygen', np.nan)
        bod = row.get('bod', np.nan)
        ph = row.get('ph', np.nan)
        
        weights = {'DO': 0.4, 'BOD': 0.4, 'pH': 0.2}
        risks = {}
        
        if pd.notna(do):
            if do >= 5:
                risks['DO'] = 0
            else:
                risks['DO'] = min(100, max(0, ((5 - do) / 5) * 100))
                
        if pd.notna(bod):
            if bod <= 3:
                risks['BOD'] = 0
            else:
                # Cap BOD risk at BOD=50
                risks['BOD'] = min(100, max(0, ((bod - 3) / 47) * 100))
                
        if pd.notna(ph):
            if 6.5 <= ph <= 8.5:
                risks['pH'] = 0
            elif ph < 6.5:
                risks['pH'] = min(100, max(0, ((6.5 - ph) / 6.5) * 100))
            else: # ph > 8.5
                risks['pH'] = min(100, max(0, ((ph - 8.5) / 5.5) * 100))
                
        core_count = len(risks)
        
        if core_count == 3:
            risk_confidence = 'High'
        elif core_count == 2:
            risk_confidence = 'Medium'
        elif core_count == 1:
            risk_confidence = 'Low'
        else:
            risk_confidence = 'Unknown'
            
        if not risks:
            return pd.Series({'risk_score': np.nan, 'risk_category': 'Unknown', 'risk_confidence': risk_confidence})
            
        total_weight = sum(weights[k] for k in risks.keys())
        normalized_weights = {k: weights[k]/total_weight for k in risks.keys()}
        
        final_score = sum(risks[k] * normalized_weights[k] for k in risks.keys())
        
        if final_score <= 25:
            cat = 'Low Risk'
        elif final_score <= 50:
            cat = 'Moderate Risk'
        elif final_score <= 75:
            cat = 'High Risk'
        else:
            cat = 'Severe Risk'
            
        return pd.Series({'risk_score': round(final_score, 2), 'risk_category': cat, 'risk_confidence': risk_confidence})
        
    risk_features = df.apply(get_row_risk, axis=1)
    for col in risk_features.columns:
        df[col] = risk_features[col]
    
    return df
