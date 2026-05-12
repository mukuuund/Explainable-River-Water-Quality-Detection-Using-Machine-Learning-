import pandas as pd
from pathlib import Path
root = Path(r'c:/Users/Mukun/OneDrive/Desktop/Coding/college/EVEN - 2026/Minor 2/Minor Project New/Minor Project (2)/Minor Project (2)/Minor Project')
df = pd.read_csv(root / 'data' / 'processed' / 'model_ready_phase1.csv', low_memory=False)
print('rows', len(df), 'cols', len(df.columns))
print('strict label dist')
print(df['strict_compliance_label'].value_counts(dropna=False).to_dict())
print('available label dist')
print(df['available_compliance_label'].value_counts(dropna=False).to_dict())
print('----')
for c in ['ph_mean','dissolved_oxygen_mean_mgl','bod_mean_mgl','nitrate_n_mean_mgl','conductivity_mean_umho_cm','turbidity_ntu']:
    print(c, 'notna', df[c].notna().sum(), 'nan', df[c].isna().sum(), 'sample', df[c].dropna().head(3).tolist())
print('----')
print('object cols with <30% missing:')
for c in df.select_dtypes(include='object').columns:
    miss = df[c].isna().mean()
    if miss < 0.3:
        print(c, df[c].nunique(dropna=True), miss)
