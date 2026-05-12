"""
Phase 1: Water Quality Data Preprocessing Pipeline
India Water Quality Datasets (2022 & 2023)
"""

import pandas as pd
import numpy as np
import re
import os
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# ── 1. LOAD & COMBINE ──────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df22 = pd.read_csv(os.path.join(BASE_DIR, 'india_water_quality_2022_complete__1_.csv'))
df23 = pd.read_csv(os.path.join(BASE_DIR, 'india_water_quality_2023_complete.csv'))
df22['Year'] = 2022
df23['Year'] = 2023
df = pd.concat([df22, df23], ignore_index=True)
print(f"Combined shape: {df.shape}")

NUMERIC_COLS = [
    'Temperature_Min_C', 'Temperature_Max_C',
    'Dissolved_Oxygen_Min_mgL', 'Dissolved_Oxygen_Max_mgL',
    'pH_Min', 'pH_Max',
    'Conductivity_Min_umho_cm', 'Conductivity_Max_umho_cm',
    'BOD_Min_mgL', 'BOD_Max_mgL',
    'Nitrate_N_Min_mgL', 'Nitrate_N_Max_mgL',
    'Fecal_Coliform_Min_MPN100ml', 'Fecal_Coliform_Max_MPN100ml',
    'Total_Coliform_Min_MPN100ml', 'Total_Coliform_Max_MPN100ml',
    'Fecal_Streptococci_Min_MPN100ml', 'Fecal_Streptococci_Max_MPN100ml',
]

# ── 2. BDL HANDLING & TYPE COERCION ───────────────────────────────────────────
# BDL (Below Detection Limit) -> 0  |  blanks/NA -> NaN for imputation
for col in NUMERIC_COLS:
    df[col] = (
        df[col].astype(str).str.strip().str.upper()
        .replace({'BDL': '0', 'NAN': np.nan, 'NA': np.nan, '': np.nan, 'NONE': np.nan})
    )
    df[col] = pd.to_numeric(df[col], errors='coerce')

# ── 3. MISSING VALUE IMPUTATION ───────────────────────────────────────────────
# Median per (River_Basin, Year) group; global median fallback
before_null = df[NUMERIC_COLS].isnull().sum().sum()
for col in NUMERIC_COLS:
    group_med  = df.groupby(['River_Basin', 'Year'])[col].transform('median')
    global_med = df[col].median()
    df[col]    = df[col].fillna(group_med).fillna(global_med)
after_null = df[NUMERIC_COLS].isnull().sum().sum()
print(f"Missing values  before: {before_null}  |  after: {after_null}")

# ── 4. DERIVED MEAN PARAMETERS ────────────────────────────────────────────────
df['Temperature_Mean_C']               = (df['Temperature_Min_C']               + df['Temperature_Max_C'])               / 2
df['Dissolved_Oxygen_Mean_mgL']        = (df['Dissolved_Oxygen_Min_mgL']        + df['Dissolved_Oxygen_Max_mgL'])        / 2
df['pH_Mean']                          = (df['pH_Min']                          + df['pH_Max'])                          / 2
df['Conductivity_Mean_umho_cm']        = (df['Conductivity_Min_umho_cm']        + df['Conductivity_Max_umho_cm'])        / 2
df['BOD_Mean_mgL']                     = (df['BOD_Min_mgL']                     + df['BOD_Max_mgL'])                     / 2
df['Nitrate_N_Mean_mgL']               = (df['Nitrate_N_Min_mgL']               + df['Nitrate_N_Max_mgL'])               / 2
df['Fecal_Coliform_Mean_MPN100ml']     = (df['Fecal_Coliform_Min_MPN100ml']     + df['Fecal_Coliform_Max_MPN100ml'])     / 2
df['Total_Coliform_Mean_MPN100ml']     = (df['Total_Coliform_Min_MPN100ml']     + df['Total_Coliform_Max_MPN100ml'])     / 2
df['Fecal_Streptococci_Mean_MPN100ml'] = (df['Fecal_Streptococci_Min_MPN100ml'] + df['Fecal_Streptococci_Max_MPN100ml']) / 2

MEAN_COLS = [
    'Temperature_Mean_C', 'Dissolved_Oxygen_Mean_mgL', 'pH_Mean',
    'Conductivity_Mean_umho_cm', 'BOD_Mean_mgL', 'Nitrate_N_Mean_mgL',
    'Fecal_Coliform_Mean_MPN100ml', 'Total_Coliform_Mean_MPN100ml',
    'Fecal_Streptococci_Mean_MPN100ml',
]

# ── 5. COMPLIANCE LABELS  (CPCB / IS 2296) ────────────────────────────────────
def classify_water(row):
    do  = row['Dissolved_Oxygen_Mean_mgL']
    bod = row['BOD_Mean_mgL']
    tc  = row['Total_Coliform_Mean_MPN100ml']
    ph  = row['pH_Mean']
    ec  = row['Conductivity_Mean_umho_cm']
    if   do >= 6 and bod <= 2 and tc <= 50:               return 'Class_A'
    elif do >= 5 and bod <= 3 and tc <= 500:              return 'Class_B'
    elif do >= 4 and bod <= 3 and tc <= 5000:             return 'Class_C'
    elif do >= 4 and bod <= 3:                            return 'Class_D'
    elif 6 <= ph <= 8.5 and bod <= 3 and ec <= 2250:      return 'Class_E'
    else:                                                 return 'Non_Compliant'

df['Compliance_Label'] = df.apply(classify_water, axis=1)
label_map = {'Class_A': 0, 'Class_B': 1, 'Class_C': 2, 'Class_D': 3, 'Class_E': 4, 'Non_Compliant': 5}
df['Compliance_Label_Encoded'] = df['Compliance_Label'].map(label_map)
df['Is_Safe'] = df['Compliance_Label'].isin(['Class_A', 'Class_B', 'Class_C']).astype(int)

print("\nCompliance label distribution:")
print(df['Compliance_Label'].value_counts())

# ── 6. CLASS IMBALANCE — INVERSE-FREQUENCY WEIGHTS ────────────────────────────
total        = len(df)
label_counts = Counter(df['Compliance_Label'])
n_classes    = len(label_counts)
df['Class_Weight'] = df['Compliance_Label'].map(
    {lbl: total / (n_classes * cnt) for lbl, cnt in label_counts.items()}
)
safe_counts = Counter(df['Is_Safe'])
df['Binary_Class_Weight'] = df['Is_Safe'].map(
    {v: total / (2 * cnt) for v, cnt in safe_counts.items()}
)
print("\nClass weights:")
for lbl, cnt in sorted(label_counts.items()):
    print(f"  {lbl:15s}: n={cnt:4d}  weight={total/(n_classes*cnt):.4f}")

# ── 7. SPATIAL TAG EXTRACTION ─────────────────────────────────────────────────
def extract_spatial_tags(loc):
    u = str(loc).upper()
    tags = []
    if re.search(r'\bU/S\b|\bUPSTREAM\b',        u): tags.append('UPSTREAM')
    if re.search(r'\bD/S\b|\bDOWNSTREAM\b',       u): tags.append('DOWNSTREAM')
    if re.search(r'\bBEFORE\b',                    u): tags.append('BEFORE')
    if re.search(r'\bAFTER\b',                     u): tags.append('AFTER')
    if re.search(r'\bCONF\.?\b|\bCONFLUENCE\b',   u): tags.append('CONFLUENCE')
    if re.search(r'\bDISCHARGE\b',                 u): tags.append('DISCHARGE')
    return '|'.join(tags) if tags else 'NONE'

def get_primary_position(loc):
    u = str(loc).upper()
    if re.search(r'\bU/S\b|\bUPSTREAM\b',   u): return 'UPSTREAM'
    if re.search(r'\bD/S\b|\bDOWNSTREAM\b', u): return 'DOWNSTREAM'
    if re.search(r'\bBEFORE\b',              u): return 'BEFORE'
    if re.search(r'\bAFTER\b',               u): return 'AFTER'
    return 'UNSPECIFIED'

df['Spatial_Tags']     = df['Monitoring_Location'].apply(extract_spatial_tags)
df['Primary_Position'] = df['Monitoring_Location'].apply(get_primary_position)
print("\nPrimary position distribution:")
print(df['Primary_Position'].value_counts())

# ── 8. STATION PAIR / GROUP LABELS ────────────────────────────────────────────
_SPATIAL_RE = (
    r'\bU/S\b|\bD/S\b|\bUPSTREAM\b|\bDOWNSTREAM\b|'
    r'\bBEFORE\b|\bAFTER\b|\bCONF\.?\b|\bCONFLUENCE\b|'
    r'\bDISCHARGE\b|\bOF\b|\bAT\b|\bNEAR\b'
)
df['Station_Base_Name'] = df['Monitoring_Location'].apply(
    lambda loc: re.sub(r'\s+', ' ', re.sub(_SPATIAL_RE, ' ', str(loc).upper())).strip(' ,.-')
)
df['Station_Group'] = df['River_Basin'].str.upper().str.strip() + '|' + df['Station_Base_Name']

# U/S–D/S pair flag via groupby-join (avoids apply column loss)
pos_sets   = df.groupby(['Station_Group', 'Year'])['Primary_Position'].apply(set)
pair_flag  = pos_sets.apply(lambda s: 'UPSTREAM' in s and 'DOWNSTREAM' in s)
pair_flag.name = 'Has_US_DS_Pair'
df = df.join(pair_flag, on=['Station_Group', 'Year'])

print(f"\nRows in a U/S–D/S pair     : {df['Has_US_DS_Pair'].sum()}")
print(f"Unique paired station groups: {df[df['Has_US_DS_Pair']]['Station_Group'].nunique()}")

# ── 9. FEATURE SCALING ────────────────────────────────────────────────────────
SCALE_COLS = MEAN_COLS

mm_vals = MinMaxScaler().fit_transform(df[SCALE_COLS])
zs_vals = StandardScaler().fit_transform(df[SCALE_COLS])
for i, col in enumerate(SCALE_COLS):
    df[col + '_MinMax'] = mm_vals[:, i]
    df[col + '_Zscore'] = zs_vals[:, i]

MINMAX_COLS = [c + '_MinMax' for c in SCALE_COLS]
ZSCORE_COLS = [c + '_Zscore'  for c in SCALE_COLS]
print("\nFeature scaling applied (Min-Max + Z-score).")

# ── 10. ASSEMBLE & EXPORT ─────────────────────────────────────────────────────
ID_COLS      = ['Station_Code', 'Monitoring_Location', 'State', 'River_Basin', 'Year']
SPATIAL_COLS = ['Spatial_Tags', 'Primary_Position', 'Station_Base_Name', 'Station_Group', 'Has_US_DS_Pair']
LABEL_COLS   = ['Compliance_Label', 'Compliance_Label_Encoded', 'Is_Safe', 'Class_Weight', 'Binary_Class_Weight']

final_cols = ID_COLS + SPATIAL_COLS + LABEL_COLS + NUMERIC_COLS + MEAN_COLS + MINMAX_COLS + ZSCORE_COLS
df_out     = df[final_cols].copy()

out_path = os.path.join(BASE_DIR, 'india_water_quality_preprocessed_phase1.csv')
df_out.to_csv(out_path, index=False)

print(f"\n✅  Saved  →  {out_path}")
print(f"    Shape  : {df_out.shape[0]} rows × {df_out.shape[1]} columns")
print(f"\n    ID / Meta         : {len(ID_COLS)}")
print(f"    Spatial tags      : {len(SPATIAL_COLS)}")
print(f"    Labels / Weights  : {len(LABEL_COLS)}")
print(f"    Raw parameters    : {len(NUMERIC_COLS)}")
print(f"    Mean parameters   : {len(MEAN_COLS)}")
print(f"    MinMax scaled     : {len(MINMAX_COLS)}")
print(f"    Z-score scaled    : {len(ZSCORE_COLS)}")