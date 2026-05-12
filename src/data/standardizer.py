import pandas as pd
import numpy as np
import re
import logging
import calendar

def infer_water_body_name_from_station(station_name):
    if pd.isna(station_name) or not isinstance(station_name, str):
        return np.nan
        
    s = station_name.strip()
    
    # "Godavari river at Jaikwadi Dam..." -> Godavari
    # "Mithi river near Road bridge..." -> Mithi
    # "Tarapur MIDC Nalla..." -> Tarapur MIDC Nalla
    m = re.match(r'^([\w\s\.]+?)\s+(river|creek|sea|nala|nalla|dam|lake)\b', s, re.IGNORECASE)
    if m:
        return m.group(1).strip().title()
        
    # "BPT, Navapur..."
    if "BPT" in s or "bpt" in s.lower():
        parts = s.split(',')
        if len(parts) >= 2:
            return parts[0].strip() + "/" + parts[1].strip()
            
    # "Bindusara river at Beed"
    m = re.match(r'^([\w\s]+?)\s+river', s, re.IGNORECASE)
    if m:
        return m.group(1).strip().title()
        
    return np.nan


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizes column names and coalesces canonical parameters.
    """
    df = df.copy()
    
    # 1. Basic snake_case renaming of original columns to ensure clean access
    new_cols = []
    seen = {}
    for col in df.columns:
        snake_cased = re.sub(r'[^a-zA-Z0-9]', '_', str(col).strip().lower())
        snake_cased = re.sub(r'_+', '_', snake_cased).strip('_')
        
        if snake_cased not in seen:
            seen[snake_cased] = 1
            new_cols.append(snake_cased)
        else:
            new_cols.append(f"{snake_cased}_{seen[snake_cased]}")
            seen[snake_cased] += 1
            
    df.columns = new_cols
    
    # 2. Exclusions for coalescing
    exclusions = ['normalized', 'scaled', 'encoded', 'minmax', 'zscore', 'class_weight', 'binary_class_weight', 'is_safe', 'compliance_label_encoded']
    
    # 3. Canonical mappings
    canonical_mappings = {
        'dissolved_oxygen': r'(^do$|dissolved_o2|^do_mg_l$|dissolved_oxygen)',
        'bod': r'(^bod$|biochemical_oxygen_demand|^bod_mg_l$|bod_mean_mgl)',
        'ph': r'(^ph$|^p_h$|^ph_value$)',
        'temperature': r'(^temp$|temperature)',
        'conductivity': r'(^conductivity$|conductance|electric_conductivity)',
        'nitrate': r'(^nitrate$|^nitrate_n$|^nitrate_mg_l$|nitrite_n_nitrate_n|^no3$)',
        'fecal_coliform': r'(^fecal_coliform|faecal_coliform)',
        'total_coliform': r'(^total_coliform)',
        'fecal_streptococci': r'(strepto|fecal_streptococci|faecal_streptococci)',
        'turbidity': r'(turbidity)',
        'cod': r'(^cod$|chemical_oxygen_demand)',
        'total_dissolved_solids': r'(^tds$|total_dissolved_solids)',
        'station_name': r'(station.*name|stn.*name|station.*code|monitoring.*location|^location$)',
        'river_name': r'(name_of_water_body|water_body_name|river.*name)',
        'water_body_type': r'(type_water_body|water_body_type)',
        'sampling_date': r'(^date$|sample_date|monitoring_date|sampling_date|^timestamp$)',
        'month': r'(^month$)',
        'year': r'(^year$)',
        'season': r'(^season$)',
        'latitude': r'(^lat$|^latitude$)',
        'longitude': r'(^lon$|^longitude$)'
    }
    
    # 4. Coalescing logic
    for canon_name, regex_pattern in canonical_mappings.items():
        candidates = []
        for c in df.columns:
            if re.search(regex_pattern, c) and not any(ex in c for ex in exclusions) and c != canon_name:
                candidates.append(c)
                
        if candidates:
            # Check if canonical name already exists from previous clean mapping
            if canon_name in df.columns:
                s = df[canon_name].copy()
                for cand in candidates:
                    s = s.combine_first(df[cand])
                df[canon_name] = s
            else:
                s = df[candidates[0]].copy()
                for cand in candidates[1:]:
                    s = s.combine_first(df[cand])
                df[canon_name] = s
            logging.info(f"Coalesced {canon_name} from {candidates}")
        else:
            if canon_name not in df.columns:
                df[canon_name] = np.nan
            
    # 5. Clean numeric columns
    numeric_features = [
        'dissolved_oxygen', 'bod', 'ph', 'temperature', 'conductivity',
        'nitrate', 'fecal_coliform', 'total_coliform', 'fecal_streptococci',
        'turbidity', 'cod', 'total_dissolved_solids'
    ]
    for col in numeric_features:
        if col in df.columns:
            df[col] = clean_numeric_value(df[col])
            
    # 6. Generate Season if missing
    if 'season' not in df.columns or df['season'].isna().all():
        df['season'] = generate_season(df)
        
    # 7. Fix generic river_name
    generic_types = ['river', 'creek', 'sea', 'nala', 'nalla', 'dam', 'lake', 'pond', 'well']
    if 'river_name' in df.columns and 'station_name' in df.columns:
        df['river_name_source'] = 'existing_valid_value'
        
        is_generic = df['river_name'].astype(str).str.lower().str.strip().isin(generic_types)
        is_missing = df['river_name'].isna() | (df['river_name'].astype(str).str.strip() == '')
        
        needs_fix = is_generic | is_missing
        
        if needs_fix.any():
            inferred = df.loc[needs_fix, 'station_name'].apply(infer_water_body_name_from_station)
            
            # Where inferred is valid, use it
            valid_inferred = inferred.notna() & (inferred.astype(str).str.strip() != '')
            
            df.loc[needs_fix[valid_inferred].index, 'river_name'] = inferred[valid_inferred]
            df.loc[needs_fix[valid_inferred].index, 'river_name_source'] = 'inferred_from_station_name'
            
            # Still missing?
            still_missing = df['river_name'].isna() | (df['river_name'].astype(str).str.strip() == '')
            df.loc[still_missing, 'river_name_source'] = 'unknown'
            
    return df

def clean_numeric_value(series: pd.Series) -> pd.Series:
    """
    Cleans a pandas Series containing mixed numeric/string values.
    Handles 'BDL', 'ND', 'NA', empty strings, and values with units.
    """
    # Create a copy
    s = series.astype(str).str.strip().copy()
    
    # Replace common non-numeric representations with NaN
    s = s.replace(r'(?i)^(nd|na|none|null|)$', np.nan, regex=True)
    
    # Handle "BDL" and variants by stripping them or treating as NaN/0
    # "0.5(BDL)" -> "0.5"
    s = s.str.replace(r'\(BDL\)', '', regex=True, flags=re.IGNORECASE)
    # Standalone "BDL" or "Below Detection Limit" -> NaN (chosen consistently to avoid false zeros)
    s = s.replace(r'(?i)^(bdl|below detection limit)$', np.nan, regex=True)
    
    # Extract leading numbers, ignoring trailing units
    # e.g., "1.8 mg/l" -> "1.8"
    s = s.str.extract(r'([-+]?\d*\.?\d+)', expand=False)
    
    return pd.to_numeric(s, errors='coerce')

def generate_season(df: pd.DataFrame) -> pd.Series:
    """
    Generates season from sampling_date or month.
    """
    months = pd.Series(index=df.index, dtype=float)
    
    if 'sampling_date' in df.columns:
        dt = pd.to_datetime(df['sampling_date'], errors='coerce')
        months = dt.dt.month
        
    if months.isna().any() and 'month' in df.columns:
        # If month is a string like 'July'
        month_map = {v.lower(): k for k,v in enumerate(calendar.month_name) if k != 0}
        month_map.update({v.lower(): k for k,v in enumerate(calendar.month_abbr) if k != 0})
        
        m_str = df.loc[months.isna(), 'month'].astype(str).str.lower().str.strip()
        m_num = pd.to_numeric(m_str, errors='coerce')
        
        # Where it couldn't be parsed as number, try mapping
        m_num[m_num.isna()] = m_str[m_num.isna()].map(month_map)
        months.loc[months.isna()] = m_num

    # Determine season based on Indian context
    # Winter: Dec, Jan, Feb
    # Pre-Monsoon: Mar, Apr, May
    # Monsoon: Jun, Jul, Aug, Sep
    # Post-Monsoon: Oct, Nov
    
    conditions = [
        months.isin([12, 1, 2]),
        months.isin([3, 4, 5]),
        months.isin([6, 7, 8, 9]),
        months.isin([10, 11])
    ]
    choices = ['Winter', 'Pre-Monsoon', 'Monsoon', 'Post-Monsoon']
    
    return pd.Series(np.select(conditions, choices, default=np.nan), index=df.index)
