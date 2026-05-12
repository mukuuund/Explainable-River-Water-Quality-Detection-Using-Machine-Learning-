import pandas as pd
import numpy as np
import re
import logging
from typing import Dict, List, Tuple, Any

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def get_canonical_mappings() -> Dict[str, Dict[str, Any]]:
    # Create explicit aliases based on user requirements.
    # Note: we will strip non-alphanumeric characters for comparison.
    return {
        # Core
        'dissolved_oxygen': {
            'group': 'core', 
            'aliases': ['do', 'dissolvedoxygen', 'dissolvedo2', 'dissolvedoxygenmgl', 'domgl']
        },
        'bod': {
            'group': 'core', 
            'aliases': ['bod', 'biochemicaloxygendemand', 'bodmgl', 'biochemicaloxygendemandmgl']
        },
        'ph': {
            'group': 'core', 
            'aliases': ['ph', 'phvalue', 'ph1', 'ph2', 'potentialofhydrogenph']
        },
        
        # Physical
        'temperature': {
            'group': 'physical', 
            'aliases': ['temp', 'temperature', 'temperatureoc']
        },
        'conductivity': {
            'group': 'physical', 
            'aliases': ['conductivity', 'conductance', 'specificconductance', 'electricalconductivity', 'electricconductivity', 'electricconductivityscm', 'ec', 'ecuscm', 'electricconductivitymscm', 'electricconductivityuscm']
        },
        'turbidity': {
            'group': 'physical', 
            'aliases': ['turbidity', 'turbidityntu']
        },
        'total_dissolved_solids': {
            'group': 'physical', 
            'aliases': ['totaldissolvedsolids', 'totaldissolvedsolidsmgl', 'tds']
        },
        'total_solids': {
            'group': 'physical', 
            'aliases': ['ts', 'totalsolids', 'tfs', 'totalsolidsmgl']
        },
        'total_suspended_solids': {
            'group': 'physical', 
            'aliases': ['tss', 'totalsuspendedsolids', 'totalsuspendedsolidsmgl']
        },
        'flow': {
            'group': 'physical', 
            'aliases': ['flow']
        },
        'depth': {
            'group': 'physical', 
            'aliases': ['depth', 'approxdepth']
        },
        'weather': {
            'group': 'physical', 
            'aliases': ['weather']
        },
        'colour': {
            'group': 'physical', 
            'aliases': ['colour', 'color']
        },
        'odour': {
            'group': 'physical', 
            'aliases': ['odour', 'odor']
        },
        'floating_matter': {
            'group': 'physical', 
            'aliases': ['floatingmatter']
        },
        'visibility_effluent_discharge': {
            'group': 'physical', 
            'aliases': ['visibilityeffluentdischarge']
        },
        
        # Biological
        'fecal_coliform': {
            'group': 'biological', 
            'aliases': ['fecalcoliform', 'faecalcoliform', 'fc', 'fecalcoliformmpn100ml']
        },
        'total_coliform': {
            'group': 'biological', 
            'aliases': ['totalcoliform', 'tc', 'totalcoliformmpn100ml']
        },
        'fecal_streptococci': {
            'group': 'biological', 
            'aliases': ['fecalstreptococci', 'faecalstreptococci']
        },
        'e_coli': {
            'group': 'biological', 
            'aliases': ['ecoli']
        },
        
        # Chemical
        'nitrate': {
            'group': 'chemical', 
            'aliases': ['nitrate', 'nitraten', 'nitratemgl', 'no3', 'nitritennitraten', 'nitritennitratenmgnl', 'nitritennitratenmgnl', 'nitratenmgnl']
        },
        'nitrite': {
            'group': 'chemical', 
            'aliases': ['nitrite', 'nitriten']
        },
        'ammonia': {
            'group': 'chemical', 
            'aliases': ['ammonia', 'amonian', 'ammonian', 'ammonicalnitrogen', 'amonianmgnl']
        },
        'phosphate': {
            'group': 'chemical', 
            'aliases': ['phosphate', 'totalphosphorus', 'totalphosphorusmgpl', 'orthophosphate']
        },
        'carbonate': {
            'group': 'chemical', 
            'aliases': ['carbonate', 'carbonatemgl']
        },
        'bicarbonate': {
            'group': 'chemical', 
            'aliases': ['bicarbonate', 'bicarbonatemgl']
        },
        'alkalinity': {
            'group': 'chemical', 
            'aliases': ['alkalinity', 'totalalkalinity', 'totalalkalinitymglascaco3']
        },
        'chloride': {
            'group': 'chemical', 
            'aliases': ['chloride', 'chlorides', 'chloridemgl']
        },
        'sulphate': {
            'group': 'chemical', 
            'aliases': ['sulphate', 'sulfate', 'sulphatemgl', 'sulfatemgl']
        },
        'fluoride': {
            'group': 'chemical', 
            'aliases': ['fluoride', 'fluoridemgl']
        },
        'calcium': {
            'group': 'chemical', 
            'aliases': ['calcium', 'calciummgl']
        },
        'magnesium': {
            'group': 'chemical', 
            'aliases': ['magnesium', 'magnesiummgl']
        },
        'sodium': {
            'group': 'chemical', 
            'aliases': ['sodium', 'sodiummgl']
        },
        'potassium': {
            'group': 'chemical', 
            'aliases': ['potassium', 'potassiummgl']
        },
        'hardness': {
            'group': 'chemical', 
            'aliases': ['hardness', 'totalhardness', 'totalhardnessmgcaco3l', 'hardnesscaco3', 'hardnessascaco3', 'hardnesscalciummgcaco3l', 'hardnessmagnesiummglascaco3']
        },
        'cod': {
            'group': 'chemical', 
            'aliases': ['cod', 'chemicaloxygendemand', 'chemicaloxygendemandmgl']
        },
        'sodium_adsorption_ratio': {
            'group': 'chemical', 
            'aliases': ['sar', 'sodiumadsorptionratio', 'sodiumadsorptionratio']
        },
        
        # Heavy Metals
        'arsenic': {'group': 'chemical', 'aliases': ['arsenic', 'arsenicmgl']},
        'cadmium': {'group': 'chemical', 'aliases': ['cadmium', 'cadmiummgl']},
        'chromium': {'group': 'chemical', 'aliases': ['chromium', 'chromiummgl']},
        'copper': {'group': 'chemical', 'aliases': ['copper', 'coppermgl']},
        'iron': {'group': 'chemical', 'aliases': ['iron', 'ironmgl']},
        'lead': {'group': 'chemical', 'aliases': ['lead', 'leadmgl']},
        'manganese': {'group': 'chemical', 'aliases': ['manganese', 'manganesemgl']},
        'mercury': {'group': 'chemical', 'aliases': ['mercury', 'mercurymgl']},
        'nickel': {'group': 'chemical', 'aliases': ['nickel', 'nickelmgl']},
        'zinc': {'group': 'chemical', 'aliases': ['zinc', 'zincmgl']},
        'boron': {'group': 'chemical', 'aliases': ['boron', 'boronmgl']},
        
        # Metadata
        'station_name': {'group': 'metadata', 'aliases': ['stationname', 'stationcode', 'sitecode', 'monitoringlocation', 'location', 'station', 'stnname']},
        'river_name': {'group': 'metadata', 'aliases': ['rivername', 'waterbody', 'nameofwaterbody', 'basin', 'tributary', 'subtributary', 'subsubtributary', 'localriver', 'river']},
        'state': {'group': 'metadata', 'aliases': ['statename', 'state', 'statelgdcode']},
        'district': {'group': 'metadata', 'aliases': ['districtname', 'district', 'districtlgdcode', 'tehsil', 'block', 'village']},
        'agency': {'group': 'metadata', 'aliases': ['agencyname', 'agency', 'monitoringagency', 'monagency']},
        'sampling_date': {'group': 'metadata', 'aliases': ['date', 'samplingdate', 'dataacquisitiontime', 'samplingtime', 'frequency']},
        'year': {'group': 'metadata', 'aliases': ['year']},
        'month': {'group': 'metadata', 'aliases': ['month']},
        'season': {'group': 'metadata', 'aliases': ['season']},
        'latitude': {'group': 'metadata', 'aliases': ['lat', 'latitude']},
        'longitude': {'group': 'metadata', 'aliases': ['lon', 'longitude']},
        'human_activities': {'group': 'metadata', 'aliases': ['humanactivities', 'majorpollutingsources', 'useofwaterindownstream', 'usebasedclass']},
        'sample_id': {'group': 'metadata', 'aliases': ['sampleid', 'slno', 'id']}
    }

def clean_col_name(col: str) -> str:
    """Removes non-alphanumeric characters to create a continuous string for alias matching"""
    return re.sub(r'[^a-zA-Z0-9]', '', str(col).lower())

def standardize_columns_and_map(df: pd.DataFrame, source_filename: str, file_state: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    
    original_cols = list(df.columns)
    new_cols = []
    col_mapping_audit = []
    
    for col in df.columns:
        # Standard snake case just for safe dataframe column names in intermediate steps
        snake_cased = re.sub(r'[^a-zA-Z0-9]', '_', str(col).strip().lower())
        snake_cased = re.sub(r'_+', '_', snake_cased).strip('_')
        new_cols.append(snake_cased)
        
    df.columns = new_cols
    
    canonical_mappings = get_canonical_mappings()
    
    out_df = pd.DataFrame()
    out_df['source_file'] = [source_filename] * len(df)
    out_df['data_origin'] = ['new_upload'] * len(df)
    
    unmapped_cols = []
    
    for i, orig_col in enumerate(original_cols):
        clean_col = new_cols[i]
        alphanum_col = clean_col_name(orig_col)
        
        matched_canon = None
        match_confidence = 'None'
        method = 'None'
        
        # Explicitly skip class
        if alphanum_col == 'class':
            unmapped_cols.append({'source_file': source_filename, 'original_column': orig_col, 'normalized_column': clean_col, 'reason': 'Ignored class variable'})
            continue
            
        # Search aliases
        for canon_name, info in canonical_mappings.items():
            if alphanum_col in info['aliases']:
                matched_canon = canon_name
                match_confidence = 'High'
                method = 'Alias Match'
                break
                
        # Heavy metal fallback ending with mgl
        if not matched_canon and alphanum_col.endswith('mgl'):
            for hm in ['arsenic', 'cadmium', 'chromium', 'copper', 'iron', 'lead', 'manganese', 'mercury', 'nickel', 'zinc', 'boron']:
                if alphanum_col.startswith(hm):
                    matched_canon = hm
                    match_confidence = 'Medium'
                    method = 'Suffix Match'
                    break
        
        # Temperature fallback
        if not matched_canon and alphanum_col.startswith('temperature'):
            matched_canon = 'temperature'
            match_confidence = 'Medium'
            method = 'Prefix Match'
            
        if matched_canon:
            if matched_canon not in out_df.columns:
                out_df[matched_canon] = df[clean_col]
            else:
                out_df[matched_canon] = out_df[matched_canon].combine_first(df[clean_col])
                
            col_mapping_audit.append({
                'source_file': source_filename,
                'original_column': orig_col,
                'normalized_column': clean_col,
                'canonical_column': matched_canon,
                'mapping_method': method,
                'mapping_confidence': match_confidence,
                'parameter_group': canonical_mappings[matched_canon]['group'],
                'non_null_count': df[clean_col].notnull().sum(),
                'example_values': str(df[clean_col].dropna().head(3).tolist())[:100],
                'notes': ''
            })
        else:
            unmapped_cols.append({
                'source_file': source_filename,
                'original_column': orig_col,
                'normalized_column': clean_col,
                'reason': 'No match found'
            })
            
    audit_df = pd.DataFrame(col_mapping_audit) if col_mapping_audit else pd.DataFrame(columns=['source_file', 'original_column', 'normalized_column', 'canonical_column', 'mapping_method', 'mapping_confidence', 'parameter_group', 'non_null_count', 'example_values', 'notes'])
    unmapped_df = pd.DataFrame(unmapped_cols) if unmapped_cols else pd.DataFrame(columns=['source_file', 'original_column', 'normalized_column', 'reason'])
    
    if 'state' not in out_df.columns and file_state != 'Unknown':
        out_df['state'] = file_state
        
    return out_df, audit_df, unmapped_df
