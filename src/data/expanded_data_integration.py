import os
import glob
import hashlib
import pandas as pd
import chardet
import logging
from typing import List, Dict, Any, Tuple

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

KNOWN_EXCLUDED_FILES = [
    'india_water_quality_preprocessed_phase1.csv',
    'nwmp_july2025.csv',
    'nwmp_august2025_mpcb_0.csv',
    'nwmp_september2025_mpcb_0.csv',
    'swq_biological_parameter_manual_cwc_ap_2021_2025.csv',
    'swq_manual_chemical_parameters_cwc_ap_1961_2020.csv',
    'swq_manual_physical_parameters_cwc_ap_1961_2020.csv'
]

def calculate_sha256(filepath: str) -> str:
    sha256_hash = hashlib.sha256()
    try:
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    except Exception as e:
        return ""

def detect_input_folder() -> str:
    possible_folders = [
        'DATAAAAAAA',
        'dataaaaaa'
    ]
    # Check lowercase starts with dataaaa
    for d in os.listdir('.'):
        if os.path.isdir(d) and d.lower().startswith('dataaaa') and d not in possible_folders:
            possible_folders.append(d)
    
    possible_folders.append(os.path.join('data', 'raw'))
    
    for f in possible_folders:
        if os.path.isdir(f):
            logging.info(f"Detected input folder: {f}")
            return f
    return '.' # fallback to current directory

def infer_state(filename: str) -> str:
    fn_lower = filename.lower()
    if '_as_' in fn_lower or 'cpcb_as' in fn_lower: return 'Assam'
    if '_ar_' in fn_lower or 'cpcb_ar' in fn_lower: return 'Arunachal Pradesh'
    if '_ka_' in fn_lower or 'cpcb_ka' in fn_lower: return 'Karnataka'
    if '_jk_' in fn_lower or 'cpcb_jk' in fn_lower: return 'Jammu and Kashmir'
    if '_ap_' in fn_lower or 'cpcb_ap' in fn_lower: return 'Andhra Pradesh'
    return 'Unknown'

def infer_parameter_group(filename: str) -> str:
    fn_lower = filename.lower()
    if 'biological' in fn_lower: return 'biological'
    if 'chemical' in fn_lower: return 'chemical'
    if 'physical' in fn_lower: return 'physical'
    return 'mixed/unknown'

def build_manifest(input_folder: str, output_manifest_path: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    os.makedirs(os.path.dirname(output_manifest_path), exist_ok=True)
    
    csv_files = glob.glob(os.path.join(input_folder, '*.csv'))
    
    # Calculate hashes of excluded files to help identify them even if renamed
    excluded_hashes = set()
    for excl in KNOWN_EXCLUDED_FILES:
        # Check both input_folder and other potential locations
        possible_paths = [os.path.join(input_folder, excl), os.path.join('data/raw', excl), excl]
        for p in possible_paths:
            if os.path.exists(p):
                excluded_hashes.add(calculate_sha256(p))
    
    manifest_data = []
    
    processed_hashes = set(excluded_hashes)
    
    stats = {
        'total_discovered': len(csv_files),
        'excluded_name': 0,
        'excluded_hash': 0,
        'new_files': 0,
        'failed_files': 0
    }
    
    for filepath in csv_files:
        filename = os.path.basename(filepath)
        
        if filename.lower() in KNOWN_EXCLUDED_FILES:
            stats['excluded_name'] += 1
            logging.info(f"Skipping known file by name: {filename}")
            continue
            
        file_hash = calculate_sha256(filepath)
        if file_hash in processed_hashes:
            stats['excluded_hash'] += 1
            logging.info(f"Skipping duplicate file by hash: {filename}")
            continue
            
        processed_hashes.add(file_hash)
        
        file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
        
        # Detect encoding
        with open(filepath, 'rb') as f:
            rawdata = f.read(100000)
            result = chardet.detect(rawdata)
            encoding = result['encoding'] or 'utf-8'
            
        row_count = 0
        column_count = 0
        load_status = 'Success'
        notes = ''
        
        try:
            df = pd.read_csv(filepath, encoding=encoding, nrows=5)
            column_count = len(df.columns)
            # Count rows without fully loading into memory
            with open(filepath, 'r', encoding=encoding, errors='replace') as f:
                row_count = sum(1 for row in f) - 1 # subtract header
                if row_count < 0: row_count = 0
        except Exception as e:
            load_status = 'Failed'
            notes = str(e)
            stats['failed_files'] += 1
            logging.error(f"Failed to read {filename}: {e}")
            continue
            
        stats['new_files'] += 1
        
        manifest_data.append({
            'source_file': filename,
            'absolute_path': os.path.abspath(filepath),
            'file_size_mb': round(file_size_mb, 2),
            'row_count': row_count,
            'column_count': column_count,
            'detected_encoding': encoding,
            'load_status': load_status,
            'duplicate_file_flag': False,
            'likely_parameter_group': infer_parameter_group(filename),
            'inferred_state_from_filename': infer_state(filename),
            'notes': notes
        })
        
    df_manifest = pd.DataFrame(manifest_data)
    df_manifest.to_csv(output_manifest_path, index=False)
    logging.info(f"Manifest written to {output_manifest_path}")
    
    return manifest_data, stats

if __name__ == "__main__":
    folder = detect_input_folder()
    build_manifest(folder, 'reports/expanded_data/new_data_file_manifest.csv')
