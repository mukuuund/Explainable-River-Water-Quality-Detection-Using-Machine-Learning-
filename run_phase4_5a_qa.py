import os
import pandas as pd
import numpy as np
import chardet

def run_qa():
    manifest = pd.read_csv('reports/expanded_data/new_data_file_manifest.csv')
    
    inventory = []
    
    for _, row in manifest.iterrows():
        if row['load_status'] == 'Failed' or row['duplicate_file_flag']:
            continue
            
        filepath = row['absolute_path']
        filename = row['source_file']
        encoding = row['detected_encoding']
        
        try:
            df = pd.read_csv(filepath, encoding=encoding)
            
            for col in df.columns:
                non_null_count = df[col].notnull().sum()
                non_null_pct = non_null_count / len(df) * 100 if len(df) > 0 else 0
                sample_vals = str(df[col].dropna().head(5).tolist())[:100]
                
                normalized = str(col).strip().lower().replace(' ', '_').replace('-', '_')
                
                inventory.append({
                    'source_file': filename,
                    'row_count': len(df),
                    'column_count': len(df.columns),
                    'original_column_name': col,
                    'normalized_column_name': normalized,
                    'non_null_count': non_null_count,
                    'non_null_percentage': non_null_pct,
                    'sample_values': sample_vals,
                    'inferred_possible_parameter': ''
                })
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            
    df_inv = pd.DataFrame(inventory)
    os.makedirs('reports/expanded_data', exist_ok=True)
    df_inv.to_csv('reports/expanded_data/phase4_5a_raw_column_inventory.csv', index=False)
    print("Generated raw column inventory.")

if __name__ == '__main__':
    run_qa()
