import os
import pandas as pd
import logging

def load_data(raw_data_dir: str = "data/raw", fallback_dir: str = "DATAAAAAAA") -> dict[str, pd.DataFrame]:
    """
    Loads all CSV files from raw_data_dir. If empty or missing, tries fallback_dir.
    Returns a dictionary of {filename: DataFrame}.
    """
    target_dir = raw_data_dir
    
    if not os.path.exists(target_dir) or not any(f.endswith('.csv') for f in os.listdir(target_dir)):
        logging.info(f"Directory {target_dir} is empty or missing CSVs. Falling back to {fallback_dir}.")
        target_dir = fallback_dir

    if not os.path.exists(target_dir):
        logging.error(f"Both {raw_data_dir} and {fallback_dir} do not exist.")
        return {}

    dataframes = {}
    for filename in os.listdir(target_dir):
        if filename.endswith('.csv'):
            filepath = os.path.join(target_dir, filename)
            try:
                # Try reading with utf-8, fallback to latin1 if needed
                try:
                    df = pd.read_csv(filepath, encoding='utf-8')
                except UnicodeDecodeError:
                    df = pd.read_csv(filepath, encoding='latin1')
                dataframes[filename] = df
                logging.info(f"Successfully loaded {filename} with shape {df.shape}")
            except Exception as e:
                logging.error(f"Error loading {filename}: {e}")
                
    return dataframes
