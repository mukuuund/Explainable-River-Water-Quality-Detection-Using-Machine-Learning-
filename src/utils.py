import os
import shutil
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def setup_project_structure(base_dir: str):
    """Creates the required project directories."""
    directories = [
        "data/raw",
        "data/processed",
        "reports/figures",
        "reports/tables",
        "src/data",
        "src/features"
    ]
    for directory in directories:
        dir_path = os.path.join(base_dir, directory)
        os.makedirs(dir_path, exist_ok=True)
        # logging.info(f"Ensured directory exists: {dir_path}")

def copy_raw_data(base_dir: str, source_dir: str = "DATAAAAAAA", target_dir: str = "data/raw"):
    """Copies CSV files from the source directory to the target raw data directory."""
    source_path = os.path.join(base_dir, source_dir)
    target_path = os.path.join(base_dir, target_dir)
    
    if not os.path.exists(source_path):
        logging.warning(f"Source directory '{source_path}' does not exist. Skipping file copy.")
        return

    os.makedirs(target_path, exist_ok=True)

    for filename in os.listdir(source_path):
        if filename.endswith(".csv"):
            src_file = os.path.join(source_path, filename)
            dst_file = os.path.join(target_path, filename)
            if not os.path.exists(dst_file):
                shutil.copy2(src_file, dst_file)
                logging.info(f"Copied {filename} to {target_dir}")
