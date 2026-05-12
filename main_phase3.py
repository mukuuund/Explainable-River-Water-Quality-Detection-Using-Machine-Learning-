import os
import sys
import subprocess
import logging

sys.path.insert(0, os.path.dirname(__file__))

from src.models.train_models_corrected import train_all
from src.models.prepare_nwmp_validation import prepare_validation

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def main():
    logging.info("Starting Phase 3...")
    
    # 1. Generate EDA Notebook
    logging.info("Generating EDA Notebook...")
    subprocess.run(["python", "scripts/create_eda_nb.py"], check=True)
    
    # 2. Train baseline models
    logging.info("Training baseline models...")
    summary = train_all()
    df_a = summary.get('df_exp_a')
    res_a = summary.get('a_results')
    best_a = summary.get('a_best_name')
    df_b = summary.get('df_exp_b')
    res_b = summary.get('b_results')
    best_b = summary.get('b_best_name')
    res_abl = None
    best_abl = None
    
    # 3. Prepare NWMP validation
    logging.info("Preparing NWMP validation...")
    prepare_validation()
    
    # 4. Print final summary
    print("\n" + "="*50)
    print("PHASE 3 FINAL SUMMARY")
    print("="*50)
    
    if df_a is not None and not df_a.empty and res_a is not None and not res_a.empty:
        print(f"\n[Experiment A: Strict Clean Model]")
        print(f"Row count: {len(df_a)}")
        print(f"Target Distribution (1=Non-Compliant):\n{df_a['target'].value_counts(normalize=True).round(3)*100}")
        best_a_row = res_a[res_a['Model'] == best_a].iloc[0]
        print(f"Best Model: {best_a}")
        print(f"Best Balanced Accuracy: {best_a_row['Balanced_Accuracy']:.4f}")
        print(f"Best Non-Compliant Recall: {best_a_row['Recall_NonCompliant']:.4f}")
        print(f"Best Non-Compliant F1-Score: {best_a_row['F1_NonCompliant']:.4f}")
    else:
        print("\n[Experiment A: Strict Clean Model] Skipped due to insufficient data.")
        
    if df_b is not None and not df_b.empty and res_b is not None and not res_b.empty:
        print(f"\n[Experiment B: Practical Operational Model]")
        print(f"Row count: {len(df_b)}")
        print(f"Target Distribution (1=Non-Compliant):\n{df_b['target'].value_counts(normalize=True).round(3)*100}")
        best_b_row = res_b[res_b['Model'] == best_b].iloc[0]
        print(f"Best Model: {best_b}")
        print(f"Best Balanced Accuracy: {best_b_row['Balanced_Accuracy']:.4f}")
        print(f"Best Non-Compliant Recall: {best_b_row['Recall_NonCompliant']:.4f}")
        print(f"Best Non-Compliant F1-Score: {best_b_row['F1_NonCompliant']:.4f}")
    else:
        print("\n[Experiment B: Practical Operational Model] Skipped due to insufficient data.")
        
    if res_abl is not None:
        best_abl_row = res_abl[res_abl['Model'] == best_abl].iloc[0]
        print(f"\n[Ablation Experiment (No DO, BOD, pH)]")
        print(f"Best Model: {best_abl}")
        print(f"Best Balanced Accuracy: {best_abl_row['Balanced_Accuracy']:.4f}")
        print(f"Best Non-Compliant Recall: {best_abl_row['Recall_NonCompliant']:.4f}")
        print(f"Best Non-Compliant F1-Score: {best_abl_row['F1_NonCompliant']:.4f}")
        
    print(f"\n[NWMP Validation Readiness]")
    print(f"Output saved to: reports/model_results/nwmp_validation_readiness.csv")
    
    print("\n[Saved Output Locations]")
    print("- Notebooks: notebooks/03_eda_model_readiness.ipynb")
    print("- Models: models/")
    print("- Metrics & Confusion Matrices: outputs/model_training/")
    print("\nPhase 3 execution complete.")

if __name__ == "__main__":
    main()
