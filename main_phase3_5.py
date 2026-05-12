import os
import subprocess
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def run_audits():
    scripts = [
        'src/models/audit_feature_leakage.py',
        'src/models/audit_duplicates.py',
        'src/models/sanity_check_shuffled_labels.py',
        'src/models/audit_group_and_restricted.py',
        'src/models/audit_true_ablation.py'
    ]
    
    for script in scripts:
        logging.info(f"Running {script}...")
        module_name = script.replace('.py', '').replace('/', '.')
        subprocess.run(["python", "-m", module_name], check=True)

def evaluate_audits():
    audit_failed = False
    reasons = []
    
    # 1. Feature Leakage
    leakage_df = pd.read_csv('reports/model_results/feature_leakage_audit.csv')
    if not leakage_df.empty:
        audit_failed = True
        reasons.append(f"Found {len(leakage_df)} leakage columns in training features.")
        
    # 2. Duplicate Audit
    dup_df = pd.read_csv('reports/model_results/duplicate_audit_report.csv')
    if not dup_df.empty:
        tt_contam = dup_df['Train_Test_Contamination_Count'].iloc[0]
        if tt_contam > 0:
            audit_failed = True
            reasons.append(f"Found {tt_contam} instances of harmful train/test contamination.")
            
    # 3. Shuffled Labels
    shuffled_df = pd.read_csv('reports/model_results/shuffled_label_sanity_metrics.csv')
    if not shuffled_df.empty:
        shuf_bal_acc = shuffled_df['Balanced_Accuracy'].iloc[0]
        shuf_roc = shuffled_df['ROC_AUC'].iloc[0]
        if shuf_bal_acc > 0.6 or shuf_roc > 0.6:
            audit_failed = True
            reasons.append(f"Shuffled-label test performed suspiciously well (Bal Acc: {shuf_bal_acc:.2f}, ROC AUC: {shuf_roc:.2f}).")
            
    # 4. Group Split
    group_df = pd.read_csv('reports/model_results/group_split_model_metrics.csv')
    if not group_df.empty:
        # Check if Random vs Group collapsed completely.
        # But wait, we only ran group split in audit_group_and_restricted. Let's see if it's < 0.5.
        group_bal_acc = group_df['Balanced_Accuracy'].mean()
        if group_bal_acc < 0.55:
            audit_failed = True
            reasons.append(f"Group split performance collapsed near random (Average Bal Acc: {group_bal_acc:.2f}).")
            
    # 5. True Auxiliary
    aux_df = pd.read_csv('reports/model_results/true_auxiliary_ablation_metrics.csv')
    if not aux_df.empty:
        random_aux = aux_df[aux_df['Split_Type'] == 'Random_Split']
        if not random_aux.empty and random_aux['Balanced_Accuracy'].iloc[0] > 0.98:
            audit_failed = True
            reasons.append(f"True auxiliary-only model still perfectly memorized the target (Bal Acc: {random_aux['Balanced_Accuracy'].iloc[0]:.4f}).")

    return audit_failed, reasons, leakage_df, dup_df, shuffled_df, group_df, aux_df

def generate_summary(audit_failed, reasons, leakage_df, dup_df, shuffled_df, group_df, aux_df):
    summary_path = 'reports/model_results/phase3_5_audit_summary.md'
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("# Phase 3.5: Model Sanity Audit and Leakage Check\n\n")
        
        if audit_failed:
            f.write("## ❌ AUDIT FAILED\n")
            f.write("The audit detected severe issues that must be addressed before proceeding to Phase 4.\n\n")
            f.write("### Reasons for Failure:\n")
            for r in reasons:
                f.write(f"- {r}\n")
        else:
            f.write("## ✅ AUDIT PASSED\n")
            f.write("No leakage, harmful contamination, or impossible memorization was detected. It is safe to proceed to Phase 4.\n")
            
        f.write("\n---\n\n")
        
        f.write("### 1. Feature Leakage Audit\n")
        if not leakage_df.empty:
            f.write("Suspicious columns found:\n")
            f.write(leakage_df.to_markdown(index=False))
        else:
            f.write("No leakage or hidden core variants found in the final features.\n")
            
        f.write("\n### 2. Duplicate Contamination Audit\n")
        f.write(dup_df.to_markdown(index=False))
        
        f.write("\n\n### 3. Shuffled-Label Sanity Result\n")
        f.write(shuffled_df.to_markdown(index=False))
        f.write("\n\n(Values should be near ~0.5. If high, indicates leakage).\n")
        
        f.write("\n### 4. Group Split & Restricted Model Comparison\n")
        f.write(group_df.to_markdown(index=False))
        
        f.write("\n\n### 5. True Auxiliary-Only Performance\n")
        f.write(aux_df.to_markdown(index=False))
        
    return summary_path

def main():
    logging.info("Starting Phase 3.5 Audits...")
    run_audits()
    
    audit_failed, reasons, leakage_df, dup_df, shuffled_df, group_df, aux_df = evaluate_audits()
    summary_path = generate_summary(audit_failed, reasons, leakage_df, dup_df, shuffled_df, group_df, aux_df)
    
    print("\n" + "="*50)
    print("PHASE 3.5 AUDIT FINAL SUMMARY")
    print("="*50)
    
    if audit_failed:
        print("\nAudit failed: revise feature set / splitting strategy before Phase 4.")
        for r in reasons:
            print(f" - {r}")
    else:
        print("\nAudit passed: safe to proceed to Phase 4.")
        
    print(f"\nDetailed report saved to: {summary_path}")

if __name__ == "__main__":
    main()
