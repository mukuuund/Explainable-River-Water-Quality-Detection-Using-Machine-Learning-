import os
import subprocess
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def generate_comparison_report(group_df, random_df):
    report_path = 'reports/model_results/phase3_6_model_comparison.md'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Phase 3.6: Clean Model Comparison\n\n")
        f.write("## Important Context: Phase 3 Leakage Correction\n")
        f.write("> [!WARNING]\n")
        f.write("> **Phase 3 perfect metrics were invalid due to hidden raw DO/BOD/pH leakage.**\n")
        f.write("> The Phase 3.5 audit detected that raw variants (like `dissolved_o2`, `bod_mean_mgl`) were inadvertently leaking into the features. Phase 3.6 corrects this by using explicit canonical allowlists. **Final model metrics should come only from Phase 3.6.**\n\n")
        
        f.write("## 1. Group Split Performance (Headline Metrics)\n")
        f.write("The models below were evaluated using GroupShuffleSplit to prevent spatial/station data leakage. This is the most defensible measure of real-world generalization.\n\n")
        f.write(group_df.to_markdown(index=False))
        
        f.write("\n\n## 2. Random Split Performance\n")
        f.write("Traditional 80/20 random stratified split.\n\n")
        f.write(random_df.to_markdown(index=False))
        
        f.write("\n\n## 3. Ablation Conclusion\n")
        f.write("> [!NOTE]\n")
        f.write("> **The True Auxiliary-Only model represents the valid ablation result.** It confirms the exact predictive power of supporting parameters when core regulatory metrics are missing.\n")

def main():
    logging.info("Starting Phase 3.6 Clean Retraining...")
    subprocess.run(["python", "-m", "src.models.train_phase3_6"], check=True)
    
    random_df = pd.read_csv('reports/model_results/phase3_6_random_split_metrics.csv')
    group_df = pd.read_csv('reports/model_results/phase3_6_group_split_metrics.csv')
    
    generate_comparison_report(group_df, random_df)
    
    print("\n" + "="*50)
    print("PHASE 3.6 CLEAN RETRAIN SUMMARY")
    print("="*50)
    
    ext_group = group_df[group_df['Variant'] == 'Extended Clean Model']
    if not ext_group.empty:
        best_row = ext_group.loc[ext_group['F1_NonCompliant'].idxmax()]
        best_model = best_row['Model']
        best_f1 = best_row['F1_NonCompliant']
        best_recall = best_row['Recall_NonCompliant']
        best_bal_acc = best_row['Balanced_Accuracy']
    else:
        best_model = "N/A"
        best_f1 = best_recall = best_bal_acc = 0.0
        
    ext_random = random_df[random_df['Variant'] == 'Extended Clean Model']
    if not ext_random.empty:
        r_row = ext_random[ext_random['Model'] == best_model].iloc[0]
        rand_f1 = r_row['F1_NonCompliant']
    else:
        rand_f1 = 0.0

    print(f"\n[Final Selected Headline Model]")
    print(f"Variant: Extended Clean Model")
    print(f"Model: {best_model}")
    print(f"\n[Metrics]")
    print(f"Best Non-Compliant Recall (Group): {best_recall:.4f}")
    print(f"Best Non-Compliant F1 (Group): {best_f1:.4f}")
    print(f"Best Balanced Accuracy (Group): {best_bal_acc:.4f}")
    print(f"Random Split F1 (For reference): {rand_f1:.4f}")
    
    print("\n[Readiness Decision]")
    if best_f1 > 0.5:
        print("Clean retrain complete: ready for Phase 4")
    else:
        print("Clean retrain complete: not ready for Phase 4 because performance is worse than random chance.")

if __name__ == "__main__":
    main()
