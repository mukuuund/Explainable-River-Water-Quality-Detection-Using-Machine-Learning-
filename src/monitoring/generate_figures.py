import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging

def generate_figures():
    logging.info("Generating dashboard-ready figures...")
    os.makedirs('reports/monitoring/figures', exist_ok=True)
    
    sns.set_theme(style="whitegrid", context="talk")
    
    # 1. Monthly compliance bar chart
    try:
        monthly_df = pd.read_csv('reports/monitoring/monthly_compliance_summary.csv')
        if not monthly_df.empty:
            plt.figure(figsize=(10, 6))
            monthly_df.set_index('month')[['predicted_compliant', 'predicted_non_compliant']].plot(
                kind='bar', stacked=True, color=['#2ecc71', '#e74c3c'], figsize=(10, 6)
            )
            plt.title('Monthly ML Predicted Compliance (NWMP 2025)')
            plt.ylabel('Number of Samples')
            plt.xlabel('Month')
            plt.xticks(rotation=0)
            plt.tight_layout()
            plt.savefig('reports/monitoring/figures/monthly_compliance.png')
            plt.close()
    except FileNotFoundError:
        logging.warning("monthly_compliance_summary.csv not found.")
        
    # 2. Risk category distribution chart
    try:
        df = pd.read_csv('data/processed/nwmp_2025_predictions.csv')
        if not df.empty and 'risk_category' in df.columns:
            plt.figure(figsize=(8, 6))
            cat_order = ['Safe', 'Low', 'Moderate', 'High', 'Severe', 'Unknown']
            palette = ['#2ecc71', '#f1c40f', '#f39c12', '#e67e22', '#c0392b', '#95a5a6']
            
            counts = df['risk_category'].value_counts()
            # Plot only present categories
            plot_cats = [c for c in cat_order if c in counts.index]
            plot_colors = [palette[cat_order.index(c)] for c in plot_cats]
            
            sns.countplot(
                data=df, 
                x='risk_category', 
                hue='risk_category', 
                order=plot_cats, 
                palette=plot_colors,
                legend=False
            )
            plt.title('Distribution of Risk Categories')
            plt.ylabel('Count')
            plt.xlabel('Risk Category')
            plt.subplots_adjust(bottom=0.25, left=0.15, right=0.95)
            plt.tight_layout()
            plt.savefig('reports/monitoring/figures/risk_category_distribution.png')
            plt.close()
    except FileNotFoundError:
        pass
        
    # 3. Top 20 hotspot bar chart
    try:
        top_hotspots = pd.read_csv('reports/monitoring/top_20_hotspots.csv')
        if not top_hotspots.empty:
            plt.figure(figsize=(12, 8))
            sns.barplot(
                data=top_hotspots, 
                x='max_risk_score', 
                y='station_name', 
                hue='hotspot_status',
                dodge=False,
                palette='Reds_r'
            )
            plt.title('Top 20 Persistent/Recurring Hotspots by Max Risk Score')
            plt.xlabel('Max Risk Score')
            plt.ylabel('')
            plt.legend(title='Hotspot Status', loc='lower right')
            plt.tight_layout()
            plt.savefig('reports/monitoring/figures/top_20_hotspots.png')
            plt.close()
    except FileNotFoundError:
        pass
        
    # 4. Alert severity distribution
    try:
        alert_summary = pd.read_csv('reports/monitoring/alert_summary.csv')
        if not alert_summary.empty:
            plt.figure(figsize=(8, 6))
            sns.barplot(
                data=alert_summary, 
                x='severity', 
                y='count', 
                hue='severity', 
                palette='rocket',
                legend=False
            )
            plt.title('Alert Severity Distribution')
            plt.xlabel('Severity')
            plt.ylabel('Number of Alerts')
            plt.subplots_adjust(bottom=0.25, left=0.15, right=0.95)
            plt.tight_layout()
            plt.savefig('reports/monitoring/figures/alert_severity_distribution.png')
            plt.close()
    except FileNotFoundError:
        pass
        
    # 5. Month-wise average risk trend
    try:
        monthly_df = pd.read_csv('reports/monitoring/monthly_compliance_summary.csv')
        if not monthly_df.empty:
            plt.figure(figsize=(10, 6))
            sns.lineplot(data=monthly_df, x='month', y='average_risk_score', marker='o', color='#c0392b', linewidth=3, markersize=10)
            plt.title('Average Risk Score Trend (July-Sep 2025)')
            plt.ylabel('Average Risk Score')
            plt.xlabel('Month')
            plt.ylim(0, 100)
            plt.tight_layout()
            plt.savefig('reports/monitoring/figures/monthly_average_risk_trend.png')
            plt.close()
    except FileNotFoundError:
        pass
        
    logging.info("Figure generation complete.")

if __name__ == "__main__":
    generate_figures()
