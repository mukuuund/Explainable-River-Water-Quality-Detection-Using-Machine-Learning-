import pandas as pd
import numpy as np
import os
import logging

def analyze_monthly_trends():
    logging.info("Analyzing monthly risk trends...")
    df = pd.read_csv('data/processed/nwmp_2025_predictions.csv')
    
    if 'month' not in df.columns or 'station_name' not in df.columns:
        logging.warning("Missing 'month' or 'station_name'. Cannot analyze trends.")
        return
        
    df['month'] = pd.to_datetime(df['sampling_date'], format='mixed', dayfirst=True).dt.month_name()
    # If parsing failed or sampling_date was bad, try to use 'month' column directly if it's there
    df['month'] = df['month'].fillna('Unknown')
    
    # Filter for July, August, September if possible, or just use what's there
    months_order = ['July', 'August', 'September']
    
    # 1. Monthly Summary
    monthly_summary = df.groupby('month').agg(
        total_samples=('predicted_non_compliant', 'count'),
        predicted_compliant=('predicted_non_compliant', lambda x: (x == 0).sum()),
        predicted_non_compliant=('predicted_non_compliant', lambda x: (x == 1).sum()),
        average_risk_score=('risk_score', 'mean'),
        severe_high_risk_count=('risk_category', lambda x: x.isin(['Severe', 'High']).sum())
    ).reset_index()
    
    os.makedirs('reports/monitoring', exist_ok=True)
    monthly_summary.to_csv('reports/monitoring/monthly_compliance_summary.csv', index=False)
    
    # 2. Station-wise Month-to-Month Change
    # Pivot risk scores by station and month
    pivot_risk = df.pivot_table(index='station_name', columns='month', values='risk_score', aggfunc='mean').reset_index()
    
    # We want to calculate the change if months are present
    available_months = [m for m in months_order if m in pivot_risk.columns]
    
    trend_data = []
    
    for _, row in pivot_risk.iterrows():
        station = row['station_name']
        scores = []
        for m in available_months:
            val = row[m]
            if pd.notna(val):
                scores.append((m, val))
                
        if len(scores) >= 2:
            first_score = scores[0][1]
            last_score = scores[-1][1]
            change = last_score - first_score
            
            # Repeated high risk? (all available scores >= 51)
            repeated_high = all(s >= 51 for _, s in scores)
            
            trend_data.append({
                'station_name': station,
                'first_month': scores[0][0],
                'last_month': scores[-1][0],
                'first_risk_score': first_score,
                'last_risk_score': last_score,
                'risk_change': change,
                'repeated_high_risk': repeated_high
            })
            
    trend_df = pd.DataFrame(trend_data)
    
    if not trend_df.empty:
        trend_df.to_csv('reports/monitoring/monthly_risk_trend_summary.csv', index=False)
        
        # Meaningful increase = change >= 15
        increasing = trend_df[trend_df['risk_change'] >= 15].sort_values('risk_change', ascending=False)
        increasing.to_csv('reports/monitoring/stations_increasing_risk.csv', index=False)
        
        decreasing = trend_df[trend_df['risk_change'] <= -15].sort_values('risk_change')
        decreasing.to_csv('reports/monitoring/stations_decreasing_risk.csv', index=False)
        
        logging.info(f"Found {len(increasing)} stations with increasing risk and {len(decreasing)} with decreasing risk.")
    else:
        logging.info("Not enough multi-month data for trend analysis.")

if __name__ == "__main__":
    analyze_monthly_trends()
