import pandas as pd
import os
import logging
import re

def parse_tags(name):
    name = str(name).lower()
    
    # Position
    if re.search(r'\b(u/s|upstream|before)\b', name):
        pos = 'Upstream'
    elif re.search(r'\b(d/s|downstream|after)\b', name):
        pos = 'Downstream'
    else:
        pos = 'Unknown'
        
    # Context
    if re.search(r'\b(drain|effluent|discharge|stp|cetp|industry|nallah|sewage)\b', name):
        context = 'Point Source / Drain'
    elif re.search(r'\b(confluence|sangam)\b', name):
        context = 'Confluence'
    elif re.search(r'\b(lake|pond|reservoir|tank)\b', name):
        context = 'Lentic Waterbody'
    elif re.search(r'\b(well|groundwater|piezometer|tubewell)\b', name):
        context = 'Groundwater'
    else:
        context = 'Riverine / General'
        
    return pd.Series([pos, context])

def tag_spatial_data():
    logging.info("Tagging spatial contexts...")
    
    # Load original dataset
    df = pd.read_csv('data/processed/nwmp_2025_predictions.csv')
    
    if 'station_name' not in df.columns:
        logging.warning("No station_name column found. Skipping spatial tagging.")
        return
        
    df[['station_position_tag', 'pollution_context_tag']] = df['station_name'].apply(parse_tags)
    
    os.makedirs('reports/monitoring', exist_ok=True)
    
    # Save back to predictions file so downstream tools can use it
    df.to_csv('data/processed/nwmp_2025_predictions.csv', index=False)
    
    # Generate summaries
    pos_summary = df['station_position_tag'].value_counts().reset_index()
    pos_summary.columns = ['station_position_tag', 'count']
    
    ctx_summary = df['pollution_context_tag'].value_counts().reset_index()
    ctx_summary.columns = ['pollution_context_tag', 'count']
    
    summary = pd.merge(pos_summary, ctx_summary, left_index=True, right_index=True, how='outer')
    summary.to_csv('reports/monitoring/upstream_downstream_tag_summary.csv', index=False)
    
    logging.info(f"Spatial tagging complete. Upstream: {pos_summary[pos_summary['station_position_tag']=='Upstream']['count'].sum() if 'Upstream' in pos_summary['station_position_tag'].values else 0}, Downstream: {pos_summary[pos_summary['station_position_tag']=='Downstream']['count'].sum() if 'Downstream' in pos_summary['station_position_tag'].values else 0}")

if __name__ == "__main__":
    tag_spatial_data()
