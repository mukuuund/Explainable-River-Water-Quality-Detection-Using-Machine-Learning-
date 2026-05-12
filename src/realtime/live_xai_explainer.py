import pandas as pd
import os

INPUT_PREDICTIONS_PATH = "data/processed/realtime/live_sensor_predictions.csv"
OUTPUT_XAI_PATH = "reports/realtime/live_xai_explanations.csv"
GLOBAL_DRIVERS_PATH = "reports/explainability/dashboard_global_drivers.csv"

def generate_xai_explanations():
    print("Generating live XAI explanations...")
    if not os.path.exists(INPUT_PREDICTIONS_PATH):
        raise FileNotFoundError(f"Input file not found: {INPUT_PREDICTIONS_PATH}")

    df = pd.read_csv(INPUT_PREDICTIONS_PATH)
    if df.empty:
        print("Dataframe is empty.")
        return df

    global_drivers_str = "BOD, dissolved_oxygen, pH, temperature, conductivity"
    if os.path.exists(GLOBAL_DRIVERS_PATH):
        try:
            drivers_df = pd.read_csv(GLOBAL_DRIVERS_PATH)
            if 'feature' in drivers_df.columns:
                top_features = drivers_df['feature'].head(5).tolist()
                global_drivers_str = ", ".join(top_features)
        except Exception as e:
            print(f"Could not load global drivers: {e}")

    def generate_row_explanation(row):
        # 1. Rule Explanation
        reasons = []
        if pd.notna(row.get('ph')):
            ph = row['ph']
            if 6.5 <= ph <= 8.5:
                reasons.append(f"pH is {ph}, within the safe range 6.5–8.5.")
            else:
                reasons.append(f"pH is {ph}, outside the safe range 6.5-8.5.")
        
        if pd.notna(row.get('dissolved_oxygen')):
            do = row['dissolved_oxygen']
            if do >= 5:
                reasons.append(f"DO is {do} mg/L, meeting the >=5 mg/L requirement.")
            else:
                reasons.append(f"DO is {do} mg/L, violating the >=5 mg/L requirement.")
                
        if pd.notna(row.get('bod')):
            bod = row['bod']
            if bod <= 3:
                reasons.append(f"BOD is {bod} mg/L, meeting the <=3 mg/L requirement.")
            else:
                reasons.append(f"BOD is {bod} mg/L, violating the <=3 mg/L requirement.")

        rule_exp = " ".join(reasons) if reasons else "No core parameter rules could be evaluated."
        
        # 2. Model Explanation
        pred_label = row.get('live_ml_predicted_compliance_label', 'Unknown')
        pred_status = row.get('live_ml_prediction_status', 'Failed')
        fallback_used = row.get('live_ml_fallback_used', False)
        
        model_exp = ""
        if fallback_used:
            model_exp = "Model inference failed; fallback rule-based estimate used."
        elif pred_status == "Success":
            model_exp = f"The ML model predicts this station is {pred_label}. "
            model_exp += f"The strongest global drivers for the model are generally: {global_drivers_str}."
            
        # 3. Missing Data Explanation
        missing_core = row.get('missing_core_parameters', 'None')
        missing_exp = ""
        if pd.notna(missing_core) and missing_core != 'None':
            missing_exp = f"Warning: {str(missing_core).replace('|', ', ')} are missing from the live feed. "
            missing_exp += "Full DO/BOD/pH compliance cannot be determined from this live record."
        else:
            missing_exp = "No major missing core parameters for this station."
            
        # 4. Final Explanation Synthesis
        final_exp = ""
        core_count = row.get('core_parameter_count', 0)
        
        if core_count == 3:
            if fallback_used:
                final_exp = f"Station {row.get('station_name', 'Unknown')} is classified {pred_label} by rule fallback. " + \
                            (str(row.get('live_violation_reasons', '')).replace('|', ' and ')) + ". " + model_exp
            else:
                final_exp = f"Station {row.get('station_name', 'Unknown')} is classified {pred_label} by rule and ML. " + \
                            (str(row.get('live_violation_reasons', '')).replace('|', ' and ')) + ". " + \
                            "The prediction has high confidence because DO, BOD, and pH are available. " + \
                            f"Top model drivers are {global_drivers_str}."
        elif core_count > 0:
            final_exp = f"Latest live data indicates {rule_exp} However, {str(missing_core).replace('|', ', ')} are missing, " + \
                        "so full regulatory compliance cannot be confirmed. The ML model provides a limited-confidence estimate " + \
                        "because key regulatory features are unavailable."
        else:
            final_exp = "No core parameters (DO, BOD, pH) are available in the live feed. Cannot assess compliance."

        return pd.Series({
            'live_rule_explanation': rule_exp,
            'live_model_explanation': model_exp,
            'live_xai_top_drivers': global_drivers_str,
            'live_missing_data_explanation': missing_exp,
            'live_final_explanation': final_exp
        })

    xai_df = df.apply(generate_row_explanation, axis=1)
    df = pd.concat([df, xai_df], axis=1)
    
    # Save a subset specifically for XAI if needed, but keeping in main df is fine too
    os.makedirs(os.path.dirname(OUTPUT_XAI_PATH), exist_ok=True)
    
    # For the separate XAI report file
    xai_cols = ['station_id', 'station_name', 'latest_timestamp', 
                'live_rule_explanation', 'live_model_explanation', 
                'live_xai_top_drivers', 'live_missing_data_explanation', 
                'live_final_explanation']
    
    out_df = df[[c for c in xai_cols if c in df.columns]]
    out_df.to_csv(OUTPUT_XAI_PATH, index=False)
    print(f"XAI explanations saved to {OUTPUT_XAI_PATH}")
    
    # Also save the enriched main dataframe
    enriched_path = INPUT_PREDICTIONS_PATH.replace(".csv", "_enriched.csv")
    df.to_csv(enriched_path, index=False)
    
    return df

if __name__ == "__main__":
    generate_xai_explanations()
