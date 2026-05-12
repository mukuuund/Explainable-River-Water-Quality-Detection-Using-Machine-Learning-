import pandas as pd
import os

INPUT_READY_PATH = "data/processed/realtime/live_sensor_model_ready.csv"
OUTPUT_COMPLIANCE_PATH = "data/processed/realtime/live_sensor_compliance.csv"

def check_rule(param, value):
    if pd.isna(value):
        return None # Missing
    
    if param == 'dissolved_oxygen':
        return value >= 5.0
    elif param == 'bod':
        return value <= 3.0
    elif param == 'ph':
        return 6.5 <= value <= 8.5
    return None

def generate_reason(param, value, passed):
    if pd.isna(value):
        return None
    if passed:
        if param == 'dissolved_oxygen':
            return f"DO is {value} (>= 5 mg/L)"
        elif param == 'bod':
            return f"BOD is {value} (<= 3 mg/L)"
        elif param == 'ph':
            return f"pH is {value} (6.5-8.5)"
    else:
        if param == 'dissolved_oxygen':
            return f"DO is {value} (< 5 mg/L)"
        elif param == 'bod':
            return f"BOD is {value} (> 3 mg/L)"
        elif param == 'ph':
            return f"pH is {value} (out of 6.5-8.5)"
    return None

def apply_compliance_logic():
    print("Applying live compliance logic...")
    if not os.path.exists(INPUT_READY_PATH):
        raise FileNotFoundError(f"File not found: {INPUT_READY_PATH}")

    df = pd.read_csv(INPUT_READY_PATH)
    if df.empty:
        print("Dataframe is empty.")
        return df

    def evaluate_row(row):
        do_val = row.get('dissolved_oxygen', pd.NA)
        bod_val = row.get('bod', pd.NA)
        ph_val = row.get('ph', pd.NA)

        do_pass = check_rule('dissolved_oxygen', do_val)
        bod_pass = check_rule('bod', bod_val)
        ph_pass = check_rule('ph', ph_val)

        evals = [('dissolved_oxygen', do_val, do_pass), 
                 ('bod', bod_val, bod_pass), 
                 ('ph', ph_val, ph_pass)]
        
        available_evals = [e for e in evals if e[2] is not None]
        missing_evals = [e for e in evals if e[2] is None]

        violations = [e for e in available_evals if e[2] == False]
        violation_reasons = [generate_reason(e[0], e[1], e[2]) for e in violations]
        
        strict_label = "Insufficient_Data"
        avail_label = "Insufficient_Data"
        scope = "Insufficient for compliance"
        confidence = "Low"

        core_count = len(available_evals)
        
        if core_count == 3:
            scope = "Full DO/BOD/pH compliance"
            confidence = "High"
            if len(violations) > 0:
                strict_label = "Non-Compliant"
                avail_label = "Non-Compliant"
            else:
                strict_label = "Compliant"
                avail_label = "Compliant"
        elif core_count > 0:
            scope = "Partial compliance based on available core parameters"
            confidence = "Medium" if core_count == 2 else "Low"
            strict_label = "Insufficient_Data"
            if len(violations) > 0:
                avail_label = "Non-Compliant_Based_On_Available_Parameters"
            else:
                avail_label = "Compliant_Based_On_Available_Parameters"
        
        return pd.Series({
            'live_available_compliance_label': avail_label,
            'live_strict_compliance_label': strict_label,
            'live_violation_reasons': " | ".join(violation_reasons) if violation_reasons else "None",
            'live_violation_count': len(violations),
            'live_rule_confidence': confidence,
            'live_compliance_scope': scope
        })

    compliance_df = df.apply(evaluate_row, axis=1)
    df = pd.concat([df, compliance_df], axis=1)

    os.makedirs(os.path.dirname(OUTPUT_COMPLIANCE_PATH), exist_ok=True)
    df.to_csv(OUTPUT_COMPLIANCE_PATH, index=False)
    print(f"Compliance applied. Saved to {OUTPUT_COMPLIANCE_PATH}")
    return df

if __name__ == "__main__":
    apply_compliance_logic()
