"""
Phase 5 – Rule-based explanation engine.
Generates transparent, human-readable compliance explanations for every NWMP record.
"""
import os, logging
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def generate_rule_explanations(predictions_path: str, out_dir: str) -> pd.DataFrame:
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(predictions_path, low_memory=False)

    explanations = []
    for _, row in df.iterrows():
        do  = row.get("dissolved_oxygen", np.nan)
        bod = row.get("bod", np.nan)
        ph  = row.get("ph", np.nan)
        lbl = row.get("available_compliance_label", "Unknown")
        risk = row.get("risk_category", "Unknown")
        viol = str(row.get("violation_reasons", "[]"))

        reasons = []
        if pd.notnull(do) and do < 5:
            reasons.append(f"DO is {do:.2f} mg/L (below 5)")
        if pd.notnull(bod) and bod > 3:
            reasons.append(f"BOD is {bod:.2f} mg/L (above 3)")
        if pd.notnull(ph) and (ph < 6.5 or ph > 8.5):
            reasons.append(f"pH is {ph:.2f} (outside 6.5-8.5)")

        if reasons:
            text = f"Non-compliant because {'; '.join(reasons)}. Risk category: {risk}."
        elif lbl == "Compliant":
            text = f"Compliant – all measured core parameters within safe limits. Risk category: {risk}."
        elif all(pd.isna(v) for v in [do, bod, ph]):
            text = "Insufficient core data (DO, BOD, pH all missing) to evaluate compliance."
        else:
            text = f"Label is '{lbl}'. Measured core values are within limits but data may be partial."

        explanations.append(text)

    df["rule_based_explanation"] = explanations

    out_cols = [
        "station_name", "river_name", "state", "district", "month",
        "dissolved_oxygen", "bod", "ph",
        "violation_reasons", "risk_score", "risk_category",
        "label_confidence", "risk_confidence",
        "rule_based_explanation",
    ]
    existing = [c for c in out_cols if c in df.columns]
    df[existing].to_csv(os.path.join(out_dir, "rule_based_explanations.csv"), index=False)
    log.info(f"Rule explanations saved for {len(df)} records.")
    return df          # full dataframe with the new column attached
