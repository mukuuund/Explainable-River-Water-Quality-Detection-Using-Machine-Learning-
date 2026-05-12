"""
Phase 5 – Local station-level and hotspot explanations.
Merges global importance context with per-station rule explanations.
"""
import os, logging
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def _build_local_row(row: pd.Series, top_global_drivers: list) -> dict:
    """Build a single enriched explanation row."""
    viol = str(row.get("violation_reasons", "[]")).lower()
    local_drivers = []
    if "do" in viol:   local_drivers.append("dissolved_oxygen")
    if "bod" in viol:  local_drivers.append("bod")
    if "ph" in viol:   local_drivers.append("ph")
    for d in top_global_drivers:
        if d not in local_drivers and len(local_drivers) < 5:
            local_drivers.append(d)

    pred_lbl = row.get("predicted_compliance_label",
                       row.get("available_compliance_label", "Unknown"))
    rule_exp = row.get("rule_based_explanation", "")
    final = (
        f"Station {row.get('station_name','?')} is predicted {pred_lbl}. "
        f"{rule_exp} Top model drivers: {', '.join(local_drivers)}."
    )
    return {
        "station_name": row.get("station_name", "Unknown"),
        "river_name":   row.get("river_name", "Unknown"),
        "state":        row.get("state", row.get("state_name", "Unknown")),
        "district":     row.get("district", "Unknown"),
        "month":        row.get("month", "Unknown"),
        "predicted_compliance_label": pred_lbl,
        "predicted_non_compliance_probability": row.get("predicted_non_compliance_probability", np.nan),
        "risk_score":      row.get("risk_score", np.nan),
        "risk_category":   row.get("risk_category", "Unknown"),
        "violation_reasons": row.get("violation_reasons", ""),
        "label_confidence":  row.get("label_confidence", "Unknown"),
        "top_5_model_drivers": str(local_drivers),
        "rule_based_explanation": rule_exp,
        "final_human_readable_explanation": final,
    }


def generate_local_explanations(
    df_with_rules: pd.DataFrame,
    hotspots_path: str,
    alerts_path: str,
    global_imp_path: str,
    out_dir: str,
):
    os.makedirs(out_dir, exist_ok=True)

    # load top global drivers
    top_drivers = []
    for p in [global_imp_path,
              global_imp_path.replace("shap_importance", "permutation_importance")]:
        if os.path.exists(p):
            top_drivers = pd.read_csv(p)["feature"].head(5).tolist()
            break

    # ── hotspot explanations ─────────────────────────────────────────
    hotspots = pd.read_csv(hotspots_path) if os.path.exists(hotspots_path) else pd.DataFrame()
    if not hotspots.empty:
        rows = []
        for _, h in hotspots.head(20).iterrows():
            stn = h.get("station_name", "Unknown")
            match = df_with_rules[df_with_rules["station_name"] == stn]
            if match.empty:
                continue
            rows.append(_build_local_row(match.iloc[0], top_drivers))
        if rows:
            pd.DataFrame(rows).to_csv(
                os.path.join(out_dir, "top_hotspot_explanations.csv"), index=False
            )
            log.info(f"Hotspot explanations: {len(rows)} stations.")

    # ── severe alert explanations ────────────────────────────────────
    alerts = pd.read_csv(alerts_path) if os.path.exists(alerts_path) else pd.DataFrame()
    if not alerts.empty:
        severe = alerts[alerts["severity"].isin(["Severe", "Critical", "High"])]
        seen, rows = set(), []
        for _, a in severe.iterrows():
            stn = a.get("station_name", "Unknown")
            if stn in seen:
                continue
            seen.add(stn)
            match = df_with_rules[df_with_rules["station_name"] == stn]
            if match.empty:
                continue
            rows.append(_build_local_row(match.iloc[0], top_drivers))
        if rows:
            pd.DataFrame(rows).to_csv(
                os.path.join(out_dir, "severe_alert_explanations.csv"), index=False
            )
            log.info(f"Severe-alert explanations: {len(rows)} stations.")

    # ── random local sample ──────────────────────────────────────────
    sample = df_with_rules.sample(min(100, len(df_with_rules)), random_state=42)
    rows = [_build_local_row(r, top_drivers) for _, r in sample.iterrows()]
    pd.DataFrame(rows).to_csv(
        os.path.join(out_dir, "local_explanations.csv"), index=False
    )
    log.info(f"Local explanations: {len(rows)} records.")
