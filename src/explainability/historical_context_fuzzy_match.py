"""
Phase 5.1 -- Historical context fuzzy matching.
Attempts relaxed matching between NWMP hotspot stations and the expanded
historical baseline, which uses different naming conventions.
"""
import os, re, logging
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# State LGD code -> name mapping (from the expanded baseline files)
_STATE_CODE_MAP = {
    1: "Jammu & Kashmir", 12: "Arunachal Pradesh", 18: "Assam",
    28: "Andhra Pradesh", 29: "Karnataka",
}

_STOPWORDS = {
    "river", "nalla", "nala", "near", "bridge", "village", "taluka",
    "district", "at", "of", "the", "in", "d/s", "u/s", "ds", "us",
    "dam", "sump", "midc", "road", "temple", "pump", "house", "intake",
    "water", "bpt", "from",
}


def _normalise(s: str) -> str:
    """Lower-case, strip punctuation, remove stopwords."""
    if pd.isna(s) or not s:
        return ""
    s = str(s).lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    tokens = [t for t in s.split() if t and t not in _STOPWORDS]
    return " ".join(tokens)


def _token_overlap(a: str, b: str) -> float:
    """Jaccard-like token overlap ratio."""
    ta = set(a.split())
    tb = set(b.split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def generate_fuzzy_historical_context(
    predictions_path: str,
    baseline_path: str,
    hotspots_path: str,
    out_dir: str,
):
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "historical_context_for_hotspots_v2.csv")

    preds = pd.read_csv(predictions_path, low_memory=False)
    hotspots = pd.read_csv(hotspots_path) if os.path.exists(hotspots_path) else pd.DataFrame()

    if not os.path.exists(baseline_path):
        log.warning("No baseline file -- nothing to match.")
        pd.DataFrame(columns=[
            "station_name", "river_name", "state",
            "attempted_match_level", "matched_baseline_source",
            "matched_parameter", "current_value", "baseline_median",
            "baseline_iqr", "context_note", "match_confidence",
        ]).to_csv(out_path, index=False)
        return

    base = pd.read_csv(baseline_path, low_memory=False)

    # Map state codes to names in baseline
    if base["state"].dtype in (int, float, np.int64, np.float64):
        base["state_name_mapped"] = base["state"].map(_STATE_CODE_MAP).fillna("Unknown")
    else:
        base["state_name_mapped"] = base["state"]

    base["river_norm"] = base["river_name"].apply(_normalise)

    params_of_interest = [
        "dissolved_oxygen", "bod", "ph", "temperature",
        "conductivity", "total_dissolved_solids",
    ]

    stations = hotspots["station_name"].tolist() if not hotspots.empty else []
    if not stations:
        stations = preds["station_name"].dropna().unique()[:50].tolist()

    rows = []
    for stn in stations:
        p_rows = preds[preds["station_name"] == stn]
        if p_rows.empty:
            continue
        p = p_rows.iloc[0]

        # Resolve state -- NWMP uses state_name, not state
        state = p.get("state_name", p.get("state", ""))
        if pd.isna(state) or not state:
            state = ""
        state_lower = str(state).strip().lower()

        # Resolve river -- try name_of_water_body first
        river_raw = p.get("name_of_water_body", p.get("river_name", ""))
        if pd.isna(river_raw) or not river_raw:
            river_raw = ""
        river_norm = _normalise(river_raw)

        # Also normalise station name for possible river extraction
        stn_norm = _normalise(stn)

        # ── hierarchical matching ────────────────────────────────
        match_level = "no_reliable_match"
        b_match = pd.DataFrame()
        match_source = ""
        confidence = "None"

        # Level 1: river + state
        if river_norm:
            cands = base[base["river_norm"].apply(lambda r: _token_overlap(r, river_norm) > 0.4)]
            if not cands.empty and state_lower:
                state_cands = cands[
                    cands["state_name_mapped"].str.lower().str.contains(state_lower, na=False)
                ]
                if not state_cands.empty:
                    b_match = state_cands
                    match_level = "river_state"
                    match_source = f"river='{river_raw}' + state='{state}'"
                    confidence = "Medium"

        # Level 2: river only
        if b_match.empty and river_norm:
            cands = base[base["river_norm"].apply(lambda r: _token_overlap(r, river_norm) > 0.4)]
            if not cands.empty:
                b_match = cands
                match_level = "river_only"
                match_source = f"river='{river_raw}'"
                confidence = "Low"

        # Level 3: station name tokens -> river name
        if b_match.empty and stn_norm:
            cands = base[base["river_norm"].apply(lambda r: _token_overlap(r, stn_norm) > 0.25)]
            if not cands.empty:
                b_match = cands
                match_level = "station_to_river_fuzzy"
                match_source = f"station='{stn}' fuzzy-matched"
                confidence = "Low"

        # Level 4: state-level fallback
        if b_match.empty and state_lower:
            cands = base[
                base["state_name_mapped"].str.lower().str.contains(state_lower, na=False)
            ]
            if not cands.empty:
                b_match = cands
                match_level = "state_only"
                match_source = f"state='{state}'"
                confidence = "Very Low"

        if b_match.empty:
            rows.append({
                "station_name": stn,
                "river_name": river_raw,
                "state": state,
                "attempted_match_level": "no_reliable_match",
                "matched_baseline_source": "",
                "matched_parameter": "",
                "current_value": np.nan,
                "baseline_median": np.nan,
                "baseline_iqr": np.nan,
                "context_note": "No historical baseline match found for this station.",
                "match_confidence": "None",
            })
            continue

        # Generate context rows for each parameter of interest
        for param in params_of_interest:
            cur = p.get(param, np.nan)
            if pd.isna(cur):
                continue
            b_param = b_match[b_match["parameter"] == param]
            if b_param.empty:
                continue
            med = b_param["median"].mean()
            mn  = b_param["min"].mean()
            mx  = b_param["max"].mean()
            std = b_param["std"].mean()
            iqr = (mx - mn) if pd.notna(mx) and pd.notna(mn) else np.nan

            if pd.notna(med):
                direction = "above" if cur > med else "at/below"
                note = f"Current {param}={cur:.2f} is {direction} historical median={med:.2f}"
                if pd.notna(mx) and cur > mx:
                    note += f" (EXCEEDS historical max={mx:.2f})"
            else:
                note = f"Current {param}={cur:.2f}, insufficient baseline data"

            rows.append({
                "station_name": stn,
                "river_name": river_raw,
                "state": state,
                "attempted_match_level": match_level,
                "matched_baseline_source": match_source,
                "matched_parameter": param,
                "current_value": cur,
                "baseline_median": med,
                "baseline_iqr": iqr,
                "context_note": note,
                "match_confidence": confidence,
            })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(out_path, index=False)
    log.info(f"Fuzzy historical context: {len(df_out)} rows, "
             f"{df_out['match_confidence'].ne('None').sum()} matched.")
    return df_out
