"""
Phase 5 – Historical baseline context for hotspot stations.
Compares current NWMP values against the expanded multi-state historical baseline.
"""
import os, logging
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def generate_historical_context(
    predictions_path: str,
    baseline_path: str,
    hotspots_path: str,
    out_dir: str,
):
    os.makedirs(out_dir, exist_ok=True)

    preds = pd.read_csv(predictions_path, low_memory=False)
    hotspots = pd.read_csv(hotspots_path) if os.path.exists(hotspots_path) else pd.DataFrame()

    if not os.path.exists(baseline_path):
        log.warning("No seasonal baseline file found – skipping historical context.")
        return

    base = pd.read_csv(baseline_path, low_memory=False)
    if "parameter" not in base.columns:
        log.warning("Baseline file has no 'parameter' column – skipping.")
        return

    params = ["dissolved_oxygen", "bod", "ph", "temperature",
              "conductivity", "total_dissolved_solids"]

    out_rows = []

    stations = hotspots["station_name"].tolist() if not hotspots.empty else []
    if not stations:
        stations = preds["station_name"].dropna().unique()[:50].tolist()

    for stn in stations:
        p_rows = preds[preds["station_name"] == stn]
        if p_rows.empty:
            continue
        p = p_rows.iloc[0]
        river = p.get("river_name", "")
        state = p.get("state", p.get("state_name", ""))

        # hierarchical matching
        b_match = base[(base["state"] == state) & (base["river_name"] == river)] if river else pd.DataFrame()
        match_level = "river_state"
        if b_match.empty:
            b_match = base[base["river_name"] == river] if river else pd.DataFrame()
            match_level = "river_only"
        if b_match.empty:
            b_match = base[base["state"] == state] if state else pd.DataFrame()
            match_level = "state_only"
        if b_match.empty:
            continue

        ctx = []
        for param in params:
            cur = p.get(param, np.nan)
            if pd.isna(cur):
                continue
            b_param = b_match[b_match["parameter"] == param]
            if b_param.empty:
                continue
            med = b_param["median"].mean()
            mx  = b_param["max"].mean()
            if pd.notna(med):
                direction = "above" if cur > med else "at/below"
                ctx.append(f"{param} current={cur:.2f} vs historical median={med:.2f} ({direction})")
            if pd.notna(mx) and cur > mx:
                ctx.append(f"WARNING: {param} current={cur:.2f} exceeds historical max={mx:.2f}")

        if ctx:
            out_rows.append({
                "station_name": stn,
                "river_name":   river,
                "state":        state,
                "match_level":  match_level,
                "historical_context": "; ".join(ctx),
            })

    out_path = os.path.join(out_dir, "historical_context_for_hotspots.csv")
    if out_rows:
        pd.DataFrame(out_rows).to_csv(out_path, index=False)
        log.info(f"Historical context generated for {len(out_rows)} stations.")
    else:
        pd.DataFrame(columns=["station_name","river_name","state","match_level","historical_context"]).to_csv(out_path, index=False)
        log.info("No historical context matches found.")
