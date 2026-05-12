# Phase 5.1: Explainability Cleanup Summary

## Final explanation method
**Permutation Importance** (SHAP TreeExplainer was attempted but failed
due to all-NaN auxiliary features in the NWMP operational dataset).

## Reason SHAP fallback was needed
The NWMP prediction data has `season`, `river_name`, and most auxiliary
features as NaN. The model pipeline's median imputer skips all-NaN
columns, producing a dimensionality mismatch that SHAP cannot handle.

## Top 5 global drivers
1. bod
2. dissolved_oxygen
3. ph
4. temperature
5. conductivity

## Top 5 auxiliary-only drivers
1. cod
2. fecal_coliform
3. conductivity
4. nitrate
5. total_coliform

## Historical context matching
- Fuzzy matching attempted: Yes
- Rows with matched context: 0
- Rows with no reliable match: 20

## Dashboard-ready files created
- `dashboard_global_drivers.csv`
- `dashboard_auxiliary_drivers.csv`
- `dashboard_hotspot_explanations.csv`
- `dashboard_explainable_alerts.csv`

## Readiness
**Ready for Phase 6 dashboard.**
