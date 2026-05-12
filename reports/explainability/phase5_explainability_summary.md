# Phase 5: Explainability and SHAP Insights -- Summary Report

## 1. Objective
Generate transparent, human-readable explanations for the operational
water quality compliance prediction system.

## 2. Model Explained
**Extended Clean Model** (Phase 3.6) -- `DecisionTreeClassifier` inside
a scikit-learn `Pipeline` with median-imputation preprocessing.

Features used by the model:
`dissolved_oxygen, bod, ph, temperature, conductivity, nitrate,
fecal_coliform, total_coliform, fecal_streptococci, turbidity, cod,
total_dissolved_solids, season, river_name`

## 3. Explanation Method

| Step | Method | Outcome |
|------|--------|---------|
| SHAP TreeExplainer | Attempted | **Failed** -- several features were entirely NaN in the NWMP dataset |
| Permutation Importance | Fallback | **Succeeded** -- used for global feature importance |

> **Terminology note:** Because SHAP was not successfully applied,
> global importance outputs are labelled **"permutation importance"**
> rather than "SHAP values."

## 4. Global Permutation Importance Findings
Top 5 drivers: **bod, dissolved_oxygen, ph, temperature, conductivity**

**Expected interpretation:** BOD and DO dominate because the compliance
label is **rule-derived** from DO/BOD/pH thresholds. The model has
learned to automate these regulatory rules. This is *automated
regulatory compliance explanation*, not independent pollution discovery.

## 5. Auxiliary-Only Model Findings
Top 5 secondary drivers: **cod, fecal_coliform, conductivity, nitrate, total_coliform**

**Academic significance:** The auxiliary-only model is trained *without*
DO, BOD, or pH. Its importance ranking reveals which secondary
indicators carry the strongest predictive signal for non-compliance
when core regulatory parameters are unavailable.

## 6. Local & Hotspot Explanations
- Hotspot explanations generated: **20**
- Severe-alert explanations: **191 unique stations**
- Explainable alerts: **516**
- Disagreement cases explained: **4**

## 7. Historical Baseline Context
Matching between NWMP hotspot stations and the expanded historical
multi-state baseline was attempted. Due to naming convention
differences (NWMP uses long descriptive station names; baseline uses
short river names with numeric state codes), exact matching yielded
0 results. Fuzzy matching (Phase 5.1) was applied as a follow-up.

## 8. Limitations
1. **Rule-derived target**: Extended Clean Model importance is dominated by DO/BOD/pH.
2. **Permutation importance, not SHAP**: Explains model reliance, not per-sample effects.
3. **Auxiliary importance is associative**: Predictive association, not regulatory confirmation.
4. **Historical context is contextual**: Does not replace field/lab validation.
5. **Alerts are decision-support**: Not official regulatory declarations.

## 9. Recommendation
Phase 5 complete. Proceed to Phase 6 (Dashboard) after Phase 5.1 cleanup validation.
