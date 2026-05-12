    # Auxiliary-Only Model Interpretation

    ## Question Answered
    *Which non-core indicators are most useful for estimating non-compliance
    risk when DO, BOD, and pH are unavailable?*

    ## Method
    A HistGradientBoostingClassifier was trained on auxiliary features only
    (no DO, BOD, pH) using the operational NWMP dataset.  Importance was
    measured via permutation importance (10 repeats).

    ## Top 5 Drivers
    - **cod** (importance 0.0781)
- **fecal_coliform** (importance 0.0638)
- **conductivity** (importance 0.0596)
- **nitrate** (importance 0.0278)
- **total_coliform** (importance 0.0275)

    ## Interpretation
    These secondary indicators carry the strongest predictive signal for
    non-compliance when core regulatory parameters are absent.  They can
    guide early-warning systems or triage field-sampling priorities.

    > **Note:** This is an academic/predictive analysis.  Regulatory
    > compliance ultimately requires direct DO/BOD/pH measurement.
