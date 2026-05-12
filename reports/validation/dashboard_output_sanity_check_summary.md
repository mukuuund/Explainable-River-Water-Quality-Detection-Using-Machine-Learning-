# Dashboard Output Sanity Check

| Category | Check | Status | Details |
|----------|-------|--------|---------|
| Overview Metrics | total_nwmp_rows is not N/A | PASS | Value: 666.0 |
| Overview Metrics | stations_monitored is not N/A | PASS | Value: 223.0 |
| Overview Metrics | non_compliance_recall is not N/A | PASS | Value: 0.0 |
| Overview Metrics | leakage_safe_f1 is not N/A | PASS | Value: 0.0 |
| Overview Metrics | persistent_hotspots is not N/A | PASS | Value: 134.0 |
| Overview Metrics | total_alerts is not N/A | PASS | Value: 516.0 |
| Overview Metrics | high_severe_alerts is not N/A | PASS | Value: 495.0 |
| Overview Metrics | expanded_baseline_rows is not N/A | PASS | Value: 49238.0 |
| Operational | File exists | PASS |  |
| Operational | At least 600 rows | PASS | Rows: 666 |
| Operational | Column station_name exists | PASS |  |
| Operational | Column river_name exists | PASS |  |
| Operational | Column water_body_type exists | PASS |  |
| Operational | Column ml_predicted_compliance_label exists | PASS |  |
| Operational | Probability column exists | PASS |  |
| Operational | Generic river_name rate <= 6% | PASS | Rate: 5.86% |
| Healthcheck | Model healthcheck files exist | PASS |  |
| SHAP | Success files exist OR failure report | PASS | Success files found |
| LIME | live_lime_explanations.csv exists | PASS |  |
| LIME | At least one lime_status = success | PASS |  |
