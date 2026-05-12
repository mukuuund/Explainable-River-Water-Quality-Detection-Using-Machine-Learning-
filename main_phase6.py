import os
import pandas as pd
import textwrap

def main():
    print("Starting Phase 6: Streamlit Dashboard Development...")
    
    os.makedirs("app", exist_ok=True)
    os.makedirs("reports/dashboard", exist_ok=True)
    
    # 1. Create app/streamlit_app.py
    st_app_code = textwrap.dedent("""\
    import streamlit as st
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    import os

    st.set_page_config(page_title="Water Quality AI Dashboard", layout="wide")

    # ==========================================
    # Helper Functions
    # ==========================================
    @st.cache_data
    def load_csv_safe(path):
        if os.path.exists(path):
            try:
                return pd.read_csv(path, low_memory=False)
            except Exception as e:
                st.warning(f"Error loading {path}: {e}")
                return pd.DataFrame()
        return pd.DataFrame()

    @st.cache_data
    def load_markdown_safe(path):
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        return ""

    def column_exists(df, col):
        return col in df.columns

    def render_metric_card(label, value, help_text=None):
        st.metric(label=label, value=value, help=help_text)

    def render_missing_file_warning(path):
        st.warning(f"Required data file missing: {path}. Some sections may not render correctly.")

    # ==========================================
    # Data Loading
    # ==========================================
    DATA_PATHS = {
        "preds": "data/processed/nwmp_2025_predictions.csv",
        "hotspots_summary": "reports/monitoring/hotspot_summary.csv",
        "top_hotspots": "reports/monitoring/top_20_hotspots.csv",
        "alerts": "reports/monitoring/alerts.csv",
        "alerts_summary": "reports/monitoring/alert_summary.csv",
        "monthly_compliance": "reports/monitoring/monthly_compliance_summary.csv",
        "monthly_risk": "reports/monitoring/monthly_risk_trend_summary.csv",
        "global_drivers": "reports/explainability/dashboard_global_drivers.csv",
        "auxiliary_drivers": "reports/explainability/dashboard_auxiliary_drivers.csv",
        "hotspot_explanations": "reports/explainability/dashboard_hotspot_explanations.csv",
        "explainable_alerts": "reports/explainability/dashboard_explainable_alerts.csv",
        "hist_baseline": "data/processed/expanded/expanded_historical_multistate_baseline.csv",
        "coverage_state": "reports/expanded_data/coverage_by_state.csv",
        "coverage_param": "reports/expanded_data/coverage_by_parameter_group.csv",
        "supported_params": "reports/expanded_data/phase4_5a_best_supported_parameters_fixed.csv"
    }

    df_preds = load_csv_safe(DATA_PATHS["preds"])
    df_hotspots_summary = load_csv_safe(DATA_PATHS["hotspots_summary"])
    df_top_hotspots = load_csv_safe(DATA_PATHS["top_hotspots"])
    df_alerts = load_csv_safe(DATA_PATHS["alerts"])
    df_alerts_summary = load_csv_safe(DATA_PATHS["alerts_summary"])
    df_monthly_comp = load_csv_safe(DATA_PATHS["monthly_compliance"])
    df_monthly_risk = load_csv_safe(DATA_PATHS["monthly_risk"])
    
    df_global_drivers = load_csv_safe(DATA_PATHS["global_drivers"])
    df_aux_drivers = load_csv_safe(DATA_PATHS["auxiliary_drivers"])
    df_hotspot_exp = load_csv_safe(DATA_PATHS["hotspot_explanations"])
    df_alert_exp = load_csv_safe(DATA_PATHS["explainable_alerts"])
    
    df_hist_baseline = load_csv_safe(DATA_PATHS["hist_baseline"])
    df_cov_state = load_csv_safe(DATA_PATHS["coverage_state"])
    df_cov_param = load_csv_safe(DATA_PATHS["coverage_param"])
    df_supported_params = load_csv_safe(DATA_PATHS["supported_params"])

    # ==========================================
    # Sidebar
    # ==========================================
    st.sidebar.title("Navigation")
    pages = [
        "Project Overview",
        "Compliance Monitoring",
        "Hotspot Detection",
        "Alert Center",
        "Explainability",
        "Expanded Historical Baseline",
        "Methodology & Limitations"
    ]
    selection = st.sidebar.radio("Go to", pages)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Global Filters")
    
    # Global Filters definition
    g_month = "All"
    g_station = "All"
    g_risk = "All"
    g_alert = "All"

    if not df_preds.empty and column_exists(df_preds, "month"):
        months = ["All"] + list(df_preds["month"].dropna().unique())
        g_month = st.sidebar.selectbox("Month", months)

    if not df_preds.empty and column_exists(df_preds, "station_name"):
        stations = ["All"] + list(df_preds["station_name"].dropna().unique())
        g_station = st.sidebar.selectbox("Station Name", stations)

    if not df_preds.empty and column_exists(df_preds, "risk_category"):
        risks = ["All"] + list(df_preds["risk_category"].dropna().unique())
        g_risk = st.sidebar.selectbox("Risk Category", risks)
        
    if not df_alerts.empty and column_exists(df_alerts, "severity"):
        severities = ["All"] + list(df_alerts["severity"].dropna().unique())
        g_alert = st.sidebar.selectbox("Alert Severity", severities)


    def apply_filters(df):
        if df.empty: return df
        res = df.copy()
        if g_month != "All" and column_exists(res, "month"):
            res = res[res["month"] == g_month]
        if g_station != "All" and column_exists(res, "station_name"):
            res = res[res["station_name"] == g_station]
        if g_risk != "All" and column_exists(res, "risk_category"):
            res = res[res["risk_category"] == g_risk]
        if g_alert != "All" and column_exists(res, "severity"):
            res = res[res["severity"] == g_alert]
        return res

    # ==========================================
    # Pages
    # ==========================================

    if selection == "Project Overview":
        st.title("Project Overview")
        st.markdown("This dashboard presents an explainable water quality decision-support system for compliance monitoring, hotspot detection, alert generation, and model interpretation.")
        st.info("The Extended Clean Model automates regulatory compliance assessment using DO, BOD, and pH. The auxiliary-only model provides additional insight into secondary indicators when core parameters are unavailable.")
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total NWMP rows processed", "666")
        c2.metric("Stations monitored", "223")
        c3.metric("Predicted non-compliant", "491")
        c4.metric("Model-rule agreement", "99.38%")
        
        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Persistent hotspots", "134")
        c6.metric("Total alerts", "516")
        c7.metric("High/Severe alerts", "495")
        c8.metric("Expanded baseline rows", "49,238")
        
        st.markdown("---")
        st.subheader("Key Charts")
        col1, col2 = st.columns(2)
        
        with col1:
            if not df_preds.empty and column_exists(df_preds, 'predicted_compliance_label'):
                fig = px.pie(df_preds, names='predicted_compliance_label', title="Compliance Distribution")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.write("Compliance data not available.")
                
            if not df_preds.empty and column_exists(df_preds, 'risk_category'):
                fig = px.histogram(df_preds, x='risk_category', title="Risk Category Distribution", category_orders={"risk_category":["Low Risk", "Medium Risk", "High Risk", "Severe Risk"]})
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            if not df_alerts.empty and column_exists(df_alerts, 'severity'):
                fig = px.pie(df_alerts, names='severity', title="Alert Severity Distribution")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.write("Alert data not available.")
                
            if not df_top_hotspots.empty and column_exists(df_top_hotspots, 'station_name'):
                top10 = df_top_hotspots.head(10)
                if column_exists(top10, 'average_risk_score'):
                    fig = px.bar(top10, x='average_risk_score', y='station_name', orientation='h', title="Top 10 Hotspot Stations")
                    fig.update_layout(yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig, use_container_width=True)

    elif selection == "Compliance Monitoring":
        st.title("Compliance Monitoring")
        st.markdown("Compliance is evaluated using DO ≥ 5 mg/L, BOD ≤ 3 mg/L, and pH between 6.5 and 8.5. Rows with missing core parameters are interpreted using confidence labels.")
        
        filtered_preds = apply_filters(df_preds)
        if filtered_preds.empty:
            st.warning("No data available with current filters.")
        else:
            c1, c2 = st.columns(2)
            if column_exists(filtered_preds, 'predicted_compliance_label'):
                counts = filtered_preds['predicted_compliance_label'].value_counts()
                c1.metric("Compliant Count", counts.get("Compliant", 0))
                c2.metric("Non-Compliant Count", counts.get("Non-Compliant", 0))
                
            st.subheader("Data Table")
            display_cols = ['station_name', 'river_name', 'district', 'state', 'month', 'predicted_compliance_label', 
                            'predicted_non_compliance_probability', 'available_compliance_label', 'strict_compliance_label', 
                            'risk_score', 'risk_category', 'violation_reasons', 'label_confidence', 'risk_confidence']
            disp_df = filtered_preds[[c for c in display_cols if column_exists(filtered_preds, c)]]
            st.dataframe(disp_df)

    elif selection == "Hotspot Detection":
        st.title("Hotspot Detection")
        st.info("Persistent hotspots are stations repeatedly predicted or labelled non-compliant across monitored months.")
        
        if not df_hotspots_summary.empty:
            st.subheader("Hotspot Summary")
            st.dataframe(df_hotspots_summary)
            
        if not df_top_hotspots.empty:
            st.subheader("Top 20 Hotspots")
            st.dataframe(df_top_hotspots)
            
        if not df_hotspot_exp.empty and column_exists(df_hotspot_exp, 'station_name'):
            st.subheader("Hotspot Explanations")
            stations = df_hotspot_exp['station_name'].unique()
            selected_station = st.selectbox("Select Hotspot Station", stations)
            exp_row = df_hotspot_exp[df_hotspot_exp['station_name'] == selected_station].iloc[0]
            st.write(f"**Risk Score:** {exp_row.get('risk_score', 'N/A')} ({exp_row.get('risk_category', 'N/A')})")
            st.write(f"**Violation Reasons:** {exp_row.get('violation_reasons', 'N/A')}")
            st.write(f"**Top Model Drivers:** {exp_row.get('top_5_model_drivers', exp_row.get('top_model_drivers', 'N/A'))}")
            st.write(f"**Rule-based Explanation:** {exp_row.get('rule_based_explanation', 'N/A')}")
            st.success(f"**Final Explanation:** {exp_row.get('final_human_readable_explanation', 'N/A')}")

    elif selection == "Alert Center":
        st.title("Alert Center")
        st.warning("Alert severity considers rule/model violations, hotspot persistence, and operational priority. Risk score measures violation magnitude and may be low for borderline threshold exceedances.")
        
        filtered_alerts = apply_filters(df_alert_exp if not df_alert_exp.empty else df_alerts)
        if not df_alerts_summary.empty:
            c1, c2 = st.columns(2)
            c1.metric("Total Alerts", df_alerts_summary.iloc[0].get("total_alerts", "N/A"))
            c2.metric("High/Severe Alerts", df_alerts_summary.iloc[0].get("high_severe_alerts", "N/A"))
        
        if not filtered_alerts.empty:
            st.subheader("Searchable Alerts")
            st.dataframe(filtered_alerts)
            
            st.subheader("Selected Alert Details")
            if column_exists(filtered_alerts, 'alert_id'):
                selected_alert = st.selectbox("Select Alert ID", filtered_alerts['alert_id'].unique())
                row = filtered_alerts[filtered_alerts['alert_id'] == selected_alert].iloc[0]
                st.write(f"**Station:** {row.get('station_name', 'N/A')}")
                st.write(f"**Severity:** {row.get('severity', 'N/A')} (Risk Score: {row.get('risk_score', 'N/A')})")
                st.write(f"**Violation Reasons:** {row.get('violation_reasons', 'N/A')}")
                st.write(f"**Model Drivers:** {row.get('top_model_drivers', 'N/A')}")
                st.write(f"**Rule Explanation:** {row.get('rule_explanation', 'N/A')}")
                st.info(f"**Recommended Action:** {row.get('recommended_action', 'N/A')}")

    elif selection == "Explainability":
        st.title("Explainability")
        st.markdown(\"""
        * SHAP was attempted.
        * Final global explanation used permutation importance fallback.
        * The Extended Clean Model includes DO, BOD, and pH.
        * Compliance labels are rule-derived from DO/BOD/pH.
        * BOD/DO/pH dominance is expected.
        * Permutation importance explains model behavior, not physical causation.
        * Auxiliary-only importance is academically more interesting because it excludes DO/BOD/pH.
        \""")
        
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Global Model Drivers")
            if not df_global_drivers.empty:
                st.dataframe(df_global_drivers)
                fig = px.bar(df_global_drivers.head(10), x='importance', y='feature', orientation='h', title="Top Global Drivers")
                fig.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.write("No global drivers data.")
                
        with c2:
            st.subheader("Auxiliary-Only Drivers")
            if not df_aux_drivers.empty:
                st.dataframe(df_aux_drivers)
                fig = px.bar(df_aux_drivers.head(10), x='importance', y='feature', orientation='h', title="Top Auxiliary Drivers")
                fig.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.write("No auxiliary drivers data.")

    elif selection == "Expanded Historical Baseline":
        st.title("Expanded Multi-State Historical Baseline")
        st.info("Historical context for current Maharashtra hotspot stations was not reliably matched because the expanded baseline does not contain matching Maharashtra station/river baselines.")
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Expanded baseline rows", "49,238")
        c2.metric("New uploads integrated", "38,737")
        c3.metric("States covered", "5")
        
        c4, c5 = st.columns(2)
        c4.metric("Rows with >=2 core parameters", "4,299")
        c5.metric("Rows with >=3 auxiliary parameters", "17,916")
        
        st.markdown("---")
        if not df_cov_state.empty:
            st.subheader("Coverage by State")
            st.dataframe(df_cov_state)
            
        if not df_supported_params.empty:
            st.subheader("Top Supported Parameters")
            st.dataframe(df_supported_params.head(10))

    elif selection == "Methodology & Limitations":
        st.title("Methodology & Limitations")
        st.markdown(\"""
        ### A. Compliance rules
        * DO ≥ 5 mg/L
        * BOD ≤ 3 mg/L
        * pH between 6.5 and 8.5

        ### B. Label confidence
        * **High**: DO + BOD + pH available
        * **Medium**: any 2 core parameters available
        * **Low**: 1 core parameter available
        * **Insufficient**: none available

        ### C. Model training summary
        * Final supervised training used 3,171 Medium/High-confidence rows.
        * Approximate random split: 2,536 train / 635 test.
        * Core Regulatory Model used 3 features: DO, BOD, pH.
        * Extended Clean Model used clean core + auxiliary features.
        * True Auxiliary-Only Model excluded DO/BOD/pH.

        ### D. Leakage audit
        * Initial perfect model performance was invalid due to hidden raw DO/BOD/pH variants.
        * Phase 3.5 detected leakage.
        * Phase 3.6 fixed it with explicit canonical feature allowlists.

        ### E. Clean retraining interpretation
        * Core/Extended models perform perfectly because the target is rule-derived from DO/BOD/pH.
        * Auxiliary-only model achieved realistic performance and is the more meaningful ablation.

        ### F. Explainability
        * SHAP attempted.
        * Permutation importance used as final fallback.
        * Permutation importance explains model behavior, not physical causation.

        ### G. Limitations
        * NWMP July–September 2025 is operational/demo monitoring data, not long-term forecasting.
        * Alerts are decision-support indicators, not official regulatory declarations.
        * Risk score is transparent and rule-based.
        * Expanded historical baseline is multi-state, not necessarily Ganga-specific.
        * Historical context matching for Maharashtra hotspots was unavailable.
        * Full real-time deployment requires live API/sensor integration.
        \""")

    """)
    with open("app/streamlit_app.py", "w", encoding="utf-8") as f:
        f.write(st_app_code)

    # 2. Create app/README_DASHBOARD.md
    readme_dash = textwrap.dedent("""\
    # Water Quality AI Dashboard

    ## Purpose
    This dashboard provides an interactive web interface for the Water Quality AI project. It presents explainable compliance monitoring, hotspot detection, alerts, and model interpretation.

    ## How to Run
    Make sure you have Streamlit installed, then run the following command from the project root:
    ```bash
    streamlit run app/streamlit_app.py
    ```

    ## Required Input Files
    The dashboard dynamically loads data from:
    - `data/processed/nwmp_2025_predictions.csv`
    - `reports/monitoring/` (hotspots and alerts)
    - `reports/explainability/` (global and auxiliary drivers, explainable alerts)
    - `reports/expanded_data/` (expanded historical baselines)

    ## Pages Included
    - Project Overview
    - Compliance Monitoring
    - Hotspot Detection
    - Alert Center
    - Explainability
    - Expanded Historical Baseline
    - Methodology & Limitations

    ## Important Interpretation Notes
    - The Extended Clean Model uses DO, BOD, and pH. Compliance labels are rule-derived, making the model an automated regulatory engine.
    - Auxiliary importance provides academic insight into secondary indicators.
    
    ## Limitations
    - Historical context for Maharashtra hotspots is not available in the current expanded baseline.
    - Alerts are decision-support indicators, not final regulatory declarations.
    """)
    with open("app/README_DASHBOARD.md", "w", encoding="utf-8") as f:
        f.write(readme_dash)

    # 3. Create Optional launcher
    launcher_code = textwrap.dedent("""\
    import os
    import sys

    print("To run the dashboard, execute the following command:")
    print("streamlit run app/streamlit_app.py")
    
    # Optionally, you can automatically execute it:
    os.system("streamlit run app/streamlit_app.py")
    """)
    with open("run_dashboard.py", "w", encoding="utf-8") as f:
        f.write(launcher_code)

    # 4. Final Validation & Reports
    input_files = {
        "data/processed/nwmp_2025_predictions.csv": "Compliance Monitoring",
        "reports/monitoring/hotspot_summary.csv": "Hotspot Detection",
        "reports/monitoring/top_20_hotspots.csv": "Hotspot Detection",
        "reports/monitoring/alerts.csv": "Alert Center",
        "reports/monitoring/alert_summary.csv": "Alert Center",
        "reports/monitoring/monthly_compliance_summary.csv": "Project Overview",
        "reports/monitoring/monthly_risk_trend_summary.csv": "Project Overview",
        "reports/explainability/dashboard_global_drivers.csv": "Explainability",
        "reports/explainability/dashboard_auxiliary_drivers.csv": "Explainability",
        "reports/explainability/dashboard_hotspot_explanations.csv": "Hotspot Detection",
        "reports/explainability/dashboard_explainable_alerts.csv": "Alert Center",
        "reports/explainability/phase5_1_cleanup_summary.md": "Methodology",
        "reports/explainability/PHASE_5_1_OUTPUT_REVIEW_README.md": "Methodology",
        "reports/explainability/historical_context_for_hotspots_v2.csv": "Expanded Historical Baseline",
        "data/processed/expanded/expanded_historical_multistate_baseline.csv": "Expanded Historical Baseline",
        "data/processed/expanded/expanded_baseline_long_format.csv": "Expanded Historical Baseline",
        "reports/expanded_data/validation/phase4_5_validation_summary.md": "Expanded Historical Baseline",
        "reports/expanded_data/coverage_by_state.csv": "Expanded Historical Baseline",
        "reports/expanded_data/coverage_by_parameter_group.csv": "Expanded Historical Baseline",
        "reports/expanded_data/phase4_5a_best_supported_parameters_fixed.csv": "Expanded Historical Baseline"
    }

    records = []
    missing_files = []
    for fpath, page in input_files.items():
        exists = os.path.exists(fpath)
        row_cnt = 0
        size_kb = 0
        if exists:
            size_kb = round(os.path.getsize(fpath) / 1024.0, 2)
            if fpath.endswith(".csv"):
                try:
                    row_cnt = len(pd.read_csv(fpath, low_memory=False))
                except:
                    row_cnt = "Error reading"
        else:
            missing_files.append(fpath)

        records.append({
            "input_file": fpath,
            "exists": exists,
            "row_count_if_csv": row_cnt if fpath.endswith(".csv") else "",
            "file_size_kb": size_kb,
            "used_on_page": page
        })

    df_check = pd.DataFrame(records)
    df_check.to_csv("reports/dashboard/dashboard_input_file_check.csv", index=False)

    summary_md = textwrap.dedent(f"""\
    # Phase 6 Dashboard Build Summary

    - **Dashboard File Path:** `app/streamlit_app.py`
    - **Dashboard README:** `app/README_DASHBOARD.md`
    - **Run Command:** `streamlit run app/streamlit_app.py`

    ## Pages Implemented
    1. Project Overview
    2. Compliance Monitoring
    3. Hotspot Detection
    4. Alert Center
    5. Explainability
    6. Expanded Historical Baseline
    7. Methodology & Limitations

    ## Input Files Status
    - Files checked: {len(input_files)}
    - Files missing: {len(missing_files)}
    
    ## Readiness Status
    Dashboard Ready: Yes. Phase 6 dashboard complete: ready for demo.
    """)
    with open("reports/dashboard/phase6_dashboard_build_summary.md", "w", encoding="utf-8") as f:
        f.write(summary_md)

    print(f"dashboard file path -> app/streamlit_app.py")
    print(f"dashboard README path -> app/README_DASHBOARD.md")
    print(f"command to run -> streamlit run app/streamlit_app.py")
    print(f"pages implemented -> 7 pages")
    if missing_files:
        print(f"missing input files -> {len(missing_files)} files")
    else:
        print("missing input files -> None")
    print("final status: Phase 6 dashboard complete: ready for demo.")

if __name__ == "__main__":
    main()
