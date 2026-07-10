import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import requests
import time

from dashboard.components.sidebar import render_sidebar
from dashboard.components.risk_card import render_risk_card, render_gauge_chart
from dashboard.components.shap_chart import (
    render_patient_shap_chart,
    render_global_importance_chart,
)
from dashboard.components.leaderboard import render_leaderboard
from dashboard.styles import CUSTOM_CSS

# Page configuration
st.set_page_config(
    page_title="PRRP - Patient Readmission Risk Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject custom clinical theme styling
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def render_header(title: str, eyebrow: str):
    """Renders the clinical header with uppercase eyebrow and serif title."""
    st.markdown(
        f'<div class="app-header"><div class="eyebrow">{eyebrow}</div>'
        f"<h1>{title}</h1></div>",
        unsafe_allow_html=True,
    )


# 1. Render Sidebar & Get navigation selection + API URL
page, api_url = render_sidebar()

# 2. Render Selected Page Content
if page == "Score Patient":
    render_header("Score Patient", "Single Patient Assessment")

    col_form, col_result = st.columns([3, 2], gap="large")

    with col_form:
        with st.form("patient_form"):
            st.markdown("### Patient Demographics & Clinical Profile")
            st.caption(
                "Fill in the fields below to calculate readmission risk probability."
            )

            c1, c2 = st.columns(2)
            with c1:
                age = st.selectbox(
                    "Age Bracket",
                    [
                        "[0-10)",
                        "[10-20)",
                        "[20-30)",
                        "[30-40)",
                        "[40-50)",
                        "[50-60)",
                        "[60-70)",
                        "[70-80)",
                        "[80-90)",
                        "[90-100)",
                    ],
                    index=6,  # default to [60-70)
                )
                time_in_hospital = st.slider(
                    "Time in Hospital (Days)", min_value=1, max_value=14, value=4
                )
                num_procedures = st.slider(
                    "Number of Procedures", min_value=0, max_value=6, value=2
                )
                num_medications = st.slider(
                    "Number of Medications", min_value=1, max_value=81, value=15
                )

            with c2:
                number_diagnoses = st.slider(
                    "Number of Diagnoses", min_value=1, max_value=16, value=7
                )
                A1Cresult = st.selectbox(
                    "A1C Result", ["None", "Normal", ">7", ">8"], index=0
                )
                insulin = st.selectbox(
                    "Insulin Status", ["No", "Steady", "Up", "Down"], index=0
                )
                diabetesMed = st.radio(
                    "Prescribed Diabetes Med?", ["No", "Yes"], index=1, horizontal=True
                )

            submitted = st.form_submit_button("Calculate Risk & Explain")

    with col_result:
        # Initialize session state for predictions
        if "prediction_result" not in st.session_state:
            st.session_state["prediction_result"] = None

        if submitted:
            patient_data = {
                "age": age,
                "time_in_hospital": int(time_in_hospital),
                "num_procedures": int(num_procedures),
                "num_medications": int(num_medications),
                "number_diagnoses": int(number_diagnoses),
                "A1Cresult": A1Cresult,
                "insulin": insulin,
                "diabetesMed": diabetesMed,
            }

            with st.spinner("Analyzing risk..."):
                try:
                    predict_resp = requests.post(
                        f"{api_url}/predict", json=patient_data, timeout=5.0
                    )

                    if predict_resp.status_code == 200:
                        pred_res = predict_resp.json()
                        st.session_state["prediction_result"] = {
                            "patient_id": pred_res.get("patient_id"),
                            "risk_score": pred_res.get("risk_score"),
                            "risk_tier": pred_res.get("risk_tier"),
                            "inference_ms": pred_res.get("inference_ms", 0.0),
                            "baseline_inputs": patient_data,
                        }
                        st.session_state["scored_patient_id"] = pred_res.get(
                            "patient_id"
                        )
                        st.session_state["show_explanations"] = True
                    else:
                        st.error(f"Prediction Service Error: {predict_resp.text}")
                        st.session_state["prediction_result"] = None
                        st.session_state["show_explanations"] = False
                except requests.exceptions.ConnectionError:
                    st.error(
                        "Can't reach the prediction service. Check that the API is running and try again."
                    )
                    st.session_state["prediction_result"] = None
                    st.session_state["show_explanations"] = False
                except Exception as e:
                    st.error(f"Failed to connect to API Service: {e}")
                    st.session_state["prediction_result"] = None
                    st.session_state["show_explanations"] = False

        # Render active prediction details if present in session state
        pred_res = st.session_state.get("prediction_result")
        if pred_res:
            patient_id = pred_res["patient_id"]
            risk_score = pred_res["risk_score"]
            risk_tier = pred_res["risk_tier"]
            inference_ms = pred_res["inference_ms"]
            baseline_inputs = pred_res["baseline_inputs"]

            st.markdown(
                '<div class="clinical-card" style="padding: 0px; border: none;">',
                unsafe_allow_html=True,
            )
            render_risk_card(risk_score, risk_tier, inference_ms)
            render_gauge_chart(risk_score, risk_tier)
            st.markdown("</div>", unsafe_allow_html=True)

            # ==================== "WHAT-IF" CLINICAL SANDBOX ====================
            st.write("")
            with st.expander("🛠️ Patient Risk 'What-If' Sandbox", expanded=True):
                st.caption(
                    "Perturb parameters below to simulate clinical risk changes in real-time."
                )

                c_sb1, c_sb2 = st.columns(2)
                with c_sb1:
                    sb_time = st.slider(
                        "Sandbox Time in Hospital",
                        1,
                        14,
                        int(baseline_inputs["time_in_hospital"]),
                        key=f"sb_time_{patient_id}",
                    )
                    sb_proc = st.slider(
                        "Sandbox Procedures",
                        0,
                        6,
                        int(baseline_inputs["num_procedures"]),
                        key=f"sb_proc_{patient_id}",
                    )
                    sb_meds = st.slider(
                        "Sandbox Medications",
                        1,
                        81,
                        int(baseline_inputs["num_medications"]),
                        key=f"sb_meds_{patient_id}",
                    )
                with c_sb2:
                    sb_diag = st.slider(
                        "Sandbox Diagnoses",
                        1,
                        16,
                        int(baseline_inputs["number_diagnoses"]),
                        key=f"sb_diag_{patient_id}",
                    )
                    sb_a1c = st.selectbox(
                        "Sandbox A1C Result",
                        ["None", "Normal", ">7", ">8"],
                        index=["None", "Normal", ">7", ">8"].index(
                            baseline_inputs["A1Cresult"]
                        ),
                        key=f"sb_a1c_{patient_id}",
                    )
                    sb_ins = st.selectbox(
                        "Sandbox Insulin",
                        ["No", "Steady", "Up", "Down"],
                        index=["No", "Steady", "Up", "Down"].index(
                            baseline_inputs["insulin"]
                        ),
                        key=f"sb_ins_{patient_id}",
                    )

                has_changed = (
                    sb_time != baseline_inputs["time_in_hospital"]
                    or sb_proc != baseline_inputs["num_procedures"]
                    or sb_meds != baseline_inputs["num_medications"]
                    or sb_diag != baseline_inputs["number_diagnoses"]
                    or sb_a1c != baseline_inputs["A1Cresult"]
                    or sb_ins != baseline_inputs["insulin"]
                )

                if has_changed:
                    sandbox_patient = baseline_inputs.copy()
                    sandbox_patient["time_in_hospital"] = int(sb_time)
                    sandbox_patient["num_procedures"] = int(sb_proc)
                    sandbox_patient["num_medications"] = int(sb_meds)
                    sandbox_patient["number_diagnoses"] = int(sb_diag)
                    sandbox_patient["A1Cresult"] = sb_a1c
                    sandbox_patient["insulin"] = sb_ins

                    try:
                        sb_resp = requests.post(
                            f"{api_url}/predict", json=sandbox_patient, timeout=3.0
                        )
                        if sb_resp.status_code == 200:
                            sb_score = sb_resp.json().get("risk_score", 0.0)
                            delta = sb_score - risk_score
                            delta_pct = delta * 100.0

                            st.write("")
                            if delta_pct < -0.1:
                                st.success(
                                    f"Sandbox Risk: **{sb_score * 100.0:.1f}%** (⬇️ **{abs(delta_pct):.1f}%** Risk Reduction!)"
                                )
                            elif delta_pct > 0.1:
                                st.warning(
                                    f"Sandbox Risk: **{sb_score * 100.0:.1f}%** (⬆️ **{abs(delta_pct):.1f}%** Risk Increase)"
                                )
                            else:
                                st.info(
                                    f"Sandbox Risk: **{sb_score * 100.0:.1f}%** (No significant change)"
                                )
                    except Exception as e:
                        st.error(f"Sandbox prediction offline: {e}")
                else:
                    st.info("Adjust the sliders/inputs above to run a risk simulation.")
        else:
            st.markdown(
                '<div class="clinical-card" style="text-align:center; color:#6B6B6E; padding:48px 24px;">'
                "Fill in the patient profile and select <b>Calculate Risk & Explain</b> to see the assessment."
                "</div>",
                unsafe_allow_html=True,
            )

    # 3. Explanation Section
    if st.session_state.get("show_explanations") and st.session_state.get(
        "scored_patient_id"
    ):
        st.divider()
        st.markdown("### Explainable AI Diagnostics")
        patient_id = st.session_state["scored_patient_id"]

        with st.spinner(
            "Generating SHAP and LIME diagnostics (polling background worker...)..."
        ):
            try:
                exp_res = None
                for _ in range(15):  # poll for up to 7.5 seconds
                    explain_resp = requests.get(
                        f"{api_url}/explain/{patient_id}", timeout=5.0
                    )
                    if explain_resp.status_code == 200:
                        exp_res = explain_resp.json()
                        break
                    elif explain_resp.status_code == 202:
                        time.sleep(0.5)
                    else:
                        break

                if exp_res:
                    shap_vals = exp_res.get("shap_values", {})
                    top_factors = exp_res.get("top_risk_factors", [])
                    lime_html = exp_res.get("lime_html", "")

                    exp_col1, exp_col2 = st.columns(2, gap="large")

                    with exp_col1:
                        render_patient_shap_chart(shap_vals)

                        st.info(
                            f"**Top Risk Drivers for this Patient:**\n"
                            f"1. `{top_factors[0] if len(top_factors) > 0 else 'n/a'}`\n"
                            f"2. `{top_factors[1] if len(top_factors) > 1 else 'n/a'}`\n"
                            f"3. `{top_factors[2] if len(top_factors) > 2 else 'n/a'}`"
                        )

                    with exp_col2:
                        st.markdown("#### Patient LIME Explanation")
                        components.html(lime_html, height=450, scrolling=True)
                else:
                    st.warning(
                        "Live explanation unavailable. Showing global feature importance as fallback."
                    )
                    render_global_importance_chart()
            except requests.exceptions.Timeout:
                st.warning(
                    "Live explanation unavailable (timeout). Showing global feature importance as fallback."
                )
                render_global_importance_chart()
            except Exception:
                st.warning(
                    "Live explanation unavailable. Showing global feature importance as fallback."
                )
                render_global_importance_chart()

elif page == "Bulk Upload":
    render_header("Bulk Upload", "Batch Patient Assessment")

    st.markdown('<div class="clinical-card">', unsafe_allow_html=True)
    st.caption(
        "Upload a CSV file containing multiple patient clinical profiles to evaluate all readmission risk scores simultaneously."
    )

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        df_uploaded = pd.read_csv(uploaded_file)
        st.markdown("#### Preview Uploaded Data")
        st.dataframe(df_uploaded.head(5), use_container_width=True)

        # Verify required columns and values
        req_cols = [
            "age",
            "time_in_hospital",
            "num_procedures",
            "num_medications",
            "number_diagnoses",
            "A1Cresult",
            "insulin",
            "diabetesMed",
        ]
        missing_cols = [c for c in req_cols if c not in df_uploaded.columns]

        if missing_cols:
            st.error(
                f"Invalid dataset columns. Missing required columns: {', '.join(missing_cols)}"
            )
        else:
            # Check row-level missing values
            invalid_rows = []
            missing_fields_set = set()
            for idx, row in df_uploaded.iterrows():
                row_missing = [
                    c for c in req_cols if pd.isna(row[c]) or str(row[c]).strip() == ""
                ]
                if row_missing:
                    invalid_rows.append(idx + 1)
                    missing_fields_set.update(row_missing)

            if invalid_rows:
                st.error(
                    f"{len(invalid_rows)} rows are missing required fields: {', '.join(sorted(missing_fields_set))}. Fix and re-upload."
                )
            else:
                if st.button("Run Bulk Risk Score Analysis", type="primary"):
                    scores = []
                    tiers = []
                    bar = st.progress(0)

                    rows_count = len(df_uploaded)
                    for idx, row in df_uploaded.iterrows():
                        patient_json = row[req_cols].to_dict()

                        # Convert sliders to integers
                        for numeric_col in [
                            "time_in_hospital",
                            "num_procedures",
                            "num_medications",
                            "number_diagnoses",
                        ]:
                            patient_json[numeric_col] = int(patient_json[numeric_col])

                        try:
                            resp = requests.post(
                                f"{api_url}/predict", json=patient_json
                            )
                            if resp.status_code == 200:
                                res = resp.json()
                                scores.append(res.get("risk_score"))
                                tiers.append(res.get("risk_tier"))
                            else:
                                scores.append(None)
                                tiers.append("error")
                        except Exception:
                            scores.append(None)
                            tiers.append("offline")

                        bar.progress((idx + 1) / rows_count)

                    df_result = df_uploaded.copy()
                    df_result["Risk Score"] = scores
                    df_result["Risk Tier"] = tiers

                    st.success(f"Successfully processed {rows_count} patients.")
                    st.dataframe(df_result, use_container_width=True)

                    # Download button
                    csv_data = df_result.to_csv(index=False)
                    st.download_button(
                        label="Download Scored Results CSV",
                        data=csv_data,
                        file_name="scored_patient_readmissions.csv",
                        mime="text/csv",
                    )

                    # ==================== COHORT ANALYTICS ====================
                    import plotly.express as px

                    st.write("---")
                    st.subheader("📊 Batch Cohort Analytics")

                    c_ana1, c_ana2 = st.columns(2)

                    with c_ana1:
                        # Pie chart of risk tiers
                        tier_counts = (
                            df_result["Risk Tier"].value_counts().reset_index()
                        )
                        tier_counts.columns = ["Risk Tier", "Count"]

                        # Capitalize names for display
                        tier_counts["Risk Tier"] = tier_counts[
                            "Risk Tier"
                        ].str.capitalize()

                        # Map clinical theme colors
                        color_map = {
                            "Low": "#2A9D8F",
                            "Medium": "#D4A24E",
                            "High": "#E76F51",
                            "Error": "#6B6B6E",
                            "Offline": "#6B6B6E",
                        }

                        fig_pie = px.pie(
                            tier_counts,
                            values="Count",
                            names="Risk Tier",
                            title="Patient Risk Tier Distribution",
                            color="Risk Tier",
                            color_discrete_map=color_map,
                            hole=0.4,
                        )
                        fig_pie.update_layout(
                            showlegend=True, margin=dict(t=40, b=0, l=0, r=0)
                        )
                        st.plotly_chart(fig_pie, use_container_width=True)

                    with c_ana2:
                        # Histogram of risk probabilities
                        df_result_valid = df_result[
                            df_result["Risk Score"].notna()
                        ].copy()
                        df_result_valid["Risk Probability (%)"] = (
                            df_result_valid["Risk Score"] * 100.0
                        )

                        fig_hist = px.histogram(
                            df_result_valid,
                            x="Risk Probability (%)",
                            nbins=20,
                            title="Distribution of Patient Risk Scores",
                            color_discrete_sequence=["#0F2238"],
                        )
                        fig_hist.update_layout(
                            xaxis_title="Predicted Risk Score (%)",
                            yaxis_title="Patient Count",
                            margin=dict(t=40, b=40, l=40, r=20),
                        )
                        st.plotly_chart(fig_hist, use_container_width=True)

                    # High-Risk Cohort Alert
                    high_risk_patients = df_result[df_result["Risk Tier"] == "high"]
                    if not high_risk_patients.empty:
                        st.warning(
                            f"⚠️ **High-Risk Action Needed**: Identified {len(high_risk_patients)} patients with readmission risk > 65%. "
                            f"Please coordinate priority transitional care interventions for these patient records."
                        )
                        with st.expander("Review High-Risk Patient Indices"):
                            st.write(high_risk_patients.index.tolist())
        st.markdown("</div>", unsafe_allow_html=True)

elif page == "Model Comparison":
    render_header("Model Comparison", "Leaderboard & Methodology")

    try:
        resp = requests.get(f"{api_url}/models", timeout=3.0)
        if resp.status_code == 200:
            models_data = resp.json().get("models", [])
        else:
            models_data = []
    except Exception:
        models_data = []

    render_leaderboard(models_data)

elif page == "Fairness & Drift":
    render_header("Fairness & Drift Monitoring", "Healthcare Compliance & Audits")

    import plotly.express as px
    import json
    import os

    tab_fairness, tab_drift = st.tabs(
        ["🏥 Subgroup Fairness Analysis", "📊 Data & Prediction Drift"]
    )

    with tab_fairness:
        st.markdown(
            '<div class="clinical-card">'
            "<h3>Fairness & Subgroup Diagnostics</h3>"
            "<p>Clinical systems demand equity across protected demographics. Below is a metrics breakdown "
            "evaluating accuracy, discrimination risk (ROC-AUC), and selection rates across Race and Gender subgroups, "
            "along with sample size (n) statistics to evaluate credibility.</p>"
            "</div>",
            unsafe_allow_html=True,
        )

        fairness_path = "data/fairness_metrics.json"
        if not os.path.exists(fairness_path):
            st.warning(
                "⚠️ **Fairness metrics file not found.** Please run the fairness analysis script inside the API container first:"
            )
            st.code(
                "docker compose exec api python scripts/analyze_fairness.py",
                language="bash",
            )
        else:
            with open(fairness_path, "r") as f:
                metrics_data = json.load(f)

            # --- RACE ANALYSIS ---
            st.subheader("Subgroup Slices by Race")
            race_df = pd.DataFrame(metrics_data["race"])

            # Map Demographic Parity Ratio from JSON
            parity_race = metrics_data["demographic_parity_ratio_race"]
            race_df["Demographic Parity Ratio"] = race_df["name"].map(parity_race)

            # Subgroup sizes warning if skew is present
            skew_warning = any(race_df["sample_size"] < 100)
            if skew_warning:
                st.info(
                    "💡 **Statistical Context**: Subgroups with small sample sizes (e.g. n < 100) are "
                    "subject to high variance. Review performance metrics carefully alongside sample counts."
                )

            # Sliced metrics side by side
            col1, col2 = st.columns(2)
            with col1:
                fig_acc = px.bar(
                    race_df,
                    x="name",
                    y="roc_auc",
                    color="name",
                    labels={"name": "Race Group", "roc_auc": "ROC-AUC Score"},
                    title="Model Discrimination (ROC-AUC) by Race",
                    color_discrete_sequence=[
                        "#2A9D8F",
                        "#E76F51",
                        "#5C6B73",
                        "#F4A261",
                    ],
                )
                fig_acc.update_layout(showlegend=False)
                st.plotly_chart(fig_acc, use_container_width=True)

            with col2:
                fig_sr = px.bar(
                    race_df,
                    x="name",
                    y="selection_rate",
                    color="name",
                    labels={
                        "name": "Race Group",
                        "selection_rate": "Selection Rate (High Risk %)",
                    },
                    title="Selection Rate by Race (Target Positive Prediction Rate)",
                    color_discrete_sequence=[
                        "#2A9D8F",
                        "#E76F51",
                        "#5C6B73",
                        "#F4A261",
                    ],
                )
                fig_sr.update_layout(showlegend=False)
                st.plotly_chart(fig_sr, use_container_width=True)

            # Table Breakdown
            st.caption("Detailed Race Metrics Breakdown:")
            st.table(
                race_df.rename(
                    columns={
                        "name": "Race Subgroup",
                        "sample_size": "Sample Count (n)",
                        "accuracy": "Accuracy",
                        "roc_auc": "ROC-AUC",
                        "selection_rate": "Selection Rate",
                        "Demographic Parity Ratio": "Demographic Parity (vs Caucasian)",
                    }
                ).set_index("Race Subgroup")
            )

            # --- GENDER ANALYSIS ---
            st.subheader("Subgroup Slices by Gender")
            gender_df = pd.DataFrame(metrics_data["gender"])

            # Map Demographic Parity Ratio from JSON
            parity_gender = metrics_data["demographic_parity_ratio_gender"].get(
                "Female_vs_Male", 1.0
            )
            gender_df["Demographic Parity Ratio"] = [
                parity_gender if name == "Female" else 1.0 for name in gender_df["name"]
            ]

            col3, col4 = st.columns(2)
            with col3:
                fig_gender_auc = px.bar(
                    gender_df,
                    x="name",
                    y="roc_auc",
                    color="name",
                    labels={"name": "Gender", "roc_auc": "ROC-AUC Score"},
                    title="Model Discrimination (ROC-AUC) by Gender",
                    color_discrete_sequence=["#2A9D8F", "#E76F51"],
                )
                fig_gender_auc.update_layout(showlegend=False)
                st.plotly_chart(fig_gender_auc, use_container_width=True)

            with col4:
                fig_gender_sr = px.bar(
                    gender_df,
                    x="name",
                    y="selection_rate",
                    color="name",
                    labels={"name": "Gender", "selection_rate": "Selection Rate"},
                    title="Selection Rate by Gender",
                    color_discrete_sequence=["#2A9D8F", "#E76F51"],
                )
                fig_gender_sr.update_layout(showlegend=False)
                st.plotly_chart(fig_gender_sr, use_container_width=True)

            # Table Breakdown
            st.caption("Detailed Gender Metrics Breakdown:")
            st.table(
                gender_df.rename(
                    columns={
                        "name": "Gender",
                        "sample_size": "Sample Count (n)",
                        "accuracy": "Accuracy",
                        "roc_auc": "ROC-AUC",
                        "selection_rate": "Selection Rate",
                        "Demographic Parity Ratio": "Demographic Parity Ratio (Female vs Male)",
                    }
                ).set_index("Gender")
            )

    with tab_drift:
        st.markdown(
            '<div class="clinical-card">'
            "<h3>Data & Target Distribution Drift Diagnostics</h3>"
            "<p>Post-deployment monitoring flags changes in data features (covariate drift) "
            "or target labels (prediction drift) over time compared to baseline training distributions.</p>"
            "</div>",
            unsafe_allow_html=True,
        )

        drift_report_path = "dashboard/drift_report.html"
        if not os.path.exists(drift_report_path):
            st.warning(
                "⚠️ **Evidently AI Drift Report not found.** Please run the drift analysis script inside the API container first:"
            )
            st.code(
                "docker compose exec api python scripts/detect_drift.py",
                language="bash",
            )
        else:
            # Load and display Evidently report in iframe
            with open(drift_report_path, "r", encoding="utf-8") as f:
                drift_html = f.read()
            components.html(drift_html, height=900, scrolling=True)

elif page == "About":
    render_header("About", "Clinical Methodology")

    st.markdown(
        '<div class="clinical-card">'
        "<h3>Methodology</h3>"
        "<p>This tool predicts the probability of a patient being readmitted to the "
        "hospital within 30 days of discharge. It compares a classical machine "
        "learning suite (Logistic Regression, SVM, Random Forest, XGBoost, Stacking Ensemble) "
        "against a deep learning model (Keras ANN), selecting the best performer "
        "by PR-AUC as the production champion model.</p>"
        "<p>Predictions are explained per-patient using SHAP and LIME. "
        "No patient data is stored — each prediction is computed and "
        "discarded statelessly.</p>"
        "<p style='color:#6B6B6E; font-size:13px; margin-top:24px;'>Built as a decision-support assistant. "
        "Not validated for clinical use without institutional population calibration.</p>"
        "</div>",
        unsafe_allow_html=True,
    )
