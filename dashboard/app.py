import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import requests

from dashboard.components.sidebar import render_sidebar
from dashboard.components.risk_card import render_risk_card, render_gauge_chart
from dashboard.components.shap_chart import render_patient_shap_chart, render_global_importance_chart
from dashboard.components.leaderboard import render_leaderboard
from dashboard.styles import CUSTOM_CSS

# Page configuration
st.set_page_config(
    page_title="PRRP - Patient Readmission Risk Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
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
        st.markdown('<div class="clinical-card">', unsafe_allow_html=True)
        st.markdown("### Patient Demographics & Clinical Profile")
        st.caption("Fill in the fields below to calculate readmission risk probability.")
        
        with st.form("patient_form"):
            c1, c2 = st.columns(2)
            with c1:
                age = st.selectbox(
                    "Age Bracket", 
                    ["[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)", "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"],
                    index=6  # default to [60-70)
                )
                time_in_hospital = st.slider("Time in Hospital (Days)", min_value=1, max_value=14, value=4)
                num_procedures = st.slider("Number of Procedures", min_value=0, max_value=6, value=2)
                num_medications = st.slider("Number of Medications", min_value=1, max_value=81, value=15)
                
            with c2:
                number_diagnoses = st.slider("Number of Diagnoses", min_value=1, max_value=16, value=7)
                A1Cresult = st.selectbox(
                    "A1C Result", 
                    ["None", "Normal", ">7", ">8"],
                    index=0
                )
                insulin = st.selectbox(
                    "Insulin Status",
                    ["No", "Steady", "Up", "Down"],
                    index=0
                )
                diabetesMed = st.radio(
                    "Prescribed Diabetes Med?",
                    ["No", "Yes"],
                    index=1,
                    horizontal=True
                )
                
            submitted = st.form_submit_button("Calculate Risk & Explain")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_result:
        if submitted:
            patient_data = {
                "age": age,
                "time_in_hospital": int(time_in_hospital),
                "num_procedures": int(num_procedures),
                "num_medications": int(num_medications),
                "number_diagnoses": int(number_diagnoses),
                "A1Cresult": A1Cresult,
                "insulin": insulin,
                "diabetesMed": diabetesMed
            }
            
            with st.spinner("Analyzing risk..."):
                try:
                    predict_resp = requests.post(f"{api_url}/predict", json=patient_data, timeout=5.0)
                    
                    if predict_resp.status_code == 200:
                        pred_res = predict_resp.json()
                        patient_id = pred_res.get("patient_id")
                        risk_score = pred_res.get("risk_score")
                        risk_tier = pred_res.get("risk_tier")
                        inference_ms = pred_res.get("inference_ms", 0.0)
                        
                        st.markdown('<div class="clinical-card" style="padding: 0px; border: none;">', unsafe_allow_html=True)
                        render_risk_card(risk_score, risk_tier, inference_ms)
                        render_gauge_chart(risk_score, risk_tier)
                        st.markdown("</div>", unsafe_allow_html=True)
                        
                        # Store prediction variables in session state for explanation block below
                        st.session_state["scored_patient_id"] = patient_id
                        st.session_state["show_explanations"] = True
                    else:
                        st.error(f"Prediction Service Error: {predict_resp.text}")
                        st.session_state["show_explanations"] = False
                except requests.exceptions.ConnectionError:
                    st.error("Can't reach the prediction service. Check that the API is running and try again.")
                    st.session_state["show_explanations"] = False
                except Exception as e:
                    st.error(f"Failed to connect to API Service: {e}")
                    st.session_state["show_explanations"] = False
        else:
            st.markdown(
                '<div class="clinical-card" style="text-align:center; color:#6B6B6E; padding:48px 24px;">'
                "Fill in the patient profile and select <b>Calculate Risk & Explain</b> to see the assessment."
                "</div>",
                unsafe_allow_html=True,
            )

    # 3. Explanation Section
    if st.session_state.get("show_explanations") and st.session_state.get("scored_patient_id"):
        st.divider()
        st.markdown("### Explainable AI Diagnostics")
        patient_id = st.session_state["scored_patient_id"]
        
        with st.spinner("Generating SHAP and LIME diagnostics..."):
            try:
                explain_resp = requests.get(f"{api_url}/explain/{patient_id}", timeout=5.0)
                if explain_resp.status_code == 200:
                    exp_res = explain_resp.json()
                    shap_vals = exp_res.get("shap_values", {})
                    top_factors = exp_res.get("top_risk_factors", [])
                    lime_html = exp_res.get("lime_html", "")
                    
                    exp_col1, exp_col2 = st.columns(2, gap="large")
                    
                    with exp_col1:
                        render_patient_shap_chart(shap_vals)
                        
                        # Top risk factor callouts
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
                    st.warning("Live explanation unavailable. Showing global feature importance as fallback.")
                    render_global_importance_chart()
            except requests.exceptions.Timeout:
                st.warning("Live explanation unavailable (timeout). Showing global feature importance as fallback.")
                render_global_importance_chart()
            except Exception:
                st.warning("Live explanation unavailable. Showing global feature importance as fallback.")
                render_global_importance_chart()

elif page == "Bulk Upload":
    render_header("Bulk Upload", "Batch Patient Assessment")
    
    st.markdown('<div class="clinical-card">', unsafe_allow_html=True)
    st.caption("Upload a CSV file containing multiple patient clinical profiles to evaluate all readmission risk scores simultaneously.")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    
    if uploaded_file is not None:
        df_uploaded = pd.read_csv(uploaded_file)
        st.markdown("#### Preview Uploaded Data")
        st.dataframe(df_uploaded.head(5), use_container_width=True)
        
        # Verify required columns and values
        req_cols = ["age", "time_in_hospital", "num_procedures", "num_medications", "number_diagnoses", "A1Cresult", "insulin", "diabetesMed"]
        missing_cols = [c for c in req_cols if c not in df_uploaded.columns]
        
        if missing_cols:
            st.error(f"Invalid dataset columns. Missing required columns: {', '.join(missing_cols)}")
        else:
            # Check row-level missing values
            invalid_rows = []
            missing_fields_set = set()
            for idx, row in df_uploaded.iterrows():
                row_missing = [c for c in req_cols if pd.isna(row[c]) or str(row[c]).strip() == ""]
                if row_missing:
                    invalid_rows.append(idx + 1)
                    missing_fields_set.update(row_missing)
            
            if invalid_rows:
                st.error(f"{len(invalid_rows)} rows are missing required fields: {', '.join(sorted(missing_fields_set))}. Fix and re-upload.")
            else:
                if st.button("Run Bulk Risk Score Analysis", type="primary"):
                    scores = []
                    tiers = []
                    bar = st.progress(0)
                    
                    rows_count = len(df_uploaded)
                    for idx, row in df_uploaded.iterrows():
                        patient_json = row[req_cols].to_dict()
                        
                        # Convert sliders to integers
                        for numeric_col in ["time_in_hospital", "num_procedures", "num_medications", "number_diagnoses"]:
                            patient_json[numeric_col] = int(patient_json[numeric_col])
                            
                        try:
                            resp = requests.post(f"{api_url}/predict", json=patient_json)
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
                        mime="text/csv"
                    )
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

elif page == "About":
    render_header("About", "Clinical Methodology")
    
    st.markdown(
        '<div class="clinical-card">'
        "<h3>Methodology</h3>"
        "<p>This tool predicts the probability of a patient being readmitted to the "
        "hospital within 30 days of discharge. It compares a classical machine "
        "learning suite (Logistic Regression, SVM, Random Forest, XGBoost) "
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
