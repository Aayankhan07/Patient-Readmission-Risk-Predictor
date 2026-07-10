import os
import streamlit as st
import requests


def render_sidebar():
    """Renders the Streamlit sidebar with navigation, API status checks and configs."""
    st.sidebar.markdown(
        "<div style='font-family:Source Serif 4, serif; font-size:18px; "
        "font-weight:600; color:white; padding:8px 0 24px 0;'>"
        "Readmission<br/>Risk Predictor</div>",
        unsafe_allow_html=True,
    )

    # 1. Page Navigation
    page = st.sidebar.radio(
        "Navigation",
        options=[
            "Score Patient",
            "Bulk Upload",
            "Model Comparison",
            "Fairness & Drift",
            "About",
        ],
        label_visibility="collapsed",
    )

    st.sidebar.divider()

    # 2. API Configuration
    st.sidebar.subheader("Service Connection")
    default_api_url = os.environ.get("API_URL", "http://localhost:8000")
    api_url = st.sidebar.text_input("FastAPI Base URL", value=default_api_url)

    # 3. Connection Health Check
    try:
        response = requests.get(f"{api_url}/health", timeout=3)
        if response.status_code == 200:
            health_data = response.json()
            if health_data.get("status") == "ok":
                st.sidebar.success("API Connected")
            else:
                st.sidebar.warning("API Degrading (No Model Loaded)")
        else:
            st.sidebar.error("API Error Code")
    except Exception:
        st.sidebar.error("Prediction service unreachable.")

    st.sidebar.divider()

    # 4. Model Information
    st.sidebar.subheader("System Info")
    try:
        models_resp = requests.get(f"{api_url}/models", timeout=10)
        if models_resp.status_code == 200:
            models_data = models_resp.json()
            # Find the champion model in the models list
            champion = {}
            for m in models_data.get("models", []):
                if m.get("is_champion"):
                    champion = m
                    break
            if champion:
                st.sidebar.info(
                    f"**Champion Family:**\n"
                    f"{champion.get('name', 'unknown')}\n\n"
                    f"**Status:**\n"
                    f"`{champion.get('stage', 'n/a')}`"
                )
            else:
                st.sidebar.warning("No champion loaded.")
    except Exception:
        st.sidebar.caption("Could not load model info.")

    st.sidebar.divider()
    st.sidebar.caption("PRRP v1.0.0 © June 2026")

    return page, api_url
