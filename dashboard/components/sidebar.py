import streamlit as st
import requests

def render_sidebar():
    """Renders the Streamlit sidebar with API status checks and configs."""
    st.sidebar.image(
        "https://cdn-icons-png.flaticon.com/512/3004/3004458.png", 
        width=80
    )
    st.sidebar.title("PRRP Control Panel")
    st.sidebar.markdown(
        "*Patient Readmission Risk Predictor*"
    )
    st.sidebar.divider()
    
    # 1. API Configuration
    st.sidebar.subheader("Service Connection")
    api_url = st.sidebar.text_input("FastAPI Base URL", value="http://localhost:8000")
    
    # 2. Connection Health Check
    try:
        response = requests.get(f"{api_url}/health", timeout=3)
        if response.status_code == 200:
            health_data = response.json()
            if health_data.get("status") == "ok":
                st.sidebar.success("API Connected")
                st.sidebar.caption(
                    f"Model: {health_data.get('version')} (Loaded)"
                )
            else:
                st.sidebar.warning("API Degrading (No Model Loaded)")
        else:
            st.sidebar.error("API Error Code")
    except Exception:
        st.sidebar.error("Prediction service unreachable.")
        
    st.sidebar.divider()
    
    # 3. Model Information
    st.sidebar.subheader("System Info")
    try:
        models_resp = requests.get(f"{api_url}/models", timeout=3)
        if models_resp.status_code == 200:
            models_data = models_resp.json()
            champion = models_data.get("champion_model", {})
            st.sidebar.info(
                f"**Champion Family:**\n"
                f"{champion.get('type', 'unknown')}\n\n"
                f"**MLflow Run:**\n"
                f"`{champion.get('run_id', 'n/a')[:12]}...`"
            )
    except Exception:
        st.sidebar.caption("Could not load model info.")
        
    st.sidebar.divider()
    st.sidebar.caption("PRRP v1.0.0 © June 2026")
    
    return api_url
