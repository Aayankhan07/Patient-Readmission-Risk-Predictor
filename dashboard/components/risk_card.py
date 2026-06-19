import streamlit as st
import plotly.graph_objects as go

def get_tier_color(tier: str) -> str:
    if tier == "low":
        return "#00E676"  # Emerald Green
    elif tier == "medium":
        return "#FFD600"  # Amber Orange
    else:
        return "#FF1744"  # Coral Red

def render_risk_card(score: float, tier: str, inference_ms: float):
    """Renders a beautiful premium CSS risk card representing the patient readmission risk."""
    color = get_tier_color(tier)
    percent = score * 100
    
    card_html = f"""
    <div style="
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-left: 6px solid {color};
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
        margin-bottom: 25px;
        transition: transform 0.2s ease-in-out;
    " onmouseover="this.style.transform='scale(1.02)'" onmouseout="this.style.transform='scale(1.0)'">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h4 style="color: #A0AAB2; margin: 0; font-size: 14px; text-transform: uppercase; letter-spacing: 1px;">Patient Risk Score</h4>
                <h2 style="color: #FFFFFF; margin: 8px 0; font-size: 36px; font-weight: 700;">{percent:.1f}%</h2>
            </div>
            <div style="
                background: {color}22;
                border: 1px solid {color};
                border-radius: 20px;
                padding: 6px 16px;
                color: {color};
                font-weight: 700;
                text-transform: uppercase;
                font-size: 14px;
            ">
                {tier} Risk
            </div>
        </div>
        <div style="display: flex; gap: 20px; margin-top: 15px; border-top: 1px solid rgba(255,255,255,0.05); padding-top: 15px;">
            <div>
                <span style="color: #A0AAB2; font-size: 12px;">Inference Latency:</span>
                <strong style="color: #FFFFFF; font-size: 12px; margin-left: 5px;">{inference_ms:.1f} ms</strong>
            </div>
            <div>
                <span style="color: #A0AAB2; font-size: 12px;">Decision Threshold:</span>
                <strong style="color: #FFFFFF; font-size: 12px; margin-left: 5px;">35.0% / 65.0%</strong>
            </div>
        </div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)

def render_gauge_chart(score: float, tier: str):
    """Generates an interactive Plotly radial gauge chart."""
    color = get_tier_color(tier)
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        number={'suffix': "%", 'font': {'color': '#FFFFFF', 'size': 40}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#A0AAB2"},
            'bar': {'color': color},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 2,
            'bordercolor': "rgba(255,255,255,0.1)",
            'steps': [
                {'range': [0, 35], 'color': 'rgba(0, 230, 118, 0.15)'},
                {'range': [35, 65], 'color': 'rgba(255, 214, 0, 0.15)'},
                {'range': [65, 100], 'color': 'rgba(255, 23, 68, 0.15)'}
            ],
            'threshold': {
                'line': {'color': "#FFFFFF", 'width': 3},
                'thickness': 0.75,
                'value': score * 100
            }
        }
    ))

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': "#FFFFFF", 'family': "Outfit, Inter, sans-serif"},
        height=220,
        margin=dict(l=20, r=20, t=30, b=20)
    )
    
    st.plotly_chart(fig, use_container_width=True)
