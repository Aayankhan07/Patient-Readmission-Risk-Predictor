import streamlit as st
import plotly.graph_objects as go
import pandas as pd

def render_patient_shap_chart(shap_values: dict, base_value: float = 0.5):
    """
    Renders an interactive Plotly horizontal bar chart representing 
    feature contributions (SHAP values) to the risk score.
    """
    # Sort features by absolute contribution
    sorted_shaps = sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)
    
    # Take top 10 features for display clarity
    top_shaps = sorted_shaps[:10]
    
    features = [item[0] for item in top_shaps]
    values = [item[1] for item in top_shaps]
    
    # Reverse so largest impact is at the top of the horizontal bar chart
    features.reverse()
    values.reverse()
    
    # Colors: Coral Red for positive impact, Emerald Green for negative impact
    colors = ['#FF1744' if val >= 0 else '#00E676' for val in values]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=features,
        x=values,
        orientation='h',
        marker_color=colors,
        text=[f"{val:+.3f}" for val in values],
        textposition='outside',
        textfont=dict(color='#FFFFFF', size=11),
        hovertemplate="<b>Feature:</b> %{y}<br><b>SHAP Value:</b> %{x:+.4f}<extra></extra>",
    ))
    
    fig.update_layout(
        title={
            'text': "Top Feature Contributions (SHAP Values)",
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 16, 'color': '#FFFFFF'}
        },
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            title="Impact on Prediction Risk",
            titlefont=dict(color="#A0AAB2"),
            tickfont=dict(color="#A0AAB2"),
            gridcolor="rgba(255,255,255,0.05)",
            zerolinecolor="rgba(255,255,255,0.2)"
        ),
        yaxis=dict(
            tickfont=dict(color="#FFFFFF", size=11),
            gridcolor="rgba(0,0,0,0)"
        ),
        margin=dict(l=150, r=40, t=50, b=40),
        height=350,
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_global_importance_chart():
    """Renders a fallback global feature importance chart based on average metrics."""
    # Top features typical for Kaggle hospital readmission dataset
    importance_data = {
        "Feature": ["number_inpatient", "discharge_disposition_id", "time_in_hospital", 
                    "num_medications", "number_diagnoses", "num_lab_procedures", 
                    "age", "insulin_Steady", "number_emergency", "diabetesMed_Yes"],
        "Importance": [0.24, 0.18, 0.13, 0.11, 0.09, 0.08, 0.06, 0.05, 0.04, 0.02]
    }
    df = pd.DataFrame(importance_data)
    
    fig = go.Figure(go.Bar(
        x=df["Importance"],
        y=df["Feature"],
        orientation='h',
        marker=dict(
            color=df["Importance"],
            colorscale=[[0, '#4FACFE'], [1, '#00F2FE']]
        ),
        hovertemplate="<b>Feature:</b> %{y}<br><b>Relative Importance:</b> %{x:.2f}<extra></extra>"
    ))
    
    # Reverse y axis to have the most important feature at the top
    fig.update_yaxes(autorange="reversed")
    
    fig.update_layout(
        title={
            'text': "Global Feature Importance Summary",
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 16, 'color': '#FFFFFF'}
        },
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            title="Relative Importance Score",
            titlefont=dict(color="#A0AAB2"),
            tickfont=dict(color="#A0AAB2"),
            gridcolor="rgba(255,255,255,0.05)"
        ),
        yaxis=dict(
            tickfont=dict(color="#FFFFFF"),
            gridcolor="rgba(0,0,0,0)"
        ),
        margin=dict(l=150, r=40, t=50, b=40),
        height=350,
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
