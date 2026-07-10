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

    # Colors: Amber-Red (#E76F51) for positive impact, Teal (#2A9D8F) for negative impact
    colors = ["#E76F51" if val >= 0 else "#2A9D8F" for val in values]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            y=features,
            x=values,
            orientation="h",
            marker_color=colors,
            text=[f"{val:+.3f}" for val in values],
            textposition="outside",
            textfont=dict(color="#1C1C1E", size=11, family="IBM Plex Sans, sans-serif"),
            hovertemplate="<b>Feature:</b> %{y}<br><b>SHAP Value:</b> %{x:+.4f}<extra></extra>",
        )
    )

    fig.update_layout(
        title={
            "text": "Top Feature Contributions (SHAP Values)",
            "y": 0.95,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
            "font": {"size": 16, "color": "#0F2238", "family": "Source Serif 4, serif"},
        },
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            title=dict(
                text="Impact on Prediction Risk",
                font=dict(color="#6B6B6E", family="IBM Plex Sans, sans-serif"),
            ),
            tickfont=dict(color="#6B6B6E", family="IBM Plex Sans, sans-serif"),
            gridcolor="rgba(0,0,0,0.05)",
            zerolinecolor="rgba(0,0,0,0.2)",
        ),
        yaxis=dict(
            tickfont=dict(color="#1C1C1E", size=11, family="IBM Plex Sans, sans-serif"),
            gridcolor="rgba(0,0,0,0)",
        ),
        margin=dict(l=150, r=40, t=50, b=40),
        height=350,
        showlegend=False,
    )

    st.plotly_chart(fig, use_container_width=True)


def render_global_importance_chart():
    """Renders a fallback global feature importance chart based on average metrics."""
    # Top features typical for Kaggle hospital readmission dataset
    importance_data = {
        "Feature": [
            "number_inpatient",
            "discharge_disposition_id",
            "time_in_hospital",
            "num_medications",
            "number_diagnoses",
            "num_lab_procedures",
            "age",
            "insulin_Steady",
            "number_emergency",
            "diabetesMed_Yes",
        ],
        "Importance": [0.24, 0.18, 0.13, 0.11, 0.09, 0.08, 0.06, 0.05, 0.04, 0.02],
    }
    df = pd.DataFrame(importance_data)

    fig = go.Figure(
        go.Bar(
            x=df["Importance"],
            y=df["Feature"],
            orientation="h",
            marker=dict(color="#0F2238"),
            hovertemplate="<b>Feature:</b> %{y}<br><b>Relative Importance:</b> %{x:.2f}<extra></extra>",
        )
    )

    # Reverse y axis to have the most important feature at the top
    fig.update_yaxes(autorange="reversed")

    fig.update_layout(
        title={
            "text": "Global Feature Importance Summary",
            "y": 0.95,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
            "font": {"size": 16, "color": "#0F2238", "family": "Source Serif 4, serif"},
        },
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            title=dict(
                text="Relative Importance Score",
                font=dict(color="#6B6B6E", family="IBM Plex Sans, sans-serif"),
            ),
            tickfont=dict(color="#6B6B6E", family="IBM Plex Sans, sans-serif"),
            gridcolor="rgba(0,0,0,0.05)",
        ),
        yaxis=dict(
            tickfont=dict(color="#1C1C1E", family="IBM Plex Sans, sans-serif"),
            gridcolor="rgba(0,0,0,0)",
        ),
        margin=dict(l=150, r=40, t=50, b=40),
        height=350,
        showlegend=False,
    )

    st.plotly_chart(fig, use_container_width=True)
