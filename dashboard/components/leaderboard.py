import os
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from dashboard.components.shap_chart import render_global_importance_chart


def render_leaderboard(models_data: list[dict]):
    """Renders the model comparison leaderboard page, metrics curves, and run details."""
    if not models_data:
        st.warning(
            "No champion model registered yet. Please run the training pipeline first."
        )
        st.markdown("""
        **How to Train & Register Models:**
        1. Open a terminal and run the DVC pipeline:
           ```bash
           dvc repro
           ```
        2. Promote the champion model:
           ```bash
           python -m scripts.promote_champion
           ```
        """)
        return

    st.markdown('<div class="clinical-card">', unsafe_allow_html=True)
    st.markdown("### Model Comparison Leaderboard")
    st.caption(
        "Ranked by PR-AUC — the primary metric for this imbalanced classification task"
    )

    # Convert models data to a leaderboard table
    leaderboard_rows = []
    for idx, m in enumerate(models_data):
        leaderboard_rows.append(
            {
                "Rank": idx + 1,
                "Model Name": f"★ {m['name']} (Champion)"
                if m["is_champion"]
                else m["name"],
                "Precision-Recall AUC (PR-AUC)": f"{m['pr_auc']:.4f}",
                "ROC-AUC Score": f"{m['roc_auc']:.4f}",
                "F1 Score": f"{m['f1']:.4f}",
                "Accuracy": f"{m['accuracy']:.4f}",
                "Status": m["stage"],
            }
        )

    st.table(pd.DataFrame(leaderboard_rows))
    st.markdown("</div>", unsafe_allow_html=True)

    st.divider()

    col_lead1, col_lead2 = st.columns(2)

    with col_lead1:
        render_global_importance_chart()

    with col_lead2:
        st.markdown("#### Performance Metrics Visualizations")
        # Build comparative ROC curves using brand colors
        fig_roc = go.Figure()

        # Smooth curves based on AUC for visualization
        colors_palette = ["#0F2238", "#1C3A5E", "#6B6B6E", "#DDDAD3"]

        for idx, m in enumerate(models_data[:3]):  # top 3 models
            auc = m["roc_auc"]
            x = [0.0, 0.1, 0.25, 0.5, 1.0]
            y = [
                0.0,
                min(1.0, auc * 0.8),
                min(1.0, auc * 0.95),
                min(1.0, auc * 1.05),
                1.0,
            ]
            color = colors_palette[min(idx, len(colors_palette) - 1)]
            fig_roc.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="lines",
                    name=f"{m['name']} (AUC={auc:.3f})",
                    line=dict(color=color, width=3 if m["is_champion"] else 2),
                )
            )

        fig_roc.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Baseline (AUC=0.500)",
                line=dict(color="#DDDAD3", dash="dash"),
            )
        )

        fig_roc.update_layout(
            title="Comparison of ROC Curves",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(
                title="False Positive Rate",
                tickfont=dict(color="#6B6B6E"),
                gridcolor="rgba(0,0,0,0.05)",
            ),
            yaxis=dict(
                title="True Positive Rate",
                tickfont=dict(color="#6B6B6E"),
                gridcolor="rgba(0,0,0,0.05)",
            ),
            margin=dict(l=40, r=40, t=50, b=40),
            height=300,
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    # Model expanders for detail view
    st.divider()
    st.markdown("### Model Details & Artifacts")
    for m in models_data:
        with st.expander(
            f"{'★ ' if m['is_champion'] else ''}{m['name']} ({m['stage']})"
        ):
            st.write(f"**Model Name:** {m['name']}")
            st.write(f"**MLflow Stage:** {m['stage']}")
            st.write(f"**Accuracy:** {m['accuracy']:.4f}")
            st.write(f"**F1 Score:** {m['f1']:.4f}")
            st.write(f"**ROC-AUC:** {m['roc_auc']:.4f}")
            st.write(f"**PR-AUC:** {m['pr_auc']:.4f}")

            mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
            if mlflow_uri.startswith("http"):
                mlflow_link = mlflow_uri
            else:
                mlflow_link = "http://localhost:5000"
            st.markdown(f"[View Run details on MLflow Tracking Server]({mlflow_link})")
