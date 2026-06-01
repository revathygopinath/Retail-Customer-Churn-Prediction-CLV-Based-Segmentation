"""
dashboard/components/sidebar.py — Sidebar navigation and pipeline status.
"""

import streamlit as st
from pathlib import Path


PAGES = {
    "Home":                   "home",
    "Executive Overview":     "executive_overview",
    "Customer Segmentation":  "customer_segmentation",
    "Predictive Intelligence":"predictive_intelligence",
    "Retention Strategy":     "retention_strategy",
}

PAGE_ICONS = {
    "Home":                   "house",
    "Executive Overview":     "bar-chart-line",
    "Customer Segmentation":  "people",
    "Predictive Intelligence":"cpu",
    "Retention Strategy":     "bullseye",
}


def render_sidebar(outputs_ready: bool) -> str:
    """
    Renders the left-hand navigation sidebar.
    Returns the selected page name.
    """
    with st.sidebar:
        st.markdown("### Retail Churn Analytics")
        st.markdown("---")

        if not outputs_ready:
            st.warning(
                "Pipeline outputs not found.\n\n"
                "Run the pipeline first:\n\n"
                "```\npython pipelines/run_pipeline.py\n```"
            )
            st.markdown("---")

        selected = st.radio(
            "Navigation",
            list(PAGES.keys()),
            label_visibility="collapsed",
        )

        st.markdown("---")

        # Pipeline status indicators
        st.markdown("**Output Status**")
        outputs_path = Path("outputs")
        checks = {
            "Predictions":        outputs_path / "predictions" / "customer_predictions.csv",
            "Model Metrics":      outputs_path / "metrics" / "model_metrics.json",
            "KPIs":               outputs_path / "metrics" / "kpis.json",
            "SHAP":               outputs_path / "plots" / "shap_summary.png",
            "Business Dashboard": outputs_path / "plots" / "business_dashboard.png",
        }
        for label, path in checks.items():
            icon = "✓" if path.exists() else "○"
            color = "green" if path.exists() else "#aaa"
            st.markdown(
                f"<span style='color:{color};font-size:0.82rem'>{icon} {label}</span>",
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.markdown(
            "<div style='font-size:0.75rem;color:#94a3b8'>"
            "UCI Online Retail Dataset<br>"
            "Dec 2010 – Dec 2011<br>"
            "Currency: ₹ (£1 = ₹107)"
            "</div>",
            unsafe_allow_html=True,
        )

    return selected
