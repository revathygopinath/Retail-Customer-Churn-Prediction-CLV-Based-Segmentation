"""
dashboard/components/kpi_cards.py — Reusable KPI card renderer.
"""

import streamlit as st


def render_kpi_card(label: str, value: str, delta: str = "", delta_color: str = "normal"):
    """Render a single metric using st.metric (styled via custom.css)."""
    st.metric(label=label, value=value, delta=delta if delta else None,
              delta_color=delta_color)


def render_kpi_row(kpis: dict):
    """
    Render the top KPI row using st.columns.
    kpis: {label: (value_str, delta_str, delta_color)}
    """
    cols = st.columns(len(kpis))
    for col, (label, (value, delta, delta_color)) in zip(cols, kpis.items()):
        with col:
            st.metric(
                label=label,
                value=value,
                delta=delta if delta else None,
                delta_color=delta_color,
            )
