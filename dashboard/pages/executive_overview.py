"""
dashboard/pages/executive_overview.py
Page 1 — Executive Overview: business impact of churn.
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

from dashboard.components.tables import (
    load_kpis, load_processed_data, load_segment_summary, plot_path,
)
from dashboard.components.charts import (
    segment_bar_chart, segment_revenue_bar,
    churn_rate_by_segment, monthly_trend_chart,
)
from dashboard.components.kpi_cards import render_kpi_row


def render():
    st.markdown("## Executive Overview")
    st.markdown(
        "A high-level summary of churn exposure, potential revenue exposure, and "
        "customer distribution across segments."
    )
    st.markdown("---")

    kpis = load_kpis()
    df   = load_processed_data()

    # ── KPI Row ───────────────────────────────────────────────────────────
    churn_pct  = kpis.get("churn_rate", 0) * 100
    rev_m      = kpis.get("total_revenue_inr", 0) / 1_000_000
    risk_rev_m = kpis.get("revenue_at_risk_inr", 0) / 1_000_000
    risk_pct   = (risk_rev_m / rev_m * 100) if rev_m else 0

    kpi_dict = {
        "Active Customer Base":          (f"{kpis.get('total_customers', 0):,}",
                                          "", "off"),
        "Historical Churn Rate":         (f"{churn_pct:.1f}%",
                                          "% inactive in Oct\u2013Dec 2011", "off"),
        "Customers at High Churn Risk":  (f"{kpis.get('high_risk_count', 0):,}",
                                          "Churn probability \u2265 0.5", "off"),
        "Observed Revenue":              (f"\u20b9{rev_m:.1f}M",
                                          "", "off"),
        "Potential Revenue Exposure":    (f"\u20b9{risk_rev_m:.1f}M",
                                          "Revenue linked to high-risk customers", "inverse"),
    }
    render_kpi_row(kpi_dict)

    st.markdown("---")

    # ── Business Impact Statement ─────────────────────────────────────────
    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("#### Churn Business Impact")
        hp_count = kpis.get("high_priority_customers", 0)
        st.markdown(
            f"""
            <div class='insight-box'>
            With a historical churn rate of <strong>{churn_pct:.1f}%</strong> across
            <strong>{kpis.get('total_customers', 0):,}</strong> customers,
            the model estimates an expected future revenue loss of
            <strong>\u20b9{risk_rev_m:.1f}M</strong> based on customer lifetime value
            and predicted churn probabilities.<br><br>
            <strong>{hp_count} High Priority customers</strong> require immediate
            retention action (high churn probability + high CLV).
            Acquiring a new customer costs 5\u20137\u00d7 more than retaining one.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown("#### Priority Group Breakdown")
        if "Priority_Group" in df.columns:
            priority_counts = df["Priority_Group"].value_counts().reset_index()
            priority_counts.columns = ["Group", "Customers"]
            st.dataframe(priority_counts, use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── Segment Charts ────────────────────────────────────────────────────
    st.markdown("#### Customer Segmentation at a Glance")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(segment_bar_chart(df), use_container_width=True)
    with c2:
        st.plotly_chart(segment_revenue_bar(df), use_container_width=True)

    st.plotly_chart(churn_rate_by_segment(df), use_container_width=True)

    st.markdown("---")

    # ── Monthly Trend — single chart only ────────────────────────────────
    st.markdown("#### Observation Window Business Trends")
    img_path = plot_path("churn_trend.png")
    if img_path.exists():
        st.image(str(img_path), caption="Monthly Revenue & Monthly Active Customers")
    else:
        st.info("Run the pipeline to generate the trend chart.")

    # ── Segment Summary Table ─────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Segment Summary")
    seg_df = load_segment_summary()
    if not seg_df.empty:
        seg_display = seg_df.copy()
        if "Churn_Rate" in seg_display.columns:
            seg_display["Churn_Rate"] = seg_display["Churn_Rate"].map(lambda v: f"{v:.1%}")
        if "Avg_ChurnProb" in seg_display.columns:
            seg_display["Avg_ChurnProb"] = seg_display["Avg_ChurnProb"].map(lambda v: f"{v:.1%}")
        if "Total_Revenue" in seg_display.columns:
            seg_display["Total_Revenue"] = seg_display["Total_Revenue"].map(lambda v: f"\u20b9{v:,.0f}")
        if "Avg_CLV" in seg_display.columns:
            seg_display["Avg_CLV"] = seg_display["Avg_CLV"].map(lambda v: f"\u20b9{v:,.0f}")
        st.dataframe(seg_display, use_container_width=True, hide_index=True)
