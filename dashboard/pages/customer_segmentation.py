"""
dashboard/pages/customer_segmentation.py
Page 2 — Customer Segmentation: CLV, cohort, priority analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from dashboard.components.tables import load_processed_data, load_segment_summary, plot_path
from dashboard.components.charts import (
    clv_vs_churn_scatter, clv_segment_box, priority_pie, priority_revenue_bar, COLORS, LAYOUT_BASE,
)


def _fmt_inr(val: float) -> str:
    """Format a value as INR with M/K suffix."""
    if val >= 1_000_000:
        return f"\u20b9{val/1_000_000:.1f}M"
    elif val >= 1_000:
        return f"\u20b9{val/1_000:.0f}K"
    else:
        return f"\u20b9{val:,.0f}"


def render():
    st.markdown("## Customer Segmentation")
    st.markdown(
        "Identify which customer groups drive revenue, which are at risk, "
        "and how CLV is distributed across segments."
    )
    st.markdown("---")

    df = load_processed_data()

    # ── CLV Distribution ──────────────────────────────────────────────────
    st.markdown("#### Customer Lifetime Value Analysis")

    if "Adjusted_CLV" in df.columns and "Segment" in df.columns:
        c3, c4 = st.columns(2)
        with c3:
            st.plotly_chart(clv_segment_box(df), use_container_width=True)
        with c4:
            clv_stats = (
                df.groupby("Segment")["Adjusted_CLV"]
                .agg(["mean", "median", "sum"])
                .rename(columns={"mean": "Avg CLV", "median": "Median CLV", "sum": "Total CLV"})
                .reset_index()
            )
            # Sort descending by Avg CLV
            clv_stats = clv_stats.sort_values("Avg CLV", ascending=False)
            # Format with M/K
            clv_stats["Avg CLV"]    = clv_stats["Avg CLV"].map(_fmt_inr)
            clv_stats["Median CLV"] = clv_stats["Median CLV"].map(_fmt_inr)
            clv_stats["Total CLV"]  = clv_stats["Total CLV"].map(_fmt_inr)
            st.markdown("**CLV Statistics by Segment**")
            st.dataframe(clv_stats, use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── Priority Groups ───────────────────────────────────────────────────
    st.markdown("#### Priority Group Analysis")
    st.markdown(
        "<div class='insight-box'>"
        "Although Nurture customers represent the majority of the customer base, "
        "Loyalty customers contribute nearly all business revenue, highlighting the "
        "importance of protecting high-value retained customers."
        "</div>",
        unsafe_allow_html=True,
    )

    if "Priority_Group" in df.columns and "Adjusted_CLV" in df.columns:
        clv_threshold = df.loc[df["Adjusted_CLV"] > 0, "Adjusted_CLV"].quantile(0.65)

        c5, c6 = st.columns(2)
        with c5:
            # Rename for display
            fig_pie = priority_pie(df)
            fig_pie.update_layout(title="Customer Distribution by Priority Group")
            st.plotly_chart(fig_pie, use_container_width=True)
        with c6:
            st.plotly_chart(priority_revenue_bar(df), use_container_width=True)

        st.plotly_chart(
            clv_vs_churn_scatter(df, clv_threshold),
            use_container_width=True,
        )

    st.markdown("---")

    # ── Cohort Retention Heatmap ──────────────────────────────────────────
    st.markdown("#### Cohort Retention Analysis")
    cohort_path = plot_path("cohort_retention.png")
    if cohort_path.exists():
        st.image(str(cohort_path), caption="Cohort Retention Heatmap")
        st.markdown(
            "<div class='insight-box'>"
            "Each row represents the acquisition cohort (month of first purchase). "
            "Values show what percentage of customers remained active in subsequent months. "
            "Darker cells indicate higher retention."
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        st.info("Cohort retention chart not found. Re-run the pipeline.")

    # ── Customer Retention Recommendations Table ──────────────────────────
    st.markdown("---")
    st.markdown("#### Customer Retention Recommendations")

    ACTION_SHORT = {
        "Immediate retention offer (discount / VIP upgrade)": "Immediate Retention Offer",
        "Loyalty rewards & upsell campaign":                   "Loyalty & Upsell",
        "Low-cost re-engagement (email / push)":               "Re-engagement Campaign",
        "No immediate action \u2014 monitor":                  "Monitor Only",
    }

    PRIORITY_COLOR_MAP = {
        "High Priority": "#fde8e8",
        "Loyalty":       "#d1fae5",
        "Nurture":       "#fef3c7",
        "Low Priority":  "#f1f5f9",
    }

    if "Segment" in df.columns and "Priority_Group" in df.columns:
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            seg_filter = st.selectbox(
                "Filter by Segment",
                ["All"] + sorted(df["Segment"].unique().tolist()),
                key="seg_filter",
            )
        with col_f2:
            pg_filter = st.selectbox(
                "Filter by Priority Group",
                ["All", "High Priority", "Loyalty", "Nurture", "Low Priority"],
                key="pg_filter",
            )

        show_df = df.copy()
        if seg_filter != "All":
            show_df = show_df[show_df["Segment"] == seg_filter]
        if pg_filter != "All":
            show_df = show_df[show_df["Priority_Group"] == pg_filter]

        # Build display columns — add Expected Revenue Loss
        display_cols = [c for c in [
            "CustomerID", "Segment", "Priority_Group", "Churn_Prob",
            "Recency", "Frequency", "Monetary", "Adjusted_CLV",
            "Expected_Revenue_Loss", "Action"
        ] if c in show_df.columns]

        top100 = show_df.sort_values("Churn_Prob", ascending=False).head(100)[display_cols].copy()

        # Format columns
        if "Churn_Prob" in top100.columns:
            top100["Churn_Prob"] = top100["Churn_Prob"].map(lambda v: f"{v:.1%}")
        if "Monetary" in top100.columns:
            top100["Monetary"] = top100["Monetary"].map(lambda v: f"\u20b9{v:,.0f}")
        if "Adjusted_CLV" in top100.columns:
            top100["Adjusted_CLV"] = top100["Adjusted_CLV"].map(_fmt_inr)
        if "Expected_Revenue_Loss" in top100.columns:
            top100["Expected_Revenue_Loss"] = top100["Expected_Revenue_Loss"].map(_fmt_inr)
        if "Action" in top100.columns:
            top100["Action"] = top100["Action"].map(lambda a: ACTION_SHORT.get(a, a))

        # Rename columns for clarity
        top100 = top100.rename(columns={
            "Expected_Revenue_Loss": "Exp. Revenue Loss",
            "Adjusted_CLV": "Adj. CLV",
            "Priority_Group": "Priority",
        })

        # Color-highlight rows by priority group
        def highlight_priority(row):
            pg = row.get("Priority", "")
            bg = PRIORITY_COLOR_MAP.get(pg, "#ffffff")
            return [f"background-color: {bg}"] * len(row)

        styled = top100.style.apply(highlight_priority, axis=1)
        st.dataframe(styled, use_container_width=True, hide_index=True)
        st.caption(
            f"Showing top 100 of {len(show_df):,} customers (sorted by churn probability). "
            f"Red = High Priority | Green = Loyalty | Yellow = Nurture"
        )
