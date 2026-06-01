"""
dashboard/pages/retention_strategy.py
Page 4 — Retention Strategy: actionable recommendations and campaign simulator.
Revenue at Risk = sum(Adjusted_CLV x Churn_Prob)  [new formula]
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from dashboard.components.tables import (
    load_processed_data, load_kpis, load_business_validation, plot_path,
)
from dashboard.components.charts import LAYOUT_BASE, COLORS


def _fmt_inr(val: float) -> str:
    """Format as INR with M/K suffix."""
    if val >= 1_000_000:
        return f"\u20b9{val/1_000_000:.2f}M"
    elif val >= 1_000:
        return f"\u20b9{val/1_000:.1f}K"
    return f"\u20b9{val:,.0f}"


def _clv_at_risk_chart(df: pd.DataFrame) -> go.Figure:
    """
    Bar chart: sum of Adjusted_CLV for customers with Churn_Prob >= 0.5,
    grouped by Priority_Group.
    """
    high_risk = df[df["Churn_Prob"] >= 0.5].copy()
    grp = (
        high_risk.groupby("Priority_Group")["Expected_Revenue_Loss"]
        .sum()
        .div(1_000_000)
        .reset_index()
    )
    grp.columns = ["Priority_Group", "Revenue_Loss_M"]
    grp = grp.sort_values("Revenue_Loss_M", ascending=False)

    fig = px.bar(
        grp, x="Priority_Group", y="Revenue_Loss_M",
        color="Priority_Group",
        color_discrete_map=COLORS,
        title="Future Customer Value at Risk by Priority Group",
        text=grp["Revenue_Loss_M"].apply(lambda v: f"\u20b9{v:.2f}M"),
    )
    fig.update_traces(textposition="outside", textfont_size=11)
    fig.update_layout(
        **LAYOUT_BASE, showlegend=False,
        yaxis_title="Expected Revenue Loss (\u20b9M)",
    )
    return fig


def render():
    st.markdown("## Retention Strategy")
    st.markdown(
        "Actionable recommendations for each customer group and "
        "estimated revenue impact of targeted retention campaigns."
    )
    st.markdown("---")

    df   = load_processed_data()
    kpis = load_kpis()
    bv   = load_business_validation()

    # ── Revenue Summary KPIs ──────────────────────────────────────────────
    # New formula: revenue_at_risk = sum(Adjusted_CLV x Churn_Prob)
    risk_m   = kpis.get("revenue_at_risk_inr", 0) / 1_000_000
    loy_clv  = bv.get("loyalty_clv_inr", 0)        / 1_000_000

    # High Priority expected revenue loss from validation json
    hp_erl_m = bv.get("hp_expected_revenue_loss_inr", 0) / 1_000_000

    c1, c2, c3 = st.columns(3)
    c1.metric("Potential Revenue Exposure",
              f"\u20b9{risk_m:.1f}M",
              help="Expected Revenue Loss = \u03a3(Adjusted_CLV \u00d7 Churn_Prob)")
    c2.metric("Loyalty Group CLV to Protect",
              f"\u20b9{loy_clv:.1f}M",
              help="Total CLV of Low-Risk, High-Value customers")
    c3.metric("High Priority Expected Loss",
              f"\u20b9{hp_erl_m:.2f}M",
              help="Expected Revenue Loss for High Priority customers only")

    st.markdown("---")

    # ── Recommended Actions Table ─────────────────────────────────────────
    st.markdown("#### Recommended Actions by Priority Group")
    recommendations = pd.DataFrame([
        {"Priority Group": "High Priority", "Profile": "High churn risk + High CLV",
         "Action": "Immediate Retention Offer", "Channel": "Phone / Personal email",
         "Expected Benefit": "Prevent high-value loss"},
        {"Priority Group": "Loyalty",       "Profile": "Low churn risk + High CLV",
         "Action": "Loyalty & Upsell",          "Channel": "Email / App push",
         "Expected Benefit": "Increase wallet share"},
        {"Priority Group": "Nurture",       "Profile": "High churn risk + Low CLV",
         "Action": "Re-engagement Campaign",     "Channel": "Automated email",
         "Expected Benefit": "Win-back at low cost"},
        {"Priority Group": "Low Priority",  "Profile": "Low churn risk + Low CLV",
         "Action": "Monitor Only",               "Channel": "Newsletter",
         "Expected Benefit": "Passive retention"},
    ])
    st.dataframe(recommendations, use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── Future Customer Value at Risk ─────────────────────────────────────
    st.markdown("#### Future Customer Value at Risk by Priority Group")
    st.caption("Sum of Adjusted CLV for customers with churn probability \u2265 0.5")
    if "Priority_Group" in df.columns and "Expected_Revenue_Loss" in df.columns:
        st.plotly_chart(_clv_at_risk_chart(df), use_container_width=True)

    # ── Retention Campaign Simulator ──────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Retention Campaign Simulator")
    st.markdown("Estimate the financial impact of a targeted retention campaign.")

    high_risk_df = df[df["Churn_Prob"] >= 0.5].copy() if "Churn_Prob" in df.columns else pd.DataFrame()

    if not high_risk_df.empty:
        col_a, col_b = st.columns(2)
        with col_a:
            retention_rate = st.slider(
                "Campaign Success Rate (%)",
                min_value=5, max_value=60, value=20, step=5,
            )
        with col_b:
            campaign_cost_per_customer = st.slider(
                "Campaign Cost per Customer (\u20b9)",
                min_value=500, max_value=10000, value=2000, step=500,
            )

        group_filter = st.selectbox(
            "Target Group",
            ["All High Risk", "High Priority only", "Nurture only"],
        )

        if group_filter == "High Priority only":
            target = high_risk_df[high_risk_df["Priority_Group"] == "High Priority"]
        elif group_filter == "Nurture only":
            target = high_risk_df[high_risk_df["Priority_Group"] == "Nurture"]
        else:
            target = high_risk_df

        retained_count   = int(len(target) * retention_rate / 100)
        retained_revenue = target["Adjusted_CLV"].nlargest(retained_count).sum() / 1_000_000
        total_cost       = retained_count * campaign_cost_per_customer / 1_000_000
        net_profit       = retained_revenue - total_cost

        # Four metrics — no ROI
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Customers Targeted",  f"{len(target):,}")
        r2.metric("Expected Retained",   f"{retained_count:,}")
        r3.metric("Revenue Retained",    f"\u20b9{retained_revenue:.2f}M")
        r4.metric("Campaign Cost",       f"\u20b9{total_cost:.2f}M")

        # Net Profit highlight row
        np_color = "#d1fae5" if net_profit >= 0 else "#fde8e8"
        np_sign  = "+" if net_profit >= 0 else ""
        st.markdown(
            f"<div style='background:{np_color};border-radius:8px;padding:1rem 1.5rem;"
            f"margin-top:0.75rem;border:1px solid #e2e8f0;text-align:center'>"
            f"<span style='font-size:0.85rem;color:#64748b;font-weight:600'>NET PROFIT</span><br>"
            f"<span style='font-size:1.8rem;font-weight:700;color:#1e293b'>"
            f"{np_sign}\u20b9{net_profit:.2f}M</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

        st.markdown(
            f"<div class='insight-box' style='margin-top:0.75rem'>"
            f"Targeting <strong>{len(target):,} customers</strong> with a "
            f"<strong>{retention_rate}% campaign success rate</strong> is estimated to "
            f"retain <strong>{retained_count:,} customers</strong>, recovering "
            f"<strong>\u20b9{retained_revenue:.2f}M</strong> in CLV against a "
            f"campaign cost of <strong>\u20b9{total_cost:.2f}M</strong>, "
            f"yielding a net profit of <strong>{np_sign}\u20b9{net_profit:.2f}M</strong>."
            f"</div>",
            unsafe_allow_html=True,
        )
    else:
        st.info("Run the pipeline to enable the simulator.")

    # ── High Priority Customer List ───────────────────────────────────────
    st.markdown("---")
    st.markdown("#### High Priority Customers \u2014 Action List")
    if "Priority_Group" in df.columns:
        hp = df[df["Priority_Group"] == "High Priority"].sort_values(
            "Churn_Prob", ascending=False
        )
        display_cols = [c for c in [
            "CustomerID", "Churn_Prob", "Adjusted_CLV",
            "Expected_Revenue_Loss", "Recency", "Frequency", "Segment", "Action"
        ] if c in hp.columns]

        top_hp = hp[display_cols].head(50).copy()
        if "Churn_Prob" in top_hp.columns:
            top_hp["Churn_Prob"] = top_hp["Churn_Prob"].map(lambda v: f"{v:.1%}")
        if "Adjusted_CLV" in top_hp.columns:
            top_hp["Adjusted_CLV"] = top_hp["Adjusted_CLV"].map(_fmt_inr)
        if "Expected_Revenue_Loss" in top_hp.columns:
            top_hp["Expected_Revenue_Loss"] = top_hp["Expected_Revenue_Loss"].map(_fmt_inr)
        if "Action" in top_hp.columns:
            top_hp["Action"] = top_hp["Action"].map(
                lambda a: "Immediate Retention Offer" if "Immediate" in a else a
            )
        top_hp = top_hp.rename(columns={
            "Expected_Revenue_Loss": "Exp. Revenue Loss",
            "Adjusted_CLV": "Adj. CLV",
        })
        st.dataframe(top_hp, use_container_width=True, hide_index=True)
        st.caption(
            f"Showing top 50 of {len(hp):,} High Priority customers, "
            f"sorted by churn probability"
        )
