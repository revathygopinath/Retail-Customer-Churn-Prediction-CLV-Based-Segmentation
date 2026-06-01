"""
dashboard/components/charts.py — Plotly chart factory functions.
All charts use a clean light theme consistent with the CSS.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

COLORS = {
    "Champions": "#2ecc71",
    "Loyal":     "#3498db",
    "At Risk":   "#e74c3c",
    "Dormant":   "#95a5a6",
    "Others":    "#f39c12",
    "High Priority": "#e74c3c",
    "Loyalty":       "#2ecc71",
    "Nurture":       "#f39c12",
    "Low Priority":  "#95a5a6",
}

LAYOUT_BASE = dict(
    template="plotly_white",
    font=dict(family="Inter, sans-serif", size=12, color="#1e293b"),
    plot_bgcolor="#f8fafc",
    paper_bgcolor="#ffffff",
    margin=dict(l=40, r=20, t=50, b=40),
)


def segment_bar_chart(customer_df: pd.DataFrame) -> go.Figure:
    seg_counts = customer_df["Segment"].value_counts().reset_index()
    seg_counts.columns = ["Segment", "Customers"]
    fig = px.bar(
        seg_counts, x="Segment", y="Customers",
        color="Segment",
        color_discrete_map=COLORS,
        title="Customer Count by Segment",
        text="Customers",
    )
    fig.update_traces(textposition="outside", textfont_size=11)
    fig.update_layout(**LAYOUT_BASE, showlegend=False)
    return fig


def segment_revenue_bar(customer_df: pd.DataFrame) -> go.Figure:
    seg_rev = (
        customer_df.groupby("Segment")["Monetary"]
        .sum()
        .div(1_000_000)
        .reset_index()
    )
    seg_rev.columns = ["Segment", "Revenue_M"]
    seg_rev = seg_rev.sort_values("Revenue_M", ascending=False)
    fig = px.bar(
        seg_rev, x="Segment", y="Revenue_M",
        color="Segment",
        color_discrete_map=COLORS,
        title="Revenue by Segment (₹ Millions)",
        text=seg_rev["Revenue_M"].apply(lambda v: f"₹{v:.1f}M"),
    )
    fig.update_traces(textposition="outside", textfont_size=11)
    fig.update_layout(**LAYOUT_BASE, showlegend=False, yaxis_title="Revenue (₹M)")
    return fig


def churn_rate_by_segment(customer_df: pd.DataFrame) -> go.Figure:
    seg_churn = (
        customer_df.groupby("Segment")["Churn"].mean()
        .mul(100).round(1)
        .reset_index()
    )
    seg_churn.columns = ["Segment", "Churn_Rate"]
    seg_churn = seg_churn.sort_values("Churn_Rate", ascending=True)
    fig = px.bar(
        seg_churn, x="Churn_Rate", y="Segment",
        orientation="h",
        color="Segment",
        color_discrete_map=COLORS,
        title="Churn Rate by Segment (%)",
        text=seg_churn["Churn_Rate"].apply(lambda v: f"{v:.1f}%"),
    )
    fig.update_traces(textposition="outside", textfont_size=11)
    fig.update_layout(**LAYOUT_BASE, showlegend=False, xaxis_title="Churn Rate (%)")
    fig.add_vline(x=50, line_dash="dash", line_color="#e74c3c", opacity=0.6)
    return fig


def churn_prob_histogram(customer_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    low  = customer_df[customer_df["Churn_Prob"] <  0.5]["Churn_Prob"]
    high = customer_df[customer_df["Churn_Prob"] >= 0.5]["Churn_Prob"]
    fig.add_trace(go.Histogram(x=low,  name="Low Risk (<0.5)",  marker_color="#2ecc71",
                               opacity=0.8, nbinsx=25))
    fig.add_trace(go.Histogram(x=high, name="High Risk (≥0.5)", marker_color="#e74c3c",
                               opacity=0.8, nbinsx=25))
    fig.add_vline(x=0.5, line_dash="dash", line_color="#1e293b", line_width=2,
                  annotation_text="Threshold (0.5)", annotation_position="top right")
    fig.update_layout(
        **LAYOUT_BASE, barmode="overlay",
        title="Churn Probability Distribution",
        xaxis_title="Churn Probability", yaxis_title="Customers",
    )
    return fig


def priority_pie(customer_df: pd.DataFrame) -> go.Figure:
    ac = customer_df["Priority_Group"].value_counts().reset_index()
    ac.columns = ["Priority_Group", "Count"]
    fig = px.pie(
        ac, names="Priority_Group", values="Count",
        color="Priority_Group",
        color_discrete_map=COLORS,
        title="Customer Mix by Priority Group",
        hole=0.35,
    )
    fig.update_traces(textposition="inside", textinfo="percent+label", textfont_size=11)
    fig.update_layout(**LAYOUT_BASE, showlegend=True)
    return fig


def priority_revenue_bar(customer_df: pd.DataFrame) -> go.Figure:
    pct_rev = (
        customer_df.groupby("Priority_Group")["Adjusted_CLV"].sum() /
        customer_df["Adjusted_CLV"].sum() * 100
    ).reset_index()
    pct_rev.columns = ["Priority_Group", "Revenue_Pct"]
    pct_rev = pct_rev.sort_values("Revenue_Pct", ascending=False)
    fig = px.bar(
        pct_rev, x="Priority_Group", y="Revenue_Pct",
        color="Priority_Group",
        color_discrete_map=COLORS,
        title="Revenue Share by Priority Group (%)",
        text=pct_rev["Revenue_Pct"].apply(lambda v: f"{v:.1f}%"),
    )
    fig.update_traces(textposition="outside", textfont_size=11)
    fig.update_layout(**LAYOUT_BASE, showlegend=False, yaxis_title="Revenue Share (%)")
    return fig


def clv_vs_churn_scatter(customer_df: pd.DataFrame, clv_threshold: float) -> go.Figure:
    fig = px.scatter(
        customer_df,
        x=np.log1p(customer_df["Adjusted_CLV"]),
        y="Churn_Prob",
        color="Priority_Group",
        color_discrete_map=COLORS,
        opacity=0.55,
        size_max=8,
        title="Customer Prioritization Map: CLV vs Churn Risk",
        labels={"x": "Log Adjusted CLV (\u20b9)", "Churn_Prob": "Predicted Churn Probability"},
        hover_data=["CustomerID", "Segment"],
    )
    fig.add_hline(y=0.5, line_dash="dash", line_color="#e74c3c", opacity=0.7,
                  annotation_text="Risk Threshold")
    fig.add_vline(x=np.log1p(clv_threshold), line_dash="dash", line_color="#1e293b",
                  opacity=0.5, annotation_text="CLV Threshold")
    fig.add_annotation(
        x=0.98, y=0.98, xref="paper", yref="paper",
        text="Top-right quadrant represents valuable customers at highest retention risk.",
        showarrow=False, font=dict(size=10, color="#64748b"),
        align="right", bgcolor="rgba(255,255,255,0.8)",
        bordercolor="#e2e8f0", borderwidth=1,
    )
    fig.update_layout(**LAYOUT_BASE, yaxis_title="Predicted Churn Probability")
    return fig


def roc_curve_chart(roc_data: list) -> go.Figure:
    """
    roc_data = [(name, fpr, tpr, auc_score, color), ...]
    """
    fig = go.Figure()
    for name, fpr, tpr, auc_val, color in roc_data:
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines", name=f"{name} (AUC={auc_val:.4f})",
            line=dict(color=color, width=2),
        ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines", name="Random",
        line=dict(color="#94a3b8", dash="dash", width=1),
    ))
    fig.update_layout(
        **LAYOUT_BASE,
        title="ROC Curve Comparison",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate (Recall)",
        legend=dict(x=0.6, y=0.05),
    )
    return fig


def shap_bar_plotly(feat_importance: pd.DataFrame) -> go.Figure:
    df = feat_importance.sort_values("SHAP_Importance", ascending=True)
    fig = px.bar(
        df, x="SHAP_Importance", y="Feature", orientation="h",
        title="Mean |SHAP Value| — Global Feature Importance",
        color="SHAP_Importance",
        color_continuous_scale=[[0, "#bfdbfe"], [1, "#1d4ed8"]],
        text=df["SHAP_Importance"].apply(lambda v: f"{v:.4f}"),
    )
    fig.update_traces(textposition="outside", textfont_size=10)
    fig.update_layout(**LAYOUT_BASE, showlegend=False,
                      xaxis_title="Mean |SHAP Value|", coloraxis_showscale=False)
    return fig


def threshold_line_chart(thresh_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=thresh_df["Threshold"], y=thresh_df["Precision"],
                             mode="lines+markers", name="Precision",
                             marker_symbol="circle",  line_color="#3498db"))
    fig.add_trace(go.Scatter(x=thresh_df["Threshold"], y=thresh_df["Recall"],
                             mode="lines+markers", name="Recall",
                             marker_symbol="square",  line_color="#e74c3c"))
    fig.add_trace(go.Scatter(x=thresh_df["Threshold"], y=thresh_df["F1"],
                             mode="lines+markers", name="F1",
                             marker_symbol="triangle-up", line_color="#2ecc71"))
    fig.add_vline(x=0.5, line_dash="dash", line_color="#1e293b", opacity=0.6,
                  annotation_text="Default (0.5)")
    fig.update_layout(
        **LAYOUT_BASE,
        title="Precision / Recall / F1 vs Classification Threshold",
        xaxis_title="Decision Threshold", yaxis_title="Score",
    )
    return fig


def monthly_trend_chart(df_obs: pd.DataFrame) -> go.Figure:
    df = df_obs.copy()
    df["YearMonth"] = df["InvoiceDate"].dt.to_period("M").astype(str)
    monthly = df.groupby("YearMonth").agg(
        Revenue   = ("TotalPrice",  "sum"),
        Customers = ("CustomerID",  "nunique"),
    ).reset_index()

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(x=monthly["YearMonth"], y=monthly["Revenue"] / 1_000_000,
               name="Revenue (₹M)", marker_color="#3498db", opacity=0.7),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=monthly["YearMonth"], y=monthly["Customers"],
                   mode="lines+markers", name="Active Customers",
                   line_color="#e74c3c", line_width=2),
        secondary_y=True,
    )
    fig.update_layout(
        **LAYOUT_BASE, title="Monthly Revenue and Active Customers",
        xaxis_title="Month", legend=dict(x=0.01, y=0.99),
    )
    fig.update_yaxes(title_text="Revenue (₹M)", secondary_y=False)
    fig.update_yaxes(title_text="Active Customers", secondary_y=True)
    return fig


def clv_segment_box(customer_df: pd.DataFrame) -> go.Figure:
    fig = px.box(
        customer_df, x="Segment", y=np.log1p(customer_df["Adjusted_CLV"]),
        color="Segment", color_discrete_map=COLORS,
        title="CLV Distribution by Segment (log scale)",
        labels={"y": "Log Adjusted CLV (₹)"},
    )
    fig.update_layout(**LAYOUT_BASE, showlegend=False)
    return fig
