"""
visualization.py — All chart generation functions.
Each function mirrors the exact plot from the corresponding notebook section
and saves a PNG to outputs/plots/.

Sections covered:
  7  — Segment overview (count + revenue by segment)
  8  — RFM distributions
  9  — Mutual information bar chart
  10 — KDE churn separation
  13 — ROC curves + confusion matrix
  14 — Threshold analysis
  16 — Churn probability distribution
  18 — Business intelligence dashboard (6-panel)
"""

import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay

from src.config import (
    PLOTS_DIR, SEGMENT_COLORS, PRIORITY_COLORS, PLT_STYLE, RISK_THRESHOLD
)
from src.utils import get_logger

logger = get_logger()
plt.rcParams.update(PLT_STYLE)


# ── Helper ────────────────────────────────────────────────────────────────────

def _save(name: str) -> None:
    path = PLOTS_DIR / name
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {path}")


# ── Section 7 — Segment Overview ─────────────────────────────────────────────

def plot_segment_overview(customer_df: pd.DataFrame) -> None:
    seg_counts = customer_df["Segment"].value_counts()
    seg_rev    = customer_df.groupby("Segment")["Monetary"].sum().sort_values(ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Customer Segmentation Overview", fontsize=15, fontweight="bold")

    colors_ord = [SEGMENT_COLORS.get(s, "#95a5a6") for s in seg_counts.index]
    bars = axes[0].bar(seg_counts.index, seg_counts.values, color=colors_ord, edgecolor="white")
    axes[0].set_title("Customer Count by Segment")
    axes[0].set_ylabel("Customers")
    for b, v in zip(bars, seg_counts.values):
        axes[0].text(
            b.get_x() + b.get_width() / 2, b.get_height() + 10,
            f"{v:,}", ha="center", fontsize=10, fontweight="bold"
        )

    rev_cols = [SEGMENT_COLORS.get(s, "#95a5a6") for s in seg_rev.index]
    bars2 = axes[1].bar(seg_rev.index, seg_rev.values / 1000, color=rev_cols, edgecolor="white")
    axes[1].set_title("Revenue by Segment (₹ thousands)")
    axes[1].set_ylabel("Revenue (₹k)")
    for b, v in zip(bars2, seg_rev.values):
        axes[1].text(
            b.get_x() + b.get_width() / 2, b.get_height() + 1,
            f"₹{v/1000:,.0f}k", ha="center", fontsize=9, fontweight="bold"
        )

    plt.tight_layout()
    _save("segment_overview.png")


# ── Section 8 — RFM Distributions ────────────────────────────────────────────

def plot_rfm_distributions(customer_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle("RFM Distributions (Observation Window)", fontsize=15, fontweight="bold")

    for ax, col, color, xlabel in zip(
        axes,
        ["Recency", "Frequency", "Monetary"],
        ["#3498db", "#2ecc71", "#e74c3c"],
        ["Days Since Last Purchase", "Number of Orders", "Total Spend (₹)"],
    ):
        ax.hist(customer_df[col], bins=40, color=color, edgecolor="white", linewidth=0.5)
        ax.set_title(col)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Customers")
        med = customer_df[col].median()
        ax.axvline(med, color="black", linestyle="--", linewidth=1.2, label=f"Median: {med:,.0f}")
        ax.legend(fontsize=9)

    plt.tight_layout()
    _save("rfm_distributions.png")


# ── Section 9 — Mutual Information ───────────────────────────────────────────

def plot_mutual_information(mi_df: pd.DataFrame, threshold: float = 0.07) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#e74c3c" if s > threshold else "#3498db" for s in mi_df["MI Score"]]
    ax.barh(mi_df["Feature"], mi_df["MI Score"], color=colors, edgecolor="white")
    for b, v in zip(ax.patches, mi_df["MI Score"]):
        ax.text(v + 0.001, b.get_y() + b.get_height() / 2, f"{v:.4f}", va="center", fontsize=9)
    ax.axvline(threshold, color="black", linestyle="--", alpha=0.5, label=f"Threshold ({threshold})")
    ax.set_xlabel("Mutual Information Score")
    ax.set_title("Feature Importance — Mutual Information vs Churn")
    ax.legend()
    plt.tight_layout()
    _save("mutual_information.png")


# ── Section 10 — KDE Churn Separation ────────────────────────────────────────

def plot_kde_separation(customer_df: pd.DataFrame, feature_candidates: list) -> None:
    cols = 3
    rows = math.ceil(len(feature_candidates) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(18, 5 * rows))
    fig.suptitle("Churn Separation by Feature", fontsize=16, fontweight="bold", y=1.01)
    axes = axes.flatten()

    for i, feat in enumerate(feature_candidates):
        sns.kdeplot(
            data=customer_df, x=feat, hue="Churn", fill=True,
            palette={0: "#2ecc71", 1: "#e74c3c"}, alpha=0.5, ax=axes[i],
        )
        axes[i].set_title(feat)
        axes[i].set_xlabel("")
        axes[i].legend(title="Churn", labels=["Retained", "Churned"], fontsize=8)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    _save("kde_separation.png")


# ── Section 13 — Model Evaluation ────────────────────────────────────────────

def plot_model_evaluation(y_test, models_dict: dict, y_pred_xgb) -> None:
    """
    models_dict = {name: (y_prob, color)}
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Model Evaluation", fontsize=15, fontweight="bold")

    ax = axes[0]
    for name, (y_prob, color) in models_dict.items():
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        ax.plot(
            fpr, tpr,
            label=f"{name} AUC={roc_auc_score(y_test, y_prob):.4f}",
            linewidth=2, color=color,
        )
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.set_title("ROC Curve Comparison")
    ax.legend(loc="lower right", fontsize=9)

    cm   = confusion_matrix(y_test, y_pred_xgb)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Retained", "Churned"])
    disp.plot(ax=axes[1], colorbar=False, cmap="Blues")
    axes[1].set_title("XGBoost — Confusion Matrix")

    plt.tight_layout()
    _save("model_evaluation.png")

    # Save confusion matrix separately
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    disp2 = ConfusionMatrixDisplay(cm, display_labels=["Retained", "Churned"])
    disp2.plot(ax=ax2, colorbar=False, cmap="Blues")
    ax2.set_title("XGBoost — Confusion Matrix")
    plt.tight_layout()
    _save("confusion_matrix.png")

    # Save ROC separately
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    for name, (y_prob, color) in models_dict.items():
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        ax3.plot(fpr, tpr, label=f"{name} AUC={roc_auc_score(y_test, y_prob):.4f}",
                 linewidth=2, color=color)
    ax3.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random")
    ax3.set_xlabel("False Positive Rate")
    ax3.set_ylabel("True Positive Rate (Recall)")
    ax3.set_title("ROC Curve Comparison")
    ax3.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    _save("roc_curve.png")


# ── Section 14 — Threshold Analysis ──────────────────────────────────────────

def plot_threshold_analysis(thresh_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(thresh_df["Threshold"], thresh_df["Precision"], "o-", label="Precision", color="#3498db")
    ax.plot(thresh_df["Threshold"], thresh_df["Recall"],    "s-", label="Recall",    color="#e74c3c")
    ax.plot(thresh_df["Threshold"], thresh_df["F1"],        "^-", label="F1",        color="#2ecc71")
    ax.axvline(0.5, color="black", linestyle="--", alpha=0.7, label="Default (0.5)")
    ax.set_xlabel("Decision Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Precision / Recall / F1 vs Classification Threshold")
    ax.legend()
    plt.tight_layout()
    _save("threshold_analysis.png")


# ── Section 16 — Churn Probability Distribution ──────────────────────────────

def plot_churn_prob_distribution(customer_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(customer_df["Churn_Prob"], bins=40, color="#3498db", edgecolor="white", linewidth=0.5)
    ax.axvline(
        RISK_THRESHOLD, color="#e74c3c", linestyle="--",
        linewidth=2, label=f"Risk Threshold ({RISK_THRESHOLD})"
    )
    ax.set_xlabel("Churn Probability")
    ax.set_ylabel("Customers")
    ax.set_title("Churn Probability Distribution — All Customers")
    ax.legend()
    plt.tight_layout()
    _save("churn_prob_dist.png")


# ── Section 18 — Business Intelligence Dashboard ─────────────────────────────

def plot_business_dashboard(customer_df: pd.DataFrame, clv_threshold: float) -> None:
    pcolors = PRIORITY_COLORS

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(
        "Retail Churn — Business Intelligence Dashboard (₹)",
        fontsize=17, fontweight="bold", y=1.01,
    )

    # 1 — Scatter: CLV vs Churn Probability
    ax1 = fig.add_subplot(2, 3, 1)
    for pg, grp in customer_df.groupby("Priority_Group"):
        ax1.scatter(
            np.log1p(grp["Adjusted_CLV"]), grp["Churn_Prob"],
            label=pg, color=pcolors.get(pg, "#aaa"), alpha=0.5, s=18,
        )
    ax1.axhline(RISK_THRESHOLD, color="red", linestyle="--", linewidth=1.2, alpha=0.8)
    ax1.axvline(np.log1p(clv_threshold), color="black", linestyle="--", linewidth=1.2, alpha=0.6)
    ax1.set_xlabel("Log Adjusted CLV (₹)")
    ax1.set_ylabel("Churn Probability")
    ax1.set_title("Customer Segmentation Map")
    ax1.legend(fontsize=8, markerscale=1.5)

    # 2 — Revenue contribution by priority
    ax2 = fig.add_subplot(2, 3, 2)
    pct_rev = (
        customer_df.groupby("Priority_Group")["Adjusted_CLV"].sum() /
        customer_df["Adjusted_CLV"].sum() * 100
    ).sort_values(ascending=False)
    bars = ax2.bar(
        pct_rev.index, pct_rev.values,
        color=[pcolors.get(p, "#aaa") for p in pct_rev.index], edgecolor="white",
    )
    ax2.set_ylabel("Revenue Share (%)")
    ax2.set_title("Revenue by Priority Group")
    ax2.tick_params(axis="x", rotation=15)
    for b, v in zip(bars, pct_rev.values):
        ax2.text(
            b.get_x() + b.get_width() / 2, b.get_height() + 0.3,
            f"{v:.1f}%", ha="center", fontsize=10, fontweight="bold",
        )

    # 3 — Avg churn probability by segment
    ax3 = fig.add_subplot(2, 3, 3)
    scp = customer_df.groupby("Segment")["Churn_Prob"].mean().sort_values(ascending=False)
    bars3 = ax3.bar(
        scp.index, scp.values,
        color=[SEGMENT_COLORS.get(s, "#95a5a6") for s in scp.index], edgecolor="white",
    )
    ax3.axhline(0.5, color="red", linestyle="--", linewidth=1.5, alpha=0.8)
    ax3.set_ylabel("Avg Churn Probability")
    ax3.set_title("Avg Churn Prob by Segment")
    ax3.tick_params(axis="x", rotation=15)
    for b, v in zip(bars3, scp.values):
        ax3.text(
            b.get_x() + b.get_width() / 2, b.get_height() + 0.005,
            f"{v:.1%}", ha="center", fontsize=9, fontweight="bold",
        )

    # 4 — CLV distribution by priority (KDE)
    ax4 = fig.add_subplot(2, 3, 4)
    for pg, grp in customer_df.groupby("Priority_Group"):
        sns.kdeplot(
            np.log1p(grp["Adjusted_CLV"]), ax=ax4,
            label=pg, fill=True, alpha=0.35, color=pcolors.get(pg, "#aaa"),
        )
    ax4.set_xlabel("Log Adjusted CLV (₹)")
    ax4.set_title("CLV Distribution by Priority")
    ax4.legend(fontsize=8)

    # 5 — Pie chart: customer mix by priority
    ax5 = fig.add_subplot(2, 3, 5)
    ac = customer_df["Priority_Group"].value_counts()
    ax5.pie(
        ac.values, labels=ac.index,
        colors=[pcolors.get(p, "#aaa") for p in ac.index],
        autopct="%1.1f%%", startangle=140, textprops={"fontsize": 9},
    )
    ax5.set_title("Customer Mix by Priority")

    # 6 — Risk banding histogram
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.hist(
        customer_df[customer_df["Churn_Prob"] <  0.5]["Churn_Prob"],
        bins=30, color="#2ecc71", alpha=0.75, label="Low Risk (<0.5)",
    )
    ax6.hist(
        customer_df[customer_df["Churn_Prob"] >= 0.5]["Churn_Prob"],
        bins=30, color="#e74c3c", alpha=0.75, label="High Risk (≥0.5)",
    )
    ax6.axvline(0.5, color="black", linestyle="--", linewidth=1.5)
    ax6.set_xlabel("Churn Probability")
    ax6.set_ylabel("Customers")
    ax6.set_title("Risk Banding")
    ax6.legend()

    plt.tight_layout()
    _save("business_dashboard.png")


# ── Churn Trend Proxy ─────────────────────────────────────────────────────────

def plot_churn_trend(df_obs: pd.DataFrame) -> None:
    """Monthly transaction volume trend (proxy for churn trend dashboard)."""
    df_obs = df_obs.copy()
    df_obs["YearMonth"] = df_obs["InvoiceDate"].dt.to_period("M")
    monthly = df_obs.groupby("YearMonth").agg(
        Revenue   = ("TotalPrice",  "sum"),
        Customers = ("CustomerID",  "nunique"),
        Orders    = ("InvoiceNo",   "nunique"),
    ).reset_index()
    monthly["YearMonth_str"] = monthly["YearMonth"].astype(str)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Monthly Business Trend (Observation Window)", fontsize=14, fontweight="bold")

    axes[0].plot(monthly["YearMonth_str"], monthly["Revenue"] / 1_000_000,
                 "o-", color="#3498db", linewidth=2)
    axes[0].set_title("Monthly Revenue (₹ Millions)")
    axes[0].set_xlabel("Month")
    axes[0].set_ylabel("Revenue (₹M)")
    axes[0].tick_params(axis="x", rotation=45)

    axes[1].plot(monthly["YearMonth_str"], monthly["Customers"],
                 "s-", color="#2ecc71", linewidth=2)
    axes[1].set_title("Monthly Active Customers")
    axes[1].set_xlabel("Month")
    axes[1].set_ylabel("Unique Customers")
    axes[1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    _save("churn_trend.png")


# ── Cohort Retention Heatmap ──────────────────────────────────────────────────

def plot_cohort_retention(df_obs: pd.DataFrame) -> None:
    """Simple cohort retention heatmap by acquisition month."""
    try:
        df = df_obs.copy()
        df["InvoiceMonth"] = df["InvoiceDate"].dt.to_period("M")
        first_month = df.groupby("CustomerID")["InvoiceMonth"].min().rename("CohortMonth")
        df = df.merge(first_month.reset_index(), on="CustomerID")
        df["CohortIndex"] = (
            df["InvoiceMonth"].dt.to_timestamp() -
            df["CohortMonth"].dt.to_timestamp()
        ).dt.days // 30

        cohort_data = df.groupby(["CohortMonth", "CohortIndex"])["CustomerID"].nunique().reset_index()
        cohort_pivot = cohort_data.pivot(index="CohortMonth", columns="CohortIndex", values="CustomerID")
        cohort_size  = cohort_pivot.iloc[:, 0]
        retention    = cohort_pivot.divide(cohort_size, axis=0).iloc[:, :9]  # first 9 months

        fig, ax = plt.subplots(figsize=(13, 7))
        sns.heatmap(
            retention, annot=True, fmt=".0%", cmap="YlGn",
            linewidths=0.5, ax=ax, cbar_kws={"label": "Retention Rate"},
        )
        ax.set_title("Cohort Retention Heatmap (by Acquisition Month)", fontsize=13, fontweight="bold")
        ax.set_xlabel("Months Since First Purchase")
        ax.set_ylabel("Cohort (Acquisition Month)")
        plt.tight_layout()
        _save("cohort_retention.png")
    except Exception as e:
        logger.warning(f"Cohort heatmap skipped: {e}")


# ── CLV Distribution ──────────────────────────────────────────────────────────

def plot_clv_distribution(customer_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Customer Lifetime Value Distribution", fontsize=14, fontweight="bold")

    axes[0].hist(np.log1p(customer_df["Adjusted_CLV"]), bins=40,
                 color="#3498db", edgecolor="white", linewidth=0.5)
    axes[0].set_xlabel("Log Adjusted CLV (₹)")
    axes[0].set_ylabel("Customers")
    axes[0].set_title("CLV Distribution (log scale)")

    seg_clv = customer_df.groupby("Segment")["Adjusted_CLV"].mean().sort_values(ascending=False)
    colors  = [SEGMENT_COLORS.get(s, "#95a5a6") for s in seg_clv.index]
    axes[1].bar(seg_clv.index, seg_clv.values / 1000, color=colors, edgecolor="white")
    axes[1].set_xlabel("Segment")
    axes[1].set_ylabel("Avg CLV (₹k)")
    axes[1].set_title("Average CLV by Segment")
    axes[1].tick_params(axis="x", rotation=15)

    plt.tight_layout()
    _save("clv_distribution.png")
