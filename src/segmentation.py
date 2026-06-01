"""
segmentation.py — RFM scoring, rule-based customer segmentation, and CLV
priority grouping.
Mirrors Notebook Sections 7 (RFM Scoring & Segmentation) and 17 (CLV & Priority).
"""

import pandas as pd
import numpy as np

from src.config import (
    SEGMENT_COLORS, PRIORITY_COLORS, ACTIONS,
    RISK_THRESHOLD, CLV_PERCENTILE, ONE_TIMER_PENALTY,
)
from src.utils import get_logger

logger = get_logger()


# ── RFM Scoring ───────────────────────────────────────────────────────────────

def create_rfm_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Score each RFM dimension 1-5 via quintiles."""
    rfm = df.copy()
    rfm["R_score"] = pd.qcut(rfm["Recency"],                              5, labels=[5, 4, 3, 2, 1])
    rfm["F_score"] = pd.qcut(rfm["Frequency"].rank(method="first"),       5, labels=[1, 2, 3, 4, 5])
    rfm["M_score"] = pd.qcut(rfm["Monetary"],                             5, labels=[1, 2, 3, 4, 5])
    rfm["RFM_Score"] = (
        rfm["R_score"].astype(str) +
        rfm["F_score"].astype(str) +
        rfm["M_score"].astype(str)
    )
    return rfm


# ── Segment Assignment ────────────────────────────────────────────────────────

def assign_segment(row) -> str:
    """Updated rule-based segment assignment."""
    r = int(row["R_score"])
    f = int(row["F_score"])
    m = int(row["M_score"])
    if r == 5 and f >= 4 and m >= 4:
        return "Champions"
    elif f >= 4 and r >= 3:
        return "Loyal"
    elif r <= 2 and f >= 2:
        return "At Risk"
    elif r == 1:
        return "Dormant"
    else:
        return "Others"


def add_segments(customer_df: pd.DataFrame) -> pd.DataFrame:
    """Apply RFM scoring and segment assignment, print summary."""
    customer_df = create_rfm_scores(customer_df)
    customer_df["Segment"] = customer_df.apply(assign_segment, axis=1)

    seg_counts = customer_df["Segment"].value_counts()
    logger.info("Segment distribution:")
    for seg, cnt in seg_counts.items():
        logger.info(f"  {seg:<12}: {cnt:,}")

    logger.info("\nChurn Rate by Segment:")
    seg_churn = customer_df.groupby("Segment")["Churn"].mean().sort_values(ascending=True)
    for seg, rate in seg_churn.items():
        bar = "#" * int(rate * 30)
        logger.info(f"  {seg:<12} {rate:.1%}  {bar}")

    return customer_df


# ── CLV & Priority ────────────────────────────────────────────────────────────

def compute_clv(customer_df: pd.DataFrame) -> pd.DataFrame:
    """
    CLV = AvgBasketSize x Frequency x ActiveMonths
    One-time buyers receive a 95% discount on CLV (no loyalty signal).
    """
    customer_df = customer_df.copy()
    customer_df["CLV"] = (
        customer_df["AvgBasketSize"] *
        customer_df["Frequency"] *
        customer_df["ActiveMonths"]
    )
    customer_df["Adjusted_CLV"] = np.where(
        customer_df["Frequency"] == 1,
        customer_df["CLV"] * ONE_TIMER_PENALTY,
        customer_df["CLV"],
    )
    return customer_df


def assign_priority(row) -> str:
    """Priority matrix: high/low risk x high/low value."""
    r, v = row["Risk_Level"], row["Value_Level"]
    if   r == "High Risk"  and v == "High Value": return "High Priority"
    elif r == "Low Risk"   and v == "High Value": return "Loyalty"
    elif r == "High Risk"  and v == "Low Value":  return "Nurture"
    else:                                          return "Low Priority"


def add_priority_groups(customer_df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign Risk_Level, Value_Level, Priority_Group, and Action columns.
    Revenue at Risk uses new formula: sum(Adjusted_CLV x Churn_Prob)
    """
    customer_df = compute_clv(customer_df)

    # Risk level
    customer_df["Risk_Level"] = np.where(
        customer_df["Churn_Prob"] >= RISK_THRESHOLD, "High Risk", "Low Risk"
    )

    # Value level
    clv_threshold = customer_df["Adjusted_CLV"].quantile(CLV_PERCENTILE)
    customer_df["Value_Level"] = np.where(
        (customer_df["Adjusted_CLV"] >= clv_threshold) &
        (customer_df["Frequency"] > 1),
        "High Value", "Low Value",
    )

    customer_df["Priority_Group"] = customer_df.apply(assign_priority, axis=1)
    customer_df["Action"]         = customer_df["Priority_Group"].map(ACTIONS)

    # Expected Revenue Loss = sum(Adjusted_CLV x Churn_Prob)
    customer_df["Expected_Revenue_Loss"] = (
        customer_df["Adjusted_CLV"] * customer_df["Churn_Prob"]
    )

    logger.info("Priority Group Summary (INR):")
    summary = customer_df.groupby("Priority_Group").agg(
        Customers              = ("CustomerID",             "count"),
        Avg_ChurnProb          = ("Churn_Prob",             "mean"),
        Avg_CLV_INR            = ("Adjusted_CLV",           "mean"),
        Total_CLV_INR          = ("Adjusted_CLV",           "sum"),
        Expected_Revenue_Loss  = ("Expected_Revenue_Loss",  "sum"),
    ).round(2)
    logger.info(f"\n{summary.to_string()}")

    hp  = customer_df[customer_df["Priority_Group"] == "High Priority"]
    loy = customer_df[customer_df["Priority_Group"] == "Loyalty"]
    total_erl = customer_df["Expected_Revenue_Loss"].sum()
    logger.info(f"Total Expected Revenue Loss (new formula): INR{total_erl:,.0f}")
    logger.info(f"Loyalty Group CLV to Protect: INR{loy['Adjusted_CLV'].sum():,.0f}")

    return customer_df, clv_threshold
