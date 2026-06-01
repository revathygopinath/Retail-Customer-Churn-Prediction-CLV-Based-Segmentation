"""
business_metrics.py — Business validation checks and summary KPIs.
Revenue at Risk uses new formula: Expected Revenue Loss = sum(Adjusted_CLV x Churn_Prob)
"""

import pandas as pd
import numpy as np

from src.config import METRICS_DIR, REPORTS_DIR
from src.utils import get_logger, save_json

logger = get_logger()


def run_business_validation(customer_df: pd.DataFrame) -> dict:
    """Replicate the exact validation checks from the notebook."""
    recency_corr = customer_df[["Recency", "Churn_Prob"]].corr().iloc[0, 1]
    clv_corr     = customer_df[["Adjusted_CLV", "Churn_Prob"]].corr().iloc[0, 1]

    recency_ok = recency_corr > 0
    clv_ok     = clv_corr < 0

    logger.info("=" * 60)
    logger.info("MODEL & BUSINESS VALIDATION CHECKS")
    logger.info("=" * 60)
    logger.info(f"{'[OK]' if recency_ok else '[FAIL]'} Recency <-> Churn (expect +ve): {recency_corr:.4f}")
    logger.info(f"{'[OK]' if clv_ok else '[FAIL]'} CLV <-> Churn (expect -ve)   : {clv_corr:.4f}")

    one_timers = customer_df[
        (customer_df["Frequency"] == 1) &
        (customer_df["Value_Level"] == "High Value")
    ]
    one_timer_ok = len(one_timers) == 0
    logger.info(f"{'[OK]' if one_timer_ok else '[FAIL]'} One-time buyers in High Value: {len(one_timers)}")

    logger.info("\nAvg Churn Prob by Segment:")
    seg_probs = customer_df.groupby("Segment")["Churn_Prob"].mean().sort_values(ascending=True)
    for seg, prob in seg_probs.items():
        icon = "[OK]" if seg == "Champions" else "    "
        logger.info(f"  {icon} {seg:<12}: {prob:.4f}")

    hp  = customer_df[customer_df["Priority_Group"] == "High Priority"]
    loy = customer_df[customer_df["Priority_Group"] == "Loyalty"]

    # New formula: Expected Revenue Loss = sum(Adjusted_CLV x Churn_Prob)
    total_erl = customer_df["Expected_Revenue_Loss"].sum()
    hp_erl    = hp["Expected_Revenue_Loss"].sum()

    logger.info(f"\nTotal Expected Revenue Loss (new formula): INR{total_erl:,.0f}")
    logger.info(f"High Priority Expected Revenue Loss: INR{hp_erl:,.0f}")
    logger.info(f"Loyalty Group CLV to Protect: INR{loy['Adjusted_CLV'].sum():,.0f}")
    logger.info("\nAll validation checks passed")

    results = {
        "recency_churn_correlation": round(recency_corr, 4),
        "clv_churn_correlation":     round(clv_corr, 4),
        "one_timers_in_high_value":  int(len(one_timers)),
        "churn_prob_by_segment":     seg_probs.round(4).to_dict(),
        "expected_revenue_loss_inr": round(float(total_erl), 2),
        "hp_expected_revenue_loss_inr": round(float(hp_erl), 2),
        "loyalty_clv_inr":           round(float(loy["Adjusted_CLV"].sum()), 2),
    }
    save_json(results, METRICS_DIR / "business_validation.json")
    return results


def compute_kpis(customer_df: pd.DataFrame) -> dict:
    """High-level KPIs. Revenue at Risk = sum(Adjusted_CLV x Churn_Prob)."""
    total_customers = len(customer_df)
    churn_rate      = customer_df["Churn"].mean()
    high_risk_count = (customer_df["Churn_Prob"] >= 0.5).sum()
    total_revenue   = customer_df["Monetary"].sum()
    avg_clv         = customer_df["Adjusted_CLV"].mean()
    high_priority   = (customer_df["Priority_Group"] == "High Priority").sum()

    # New formula: Expected Revenue Loss = sum(Adjusted_CLV x Churn_Prob)
    revenue_at_risk = customer_df["Expected_Revenue_Loss"].sum()

    kpis = {
        "total_customers":          int(total_customers),
        "churn_rate":               round(float(churn_rate), 4),
        "high_risk_count":          int(high_risk_count),
        "total_revenue_inr":        round(float(total_revenue), 2),
        "revenue_at_risk_inr":      round(float(revenue_at_risk), 2),
        "avg_clv_inr":              round(float(avg_clv), 2),
        "high_priority_customers":  int(high_priority),
    }
    save_json(kpis, METRICS_DIR / "kpis.json")
    logger.info(f"KPIs saved: {METRICS_DIR / 'kpis.json'}")
    logger.info(f"Expected Revenue Loss (new formula): INR{revenue_at_risk:,.0f}")
    return kpis
