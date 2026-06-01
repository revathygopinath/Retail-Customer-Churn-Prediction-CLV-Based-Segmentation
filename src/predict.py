"""
predict.py — Score all customers using the production model.
Mirrors Notebook Section 16 (Production Model — Score All Customers).
"""

import pandas as pd
import numpy as np

from src.config import FEATURES, PREDICTIONS_DIR, RISK_THRESHOLD
from src.utils import get_logger

logger = get_logger()


def score_customers(customer_df: pd.DataFrame, final_model) -> pd.DataFrame:
    """
    Add Churn_Prob column to customer_df using the production model
    trained on the full dataset (matches notebook Section 16).
    """
    X_all = customer_df[FEATURES]
    customer_df = customer_df.copy()
    customer_df["Churn_Prob"] = final_model.predict_proba(X_all)[:, 1]
    customer_df["Churn_Pred"] = (customer_df["Churn_Prob"] >= RISK_THRESHOLD).astype(int)

    logger.info("Churn Probability Distribution:")
    logger.info(customer_df["Churn_Prob"].describe().round(4).to_string())
    return customer_df


def save_predictions(customer_df: pd.DataFrame) -> None:
    """Save all predictions and high-risk subset."""
    # Full predictions
    cols = [
        "CustomerID", "Churn_Prob", "Churn_Pred", "Churn",
        "Segment", "Priority_Group", "Action", "Expected_Revenue_Loss",
        "Recency", "Frequency", "Monetary", "Adjusted_CLV",
    ]
    save_cols = [c for c in cols if c in customer_df.columns]
    all_path  = PREDICTIONS_DIR / "customer_predictions.csv"
    customer_df[save_cols].to_csv(all_path, index=False)
    logger.info(f"All predictions saved: {all_path}")

    # High-risk customers
    high_risk = customer_df[customer_df["Churn_Prob"] >= RISK_THRESHOLD][save_cols].sort_values(
        "Churn_Prob", ascending=False
    )
    hr_path = PREDICTIONS_DIR / "high_risk_customers.csv"
    high_risk.to_csv(hr_path, index=False)
    logger.info(f"High-risk customers ({len(high_risk):,}) saved: {hr_path}")

    # Segment-wise
    for seg in customer_df["Segment"].unique():
        seg_df   = customer_df[customer_df["Segment"] == seg][save_cols]
        seg_path = PREDICTIONS_DIR / f"segment_{seg.lower().replace(' ', '_')}.csv"
        seg_df.to_csv(seg_path, index=False)
    logger.info(f"Segment-wise CSVs saved to: {PREDICTIONS_DIR}")
