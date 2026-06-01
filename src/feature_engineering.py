"""
feature_engineering.py — RFM computation, behavioural features, churn label.
Mirrors Notebook Sections 5 (RFM & Behavioral Features) and 6 (Churn Label).

Features built:
  Core RFM:
    Recency             — days since last purchase (lower = more recent)
    Frequency           — distinct invoices
    Monetary            — total spend (₹)

  Behavioural:
    ActiveDays          — distinct calendar days with a purchase
    ActiveMonths        — distinct calendar months active
    CustomerAge         — days from first purchase to snapshot
    AvgBasketSize       — mean invoice total (₹)
    ProductDiversity    — unique SKUs purchased
    AvgPurchaseInterval — mean gap (days) between consecutive purchases

  Log-transforms (right-skewed features):
    Monetary_log, Frequency_log, AvgBasketSize_log, AvgPurchaseInterval_log

Churn label:
    Churn = 1 if customer absent from future window, else 0
"""

import pandas as pd
import numpy as np

from src.config import CUTOFF_DATE
from src.utils import get_logger

logger = get_logger()


# ── RFM Core ──────────────────────────────────────────────────────────────────

def build_rfm(df_obs: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Recency, Frequency, Monetary per customer.
    snapshot_date = max(InvoiceDate) + 1 day  (matches notebook).
    """
    snapshot_date = df_obs["InvoiceDate"].max() + pd.Timedelta(days=1)
    logger.info(f"Snapshot date: {snapshot_date.date()}")

    rfm = df_obs.groupby("CustomerID").agg(
        Recency   = ("InvoiceDate",  lambda x: (snapshot_date - x.max()).days),
        Frequency = ("InvoiceNo",    "nunique"),
        Monetary  = ("TotalPrice",   "sum"),
    ).reset_index()

    logger.info(f"RFM shape: {rfm.shape}")
    return rfm, snapshot_date


# ── Behavioural Features ──────────────────────────────────────────────────────

def build_behavioral_features(df_obs: pd.DataFrame, rfm: pd.DataFrame, snapshot_date) -> pd.DataFrame:
    """
    Adds the 6 behavioural features described in the notebook docstring.
    Single-purchase customers get AvgPurchaseInterval = CustomerAge (conservative proxy).
    """
    df = df_obs.copy()
    df["InvoiceDay"] = df["InvoiceDate"].dt.date
    df["YearMonth"]  = df["InvoiceDate"].dt.to_period("M")

    active_days   = df.groupby("CustomerID")["InvoiceDay"].nunique().rename("ActiveDays")
    active_months = df.groupby("CustomerID")["YearMonth"].nunique().rename("ActiveMonths")
    first_purch   = df.groupby("CustomerID")["InvoiceDate"].min().rename("FirstPurchase_")
    customer_age  = ((snapshot_date - first_purch).dt.days).rename("CustomerAge")

    inv_totals    = df.groupby(["CustomerID", "InvoiceNo"])["TotalPrice"].sum()
    avg_basket    = inv_totals.groupby(level="CustomerID").mean().rename("AvgBasketSize")

    product_div   = df.groupby("CustomerID")["StockCode"].nunique().rename("ProductDiversity")

    df_s           = df.sort_values(["CustomerID", "InvoiceDate"])
    df_s["PrevPurchase"] = df_s.groupby("CustomerID")["InvoiceDate"].shift(1)
    df_s["PurchaseGap"]  = (df_s["InvoiceDate"] - df_s["PrevPurchase"]).dt.days
    avg_gap        = df_s.groupby("CustomerID")["PurchaseGap"].mean().rename("AvgPurchaseInterval")

    feats = rfm.set_index("CustomerID")
    for s in [active_days, active_months, customer_age, avg_basket, product_div, avg_gap]:
        feats = feats.join(s, how="left")

    return feats.reset_index()


# ── Log-Transforms ────────────────────────────────────────────────────────────

def add_log_transforms(customer_df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply log1p transforms to right-skewed columns (matches notebook).
    """
    customer_df = customer_df.copy()
    # Single-purchase gap fix — use CustomerAge as conservative proxy
    customer_df["AvgPurchaseInterval"] = customer_df["AvgPurchaseInterval"].fillna(
        customer_df["CustomerAge"]
    )
    for col in ["Monetary", "Frequency", "AvgBasketSize", "AvgPurchaseInterval"]:
        customer_df[f"{col}_log"] = np.log1p(customer_df[col])
    return customer_df


# ── Churn Label ───────────────────────────────────────────────────────────────

def create_churn_label(customer_df: pd.DataFrame, df_future: pd.DataFrame) -> pd.DataFrame:
    """
    Churn = 1 if a customer does not appear in the future window.
    """
    future_customers = df_future["CustomerID"].unique()
    customer_df = customer_df.copy()
    customer_df["Churn"] = (~customer_df["CustomerID"].isin(future_customers)).astype(int)

    dist = customer_df["Churn"].value_counts(normalize=True).mul(100).round(1)
    logger.info(f"Retained (0): {dist.get(0, 0):.1f}%  |  Churned (1): {dist.get(1, 0):.1f}%")
    logger.info("Classes roughly balanced — no aggressive resampling needed")
    return customer_df


# ── Master Builder ────────────────────────────────────────────────────────────

def build_customer_features(df_obs: pd.DataFrame, df_future: pd.DataFrame) -> pd.DataFrame:
    """
    Orchestrates all feature-engineering steps and returns the full
    customer-level feature matrix with churn label.
    """
    rfm, snapshot_date  = build_rfm(df_obs)
    customer_df         = build_behavioral_features(df_obs, rfm, snapshot_date)
    customer_df         = add_log_transforms(customer_df)
    customer_df         = create_churn_label(customer_df, df_future)
    logger.info(f"Feature matrix shape : {customer_df.shape}")
    logger.info(f"Columns: {customer_df.columns.tolist()}")
    return customer_df
