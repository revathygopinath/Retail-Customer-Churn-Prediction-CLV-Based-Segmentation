"""
preprocessing.py — Data cleaning and temporal window definition.
Mirrors Notebook Sections 3 (Data Cleaning) and 4 (Temporal Window + INR).

Cleaning steps (exact notebook order):
  1. Parse InvoiceDate, cast CustomerID / InvoiceNo / StockCode types
  2. Drop missing CustomerID
  3. Remove cancelled invoices (InvoiceNo starts with 'C')
  4. Remove rows with Quantity <= 0 or UnitPrice <= 0
  5. Cap 99th-percentile outliers for Quantity and UnitPrice
  6. Drop exact duplicate rows

Temporal split:
  Observation window : Dec 2010 – Sep 2011  (features)
  Prediction window  : Oct – Dec 2011       (churn label)

INR conversion is applied to the observation window BEFORE feature engineering
so every downstream INR figure is consistent (£1 = INR107).
"""

import pandas as pd
import numpy as np
from pathlib import Path

from src.config import CUTOFF_DATE, GBP_TO_INR
from src.utils import get_logger

logger = get_logger()


# ── Cleaning ──────────────────────────────────────────────────────────────────

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all cleaning steps from the notebook and return clean DataFrame.
    Logs row counts removed at each step.
    """
    df_clean = df.copy()

    # Type casts
    df_clean["InvoiceDate"] = pd.to_datetime(df_clean["InvoiceDate"], errors="coerce")
    df_clean["CustomerID"]  = df_clean["CustomerID"].astype("Int64")
    df_clean["InvoiceNo"]   = df_clean["InvoiceNo"].astype(str)
    df_clean["StockCode"]   = df_clean["StockCode"].astype(str)

    cleaning_steps = [
        ("Drop missing CustomerID",    lambda d: d.dropna(subset=["CustomerID"])),
        ("Remove cancellations",       lambda d: d[~d["InvoiceNo"].str.startswith("C")]),
        ("Remove invalid qty/price",   lambda d: d[(d["Quantity"] > 0) & (d["UnitPrice"] > 0)]),
    ]

    for label, fn in cleaning_steps:
        before     = len(df_clean)
        df_clean   = fn(df_clean)
        removed    = before - len(df_clean)
        logger.info(f"  {label:<35}: {removed:,} rows removed")

    # 99th-percentile outlier cap
    qty_cap   = df_clean["Quantity"].quantile(0.99)
    price_cap = df_clean["UnitPrice"].quantile(0.99)
    before    = len(df_clean)
    df_clean  = df_clean[
        (df_clean["Quantity"]   <= qty_cap) &
        (df_clean["UnitPrice"]  <= price_cap)
    ]
    logger.info(
        f"  {'99th-pct outlier cap':<35}: {before - len(df_clean):,} rows "
        f"(qty<={qty_cap:.0f}, price<={price_cap:.2f})"
    )

    # Drop exact duplicates
    before   = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    logger.info(f"  {'Drop exact duplicates':<35}: {before - len(df_clean):,} rows removed")

    logger.info(f"Clean shape       : {df_clean.shape}")
    logger.info(f"Remaining customers: {df_clean['CustomerID'].nunique():,}")
    logger.info(
        f"Date range        : {df_clean['InvoiceDate'].min().date()} -> "
        f"{df_clean['InvoiceDate'].max().date()}"
    )
    return df_clean


# ── Temporal Split + INR Conversion ──────────────────────────────────────────

def create_temporal_windows(df_clean: pd.DataFrame):
    """
    Split the clean DataFrame into observation and future windows.
    Applies GBP -> INR conversion and computes TotalPrice on the
    observation window only (matching the notebook exactly).

    Returns
    -------
    df_obs    : observation window with UnitPrice and TotalPrice in INR
    df_future : prediction window (raw GBP — only used to derive churn label)
    """
    cutoff  = pd.to_datetime(CUTOFF_DATE)
    df_obs    = df_clean[df_clean["InvoiceDate"] <= cutoff].copy()
    df_future = df_clean[df_clean["InvoiceDate"] >  cutoff].copy()

    # Apply INR conversion BEFORE feature engineering
    df_obs["UnitPrice"]  = df_obs["UnitPrice"] * GBP_TO_INR
    df_obs["TotalPrice"] = df_obs["Quantity"]  * df_obs["UnitPrice"]

    logger.info(
        f"Observation window: {df_obs['InvoiceDate'].min().date()} -> "
        f"{df_obs['InvoiceDate'].max().date()}  |  "
        f"{df_obs['CustomerID'].nunique():,} customers"
    )
    logger.info(
        f"Future window     : {df_future['InvoiceDate'].min().date()} -> "
        f"{df_future['InvoiceDate'].max().date()}  |  "
        f"{df_future['CustomerID'].nunique():,} customers"
    )
    return df_obs, df_future
