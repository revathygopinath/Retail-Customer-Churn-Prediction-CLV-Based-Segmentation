"""
data_loader.py — Raw data ingestion and initial exploration.
Mirrors Notebook Sections 2 (Data Loading & Initial Exploration).
"""

import pandas as pd
from pathlib import Path

from src.config import RAW_DATA_PATH, GBP_TO_INR
from src.utils import get_logger

logger = get_logger()


def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Load the UCI Online Retail CSV.

    Returns
    -------
    pd.DataFrame  — raw transactions with original column names.
    """
    logger.info(f"Loading dataset from: {path}")
    df = pd.read_csv(path, encoding="latin1")
    logger.info(f"Raw shape         : {df.shape}")
    logger.info(f"Date range        : {df['InvoiceDate'].min()} -> {df['InvoiceDate'].max()}")
    logger.info(f"Unique customers  : {df['CustomerID'].nunique():,}")
    logger.info(f"Unique products   : {df['StockCode'].nunique():,}")
    logger.info(f"Countries         : {df['Country'].nunique()}")
    return df


def get_missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame summarising missing values per column."""
    missing     = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    summary     = pd.DataFrame({"Missing": missing, "Pct (%)": missing_pct})
    return summary[summary["Missing"] > 0]
