"""
dashboard/components/tables.py — Data loading helpers for the Streamlit dashboard.
All functions load from outputs/ — the dashboard never retrains the model.
"""

import json
import pandas as pd
import streamlit as st
from pathlib import Path

from src.config import PREDICTIONS_DIR, METRICS_DIR, PLOTS_DIR

OUTPUTS_DIR = Path("outputs")


def outputs_exist() -> bool:
    """Return True if the minimum required outputs are present."""
    required = [
        PREDICTIONS_DIR / "customer_predictions.csv",
        METRICS_DIR / "model_metrics.json",
        METRICS_DIR / "kpis.json",
    ]
    return all(p.exists() for p in required)


@st.cache_data(ttl=300)
def load_predictions() -> pd.DataFrame:
    return pd.read_csv(PREDICTIONS_DIR / "customer_predictions.csv")


@st.cache_data(ttl=300)
def load_processed_data() -> pd.DataFrame:
    path = PREDICTIONS_DIR / "customer_features_processed.csv"
    if path.exists():
        return pd.read_csv(path)
    return load_predictions()


@st.cache_data(ttl=300)
def load_kpis() -> dict:
    with open(METRICS_DIR / "kpis.json") as f:
        return json.load(f)


@st.cache_data(ttl=300)
def load_model_metrics() -> dict:
    with open(METRICS_DIR / "model_metrics.json") as f:
        return json.load(f)


@st.cache_data(ttl=300)
def load_overfitting() -> dict:
    path = METRICS_DIR / "overfitting_diagnosis.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


@st.cache_data(ttl=300)
def load_threshold_df() -> pd.DataFrame:
    path = METRICS_DIR / "threshold_analysis.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data(ttl=300)
def load_feature_importance() -> pd.DataFrame:
    path = METRICS_DIR / "feature_importance.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data(ttl=300)
def load_segment_summary() -> pd.DataFrame:
    path = PREDICTIONS_DIR / "segment_summary.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data(ttl=300)
def load_business_validation() -> dict:
    path = METRICS_DIR / "business_validation.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def plot_path(name: str) -> Path:
    return PLOTS_DIR / name
