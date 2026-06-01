"""
config.py — Central configuration for the Retail Churn ML Pipeline.
All constants, paths, model parameters, and business rules live here.
"""

import os
from pathlib import Path

# ── Project Root ──────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent

# ── Data Paths ────────────────────────────────────────────────────────────────
DATA_DIR       = ROOT_DIR / "data"
RAW_DATA_PATH  = DATA_DIR / "OnlineRetail.csv"

# ── Output Paths ──────────────────────────────────────────────────────────────
OUTPUTS_DIR      = ROOT_DIR / "outputs"
PLOTS_DIR        = OUTPUTS_DIR / "plots"
REPORTS_DIR      = OUTPUTS_DIR / "reports"
PREDICTIONS_DIR  = OUTPUTS_DIR / "predictions"
METRICS_DIR      = OUTPUTS_DIR / "metrics"
LOGS_DIR         = OUTPUTS_DIR / "logs"
MODELS_DIR       = ROOT_DIR / "models"

# ── Currency ──────────────────────────────────────────────────────────────────
GBP_TO_INR = 107  # Fixed 2010–2011 period rate for reproducibility

# ── Temporal Windows ──────────────────────────────────────────────────────────
CUTOFF_DATE = "2011-09-30"   # observation window end / prediction window start
MID_DATE    = "2011-06-30"   # early/late cohort split for stratified sampling

# ── Feature Engineering ───────────────────────────────────────────────────────
FEATURE_CANDIDATES = [
    "ActiveDays",
    "Frequency_log",
    "ActiveMonths",
    "Monetary_log",
    "ProductDiversity",
    "AvgPurchaseInterval",
    "Recency",
]

MI_THRESHOLD = 0.07   # minimum mutual information score to include a feature

# Final selected features (those with MI > threshold)
FEATURES = [
    "ActiveDays",
    "Frequency_log",
    "ActiveMonths",
    "Monetary_log",
    "ProductDiversity",
    "AvgPurchaseInterval",
]

# ── RFM Segmentation Rules ────────────────────────────────────────────────────
SEGMENT_COLORS = {
    "Champions": "#2ecc71",
    "Loyal":     "#3498db",
    "At Risk":   "#e74c3c",
    "Dormant":   "#95a5a6",
    "Others":    "#f39c12",
}

# ── Model Parameters ──────────────────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE    = 0.20
CV_FOLDS     = 5

XGB_PARAMS = dict(
    n_estimators      = 300,
    max_depth         = 3,
    learning_rate     = 0.05,
    subsample         = 0.8,
    colsample_bytree  = 0.8,
    min_child_weight  = 5,
    gamma             = 0.1,
    reg_alpha         = 0.1,
    reg_lambda        = 1.0,
    eval_metric       = "logloss",
    random_state      = RANDOM_STATE,
)

RF_PARAMS = dict(
    n_estimators    = 300,
    max_depth       = 8,
    min_samples_leaf= 10,
    class_weight    = "balanced",
    random_state    = RANDOM_STATE,
)

LR_PARAMS = dict(
    max_iter      = 1000,
    class_weight  = "balanced",
    random_state  = RANDOM_STATE,
)

# ── Business Logic ────────────────────────────────────────────────────────────
RISK_THRESHOLD    = 0.50   # churn probability cutoff for High Risk
CLV_PERCENTILE    = 0.65   # percentile above which a customer is "High Value"
ONE_TIMER_PENALTY = 0.05   # CLV discount for single-purchase customers

ACTIONS = {
    "High Priority": "Immediate retention offer (discount / VIP upgrade)",
    "Loyalty":       "Loyalty rewards & upsell campaign",
    "Nurture":       "Low-cost re-engagement (email / push)",
    "Low Priority":  "No immediate action — monitor",
}

PRIORITY_COLORS = {
    "High Priority": "#e74c3c",
    "Loyalty":       "#2ecc71",
    "Nurture":       "#f39c12",
    "Low Priority":  "#95a5a6",
}

# ── Plot Style ────────────────────────────────────────────────────────────────
PLT_STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor":   "#f8f9fa",
    "axes.grid":        True,
    "grid.alpha":       0.4,
    "font.family":      "DejaVu Sans",
    "axes.titlesize":   13,
    "axes.titleweight": "bold",
}

PALETTE = ["#2ecc71", "#e74c3c", "#3498db", "#f39c12"]
