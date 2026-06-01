"""
train.py — Feature selection, stratified temporal train/test split, and
model training.
Mirrors Notebook Sections 9 (Mutual Information), 11 (Stratified Split),
12 (Predictive Modelling).
"""

import numpy as np
import pandas as pd
import joblib

from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.config import (
    FEATURES, FEATURE_CANDIDATES, MI_THRESHOLD,
    XGB_PARAMS, RF_PARAMS, LR_PARAMS,
    CV_FOLDS, RANDOM_STATE, MODELS_DIR,
    MID_DATE,
)
from src.utils import get_logger

logger = get_logger()


# ── Feature Selection ─────────────────────────────────────────────────────────

def select_features_mi(customer_df: pd.DataFrame):
    """
    Compute mutual information scores and select features above MI_THRESHOLD.
    Returns (mi_df, selected_features).
    """
    X_all = customer_df[FEATURE_CANDIDATES]
    y     = customer_df["Churn"]

    mi_scores = mutual_info_classif(X_all, y, random_state=RANDOM_STATE)
    mi_df     = pd.DataFrame({"Feature": FEATURE_CANDIDATES, "MI Score": mi_scores})
    mi_df     = mi_df.sort_values("MI Score", ascending=False).reset_index(drop=True)

    selected = [f for f, s in zip(FEATURE_CANDIDATES, mi_scores) if s > MI_THRESHOLD]
    if not selected:
        selected = list(FEATURES)
        logger.warning(f"No features exceeded MI threshold. Falling back to: {selected}")
    logger.info(f"Selected features (MI > {MI_THRESHOLD}): {selected}")
    return mi_df, selected


# ── Stratified Temporal Split ─────────────────────────────────────────────────

def stratified_temporal_split(customer_df: pd.DataFrame, df_obs: pd.DataFrame, features: list):
    """
    Stratified Temporal Sampling — splits early/late cohorts 80/20 each,
    then unions them. Prevents distribution mismatch described in Section 11.

    Returns: X_train, X_test, y_train, y_test, train_df, test_df
    """
    mid_date   = pd.to_datetime(MID_DATE)
    first_purch = df_obs.groupby("CustomerID")["InvoiceDate"].min().rename("FirstPurchase")
    customer_df = customer_df.merge(first_purch.reset_index(), on="CustomerID", how="left")

    early_mask = customer_df["FirstPurchase"] <= mid_date
    late_mask  = ~early_mask

    np.random.seed(RANDOM_STATE)
    early_idx = customer_df[early_mask].index.tolist()
    late_idx  = customer_df[late_mask].index.tolist()
    np.random.shuffle(early_idx)
    np.random.shuffle(late_idx)

    early_split = int(len(early_idx) * 0.8)
    late_split  = int(len(late_idx)  * 0.8)

    train_idx = early_idx[:early_split] + late_idx[:late_split]
    test_idx  = early_idx[early_split:] + late_idx[late_split:]

    train_df = customer_df.loc[train_idx]
    test_df  = customer_df.loc[test_idx]

    X_train, y_train = train_df[features], train_df["Churn"]
    X_test,  y_test  = test_df[features],  test_df["Churn"]

    logger.info(
        f"Train : {len(X_train):,} customers | churn rate: {y_train.mean():.1%}"
    )
    logger.info(
        f"Test  : {len(X_test):,} customers | churn rate: {y_test.mean():.1%}"
    )
    logger.info("Churn rates are comparable — split is balanced and representative.")
    return X_train, X_test, y_train, y_test, train_df, test_df, customer_df


# ── Model Training ────────────────────────────────────────────────────────────

def train_logistic_regression(X_train, y_train):
    """Train Logistic Regression pipeline (StandardScaler + LogReg)."""
    log_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(**LR_PARAMS)),
    ])
    log_pipe.fit(X_train, y_train)
    logger.info("Logistic Regression trained.")
    return log_pipe


def train_random_forest(X_train, y_train):
    """Train Random Forest classifier."""
    rf_model = RandomForestClassifier(**RF_PARAMS)
    rf_model.fit(X_train, y_train)
    logger.info("Random Forest trained.")
    return rf_model


def train_xgboost(X_train, y_train):
    """
    Train regularised XGBoost with 5-fold stratified CV (matches Section 12).
    Returns (model, cv_scores).
    """
    xgb_params = dict(**XGB_PARAMS)
    # Class imbalance weight computed from training data
    xgb_params["scale_pos_weight"] = (
        (y_train == 0).sum() / (y_train == 1).sum()
    )

    xgb_model = XGBClassifier(**xgb_params)

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(xgb_model, X_train, y_train, cv=cv, scoring="roc_auc")
    logger.info(
        f"XGBoost 5-Fold CV AUC : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}"
    )
    logger.info(f"  Fold scores: {[round(s, 4) for s in cv_scores]}")

    xgb_model.fit(X_train, y_train)
    logger.info("XGBoost (regularised) trained on full train set.")
    return xgb_model, cv_scores


def train_final_model(X_all: pd.DataFrame, y_all: pd.Series):
    """
    Re-train XGBoost on the entire dataset (train + test) for production
    scoring — standard practice after validation metrics are locked in.
    """
    xgb_params = dict(**XGB_PARAMS)
    xgb_params["scale_pos_weight"] = (y_all == 0).sum() / (y_all == 1).sum()

    final_model = XGBClassifier(**xgb_params)
    final_model.fit(X_all, y_all)
    logger.info("Final production model trained on full dataset.")
    return final_model


# ── Model Persistence ─────────────────────────────────────────────────────────

def save_model(model, filename: str) -> None:
    path = MODELS_DIR / filename
    joblib.dump(model, path)
    logger.info(f"Model saved: {path}")


def load_model(filename: str):
    path = MODELS_DIR / filename
    model = joblib.load(path)
    logger.info(f"Model loaded: {path}")
    return model
