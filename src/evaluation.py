"""
evaluation.py — Model evaluation, overfitting diagnosis, and threshold analysis.
Mirrors Notebook Sections 13 (Model Evaluation & Overfitting Diagnosis)
and 14 (Classification Threshold Analysis).
"""

import numpy as np
import pandas as pd
import json

from sklearn.metrics import (
    roc_auc_score, roc_curve,
    confusion_matrix, f1_score, precision_score, recall_score,
)

from src.config import METRICS_DIR
from src.utils import get_logger, save_json

logger = get_logger()


# ── Per-Model Metrics ─────────────────────────────────────────────────────────

def compute_metrics(name: str, y_test, y_prob, y_pred) -> dict:
    """Return a dict of evaluation metrics for one model."""
    return {
        "model":     name,
        "roc_auc":   round(roc_auc_score(y_test, y_prob), 4),
        "f1_macro":  round(f1_score(y_test, y_pred, average="macro"), 4),
        "precision": round(precision_score(y_test, y_pred, average="macro"), 4),
        "recall":    round(recall_score(y_test, y_pred, average="macro"), 4),
    }


def compare_models(results: list) -> pd.DataFrame:
    """
    Print and return a comparison table.
    `results` is a list of dicts from compute_metrics().
    """
    df = pd.DataFrame(results)
    logger.info("\n" + "=" * 58)
    logger.info("MODEL COMPARISON SUMMARY")
    logger.info("=" * 58)
    for row in results:
        logger.info(f"\n{row['model']}")
        logger.info(f"  ROC-AUC   : {row['roc_auc']}")
        logger.info(f"  F1 (macro): {row['f1_macro']}")
        logger.info(f"  Precision : {row['precision']}")
        logger.info(f"  Recall    : {row['recall']}")
    return df


# ── Overfitting Diagnosis ─────────────────────────────────────────────────────

def diagnose_overfitting(xgb_model, X_train, y_train, X_test, y_test, cv_scores) -> dict:
    """Compute train/test AUC gap and print overfitting verdict (notebook logic)."""
    train_auc = roc_auc_score(y_train, xgb_model.predict_proba(X_train)[:, 1])
    test_auc  = roc_auc_score(y_test,  xgb_model.predict_proba(X_test)[:, 1])
    delta     = train_auc - test_auc
    cv_mean   = cv_scores.mean()

    logger.info("\n" + "=" * 58)
    logger.info("OVERFITTING DIAGNOSIS")
    logger.info("=" * 58)
    logger.info(f"  Train AUC         : {train_auc:.4f}")
    logger.info(f"  Test AUC          : {test_auc:.4f}")
    logger.info(f"  Delta (train-test): {delta:.4f}")
    logger.info(f"  CV AUC (5-fold)   : {cv_mean:.4f} ± {cv_scores.std():.4f}")

    if delta < 0.03:
        verdict = "No meaningful overfitting — train/test gap < 0.03"
        flag    = "OK"
    elif delta < 0.06:
        verdict = "Mild overfitting — gap 0.03–0.06, acceptable for tabular data"
        flag    = "WARN"
    else:
        verdict = "Overfitting detected — gap > 0.06, further regularisation needed"
        flag    = "ALERT"

    logger.info(f"\n  Verdict: {verdict}")
    logger.info("  Interpretation:")
    logger.info("  - CV AUC ~ Test AUC -> model generalises consistently across folds")
    logger.info("  - If Test AUC > Train AUC, suspect cohort mismatch in split (fixed)")

    result = {
        "train_auc": round(train_auc, 4),
        "test_auc":  round(test_auc, 4),
        "delta":     round(delta, 4),
        "cv_auc":    round(cv_mean, 4),
        "cv_std":    round(cv_scores.std(), 4),
        "verdict":   verdict,
        "flag":      flag,
    }
    return result


# ── Threshold Analysis ────────────────────────────────────────────────────────

def threshold_analysis(y_test, y_prob_xgb) -> pd.DataFrame:
    """
    Sweep decision thresholds 0.30 → 0.70 and compute Precision/Recall/F1.
    Matches notebook Section 14 exactly.
    """
    records = []
    for t in np.arange(0.3, 0.75, 0.05):
        preds = (y_prob_xgb >= t).astype(int)
        records.append({
            "Threshold":       round(t, 2),
            "Precision":       round(precision_score(y_test, preds, zero_division=0), 4),
            "Recall":          round(recall_score(y_test, preds, zero_division=0), 4),
            "F1":              round(f1_score(y_test, preds, zero_division=0), 4),
            "Churners_Caught": int(preds.sum()),
        })
    thresh_df = pd.DataFrame(records)
    logger.info("\nThreshold Analysis:")
    logger.info(thresh_df.to_string(index=False))
    logger.info("\nSelected threshold: 0.5")
    return thresh_df


# ── Save Metrics ──────────────────────────────────────────────────────────────

def save_all_metrics(model_results: list, overfitting: dict, thresh_df: pd.DataFrame) -> None:
    """Persist model metrics and threshold analysis to outputs/metrics/."""
    metrics_path   = METRICS_DIR / "model_metrics.json"
    overfitting_path = METRICS_DIR / "overfitting_diagnosis.json"
    thresh_path    = METRICS_DIR / "threshold_analysis.csv"

    save_json({"models": model_results, "selected_threshold": 0.5}, metrics_path)
    save_json(overfitting, overfitting_path)
    thresh_df.to_csv(thresh_path, index=False)

    logger.info(f"Metrics saved to: {METRICS_DIR}")
