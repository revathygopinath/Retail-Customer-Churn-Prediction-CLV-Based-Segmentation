"""
run_pipeline.py — End-to-end ML pipeline execution script.

Run from the project root:
    python pipelines/run_pipeline.py

The script mirrors every step of the notebook in order, logs progress to the
terminal, and saves all outputs to the outputs/ folder.
"""

import sys
import os

# ── Windows UTF-8 fix: must run BEFORE any other import ─────────────────────
# Prevents UnicodeEncodeError for Rs-sign, arrows etc. on cp1252 terminals.
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")


import time
import traceback

# ── Ensure project root is on PYTHONPATH ─────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    RAW_DATA_PATH, FEATURES, FEATURE_CANDIDATES, MI_THRESHOLD,
    PLOTS_DIR, PREDICTIONS_DIR, METRICS_DIR, REPORTS_DIR, MODELS_DIR,
)
from src.utils import get_logger, ensure_output_dirs, print_section, save_json
from src.data_loader import load_raw_data, get_missing_summary
from src.preprocessing import clean_data, create_temporal_windows
from src.feature_engineering import build_customer_features
from src.segmentation import add_segments, add_priority_groups
from src.train import (
    select_features_mi, stratified_temporal_split,
    train_logistic_regression, train_random_forest,
    train_xgboost, train_final_model, save_model,
)
from src.evaluation import (
    compute_metrics, compare_models, diagnose_overfitting,
    threshold_analysis, save_all_metrics,
)
from src.explainability import (
    compute_shap, save_shap_summary_plot, save_shap_bar_plot,
    build_feature_importance_table,
)
from src.predict import score_customers, save_predictions
from src.business_metrics import run_business_validation, compute_kpis
from src.visualization import (
    plot_segment_overview, plot_rfm_distributions,
    plot_mutual_information, plot_kde_separation,
    plot_model_evaluation, plot_threshold_analysis,
    plot_churn_prob_distribution, plot_business_dashboard,
    plot_churn_trend, plot_cohort_retention, plot_clv_distribution,
)

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score


# ── Logger ────────────────────────────────────────────────────────────────────
logger = get_logger("pipeline")

GBP_TO_INR = 107


def run_pipeline():
    start_time = time.time()

    logger.info("=" * 65)
    logger.info("  RETAIL CHURN PREDICTION & CLV SEGMENTATION PIPELINE")
    logger.info("  UCI Online Retail Dataset  |  GBP1 = INR107")
    logger.info("=" * 65)

    ensure_output_dirs()

    # ── STEP 1 — Data Loading ─────────────────────────────────────────────
    print_section("STEP 1 — Loading Dataset")
    logger.info("Loading dataset...")
    if not RAW_DATA_PATH.exists():
        logger.error(
            f"Dataset not found at: {RAW_DATA_PATH}\n"
            "  Download from: https://archive.ics.uci.edu/ml/datasets/Online+Retail\n"
            "  Place the CSV file at: data/OnlineRetail.csv"
        )
        sys.exit(1)

    df = load_raw_data()
    missing_summary = get_missing_summary(df)
    logger.info(f"\nMissing Values:\n{missing_summary.to_string()}")

    # ── STEP 2 — Preprocessing ────────────────────────────────────────────
    print_section("STEP 2 — Data Cleaning")
    logger.info("Performing preprocessing...")
    df_clean = clean_data(df)

    # ── STEP 3 — Temporal Windows + INR ──────────────────────────────────
    print_section("STEP 3 — Temporal Window & INR Conversion")
    df_obs, df_future = create_temporal_windows(df_clean)

    # ── STEP 4 — Feature Engineering ─────────────────────────────────────
    print_section("STEP 4 — Feature Engineering")
    logger.info("Creating features...")
    customer_df = build_customer_features(df_obs, df_future)

    # ── STEP 5 — RFM Segmentation ─────────────────────────────────────────
    print_section("STEP 5 — RFM Scoring & Segmentation")
    customer_df = add_segments(customer_df)

    # ── STEP 6 — EDA Plots ────────────────────────────────────────────────
    print_section("STEP 6 — Exploratory Data Analysis (Saving Charts)")
    logger.info("Plotting segment overview...")
    plot_segment_overview(customer_df)
    logger.info("Plotting RFM distributions...")
    plot_rfm_distributions(customer_df)
    logger.info("Plotting churn trend...")
    plot_churn_trend(df_obs)
    logger.info("Plotting cohort retention heatmap...")
    plot_cohort_retention(df_obs)

    # RFM stats
    logger.info("\nKey stats (Monetary in INR):")
    logger.info(customer_df[["Recency", "Frequency", "Monetary"]].describe().round(2).to_string())

    # ── STEP 7 — Feature Selection ────────────────────────────────────────
    print_section("STEP 7 — Feature Selection via Mutual Information")
    mi_df, selected_features = select_features_mi(customer_df)
    plot_mutual_information(mi_df, threshold=MI_THRESHOLD)

    # KDE separation
    logger.info("Plotting KDE churn separation...")
    plot_kde_separation(customer_df, FEATURE_CANDIDATES)

    # ── STEP 8 — Train / Test Split ───────────────────────────────────────
    print_section("STEP 8 — Stratified Temporal Train / Test Split")
    X_train, X_test, y_train, y_test, train_df, test_df, customer_df = \
        stratified_temporal_split(customer_df, df_obs, selected_features)

    # ── STEP 9 — Model Training ───────────────────────────────────────────
    print_section("STEP 9 — Training Models")
    logger.info("Training churn model — Logistic Regression...")
    lr_model = train_logistic_regression(X_train, y_train)

    logger.info("Training churn model — Random Forest...")
    rf_model = train_random_forest(X_train, y_train)

    logger.info("Training churn model — XGBoost (regularised, 5-fold CV)...")
    xgb_model, cv_scores = train_xgboost(X_train, y_train)

    # Predictions on test set
    y_prob_lr  = lr_model.predict_proba(X_test)[:, 1]
    y_pred_lr  = lr_model.predict(X_test)
    y_prob_rf  = rf_model.predict_proba(X_test)[:, 1]
    y_pred_rf  = rf_model.predict(X_test)
    y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]
    y_pred_xgb = xgb_model.predict(X_test)

    logger.info(f"Logistic Regression AUC : {roc_auc_score(y_test, y_prob_lr):.4f}")
    logger.info(f"Random Forest AUC       : {roc_auc_score(y_test, y_prob_rf):.4f}")
    logger.info(f"XGBoost Test AUC        : {roc_auc_score(y_test, y_prob_xgb):.4f}")

    # ── STEP 10 — Model Evaluation ────────────────────────────────────────
    print_section("STEP 10 — Evaluating Model")
    model_results = [
        compute_metrics("Logistic Regression",    y_test, y_prob_lr,  y_pred_lr),
        compute_metrics("Random Forest",           y_test, y_prob_rf,  y_pred_rf),
        compute_metrics("XGBoost (regularised)",   y_test, y_prob_xgb, y_pred_xgb),
    ]
    compare_models(model_results)
    overfitting = diagnose_overfitting(xgb_model, X_train, y_train, X_test, y_test, cv_scores)

    thresh_df = threshold_analysis(y_test, y_prob_xgb)

    # Evaluation plots
    models_dict = {
        "Logistic Regression":   (y_prob_lr,  "#3498db"),
        "Random Forest":         (y_prob_rf,  "#f39c12"),
        "XGBoost (regularised)": (y_prob_xgb, "#e74c3c"),
    }
    logger.info("Saving evaluation plots...")
    plot_model_evaluation(y_test, models_dict, y_pred_xgb)
    plot_threshold_analysis(thresh_df)

    save_all_metrics(model_results, overfitting, thresh_df)

    # Save validation models
    save_model(xgb_model, "xgb_validation.pkl")
    save_model(lr_model,  "lr_model.pkl")
    save_model(rf_model,  "rf_model.pkl")

    # ── STEP 11 — SHAP Explainability ─────────────────────────────────────
    print_section("STEP 11 — Generating SHAP Explanations")
    from src.explainability import SHAP_AVAILABLE
    if not SHAP_AVAILABLE:
        logger.warning("SHAP not installed — SHAP plots will be skipped. Install with: pip install shap")
    logger.info("Generating SHAP explanations...")
    _, shap_values = compute_shap(xgb_model, X_test)
    save_shap_summary_plot(shap_values, X_test)
    save_shap_bar_plot(shap_values, X_test)
    X_all = customer_df[selected_features]
    y_all = customer_df["Churn"]
    feat_importance = build_feature_importance_table(xgb_model, shap_values, X_all, y_all, X_test)

    # ── STEP 12 — Production Model (Score All Customers) ──────────────────
    print_section("STEP 12 — Production Model: Score All Customers")
    logger.info("Training final production model on full dataset...")
    final_model = train_final_model(X_all, y_all)
    save_model(final_model, "xgb_final.pkl")
    customer_df = score_customers(customer_df, final_model)

    # ── STEP 13 — CLV & Priority Segmentation ────────────────────────────
    print_section("STEP 13 — CLV & Priority Segmentation")
    customer_df, clv_threshold = add_priority_groups(customer_df)

    # ── STEP 14 — Business Plots ──────────────────────────────────────────
    print_section("STEP 14 — Generating Business Visualizations")
    logger.info("Generating churn probability distribution...")
    plot_churn_prob_distribution(customer_df)
    logger.info("Generating CLV distribution...")
    plot_clv_distribution(customer_df)
    logger.info("Generating business intelligence dashboard...")
    plot_business_dashboard(customer_df, clv_threshold)

    # ── STEP 15 — Business Validation ────────────────────────────────────
    print_section("STEP 15 — Business & Model Validation")
    run_business_validation(customer_df)

    # ── STEP 16 — KPIs & Save Outputs ────────────────────────────────────
    print_section("STEP 16 — Computing KPIs & Saving Outputs")
    compute_kpis(customer_df)

    # Save processed data
    processed_path = PREDICTIONS_DIR / "customer_features_processed.csv"
    customer_df.to_csv(processed_path, index=False)
    logger.info(f"Processed customer data saved: {processed_path}")

    # Save segment-level summaries
    seg_summary = customer_df.groupby("Segment").agg(
        Customers     = ("CustomerID",   "count"),
        Churn_Rate    = ("Churn",        "mean"),
        Avg_ChurnProb = ("Churn_Prob",   "mean"),
        Avg_CLV       = ("Adjusted_CLV", "mean"),
        Total_Revenue = ("Monetary",     "sum"),
    ).round(2).reset_index()
    seg_summary.to_csv(PREDICTIONS_DIR / "segment_summary.csv", index=False)

    save_predictions(customer_df)

    # ── Done ──────────────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    logger.info("\n" + "=" * 65)
    logger.info(f"  PIPELINE COMPLETED SUCCESSFULLY in {elapsed:.1f}s")
    logger.info("  All outputs saved to: outputs/")
    logger.info("=" * 65)
    logger.info("\nNext step — launch the dashboard:")
    logger.info("  streamlit run app.py")


if __name__ == "__main__":
    try:
        run_pipeline()
    except Exception:
        logger.error("Pipeline failed with error:")
        traceback.print_exc()
        sys.exit(1)
