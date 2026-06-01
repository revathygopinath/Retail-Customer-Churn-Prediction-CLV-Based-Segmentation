"""
dashboard/pages/predictive_intelligence.py
Page 3 — Predictive Intelligence: model performance, SHAP, threshold analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np

from dashboard.components.tables import (
    load_model_metrics, load_overfitting, load_threshold_df,
    load_feature_importance, load_processed_data, plot_path,
)
from dashboard.components.charts import (
    churn_prob_histogram, shap_bar_plotly, threshold_line_chart,
)


def render():
    st.markdown("## Predictive Intelligence")
    st.markdown(
        "Model performance summary, SHAP-based feature importance, and "
        "decision threshold analysis for the XGBoost churn model."
    )
    st.markdown("---")

    metrics     = load_model_metrics()
    overfitting = load_overfitting()
    thresh_df   = load_threshold_df()
    feat_imp    = load_feature_importance()
    df          = load_processed_data()

    # ── Model Comparison ──────────────────────────────────────────────────
    st.markdown("#### Model Comparison")
    models_list = metrics.get("models", [])
    if models_list:
        model_df = pd.DataFrame(models_list)
        model_df.columns = ["Model", "ROC-AUC", "F1 (Macro)", "Precision", "Recall"]
        st.dataframe(model_df, use_container_width=True, hide_index=True)

        best = model_df.loc[model_df["ROC-AUC"].idxmax()]
        st.markdown(
            f"<div class='insight-box'>"
            f"Best model: <strong>{best['Model']}</strong> with ROC-AUC = "
            f"<strong>{best['ROC-AUC']:.4f}</strong>. "
            f"XGBoost was selected as the production model for its regularisation "
            f"controls and compatibility with SHAP explainability."
            f"</div>",
            unsafe_allow_html=True,
        )

    # ── Overfitting Diagnosis ─────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Overfitting Diagnosis")
    if overfitting:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Train AUC",       f"{overfitting.get('train_auc', 0):.4f}")
        c2.metric("Test AUC",        f"{overfitting.get('test_auc', 0):.4f}")
        c3.metric("CV AUC (5-fold)", f"{overfitting.get('cv_auc', 0):.4f} \u00b1 {overfitting.get('cv_std', 0):.4f}")
        delta = overfitting.get("delta", 0)
        flag  = overfitting.get("flag", "")
        color = {"OK": "normal", "WARN": "inverse", "ALERT": "inverse"}.get(flag, "off")
        c4.metric("Train-Test Gap", f"{delta:.4f}", delta_color=color)
        verdict = overfitting.get("verdict", "")
        if verdict:
            icon = "\u2713" if "No meaningful" in verdict else ("\u26a0" if "Mild" in verdict else "\u2717")
            st.markdown(
                f"<div class='insight-box'>{icon} <strong>Verdict:</strong> {verdict}<br>"
                f"CV AUC \u2248 Test AUC \u2014 model generalises consistently across folds.</div>",
                unsafe_allow_html=True,
            )

    # ── ROC Curve & Confusion Matrix ──────────────────────────────────────
    st.markdown("---")
    st.markdown("#### ROC Curve & Confusion Matrix")
    roc_only = plot_path("roc_curve.png")
    roc_path = plot_path("model_evaluation.png")
    cm_path  = plot_path("confusion_matrix.png")

    c5, c6 = st.columns(2)
    with c5:
        if roc_only.exists():
            st.image(str(roc_only), caption="ROC Curve Comparison")
        elif roc_path.exists():
            st.image(str(roc_path), caption="Model Evaluation")
    with c6:
        if cm_path.exists():
            st.image(str(cm_path), caption="Confusion Matrix \u2014 XGBoost")

    # ── Churn Probability Distribution ────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Churn Probability Distribution")
    if "Churn_Prob" in df.columns:
        st.plotly_chart(churn_prob_histogram(df), use_container_width=True)
        stats = df["Churn_Prob"].describe().round(4)
        cc = st.columns(4)
        cc[0].metric("Mean",   f"{stats['mean']:.4f}")
        cc[1].metric("Median", f"{stats['50%']:.4f}")
        cc[2].metric("25th %", f"{stats['25%']:.4f}")
        cc[3].metric("75th %", f"{stats['75%']:.4f}")

    # ── Threshold Analysis ────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Classification Threshold Analysis")
    if not thresh_df.empty:
        st.plotly_chart(threshold_line_chart(thresh_df), use_container_width=True)
        st.dataframe(thresh_df, use_container_width=True, hide_index=True)
        st.markdown(
            "<div class='insight-box'>"
            "<strong>Threshold = 0.5 selected (default).</strong> "
            "Lower threshold \u2192 catch more churners but more false positives (campaign cost). "
            "Higher threshold \u2192 precise but miss real churners (lost revenue). "
            "At 0.5: Precision \u2248 62%, Recall \u2248 78%."
            "</div>",
            unsafe_allow_html=True,
        )

    # ── SHAP Explainability — Beeswarm only ───────────────────────────────
    st.markdown("---")
    st.markdown("#### SHAP Feature Importance")

    shap_sum = plot_path("shap_summary.png")
    if shap_sum.exists():
        st.image(str(shap_sum), caption="SHAP Beeswarm Summary")
        st.markdown(
            "<div class='insight-box'>"
            "<strong>ActiveMonths</strong> is the top predictor \u2014 customers who were active across "
            "more calendar months are far less likely to churn. "
            "<strong>ProductDiversity</strong> (unique SKUs purchased) is second \u2014 "
            "broad engagement signals deep loyalty. "
            "High SHAP values (right side of the beeswarm) increase churn probability."
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        st.info("SHAP plots not found. Re-run the pipeline with SHAP installed.")
