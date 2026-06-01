"""
explainability.py — SHAP-based model explainability.
Mirrors Notebook Section 15 (Model Explainability — SHAP Values).

SHAP is an optional dependency. If it is not installed the functions
return gracefully with a warning rather than crashing the pipeline.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── SHAP optional import ──────────────────────────────────────────────────────
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

from src.config import FEATURES, FEATURE_CANDIDATES, PLOTS_DIR, METRICS_DIR, PLT_STYLE
from src.utils import get_logger, save_json

logger = get_logger()
plt.rcParams.update(PLT_STYLE)


def _shap_unavailable(fn_name: str):
    logger.warning(
        f"SHAP is not installed — skipping {fn_name}. "
        "Install with: pip install shap  or  conda install -c conda-forge shap"
    )


def compute_shap(model, X_test: pd.DataFrame):
    """Compute SHAP values using TreeExplainer. Returns (explainer, shap_values) or (None, None)."""
    if not SHAP_AVAILABLE:
        _shap_unavailable("compute_shap")
        return None, None

    logger.info("Computing SHAP values...")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    logger.info("SHAP values computed.")
    return explainer, shap_values


def save_shap_summary_plot(shap_values, X_test: pd.DataFrame) -> None:
    if not SHAP_AVAILABLE or shap_values is None:
        _shap_unavailable("save_shap_summary_plot")
        return

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.title(
        "SHAP Summary — Feature Impact on Churn Probability",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    out = PLOTS_DIR / "shap_summary.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {out}")


def save_shap_bar_plot(shap_values, X_test: pd.DataFrame) -> None:
    if not SHAP_AVAILABLE or shap_values is None:
        _shap_unavailable("save_shap_bar_plot")
        return

    plt.figure(figsize=(9, 5))
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.title(
        "Mean |SHAP Value| — Global Feature Importance",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    out = PLOTS_DIR / "shap_bar.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {out}")


def build_feature_importance_table(model, shap_values, X_all, y_all, X_test) -> pd.DataFrame:
    """
    Three-way feature importance comparison: MI Score | XGB Gain | SHAP Importance.
    Falls back to MI + XGB Gain only if SHAP is unavailable.
    """
    from sklearn.feature_selection import mutual_info_classif

    xgb_imp = model.get_booster().get_score(importance_type="gain")
    mi_vals  = dict(zip(FEATURES, mutual_info_classif(X_all, y_all, random_state=42)))

    if SHAP_AVAILABLE and shap_values is not None:
        shap_imp = dict(zip(FEATURES, np.abs(shap_values).mean(0)))
    else:
        # Use XGB gain normalised as a proxy when SHAP unavailable
        total_gain = sum(xgb_imp.values()) or 1.0
        shap_imp   = {f: xgb_imp.get(f, 0) / total_gain for f in FEATURES}
        logger.warning("SHAP unavailable — using normalised XGB Gain as importance proxy.")

    compare = pd.DataFrame({
        "Feature":          FEATURES,
        "MI_Score":         [mi_vals.get(f, 0)  for f in FEATURES],
        "XGB_Gain":         [xgb_imp.get(f, 0)  for f in FEATURES],
        "SHAP_Importance":  [shap_imp.get(f, 0) for f in FEATURES],
    }).sort_values("SHAP_Importance", ascending=False).reset_index(drop=True)

    logger.info("Feature Importance Comparison:")
    logger.info(compare.to_string(index=False))

    compare.to_csv(METRICS_DIR / "feature_importance.csv", index=False)
    logger.info(f"Feature importance saved to: {METRICS_DIR / 'feature_importance.csv'}")
    return compare
