import warnings
warnings.filterwarnings("ignore")
import streamlit as st
import sys
import os

# ── Ensure project root on path ───────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pathlib import Path
from dashboard.components.sidebar import render_sidebar
from dashboard.components.tables import outputs_exist


# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Retail Churn Analytics",
    page_icon="chart_with_upwards_trend",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load Custom CSS ───────────────────────────────────────────────────────────
css_path = Path("dashboard/styles/custom.css")
if css_path.exists():
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# ── Sidebar Navigation ────────────────────────────────────────────────────────
ready    = outputs_exist()
selected = render_sidebar(ready)


# ── Home Page ─────────────────────────────────────────────────────────────────
if selected == "Home":
    st.markdown("# Retail Customer Churn Prediction")
    st.markdown("### End-to-End ML Solution for Customer Retention")
    st.markdown("---")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("#### Business Problem")
        st.markdown(
            """
            In e-commerce and retail, acquiring a new customer costs **5–7× more**
            than retaining an existing one. Churn — a customer stopping purchases —
            directly erodes revenue and increases acquisition spend.

            This project builds a **production-ready ML pipeline** that:
            1. Predicts which customers are likely to churn using behavioral features + XGBoost
            2. Quantifies the revenue at risk using Customer Lifetime Value (CLV)
            3. Segments customers into actionable priority groups so retention
               efforts are targeted, not wasteful
            """
        )

        st.markdown("#### Dataset")
        st.markdown(
            """
            **UCI Online Retail Dataset** — 541,909 transactions from a UK-based
            online retailer (Dec 2010 – Dec 2011).

            | Attribute | Value |
            |-----------|-------|
            | Raw Transactions | 541,909 |
            | Clean Transactions | ~385,081 |
            | Unique Customers (obs.) | 3,565 |
            | Countries | 38 |
            | Currency | ₹ (£1 = ₹107, fixed 2010–11 rate) |
            """
        )

    with col2:
        st.markdown("#### Key Deliverables")
        deliverables = [
            "Churn probability score for every customer",
            "CLV-adjusted risk segmentation (4 priority groups)",
            "Recommended retention action per customer",
            "Revenue-at-risk quantification in ₹",
            "SHAP-based model explainability",
            "Cohort retention analysis",
        ]
        for d in deliverables:
            st.markdown(f"- {d}")

        st.markdown("#### Currency Note")
        st.info(
            "All monetary values are in Indian Rupees (₹). "
            "A fixed rate of ₹107 = £1 is applied — this approximates the "
            "GBP/INR rate during the dataset period (2010–2011) and is kept "
            "fixed for reproducibility."
        )

    st.markdown("---")

    # ── ML Workflow ───────────────────────────────────────────────────────
    st.markdown("#### ML Pipeline Workflow")
    workflow_cols = st.columns(8)
    steps = [
        ("1", "Data\nLoading"),
        ("2", "Cleaning &\nPreprocessing"),
        ("3", "Feature\nEngineering"),
        ("4", "RFM\nSegmentation"),
        ("5", "Model\nTraining"),
        ("6", "Evaluation &\nSHAP"),
        ("7", "Production\nScoring"),
        ("8", "CLV\nPriority"),
    ]
    for col, (num, label) in zip(workflow_cols, steps):
        with col:
            st.markdown(
                f"<div style='text-align:center;padding:0.5rem;background:#f0f9ff;"
                f"border:1px solid #bae6fd;border-radius:8px;font-size:0.78rem'>"
                f"<strong style='color:#1d4ed8'>{num}</strong><br>{label}</div>",
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # ── RFM Segments ──────────────────────────────────────────────────────
    st.markdown("#### Customer Segments")
    seg_cols = st.columns(5)
    segments = [
        ("Champions",  "#2ecc71", "R≥4, F≥4, M≥4 — Best customers. Recent, frequent, high-spend."),
        ("Loyal",      "#3498db", "F≥4 — Purchase frequently. Solid base."),
        ("At Risk",    "#e74c3c", "R≤2, F≥2 — Were good customers; haven't returned recently."),
        ("Dormant",    "#95a5a6", "R=1 — Haven't purchased in a very long time."),
        ("Others",     "#f39c12", "Remaining customers not fitting the above rules."),
    ]
    for col, (name, color, desc) in zip(seg_cols, segments):
        with col:
            st.markdown(
                f"<div style='border-left:4px solid {color};padding:0.5rem 0.75rem;"
                f"background:#f8fafc;border-radius:0 6px 6px 0;font-size:0.82rem'>"
                f"<strong>{name}</strong><br><span style='color:#64748b'>{desc}</span></div>",
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # ── Pipeline Status ───────────────────────────────────────────────────
    st.markdown("#### Setup Instructions")
    if not ready:
        st.warning("Pipeline outputs not found. Follow the steps below.")

    tab1, tab2 = st.tabs(["Setup & Run", "Project Structure"])
    with tab1:
        st.code(
            """# 1. Install dependencies
pip install -r requirements.txt

# 2. Place the dataset
#    Download from: https://archive.ics.uci.edu/ml/datasets/Online+Retail
#    Place at:      data/OnlineRetail.csv

# 3. Run the ML pipeline (from project root)
python pipelines/run_pipeline.py

# 4. Launch the dashboard
streamlit run app.py""",
            language="bash",
        )
    with tab2:
        st.code(
            """retail_churn/
├── app.py                    # Streamlit entry point
├── requirements.txt
├── README.md
├── data/                     # OnlineRetail.csv
├── models/                   # Saved model files
├── outputs/
│   ├── plots/                # All generated charts
│   ├── predictions/          # Customer predictions CSV
│   ├── metrics/              # Model metrics JSON
│   ├── reports/
│   └── logs/
├── pipelines/
│   └── run_pipeline.py       # Main pipeline script
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── segmentation.py
│   ├── train.py
│   ├── evaluation.py
│   ├── explainability.py
│   ├── predict.py
│   ├── business_metrics.py
│   ├── visualization.py
│   └── utils.py
└── dashboard/
    ├── pages/
    │   ├── executive_overview.py
    │   ├── customer_segmentation.py
    │   ├── predictive_intelligence.py
    │   └── retention_strategy.py
    ├── components/
    │   ├── kpi_cards.py
    │   ├── charts.py
    │   ├── sidebar.py
    │   └── tables.py
    └── styles/
        └── custom.css""",
            language="text",
        )

    st.markdown("---")
    st.markdown("#### Navigate to Dashboard Pages")
    cols = st.columns(4)
    pages = [
        ("Executive Overview",      "High-level KPIs, revenue at risk, churn by segment"),
        ("Customer Segmentation",   "RFM groups, CLV distribution, cohort retention"),
        ("Predictive Intelligence", "Model performance, SHAP importance, threshold analysis"),
        ("Retention Strategy",      "Action recommendations, priority matrix, campaign simulator"),
    ]
    for col, (name, desc) in zip(cols, pages):
        with col:
            st.markdown(
                f"<div style='background:#f8fafc;border:1px solid #e2e8f0;"
                f"border-radius:8px;padding:1rem;text-align:center;height:120px'>"
                f"<strong style='font-size:0.9rem'>{name}</strong><br>"
                f"<span style='font-size:0.78rem;color:#64748b'>{desc}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

# ── Dashboard Pages ───────────────────────────────────────────────────────────
elif selected == "Executive Overview":
    if not ready:
        st.error("Pipeline outputs not found. Run: `python pipelines/run_pipeline.py`")
    else:
        from dashboard.pages import executive_overview
        executive_overview.render()

elif selected == "Customer Segmentation":
    if not ready:
        st.error("Pipeline outputs not found. Run: `python pipelines/run_pipeline.py`")
    else:
        from dashboard.pages import customer_segmentation
        customer_segmentation.render()

elif selected == "Predictive Intelligence":
    if not ready:
        st.error("Pipeline outputs not found. Run: `python pipelines/run_pipeline.py`")
    else:
        from dashboard.pages import predictive_intelligence
        predictive_intelligence.render()

elif selected == "Retention Strategy":
    if not ready:
        st.error("Pipeline outputs not found. Run: `python pipelines/run_pipeline.py`")
    else:
        from dashboard.pages import retention_strategy
        retention_strategy.render()
