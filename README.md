# Retail Customer Churn Prediction & CLV-Based Segmentation

> **End-to-End Machine Learning Solution for Customer Retention**  
> Dataset: UCI Online Retail | Models: XGBoost, Random Forest, Logistic Regression  
> Currency: ₹ (£1 = ₹107, fixed 2010–2011 period rate)

---

## Quick Start

```bash
# 1. Install dependencies (use the smart installer — handles SHAP on Windows)
python install.py

# 2. Place the dataset at: data/OnlineRetail.csv
#    Download: https://archive.ics.uci.edu/ml/datasets/Online+Retail

# 3. Run the ML pipeline
python pipelines/run_pipeline.py

# 4. Launch the dashboard
streamlit run app.py
```

---

## Installation — SHAP on Windows

SHAP requires a C++ compiler to build from source. The smart installer
(`install.py`) handles this automatically with three fallback strategies.
If it still fails, use one of these manual options:

### Option A — Conda (Recommended for Windows)
```bash
conda install -c conda-forge shap
pip install pandas numpy scikit-learn xgboost matplotlib seaborn plotly streamlit joblib
```

### Option B — Install Visual C++ Build Tools
1. Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
2. Install "Desktop development with C++"
3. Then: `pip install -r requirements.txt`

### Option C — Run without SHAP
The pipeline and dashboard work fine without SHAP.
SHAP plots are skipped gracefully — all other outputs are produced normally.
```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn plotly streamlit joblib
python pipelines/run_pipeline.py   # SHAP steps are auto-skipped
```

---

## Business Problem

In e-commerce and retail, acquiring a new customer costs **5–7× more** than retaining
an existing one. This project builds a production-ready system that:

1. **Predicts** which customers are likely to churn using behavioral features + XGBoost
2. **Quantifies** revenue at risk using Customer Lifetime Value (CLV)
3. **Segments** customers into 4 actionable priority groups so retention efforts are targeted

---

## Key Results

| Metric | Value |
|--------|-------|
| Dataset | 541,909 raw transactions → 3,565 customer profiles |
| Churn Rate | ~48.6% (balanced dataset) |
| XGBoost CV AUC | 0.7287 ± 0.0149 (5-fold) |
| XGBoost Test AUC | 0.7164 |
| Top Feature (SHAP) | ActiveMonths (0.479) |

---

## Project Structure

```
retail_churn/
├── app.py                        # Streamlit dashboard entry point
├── install.py                    # Smart installer (handles SHAP on Windows)
├── requirements.txt
├── README.md
│
├── data/
│   └── OnlineRetail.csv          # Place dataset here
│
├── models/                       # Saved model .pkl files
│
├── outputs/
│   ├── plots/                    # All generated charts (.png)
│   ├── predictions/              # Customer CSVs
│   ├── metrics/                  # Model metrics JSON files
│   ├── reports/
│   └── logs/
│
├── pipelines/
│   └── run_pipeline.py           # Main pipeline — run from terminal
│
├── src/
│   ├── config.py                 # All constants and parameters
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── segmentation.py
│   ├── train.py
│   ├── evaluation.py
│   ├── explainability.py         # SHAP (optional — skipped if not installed)
│   ├── predict.py
│   ├── business_metrics.py
│   ├── visualization.py
│   └── utils.py
│
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
        └── custom.css
```

---

## ML Pipeline Steps

| Step | What Happens | Notebook Section |
|------|-------------|-----------------|
| 1 | Load UCI Online Retail CSV | §2 |
| 2 | Clean data (drop NAs, cancellations, outliers) | §3 |
| 3 | Temporal windows + INR conversion (£1=₹107) | §4 |
| 4 | RFM + 6 behavioral features + churn label | §5–6 |
| 5 | RFM scoring (1–5) + rule-based segmentation | §7 |
| 6 | EDA charts saved to outputs/plots/ | §8 |
| 7 | Mutual information feature selection | §9 |
| 8 | KDE churn separation plots | §10 |
| 9 | Stratified temporal train/test split (80/20) | §11 |
| 10 | Train LR + RF + XGBoost (5-fold CV) | §12 |
| 11 | Evaluate models, overfitting diagnosis | §13 |
| 12 | Threshold analysis (0.30–0.70) | §14 |
| 13 | SHAP explainability (if installed) | §15 |
| 14 | Production model on full dataset | §16 |
| 15 | CLV + priority group assignment | §17 |
| 16 | Business intelligence dashboard plots | §18 |
| 17 | Business validation checks + KPIs | §19 |

---

## Dashboard Pages

| Page | Content |
|------|---------|
| **Home** | Project overview, workflow diagram, setup guide |
| **Executive Overview** | KPIs, revenue at risk, monthly trend, segment table |
| **Customer Segmentation** | RFM groups, CLV distribution, cohort heatmap, explorer |
| **Predictive Intelligence** | Model comparison, ROC, SHAP, threshold analysis |
| **Retention Strategy** | Priority matrix, action table, campaign ROI simulator |

---

## XGBoost Regularisation Settings

```python
max_depth         = 3    # shallow trees — primary regularisation
min_child_weight  = 5    # 5 samples minimum per leaf
subsample         = 0.8  # row subsampling
colsample_bytree  = 0.8  # column subsampling
reg_alpha         = 0.1  # L1 weight regularisation
reg_lambda        = 1.0  # L2 weight regularisation
gamma             = 0.1  # min gain required to make a split
```

---

*Dataset: UCI Machine Learning Repository — Online Retail*
