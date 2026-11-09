# EVAura — Student Mental Health Risk Prediction Pipeline

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.14-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()
[![Docker Compose](https://img.shields.io/badge/Docker%20Compose-Airflow%20%2B%20Streamlit-orange.svg)]()

EVAura is an end-to-end Machine Learning system that predicts student mental health risk levels (**Low**, **Medium**, **High**) based on academic workload, sleep patterns, screen time habits, and financial/family support factors.

---

## Key Results Summary

- **Best Model:** Tuned Logistic Regression ($C=0.5$, $\ell_2$ regularization, `lbfgs` solver)
- **Test Accuracy:** **$75.80\%$**
- **Test Macro F1:** **$0.7571$**
- **Test ROC-AUC:** **$0.9109$**
- **High-Risk Recall:** **$81.18\%$** ($0$ High-Risk students misclassified as Low-Risk)
- **Top Predictor:** `Sleep_Hours` (Permutation Importance drop: $\Delta F_1 = 0.241$), followed by `Stress_to_Support_Ratio` and `Academic_Effort_Index`.

Detailed report: [reports/summary_report.md](file:///home/neo/Projects/python/EVAura/reports/summary_report.md)  
Architecture document: [docs/architecture.md](file:///home/neo/Projects/python/EVAura/docs/architecture.md)

---

## Project Structure

```
EVAura/
├── docker/
│   ├── airflow/                     # Airflow container Dockerfile & requirements
│   └── streamlit/                   # Streamlit container Dockerfile & requirements
├── docker-compose.yml               # Multi-container orchestration (Airflow + Streamlit + Postgres)
│
├── data/
│   ├── raw/smh_raw.csv              # Original student mental health dataset
│   ├── processed/                   # Prepared train/test splits (80/20)
│   └── samples/                     # Batch prediction test sample CSVs
│
├── notebooks/
│   ├── 01_eda.ipynb                 # Exploratory Data Analysis & statistical audits
│   ├── 02_preprocessing.ipynb       # Imputation, 1.5*IQR capping, feature engineering & scaling
│   ├── 03_model_training.ipynb      # 5-fold Stratified CV comparing 5 candidate models
│   └── 04_tuning_evaluation.ipynb   # Hyperparameter tuning, test set evaluation, artifact exports
│
├── src/
│   ├── data/load_data.py            # Data loading & saving utility functions
│   ├── features/outliers.py         # IQROutlierCapper (scikit-learn TransformerMixin)
│   ├── features/preprocessing.py    # FeatureEngineerSelector (scikit-learn TransformerMixin)
│   ├── models/train.py              # Candidate training & comparison script
│   ├── models/tune.py               # Hyperparameter search execution
│   ├── models/evaluate.py           # Model evaluation & metrics.json export
│   └── models/predict.py            # Real-time single & batch prediction inference
│
├── models/
│   ├── best_pipeline.joblib         # Fitted StandardScaler + LogisticRegression pipeline
│   ├── best_model.joblib            # Trained estimator (C=0.5)
│   ├── scaler.joblib                # Fitted StandardScaler
│   └── model_metadata.json          # Architecture, parameters, versions, and test metrics
│
├── reports/
│   ├── metrics.json                 # Test metrics consumed by Streamlit dashboard
│   ├── model_comparison.csv         # 5-fold cross-validation results across all models
│   ├── tuning_results.csv           # GridSearch candidate scores
│   ├── summary_report.md            # Detailed narrative summary of project results
│   └── figures/                     # Confusion matrices, ROC curves, feature importance plots
│
└── app/
    ├── app.py                       # Streamlit multi-page web application
    └── pages/
        ├── 1_Model_Performance.py   # Performance dashboard (reads reports/metrics.json)
        └── 2_Predict.py             # Form & CSV batch prediction interface
```

---

## Optimal Preprocessing Pipeline

1. **Cleaning:**
   - Drop `Student_ID`.
   - Set impossible values (`Social_Media_Hours > Screen_Time_Hours`) to NaN.
   - Text standardization (whitespace stripping, title case).
2. **Missing Values:** Median imputation for numerical features; most frequent for categorical features.
3. **Outlier Mitigation:**
   - Evaluated with Tukey $1.5 \times \text{IQR}$ rule on continuous features.
   - Values capped (Winsorized) to $[\text{Lower Bound}, \text{Upper Bound}]$ computed strictly on `x_train`.
   - Preserves $100\%$ of data instances and safely generalizes to production inference.
4. **Feature Engineering & Selection:**
   - Created interaction features: `Stress_to_Support_Ratio`, `Academic_Effort_Index`, `Non_Social_Screen_Time`.
   - Low variance filter ($\sigma^2 < 0.02$) dropped near-zero variance features (`Social_Media_Share`).
   - Multicollinearity filter ($|r| \ge 0.75$) dropped redundant duplicates (`Sleep_Deficit`, `Study_to_Sleep_Ratio`, `Study_Hours_Per_Week`).
   - Target correlation filter ($|r| < 0.05$) eliminated uninformative noise (`Age`).
5. **Encoding & Normalization:**
   - `OrdinalEncoder` for ordinal scales (`Physical_Activity`, `Diet_Quality`).
   - `OneHotEncoder(drop='first')` for `Gender`.
   - `StandardScaler` to ensure zero mean and unit variance for distance and gradient-based models.

---

## Model Comparison & Tuning Overview

### 5-Fold Stratified Cross-Validation
| Model | CV Accuracy | CV Macro F1 | Train Macro F1 | Generalization Gap | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **$0.7533 \pm 0.0245$** | **$0.7530 \pm 0.0257$** | $0.7674$ | **$+0.014$** | **Shortlisted** |
| **Support Vector Machine** | $0.7260 \pm 0.0259$ | $0.7275 \pm 0.0242$ | $0.8499$ | $+0.122$ | Benchmarked |
| **XGBoost (Default)** | $0.6840 \pm 0.0176$ | $0.6827 \pm 0.0161$ | $1.0000$ | $+0.317$ | Shortlisted |
| **Random Forest (Default)** | $0.6740 \pm 0.0184$ | $0.6710 \pm 0.0200$ | $1.0000$ | $+0.329$ | Shortlisted |
| **Dummy Baseline** | $0.3353 \pm 0.0280$ | $0.3350 \pm 0.0283$ | $0.3403$ | $\approx 0$ | Reference |

### Hyperparameter Tuning Gain
- **Logistic Regression ($C=0.5$):** Tuned CV Macro F1 = **$0.7558$** (Winner)
- **XGBoost (`max_depth=3, lr=0.1, n_est=120`):** Tuned CV Macro F1 = **$0.7088$** ($+2.61\%$)
- **Random Forest (`max_depth=8, min_leaf=8`):** Tuned CV Macro F1 = **$0.6920$** ($+2.10\%$)

---

## Quickstart & Running the Application

### 1. Run with Docker Compose
```bash
docker compose up --build -d
```
- **Streamlit Web Application:** `http://localhost:8501`
- **Airflow Web UI:** `http://localhost:8080` (Username: `airflow`, Password: `airflow`)

### 2. Run Streamlit App Locally
```bash
streamlit run app/app.py --server.port=8501
```

### 3. Test Airflow DAG Locally
```bash
export AIRFLOW_HOME="$(pwd)/.airflow"
export PYTHONPATH="$(pwd)"

# Test DAG task execution
python -m airflow tasks test mental_health_pipeline_dag load_raw_data 2026-01-01
```

### 4. Run Unit Tests
```bash
python -m unittest discover -s tests -p "test_*.py"
```

### 5. Run Standalone Inference
```bash
python src/models/predict.py
```

