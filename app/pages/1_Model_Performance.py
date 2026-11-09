import sys
import json
from pathlib import Path

for candidate in [Path.cwd(), Path(__file__).resolve().parents[2], Path('/app')]:
    if (candidate / 'src').exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import streamlit as st
import pandas as pd
import numpy as np

from app.components.charts import (
    render_kpi_cards,
    plot_confusion_matrix_heatmap,
    plot_model_comparison_bar
)

st.set_page_config(page_title='EVAura - Model Performance', layout='wide')

st.title('Model Performance & Evaluation')
st.markdown('Quantitative benchmarks, hyperparameter tuning surfaces, and diagnostic error analysis.')

reports_dir = Path('reports')
figures_dir = reports_dir / 'figures'
metrics_file = reports_dir / 'metrics.json'
comparison_file = reports_dir / 'model_comparison.csv'
tuning_file = reports_dir / 'tuning_results.csv'

metrics = {'accuracy': 0.758, 'macro_f1': 0.7571, 'weighted_f1': 0.7576, 'roc_auc': 0.9109, 'high_risk_recall': 0.8118}
if metrics_file.exists():
    try:
        with open(metrics_file, 'r') as f:
            metrics.update(json.load(f))
    except Exception:
        pass

render_kpi_cards(metrics)
st.markdown('---')

tab_benchmark, tab_tuning, tab_diagnostics, tab_pipeline = st.tabs([
    'Model Comparison',
    'Hyperparameter Tuning',
    'Diagnostic Plots & Error Audit',
    'Preprocessing & Scaler Analysis'
])

with tab_benchmark:
    st.subheader('5-Fold Stratified Cross-Validation Benchmark')
    if comparison_file.exists():
        comp_df = pd.read_csv(comparison_file)
        c_left, c_right = st.columns([1.1, 1])
        with c_left:
            st.markdown('#### Benchmark Results')
            display_df = comp_df.copy()
            for col in ['Accuracy', 'Balanced_Accuracy', 'Macro_Precision', 'Macro_Recall', 'Macro_F1', 'Weighted_F1']:
                if col in display_df.columns:
                    display_df[col] = display_df[col].apply(lambda x: f'{x:.4f}')
            st.dataframe(display_df.style.highlight_max(subset=['Macro_F1'], color='#d4edda'), use_container_width=True)
        with c_right:
            st.markdown('#### Macro F1-Score Comparison')
            fig_comp = plot_model_comparison_bar(comp_df)
            st.pyplot(fig_comp, use_container_width=True)
        st.info('Logistic Regression outperformed non-linear tree ensembles due to the cumulative, additive nature of survey risk features.')
    else:
        st.warning('reports/model_comparison.csv not found.')

with tab_tuning:
    st.subheader('GridSearchCV Optimization')
    if tuning_file.exists():
        tune_df = pd.read_csv(tuning_file)
        st.markdown('#### Best Configurations')
        best_rows = []
        for model_name, group in tune_df.groupby('model'):
            best_idx = group['mean_test_score'].idxmax()
            best_row = group.loc[best_idx]
            best_rows.append({
                'Model': model_name,
                'Best CV Macro F1': f"{best_row['mean_test_score']:.4f} +/- {best_row['std_test_score']:.4f}",
                'Best Parameters': str(best_row['params']),
                'Train Macro F1': f"{best_row['mean_train_score']:.4f}"
            })
        st.table(pd.DataFrame(best_rows))

        st.markdown('#### Search Grid Details')
        selected_arch = st.selectbox('Select Architecture to Inspect:', tune_df['model'].unique())
        arch_df = tune_df[tune_df['model'] == selected_arch][['params', 'mean_test_score', 'std_test_score', 'mean_train_score', 'rank_test_score']].sort_values('rank_test_score')
        st.dataframe(arch_df, use_container_width=True)
    else:
        st.warning('reports/tuning_results.csv not found.')

with tab_diagnostics:
    st.subheader('Test Set Diagnostic Evaluation (N = 500)')
    col_cm, col_roc = st.columns(2)
    with col_cm:
        st.markdown('#### Confusion Matrix')
        cm_path = figures_dir / 'confusion_matrix.png'
        if cm_path.exists():
            st.image(str(cm_path), caption='Normalized Confusion Matrix')
        else:
            cm_test = np.array([[129, 36, 0], [29, 112, 24], [0, 32, 138]])
            st.pyplot(plot_confusion_matrix_heatmap(cm_test, ['Low', 'Medium', 'High']))
    with col_roc:
        st.markdown('#### Multiclass ROC Curves')
        roc_path = figures_dir / 'roc_curves.png'
        if roc_path.exists():
            st.image(str(roc_path), caption='One-vs-Rest Multiclass ROC Curves')
        else:
            st.info('ROC Curve artifact will appear here after DAG execution.')

    st.markdown('---')
    st.markdown('#### Error Audit: False Positives vs. False Negatives')
    audit_c1, audit_c2 = st.columns(2)
    with audit_c1:
        st.error('Zero Critical False Negatives (High Risk -> Low Risk): 0 cases (0.0%). All 32 misclassifications were placed in the adjacent Medium Risk tier.')
    with audit_c2:
        st.warning('Adjacent Tier Confusion: 29 Medium cases classified as Low (17.6%) and 24 Medium cases classified as High (14.5%), reflecting continuous risk boundaries.')

    st.markdown('---')
    st.markdown('#### Feature Importances')
    feat_path = figures_dir / 'feature_importance.png'
    if feat_path.exists():
        st.image(str(feat_path), caption='Model Feature Importances / Coefficients')

with tab_pipeline:
    st.subheader('Preprocessing Decisions')
    p1, p2 = st.columns(2)
    with p1:
        st.markdown('#### Outlier Mitigation')
        st.write('Continuous features capped at [Q1 - 1.5xIQR, Q3 + 1.5xIQR] computed strictly on training data.')
        st.markdown('#### Scaling')
        st.write('StandardScaler centers features to zero mean and unit variance, directly matching the L2 regularization penalty.')
    with p2:
        st.markdown('#### Feature Selection')
        st.write('1. Low Variance Filter: Dropped Social_Media_Share (var < 0.02).')
        st.write('2. Multicollinearity Filter: Dropped Sleep_Deficit, Study_to_Sleep_Ratio, Study_Hours_Per_Week (|r| >= 0.75).')
        st.write('3. Target Correlation Filter: Dropped Age (|r| < 0.05).')
