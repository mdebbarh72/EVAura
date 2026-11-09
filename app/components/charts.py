import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

RISK_COLORS = {'Low': '#2ecc71', 'Medium': '#f39c12', 'High': '#e74c3c'}

def render_kpi_cards(metrics: Dict[str, Any]):
    col1, col2, col3, col4 = st.columns(4)
    acc = metrics.get('accuracy', 0.758)
    f1 = metrics.get('macro_f1', 0.7571)
    auc = metrics.get('roc_auc', 0.9109)
    high_recall = metrics.get('high_risk_recall', 0.8118)
    with col1:
        st.metric(label='Test Accuracy', value=f'{acc * 100:.2f}%', delta='+42.8% vs Baseline', help='Classification accuracy on held-out test data.')
    with col2:
        st.metric(label='Macro F1-Score', value=f'{f1:.4f}', delta='+0.424 vs Baseline', help='Macro-averaged F1 across all risk classes.')
    with col3:
        st.metric(label='Multiclass ROC-AUC', value=f'{auc:.4f}', delta='Strong Discrimination', help='Macro-averaged area under the ROC curve.')
    with col4:
        st.metric(label='High-Risk Recall', value=f'{high_recall * 100:.1f}%', delta='0 High-to-Low Errors', help='Recall for high-risk students.')

def plot_probability_bars(probabilities: Dict[str, float]) -> plt.Figure:
    classes = list(probabilities.keys())
    values = [probabilities[c] * 100 for c in classes]
    colors = [RISK_COLORS.get(c, '#3498db') for c in classes]
    fig, ax = plt.subplots(figsize=(7, 2.5), facecolor='none')
    y_pos = np.arange(len(classes))
    bars = ax.barh(y_pos, values, color=colors, height=0.55, edgecolor='none')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(classes, fontsize=11, fontweight='bold')
    ax.set_xlim(0, 105)
    ax.set_xlabel('Probability (%)', fontsize=10, color='#555555')
    for bar, val in zip(bars, values):
        ax.text(val + 1.5, bar.get_y() + bar.get_height() / 2, f'{val:.1f}%', va='center', ha='left', fontsize=10, fontweight='bold', color='#333333')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#cccccc')
    ax.grid(axis='x', linestyle='--', alpha=0.3)
    plt.tight_layout()
    return fig

def plot_confusion_matrix_heatmap(cm_array: np.ndarray, labels: List[str]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 5), facecolor='none')
    cm_norm = cm_array.astype(float) / cm_array.sum(axis=1)[:, np.newaxis]
    annot = np.empty_like(cm_array, dtype=object)
    for i in range(cm_array.shape[0]):
        for j in range(cm_array.shape[1]):
            annot[i, j] = f'{cm_array[i, j]}\n({cm_norm[i, j] * 100:.1f}%)'
    sns.heatmap(cm_array, annot=annot, fmt='', cmap='Blues', xticklabels=labels, yticklabels=labels, cbar=False, ax=ax, linewidths=1.5, linecolor='white', annot_kws={'fontsize': 11, 'fontweight': 'bold'})
    ax.set_xlabel('Predicted Label', fontsize=11, fontweight='bold', labelpad=10)
    ax.set_ylabel('True Label', fontsize=11, fontweight='bold', labelpad=10)
    ax.set_title('Test Set Confusion Matrix', fontsize=13, fontweight='bold', pad=12)
    plt.tight_layout()
    return fig

def plot_model_comparison_bar(comparison_df: pd.DataFrame) -> plt.Figure:
    df_plot = comparison_df.sort_values('Macro_F1', ascending=True)
    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor='none')
    y_pos = np.arange(len(df_plot))
    colors = ['#3498db' if i < len(df_plot) - 1 else '#2ecc71' for i in range(len(df_plot))]
    bars = ax.barh(y_pos, df_plot['Macro_F1'], color=colors, height=0.55)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_plot['Model'], fontsize=10, fontweight='bold')
    ax.set_xlim(0, 1.0)
    ax.set_xlabel('Macro F1-Score (5-Fold Stratified CV)', fontsize=10, fontweight='bold')
    ax.set_title('Model Cross-Validation Benchmark', fontsize=12, fontweight='bold', pad=12)
    for bar, val in zip(bars, df_plot['Macro_F1']):
        ax.text(val + 0.02, bar.get_y() + bar.get_height() / 2, f'{val:.4f}', va='center', fontsize=9.5, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', linestyle='--', alpha=0.3)
    plt.tight_layout()
    return fig

def plot_batch_risk_pie(risk_counts: pd.Series) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5, 4.5), facecolor='none')
    labels = risk_counts.index.tolist()
    counts = risk_counts.values.tolist()
    colors = [RISK_COLORS.get(label, '#95a5a6') for label in labels]
    wedges, texts, autotexts = ax.pie(counts, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors, wedgeprops=dict(width=0.45, edgecolor='white', linewidth=2), pctdistance=0.75)
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(10)
    for text in texts:
        text.set_fontweight('bold')
        text.set_fontsize(11)
    ax.set_title('Cohort Risk Breakdown', fontsize=12, fontweight='bold', pad=10)
    plt.tight_layout()
    return fig
