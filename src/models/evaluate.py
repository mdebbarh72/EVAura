import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix, classification_report
from src.data.load_data import load_data
from src.logger import get_logger
logger = get_logger('model_evaluation')
TARGET_NAMES = ['Low', 'Medium', 'High']

def evaluate_model(model, output_json: str='reports/metrics.json'):
    logger.info('Loading test data for evaluation...')
    x_test = load_data('test_features')
    y_test = load_data('test_target').iloc[:, 0].astype(int)
    y_pred = model.predict(x_test)
    y_proba = model.predict_proba(x_test) if hasattr(model, 'predict_proba') else None
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    weighted_f1 = f1_score(y_test, y_pred, average='weighted')
    macro_prec = precision_score(y_test, y_pred, average='macro')
    macro_rec = recall_score(y_test, y_pred, average='macro')
    roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro') if y_proba is not None else None
    logger.info(f'Test Accuracy: {acc:.4f} | Macro F1: {macro_f1:.4f} | ROC-AUC: {roc_auc:.4f}')
    clf_rep = classification_report(y_test, y_pred, target_names=TARGET_NAMES, output_dict=True)
    metrics = {'model_name': type(model).__name__, 'accuracy': round(acc, 4), 'macro_f1': round(macro_f1, 4), 'weighted_f1': round(weighted_f1, 4), 'macro_precision': round(macro_prec, 4), 'macro_recall': round(macro_rec, 4), 'roc_auc': round(roc_auc, 4) if roc_auc is not None else None, 'class_metrics': {cls: {'precision': round(clf_rep[cls]['precision'], 3), 'recall': round(clf_rep[cls]['recall'], 3), 'f1': round(clf_rep[cls]['f1-score'], 3), 'support': clf_rep[cls]['support']} for cls in TARGET_NAMES}}
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f'Saved evaluation metrics to {output_json}')
    return metrics
if __name__ == '__main__':
    import joblib
    model = joblib.load('models/best_model.joblib')
    evaluate_model(model)
