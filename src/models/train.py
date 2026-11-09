import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from src.data.load_data import load_data
from src.logger import get_logger
logger = get_logger('model_training')
RANDOM_STATE = 42

def train_and_compare_models(output_csv: str='reports/model_comparison.csv') -> pd.DataFrame:
    logger.info('Loading preprocessed training data...')
    x_train = load_data('train_features')
    y_train = load_data('train_target').iloc[:, 0].astype(int)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = {'accuracy': 'accuracy', 'balanced_accuracy': 'balanced_accuracy', 'precision_macro': 'precision_macro', 'recall_macro': 'recall_macro', 'f1_macro': 'f1_macro', 'f1_weighted': 'f1_weighted'}
    models = {'Dummy Baseline': DummyClassifier(strategy='stratified', random_state=RANDOM_STATE), 'Logistic Regression': LogisticRegression(max_iter=1000, random_state=RANDOM_STATE), 'Support Vector Machine': SVC(probability=True, random_state=RANDOM_STATE), 'Random Forest': RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE), 'XGBoost': XGBClassifier(n_estimators=100, random_state=RANDOM_STATE, eval_metric='mlogloss')}
    records = []
    for name, model in models.items():
        logger.info(f'Running cross-validation for {name}...')
        res = cross_validate(model, x_train, y_train, cv=cv, scoring=scoring, return_train_score=True, n_jobs=-1)
        records.append({'Model': name, 'Accuracy': res['test_accuracy'].mean(), 'Accuracy_std': res['test_accuracy'].std(), 'Balanced_Accuracy': res['test_balanced_accuracy'].mean(), 'Macro_Precision': res['test_precision_macro'].mean(), 'Macro_Recall': res['test_recall_macro'].mean(), 'Macro_F1': res['test_f1_macro'].mean(), 'Macro_F1_std': res['test_f1_macro'].std(), 'Weighted_F1': res['test_f1_weighted'].mean(), 'Train_Macro_F1': res['train_f1_macro'].mean()})
    df_comparison = pd.DataFrame(records).sort_values('Macro_F1', ascending=False)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_comparison.to_csv(output_csv, index=False)
    logger.info(f'Saved model comparison to {output_csv}')
    return df_comparison
if __name__ == '__main__':
    train_and_compare_models()
