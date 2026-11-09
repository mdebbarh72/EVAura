import os
import time
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from src.data.load_data import load_data
from src.logger import get_logger
logger = get_logger('model_tuning')
RANDOM_STATE = 42

def tune_models(output_csv: str='reports/tuning_results.csv'):
    logger.info('Loading training data for tuning...')
    x_train = load_data('train_features')
    y_train = load_data('train_target').iloc[:, 0].astype(int)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    searches = {'Logistic Regression': GridSearchCV(LogisticRegression(random_state=RANDOM_STATE), param_grid={'C': [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0], 'solver': ['lbfgs'], 'max_iter': [1000]}, cv=cv, scoring='f1_macro', n_jobs=-1, return_train_score=True), 'Random Forest': GridSearchCV(RandomForestClassifier(random_state=RANDOM_STATE), param_grid={'n_estimators': [100, 150], 'max_depth': [4, 6, 8, 10], 'min_samples_leaf': [4, 8, 12]}, cv=cv, scoring='f1_macro', n_jobs=-1, return_train_score=True), 'XGBoost': GridSearchCV(XGBClassifier(random_state=RANDOM_STATE, eval_metric='mlogloss'), param_grid={'n_estimators': [80, 120], 'max_depth': [2, 3, 4], 'learning_rate': [0.03, 0.05, 0.1], 'subsample': [0.8, 1.0]}, cv=cv, scoring='f1_macro', n_jobs=-1, return_train_score=True)}
    results = []
    best_models = {}
    for name, search in searches.items():
        logger.info(f'Tuning {name}...')
        t0 = time.perf_counter()
        search.fit(x_train, y_train)
        elapsed = time.perf_counter() - t0
        logger.info(f'Finished {name} in {elapsed:.2f}s | Best Macro F1: {search.best_score_:.4f}')
        cv_df = pd.DataFrame(search.cv_results_)
        cv_df['model'] = name
        results.append(cv_df)
        best_models[name] = search.best_estimator_
    all_tuning_df = pd.concat(results, ignore_index=True)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    all_tuning_df.to_csv(output_csv, index=False)
    logger.info(f'Saved tuning results to {output_csv}')
    return (best_models, all_tuning_df)
if __name__ == '__main__':
    tune_models()
