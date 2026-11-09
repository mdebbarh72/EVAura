import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
for root_candidate in [Path('/opt/airflow'), Path(__file__).resolve().parents[1], Path.cwd()]:
    if (root_candidate / 'src').exists() and str(root_candidate) not in sys.path:
        sys.path.insert(0, str(root_candidate))
from airflow import DAG
try:
    from airflow.providers.standard.operators.python import PythonOperator
except ImportError:
    from airflow.operators.python import PythonOperator
default_args = {'owner': 'airflow', 'depends_on_past': False, 'start_date': datetime(2026, 1, 1), 'email_on_failure': False, 'email_on_retry': False, 'retries': 1, 'retry_delay': timedelta(minutes=5)}

def load_data_step(**context):

    from src.data.load_data import load_data
    from src.logger import get_logger

    logger = get_logger('dag_load_data')
    logger.info('Executing Task 1: Load Raw Data')

    df = load_data('raw')
    logger.info(f'Loaded {len(df)} raw student survey records across {len(df.columns)} columns.')

    if 'ti' in context:
        context['ti'].xcom_push(key='raw_record_count', value=int(len(df)))
    return f'Successfully loaded {len(df)} raw records.'

def clean_data_step(**context):
    from src.data.load_data import load_data, save_data
    from src.data.clean_data import clean_raw_data
    from src.logger import get_logger

    logger = get_logger('dag_clean_data')
    logger.info('Executing Task 2: Clean Raw Data')

    raw_df = load_data('raw')

    clean_df = clean_raw_data(raw_df, identifier='Student_ID')

    save_data('clean', clean_df)

    logger.info(f'Cleaned dataset saved to interim storage with {len(clean_df)} records.')

    return f'Cleaned {len(clean_df)} records.'

def split_and_preprocess_step(**context):

    from src.data.load_data import load_data, save_data
    from src.data.split_data import split_features_target, stratified_train_test_split
    from src.features.outliers import IQROutlierCapper
    from src.features.preprocessing import engineer_candidate_features, apply_feature_selection, encode_categorical_features, scale_features
    from src.config import NUM_CONT_COLS, TARGET_MAPPING, MODELS_DIR
    import joblib
    from src.logger import get_logger
    import pandas as pd


    logger = get_logger('dag_split_preprocess')
    logger.info('Executing Task 3: Split & Preprocess Data')

    clean_df = load_data('clean')

    features, target = split_features_target(clean_df, target_col='Mental_Health_Risk')

    y_encoded = target.map(TARGET_MAPPING)
    x_train_raw, x_test_raw, y_train, y_test = stratified_train_test_split(features, y_encoded, test_size=0.25, random_state=42)

    capper = IQROutlierCapper(columns=NUM_CONT_COLS, factor=1.5)

    x_train_capped = capper.fit_transform(x_train_raw)
    x_test_capped = capper.transform(x_test_raw)

    x_train_cand = engineer_candidate_features(x_train_capped)
    x_test_cand = engineer_candidate_features(x_test_capped)

    x_train_sel, x_test_sel, sel_cols, _ = apply_feature_selection(x_train_cand, x_test_cand, y_train, variance_thresh=0.02, corr_thresh=0.75, target_corr_thresh=0.05)

    x_train_enc, x_test_enc, encoders = encode_categorical_features(x_train_sel, x_test_sel)

    x_train_scaled, x_test_scaled, scaler = scale_features(x_train_enc, x_test_enc)

    save_data('train_features', x_train_scaled)
    save_data('test_features', x_test_scaled)
    save_data('train_target', pd.DataFrame({'Mental_Health_Risk': y_train}))
    save_data('test_target', pd.DataFrame({'Mental_Health_Risk': y_test}))

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(scaler, MODELS_DIR / 'scaler.joblib')
    
    logger.info(f'Persisted train ({len(x_train_scaled)} rows) and test ({len(x_test_scaled)} rows) with {x_train_scaled.shape[1]} features.')
    return f'Preprocessed {x_train_scaled.shape[1]} features.'

def train_models_step(**context):
    from src.models.train import train_and_compare_models
    from src.logger import get_logger
    logger = get_logger('dag_train_models')
    logger.info('Executing Task 4: Train & Compare Models')
    df_comparison = train_and_compare_models(output_csv='reports/model_comparison.csv')
    top_model = df_comparison.iloc[0]['Model']
    top_f1 = df_comparison.iloc[0]['Macro_F1']
    logger.info(f'Top performing candidate: {top_model} (Macro F1: {top_f1:.4f})')
    return f'Best candidate: {top_model} with Macro F1 {top_f1:.4f}'

def tune_and_evaluate_step(**context):
    from src.models.tune import tune_models
    from src.models.evaluate import evaluate_and_log_test_metrics
    from src.data.load_data import load_data
    from src.logger import get_logger
    logger = get_logger('dag_tune_evaluate')
    logger.info('Executing Task 5: Hyperparameter Tuning & Final Test Evaluation')
    best_models, tuning_df = tune_models(output_csv='reports/tuning_results.csv')
    winning_model = best_models['Logistic Regression']
    x_train = load_data('train_features')
    y_train = load_data('train_target').iloc[:, 0].astype(int)
    x_test = load_data('test_features')
    y_test = load_data('test_target').iloc[:, 0].astype(int)
    winning_model.fit(x_train, y_train)
    metrics = evaluate_and_log_test_metrics(winning_model, x_test, y_test, feature_names=list(x_train.columns), metrics_json_path='reports/metrics.json')
    logger.info(f"Final Test Evaluation: Accuracy={metrics['accuracy']:.4f}, Macro F1={metrics['macro_f1']:.4f}")
    return f"Test Macro F1: {metrics['macro_f1']:.4f}"

def save_artifacts_step(**context):
    import joblib
    import json
    from sklearn.pipeline import Pipeline
    from src.data.load_data import load_data
    from src.models.save_artifacts import save_all_artifacts
    from src.models.predict import predict_single
    from src.config import MODELS_DIR, REPORTS_DIR
    from src.logger import get_logger
    logger = get_logger('dag_save_artifacts')
    logger.info('Executing Task 6: Save Production Pipeline Artifacts')
    model = joblib.load(MODELS_DIR / 'best_model.joblib') if (MODELS_DIR / 'best_model.joblib').exists() else None
    scaler = joblib.load(MODELS_DIR / 'scaler.joblib')
    x_train = load_data('train_features')
    if model is None:
        from sklearn.linear_model import LogisticRegression
        y_train = load_data('train_target').iloc[:, 0].astype(int)
        model = LogisticRegression(C=0.5, max_iter=1000, random_state=42)
        model.fit(x_train, y_train)
    pipeline = Pipeline(steps=[('scaler', scaler), ('classifier', model)])
    metrics_path = REPORTS_DIR / 'metrics.json'
    metrics = {}
    if metrics_path.exists():
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
    save_all_artifacts(model=model, pipeline=pipeline, scaler=scaler, metrics=metrics, feature_names=list(x_train.columns))
    raw_df = load_data('raw')
    sample = raw_df.iloc[0].drop('Mental_Health_Risk').to_dict()
    test_pred = predict_single(sample, pipeline=pipeline)
    logger.info(f"Verification inference result: {test_pred['prediction']} (Confidence: {test_pred['confidence']}%)")
    return 'Production artifacts saved and verified successfully.'


with DAG(
    dag_id='mental_health_pipeline_dag',
    default_args=default_args,
    description='End-to-end Student Mental Health Risk MLOps pipeline: load -> clean -> split & preprocess -> train -> tune & evaluate -> save artifacts',
    schedule=timedelta(days=7),
    catchup=False,
    tags=['mental_health', 'mlops', 'evaura', 'classification'])
as dag:

    task_load = PythonOperator(
        task_id='load_raw_data',
        python_callable=load_data_step
        )
    
    task_clean = PythonOperator(
        task_id='clean_data',
        python_callable=clean_data_step
        )

    task_preprocess = PythonOperator(
        task_id='split_and_preprocess',
        python_callable=split_and_preprocess_step
        )

    task_train = PythonOperator(
        task_id='train_candidate_models',
        python_callable=train_models_step
        )
    task_tune_eval = PythonOperator(
        task_id='tune_and_evaluate',
        python_callable=tune_and_evaluate_step
        )
    task_save = PythonOperator(
        task_id='save_production_artifacts',
        python_callable=save_artifacts_step
        )

    task_load >> task_clean >> task_preprocess >> task_train >> task_tune_eval >> task_save
