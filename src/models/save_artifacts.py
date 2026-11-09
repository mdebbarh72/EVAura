import os
import json
import datetime
import joblib
from src.logger import get_logger
logger = get_logger('save_artifacts')

def save_all_artifacts(model, pipeline, scaler, metrics=None, feature_names=None):
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/best_model.joblib')
    joblib.dump(pipeline, 'models/best_pipeline.joblib')
    if scaler is not None:
        joblib.dump(scaler, 'models/scaler.joblib')
    metadata = {'model_name': type(model).__name__, 'architecture': str(model), 'hyperparameters': model.get_params() if hasattr(model, 'get_params') else {}, 'created_at': datetime.datetime.now().isoformat(), 'library_versions': {'scikit_learn': '1.4.1.post1', 'numpy': '1.26.4', 'pandas': '2.2.1'}, 'features': feature_names or [], 'n_features': len(feature_names) if feature_names else 0, 'test_metrics': metrics or {}}
    with open('models/model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    logger.info('Successfully persisted models/best_pipeline.joblib, models/scaler.joblib, models/model_metadata.json')
if __name__ == '__main__':
    pass
