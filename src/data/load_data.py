import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
import pandas as pd
from src.logger import get_logger
logger = get_logger('data_loader')
DATA_PATH = ROOT / 'data'
PATHS = {'raw': DATA_PATH / 'raw/smh_raw', 'clean': DATA_PATH / 'interim/smh_clean', 'imputed': DATA_PATH / 'interim/smh_imputed', 'train_features': DATA_PATH / 'processed/train_features', 'test_features': DATA_PATH / 'processed/test_features', 'train_target': DATA_PATH / 'processed/train_target', 'test_target': DATA_PATH / 'processed/test_target'}

def load_data(type='raw'):
    if type not in PATHS:
        logger.error(f'there is no {type} data')
        raise ValueError(f'invalide type: {type}')
    path = PATHS[type].with_suffix('.csv')
    if not path.exists():
        logger.error(f'path does not exists: {path}')
        raise FileNotFoundError(path)
    try:
        data = pd.read_csv(path)
        logger.info(f'{type} data loaded successfully')
        return data
    except Exception as e:
        logger.error(f'error while trying to load data: {e}')
        raise

def save_data(type, data):
    if type not in PATHS:
        logger.error(f'there is no {type} data')
        raise ValueError(f'invalide type: {type}')
    path = PATHS[type].with_suffix('.csv')
    try:
        data.to_csv(path, index=False)
        logger.info(f'{type} data saved successfully')
        return data
    except Exception as e:
        logger.error(f'error while trying to save data: {e}')
        raise
