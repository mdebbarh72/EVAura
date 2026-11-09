import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from typing import Optional, List
import numpy as np
import pandas as pd
from src.config import IDENTIFIER
from src.logger import get_logger
logger = get_logger('clean_data')

def clean_raw_data(df: pd.DataFrame, identifier: str=IDENTIFIER) -> pd.DataFrame:
    df_clean = df.copy()
    if identifier in df_clean.columns:
        df_clean = df_clean.drop(columns=[identifier])
        logger.info(f"Dropped identifier column: '{identifier}'")
    cat_cols = df_clean.select_dtypes(include=['object', 'string']).columns.tolist()
    if 'Mental_Health_Risk' in cat_cols:
        cat_cols.remove('Mental_Health_Risk')
    for col in cat_cols:
        df_clean[col] = df_clean[col].astype(str).str.strip().str.capitalize()
    logger.info(f'Standardized text casing on {len(cat_cols)} categorical columns')
    if 'Social_Media_Hours' in df_clean.columns and 'Screen_Time_Hours' in df_clean.columns:
        impossible_mask = df_clean['Social_Media_Hours'] > df_clean['Screen_Time_Hours']
        n_impossible = int(impossible_mask.sum())
        if n_impossible > 0:
            df_clean['Social_Media_Hours'] = np.where(impossible_mask, np.nan, df_clean['Social_Media_Hours'])
            logger.info(f'Replaced {n_impossible} impossible values (Social Media > Screen Time) with NaN')
    return df_clean

def check_missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    null_counts = df.isnull().sum()
    null_pct = (df.isnull().mean() * 100).round(2)
    summary = pd.DataFrame({'Missing_Count': null_counts, 'Missing_Percentage': null_pct})
    return summary[summary['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)
