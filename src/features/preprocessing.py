import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from typing import List, Dict, Tuple, Any, Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from src.config import ORDINAL_ORDERS, NOMINAL_COLS, BINARY_COLS, VARIANCE_THRESHOLD, CORR_THRESHOLD, TARGET_CORR_THRESHOLD, TARGET_MAPPING
from src.logger import get_logger
logger = get_logger('preprocessing')

def engineer_candidate_features(df_in: pd.DataFrame) -> pd.DataFrame:
    df_feat = df_in.copy()
    df_feat['Sleep_Deficit'] = np.maximum(0.0, 7.5 - df_feat['Sleep_Hours'])
    df_feat['Study_to_Sleep_Ratio'] = df_feat['Study_Hours_Per_Week'] / (df_feat['Sleep_Hours'] * 7.0 + 1e-05)
    df_feat['Social_Media_Share'] = df_feat['Social_Media_Hours'] / (df_feat['Screen_Time_Hours'] + 1e-05)
    df_feat['Non_Social_Screen_Time'] = np.maximum(0.0, df_feat['Screen_Time_Hours'] - df_feat['Social_Media_Hours'])
    df_feat['Stress_to_Support_Ratio'] = df_feat['Financial_Stress'] / (df_feat['Family_Support'] + 1e-05)
    df_feat['Academic_Effort_Index'] = df_feat['GPA'] * (df_feat['Study_Hours_Per_Week'] / 10.0)
    return df_feat

def apply_feature_selection(x_train_cand: pd.DataFrame, x_test_cand: pd.DataFrame, y_train: pd.Series, variance_thresh: float=VARIANCE_THRESHOLD, corr_thresh: float=CORR_THRESHOLD, target_corr_thresh: float=TARGET_CORR_THRESHOLD) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], Dict[str, Any]]:
    if y_train.dtype == object or isinstance(y_train.iloc[0], str):
        y_num = y_train.map(TARGET_MAPPING)
    else:
        y_num = y_train
    numeric_cols = x_train_cand.select_dtypes(include=[np.number]).columns.tolist()
    variances = x_train_cand[numeric_cols].var()
    low_var_drops = variances[variances < variance_thresh].index.tolist()
    cols_pass_var = [c for c in numeric_cols if c not in low_var_drops]
    target_corrs = x_train_cand[cols_pass_var].apply(lambda col: col.corr(y_num)).abs()
    pairwise_corr = x_train_cand[cols_pass_var].corr().abs()
    redundant_drops = set()
    for i in range(len(cols_pass_var)):
        for j in range(i + 1, len(cols_pass_var)):
            f1 = cols_pass_var[i]
            f2 = cols_pass_var[j]
            if pairwise_corr.loc[f1, f2] >= corr_thresh:
                drop_f = f2 if target_corrs[f1] >= target_corrs[f2] else f1
                redundant_drops.add(drop_f)
    cols_pass_corr = [c for c in cols_pass_var if c not in redundant_drops]
    rem_target_corrs = target_corrs[cols_pass_corr]
    low_target_drops = rem_target_corrs[rem_target_corrs < target_corr_thresh].index.tolist()
    selected_num = [c for c in cols_pass_corr if c not in low_target_drops]
    cat_cols = [c for c in x_train_cand.columns if c not in numeric_cols]
    final_features = selected_num + cat_cols
    x_train_sel = x_train_cand[final_features].copy()
    x_test_sel = x_test_cand[final_features].copy()
    report = {'low_variance_dropped': low_var_drops, 'collinear_dropped': list(redundant_drops), 'low_target_dropped': low_target_drops, 'selected_numerical': selected_num, 'final_features': final_features}
    logger.info(f'Feature selection complete: retained {len(final_features)} features ({len(selected_num)} numeric, {len(cat_cols)} categorical)')
    return (x_train_sel, x_test_sel, final_features, report)

def encode_categorical_features(x_train: pd.DataFrame, x_test: pd.DataFrame, ordinal_orders: Dict[str, List[str]]=ORDINAL_ORDERS, nominal_cols: List[str]=NOMINAL_COLS, binary_cols: List[str]=BINARY_COLS) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    x_train_enc = x_train.copy()
    x_test_enc = x_test.copy()
    ord_cols = [c for c in ordinal_orders.keys() if c in x_train_enc.columns]
    if ord_cols:
        ord_categories = [ordinal_orders[col] for col in ord_cols]
        ordinal_encoder = OrdinalEncoder(categories=ord_categories, handle_unknown='use_encoded_value', unknown_value=-1)
        x_train_enc[ord_cols] = ordinal_encoder.fit_transform(x_train_enc[ord_cols])
        x_test_enc[ord_cols] = ordinal_encoder.transform(x_test_enc[ord_cols])
    else:
        ordinal_encoder = None
    for col in binary_cols:
        if col in x_train_enc.columns:
            x_train_enc[col] = (x_train_enc[col].astype(str).str.lower() == 'yes').astype(int)
            x_test_enc[col] = (x_test_enc[col].astype(str).str.lower() == 'yes').astype(int)
    active_nom_cols = [c for c in nominal_cols if c in x_train_enc.columns]
    if active_nom_cols:
        one_hot = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
        gender_train = pd.DataFrame(one_hot.fit_transform(x_train_enc[active_nom_cols]), columns=one_hot.get_feature_names_out(active_nom_cols), index=x_train_enc.index)
        gender_test = pd.DataFrame(one_hot.transform(x_test_enc[active_nom_cols]), columns=one_hot.get_feature_names_out(active_nom_cols), index=x_test_enc.index)
        x_train_final = pd.concat([x_train_enc.drop(columns=active_nom_cols), gender_train], axis=1).astype(float)
        x_test_final = pd.concat([x_test_enc.drop(columns=active_nom_cols), gender_test], axis=1).astype(float)
    else:
        one_hot = None
        x_train_final = x_train_enc.astype(float)
        x_test_final = x_test_enc.astype(float)
    encoders = {'ordinal_encoder': ordinal_encoder, 'one_hot_encoder': one_hot, 'feature_names': list(x_train_final.columns)}
    return (x_train_final, x_test_final, encoders)

def scale_features(x_train: pd.DataFrame, x_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    scaler = StandardScaler()
    x_train_scaled = pd.DataFrame(scaler.fit_transform(x_train), columns=x_train.columns, index=x_train.index)
    x_test_scaled = pd.DataFrame(scaler.transform(x_test), columns=x_test.columns, index=x_test.index)
    return (x_train_scaled, x_test_scaled, scaler)

class FeatureEngineerSelector(BaseEstimator, TransformerMixin):

    def __init__(self, variance_threshold: float=VARIANCE_THRESHOLD, corr_threshold: float=CORR_THRESHOLD, target_corr_threshold: float=TARGET_CORR_THRESHOLD):
        self.variance_threshold = variance_threshold
        self.corr_threshold = corr_threshold
        self.target_corr_threshold = target_corr_threshold
        self.selected_features_: Optional[List[str]] = None

    def fit(self, X, y=None):
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        X_cand = engineer_candidate_features(X_df)
        numeric_cols = X_cand.select_dtypes(include=[np.number]).columns.tolist()
        variances = X_cand[numeric_cols].var()
        low_var = variances[variances < self.variance_threshold].index.tolist()
        cols_pass_var = [c for c in numeric_cols if c not in low_var]
        if y is not None:
            y_num = y.map(TARGET_MAPPING) if y.dtype == object or isinstance(y.iloc[0], str) else y
            target_corrs = X_cand[cols_pass_var].apply(lambda col: col.corr(y_num)).abs()
        else:
            target_corrs = pd.Series(0.0, index=cols_pass_var)
        pairwise_corr = X_cand[cols_pass_var].corr().abs()
        redundant_drops = set()
        for i in range(len(cols_pass_var)):
            for j in range(i + 1, len(cols_pass_var)):
                f1 = cols_pass_var[i]
                f2 = cols_pass_var[j]
                if pairwise_corr.loc[f1, f2] >= self.corr_threshold:
                    drop_f = f2 if target_corrs[f1] >= target_corrs[f2] else f1
                    redundant_drops.add(drop_f)
        cols_pass_corr = [c for c in cols_pass_var if c not in redundant_drops]
        if y is not None:
            rem_corrs = target_corrs[cols_pass_corr]
            low_target = rem_corrs[rem_corrs < self.target_corr_threshold].index.tolist()
            selected_num = [c for c in cols_pass_corr if c not in low_target]
        else:
            selected_num = cols_pass_corr
        cat_cols = [c for c in X_cand.columns if c not in numeric_cols and c in X_df.columns]
        self.selected_features_ = selected_num + cat_cols
        return self

    def transform(self, X):
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        X_cand = engineer_candidate_features(X_df)
        if self.selected_features_ is not None:
            return X_cand[self.selected_features_].copy()
        return X_cand
