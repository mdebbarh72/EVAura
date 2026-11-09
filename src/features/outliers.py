import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class IQROutlierCapper(BaseEstimator, TransformerMixin):

    def __init__(self, columns=None, factor=1.5):
        self.columns = columns
        self.factor = factor
        self.lower_bounds_ = {}
        self.upper_bounds_ = {}

    def fit(self, X, y=None):
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        cols = self.columns if self.columns is not None else X_df.select_dtypes(include=[np.number]).columns.tolist()
        for col in cols:
            q1 = X_df[col].quantile(0.25)
            q3 = X_df[col].quantile(0.75)
            iqr = q3 - q1
            self.lower_bounds_[col] = q1 - self.factor * iqr
            self.upper_bounds_[col] = q3 + self.factor * iqr
        return self

    def transform(self, X):
        X_df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        for col, lower in self.lower_bounds_.items():
            if col in X_df.columns:
                upper = self.upper_bounds_[col]
                X_df[col] = X_df[col].clip(lower=lower, upper=upper)
        return X_df
