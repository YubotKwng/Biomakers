"""Feature-level imputation transformers."""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureMedianImputer(BaseEstimator, TransformerMixin):
    """Median imputer that follows sklearn fit/transform semantics."""

    def __init__(self, fill_empty: float = 0.0):
        self.fill_empty = fill_empty

    def fit(self, X, y=None):
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        med = np.nanmedian(X_arr, axis=0)
        med = np.where(np.isfinite(med), med, float(self.fill_empty))
        self.statistics_ = med.astype(float)
        self.n_features_in_ = int(X_arr.shape[1])
        self.fit_rows_ = int(X_arr.shape[0])
        return self

    def transform(self, X):
        if not hasattr(self, "statistics_"):
            raise RuntimeError("FeatureMedianImputer must be fit before transform")
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        if X_arr.shape[1] != self.n_features_in_:
            raise ValueError("X has a different number of features than during fit")
        out = X_arr.copy()
        rows, cols = np.where(~np.isfinite(out))
        if len(rows):
            out[rows, cols] = self.statistics_[cols]
        return out


__all__ = ["FeatureMedianImputer"]
