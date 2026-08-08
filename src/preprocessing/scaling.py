"""Train-only scaling transformer."""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class TrainOnlyStandardScaler(BaseEstimator, TransformerMixin):
    """Standardize arrays using statistics learned only in ``fit``."""

    def __init__(self, eps: float = 1e-12):
        self.eps = eps

    def fit(self, X, y=None):
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        self.mean_ = np.nanmean(X_arr, axis=0)
        scale = np.nanstd(X_arr, axis=0, ddof=0)
        scale = np.where(np.isfinite(scale) & (scale > float(self.eps)), scale, 1.0)
        self.scale_ = scale.astype(float)
        self.n_features_in_ = int(X_arr.shape[1])
        self.fit_rows_ = int(X_arr.shape[0])
        return self

    def transform(self, X):
        if not hasattr(self, "mean_"):
            raise RuntimeError("TrainOnlyStandardScaler must be fit before transform")
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        if X_arr.shape[1] != self.n_features_in_:
            raise ValueError("X has a different number of features than during fit")
        return (X_arr - self.mean_) / self.scale_


__all__ = ["TrainOnlyStandardScaler"]
