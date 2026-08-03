"""Closed-form ridge regression.

Functions
---------
* ``fit_ridge(X, y, alpha)`` — solves ``(X.T X + alpha I) w = X.T y``.
* ``predict_linear(X, w, intercept=0.0)`` — generic linear predictor used
  by Ridge, ElasticNet-CD, and PLS backends.
"""
from __future__ import annotations

import numpy as np

from ..data.qc import _as_float_array


def fit_ridge(X, y, alpha):
    """Closed-form ridge: ``(X.T X + alpha I) w = X.T y``.

    Fit a ridge-regression coefficient vector with an explicit intercept.
    """
    X = _as_float_array(X)
    y = _as_float_array(y)
    n, p = X.shape
    A = X.T @ X + float(alpha) * np.eye(p)
    b = X.T @ y
    try:
        w = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        w = np.linalg.pinv(A) @ b
    return w


def predict_linear(X, w, intercept=0.0):
    """Generic linear predictor: ``X @ w + intercept``.

    Predict from a linear coefficient vector and optional intercept.
    """
    return _as_float_array(X) @ _as_float_array(w) + float(intercept)


__all__ = ["fit_ridge", "predict_linear"]
