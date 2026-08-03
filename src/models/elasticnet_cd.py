"""NumPy coordinate-descent ElasticNet backend.

This lightweight ElasticNet implementation supports small-sample feature
selection experiments without requiring sklearn's estimator interface.
Predictors are expected to be standardized by the caller and the intercept is
returned on the target scale.
"""
from __future__ import annotations

import numpy as np

from ..data.qc import _as_float_array
from .ridge import predict_linear  # re-export for convenience


def soft_threshold(z, g):
    """Scalar soft-threshold operator for L1 shrinkage."""
    if z > g:
        return z - g
    if z < -g:
        return z + g
    return 0.0


def fit_elasticnet_cd(X, y, alpha, l1_ratio, max_iter=500, tol=1e-6):
    """Fit coordinate descent on standardized predictors and centered target.

    Returns ``(w, intercept)`` where ``intercept`` is ``mean(y)``.
    """
    X = _as_float_array(X)
    y = _as_float_array(y)
    n, p = X.shape

    # Center y
    y_mean = np.mean(y)
    y_c = y - y_mean

    w = np.zeros(p, dtype=float)
    X_col_norm = (X ** 2).sum(axis=0) / n

    l1 = float(alpha) * float(l1_ratio)
    l2 = float(alpha) * (1.0 - float(l1_ratio))

    # Precompute X^T y for diagnostics and possible backend extensions.
    XTy = (X.T @ y_c) / n

    for _ in range(int(max_iter)):
        w_old = w.copy()
        # Update each coordinate
        for j in range(p):
            # partial residual correlation
            r = y_c - X @ w + X[:, j] * w[j]
            rho = float((X[:, j] @ r) / n)

            denom = X_col_norm[j] + l2
            w[j] = soft_threshold(rho, l1) / (denom if denom != 0 else 1.0)

        if np.max(np.abs(w - w_old)) < tol:
            break

    return w, float(y_mean)


__all__ = ["soft_threshold", "fit_elasticnet_cd", "predict_linear"]
