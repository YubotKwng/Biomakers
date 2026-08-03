"""Two-class Fisher LDA helpers for visit-separation models.

``fit_lda_direction`` solves a regularized pooled-covariance system, and
``predict_lda_scores`` projects held-out rows onto the fitted direction.
"""
from __future__ import annotations

import numpy as np

from ..data.qc import _as_float_array


def fit_lda_direction(X, y, shrink="auto"):
    """Return the two-class LDA direction ``inv(S + lam I) (mu1 - mu0)``.

    Returns ``None`` if either class has fewer than two rows.
    """
    X = _as_float_array(X)
    y = np.asarray(y).astype(int)
    X0 = X[y == 0]
    X1 = X[y == 1]
    if len(X0) < 2 or len(X1) < 2:
        return None
    mu0 = np.mean(X0, axis=0)
    mu1 = np.mean(X1, axis=0)
    S0 = np.cov(X0, rowvar=False, ddof=1)
    S1 = np.cov(X1, rowvar=False, ddof=1)
    S0 = np.atleast_2d(S0)
    S1 = np.atleast_2d(S1)
    S = 0.5 * (S0 + S1)
    S = np.atleast_2d(S)
    p = int(S.shape[0])
    if shrink == "auto":
        lam = 1e-3 * float(np.trace(S) / max(p, 1))
    else:
        lam = float(shrink)
    S_reg = S + lam * np.eye(p)
    try:
        w = np.linalg.solve(S_reg, (mu1 - mu0))
    except np.linalg.LinAlgError:
        w = np.linalg.pinv(S_reg) @ (mu1 - mu0)
    return w


def predict_lda_scores(X, w):
    """Project ``X`` onto the fitted LDA direction ``w``."""
    X = _as_float_array(X)
    return X @ _as_float_array(w)


__all__ = ["fit_lda_direction", "predict_lda_scores"]
