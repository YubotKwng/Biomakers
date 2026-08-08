"""Minimum Message Length feature selection for linear visit separation."""
from __future__ import annotations

from typing import Sequence

import numpy as np


def _as_2d_float(X) -> np.ndarray:
    """Coerce feature input to ``(n_rows, n_features)`` float shape."""
    X_arr = np.asarray(X, dtype=float)
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(-1, 1)
    if X_arr.ndim != 2:
        raise ValueError("X must be a 2D array-like object")
    return X_arr


def mml_linear_score(X, y, ridge: float = 1e-6) -> float:
    """MML87 message length for OLS linear regression.

    Lower values are better. ``X`` may have zero columns, yielding the
    intercept-only null model.
    """
    X_arr = _as_2d_float(X)
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    if X_arr.shape[0] != y_arr.shape[0]:
        raise ValueError("X and y must have the same number of rows")
    n = int(y_arr.shape[0])
    if n == 0:
        raise ValueError("mml_linear_score requires at least one row")

    # The MML score is computed on an explicit intercept design. Passing
    # ``X`` with zero columns therefore evaluates the intercept-only null.
    Xd = np.column_stack([np.ones(n, dtype=float), X_arr])
    p = int(Xd.shape[1])
    xtx = Xd.T @ Xd
    xty = Xd.T @ y_arr
    ridge_eye = float(ridge) * np.eye(p)
    try:
        beta = np.linalg.solve(xtx + ridge_eye, xty)
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(xtx + ridge_eye) @ xty

    resid = y_arr - Xd @ beta
    rss = float(resid.T @ resid)
    sigma2 = max(rss / n, 1e-12)

    # The stated MML formula uses log|X'X|. Ridge is only a numerical fallback
    # for singular or nearly singular designs; it is not a feature-selection
    # objective change.
    sign, logdet_xtx = np.linalg.slogdet(xtx)
    if sign <= 0 or not np.isfinite(logdet_xtx):
        sign, logdet_xtx = np.linalg.slogdet(xtx + ridge_eye)
    if sign <= 0 or not np.isfinite(logdet_xtx):
        logdet_xtx = float(np.log(max(np.linalg.det(xtx + ridge_eye), 1e-300)))

    logdet_F = float(logdet_xtx - p * np.log(sigma2))
    L = (
        0.5 * p * np.log(n / 12)
        + 0.5 * logdet_F
        + 0.5 * n * np.log(2 * np.pi * sigma2)
        + 0.5 * n
    )
    return float(L)


def mml_forward_selection(X_train, y_train, feature_names: Sequence[str]) -> list[str]:
    """Greedy forward stepwise selection minimizing ``mml_linear_score``."""
    X_arr = _as_2d_float(X_train)
    names = list(feature_names)
    if X_arr.shape[1] != len(names):
        raise ValueError("feature_names length must match X_train columns")

    n = X_arr.shape[0]
    selected: list[int] = []
    remaining = list(range(len(names)))
    # Start from the null model so every selected feature must justify its
    # extra parameter cost by reducing total message length.
    best_L = mml_linear_score(np.zeros((n, 0)), y_train)

    while remaining:
        best_candidate = None
        best_candidate_L = best_L
        for j in remaining:
            cols = selected + [j]
            L = mml_linear_score(X_arr[:, cols], y_train)
            if L < best_candidate_L:
                best_candidate_L = L
                best_candidate = j

        # Stop on tiny numerical differences; the selector should add a feature
        # only when it gives a real improvement over the current message length.
        if best_candidate is None or (best_L - best_candidate_L) <= 1e-9:
            break

        selected.append(best_candidate)
        remaining.remove(best_candidate)
        best_L = best_candidate_L

    return [names[i] for i in selected]


__all__ = ["mml_linear_score", "mml_forward_selection"]
