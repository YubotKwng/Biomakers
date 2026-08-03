"""PLS1 NIPALS
"""
from __future__ import annotations

import numpy as np

from ..data.qc import _as_float_array


def fit_pls1_nipals(X, y, n_components=2, tol=1e-8, max_iter=500):
    """PLS1 with NIPALS. Returns ``(coef, intercept)``.

    Fit a one-target PLS model with NIPALS-style latent components.
    """
    X = _as_float_array(X)
    y = _as_float_array(y).reshape(-1, 1)

    # Center
    X_mean = np.mean(X, axis=0, keepdims=True)
    y_mean = np.mean(y, axis=0, keepdims=True)
    Xc = X - X_mean
    yc = y - y_mean

    n, p = Xc.shape
    W = []
    P = []
    Q = []

    X_res = Xc.copy()
    y_res = yc.copy()

    for _ in range(int(n_components)):
        # weights
        w = (X_res.T @ y_res).ravel()
        if np.allclose(w, 0):
            break
        w = w / (np.linalg.norm(w) + 1e-12)
        t = X_res @ w.reshape(-1, 1)
        # loadings
        p_vec = (X_res.T @ t).ravel() / (float(t.T @ t) + 1e-12)
        q = float((y_res.T @ t) / (float(t.T @ t) + 1e-12))

        X_res = X_res - t @ p_vec.reshape(1, -1)
        y_res = y_res - t * q

        W.append(w)
        P.append(p_vec)
        Q.append(q)

    if not W:
        coef = np.zeros(p)
    else:
        Wm = np.vstack(W).T  # p x k
        Pm = np.vstack(P).T  # p x k
        Qm = np.asarray(Q).reshape(-1, 1)  # k x 1
        # Regression coefficients: B = W (P^T W)^{-1} Q
        try:
            coef = Wm @ np.linalg.inv(Pm.T @ Wm) @ Qm
        except np.linalg.LinAlgError:
            coef = Wm @ np.linalg.pinv(Pm.T @ Wm) @ Qm
        coef = coef.ravel()

    intercept = float(y_mean.ravel() - X_mean.ravel() @ coef)
    return coef, intercept


__all__ = ["fit_pls1_nipals"]
