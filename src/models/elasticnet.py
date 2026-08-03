"""ElasticNet wrapper.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler


@dataclass
class FittedElasticNet:
    """Bundle of objects produced by :func:`fit_elasticnet`.

    Holds the trained model, the scaler used on ``X_train``, and (optionally)
    the scaler used on ``y_train`` so callers can ``inverse_transform``
    predictions back to the raw target scale.
    """

    model: ElasticNet
    x_scaler: StandardScaler
    y_scaler: Optional[StandardScaler] = None


def fit_elasticnet(
    X_train,
    y_train,
    *,
    alpha: float = 1.3,
    l1_ratio: float = 0.0,
    max_iter: int = 5000,
    random_state: int = 42,
    scale_y: bool = False,
) -> FittedElasticNet:
    """Fit a StandardScaler-wrapped sklearn ElasticNet.

    Parameters
    ----------
    X_train, y_train : array-like
        Training feature matrix (n_samples × n_features) and target vector.
    alpha : float
        Regularisation strength (sklearn's ``alpha`` — overall penalty).
    l1_ratio : float
        Mixing parameter (0 → pure L2 / Ridge, 1 → pure L1 / Lasso).
    max_iter : int, default 5000
        Maximum coordinate-descent iterations.
    random_state : int, default 42
        Forwarded to ElasticNet for reproducible optimization.
    scale_y : bool, default False
        If True, also z-score ``y_train`` and store the y-scaler. Used by
        grouped cross-validation runs where both predictors and targets are
        standardized. LOOCV, bootstrap, and composite call sites leave the
        target on its native scale, so the default is ``False``.

    Returns
    -------
    FittedElasticNet
        A bundle of (model, x_scaler, y_scaler).
    """
    x_scaler = StandardScaler()
    X_train_s = x_scaler.fit_transform(X_train)

    if scale_y:
        y_scaler = StandardScaler()
        y_train_s = y_scaler.fit_transform(np.asarray(y_train).reshape(-1, 1)).ravel()
    else:
        y_scaler = None
        y_train_s = np.asarray(y_train)

    model = ElasticNet(
        alpha=alpha,
        l1_ratio=l1_ratio,
        max_iter=max_iter,
        random_state=random_state,
    )
    model.fit(X_train_s, y_train_s)
    return FittedElasticNet(model=model, x_scaler=x_scaler, y_scaler=y_scaler)


def predict_elasticnet(fitted: FittedElasticNet, X) -> np.ndarray:
    """Predict on the raw target scale, given a :class:`FittedElasticNet`.

    Applies the stored target scaler when one was fit during training.
    """
    X_s = fitted.x_scaler.transform(X)
    pred_s = fitted.model.predict(X_s)
    if fitted.y_scaler is None:
        return pred_s
    return fitted.y_scaler.inverse_transform(pred_s.reshape(-1, 1)).ravel()


__all__ = ["FittedElasticNet", "fit_elasticnet", "predict_elasticnet"]
