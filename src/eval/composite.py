"""Composite-score utilities for longitudinal FRDA biomarker analysis.

These helpers fit ElasticNet models, convert coefficients back to raw feature
scale, and evaluate scalar composite scores under subject-level validation.
The composites are designed to compare multimodal MRI sensitivity against
clinical scales and individual imaging biomarkers.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from ..data.qc import tukey_outliers_mask
from ..data.model_safety import assert_training_frame_is_patient_only


# ---------------------------------------------------------------------------
# Full-data composite score.
# ---------------------------------------------------------------------------
def compute_composite(
    df: pd.DataFrame,
    target_col: str,
    combo: dict,
    alpha: float = 1.3,
    l1_ratio: float = 0.0,
    *,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Fit ElasticNet on the whole dataset and return ``df`` plus a
    ``composite`` column.

    The composite is ``sum_j |coef_raw_j| * X_j`` where
    ``coef_raw = coef_std / sd(X_j)`` converts the standardized-feature
    coefficient back to the raw scale.
    """
    feature_cols: list[str] = []
    for domain in combo["domains"]:
        feature_cols.extend(domain)
    assert_training_frame_is_patient_only(df, feature_cols, target_col=target_col)

    sub = df.dropna(subset=feature_cols + [target_col]).copy()
    sub = sub[~tukey_outliers_mask(sub, feature_cols, k=3.0)].copy()

    X = sub[feature_cols].values
    y = sub[target_col].values

    xs = StandardScaler()
    ys = StandardScaler()
    Xs = xs.fit_transform(X)
    ys_ = ys.fit_transform(y.reshape(-1, 1)).ravel()

    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=5000, random_state=random_seed)
    model.fit(Xs, ys_)

    coef_std = model.coef_
    feature_sd = xs.scale_
    coef_raw = coef_std / feature_sd

    composite = (np.abs(coef_raw) * X).sum(axis=1)
    sub = sub.copy()
    sub["composite"] = composite
    return sub


# ---------------------------------------------------------------------------
# Subject-level leave-one-out composite scores.
# ---------------------------------------------------------------------------
def compute_composite_loocv(
    df_long: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str,
    subject_col: str,
    alpha: float = 1.3,
    l1_ratio: float = 0.0,
    *,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Subject-leave-one-out composite predictions.

    For every held-out subject, fit ElasticNet on the remaining subjects,
    z-score X and y on the training fold, predict the held-out rows, and
    inverse-transform predictions back to the target scale. Returns the
    held-out rows concatenated with a new ``composite_pred`` column.
    """
    assert_training_frame_is_patient_only(df_long, feature_cols, target_col=target_col)
    sub = df_long.dropna(subset=list(feature_cols) + [target_col]).copy()
    sub = sub[~tukey_outliers_mask(sub, feature_cols, k=3.0)].copy()

    subjects = sub[subject_col].unique()
    results = []

    for subj in subjects:
        test_mask = sub[subject_col] == subj
        train_mask = ~test_mask

        X_train = sub.loc[train_mask, list(feature_cols)].values
        y_train = sub.loc[train_mask, target_col].values
        X_test = sub.loc[test_mask, list(feature_cols)].values

        xs = StandardScaler()
        ys = StandardScaler()
        X_train_s = xs.fit_transform(X_train)
        y_train_s = ys.fit_transform(y_train.reshape(-1, 1)).ravel()
        X_test_s = xs.transform(X_test)

        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=5000, random_state=random_seed)
        model.fit(X_train_s, y_train_s)

        y_pred_s = model.predict(X_test_s)
        y_pred = ys.inverse_transform(y_pred_s.reshape(-1, 1)).ravel()

        test_rows = sub.loc[test_mask].copy()
        test_rows["composite_pred"] = y_pred
        results.append(test_rows)

    return pd.concat(results, ignore_index=True)


# ---------------------------------------------------------------------------
# Leave-one-out ElasticNet search for progression-sensitive composites.
# ---------------------------------------------------------------------------
def run_cv_demo2(
    df: pd.DataFrame,
    target_col: str,
    combo: dict,
    subject_col: str,
    alphas: Optional[Sequence[float]] = None,
    l1_ratios: Optional[Sequence[float]] = None,
    *,
    random_seed: int = 42,
    verbose: bool = True,
) -> dict:
    """Subject-LOO with per-fold grid-search over (alpha, l1_ratio).

    Uses subject-level leave-one-out validation and per-fold grid search.
    Predictions stay on the raw target scale so clinical-score errors remain
    directly interpretable. The returned dictionary includes accuracy metrics,
    sample counts, and summaries of the selected ElasticNet hyperparameters.
    """
    if alphas is None:
        alphas = np.logspace(-2, 2, 10)
    if l1_ratios is None:
        l1_ratios = [0.1, 0.5, 0.9]

    feature_cols = [f for domain in combo["domains"] for f in domain]
    assert_training_frame_is_patient_only(df, feature_cols, target_col=target_col)
    sub = df.dropna(subset=feature_cols + [target_col]).copy()
    n_before = len(sub)

    outlier_mask = tukey_outliers_mask(sub, feature_cols, k=3.0)
    sub = sub[~outlier_mask].copy()
    n_after = len(sub)

    if verbose:
        if n_before - n_after == 0:
            print("Outlier removal for %s / %s: removed 0 (none), left %d" % (combo["name"], target_col, n_after))
        else:
            print("Outlier removal for %s / %s: removed %d, left %d" % (combo["name"], target_col, n_before - n_after, n_after))

    groups = sub[subject_col].values
    X = sub[feature_cols].values
    y = sub[target_col].values

    subjects = np.unique(groups)
    all_preds, all_true = [], []
    best_alphas, best_l1s = [], []

    for subj in subjects:
        test_mask = groups == subj
        train_mask = ~test_mask

        X_train = X[train_mask]
        y_train = y[train_mask]
        X_test = X[test_mask]
        y_test = y[test_mask]

        xs = StandardScaler()
        X_train_s = xs.fit_transform(X_train)
        X_test_s = xs.transform(X_test)

        best_rmse, best_pred = np.inf, None
        best_a, best_l = None, None
        for a in alphas:
            for l in l1_ratios:
                m = ElasticNet(alpha=a, l1_ratio=l, max_iter=5000, random_state=random_seed)
                m.fit(X_train_s, y_train)
                pred = m.predict(X_test_s)
                rmse_val = np.sqrt(mean_squared_error(y_test, pred))
                if rmse_val < best_rmse:
                    best_rmse = rmse_val
                    best_pred = pred
                    best_a = a
                    best_l = l

        all_preds.extend(best_pred)
        all_true.extend(y_test)
        if best_a is not None and best_l is not None:
            best_alphas.append(best_a)
            best_l1s.append(best_l)

    r2 = r2_score(all_true, all_preds)
    rmse_val = np.sqrt(mean_squared_error(all_true, all_preds))

    def _mode(vals):
        if not vals:
            return None
        return max(set(vals), key=vals.count)

    return {
        "r2": r2,
        "rmse": rmse_val,
        "n_rows": len(sub),
        "lambda_mode": _mode(best_alphas),
        "alpha_mode": _mode(best_l1s),
        "lambda_mean": float(np.mean(best_alphas)) if best_alphas else None,
        "alpha_mean": float(np.mean(best_l1s)) if best_l1s else None,
    }


__all__ = [
    "compute_composite",
    "compute_composite_loocv",
    "run_cv_demo2",
]
