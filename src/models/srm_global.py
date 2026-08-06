"""Global linear composite optimized for paired SRM."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

from ..data.model_safety import assert_training_frame_is_patient_only
from ..data.qc import standardize_train_test
from ..eval.cv import group_kfold_indices, resolve_split_group_col
from ..eval.metrics import (
    bootstrap_ci_d,
    compute_cohens_d,
    compute_srm,
    paired_deltas_from_long,
)
from ..features.selection import select_features


def _paired_rows_by_subject(
    subject_id: np.ndarray,
    visit: np.ndarray,
    *,
    v1: int = 1,
    v2: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return aligned visit-1 and visit-2 row indices for complete pairs."""
    df = pd.DataFrame({"_subject": subject_id, "_visit": visit})
    df["_row"] = np.arange(len(df))
    order = df.sort_values(["_subject", "_visit"])
    v1_rows = order[order["_visit"] == v1][["_subject", "_row"]].set_index("_subject")
    v2_rows = order[order["_visit"] == v2][["_subject", "_row"]].set_index("_subject")
    common = v1_rows.index.intersection(v2_rows.index)
    return (
        v1_rows.loc[common]["_row"].to_numpy(),
        v2_rows.loc[common]["_row"].to_numpy(),
        common.to_numpy(),
    )


@dataclass
class SRMGlobalLinear:
    """Plain linear composite with fold-trained global imaging weights."""

    ridge: float = 1e-6
    covariance_shrinkage: float = 0.0
    coef_: np.ndarray | None = field(default=None, init=False)
    feature_names_: list[str] | None = field(default=None, init=False)

    def fit(self, X, subject_id, visit) -> "SRMGlobalLinear":
        """Fit the global imaging direction from training-fold paired deltas."""
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        p = int(X_arr.shape[1])
        if p == 0:
            self.coef_ = np.zeros(0, dtype=float)
            return self

        rows_v1, rows_v2, _ = _paired_rows_by_subject(
            np.asarray(subject_id), np.asarray(visit).astype(int)
        )
        if len(rows_v1) < 2:
            self.coef_ = np.zeros(p, dtype=float)
            return self

        # SRM optimisation is performed on within-subject change vectors, not
        # on raw visit rows. The fitted score is later applied back to visits.
        delta = X_arr[rows_v2] - X_arr[rows_v1]
        mu = np.nanmean(delta, axis=0)
        Sigma = np.cov(delta, rowvar=False, ddof=1)
        Sigma = np.atleast_2d(Sigma)
        shrink = float(np.clip(self.covariance_shrinkage, 0.0, 1.0))
        if shrink > 0:
            # Shrink noisy feature-feature covariances toward the diagonal.
            # This keeps per-feature variance while reducing unstable
            # cross-feature correlations in small samples.
            Sigma = (1.0 - shrink) * Sigma + shrink * np.diag(np.diag(Sigma))
        reg = Sigma + float(self.ridge) * np.eye(p)
        try:
            w = np.linalg.solve(reg, mu)
        except np.linalg.LinAlgError:
            w = np.linalg.pinv(reg) @ mu
        self.coef_ = np.asarray(w, dtype=float).reshape(-1)
        return self

    def score(self, X) -> np.ndarray:
        """Project visit-level imaging rows onto the fitted SRM direction."""
        if self.coef_ is None:
            raise RuntimeError("fit() must be called before score()")
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        if X_arr.shape[1] == 0:
            return np.zeros(X_arr.shape[0], dtype=float)
        return X_arr @ self.coef_


def srm_global_loocv(
    df_long: pd.DataFrame,
    feature_cols: Sequence[str],
    subject_col: str,
    visit_col: str = "visit",
    *,
    selection_method: str = "none",
    k: int = 8,
    ridge: float = 1e-6,
    covariance_shrinkage: float = 0.0,
    cv_n_splits: int | None = None,
    random_seed: int = 42,
    compute_ci: bool = True,
    split_group_col: str | None = None,
) -> dict:
    """Run fold-safe subject-level validation for ``SRMGlobalLinear``.

    If ``cv_n_splits`` is ``None`` or at least the number of subjects, this is
    subject-level LOO. Otherwise it uses subject-grouped K-fold splits.
    """
    feats_present = [f for f in feature_cols if f in df_long.columns]
    if not feats_present:
        return {
            "oof_df": pd.DataFrame(),
            "n_subjects": 0,
            "d_score": np.nan,
            "srm": np.nan,
            "d_ci_low": np.nan,
            "d_ci_high": np.nan,
            "selected_features_by_fold": [],
        }

    assert_training_frame_is_patient_only(df_long, feats_present)
    resolved_split_group_col = resolve_split_group_col(df_long, subject_col, split_group_col)
    cols = [subject_col, visit_col] + feats_present
    if resolved_split_group_col not in cols:
        cols.append(resolved_split_group_col)
    # Drop missing values after resolving the feature list so every fold sees
    # the same complete-case modelling matrix.
    sub = df_long[cols].dropna().copy()
    counts = sub.groupby(subject_col)[visit_col].nunique()
    valid_subjects = counts[counts == 2].index
    sub = sub[sub[subject_col].isin(valid_subjects)].copy()
    oof_rows = []
    selected_by_fold: list[list[str]] = []
    groups = sub[resolved_split_group_col].values
    split_groups = np.asarray(sub[resolved_split_group_col].unique())
    # ``split_groups`` are participant ids when available. This keeps V1V2 and
    # V2V3 intervals from the same participant on the same side of the split.
    use_kfold = cv_n_splits is not None and 1 < int(cv_n_splits) < len(split_groups)
    splits = (
        group_kfold_indices(groups, n_splits=int(cv_n_splits), seed=random_seed)
        if use_kfold
        else (
            (np.where(groups != sid)[0], np.where(groups == sid)[0])
            for sid in split_groups
        )
    )

    for train_idx, test_idx in splits:
        train_df = sub.iloc[train_idx].copy()
        test_df = sub.iloc[test_idx].copy()
        # Feature selection is deliberately inside the outer loop: the held-out
        # participant group cannot influence MI/MML ranking or top-k choice.
        y_select = (train_df[visit_col].values == 2).astype(int)
        feats = select_features(selection_method, train_df[feats_present], y_select, feats_present, k=k)
        selected_by_fold.append(list(feats))

        X_train = train_df[feats].values if feats else np.zeros((len(train_df), 0))
        X_test = test_df[feats].values if feats else np.zeros((len(test_df), 0))
        X_train_s, X_test_s, _, _ = standardize_train_test(X_train, X_test)

        model = SRMGlobalLinear(ridge=ridge, covariance_shrinkage=covariance_shrinkage).fit(
            X_train_s,
            train_df[subject_col].values,
            train_df[visit_col].values,
        )
        scores = model.score(X_test_s)
        for sid, v, sc in zip(test_df[subject_col].values, test_df[visit_col].values, scores):
            oof_rows.append({subject_col: sid, visit_col: int(v), "score": float(sc)})

    oof_df = pd.DataFrame(oof_rows)
    deltas = paired_deltas_from_long(
        oof_df.rename(columns={"score": "value"}), subject_col, visit_col, "value"
    )
    d_out = compute_cohens_d(deltas)
    srm_out = compute_srm(deltas)
    if compute_ci:
        _, d_lo, d_hi = bootstrap_ci_d(
            oof_df.rename(columns={"score": "value"}), subject_col, visit_col, "value"
        )
    else:
        d_lo = np.nan
        d_hi = np.nan
    return {
        "oof_df": oof_df,
        "n_subjects": int(d_out["n"]),
        "d_score": d_out["d"],
        "srm": srm_out["srm"],
        "d_ci_low": d_lo,
        "d_ci_high": d_hi,
        "selected_features_by_fold": selected_by_fold,
        "cv_n_splits": int(cv_n_splits) if use_kfold else int(len(split_groups)),
        "cv_mode": "group_kfold" if use_kfold else "loo",
        "split_group_col": resolved_split_group_col,
        "n_split_groups": int(len(split_groups)),
        "ridge": float(ridge),
        "covariance_shrinkage": float(covariance_shrinkage),
    }


__all__ = ["SRMGlobalLinear", "srm_global_loocv"]
