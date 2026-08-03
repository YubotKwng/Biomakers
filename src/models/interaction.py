"""Interaction-term linear composite biomarker (patient-adaptive weighting).

Implements a linear composite score with subject-level modulators acting only
as *weight modulators* via multiplicative interactions (no raw Z main effects):

    score_it = X_it @ beta + (X_it * Z_i) @ gamma

where X is imaging, Z are subject-level covariates (e.g. GAA1, age_at_onset,
dur), and beta/gamma are learned weights.

Fitting objective (small-sample, interpretable, torch-free)
-----------------------------------------------------------
The training target is longitudinal sensitivity (paired Cohen's d_z / SRM).
We approximate a stable, sparse direction for change by fitting an ElasticNet
to per-subject paired differences of the *standardised* design:

    Δdesign_i = design_i(v2) - design_i(v1)
    fit w:  Δdesign_i @ w ≈ 1

When fit_intercept=False and y is constant, this finds a sparse direction in
feature space aligned with the mean change vector, with shrinkage for stability.
Hyperparameters can be tuned by inner subject-group CV to maximise SRM on
held-out subjects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet

from ..config import Config, DEFAULT_CONFIG
from ..eval.cv import group_kfold_indices
from ..eval.metrics import paired_cohens_d
from ..features.interactions import expand_interactions


def _paired_rows_by_subject(
    subject_id: np.ndarray,
    visit: np.ndarray,
    *,
    v1: int = 1,
    v2: int = 2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return aligned (rows_v1, rows_v2, common_subjects)."""
    df = pd.DataFrame({"_subject": subject_id, "_visit": visit})
    df["_row"] = np.arange(len(df))
    order = df.sort_values(["_subject", "_visit"])
    v1_rows = order[order["_visit"] == v1][["_subject", "_row"]].set_index("_subject")
    v2_rows = order[order["_visit"] == v2][["_subject", "_row"]].set_index("_subject")
    common = v1_rows.index.intersection(v2_rows.index)
    rows_v1 = v1_rows.loc[common]["_row"].to_numpy()
    rows_v2 = v2_rows.loc[common]["_row"].to_numpy()
    return rows_v1, rows_v2, common.to_numpy()


@dataclass
class InteractionLinearComposite:
    """Sparse linear composite with X⊗Z interactions (ElasticNet on Δdesign)."""

    config: Config = field(default_factory=lambda: DEFAULT_CONFIG)

    coef_: Optional[np.ndarray] = field(default=None, init=False)
    coef_main_: Optional[np.ndarray] = field(default=None, init=False)
    coef_interaction_: Optional[np.ndarray] = field(default=None, init=False)
    feature_names_x_: Optional[list[str]] = field(default=None, init=False)
    feature_names_z_: Optional[list[str]] = field(default=None, init=False)
    design_columns_: Optional[list[str]] = field(default=None, init=False)
    x_mean_: Optional[pd.Series] = field(default=None, init=False)
    x_sd_: Optional[pd.Series] = field(default=None, init=False)
    z_mean_: Optional[pd.Series] = field(default=None, init=False)
    z_sd_: Optional[pd.Series] = field(default=None, init=False)
    best_params_: Optional[dict] = field(default=None, init=False)

    # ------------------------------------------------------------------
    # Standardisation + design
    # ------------------------------------------------------------------
    def _standardise(
        self, X: pd.DataFrame, Z: pd.DataFrame, *, fit: bool
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        eps = self.config.interaction_eps
        if fit:
            self.x_mean_ = X.mean(axis=0)
            self.x_sd_ = X.std(axis=0, ddof=0).clip(lower=eps)
            self.z_mean_ = Z.mean(axis=0)
            self.z_sd_ = Z.std(axis=0, ddof=0).clip(lower=eps)
        if self.x_mean_ is None or self.x_sd_ is None or self.z_mean_ is None or self.z_sd_ is None:
            raise RuntimeError("Standardiser not fit; call fit() first.")
        X_std = (X - self.x_mean_) / self.x_sd_
        Z_std = (Z - self.z_mean_) / self.z_sd_
        return X_std, Z_std

    def _build_design(self, X_std: pd.DataFrame, Z_std: pd.DataFrame) -> np.ndarray:
        n, p = X_std.shape
        inter = expand_interactions(X_std, Z_std)
        intercept = np.ones((n, 1), dtype=float)
        D = np.concatenate([intercept, X_std.values, inter.values], axis=1)
        self.design_columns_ = ["intercept"] + list(X_std.columns) + list(inter.columns)
        return D

    # ------------------------------------------------------------------
    # Training objective on paired differences
    # ------------------------------------------------------------------
    def _delta_design(
        self,
        D: np.ndarray,
        subject_id: np.ndarray,
        visit: np.ndarray,
    ) -> np.ndarray:
        rows_v1, rows_v2, _ = _paired_rows_by_subject(subject_id, visit)
        Delta = D[rows_v2] - D[rows_v1]
        # Drop intercept column (all zeros after differencing).
        return Delta[:, 1:]

    def _fit_en(self, Delta: np.ndarray, *, alpha: float, l1_ratio: float) -> np.ndarray:
        y = np.ones(Delta.shape[0], dtype=float)
        model = ElasticNet(
            alpha=float(alpha),
            l1_ratio=float(l1_ratio),
            fit_intercept=False,
            max_iter=20000,
            random_state=self.config.random_state,
        )
        model.fit(Delta, y)
        w_eff = model.coef_.astype(float, copy=False)
        # Pad intercept (0.0) so downstream code can keep [1|X|X⊗Z] convention.
        return np.concatenate([[0.0], w_eff])

    def _tune_by_inner_cv_srm(
        self,
        D: np.ndarray,
        subject_id: np.ndarray,
        visit: np.ndarray,
        *,
        alpha_grid: Iterable[float],
        l1_ratio_grid: Iterable[float],
    ) -> tuple[float, float]:
        """Select (alpha, l1_ratio) maximising mean inner-fold SRM on Δscores."""
        cfg = self.config
        rows_v1, rows_v2, common_subjects = _paired_rows_by_subject(subject_id, visit)
        # Paired-diff dataset is one row per subject.
        Delta = (D[rows_v2] - D[rows_v1])[:, 1:]
        groups = common_subjects

        best = (-np.inf, None, None)
        for a in alpha_grid:
            for l1 in l1_ratio_grid:
                fold_ds = []
                for tr_idx, va_idx in group_kfold_indices(
                    groups, n_splits=cfg.interaction_inner_cv_splits, seed=cfg.random_state
                ):
                    w = self._fit_en(Delta[tr_idx], alpha=a, l1_ratio=l1)
                    delta_scores = Delta[va_idx] @ w[1:]
                    # SRM / d_z on Δscores for validation subjects.
                    d = float(delta_scores.mean() / (delta_scores.std(ddof=1) + cfg.interaction_eps)) \
                        if len(delta_scores) >= 2 else float("nan")
                    if np.isfinite(d):
                        fold_ds.append(d)
                mean_d = float(np.mean(fold_ds)) if fold_ds else float("-inf")
                if mean_d > best[0]:
                    best = (mean_d, float(a), float(l1))
        if best[1] is None or best[2] is None:
            return float(cfg.interaction_en_alpha), float(cfg.interaction_en_l1_ratio)
        return float(best[1]), float(best[2])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(
        self,
        X: pd.DataFrame,
        Z: pd.DataFrame,
        subject_id: pd.Series | np.ndarray,
        visit: pd.Series | np.ndarray,
    ) -> "InteractionLinearComposite":
        subject_arr = np.asarray(subject_id)
        visit_arr = np.asarray(visit).astype(int)

        X_std, Z_std = self._standardise(X, Z, fit=True)
        D = self._build_design(X_std, Z_std)

        if self.config.interaction_tune_inner_cv:
            best_alpha, best_l1 = self._tune_by_inner_cv_srm(
                D,
                subject_arr,
                visit_arr,
                alpha_grid=self.config.interaction_en_alpha_grid,
                l1_ratio_grid=self.config.interaction_en_l1_ratio_grid,
            )
        else:
            best_alpha, best_l1 = float(self.config.interaction_en_alpha), float(self.config.interaction_en_l1_ratio)

        Delta = self._delta_design(D, subject_arr, visit_arr)
        w_final = self._fit_en(Delta, alpha=best_alpha, l1_ratio=best_l1)

        self.feature_names_x_ = list(X.columns)
        self.feature_names_z_ = list(Z.columns)
        p = len(self.feature_names_x_)
        q = len(self.feature_names_z_)
        self.coef_ = w_final
        self.coef_main_ = w_final[1: 1 + p]
        self.coef_interaction_ = w_final[1 + p:].reshape(p, q)
        self.best_params_ = {"alpha": best_alpha, "l1_ratio": best_l1}
        return self

    def score(self, X: pd.DataFrame, Z: pd.DataFrame) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("fit() must be called before score()")
        X_std, Z_std = self._standardise(X, Z, fit=False)
        D = self._build_design(X_std, Z_std)
        return D @ self.coef_

    def imaging_weights(self, z_star: dict) -> pd.Series:
        if self.coef_main_ is None or self.coef_interaction_ is None:
            raise RuntimeError("fit() must be called before imaging_weights()")
        if self.z_mean_ is None or self.z_sd_ is None:
            raise RuntimeError("Standardiser not fit; call fit() first.")
        z_raw = pd.Series(
            {k: float(z_star.get(k, self.z_mean_[k])) for k in self.feature_names_z_}  # type: ignore[arg-type]
        )
        z_std = (z_raw - self.z_mean_) / self.z_sd_
        beta = self.coef_main_ + self.coef_interaction_ @ z_std.values
        return pd.Series(beta, index=self.feature_names_x_, name="beta_at_z_star")


__all__ = ["InteractionLinearComposite"]

