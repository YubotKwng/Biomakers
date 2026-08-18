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
from ..eval.intervals import adjacent_pair_interval_effect_summary, annual_tuning_diagnostics
from ..eval.model_selection import select_hierarchical_candidate
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
    start_visit: int = 1
    end_visit: int = 2
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
            np.asarray(subject_id),
            np.asarray(visit).astype(int),
            v1=int(self.start_visit),
            v2=int(self.end_visit),
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
    z_clip: float | None = None,
    cv_n_splits: int | None = None,
    random_seed: int = 42,
    compute_ci: bool = True,
    split_group_col: str | None = None,
    start_visit: int = 1,
    end_visit: int = 2,
    selection_params: dict | None = None,
) -> dict:
    """Run fold-safe subject-level validation for ``SRMGlobalLinear``.

    If ``cv_n_splits`` is ``None`` or at least the number of subjects, this is
    subject-level LOO. Otherwise it uses subject-grouped K-fold splits.
    ``z_clip`` optionally clips train-fold-standardised imaging values before
    fitting and scoring; the default ``None`` preserves historical behaviour.
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
    interval_visits = {int(start_visit), int(end_visit)}
    visit_int = pd.to_numeric(sub[visit_col], errors="coerce").astype("Int64")
    sub = sub[visit_int.isin(interval_visits)].copy()
    counts = sub.groupby(subject_col)[visit_col].agg(lambda s: interval_visits.issubset(set(pd.to_numeric(s, errors="coerce").dropna().astype(int))))
    valid_subjects = counts[counts].index
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
        y_select = (pd.to_numeric(train_df[visit_col], errors="coerce").values == int(end_visit)).astype(int)
        feats = select_features(
            selection_method,
            train_df[feats_present],
            y_select,
            feats_present,
            k=k,
            train_frame=train_df,
            subject_col=subject_col,
            visit_col=visit_col,
            **(selection_params or {}),
        )
        selected_by_fold.append(list(feats))

        X_train = train_df[feats].values if feats else np.zeros((len(train_df), 0))
        X_test = test_df[feats].values if feats else np.zeros((len(test_df), 0))
        X_train_s, X_test_s, _, _ = standardize_train_test(X_train, X_test)
        if z_clip is not None and X_train_s.shape[1] > 0:
            clip = float(z_clip)
            X_train_s = np.clip(X_train_s, -clip, clip)
            X_test_s = np.clip(X_test_s, -clip, clip)

        model = SRMGlobalLinear(
            ridge=ridge,
            covariance_shrinkage=covariance_shrinkage,
            start_visit=int(start_visit),
            end_visit=int(end_visit),
        ).fit(
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
        "z_clip": None if z_clip is None else float(z_clip),
        "start_visit": int(start_visit),
        "end_visit": int(end_visit),
    }


def _srm_fit_score_fold(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feats: Sequence[str],
    subject_col: str,
    visit_col: str,
    *,
    ridge: float,
    covariance_shrinkage: float,
    z_clip: float | None,
    start_visit: int = 1,
    end_visit: int = 2,
) -> pd.DataFrame:
    """Fit SRM on one training split and score one held-out split."""
    X_train = train_df[list(feats)].values if feats else np.zeros((len(train_df), 0))
    X_test = test_df[list(feats)].values if feats else np.zeros((len(test_df), 0))
    X_train_s, X_test_s, _, _ = standardize_train_test(X_train, X_test)
    if z_clip is not None and X_train_s.shape[1] > 0:
        clip = float(z_clip)
        X_train_s = np.clip(X_train_s, -clip, clip)
        X_test_s = np.clip(X_test_s, -clip, clip)
    model = SRMGlobalLinear(
        ridge=ridge,
        covariance_shrinkage=covariance_shrinkage,
        start_visit=int(start_visit),
        end_visit=int(end_visit),
    ).fit(
        X_train_s,
        train_df[subject_col].values,
        train_df[visit_col].values,
    )
    scores = model.score(X_test_s)
    return pd.DataFrame({
        subject_col: test_df[subject_col].values,
        visit_col: test_df[visit_col].astype(int).values,
        "score": scores.astype(float),
    })


def srm_global_nested_loocv(
    df_long: pd.DataFrame,
    feature_cols: Sequence[str],
    subject_col: str,
    visit_col: str = "visit",
    *,
    candidates: Sequence[dict] | None = None,
    selection_method: str = "none",
    k: int = 8,
    cv_n_splits: int | None = None,
    inner_folds: int = 5,
    random_seed: int = 42,
    compute_ci: bool = True,
    split_group_col: str | None = None,
    start_visit: int = 1,
    end_visit: int = 2,
    tuning_metric: str = "annual_mean_dz",
) -> dict:
    """Nested subject-level SRM validation with train-fold parameter tuning.

    Candidate ridge, covariance-shrinkage, clipping, and selection settings are
    chosen inside each outer training fold by inner grouped CV. The outer test
    fold is then scored once with the selected candidate.
    """
    feats_present = [f for f in feature_cols if f in df_long.columns]
    if not feats_present:
        return {
            "oof_df": pd.DataFrame(),
            "chosen_params_df": pd.DataFrame(),
            "n_subjects": 0,
            "d_score": np.nan,
            "srm": np.nan,
            "d_ci_low": np.nan,
            "d_ci_high": np.nan,
            "selected_features_by_fold": [],
        }
    if candidates is None:
        candidates = [
            {"ridge": 0.0, "covariance_shrinkage": s, "z_clip": z, "selection_method": selection_method, "k": k}
            for s in (0.35, 0.4, 0.45)
            for z in (None, 2.75, 3.0, 3.25)
        ]

    assert_training_frame_is_patient_only(df_long, feats_present)
    resolved_split_group_col = resolve_split_group_col(df_long, subject_col, split_group_col)
    cols = [subject_col, visit_col] + feats_present
    if resolved_split_group_col not in cols:
        cols.append(resolved_split_group_col)
    sub = df_long[cols].dropna().copy()
    interval_visits = {int(start_visit), int(end_visit)}
    visit_int = pd.to_numeric(sub[visit_col], errors="coerce").astype("Int64")
    sub = sub[visit_int.isin(interval_visits)].copy()
    counts = sub.groupby(subject_col)[visit_col].agg(lambda s: interval_visits.issubset(set(pd.to_numeric(s, errors="coerce").dropna().astype(int))))
    sub = sub[sub[subject_col].isin(counts[counts].index)].copy()

    groups = sub[resolved_split_group_col].values
    split_groups = np.asarray(sub[resolved_split_group_col].unique())
    use_kfold = cv_n_splits is not None and 1 < int(cv_n_splits) < len(split_groups)
    outer_splits = (
        group_kfold_indices(groups, n_splits=int(cv_n_splits), seed=random_seed)
        if use_kfold
        else (
            (np.where(groups != sid)[0], np.where(groups == sid)[0])
            for sid in split_groups
        )
    )

    oof_parts: list[pd.DataFrame] = []
    chosen_rows: list[dict] = []
    selected_by_fold: list[list[str]] = []

    for outer_fold, (train_idx, test_idx) in enumerate(outer_splits, start=1):
        train_df = sub.iloc[train_idx].copy()
        test_df = sub.iloc[test_idx].copy()
        train_groups = train_df[resolved_split_group_col].values
        inner_scores: list[dict] = []
        for cand_idx, cand in enumerate(candidates, start=1):
            cand = dict(cand)
            cand_method = cand.get("selection_method", selection_method)
            cand_k = int(cand.get("k", k))
            fold_ds = []
            inner_pred_parts = []
            inner_selected: list[list[str]] = []
            for inner_train_idx, inner_val_idx in group_kfold_indices(
                train_groups,
                n_splits=inner_folds,
                seed=random_seed + outer_fold,
            ):
                inner_train = train_df.iloc[inner_train_idx].copy()
                inner_val = train_df.iloc[inner_val_idx].copy()
                y_inner_select = (
                    pd.to_numeric(inner_train[visit_col], errors="coerce").values == int(end_visit)
                ).astype(int)
                inner_feats = select_features(
                    cand_method,
                    inner_train[feats_present],
                    y_inner_select,
                    feats_present,
                    k=cand_k,
                    train_frame=inner_train,
                    subject_col=subject_col,
                    visit_col=visit_col,
                    mrmr_redundancy_lambda=float(cand.get("mrmr_redundancy_lambda", 0.25)),
                    sparse_lambda=float(cand.get("sparse_lambda", 0.01)),
                    sparse_alpha=float(cand.get("sparse_alpha", 0.5)),
                    sparse_tolerance=float(cand.get("sparse_tolerance", 1e-8)),
                )
                inner_selected.append(list(inner_feats))
                if not inner_feats:
                    continue
                pred_df = _srm_fit_score_fold(
                    inner_train,
                    inner_val,
                    inner_feats,
                    subject_col,
                    visit_col,
                    ridge=float(cand.get("ridge", 0.0)),
                    covariance_shrinkage=float(cand.get("covariance_shrinkage", 0.0)),
                    z_clip=cand.get("z_clip"),
                    start_visit=int(start_visit),
                    end_visit=int(end_visit),
                )
                deltas = paired_deltas_from_long(
                    pred_df.rename(columns={"score": "value"}), subject_col, visit_col, "value"
                )
                d_val = compute_cohens_d(deltas)["d"]
                if np.isfinite(d_val):
                    fold_ds.append(float(d_val))
                inner_pred_parts.append(pred_df)
            if not inner_pred_parts:
                inner_scores.append({
                    "candidate_idx": cand_idx,
                    "mean_validation_dz": float("-inf"),
                    "mean_validation_annual_dz": float("-inf"),
                    "se_validation_dz": 0.0,
                    "feature_count": np.inf,
                    "candidate": cand,
                    "inner_selected_features": inner_selected,
                })
                continue
            if str(tuning_metric) == "annual_mean_dz" and inner_pred_parts:
                inner_oof = pd.concat(inner_pred_parts, ignore_index=True)
                inner_intervals = adjacent_pair_interval_effect_summary(
                    inner_oof,
                    pair_col=subject_col,
                    visit_col=visit_col,
                    score_col="score",
                    n_boot=100,
                    seed=random_seed + outer_fold + cand_idx,
                )
                annual_diag = annual_tuning_diagnostics(inner_intervals)
                mean_d = annual_diag["mean_validation_annual_dz"]
                if not np.isfinite(mean_d):
                    mean_d = float("-inf")
            else:
                annual_diag = {}
                mean_d = float(np.mean(fold_ds)) if fold_ds else float("-inf")
            se_d = float(np.std(fold_ds, ddof=1) / np.sqrt(len(fold_ds))) if len(fold_ds) > 1 else 0.0
            feature_count = np.median([len(x) for x in inner_selected if x]) if inner_selected else np.inf
            inner_scores.append({
                "candidate_idx": cand_idx,
                "mean_validation_dz": mean_d,
                "mean_validation_annual_dz": annual_diag.get("mean_validation_annual_dz", mean_d),
                "dz_v1_v2": annual_diag.get("dz_v1_v2", np.nan),
                "dz_v2_v3": annual_diag.get("dz_v2_v3", np.nan),
                "annual_interval_gap": annual_diag.get("annual_interval_gap", np.nan),
                "p_progression": annual_diag.get("p_progression", np.nan),
                "se_validation_dz": se_d,
                "feature_count": feature_count,
                "candidate": cand,
                "inner_selected_features": inner_selected,
            })
        inner_score_df = pd.DataFrame(inner_scores)
        if str(tuning_metric) == "annual_mean_dz":
            choice = select_hierarchical_candidate(inner_score_df)
        else:
            choice = inner_score_df.loc[inner_score_df["mean_validation_dz"].astype(float).idxmax()]
        best_inner = float(choice["mean_validation_annual_dz"] if str(tuning_metric) == "annual_mean_dz" else choice["mean_validation_dz"])
        best_cand = dict(choice["candidate"])
        best_method = best_cand.get("selection_method", selection_method)
        best_k = int(best_cand.get("k", k))
        y_select = (pd.to_numeric(train_df[visit_col], errors="coerce").values == int(end_visit)).astype(int)
        best_feats = select_features(
            best_method,
            train_df[feats_present],
            y_select,
            feats_present,
            k=best_k,
            train_frame=train_df,
            subject_col=subject_col,
            visit_col=visit_col,
            mrmr_redundancy_lambda=float(best_cand.get("mrmr_redundancy_lambda", 0.25)),
            sparse_lambda=float(best_cand.get("sparse_lambda", 0.01)),
            sparse_alpha=float(best_cand.get("sparse_alpha", 0.5)),
            sparse_tolerance=float(best_cand.get("sparse_tolerance", 1e-8)),
        )
        selected_by_fold.append(list(best_feats))
        pred_df = _srm_fit_score_fold(
            train_df,
            test_df,
            best_feats,
            subject_col,
            visit_col,
            ridge=float(best_cand.get("ridge", 0.0)),
            covariance_shrinkage=float(best_cand.get("covariance_shrinkage", 0.0)),
            z_clip=best_cand.get("z_clip"),
            start_visit=int(start_visit),
            end_visit=int(end_visit),
        )
        oof_parts.append(pred_df)
        chosen_rows.append({
            "outer_fold": outer_fold,
            "inner_d_score": best_inner,
            "inner_tuning_metric": tuning_metric,
            "inner_dz_v1_v2": choice.get("dz_v1_v2", np.nan),
            "inner_dz_v2_v3": choice.get("dz_v2_v3", np.nan),
            "inner_annual_interval_gap": choice.get("annual_interval_gap", np.nan),
            "inner_p_progression": choice.get("p_progression", np.nan),
            "n_features": len(best_feats),
            **best_cand,
        })

    oof_df = pd.concat(oof_parts, ignore_index=True) if oof_parts else pd.DataFrame()
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
        "chosen_params_df": pd.DataFrame(chosen_rows),
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
        "inner_folds": int(inner_folds),
        "start_visit": int(start_visit),
        "end_visit": int(end_visit),
        "tuning_metric": tuning_metric,
    }


def srm_global_repeated_group_cv(
    df_long: pd.DataFrame,
    feature_cols: Sequence[str],
    subject_col: str,
    visit_col: str = "visit",
    *,
    n_splits: int = 5,
    n_repeats: int = 10,
    random_seed: int = 42,
    **kwargs,
) -> dict:
    """Run repeated grouped outer CV by reusing the fold-safe SRM evaluator."""
    rows = []
    oof_parts = []
    selected = []
    for repeat in range(1, int(n_repeats) + 1):
        res = srm_global_loocv(
            df_long,
            feature_cols,
            subject_col,
            visit_col=visit_col,
            cv_n_splits=int(n_splits),
            random_seed=random_seed + repeat - 1,
            **kwargs,
        )
        rows.append({
            "repeat": repeat,
            "d_score": res["d_score"],
            "srm": res["srm"],
            "n_subjects": res["n_subjects"],
            "d_ci_low": res["d_ci_low"],
            "d_ci_high": res["d_ci_high"],
            "cv_n_splits": res["cv_n_splits"],
            "start_visit": res.get("start_visit", 1),
            "end_visit": res.get("end_visit", 2),
        })
        if not res["oof_df"].empty:
            tmp = res["oof_df"].copy()
            tmp["repeat"] = repeat
            oof_parts.append(tmp)
        selected.extend(res["selected_features_by_fold"])
    summary = pd.DataFrame(rows)
    return {
        "summary_df": summary,
        "oof_df": pd.concat(oof_parts, ignore_index=True) if oof_parts else pd.DataFrame(),
        "selected_features_by_fold": selected,
        "mean_d_score": float(summary["d_score"].mean()) if not summary.empty else np.nan,
        "se_d_score": float(summary["d_score"].std(ddof=1) / np.sqrt(len(summary))) if len(summary) > 1 else np.nan,
        "n_repeats": int(n_repeats),
        "n_splits": int(n_splits),
    }


__all__ = [
    "SRMGlobalLinear",
    "srm_global_loocv",
    "srm_global_nested_loocv",
    "srm_global_repeated_group_cv",
]
