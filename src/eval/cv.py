"""Cross-validation utilities.
"""
from __future__ import annotations

import math
from typing import Iterator, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from .metrics import compute_cohens_d_from_oof
from .intervals import adjacent_pair_interval_effect_summary, annual_tuning_diagnostics
from .model_selection import select_hierarchical_candidate
from ..data.qc import _as_float_array, standardize_train_test, tukey_outliers_mask
from ..data.model_safety import assert_training_frame_is_patient_only
from ..features.selection import select_features


# ---------------------------------------------------------------------------
# Group-aware splitters
# ---------------------------------------------------------------------------
def group_kfold_indices(
    groups,
    n_splits: int = 5,
    seed: int = 42,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Shuffle groups and yield group-disjoint train/validation indices.

    Every row for a given subject or pair is assigned to one fold, which
    prevents within-subject leakage in longitudinal validation.
    """
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    if len(uniq) < 2:
        return
    # Shuffle unique groups, not rows, so all rows belonging to one participant
    # or interval group remain together in the same fold.
    rs = np.random.RandomState(seed)
    rs.shuffle(uniq)
    folds = np.array_split(uniq, min(n_splits, len(uniq)))
    for fold_groups in folds:
        val_mask = np.isin(groups, fold_groups)
        train_idx = np.where(~val_mask)[0]
        val_idx = np.where(val_mask)[0]
        if len(np.unique(groups[val_idx])) == 0 or len(np.unique(groups[train_idx])) == 0:
            continue
        yield train_idx, val_idx


def subject_loo_split(groups) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Subject-wise leave-one-out splitter.

    For each unique subject or pair, yields ``(train_idx, test_idx)`` so the
    test fold contains all rows for that entity and the training fold contains
    none of them.
    """
    groups = np.asarray(groups)
    subjects = np.unique(groups)
    for subj in subjects:
        test_mask = groups == subj
        train_idx = np.where(~test_mask)[0]
        test_idx = np.where(test_mask)[0]
        yield train_idx, test_idx


def resolve_split_group_col(
    df: pd.DataFrame,
    subject_col: str,
    split_group_col: str | None = None,
) -> str:
    """Return the column used to keep related progression pairs in one fold.

    TRACK-FA long data uses ``pair_id`` for one progression interval
    (for example, ``AAN001_V1V2``) and ``subject`` for the participant
    (for example, ``AAN001``). Evaluation must keep ``pair_id`` for paired
    deltas, but splitting should use ``subject`` so V1V2 and V2V3 from the
    same participant are either both training or both held out.
    """
    if split_group_col is not None:
        if split_group_col not in df.columns:
            raise KeyError(f"df missing split_group_col {split_group_col!r}")
        return split_group_col
    if subject_col != "subject" and "subject" in df.columns:
        return "subject"
    return subject_col


def nested_groupkfold_inner(
    groups_train,
    n_splits: int = 5,
    seed: int = 42,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Inner-CV splitter for nested CV — wraps ``group_kfold_indices``.

    Centralises the inner-CV step used by ``select_params_by_inner_cv_d``
    and ``tune_and_run_regression_loocv``. Yields ``(train_idx, val_idx)``
    pairs with subjects fully contained in one fold.
    """
    yield from group_kfold_indices(groups_train, n_splits=n_splits, seed=seed)


# ---------------------------------------------------------------------------
# Bootstrap and validation split helpers.
# ---------------------------------------------------------------------------
def bootstrap_subjects(df: pd.DataFrame, subject_col: str, n_boot: int, seed: int = 42):
    """Yield bootstrap data frames by sampling subjects with replacement."""
    subjects = df[subject_col].unique()
    rng = np.random.RandomState(seed)
    for _ in range(n_boot):
        samp = rng.choice(subjects, size=len(subjects), replace=True)
        yield df[df[subject_col].isin(samp)].copy()


def split_train_val_subjects(
    train_df: pd.DataFrame,
    subject_col: str,
    val_fraction: float = 0.2,
    seed: int = 42,
    split_group_col: str | None = None,
):
    """Create a reproducible subject-level train/validation split.

    This is used inside DL training folds for early stopping. The validation
    split is carved only from the current training fold.
    """
    resolved_split_group_col = resolve_split_group_col(train_df, subject_col, split_group_col)
    subjects = np.array(sorted(train_df[resolved_split_group_col].unique()))
    rng = np.random.RandomState(seed)
    rng.shuffle(subjects)
    n_val = max(1, int(round(len(subjects) * val_fraction)))
    val_subjects = set(subjects[:n_val])
    train_subjects = set(subjects[n_val:])
    if len(train_subjects) == 0:
        train_subjects = val_subjects
        val_subjects = set(subjects[:1])
    train_split = train_df[train_df[resolved_split_group_col].isin(train_subjects)].copy()
    val_split = train_df[train_df[resolved_split_group_col].isin(val_subjects)].copy()
    return train_split, val_split


# ---------------------------------------------------------------------------
# ElasticNet cross-validation runners.
# ---------------------------------------------------------------------------
def run_cv_for_combination(
    df: pd.DataFrame,
    target_col: str,
    combo: dict,
    *,
    alphas: Sequence[float],
    lambdas: Sequence[float],
    n_splits: int = 5,
    n_repeats: int = 5,
    random_seed: int = 42,
):
    """Run grouped cross-validation with per-fold ElasticNet tuning.

    Rows are grouped by subject identifier so repeated measures from the same
    participant cannot appear in both training and validation folds.
    """
    feature_cols: list[str] = []
    for domain in combo["domains"]:
        feature_cols.extend(domain)
    assert_training_frame_is_patient_only(df, feature_cols, target_col=target_col)

    sub = df.dropna(subset=feature_cols + [target_col]).copy()
    n_before = len(sub)

    outlier_mask = tukey_outliers_mask(sub, feature_cols, k=3.0)
    sub = sub[~outlier_mask].copy()
    n_after = len(sub)
    group_col = (
        "subject" if "subject" in sub.columns else (
            "melb_id" if "melb_id" in sub.columns else (
                "ID" if "ID" in sub.columns else None)))
    if group_col is None:
        raise KeyError("No subject/group id column found for GroupKFold")

    groups = sub[group_col].values
    X = sub[feature_cols].values
    y = sub[target_col].values

    all_preds: list = []
    all_true: list = []
    best_alphas: list = []
    best_l1s: list = []

    for r in range(n_repeats):
        rng = np.random.RandomState(random_seed + r)
        unique_groups = np.unique(groups)
        rng.shuffle(unique_groups)

        group_order = {g: i for i, g in enumerate(unique_groups)}
        order = np.argsort([group_order[g] for g in groups])
        Xr, yr, gr = X[order], y[order], groups[order]

        gkf = GroupKFold(n_splits=n_splits)
        for train_idx, val_idx in gkf.split(Xr, yr, gr):
            X_train, X_val = Xr[train_idx], Xr[val_idx]
            y_train, y_val = yr[train_idx], yr[val_idx]

            xs = StandardScaler()
            ys = StandardScaler()
            X_train_s = xs.fit_transform(X_train)
            X_val_s = xs.transform(X_val)
            y_train_s = ys.fit_transform(y_train.reshape(-1, 1)).ravel()

            best_rmse = np.inf
            best_pred = None
            best_a = None
            best_l = None

            for a in alphas:
                for l in lambdas:
                    model = ElasticNet(alpha=l, l1_ratio=a, max_iter=5000, random_state=random_seed)
                    model.fit(X_train_s, y_train_s)
                    y_pred_s = model.predict(X_val_s)
                    y_pred = ys.inverse_transform(y_pred_s.reshape(-1, 1)).ravel()

                    rmse_val = math.sqrt(mean_squared_error(y_val, y_pred))
                    if rmse_val < best_rmse:
                        best_rmse = rmse_val
                        best_pred = y_pred
                        best_a = a
                        best_l = l

            all_preds.extend(best_pred)
            all_true.extend(y_val)
            if best_a is not None and best_l is not None:
                best_l1s.append(best_a)
                best_alphas.append(best_l)

    r2_value = r2_score(all_true, all_preds)
    rmse_value = math.sqrt(mean_squared_error(all_true, all_preds))

    def _mode(vals):
        if not vals:
            return None
        return max(set(vals), key=vals.count)

    best_alpha_mode = _mode(best_alphas)
    best_l1_mode = _mode(best_l1s)
    best_alpha_mean = float(np.mean(best_alphas)) if best_alphas else None
    best_l1_mean = float(np.mean(best_l1s)) if best_l1s else None
    return r2_value, rmse_value, len(sub), n_before, n_after, best_alpha_mode, best_l1_mode, best_alpha_mean, best_l1_mean


def run_cv_loocv_with_coefs(
    df_long: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str,
    subject_col: str,
    *,
    l1_ratios: Optional[Sequence[float]] = None,
    reg_strengths: Optional[Sequence[float]] = None,
    random_seed: int = 42,
):
    """LOOCV with per-fold ElasticNet hyperparameter sweep + coef storage.

    Fits ElasticNet models inside subject-level leave-one-out validation.
    (``RANDOM_SEED`` parameterised as ``random_seed``).
    """
    if l1_ratios is None:
        l1_ratios = np.linspace(0, 1, 11)
    if reg_strengths is None:
        reg_strengths = np.linspace(0, 20, 201)

    assert_training_frame_is_patient_only(df_long, feature_cols, target_col=target_col)
    sub = df_long.dropna(subset=list(feature_cols) + [target_col]).copy()
    sub = sub[~tukey_outliers_mask(sub, feature_cols, k=3.0)].copy()

    groups = sub[subject_col].values
    X = sub[list(feature_cols)].values
    y = sub[target_col].values

    n_subjects = len(np.unique(groups))
    gkf = GroupKFold(n_splits=n_subjects)

    all_preds: list = []
    all_true: list = []
    oof_rows: list = []
    coef_rows: list = []

    for train_idx, test_idx in gkf.split(X, y, groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        group_test = groups[test_idx]

        xs = StandardScaler()
        X_train_s = xs.fit_transform(X_train)
        X_test_s = xs.transform(X_test)

        best_rmse, best_pred, best_a, best_l = np.inf, None, None, None
        best_model = None
        for a in reg_strengths:
            for l in l1_ratios:
                m = ElasticNet(alpha=a, l1_ratio=l, max_iter=5000, random_state=random_seed)
                m.fit(X_train_s, y_train)
                pred = m.predict(X_test_s)
                rmse_val = np.sqrt(mean_squared_error(y_test, pred))
                if rmse_val < best_rmse:
                    best_rmse, best_pred, best_a, best_l = rmse_val, pred, a, l
                    best_model = m

        all_preds.extend(best_pred)
        all_true.extend(y_test)

        for sid, v, p in zip(group_test, sub.iloc[test_idx]["visit"], best_pred):
            oof_rows.append({subject_col: sid, "visit": v, "pred": p})

        coef_rows.append(best_model.coef_)

    r2_value = r2_score(all_true, all_preds)
    rmse_value = np.sqrt(mean_squared_error(all_true, all_preds))
    oof_df = pd.DataFrame(oof_rows)
    coef_mat = np.vstack(coef_rows) if coef_rows else np.zeros((0, len(feature_cols)))
    return r2_value, rmse_value, oof_df, coef_mat


def select_params_by_inner_cv_d(
    df_long_train: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str,
    subject_col: str,
    pair_col: str | None = None,
    *,
    inner_folds: int = 5,
    l1_ratios: Optional[Sequence[float]] = None,
    alphas: Optional[Sequence[float]] = None,
    random_seed: int = 42,
):
    """Inner CV that picks (alpha, l1_ratio) maximising paired Cohen's d.

    Select ElasticNet hyperparameters by inner subject-group validation.
    The outer held-out subject is already removed before this function is
    called, so parameter selection cannot see the outer test fold.
    """
    if l1_ratios is None:
        l1_ratios = [0.1, 0.5, 0.9]
    if alphas is None:
        alphas = np.logspace(-2, 2, 10)

    assert_training_frame_is_patient_only(df_long_train, feature_cols, target_col=target_col)
    sub = df_long_train.dropna(subset=list(feature_cols) + [target_col]).copy()
    sub = sub[~tukey_outliers_mask(sub, feature_cols, k=3.0)].copy()
    if len(sub) == 0:
        return None, None, pd.DataFrame(), np.nan, np.nan

    groups = sub[subject_col].values
    n_unique = len(np.unique(groups))
    n_splits = min(int(inner_folds), int(n_unique))
    if n_splits < 2:
        return None, None, pd.DataFrame(), np.nan, np.nan

    X = sub[list(feature_cols)].values
    y = sub[target_col].values

    best_d = -np.inf
    best_rmse = np.inf
    best_alpha = None
    best_l1 = None
    best_oof_df = pd.DataFrame()

    gkf = GroupKFold(n_splits=n_splits)
    for alpha in alphas:
        for l1 in l1_ratios:
            oof_rows = []
            for train_idx, val_idx in gkf.split(X, y, groups):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                groups_val = groups[val_idx]
                visits_val = sub.iloc[val_idx]["visit"].values
                pairs_val = (
                    sub.iloc[val_idx][pair_col].values
                    if pair_col is not None and pair_col in sub.columns
                    else groups_val
                )

                xs = StandardScaler()
                X_train_s = xs.fit_transform(X_train)
                X_val_s = xs.transform(X_val)

                m = ElasticNet(alpha=float(alpha), l1_ratio=float(l1), max_iter=5000, random_state=random_seed)
                m.fit(X_train_s, y_train)
                pred = m.predict(X_val_s)

                pair_key = pair_col or subject_col
                for pid, v, p, t in zip(pairs_val, visits_val, pred, y_val):
                    oof_rows.append({pair_key: pid, "visit": v, "pred": p, "true": t})

            oof_df = pd.DataFrame(oof_rows)
            pair_key = pair_col or subject_col
            # Optimise the same progression metric used for reporting, while
            # using RMSE only as a tie-breaker between equal d_z values.
            d = compute_cohens_d_from_oof(oof_df, pair_key, pred_col="pred", visit_col="visit")
            tmp = oof_df[["true", "pred"]].dropna()
            rmse_val = np.nan if len(tmp) == 0 else float(np.sqrt(mean_squared_error(tmp["true"], tmp["pred"])))

            d_val = -np.inf if pd.isna(d) else float(d)
            rmse_v = np.inf if pd.isna(rmse_val) else float(rmse_val)

            better = (d_val > best_d) or (
                d_val == best_d and (rmse_v < best_rmse)
            ) or (
                d_val == best_d and rmse_v == best_rmse and (best_alpha is None or float(alpha) < float(best_alpha))
            )

            if better:
                best_d = d_val
                best_rmse = rmse_v
                best_alpha = float(alpha)
                best_l1 = float(l1)
                best_oof_df = oof_df

    out_d = np.nan if best_d == -np.inf else float(best_d)
    out_rmse = np.nan if best_rmse == np.inf else float(best_rmse)
    return best_alpha, best_l1, best_oof_df, out_d, out_rmse


def run_loocv_d_tuned_with_coefs(
    df_long: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str,
    subject_col: str,
    pair_col: str | None = None,
    *,
    inner_folds: int = 5,
    l1_ratios: Optional[Sequence[float]] = None,
    alphas: Optional[Sequence[float]] = None,
    random_seed: int = 42,
) -> dict:
    """Outer LOOCV with per-fold inner-CV d-tuned hyperparameters.

    Select ElasticNet hyperparameters by inner subject-group validation.
    """
    if l1_ratios is None:
        l1_ratios = [0.1, 0.5, 0.9]
    if alphas is None:
        alphas = np.logspace(-2, 2, 10)

    assert_training_frame_is_patient_only(df_long, feature_cols, target_col=target_col)
    sub = df_long.dropna(subset=list(feature_cols) + [target_col]).copy()
    sub = sub[~tukey_outliers_mask(sub, feature_cols, k=3.0)].copy()

    n_rows = int(len(sub))
    if n_rows == 0:
        return {
            "d_score": np.nan,
            "rmse": np.nan,
            "r2": np.nan,
            "n_rows": 0,
            "n_subjects": 0,
            "alpha_mode": None,
            "l1_ratio_mode": None,
            "alpha_mean": None,
            "l1_ratio_mean": None,
            "oof_df": pd.DataFrame(),
            "coef_mat": np.zeros((0, len(feature_cols))),
            "chosen_params_df": pd.DataFrame(),
        }

    groups = sub[subject_col].values
    subjects = np.unique(groups)
    n_subjects = int(len(subjects))

    oof_rows = []
    coef_rows = []
    chosen_params_rows = []

    X_all = sub[list(feature_cols)].values
    y_all = sub[target_col].values

    for test_subject in subjects:
        test_mask = groups == test_subject
        train_mask = ~test_mask

        train_df = sub.loc[train_mask].copy()
        best_alpha, best_l1, _, inner_d, inner_rmse = select_params_by_inner_cv_d(
            train_df,
            feature_cols,
            target_col,
            subject_col,
            pair_col=pair_col,
            inner_folds=inner_folds,
            l1_ratios=l1_ratios,
            alphas=alphas,
            random_seed=random_seed,
        )

        if best_alpha is None or best_l1 is None:
            continue

        X_train, X_test = X_all[train_mask], X_all[test_mask]
        y_train, y_test = y_all[train_mask], y_all[test_mask]
        visits_test = sub.loc[test_mask, "visit"].values
        pairs_test = (
            sub.loc[test_mask, pair_col].values
            if pair_col is not None and pair_col in sub.columns
            else np.asarray([test_subject] * int(test_mask.sum()))
        )

        xs = StandardScaler()
        X_train_s = xs.fit_transform(X_train)
        X_test_s = xs.transform(X_test)

        m = ElasticNet(alpha=float(best_alpha), l1_ratio=float(best_l1), max_iter=5000, random_state=random_seed)
        m.fit(X_train_s, y_train)
        pred = m.predict(X_test_s)

        chosen_params_rows.append({
            "test_subject": test_subject,
            "alpha": float(best_alpha),
            "l1_ratio": float(best_l1),
            "inner_d": inner_d,
            "inner_rmse": inner_rmse,
        })

        pair_key = pair_col or subject_col
        for pid, v, p, t in zip(pairs_test, visits_test, pred, y_test):
            oof_rows.append({pair_key: pid, "visit": v, "pred": p, "true": t})

        coef_rows.append(m.coef_)

    oof_df = pd.DataFrame(oof_rows)
    coef_mat = np.vstack(coef_rows) if coef_rows else np.zeros((0, len(feature_cols)))
    chosen_params_df = pd.DataFrame(chosen_params_rows)

    pair_key = pair_col or subject_col
    d_score = compute_cohens_d_from_oof(oof_df, pair_key, pred_col="pred", visit_col="visit")
    tmp = oof_df[["true", "pred"]].dropna() if len(oof_df) else pd.DataFrame(columns=["true", "pred"])
    rmse_val = np.nan if len(tmp) == 0 else float(np.sqrt(mean_squared_error(tmp["true"], tmp["pred"])))
    r2_val = np.nan if len(tmp) < 2 else float(r2_score(tmp["true"], tmp["pred"]))

    def _mode(vals):
        vals = [v for v in vals if v is not None and np.isfinite(v)]
        if not vals:
            return None
        return max(set(vals), key=vals.count)

    alpha_mode = _mode(chosen_params_df["alpha"].tolist()) if "alpha" in chosen_params_df.columns else None
    l1_mode = _mode(chosen_params_df["l1_ratio"].tolist()) if "l1_ratio" in chosen_params_df.columns else None
    alpha_mean = float(np.mean(chosen_params_df["alpha"])) if "alpha" in chosen_params_df.columns and len(chosen_params_df) else None
    l1_mean = float(np.mean(chosen_params_df["l1_ratio"])) if "l1_ratio" in chosen_params_df.columns and len(chosen_params_df) else None

    return {
        "d_score": d_score,
        "rmse": rmse_val,
        "r2": r2_val,
        "n_rows": n_rows,
        "n_subjects": n_subjects,
        "alpha_mode": alpha_mode,
        "l1_ratio_mode": l1_mode,
        "alpha_mean": alpha_mean,
        "l1_ratio_mean": l1_mean,
        "oof_df": oof_df,
        "coef_mat": coef_mat,
        "chosen_params_df": chosen_params_df,
    }


# ---------------------------------------------------------------------------
# LDA and regression leave-one-out runners.
# ---------------------------------------------------------------------------
def lda_loocv(
    df_long_in: pd.DataFrame,
    feature_cols: Sequence[str],
    subject_col: str,
    visit_col: str = "visit",
    selection_fn=None,
    selection_method: str = "none",
    k: int = 8,
    cv_n_splits: int | None = None,
    random_seed: int = 42,
    shrink="auto",
    covariance_shrinkage: float = 0.0,
    z_clip: float | None = None,
    compute_ci: bool = True,
    split_group_col: str | None = None,
):
    """Run subject-level LDA visit separation.

    The model learns a direction that separates baseline from follow-up visits
    in the training subjects. Held-out visit scores are converted to paired
    deltas and evaluated with Cohen's d and SRM. Uses LOO by default, or
    subject-grouped K-fold when ``cv_n_splits`` is between 2 and n_subjects-1.
    ``z_clip`` optionally clips fold-standardised imaging values before LDA.
    """
    from .metrics import (
        bootstrap_ci_d,
        compute_cohens_d,
        compute_srm,
        paired_deltas_from_long,
    )

    # Import lazily to keep linear-model dependencies local to this runner.
    try:
        from ..models.lda import fit_lda_direction, predict_lda_scores
    except ImportError as exc:  # pragma: no cover - optional model backend wiring
        raise ImportError(
            "lda_loocv requires src.models.lda with fit_lda_direction and "
            "predict_lda_scores."
        ) from exc

    feats_present = [f for f in feature_cols if f in df_long_in.columns]
    if len(feats_present) == 0:
        return {
            "oof_df": pd.DataFrame(),
            "n_subjects": 0,
            "d_score": np.nan,
            "srm": np.nan,
            "d_ci_low": np.nan,
            "d_ci_high": np.nan,
            "selected_features_by_fold": [],
        }
    assert_training_frame_is_patient_only(df_long_in, feats_present)
    resolved_split_group_col = resolve_split_group_col(df_long_in, subject_col, split_group_col)
    cols = [subject_col, visit_col] + feats_present
    if resolved_split_group_col not in cols:
        cols.append(resolved_split_group_col)
    sub = df_long_in[cols].dropna().copy()
    counts = sub.groupby(subject_col)[visit_col].nunique()
    valid_subjects = counts[counts == 2].index
    sub = sub[sub[subject_col].isin(valid_subjects)].copy()

    split_groups = np.asarray(sub[resolved_split_group_col].unique())
    oof_rows = []
    selected_features_by_fold: list[list[str]] = []
    groups = sub[resolved_split_group_col].values
    # The split group can differ from the paired-delta subject column. In
    # TRACK-FA this lets evaluation form V1V2/V2V3 deltas by pair_id while CV
    # holds out all intervals for the same participant together.
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

        feats = list(feature_cols)
        if selection_fn is not None:
            feats = selection_fn(train_df)
        elif selection_method != "none":
            y_select = (train_df[visit_col].values == 2).astype(int)
            feats = select_features(selection_method, train_df[feats_present], y_select, feats_present, k=k)
        feats = [f for f in feats if f in train_df.columns]
        selected_features_by_fold.append(list(feats))
        if len(feats) == 0:
            continue

        X_train = train_df[feats].values
        y_train = (train_df[visit_col].values == 2).astype(int)
        X_test = test_df[feats].values

        X_train_s, X_test_s, _, _ = standardize_train_test(X_train, X_test)
        if z_clip is not None and X_train_s.shape[1] > 0:
            clip = float(z_clip)
            X_train_s = np.clip(X_train_s, -clip, clip)
            X_test_s = np.clip(X_test_s, -clip, clip)
        w = fit_lda_direction(
            X_train_s,
            y_train,
            shrink=shrink,
            covariance_shrinkage=covariance_shrinkage,
        )
        if w is None:
            continue
        train_scores = predict_lda_scores(X_train_s, w)
        orient_df = pd.DataFrame({
            "_subject": train_df[subject_col].values,
            "_visit": train_df[visit_col].values,
            "_score": train_scores,
        })
        train_d = compute_cohens_d_from_oof(
            orient_df,
            subject_col="_subject",
            pred_col="_score",
            visit_col="_visit",
        )
        if np.isfinite(train_d) and train_d < 0:
            w = -w
        scores = predict_lda_scores(X_test_s, w)

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
        "selected_features_by_fold": selected_features_by_fold,
        "cv_n_splits": int(cv_n_splits) if use_kfold else int(len(split_groups)),
        "cv_mode": "group_kfold" if use_kfold else "loo",
        "split_group_col": resolved_split_group_col,
        "n_split_groups": int(len(split_groups)),
        "shrink": shrink,
        "covariance_shrinkage": float(covariance_shrinkage),
        "z_clip": None if z_clip is None else float(z_clip),
    }


def lda_nested_loocv(
    df_long_in: pd.DataFrame,
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
    tuning_metric: str = "annual_mean_dz",
) -> dict:
    """Nested LDA validation with train-fold-only hyperparameter selection.

    LDA shrinkage, covariance shrinkage, clipping, and optional feature
    selection are chosen using inner grouped CV inside each outer training fold.
    """
    from .metrics import (
        bootstrap_ci_d,
        compute_cohens_d,
        compute_srm,
        paired_deltas_from_long,
    )
    from ..models.lda import fit_lda_direction, predict_lda_scores

    feats_present = [f for f in feature_cols if f in df_long_in.columns]
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
            {"shrink": sh, "covariance_shrinkage": cs, "z_clip": z, "selection_method": selection_method, "k": k}
            for cs in (0.75, 1.0)
            for sh in ("auto", 1e-8, 0.1, 1.0, 10.0)
            for z in (None, 4.0, 3.0)
        ]

    assert_training_frame_is_patient_only(df_long_in, feats_present)
    resolved_split_group_col = resolve_split_group_col(df_long_in, subject_col, split_group_col)
    cols = [subject_col, visit_col] + feats_present
    if resolved_split_group_col not in cols:
        cols.append(resolved_split_group_col)
    sub = df_long_in[cols].dropna().copy()
    counts = sub.groupby(subject_col)[visit_col].nunique()
    sub = sub[sub[subject_col].isin(counts[counts == 2].index)].copy()

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

    def fit_score(train_df: pd.DataFrame, test_df: pd.DataFrame, feats: Sequence[str], cand: dict) -> pd.DataFrame:
        X_train = train_df[list(feats)].values
        X_test = test_df[list(feats)].values
        y_train = (train_df[visit_col].values == 2).astype(int)
        X_train_s, X_test_s, _, _ = standardize_train_test(X_train, X_test)
        if cand.get("z_clip") is not None and X_train_s.shape[1] > 0:
            clip = float(cand["z_clip"])
            X_train_s = np.clip(X_train_s, -clip, clip)
            X_test_s = np.clip(X_test_s, -clip, clip)
        w = fit_lda_direction(
            X_train_s,
            y_train,
            shrink=cand.get("shrink", "auto"),
            covariance_shrinkage=float(cand.get("covariance_shrinkage", 0.0)),
        )
        if w is None:
            return pd.DataFrame(columns=[subject_col, visit_col, "score"])
        train_scores = predict_lda_scores(X_train_s, w)
        orient_df = pd.DataFrame({
            "_subject": train_df[subject_col].values,
            "_visit": train_df[visit_col].values,
            "_score": train_scores,
        })
        train_d = compute_cohens_d_from_oof(
            orient_df, subject_col="_subject", pred_col="_score", visit_col="_visit"
        )
        if np.isfinite(train_d) and train_d < 0:
            w = -w
        return pd.DataFrame({
            subject_col: test_df[subject_col].values,
            visit_col: test_df[visit_col].astype(int).values,
            "score": predict_lda_scores(X_test_s, w).astype(float),
        })

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
                y_inner_select = (inner_train[visit_col].values == 2).astype(int)
                inner_feats = (
                    select_features(cand_method, inner_train[feats_present], y_inner_select, feats_present, k=cand_k)
                    if cand_method != "none"
                    else list(feats_present)
                )
                inner_selected.append(list(inner_feats))
                if not inner_feats:
                    continue
                pred_df = fit_score(inner_train, inner_val, inner_feats, cand)
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
                    seed=random_seed + outer_fold,
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
        y_select = (train_df[visit_col].values == 2).astype(int)
        best_feats = (
            select_features(best_method, train_df[feats_present], y_select, feats_present, k=best_k)
            if best_method != "none"
            else list(feats_present)
        )
        selected_by_fold.append(list(best_feats))
        oof_parts.append(fit_score(train_df, test_df, best_feats, best_cand))
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
        "tuning_metric": tuning_metric,
    }


def interaction_loocv(
    df_long_in: pd.DataFrame,
    feature_cols: Sequence[str],
    modulator_cols: Sequence[str],
    subject_col: str,
    visit_col: str = "visit",
    *,
    selection_method: str = "none",
    k: int = 8,
    cv_n_splits: int | None = None,
    random_seed: int = 42,
    config=None,
    compute_ci: bool = True,
    split_group_col: str | None = None,
) -> dict:
    """Subject-level validation for ``InteractionLinearComposite``.

    Imaging features are selected fold-locally; patient modulators are always
    passed through unchanged and never selected by the imaging selector.
    """
    from .metrics import (
        bootstrap_ci_d,
        compute_cohens_d,
        compute_srm,
        paired_deltas_from_long,
    )
    from ..models.interaction import InteractionLinearComposite

    feats_present = [f for f in feature_cols if f in df_long_in.columns]
    mods_present = [z for z in modulator_cols if z in df_long_in.columns]
    if len(feats_present) == 0 or len(mods_present) == 0:
        return {
            "oof_df": pd.DataFrame(),
            "n_subjects": 0,
            "d_score": np.nan,
            "srm": np.nan,
            "d_ci_low": np.nan,
            "d_ci_high": np.nan,
            "selected_features_by_fold": [],
        }

    assert_training_frame_is_patient_only(df_long_in, feats_present)
    resolved_split_group_col = resolve_split_group_col(df_long_in, subject_col, split_group_col)
    cols = [subject_col, visit_col] + feats_present + mods_present
    if resolved_split_group_col not in cols:
        cols.append(resolved_split_group_col)
    sub = df_long_in[cols].dropna().copy()
    counts = sub.groupby(subject_col)[visit_col].nunique()
    valid_subjects = counts[counts == 2].index
    sub = sub[sub[subject_col].isin(valid_subjects)].copy()

    split_groups = np.asarray(sub[resolved_split_group_col].unique())
    oof_rows = []
    selected_features_by_fold: list[list[str]] = []
    groups = sub[resolved_split_group_col].values
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

        y_select = (train_df[visit_col].values == 2).astype(int)
        feats = select_features(selection_method, train_df[feats_present], y_select, feats_present, k=k)
        feats = [f for f in feats if f in train_df.columns]
        selected_features_by_fold.append(list(feats))
        if len(feats) == 0:
            continue

        model = InteractionLinearComposite(config=config) if config is not None else InteractionLinearComposite()
        model.fit(
            train_df[feats],
            train_df[mods_present],
            train_df[subject_col].values,
            train_df[visit_col].values,
            cv_group_id=train_df[resolved_split_group_col].values,
        )
        scores = model.score(test_df[feats], test_df[mods_present])
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
        "selected_features_by_fold": selected_features_by_fold,
        "cv_n_splits": int(cv_n_splits) if use_kfold else int(len(split_groups)),
        "cv_mode": "group_kfold" if use_kfold else "loo",
        "split_group_col": resolved_split_group_col,
        "n_split_groups": int(len(split_groups)),
    }


def tune_and_run_regression_loocv(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str,
    subject_col: str,
    model_kind: str,
    selection_fn=None,
    selection_method: str = "none",
    k: int = 8,
    visit_col: str = "visit",
    cv_n_splits: int | None = None,
    inner_folds: int = 3,
    alphas: Optional[Sequence[float]] = None,
    l1_ratios: Optional[Sequence[float]] = None,
    n_components_list: Optional[Sequence[int]] = None,
    random_seed: int = 42,
    param_selection_metric: str = "rmse",
    z_clip: float | None = None,
    compute_ci: bool = True,
    split_group_col: str | None = None,
):
    """Run subject-grouped regression with inner-CV hyperparameter tuning.

    Supports ElasticNet as the retained clinical-score prediction comparator
    under subject-level validation. Uses LOO by default, or grouped K-fold via
    ``cv_n_splits``. Inner hyperparameters can be selected by RMSE or, when
    visit rows are available, by the same paired Cohen's d_z used for
    progression reporting.
    ``z_clip`` optionally clips fold-standardised imaging values before
    fitting the ElasticNet backend.
    """
    from .metrics import r2 as _r2, rmse as _rmse  # local alias

    if model_kind != "elasticnet":
        raise ValueError("Clinical-score regression comparator is now ElasticNet-only")

    feats_present = [f for f in feature_cols if f in df.columns]
    if len(feats_present) == 0:
        return {
            "oof_df": pd.DataFrame(),
            "chosen_params_df": pd.DataFrame(),
            "n_subjects": 0,
            "rmse": np.nan,
            "r2": np.nan,
        }
    assert_training_frame_is_patient_only(
        df, feats_present, target_col=target_col, allow_clinical_target=True
    )
    resolved_split_group_col = resolve_split_group_col(df, subject_col, split_group_col)
    cols = [subject_col, target_col] + feats_present
    has_visit = visit_col in df.columns
    if has_visit:
        cols.insert(1, visit_col)
    if resolved_split_group_col not in cols:
        cols.append(resolved_split_group_col)
    sub = df[cols].dropna().copy()
    split_groups = np.asarray(sub[resolved_split_group_col].unique())

    oof_rows = []
    chosen_rows = []
    selected_features_by_fold: list[list[str]] = []

    if alphas is None:
        alphas = np.array([0.01, 0.1, 1.0, 10.0])
    if l1_ratios is None:
        l1_ratios = [0.1, 0.5, 0.9]
    groups = sub[resolved_split_group_col].values
    use_kfold = cv_n_splits is not None and 1 < int(cv_n_splits) < len(split_groups)
    splits = (
        group_kfold_indices(groups, n_splits=int(cv_n_splits), seed=random_seed)
        if use_kfold
        else (
            (np.where(groups != sid)[0], np.where(groups == sid)[0])
            for sid in split_groups
        )
    )

    for fold_idx, (train_idx, test_idx) in enumerate(splits, start=1):
        train_df = sub.iloc[train_idx].copy()
        test_df = sub.iloc[test_idx].copy()

        feats = list(feature_cols)
        if selection_fn is not None:
            feats = selection_fn(train_df)
        elif selection_method != "none":
            if not has_visit:
                raise ValueError("selection_method requires visit_col to be present in df")
            y_select = (train_df[visit_col].values == 2).astype(int)
            feats = select_features(selection_method, train_df[feats_present], y_select, feats_present, k=k)
        feats = [f for f in feats if f in train_df.columns]
        selected_features_by_fold.append(list(feats))
        if len(feats) == 0:
            continue

        X_train = train_df[feats].values
        y_train = train_df[target_col].values
        X_test = test_df[feats].values
        y_test = test_df[target_col].values

        X_train_s, X_test_s, _, _ = standardize_train_test(X_train, X_test)
        if z_clip is not None and X_train_s.shape[1] > 0:
            clip = float(z_clip)
            X_train_s = np.clip(X_train_s, -clip, clip)
            X_test_s = np.clip(X_test_s, -clip, clip)

        groups_train = train_df[resolved_split_group_col].values
        for tr_idx, va_idx in group_kfold_indices(groups_train, n_splits=inner_folds, seed=random_seed):
            pass  # Exhaust once to surface invalid fold configurations early.

        selection_metric = str(param_selection_metric).lower()
        if selection_metric not in {"rmse", "d", "cohens_d", "d_score", "annual_mean_dz"}:
            raise ValueError("param_selection_metric must be 'rmse', 'd', or 'annual_mean_dz'")
        if selection_metric in {"d", "cohens_d", "d_score", "annual_mean_dz"} and not has_visit:
            raise ValueError("d-based parameter selection requires visit_col to be present in df")

        def score_candidate(params):
            """Evaluate one hyperparameter candidate on inner grouped folds."""
            rmses = []
            d_scores = []
            annual_pred_parts = []
            for tr_idx, va_idx in group_kfold_indices(groups_train, n_splits=inner_folds, seed=random_seed):
                Xt, Xv = X_train_s[tr_idx], X_train_s[va_idx]
                yt, yv = y_train[tr_idx], y_train[va_idx]

                model = ElasticNet(
                    alpha=params["alpha"],
                    l1_ratio=params["l1_ratio"],
                    fit_intercept=True,
                    max_iter=5000,
                    random_state=random_seed,
                )
                model.fit(Xt, yt)
                pred = model.predict(Xv)

                rmses.append(_rmse(yv, pred))
                if selection_metric in {"d", "cohens_d", "d_score"}:
                    va_df = train_df.iloc[va_idx][[subject_col, visit_col]].copy()
                    va_df["pred"] = pred
                    d_val = compute_cohens_d_from_oof(
                        va_df, subject_col, pred_col="pred", visit_col=visit_col
                    )
                    if np.isfinite(d_val):
                        d_scores.append(float(d_val))
                elif selection_metric == "annual_mean_dz":
                    va_df = train_df.iloc[va_idx][[subject_col, visit_col]].copy()
                    va_df["pred"] = pred
                    annual_pred_parts.append(va_df)

            se_d = float(np.std(d_scores, ddof=1) / np.sqrt(len(d_scores))) if len(d_scores) > 1 else 0.0
            if selection_metric in {"d", "cohens_d", "d_score"}:
                score = float("-inf") if not d_scores else float(np.mean(d_scores))
                return {
                    **params,
                    "score": score,
                    "mean_validation_dz": score,
                    "se_validation_dz": se_d,
                    "feature_count": len(feats),
                }
            if selection_metric == "annual_mean_dz":
                if not annual_pred_parts:
                    return {
                        **params,
                        "score": float("-inf"),
                        "mean_validation_dz": float("-inf"),
                        "mean_validation_annual_dz": float("-inf"),
                        "se_validation_dz": se_d,
                        "feature_count": len(feats),
                    }
                annual_oof = pd.concat(annual_pred_parts, ignore_index=True)
                annual_summary = adjacent_pair_interval_effect_summary(
                    annual_oof,
                    pair_col=subject_col,
                    visit_col=visit_col,
                    score_col="pred",
                    n_boot=100,
                    seed=random_seed,
                )
                annual_diag = annual_tuning_diagnostics(annual_summary)
                value = annual_diag["mean_validation_annual_dz"]
                score = float(value) if np.isfinite(value) else float("-inf")
                return {
                    **params,
                    "score": score,
                    "mean_validation_dz": score,
                    "se_validation_dz": se_d,
                    "feature_count": len(feats),
                    **annual_diag,
                }
            rmses = [r for r in rmses if np.isfinite(r)]
            score = np.inf if not rmses else float(np.mean(rmses))
            return {
                **params,
                "score": score,
                "mean_validation_dz": -score if np.isfinite(score) else float("-inf"),
                "se_validation_dz": float(np.std(rmses, ddof=1) / np.sqrt(len(rmses))) if len(rmses) > 1 else 0.0,
                "feature_count": len(feats),
            }

        candidates = [{"alpha": float(a), "l1_ratio": float(l)} for a in alphas for l in l1_ratios]

        candidate_scores = pd.DataFrame([score_candidate(params) for params in candidates])
        if candidate_scores.empty:
            best_params = None
            best = np.nan
            choice = pd.Series(dtype=object)
        elif selection_metric == "annual_mean_dz":
            choice = select_hierarchical_candidate(candidate_scores)
            best = choice["score"]
            best_params = {k: choice[k] for k in candidates[0].keys()}
        elif selection_metric in {"d", "cohens_d", "d_score"}:
            choice = candidate_scores.loc[candidate_scores["score"].astype(float).idxmax()]
            best = choice["score"]
            best_params = {k: choice[k] for k in candidates[0].keys()}
        else:
            choice = candidate_scores.loc[candidate_scores["score"].astype(float).idxmin()]
            best = choice["score"]
            best_params = {k: choice[k] for k in candidates[0].keys()}

        if best_params is None:
            continue

        model = ElasticNet(
            alpha=best_params["alpha"],
            l1_ratio=best_params["l1_ratio"],
            fit_intercept=True,
            max_iter=5000,
            random_state=random_seed,
        )
        model.fit(X_train_s, y_train)
        pred = model.predict(X_test_s)

        visits_test = test_df[visit_col].values if has_visit else np.arange(len(y_test))
        subjects_test = test_df[subject_col].values
        for sid_test, v, yt, yp in zip(subjects_test, visits_test, y_test, pred):
            row = {subject_col: sid_test, "y_true": float(yt), "y_pred": float(yp)}
            if has_visit:
                row[visit_col] = int(v)
                row["visit"] = int(v)
                row["pred"] = float(yp)
                row["true"] = float(yt)
            oof_rows.append(row)
        chosen_rows.append({
            "test_fold": fold_idx,
            "param_selection_metric": selection_metric,
            "inner_score": best,
            "inner_dz_v1_v2": choice.get("dz_v1_v2", np.nan),
            "inner_dz_v2_v3": choice.get("dz_v2_v3", np.nan),
            "inner_annual_interval_gap": choice.get("annual_interval_gap", np.nan),
            "inner_p_progression": choice.get("p_progression", np.nan),
            "inner_se_validation_dz": choice.get("se_validation_dz", np.nan),
            **best_params,
        })

    oof_df = pd.DataFrame(oof_rows)
    if oof_df.empty:
        return {
            "oof_df": oof_df,
            "n_subjects": 0,
            "d_score": np.nan,
            "srm": np.nan,
            "d_ci_low": np.nan,
            "d_ci_high": np.nan,
            "selected_features_by_fold": selected_features_by_fold,
            "z_clip": None if z_clip is None else float(z_clip),
        }
    chosen_df = pd.DataFrame(chosen_rows)
    d_score = np.nan
    d_lo = np.nan
    d_hi = np.nan
    if has_visit:
        d_score = compute_cohens_d_from_oof(oof_df, subject_col, pred_col="pred", visit_col=visit_col)
        if compute_ci:
            from .metrics import bootstrap_ci_d
            _, d_lo, d_hi = bootstrap_ci_d(
                oof_df.rename(columns={"pred": "value"}), subject_col, visit_col, "value"
            )

    return {
        "oof_df": oof_df,
        "chosen_params_df": chosen_df,
        "n_subjects": int(oof_df[subject_col].nunique()) if subject_col in oof_df.columns else int(len(oof_df)),
        "rmse": _rmse(oof_df["y_true"], oof_df["y_pred"]) if len(oof_df) else np.nan,
        "r2": _r2(oof_df["y_true"], oof_df["y_pred"]) if len(oof_df) else np.nan,
        "d_score": d_score,
        "srm": d_score,
        "d_ci_low": d_lo,
        "d_ci_high": d_hi,
        "selected_features_by_fold": selected_features_by_fold,
        "cv_n_splits": int(cv_n_splits) if use_kfold else int(len(split_groups)),
        "cv_mode": "group_kfold" if use_kfold else "loo",
        "split_group_col": resolved_split_group_col,
        "n_split_groups": int(len(split_groups)),
    }


__all__ = [
    "group_kfold_indices",
    "subject_loo_split",
    "resolve_split_group_col",
    "nested_groupkfold_inner",
    "tune_and_run_regression_loocv",
    "lda_loocv",
    "lda_nested_loocv",
    "interaction_loocv",
    "run_cv_loocv_with_coefs",
    "select_params_by_inner_cv_d",
    "run_loocv_d_tuned_with_coefs",
    "bootstrap_subjects",
    "split_train_val_subjects",
    "run_cv_for_combination",
]
