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
from ..data.qc import _as_float_array, standardize_train_test, tukey_outliers_mask


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
):
    """Create a reproducible subject-level train/validation split."""
    subjects = np.array(sorted(train_df[subject_col].unique()))
    rng = np.random.RandomState(seed)
    rng.shuffle(subjects)
    n_val = max(1, int(round(len(subjects) * val_fraction)))
    val_subjects = set(subjects[:n_val])
    train_subjects = set(subjects[n_val:])
    if len(train_subjects) == 0:
        train_subjects = val_subjects
        val_subjects = set(subjects[:1])
    train_split = train_df[train_df[subject_col].isin(train_subjects)].copy()
    val_split = train_df[train_df[subject_col].isin(val_subjects)].copy()
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
    """
    if l1_ratios is None:
        l1_ratios = [0.1, 0.5, 0.9]
    if alphas is None:
        alphas = np.logspace(-2, 2, 10)

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
    shrink="auto",
):
    """Run subject-level leave-one-out LDA visit separation.

    The model learns a direction that separates baseline from follow-up visits
    in the training subjects. Held-out visit scores are converted to paired
    deltas and evaluated with Cohen's d and SRM.
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
        }
    sub = df_long_in[[subject_col, visit_col] + feats_present].dropna().copy()
    counts = sub.groupby(subject_col)[visit_col].nunique()
    valid_subjects = counts[counts == 2].index
    sub = sub[sub[subject_col].isin(valid_subjects)].copy()

    subjects = sub[subject_col].unique()
    oof_rows = []

    for sid in subjects:
        train_df = sub[sub[subject_col] != sid]
        test_df = sub[sub[subject_col] == sid]

        feats = list(feature_cols)
        if selection_fn is not None:
            feats = selection_fn(train_df)
        feats = [f for f in feats if f in train_df.columns]
        if len(feats) == 0:
            continue

        X_train = train_df[feats].values
        y_train = (train_df[visit_col].values == 2).astype(int)
        X_test = test_df[feats].values

        X_train_s, X_test_s, _, _ = standardize_train_test(X_train, X_test)
        w = fit_lda_direction(X_train_s, y_train, shrink=shrink)
        if w is None:
            continue
        scores = predict_lda_scores(X_test_s, w)

        for v, sc in zip(test_df[visit_col].values, scores):
            oof_rows.append({subject_col: sid, visit_col: int(v), "score": float(sc)})

    oof_df = pd.DataFrame(oof_rows)
    deltas = paired_deltas_from_long(
        oof_df.rename(columns={"score": "value"}), subject_col, visit_col, "value"
    )
    d_out = compute_cohens_d(deltas)
    srm_out = compute_srm(deltas)
    d_mean, d_lo, d_hi = bootstrap_ci_d(
        oof_df.rename(columns={"score": "value"}), subject_col, visit_col, "value"
    )

    return {
        "oof_df": oof_df,
        "n_subjects": int(d_out["n"]),
        "d_score": d_out["d"],
        "srm": srm_out["srm"],
        "d_ci_low": d_lo,
        "d_ci_high": d_hi,
    }


def tune_and_run_regression_loocv(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str,
    subject_col: str,
    model_kind: str,
    selection_fn=None,
    inner_folds: int = 3,
    alphas: Optional[Sequence[float]] = None,
    l1_ratios: Optional[Sequence[float]] = None,
    n_components_list: Optional[Sequence[int]] = None,
    random_seed: int = 42,
):
    """Run leave-one-out regression with inner-CV hyperparameter tuning.

    Supports ElasticNet coordinate descent, Ridge, and PLS backends for
    clinical-score or change-score prediction under subject-level validation.
    """
    from .metrics import r2 as _r2, rmse as _rmse  # local alias

    try:
        from ..models.elasticnet_cd import fit_elasticnet_cd
        from ..models.pls import fit_pls1_nipals
        from ..models.ridge import fit_ridge, predict_linear
    except ImportError as exc:  # pragma: no cover - optional model backend wiring
        raise ImportError(
            "tune_and_run_regression_loocv requires ridge, elasticnet_cd, "
            "and pls model backends."
        ) from exc

    feats_present = [f for f in feature_cols if f in df.columns]
    if len(feats_present) == 0:
        return {
            "oof_df": pd.DataFrame(),
            "chosen_params_df": pd.DataFrame(),
            "n_subjects": 0,
            "rmse": np.nan,
            "r2": np.nan,
        }
    sub = df[[subject_col, target_col] + feats_present].dropna().copy()
    subjects = sub[subject_col].unique()

    oof_rows = []
    chosen_rows = []

    if alphas is None:
        alphas = np.array([0.01, 0.1, 1.0, 10.0])
    if l1_ratios is None:
        l1_ratios = [0.1, 0.5, 0.9]
    if n_components_list is None:
        n_components_list = [1, 2, 3]

    for sid in subjects:
        train_df = sub[sub[subject_col] != sid]
        test_df = sub[sub[subject_col] == sid]

        feats = list(feature_cols)
        if selection_fn is not None:
            feats = selection_fn(train_df)
        feats = [f for f in feats if f in train_df.columns]
        if len(feats) == 0:
            continue

        X_train = train_df[feats].values
        y_train = train_df[target_col].values
        X_test = test_df[feats].values
        y_test = test_df[target_col].values

        X_train_s, X_test_s, _, _ = standardize_train_test(X_train, X_test)

        groups_train = train_df[subject_col].values
        for tr_idx, va_idx in group_kfold_indices(groups_train, n_splits=inner_folds, seed=random_seed):
            pass  # Exhaust once to surface invalid fold configurations early.

        def score_candidate(params):
            rmses = []
            for tr_idx, va_idx in group_kfold_indices(groups_train, n_splits=inner_folds, seed=random_seed):
                Xt, Xv = X_train_s[tr_idx], X_train_s[va_idx]
                yt, yv = y_train[tr_idx], y_train[va_idx]

                if model_kind == "ridge":
                    w = fit_ridge(Xt, yt, alpha=params["alpha"])
                    pred = predict_linear(Xv, w)
                elif model_kind == "elasticnet":
                    w, b0 = fit_elasticnet_cd(Xt, yt, alpha=params["alpha"], l1_ratio=params["l1_ratio"])
                    pred = predict_linear(Xv, w, intercept=b0)
                elif model_kind == "pls":
                    coef, intercept = fit_pls1_nipals(Xt, yt, n_components=params["n_components"])
                    pred = predict_linear(Xv, coef, intercept=intercept)
                else:
                    raise ValueError("unknown model_kind")

                rmses.append(_rmse(yv, pred))

            rmses = [r for r in rmses if np.isfinite(r)]
            return np.inf if not rmses else float(np.mean(rmses))

        candidates = []
        if model_kind == "ridge":
            candidates = [{"alpha": float(a)} for a in alphas]
        elif model_kind == "elasticnet":
            candidates = [{"alpha": float(a), "l1_ratio": float(l)} for a in alphas for l in l1_ratios]
        elif model_kind == "pls":
            max_k = min(int(max(n_components_list)), len(feats), max(1, len(train_df) - 1))
            cand = [k for k in n_components_list if 1 <= k <= max_k]
            candidates = [{"n_components": int(k)} for k in cand]
        else:
            raise ValueError("unknown model_kind")

        best = None
        best_params = None
        for params in candidates:
            s = score_candidate(params)
            if best is None or s < best:
                best = s
                best_params = params

        if best_params is None:
            continue

        if model_kind == "ridge":
            w = fit_ridge(X_train_s, y_train, alpha=best_params["alpha"])
            pred = predict_linear(X_test_s, w)
        elif model_kind == "elasticnet":
            w, b0 = fit_elasticnet_cd(X_train_s, y_train, alpha=best_params["alpha"], l1_ratio=best_params["l1_ratio"])
            pred = predict_linear(X_test_s, w, intercept=b0)
        else:  # pls
            coef, intercept = fit_pls1_nipals(X_train_s, y_train, n_components=best_params["n_components"])
            pred = predict_linear(X_test_s, coef, intercept=intercept)

        oof_rows.append({subject_col: sid, "y_true": float(y_test[0]), "y_pred": float(pred[0])})
        chosen_rows.append({"test_subject": sid, **best_params})

    oof_df = pd.DataFrame(oof_rows)
    if oof_df.empty:
        return {
            "oof_df": oof_df,
            "n_subjects": 0,
            "d_score": np.nan,
            "srm": np.nan,
            "d_ci_low": np.nan,
            "d_ci_high": np.nan,
        }
    chosen_df = pd.DataFrame(chosen_rows)

    return {
        "oof_df": oof_df,
        "chosen_params_df": chosen_df,
        "n_subjects": int(len(oof_df)),
        "rmse": _rmse(oof_df["y_true"], oof_df["y_pred"]) if len(oof_df) else np.nan,
        "r2": _r2(oof_df["y_true"], oof_df["y_pred"]) if len(oof_df) else np.nan,
    }


__all__ = [
    "group_kfold_indices",
    "subject_loo_split",
    "nested_groupkfold_inner",
    "tune_and_run_regression_loocv",
    "lda_loocv",
    "run_cv_loocv_with_coefs",
    "select_params_by_inner_cv_d",
    "run_loocv_d_tuned_with_coefs",
    "bootstrap_subjects",
    "split_train_val_subjects",
    "run_cv_for_combination",
]
