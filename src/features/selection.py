"""Fold-local feature selectors for FRDA biomarker modelling."""
from __future__ import annotations

from typing import Callable, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .entropy import rank_features_by_mi
from .mml import mml_forward_selection
from .registry import FEATURE_GROUPS


# ---------------------------------------------------------------------------
# Top-k selectors operating on a pre-computed MI ranking dataframe
# ---------------------------------------------------------------------------
def select_topk_global(mi_rank_df: pd.DataFrame, k: int) -> List[str]:
    """Select the top-k features by mutual information across all groups."""
    k = int(k)
    if k <= 0:
        return []
    df = mi_rank_df.copy()
    df = df[df["mi"] > -np.inf]
    return df["feature"].head(min(k, len(df))).tolist()


def select_topk_by_group(
    mi_rank_df: pd.DataFrame,
    group_to_features: Dict[str, List[str]],
    k: int,
) -> List[str]:
    """Select top-k features by MI within each group and deduplicate them.

    Output order follows the iteration order of ``group_to_features``.
    """
    selected = []
    for g, feats in group_to_features.items():
        df = mi_rank_df[mi_rank_df["feature"].isin(feats)].copy()
        df = df[df["mi"] > -np.inf]
        selected.extend(df["feature"].head(min(int(k), len(df))).tolist())

    seen = set()
    out = []
    for f in selected:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


# ---------------------------------------------------------------------------
# Unified fold-local selector.
# ---------------------------------------------------------------------------
def select_features(
    method: str,
    X_train,
    y_train,
    feature_names: Sequence[str],
    k: int = 8,
    *,
    train_frame: pd.DataFrame | None = None,
    subject_col: str | None = None,
    visit_col: str = "visit",
    mrmr_redundancy_lambda: float = 0.25,
    sparse_lambda: float = 0.01,
    sparse_alpha: float = 0.5,
    sparse_tolerance: float = 1e-8,
) -> List[str]:
    """Select features using a training-fold-only matrix and target.

    ``X_train`` should contain the stacked training visit rows, and ``y_train``
    should be the corresponding 0/1 visit-label target for ``mi_visit`` and
    ``mml``. Progression-aware selectors additionally require ``train_frame``,
    ``subject_col``, and ``visit_col`` so annual paired changes are calculated
    from the current training fold only.
    """
    names = list(feature_names)
    method = str(method).lower()
    if method == "mi":
        # Backward-compatible alias. Keep historical MI behaviour comparable
        # while exposing the target explicitly as visit-label MI.
        method = "mi_visit"
    if method == "none":
        return list(names)
    if len(names) == 0:
        return []

    if isinstance(X_train, pd.DataFrame):
        # DataFrame input lets callers pass a wide training frame while the
        # dispatcher keeps only the feature names requested by the model.
        X_df = X_train.loc[:, [f for f in names if f in X_train.columns]].copy()
    else:
        X_arr = np.asarray(X_train, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        if X_arr.shape[1] != len(names):
            raise ValueError("X_train columns must match feature_names")
        X_df = pd.DataFrame(X_arr, columns=names)

    usable = [f for f in names if f in X_df.columns]
    if not usable:
        return []

    if method == "mi_visit":
        y_arr = np.asarray(y_train).reshape(-1)
        # MI ranks single features against visit labels; top-k selection is the
        # existing implementation reused through the unified dispatcher.
        mi_rank = rank_features_by_mi(X_df, usable, y_arr, mi_kind="mi_visit")
        return select_topk_global(mi_rank, k)
    if method == "mml":
        return mml_forward_selection(X_df[usable].values, y_train, usable)
    if method == "progression_univariate":
        effects = progression_univariate_effects(
            _require_train_frame(train_frame, usable),
            usable,
            subject_col=_require_subject_col(subject_col),
            visit_col=visit_col,
        )
        return effects.head(min(int(k), len(effects)))["feature"].tolist()
    if method == "progression_mrmr":
        return progression_mrmr_selection(
            _require_train_frame(train_frame, usable),
            usable,
            subject_col=_require_subject_col(subject_col),
            visit_col=visit_col,
            k=k,
            redundancy_lambda=mrmr_redundancy_lambda,
        )
    if method == "sparse_srm":
        weights = sparse_srm_coefficients(
            _require_train_frame(train_frame, usable),
            usable,
            subject_col=_require_subject_col(subject_col),
            visit_col=visit_col,
            sparse_lambda=sparse_lambda,
            sparse_alpha=sparse_alpha,
        )
        selected = weights[weights["abs_weight"] > float(sparse_tolerance)].copy()
        selected = selected.sort_values("abs_weight", ascending=False, kind="mergesort")
        if int(k) > 0:
            selected = selected.head(min(int(k), len(selected)))
        return selected["feature"].tolist()
    raise ValueError(
        "method must be one of {'none', 'mi', 'mi_visit', 'mml', "
        "'progression_univariate', 'progression_mrmr', 'sparse_srm'}"
    )


def _require_train_frame(train_frame: pd.DataFrame | None, features: Sequence[str]) -> pd.DataFrame:
    if train_frame is None:
        raise ValueError("progression-aware feature selection requires train_frame")
    missing = [f for f in features if f not in train_frame.columns]
    if missing:
        raise KeyError(f"train_frame missing feature columns: {missing[:10]}")
    return train_frame


def _require_subject_col(subject_col: str | None) -> str:
    if not subject_col:
        raise ValueError("progression-aware feature selection requires subject_col")
    return str(subject_col)


def _pair_interval_from_id(values: pd.Series) -> pd.Series:
    return (
        values.astype(str)
        .str.upper()
        .str.extract(r"(V\d+V\d+)", expand=False)
        .map({"V1V2": "V1->V2", "V2V3": "V2->V3"})
    )


def annual_feature_deltas(
    train_frame: pd.DataFrame,
    features: Sequence[str],
    *,
    subject_col: str,
    visit_col: str = "visit",
) -> pd.DataFrame:
    """Return annual paired feature deltas from one training fold.

    The function keeps V1->V2 and V2->V3 labels separate when the pair id
    contains ``V1V2`` or ``V2V3``. It does not use validation or test rows.
    """
    usable = [f for f in features if f in train_frame.columns]
    required = {subject_col, visit_col, *usable}
    missing = required - set(train_frame.columns)
    if missing:
        raise KeyError(f"train_frame missing required columns: {sorted(missing)}")

    rows = []
    tmp = (
        train_frame[[subject_col, visit_col, *usable]]
        .dropna(subset=[subject_col, visit_col])
        .copy()
    )
    tmp["_visit_int"] = pd.to_numeric(tmp[visit_col], errors="coerce").astype("Int64")
    tmp = tmp[tmp["_visit_int"].isin([1, 2])].copy()
    interval_by_subject = _pair_interval_from_id(tmp[subject_col])
    for pair_id, group in tmp.groupby(subject_col, sort=False):
        visits = group.sort_values("_visit_int")
        if set(visits["_visit_int"].dropna().astype(int)) != {1, 2}:
            continue
        base = visits[visits["_visit_int"] == 1].iloc[0]
        foll = visits[visits["_visit_int"] == 2].iloc[0]
        interval = interval_by_subject.loc[visits.index].dropna()
        interval_label = str(interval.iloc[0]) if len(interval) else "annual"
        row = {subject_col: pair_id, "interval": interval_label}
        for feat in usable:
            start_value = pd.to_numeric(pd.Series([base[feat]]), errors="coerce").iloc[0]
            end_value = pd.to_numeric(pd.Series([foll[feat]]), errors="coerce").iloc[0]
            row[feat] = (
                float(end_value - start_value)
                if pd.notna(start_value) and pd.notna(end_value)
                else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _cohens_dz(values: pd.Series | np.ndarray) -> float:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
    if len(arr) < 2:
        return np.nan
    sd = float(np.std(arr, ddof=1))
    if sd == 0 or not np.isfinite(sd):
        return np.nan
    return float(np.mean(arr) / sd)


def progression_univariate_effects(
    train_frame: pd.DataFrame,
    features: Sequence[str],
    *,
    subject_col: str,
    visit_col: str = "visit",
) -> pd.DataFrame:
    """Rank features by annual paired progression relevance in train data."""
    deltas = annual_feature_deltas(
        train_frame,
        features,
        subject_col=subject_col,
        visit_col=visit_col,
    )
    rows = []
    for feat in [f for f in features if f in deltas.columns]:
        d12 = _cohens_dz(deltas.loc[deltas["interval"] == "V1->V2", feat])
        d23 = _cohens_dz(deltas.loc[deltas["interval"] == "V2->V3", feat])
        finite_effects = [v for v in (d12, d23) if np.isfinite(v)]
        mean_annual = float(np.mean(finite_effects)) if finite_effects else np.nan
        gap = abs(d12 - d23) if np.isfinite(d12) and np.isfinite(d23) else np.nan
        rows.append({
            "feature": feat,
            "dz_v1_v2": d12,
            "dz_v2_v3": d23,
            "mean_annual_dz": (
                float(mean_annual)
                if np.isfinite(mean_annual)
                else np.nan
            ),
            "abs_mean_annual_dz": (
                abs(float(mean_annual))
                if np.isfinite(mean_annual)
                else -np.inf
            ),
            "annual_interval_gap": gap,
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(
            columns=[
                "feature",
                "dz_v1_v2",
                "dz_v2_v3",
                "mean_annual_dz",
                "abs_mean_annual_dz",
                "annual_interval_gap",
            ]
        )
    return out.sort_values(
        ["abs_mean_annual_dz", "annual_interval_gap", "feature"],
        ascending=[False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)


def _standardized_feature_frame(frame: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    x = frame[list(features)].apply(pd.to_numeric, errors="coerce")
    x = x.fillna(x.mean(numeric_only=True))
    sd = x.std(axis=0, ddof=1).replace(0, 1.0)
    return (x - x.mean(axis=0)) / sd


def progression_mrmr_selection(
    train_frame: pd.DataFrame,
    features: Sequence[str],
    *,
    subject_col: str,
    visit_col: str = "visit",
    k: int = 8,
    redundancy_lambda: float = 0.25,
) -> list[str]:
    """Greedy progression-aware mRMR using absolute feature correlations."""
    effects = progression_univariate_effects(
        train_frame,
        features,
        subject_col=subject_col,
        visit_col=visit_col,
    )
    if effects.empty or int(k) <= 0:
        return []
    relevance = dict(zip(effects["feature"], effects["abs_mean_annual_dz"]))
    xz = _standardized_feature_frame(train_frame, effects["feature"].tolist())
    corr = xz.corr(method="pearson").abs().fillna(0.0)
    selected: list[str] = []
    remaining = effects["feature"].tolist()
    while remaining and len(selected) < int(k):
        best_feat = None
        best_score = -np.inf
        for feat in remaining:
            redundancy = float(corr.loc[feat, selected].mean()) if selected else 0.0
            score = float(relevance.get(feat, -np.inf)) - float(redundancy_lambda) * redundancy
            tie_break = score == best_score and (best_feat is None or feat < best_feat)
            if score > best_score or tie_break:
                best_score = score
                best_feat = feat
        if best_feat is None:
            break
        selected.append(best_feat)
        remaining.remove(best_feat)
    return selected


def sparse_srm_coefficients(
    train_frame: pd.DataFrame,
    features: Sequence[str],
    *,
    subject_col: str,
    visit_col: str = "visit",
    sparse_lambda: float = 0.01,
    sparse_alpha: float = 0.5,
) -> pd.DataFrame:
    """Return deterministic ElasticNet-style sparse SRM coefficients.

    This approximates an embedded sparse SRM by estimating interval-balanced
    annual change mean/covariance on training subjects, solving a regularised
    SRM direction, then applying deterministic soft-thresholding.
    """
    usable = [f for f in features if f in train_frame.columns]
    deltas = annual_feature_deltas(
        train_frame,
        usable,
        subject_col=subject_col,
        visit_col=visit_col,
    )
    if deltas.empty or not usable:
        return pd.DataFrame({"feature": usable, "weight": 0.0, "abs_weight": 0.0})
    mats = []
    means = []
    covs = []
    for _, group in deltas.groupby("interval", sort=True):
        mat = group[usable].apply(pd.to_numeric, errors="coerce").dropna().to_numpy(dtype=float)
        if mat.shape[0] < 2:
            continue
        mats.append(mat)
        means.append(np.mean(mat, axis=0))
        covs.append(np.atleast_2d(np.cov(mat, rowvar=False, ddof=1)))
    if not mats:
        return pd.DataFrame({"feature": usable, "weight": 0.0, "abs_weight": 0.0})
    mu = np.mean(np.vstack(means), axis=0)
    sigma = np.mean(np.stack(covs, axis=0), axis=0)
    lam = float(sparse_lambda)
    alpha = float(np.clip(sparse_alpha, 0.0, 1.0))
    reg = sigma + (lam * (1.0 - alpha) + 1e-8) * np.eye(len(usable))
    try:
        raw_w = np.linalg.solve(reg, mu)
    except np.linalg.LinAlgError:
        raw_w = np.linalg.pinv(reg) @ mu
    thresh = lam * alpha
    weights = np.sign(raw_w) * np.maximum(np.abs(raw_w) - thresh, 0.0)
    out = pd.DataFrame({
        "feature": usable,
        "weight": weights.astype(float),
        "abs_weight": np.abs(weights).astype(float),
    })
    return out.sort_values("abs_weight", ascending=False, kind="mergesort").reset_index(drop=True)


def feature_stability_report(
    per_fold_features: list[list[str]],
    all_features: list[str],
) -> pd.DataFrame:
    """Report per-feature retention and pairwise selected-set Jaccard."""
    fold_sets = [set(fs) for fs in per_fold_features]
    n_folds = len(fold_sets)
    summary = feature_set_jaccard_summary(per_fold_features)

    rows = []
    for feat in list(all_features):
        count = int(sum(feat in fs for fs in fold_sets))
        retention = float(count / n_folds) if n_folds else np.nan
        rows.append({
            "feature": feat,
            "n_selections": count,
            "selection_frequency": retention,
            "retention_rate": retention,
            **summary,
        })
    return pd.DataFrame(rows).sort_values(
        "retention_rate", ascending=False, kind="mergesort"
    ).reset_index(drop=True)


def feature_set_jaccard_summary(per_fold_features: Sequence[Sequence[str]]) -> dict:
    """Summarise pairwise Jaccard overlap for selected feature sets."""
    fold_sets = [set(fs) for fs in per_fold_features]
    vals = []
    for i in range(len(fold_sets)):
        for j in range(i + 1, len(fold_sets)):
            a, b = fold_sets[i], fold_sets[j]
            if not a and not b:
                vals.append(1.0)
            elif not a or not b:
                vals.append(0.0)
            else:
                vals.append(len(a & b) / len(a | b))
    arr = np.asarray(vals, dtype=float)
    if len(arr) == 0:
        return {
            "mean_jaccard": 1.0 if fold_sets else np.nan,
            "median_jaccard": np.nan,
            "iqr_jaccard": np.nan,
            "min_jaccard": np.nan,
            "max_jaccard": np.nan,
        }
    return {
        "mean_jaccard": float(np.mean(arr)),
        "median_jaccard": float(np.median(arr)),
        "iqr_jaccard": float(np.percentile(arr, 75) - np.percentile(arr, 25)),
        "min_jaccard": float(np.min(arr)),
        "max_jaccard": float(np.max(arr)),
    }


def feature_domain_coverage(
    selected_features: Sequence[str],
    feature_groups: Mapping[str, Sequence[str]] | None = None,
) -> dict:
    """Count represented MRI domains for a selected subset."""
    groups = feature_groups or FEATURE_GROUPS
    selected = set(selected_features)
    represented = [
        domain for domain, feats in groups.items()
        if selected.intersection(set(feats))
    ]
    return {
        "n_selected_features": int(len(selected)),
        "n_domains": int(len(represented)),
        "represented_domains": ", ".join(represented),
    }


def select_one_se_candidate(results: pd.DataFrame) -> pd.Series:
    """Backward-compatible alias for hierarchical one-SE model selection."""
    from ..eval.model_selection import select_hierarchical_candidate

    return select_hierarchical_candidate(results)


# ---------------------------------------------------------------------------
# Descriptive registry helper for multi-source MI rankings.
# ---------------------------------------------------------------------------
def _global_rank(
    entropy_df: pd.DataFrame,
    source_col: str,
    group: Optional[str] = None,
) -> List[str]:
    """Rank features in ``entropy_df`` by ``source_col``.

    The input has one row per feature/group combination and may contain
    multiple rows for the same feature, so the returned list is deduplicated.
    """
    df = entropy_df.copy()
    if group is not None:
        df = df[df["group"] == group]
    df = df.dropna(subset=[source_col]).copy()
    df = df.sort_values(source_col, ascending=False)

    seen = set()
    feats = []
    for f in df["feature"]:
        if f not in seen:
            seen.add(f)
            feats.append(f)
    return feats


# ---------------------------------------------------------------------------
# Leakage-safe selection-function factory.
# ---------------------------------------------------------------------------
def make_selection_fn(
    task: str,
    selection_mode: str,
    k_selected: int,
    entropy_source: str,
    all_features: List[str],
    feature_groups: Dict[str, List[str]],
) -> Callable[[pd.DataFrame], List[str]]:
    """Return a training-fold feature selector.

    The closure recomputes MI rankings on the training fold only. ``task`` is
    ``'lda_visit'``; clinical-score regression selectors are disabled so
    clinical scores cannot become model-training targets.
    """
    k_selected = int(k_selected)
    if task != "lda_visit":
        raise ValueError(
            "Only lda_visit feature selection is allowed; clinical-score targets are disabled."
        )

    def select_for_train(train_df: pd.DataFrame) -> List[str]:
        """Select features from one caller-supplied training fold."""
        y = (train_df["visit"].values == 2).astype(int)
        mi_rank = rank_features_by_mi(train_df, all_features, y, mi_kind="mi_visit")

        if selection_mode == "entropy_topk_global":
            return select_topk_global(mi_rank, k_selected)

        # entropy_topk_group
        return select_topk_by_group(mi_rank, feature_groups, k_selected)

    return select_for_train


__all__ = [
    "select_topk_global",
    "select_topk_by_group",
    "select_features",
    "annual_feature_deltas",
    "progression_univariate_effects",
    "progression_mrmr_selection",
    "sparse_srm_coefficients",
    "feature_stability_report",
    "feature_set_jaccard_summary",
    "feature_domain_coverage",
    "select_one_se_candidate",
    "_global_rank",
    "make_selection_fn",
]
