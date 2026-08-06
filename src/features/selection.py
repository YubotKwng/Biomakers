"""MI-based top-k feature selectors and the leakage-safe selection-fn
factory.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .entropy import rank_features_by_mi
from .mml import mml_forward_selection


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
    # dedupe while keeping order
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
) -> List[str]:
    """Select features using a training-fold-only matrix and target.

    ``X_train`` should contain the stacked training visit rows, and ``y_train``
    should be the corresponding 0/1 visit-label target for ``mi`` and ``mml``.
    Callers should invoke this inside the outer CV loop, never on the full
    dataset before splitting.
    """
    names = list(feature_names)
    method = str(method).lower()
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

    if method == "mi":
        y_arr = np.asarray(y_train).reshape(-1)
        # MI ranks single features against visit labels; top-k selection is the
        # existing implementation reused through the unified dispatcher.
        mi_rank = rank_features_by_mi(X_df, usable, y_arr, mi_kind="mi_visit")
        return select_topk_global(mi_rank, k)
    if method == "mml":
        return mml_forward_selection(X_df[usable].values, y_train, usable)
    raise ValueError("method must be one of {'mi', 'mml', 'none'}")


def feature_stability_report(
    per_fold_features: list[list[str]],
    all_features: list[str],
) -> pd.DataFrame:
    """Report per-feature retention and mean pairwise selected-set Jaccard."""
    fold_sets = [set(fs) for fs in per_fold_features]
    n_folds = len(fold_sets)
    if n_folds == 0:
        mean_jaccard = np.nan
    else:
        vals = []
        for i in range(n_folds):
            for j in range(i + 1, n_folds):
                a, b = fold_sets[i], fold_sets[j]
                # Empty selections can be meaningful. Treat two empty folds as
                # perfectly matching and one empty / one non-empty as no overlap.
                if not a and not b:
                    vals.append(1.0)
                elif not a or not b:
                    vals.append(0.0)
                else:
                    vals.append(len(a & b) / len(a | b))
        mean_jaccard = float(np.mean(vals)) if vals else 1.0

    rows = []
    for feat in list(all_features):
        retention = (
            float(sum(feat in fs for fs in fold_sets) / n_folds)
            if n_folds else np.nan
        )
        rows.append({
            "feature": feat,
            "retention_rate": retention,
            "mean_jaccard": mean_jaccard,
        })
    return pd.DataFrame(rows).sort_values(
        "retention_rate", ascending=False, kind="mergesort"
    ).reset_index(drop=True)


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
    # dedupe features (since some appear in both structural and structural_ext rows)
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
        raise ValueError("Only lda_visit feature selection is allowed; clinical-score targets are disabled.")

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
    "feature_stability_report",
    "_global_rank",
    "make_selection_fn",
]
