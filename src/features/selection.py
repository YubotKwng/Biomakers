"""MI-based top-k feature selectors and the leakage-safe selection-fn
factory.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from .entropy import rank_features_by_mi


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
    one of ``{'lda_visit', 'reg_fars1', 'reg_dfars'}``; ``selection_mode`` is
    one of ``{'entropy_topk_global', 'entropy_topk_group'}``.
    """
    k_selected = int(k_selected)

    def select_for_train(train_df: pd.DataFrame) -> List[str]:
        # train_df is either long_df (for lda) or baseline/delta df (for regression)
        if task == "lda_visit":
            y = (train_df["visit"].values == 2).astype(int)
            mi_rank = rank_features_by_mi(train_df, all_features, y, mi_kind="mi_visit")
        elif task == "reg_fars1":
            y = train_df["FARS1"].values
            mi_rank = rank_features_by_mi(train_df, all_features, y, mi_kind="mi_reg")
        else:
            y = train_df["dFARS"].values
            mi_rank = rank_features_by_mi(train_df, all_features, y, mi_kind="mi_reg")

        if selection_mode == "entropy_topk_global":
            return select_topk_global(mi_rank, k_selected)

        # entropy_topk_group
        return select_topk_by_group(mi_rank, feature_groups, k_selected)

    return select_for_train


__all__ = [
    "select_topk_global",
    "select_topk_by_group",
    "_global_rank",
    "make_selection_fn",
]
