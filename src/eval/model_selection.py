"""Hierarchical model-selection rules for progression-sensitive composites."""
from __future__ import annotations

import numpy as np
import pandas as pd


def select_hierarchical_candidate(
    results: pd.DataFrame,
    *,
    mean_col: str = "mean_validation_dz",
    se_col: str = "se_validation_dz",
) -> pd.Series:
    """Select a candidate by primary d_z and stability-oriented tie-breakers.

    Rule:
    1. Find the best mean inner-validation d_z.
    2. Keep candidates within one SE of that best model.
    3. Prefer fewer features, stronger feature-set Jaccard, coefficient-sign
       stability, P(delta>0), score-ranking stability, then mean d_z.
    """
    if results.empty:
        raise ValueError("results must contain at least one candidate")
    required = {mean_col, se_col}
    missing = required - set(results.columns)
    if missing:
        raise KeyError(f"results missing required columns: {sorted(missing)}")

    work = results.copy()
    best_idx = work[mean_col].astype(float).idxmax()
    best = work.loc[best_idx]
    threshold = float(best[mean_col]) - float(best[se_col])
    eligible = work[work[mean_col].astype(float) >= threshold].copy()
    if eligible.empty:
        eligible = work.loc[[best_idx]].copy()

    defaults = {
        "feature_count": np.inf,
        "jaccard_stability": -np.inf,
        "sign_stability": -np.inf,
        "coefficient_sign_stability": -np.inf,
        "p_progression": -np.inf,
        "directional_consistency": -np.inf,
        "score_ranking_stability": -np.inf,
    }
    for col, default in defaults.items():
        if col not in eligible.columns:
            eligible[col] = default
        eligible[col] = eligible[col].fillna(default)
    # Backward-compatible aliases used by existing notebooks/tests.
    eligible["_sign_stability_rank"] = eligible[["sign_stability", "coefficient_sign_stability"]].max(axis=1)
    eligible["_progression_rank"] = eligible[["p_progression", "directional_consistency"]].max(axis=1)
    eligible = eligible.sort_values(
        [
            "feature_count",
            "jaccard_stability",
            "_sign_stability_rank",
            "_progression_rank",
            "score_ranking_stability",
            mean_col,
        ],
        ascending=[True, False, False, False, False, False],
        kind="mergesort",
    )
    return eligible.iloc[0].drop(labels=["_sign_stability_rank", "_progression_rank"], errors="ignore")


__all__ = ["select_hierarchical_candidate"]
