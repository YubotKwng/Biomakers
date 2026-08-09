"""Hierarchical model-selection rules for progression-sensitive composites."""
from __future__ import annotations

import numpy as np
import pandas as pd


def select_hierarchical_candidate(
    results: pd.DataFrame,
    *,
    mean_col: str | None = None,
    se_col: str = "se_validation_dz",
) -> pd.Series:
    """Select a candidate by annual d_z and consistency/stability tie-breakers.

    Rule:
    1. Find the best mean annual inner-validation d_z from V1->V2 and V2->V3.
    2. Keep candidates within one SE of that best model.
    3. Prefer smaller ``abs(dz_V1_V2 - dz_V2_V3)``.
    4. Prefer higher P(delta>0).
    5. Prefer simpler and more stable models.
    """
    if results.empty:
        raise ValueError("results must contain at least one candidate")
    if mean_col is None:
        mean_col = (
            "mean_validation_annual_dz"
            if "mean_validation_annual_dz" in results.columns
            else "mean_validation_dz"
        )
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
        "annual_interval_gap": np.inf,
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
            "annual_interval_gap",
            "_progression_rank",
            "feature_count",
            "jaccard_stability",
            "_sign_stability_rank",
            "score_ranking_stability",
            mean_col,
        ],
        ascending=[True, False, True, False, False, False, False],
        kind="mergesort",
    )
    return eligible.iloc[0].drop(labels=["_sign_stability_rank", "_progression_rank"], errors="ignore")


__all__ = ["select_hierarchical_candidate"]
