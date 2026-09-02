"""Model evaluation across a priori feature-panel combinations."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from ..eval.intervals import adjacent_pair_interval_effect_summary, annual_tuning_diagnostics
from ..eval.stability import selected_feature_jaccard
from ..features.panels import a_priori_70_panel_combinations
from ..features.selection import feature_set_jaccard_summary
from ..models.srm_global import srm_global_nested_loocv


def evaluate_srm_panel_combinations(
    df_long: pd.DataFrame,
    *,
    subject_col: str = "pair_id",
    visit_col: str = "visit",
    split_group_col: str = "subject",
    cv_n_splits: int = 5,
    inner_folds: int = 2,
    random_seed: int = 42,
    n_boot: int = 300,
    max_panel_combo_size: int | None = None,
    panel_family: str = "anatomical",
    ridge: float = 0.0,
    covariance_shrinkage: float = 0.0,
    z_clip: float | None = None,
) -> dict:
    """Run nested-CV SRM on every requested a priori panel combination.

    Each panel combination is evaluated as a fixed feature set. The inner CV
    score is retained as the validation score; the outer OOF annual score is
    retained as the test score. This gives the validation-vs-test overfit check
    supervisors asked to see without selecting features on the held-out folds.
    """
    combos = a_priori_70_panel_combinations(
        family=panel_family,
        max_size=max_panel_combo_size,
    )
    rows = []
    results: dict[str, dict] = {}
    for combo_idx, combo in enumerate(combos, start=1):
        features = [f for f in combo["features"] if f in df_long.columns]
        if not features:
            continue
        candidate = {
            "ridge": float(ridge),
            "covariance_shrinkage": float(covariance_shrinkage),
            "z_clip": z_clip,
            "selection_method": "none",
            "k": len(features),
            "selection_label": "all_features_in_panel_combo",
        }
        res = srm_global_nested_loocv(
            df_long,
            features,
            subject_col=subject_col,
            visit_col=visit_col,
            candidates=[candidate],
            cv_n_splits=cv_n_splits,
            inner_folds=inner_folds,
            random_seed=random_seed,
            compute_ci=False,
            split_group_col=split_group_col,
            tuning_metric="annual_mean_dz",
        )
        intervals = adjacent_pair_interval_effect_summary(
            res["oof_df"],
            pair_col=subject_col,
            visit_col=visit_col,
            score_col="score",
            n_boot=n_boot,
            seed=random_seed + combo_idx,
        )
        test_diag = annual_tuning_diagnostics(intervals)
        chosen = res["chosen_params_df"]
        validation_score = (
            float(pd.to_numeric(chosen["inner_d_score"], errors="coerce").mean())
            if "inner_d_score" in chosen
            else np.nan
        )
        validation_d12 = (
            float(pd.to_numeric(chosen["inner_dz_v1_v2"], errors="coerce").mean())
            if "inner_dz_v1_v2" in chosen
            else np.nan
        )
        validation_d23 = (
            float(pd.to_numeric(chosen["inner_dz_v2_v3"], errors="coerce").mean())
            if "inner_dz_v2_v3" in chosen
            else np.nan
        )
        stability = feature_set_jaccard_summary(res["selected_features_by_fold"])
        row = {
            "panel_combo": combo["panel_combo"],
            "panel_family": combo.get("panel_family", panel_family),
            "panel_count": combo["panel_count"],
            "feature_count": len(features),
            "validation_score": validation_score,
            "test_score": test_diag["mean_validation_annual_dz"],
            "validation_minus_test": validation_score - test_diag["mean_validation_annual_dz"],
            "validation_dz_v1_v2": validation_d12,
            "validation_dz_v2_v3": validation_d23,
            "test_dz_v1_v2": test_diag["dz_v1_v2"],
            "test_dz_v2_v3": test_diag["dz_v2_v3"],
            "test_annual_interval_gap": test_diag["annual_interval_gap"],
            "test_p_progression": test_diag["p_progression"],
            "n_subjects": res["n_subjects"],
            "cv_mode": res.get("cv_mode"),
            "cv_n_splits": res.get("cv_n_splits"),
            "inner_folds": res.get("inner_folds"),
            "mean_jaccard": stability["mean_jaccard"],
            "features": ", ".join(features),
        }
        rows.append(row)
        results[combo["panel_combo"]] = {
            **res,
            "intervals": intervals,
            "features": features,
            "panel_combo": combo["panel_combo"],
            "summary_row": row,
        }

    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary = summary.sort_values(
            ["test_score", "test_annual_interval_gap", "feature_count"],
            ascending=[False, True, True],
            kind="mergesort",
        ).reset_index(drop=True)
        summary["rank"] = np.arange(1, len(summary) + 1)
    return {"summary": summary, "results": results}


def best_panel_combination(summary: pd.DataFrame) -> pd.Series:
    """Return the top panel-combination row from a summary table."""
    if summary is None or summary.empty:
        raise ValueError("panel-combination summary is empty")
    return summary.sort_values(
        ["test_score", "test_annual_interval_gap", "feature_count"],
        ascending=[False, True, True],
        kind="mergesort",
    ).iloc[0]


__all__ = ["best_panel_combination", "evaluate_srm_panel_combinations"]
