"""Elbow-style feature-count selection for progression-sensitive models."""
from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from ..eval.intervals import adjacent_pair_interval_effect_summary, annual_tuning_diagnostics
from ..features.selection import feature_set_jaccard_summary
from ..models.srm_global import srm_global_loocv


DEFAULT_ELBOW_K_VALUES: tuple[int, ...] = (4, 6, 8, 10, 12, 16, 20, 30, 40)


def elbow_candidate_grid(
    *,
    k_values: Sequence[int] = DEFAULT_ELBOW_K_VALUES,
    include_full_count: int | None = None,
    mrmr_lambdas: Sequence[float] = (0.25, 0.5),
) -> list[dict]:
    """Return supervisor-facing feature-count candidates for elbow review."""
    candidates: list[dict] = []
    seen: set[tuple] = set()

    def add(row: Mapping) -> None:
        params = row.get("selection_params", {}) or {}
        key = (
            row.get("selection_method"),
            int(row.get("k", 0)),
            params.get("mrmr_redundancy_lambda"),
        )
        if key in seen:
            return
        seen.add(key)
        candidates.append(dict(row))

    for k in k_values:
        k = int(k)
        if k <= 0:
            continue
        add(
            {
                "label": f"progression_univariate_k{k}",
                "selection_method": "progression_univariate",
                "k": k,
                "selection_params": {},
            }
        )
        for lam in mrmr_lambdas:
            add(
                {
                    "label": f"progression_mrmr_k{k}_lam{lam:g}",
                    "selection_method": "progression_mrmr",
                    "k": k,
                    "selection_params": {"mrmr_redundancy_lambda": float(lam)},
                }
            )
    if include_full_count is not None:
        add(
            {
                "label": f"all_{int(include_full_count)}",
                "selection_method": "none",
                "k": int(include_full_count),
                "selection_params": {},
            }
        )
    return candidates


def evaluate_elbow_feature_counts(
    df_long: pd.DataFrame,
    feature_cols: Sequence[str],
    *,
    subject_col: str = "pair_id",
    visit_col: str = "visit",
    split_group_col: str = "subject",
    cv_n_splits: int = 5,
    random_seed: int = 42,
    n_boot: int = 300,
    k_values: Sequence[int] = DEFAULT_ELBOW_K_VALUES,
    mrmr_lambdas: Sequence[float] = (0.25, 0.5),
) -> dict:
    """Evaluate k-by-method candidates with participant-level CV.

    This is a non-nested elbow screen: each candidate's feature selection is
    still fit inside each outer training fold, but the table is used for
    human-readable k selection rather than as the final unbiased estimate.
    """
    features = [f for f in feature_cols if f in df_long.columns]
    candidates = elbow_candidate_grid(
        k_values=k_values,
        include_full_count=len(features),
        mrmr_lambdas=mrmr_lambdas,
    )
    rows = []
    results: dict[str, dict] = {}
    for cand_idx, cand in enumerate(candidates, start=1):
        res = srm_global_loocv(
            df_long,
            features,
            subject_col=subject_col,
            visit_col=visit_col,
            selection_method=cand["selection_method"],
            k=int(cand["k"]),
            cv_n_splits=cv_n_splits,
            random_seed=random_seed,
            split_group_col=split_group_col,
            selection_params=cand.get("selection_params", {}),
            compute_ci=False,
        )
        intervals = adjacent_pair_interval_effect_summary(
            res["oof_df"],
            pair_col=subject_col,
            visit_col=visit_col,
            score_col="score",
            n_boot=n_boot,
            seed=random_seed + cand_idx,
        )
        diag = annual_tuning_diagnostics(intervals)
        stability = feature_set_jaccard_summary(res["selected_features_by_fold"])
        selected_counts = [len(x) for x in res["selected_features_by_fold"] if x]
        feature_count = int(np.median(selected_counts)) if selected_counts else 0
        row = {
            "label": cand["label"],
            "selection_method": cand["selection_method"],
            "k": int(cand["k"]),
            "feature_count": feature_count,
            "mean_annual_d_z": diag["mean_validation_annual_dz"],
            "d_z_v1_v2": diag["dz_v1_v2"],
            "d_z_v2_v3": diag["dz_v2_v3"],
            "annual_interval_gap": diag["annual_interval_gap"],
            "p_progression": diag["p_progression"],
            "pooled_pair_d_z": res["d_score"],
            "n_subjects": res["n_subjects"],
            "cv_mode": res.get("cv_mode"),
            "cv_n_splits": res.get("cv_n_splits"),
            "mean_jaccard": stability["mean_jaccard"],
            "selection_params": cand.get("selection_params", {}),
        }
        rows.append(row)
        results[cand["label"]] = {**res, "intervals": intervals, "summary_row": row}

    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary = summary.sort_values(
            ["mean_annual_d_z", "annual_interval_gap", "feature_count"],
            ascending=[False, True, True],
            kind="mergesort",
        ).reset_index(drop=True)
        summary["rank"] = np.arange(1, len(summary) + 1)
    return {"summary": summary, "results": results, "candidates": candidates}


def select_feature_count_elbow(
    summary: pd.DataFrame,
    *,
    score_col: str = "mean_annual_d_z",
    feature_count_col: str = "feature_count",
    validation_col: str | None = None,
    performance_tolerance: float = 0.03,
    max_validation_test_gap: float | None = None,
) -> dict:
    """Choose the smallest feature count near the best observed performance.

    The rule encodes the supervisor discussion: accept a small performance loss
    in exchange for a simpler feature set, and stop just before performance
    drops materially. If validation/test columns are supplied, candidates with
    excessive validation-test optimism can be excluded.
    """
    if summary is None or summary.empty:
        raise ValueError("summary must contain at least one candidate")
    work = summary.copy()
    work[score_col] = pd.to_numeric(work[score_col], errors="coerce")
    work[feature_count_col] = pd.to_numeric(work[feature_count_col], errors="coerce")
    work = work.dropna(subset=[score_col, feature_count_col])
    if work.empty:
        raise ValueError("summary contains no finite score/feature-count candidates")
    if validation_col and validation_col in work.columns:
        work[validation_col] = pd.to_numeric(work[validation_col], errors="coerce")
        work["validation_minus_test"] = work[validation_col] - work[score_col]
        if max_validation_test_gap is not None:
            kept = work[work["validation_minus_test"] <= float(max_validation_test_gap)].copy()
            if not kept.empty:
                work = kept
    best_idx = work[score_col].idxmax()
    best = work.loc[best_idx]
    threshold = float(best[score_col]) - float(performance_tolerance)
    eligible = work[work[score_col] >= threshold].copy()
    eligible = eligible.sort_values(
        [feature_count_col, "annual_interval_gap", score_col],
        ascending=[True, True, False],
        kind="mergesort",
    )
    chosen = eligible.iloc[0]
    review = work.sort_values([feature_count_col, score_col], ascending=[True, False]).copy()
    review["best_score"] = float(best[score_col])
    review["score_loss_vs_best"] = float(best[score_col]) - review[score_col]
    review["within_elbow_tolerance"] = review["score_loss_vs_best"] <= float(performance_tolerance)
    review["elbow_selected"] = review.index == chosen.name
    return {
        "best": best,
        "chosen": chosen,
        "eligible": eligible,
        "review_table": review.reset_index(drop=True),
        "performance_tolerance": float(performance_tolerance),
        "threshold": threshold,
        "summary": (
            f"Elbow-selected {int(chosen[feature_count_col])} features "
            f"({chosen.get('label', 'candidate')}) because its {score_col}="
            f"{float(chosen[score_col]):.3f} is within {float(performance_tolerance):.3f} "
            f"of the best {score_col}={float(best[score_col]):.3f}."
        ),
    }


def elbow_nested_candidates(
    elbow_choice: Mapping,
    *,
    ridge_grid: Iterable[float] = (0.0,),
    covariance_shrinkage_grid: Iterable[float] = (0.0, 0.25, 0.45),
    z_clip_grid: Iterable[float | None] = (None, 2.75, 3.25),
) -> list[dict]:
    """Build model-tuning candidates after elbow-selected feature count."""
    row = dict(elbow_choice)
    method = row.get("selection_method", "none")
    k = int(row.get("k", row.get("feature_count", 0)))
    params = row.get("selection_params", {})
    if isinstance(params, str):
        params = {}
    out = []
    for ridge in ridge_grid:
        for shrink in covariance_shrinkage_grid:
            for z_clip in z_clip_grid:
                cand = {
                    "ridge": float(ridge),
                    "covariance_shrinkage": float(shrink),
                    "z_clip": z_clip,
                    "selection_method": method,
                    "k": k,
                    "selection_label": row.get("label", f"{method}_k{k}"),
                }
                cand.update(dict(params or {}))
                out.append(cand)
    return out


__all__ = [
    "DEFAULT_ELBOW_K_VALUES",
    "elbow_candidate_grid",
    "elbow_nested_candidates",
    "evaluate_elbow_feature_counts",
    "select_feature_count_elbow",
]
