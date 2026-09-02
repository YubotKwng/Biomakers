"""Patient-adaptive model tuning after feature-panel reduction."""
from __future__ import annotations

from itertools import combinations
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from ..config import Config
from ..eval.cv import interaction_loocv
from ..eval.intervals import adjacent_pair_interval_effect_summary, annual_tuning_diagnostics
from ..features.selection import progression_mrmr_selection, progression_univariate_effects


DEFAULT_MODULATOR_CANDIDATES = (
    ("age",),
    ("disease_duration",),
    ("gaa_1",),
    ("gaa_2",),
    ("age", "disease_duration"),
    ("age", "gaa_1"),
    ("age", "gaa_2"),
    ("disease_duration", "gaa_1"),
    ("disease_duration", "gaa_2"),
    ("gaa_1", "gaa_2"),
    ("age", "disease_duration", "gaa_1"),
    ("age", "disease_duration", "gaa_2"),
)


def available_modulator_sets(
    columns: Sequence[str],
    *,
    candidate_sets: Sequence[Sequence[str]] = DEFAULT_MODULATOR_CANDIDATES,
) -> list[tuple[str, ...]]:
    """Return candidate patient-modulator sets available in the dataframe."""
    available = set(columns)
    out = []
    for mods in candidate_sets:
        picked = tuple(m for m in mods if m in available)
        if len(picked) == len(tuple(mods)) and picked not in out:
            out.append(picked)
    return out


def feature_rank_table(
    df_long: pd.DataFrame,
    feature_cols: Sequence[str],
    *,
    subject_col: str = "pair_id",
    visit_col: str = "visit",
) -> pd.DataFrame:
    """Return per-feature longitudinal d_z ranking used by progression mRMR."""
    effects = progression_univariate_effects(
        df_long,
        [f for f in feature_cols if f in df_long.columns],
        subject_col=subject_col,
        visit_col=visit_col,
    ).copy()
    if effects.empty:
        return effects
    effects.insert(0, "rank_by_abs_mean_annual_dz", np.arange(1, len(effects) + 1))
    return effects


def selected_feature_sets_from_rank(
    df_long: pd.DataFrame,
    base_features: Sequence[str],
    *,
    subject_col: str = "pair_id",
    visit_col: str = "visit",
    top_ks: Sequence[int] = (8, 12, 20),
    mrmr_ks: Sequence[int] = (8, 12, 20),
    mrmr_redundancy_lambda: float = 0.25,
) -> dict[str, list[str]]:
    """Build reduced feature sets for adaptive-model tuning."""
    usable = [f for f in base_features if f in df_long.columns]
    ranks = feature_rank_table(
        df_long,
        usable,
        subject_col=subject_col,
        visit_col=visit_col,
    )
    out: dict[str, list[str]] = {}
    out[f"all_{len(usable)}"] = list(usable)
    for k in top_ks:
        out[f"progression_univariate_k{k}"] = ranks.head(min(int(k), len(ranks)))["feature"].tolist()
    for k in mrmr_ks:
        out[f"progression_mrmr_k{k}_lam{mrmr_redundancy_lambda:g}"] = progression_mrmr_selection(
            df_long,
            usable,
            subject_col=subject_col,
            visit_col=visit_col,
            k=int(k),
            redundancy_lambda=float(mrmr_redundancy_lambda),
        )
    return {name: feats for name, feats in out.items() if feats}


def evaluate_patient_adaptive_candidates(
    df_long: pd.DataFrame,
    feature_sets: Mapping[str, Sequence[str]],
    modulator_sets: Sequence[Sequence[str]],
    *,
    subject_col: str = "pair_id",
    visit_col: str = "visit",
    split_group_col: str = "subject",
    cv_n_splits: int = 5,
    random_seed: int = 42,
    n_boot: int = 300,
    alpha_grid: Sequence[float] = (0.003, 0.01, 0.03, 0.1, 0.3, 0.7, 1.0, 1.3, 1.7, 2.0, 3.0, 5.0),
    l1_ratio_grid: Sequence[float] = (0.0,),
    z_clip_grid: Sequence[float | None] = (None, 4.0, 2.75, 2.0),
    inner_cv_splits: int = 3,
) -> dict:
    """Tune adaptive interaction models over reduced features and modulators."""
    rows = []
    results: dict[str, dict] = {}
    trial_id = 0
    for feature_set_name, features_raw in feature_sets.items():
        features = [f for f in features_raw if f in df_long.columns]
        if not features:
            continue
        for mods_raw in modulator_sets:
            mods = tuple(m for m in mods_raw if m in df_long.columns)
            if not mods:
                continue
            for z_clip in z_clip_grid:
                trial_id += 1
                config = Config(
                    interaction_en_alpha_grid=tuple(float(x) for x in alpha_grid),
                    interaction_en_l1_ratio_grid=tuple(float(x) for x in l1_ratio_grid),
                    interaction_inner_cv_splits=int(inner_cv_splits),
                    interaction_z_clip=z_clip,
                )
                res = interaction_loocv(
                    df_long,
                    features,
                    mods,
                    subject_col=subject_col,
                    visit_col=visit_col,
                    selection_method="none",
                    k=len(features),
                    cv_n_splits=cv_n_splits,
                    random_seed=random_seed,
                    config=config,
                    compute_ci=False,
                    split_group_col=split_group_col,
                )
                intervals = adjacent_pair_interval_effect_summary(
                    res["oof_df"],
                    pair_col=subject_col,
                    visit_col=visit_col,
                    score_col="score",
                    n_boot=n_boot,
                    seed=random_seed + trial_id,
                )
                diag = annual_tuning_diagnostics(intervals)
                chosen = res.get("chosen_params_df", pd.DataFrame())
                chosen_alpha = np.nan
                chosen_l1_ratio = np.nan
                if isinstance(chosen, pd.DataFrame) and not chosen.empty:
                    if "alpha" in chosen:
                        alpha_values = pd.to_numeric(chosen["alpha"], errors="coerce").dropna()
                        if not alpha_values.empty:
                            chosen_alpha = float(alpha_values.mode().iloc[0])
                    if "l1_ratio" in chosen:
                        l1_values = pd.to_numeric(chosen["l1_ratio"], errors="coerce").dropna()
                        if not l1_values.empty:
                            chosen_l1_ratio = float(l1_values.mode().iloc[0])
                row = {
                    "trial_id": trial_id,
                    "feature_set": feature_set_name,
                    "feature_count": len(features),
                    "modulators": ",".join(mods),
                    "modulator_count": len(mods),
                    "z_clip": z_clip,
                    "validation_score": (
                        float(pd.to_numeric(chosen.get("inner_d_score"), errors="coerce").mean())
                        if isinstance(chosen, pd.DataFrame) and "inner_d_score" in chosen
                        else np.nan
                    ),
                    "test_score": diag["mean_validation_annual_dz"],
                    "validation_minus_test": np.nan,
                    "test_dz_v1_v2": diag["dz_v1_v2"],
                    "test_dz_v2_v3": diag["dz_v2_v3"],
                    "test_annual_interval_gap": diag["annual_interval_gap"],
                    "test_p_progression": diag["p_progression"],
                    "pooled_pair_d_z": res["d_score"],
                    "n_subjects": int(res.get("n_split_groups", res["n_subjects"])),
                    "n_pairs": int(res["n_subjects"]),
                    "cv_mode": res.get("cv_mode"),
                    "cv_n_splits": res.get("cv_n_splits"),
                    "alpha": chosen_alpha,
                    "l1_ratio": chosen_l1_ratio,
                    "features": ", ".join(features),
                }
                if np.isfinite(row["validation_score"]):
                    row["validation_minus_test"] = row["validation_score"] - row["test_score"]
                rows.append(row)
                results[str(trial_id)] = {
                    **res,
                    "intervals": intervals,
                    "features": features,
                    "modulators": list(mods),
                    "summary_row": row,
                }
    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary = summary.sort_values(
            ["test_score", "test_annual_interval_gap", "feature_count", "modulator_count"],
            ascending=[False, True, True, True],
            kind="mergesort",
        ).reset_index(drop=True)
        summary["rank"] = np.arange(1, len(summary) + 1)
    return {"summary": summary, "results": results}


__all__ = [
    "DEFAULT_MODULATOR_CANDIDATES",
    "available_modulator_sets",
    "evaluate_patient_adaptive_candidates",
    "feature_rank_table",
    "selected_feature_sets_from_rank",
]
