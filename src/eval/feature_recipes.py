"""Fold-local feature recipes shared by TRACK-FA model notebooks."""
from __future__ import annotations

from itertools import combinations
from typing import Literal, Sequence

import numpy as np
import pandas as pd

from ..features.selection import progression_univariate_effects
from ..reporting.experiment_artifacts import (
    FEATURE_RECIPE_COLUMNS,
    split_participants_for_fold,
    validate_feature_recipe,
    validate_fold_manifest,
)
from .control_aware_selection import (
    control_aware_feature_ranking,
    single_feature_effect_table,
)


Strategy = Literal["frda_only", "control_aware"]


def _training_subset(
    frame: pd.DataFrame,
    participant_ids: set[str],
    *,
    participant_col: str,
) -> pd.DataFrame:
    participants = frame[participant_col].astype(str)
    return frame.loc[participants.isin(participant_ids)].copy()


def _frda_only_ranking(
    frda_train: pd.DataFrame,
    features: Sequence[str],
    *,
    pair_col: str,
    participant_col: str,
    visit_col: str,
) -> pd.DataFrame:
    effects = single_feature_effect_table(
        frda_train,
        features,
        cohort="FRDA",
        pair_col=pair_col,
        participant_col=participant_col,
        visit_col=visit_col,
    )
    progression = progression_univariate_effects(
        frda_train,
        features,
        subject_col=pair_col,
        visit_col=visit_col,
    )[["feature", "abs_mean_annual_dz", "annual_interval_gap"]]
    ranking = effects.merge(progression, on="feature", how="left")
    ranking = ranking.rename(columns={"abs_mean_annual_dz": "selection_score"})
    ranking = ranking.sort_values(
        ["selection_score", "annual_interval_gap", "feature"],
        ascending=[False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))
    return ranking


def _rank_training_data(
    frda_train: pd.DataFrame,
    control_train: pd.DataFrame | None,
    features: Sequence[str],
    *,
    strategy: Strategy,
    control_penalty: float,
    pair_col: str,
    participant_col: str,
    visit_col: str,
) -> pd.DataFrame:
    if strategy == "frda_only":
        return _frda_only_ranking(
            frda_train,
            features,
            pair_col=pair_col,
            participant_col=participant_col,
            visit_col=visit_col,
        )
    if strategy != "control_aware":
        raise ValueError(f"unknown feature-recipe strategy: {strategy!r}")
    if control_train is None or control_train.empty:
        raise ValueError("control_aware selection requires non-empty training controls")
    ranking = control_aware_feature_ranking(
        frda_train,
        control_train,
        features,
        control_penalty=float(control_penalty),
        pair_col=pair_col,
        participant_col=participant_col,
        visit_col=visit_col,
    ).rename(columns={"control_aware_score": "selection_score"})
    return ranking


def _recipe_rows(
    ranking: pd.DataFrame,
    *,
    strategy: Strategy,
    outer_fold: int,
    k: int,
    frda_train: pd.DataFrame,
    control_train: pd.DataFrame | None,
    pair_col: str,
    participant_col: str,
) -> list[dict[str, object]]:
    selected_features = set(ranking.head(min(int(k), len(ranking)))["feature"].astype(str))
    rows: list[dict[str, object]] = []
    for _, source in ranking.iterrows():
        rows.append(
            {
                "strategy": strategy,
                "outer_fold": int(outer_fold),
                "feature": str(source["feature"]),
                "rank": int(source["rank"]),
                "selected": str(source["feature"]) in selected_features,
                "selection_score": float(source["selection_score"]),
                "frda_pooled_d_z": source.get("pooled_d_z", source.get("frda_pooled_d_z", np.nan)),
                "frda_v1_v2_d_z": source.get("v1_v2_d_z", source.get("frda_v1_v2_d_z", np.nan)),
                "frda_v2_v3_d_z": source.get("v2_v3_d_z", source.get("frda_v2_v3_d_z", np.nan)),
                "control_pooled_d_z": source.get("control_pooled_d_z", np.nan),
                "control_v1_v2_d_z": source.get("control_v1_v2_d_z", np.nan),
                "control_v2_v3_d_z": source.get("control_v2_v3_d_z", np.nan),
                "train_frda_participants": int(frda_train[participant_col].nunique()),
                "train_frda_pairs": int(frda_train[pair_col].nunique()),
                "train_control_participants": (
                    int(control_train[participant_col].nunique()) if control_train is not None else 0
                ),
                "train_control_pairs": (
                    int(control_train[pair_col].nunique()) if control_train is not None else 0
                ),
            }
        )
    return rows


def build_fold_local_feature_recipe(
    frda_long: pd.DataFrame,
    features: Sequence[str],
    folds: pd.DataFrame,
    *,
    strategy: Strategy,
    k: int,
    control_long: pd.DataFrame | None = None,
    control_penalty: float = 0.5,
    pair_col: str = "pair_id",
    participant_col: str = "subject",
    visit_col: str = "visit",
) -> pd.DataFrame:
    """Select one feature recipe inside each persisted outer-training fold."""
    validate_fold_manifest(folds)
    panel = list(dict.fromkeys(map(str, features)))
    if not panel or int(k) <= 0:
        raise ValueError("features and k must be non-empty and positive")
    missing = [feature for feature in panel if feature not in frda_long.columns]
    if missing:
        raise KeyError(f"FRDA data are missing panel features: {missing[:10]}")
    if strategy == "control_aware":
        if control_long is None:
            raise ValueError("control_aware selection requires control_long")
        missing_control = [feature for feature in panel if feature not in control_long.columns]
        if missing_control:
            raise KeyError(f"control data are missing panel features: {missing_control[:10]}")

    fold_values = sorted(
        pd.to_numeric(
            folds.loc[folds["cohort"].astype(str).eq("FRDA"), "outer_fold"],
            errors="raise",
        ).astype(int).unique()
    )
    rows: list[dict[str, object]] = []
    for outer_fold in fold_values:
        frda_train_ids, _ = split_participants_for_fold(
            folds, cohort="FRDA", outer_fold=outer_fold
        )
        frda_train = _training_subset(
            frda_long, frda_train_ids, participant_col=participant_col
        )
        control_train = None
        if strategy == "control_aware":
            control_train_ids, _ = split_participants_for_fold(
                folds, cohort="Control", outer_fold=outer_fold
            )
            control_train = _training_subset(
                control_long, control_train_ids, participant_col=participant_col
            )
        ranking = _rank_training_data(
            frda_train,
            control_train,
            panel,
            strategy=strategy,
            control_penalty=control_penalty,
            pair_col=pair_col,
            participant_col=participant_col,
            visit_col=visit_col,
        )
        rows.extend(
            _recipe_rows(
                ranking,
                strategy=strategy,
                outer_fold=outer_fold,
                k=k,
                frda_train=frda_train,
                control_train=control_train,
                pair_col=pair_col,
                participant_col=participant_col,
            )
        )
    recipe = pd.DataFrame(rows, columns=FEATURE_RECIPE_COLUMNS)
    validate_feature_recipe(recipe, panel=panel)
    return recipe


def build_full_data_feature_recipe(
    frda_long: pd.DataFrame,
    features: Sequence[str],
    *,
    strategy: Strategy,
    k: int,
    control_long: pd.DataFrame | None = None,
    control_penalty: float = 0.5,
    pair_col: str = "pair_id",
    participant_col: str = "subject",
    visit_col: str = "visit",
) -> pd.DataFrame:
    """Build the non-OOF recipe used only for final refitting and deployment."""
    ranking = _rank_training_data(
        frda_long,
        control_long,
        features,
        strategy=strategy,
        control_penalty=control_penalty,
        pair_col=pair_col,
        participant_col=participant_col,
        visit_col=visit_col,
    )
    recipe = pd.DataFrame(
        _recipe_rows(
            ranking,
            strategy=strategy,
            outer_fold=0,
            k=k,
            frda_train=frda_long,
            control_train=control_long if strategy == "control_aware" else None,
            pair_col=pair_col,
            participant_col=participant_col,
        ),
        columns=FEATURE_RECIPE_COLUMNS,
    )
    validate_feature_recipe(recipe, panel=features)
    return recipe


def feature_recipe_stability(recipe: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return per-feature selection frequency and pairwise fold Jaccard values."""
    validate_feature_recipe(recipe)
    selected = recipe.loc[recipe["selected"]].copy()
    folds = sorted(recipe["outer_fold"].astype(int).unique())
    n_folds = len(folds)
    frequency = (
        selected.groupby(["strategy", "feature"], as_index=False)
        .agg(folds_selected=("outer_fold", "nunique"))
    )
    frequency["selection_frequency"] = frequency["folds_selected"] / n_folds
    frequency = frequency.sort_values(
        ["selection_frequency", "feature"], ascending=[False, True], kind="mergesort"
    ).reset_index(drop=True)

    sets = {
        fold: set(selected.loc[selected["outer_fold"].eq(fold), "feature"].astype(str))
        for fold in folds
    }
    jaccard_rows = []
    for left, right in combinations(folds, 2):
        union = sets[left] | sets[right]
        jaccard_rows.append(
            {
                "strategy": str(recipe["strategy"].iloc[0]),
                "fold_a": left,
                "fold_b": right,
                "jaccard": len(sets[left] & sets[right]) / len(union) if union else 1.0,
            }
        )
    return frequency, pd.DataFrame(jaccard_rows)


def compare_feature_recipes(
    baseline: pd.DataFrame,
    control_aware: pd.DataFrame,
) -> pd.DataFrame:
    """Summarise overlap and changes between two fold-local recipes."""
    validate_feature_recipe(baseline)
    validate_feature_recipe(control_aware)
    baseline_sets = {
        int(fold): set(group.loc[group["selected"], "feature"].astype(str))
        for fold, group in baseline.groupby("outer_fold")
    }
    control_sets = {
        int(fold): set(group.loc[group["selected"], "feature"].astype(str))
        for fold, group in control_aware.groupby("outer_fold")
    }
    if set(baseline_sets) != set(control_sets):
        raise ValueError("feature recipes do not contain identical outer folds")
    rows = []
    for fold in sorted(baseline_sets):
        before, after = baseline_sets[fold], control_sets[fold]
        union = before | after
        rows.append(
            {
                "outer_fold": fold,
                "baseline_features": len(before),
                "control_aware_features": len(after),
                "shared_features": len(before & after),
                "jaccard": len(before & after) / len(union) if union else 1.0,
                "removed_from_baseline": " | ".join(sorted(before - after)),
                "added_by_control_aware": " | ".join(sorted(after - before)),
            }
        )
    return pd.DataFrame(rows)


__all__ = [
    "build_fold_local_feature_recipe",
    "build_full_data_feature_recipe",
    "compare_feature_recipes",
    "feature_recipe_stability",
]
