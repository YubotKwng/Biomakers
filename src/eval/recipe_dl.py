"""Deep-learning evaluation for persisted TRACK-FA feature recipes."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from ..reporting.experiment_artifacts import split_participants_for_fold, validate_feature_recipe, validate_fold_manifest, validate_oof_uniqueness
from ..training.fusion import score_fusion_progression, train_fusion_model
from ..training.pair import score_pair_progression, train_pair_model
from .recipe_models import (
    _complete_rows,
    _score_metadata,
    _subset_participants,
    comparison_table,
    selected_features_for_fold,
    summarise_oof_performance,
)


def _fusion_meta(
    features: Sequence[str],
    structural_features: Sequence[str],
    diffusion_features: Sequence[str],
) -> dict[str, list[int]]:
    structural = set(structural_features)
    diffusion = set(diffusion_features)
    return {
        "struct_idx": [index for index, feature in enumerate(features) if feature in structural],
        "diff_idx": [index for index, feature in enumerate(features) if feature in diffusion],
        "back_idx": [
            index
            for index, feature in enumerate(features)
            if feature not in structural and feature not in diffusion
        ],
    }


def _pair_visit_scores(frame: pd.DataFrame, pair_deltas: Mapping[str, float], pair_col: str) -> np.ndarray:
    signs = np.where(pd.to_numeric(frame["visit"], errors="coerce").eq(1), -0.5, 0.5)
    deltas = frame[pair_col].astype(str).map(pair_deltas).to_numpy(dtype=float)
    return signs * deltas


def run_dl_recipe_comparison(
    frda_long: pd.DataFrame,
    control_long: pd.DataFrame,
    folds: pd.DataFrame,
    recipe: pd.DataFrame,
    *,
    run_id: str,
    structural_features: Sequence[str],
    diffusion_features: Sequence[str],
    device,
    fusion_kwargs: Mapping[str, object] | None = None,
    pair_kwargs: Mapping[str, object] | None = None,
    architectures: Sequence[str] = ("fusion_mlp", "pair_model"),
    seed: int = 42,
    n_boot: int = 300,
    pair_col: str = "pair_id",
    participant_col: str = "subject",
    visit_col: str = "visit",
) -> dict[str, pd.DataFrame]:
    """Train each DL architecture on FRDA and score held-out FRDA/controls."""
    validate_fold_manifest(folds)
    validate_feature_recipe(recipe)
    allowed = {"fusion_mlp", "pair_model"}
    requested = list(dict.fromkeys(architectures))
    unknown = sorted(set(requested) - allowed)
    if unknown:
        raise ValueError(f"unknown DL architectures: {unknown}")

    common = {
        "epochs": 60,
        "patience": 8,
        "lr": 3e-3,
        "weight_decay": 1e-5,
        "dropout": 0.0,
        "val_fraction": 0.2,
        "use_clinical_heads": False,
        "lambda_prog": 1.0,
        "lambda_fars": 0.0,
        "lambda_sara": 0.0,
    }
    fusion_options = {**common, "z_clip": 3.0, **dict(fusion_kwargs or {})}
    pair_options = {**common, "z_clip": None, **dict(pair_kwargs or {})}
    for options in (fusion_options, pair_options):
        if options.get("use_clinical_heads") or float(options.get("lambda_fars", 0)) or float(options.get("lambda_sara", 0)):
            raise ValueError("clinical auxiliary heads must remain disabled")

    strategies = list(dict.fromkeys(recipe["strategy"].astype(str)))
    outer_folds = sorted(pd.to_numeric(folds["outer_fold"], errors="raise").astype(int).unique())
    oof_parts: list[pd.DataFrame] = []
    diagnostic_rows: list[dict[str, object]] = []

    for architecture in requested:
        for strategy in strategies:
            for outer_fold in outer_folds:
                features = selected_features_for_fold(recipe, strategy, outer_fold)
                train_ids, frda_test_ids = split_participants_for_fold(folds, cohort="FRDA", outer_fold=outer_fold)
                _, control_test_ids = split_participants_for_fold(folds, cohort="Control", outer_fold=outer_fold)
                frda_train = _complete_rows(
                    _subset_participants(frda_long, train_ids, participant_col=participant_col),
                    features, pair_col=pair_col, participant_col=participant_col, visit_col=visit_col,
                )
                frda_test = _complete_rows(
                    _subset_participants(frda_long, frda_test_ids, participant_col=participant_col),
                    features, pair_col=pair_col, participant_col=participant_col, visit_col=visit_col,
                )
                control_test = _complete_rows(
                    _subset_participants(control_long, control_test_ids, participant_col=participant_col),
                    features, pair_col=pair_col, participant_col=participant_col, visit_col=visit_col,
                )
                scaler = StandardScaler().fit(frda_train[features].to_numpy(dtype=float))
                started = time.perf_counter()

                if architecture == "fusion_mlp":
                    meta = _fusion_meta(features, structural_features, diffusion_features)
                    options = {**fusion_options, "seed": int(seed) + outer_fold}
                    model, best_epoch = train_fusion_model(
                        frda_train, features, meta, scaler,
                        subject_col=pair_col, split_group_col=participant_col,
                        device=device, **options,
                    )
                    scored = {
                        "FRDA": score_fusion_progression(
                            model, frda_test, features, meta, scaler,
                            device=device, z_clip=options.get("z_clip"),
                        ),
                        "Control": score_fusion_progression(
                            model, control_test, features, meta, scaler,
                            device=device, z_clip=options.get("z_clip"),
                        ),
                    }
                    score_frames = {"FRDA": frda_test, "Control": control_test}
                else:
                    options = {**pair_options, "seed": int(seed) + outer_fold}
                    model, best_epoch = train_pair_model(
                        frda_train, features, scaler,
                        subject_col=pair_col, split_group_col=participant_col,
                        device=device, **options,
                    )
                    score_frames = {"FRDA": frda_test, "Control": control_test}
                    scored = {}
                    for cohort, frame in score_frames.items():
                        pair_ids, deltas = score_pair_progression(
                            model, frame, features, scaler,
                            subject_col=pair_col, device=device, z_clip=options.get("z_clip"),
                        )
                        delta_map = dict(zip(map(str, pair_ids), map(float, deltas)))
                        scored[cohort] = _pair_visit_scores(frame, delta_map, pair_col)

                for cohort, frame in score_frames.items():
                    oof_parts.append(
                        _score_metadata(
                            frame, scored[cohort], run_id=run_id, model=architecture,
                            strategy=strategy, cohort=cohort, outer_fold=outer_fold,
                            feature_count=len(features), modulator_recipe="none",
                            pair_col=pair_col, participant_col=participant_col, visit_col=visit_col,
                        )
                    )
                diagnostic_rows.append(
                    {
                        "model": architecture,
                        "selection_strategy": strategy,
                        "outer_fold": outer_fold,
                        "feature_count": len(features),
                        "train_frda_participants": int(frda_train[participant_col].nunique()),
                        "train_frda_pairs": int(frda_train[pair_col].nunique()),
                        "test_frda_participants": int(frda_test[participant_col].nunique()),
                        "test_control_participants": int(control_test[participant_col].nunique()),
                        "best_epoch": int(best_epoch),
                        "runtime_seconds": float(time.perf_counter() - started),
                        "scaler_mean_abs_max": float(np.max(np.abs(scaler.mean_))),
                        "clinical_heads_enabled": False,
                        "features": "|".join(features),
                    }
                )

    oof = pd.concat(oof_parts, ignore_index=True)
    validate_oof_uniqueness(oof)
    performance, sites = summarise_oof_performance(oof, n_boot=n_boot, seed=seed)
    return {
        "oof_visit_scores": oof,
        "performance": performance,
        "site_diagnostics": sites,
        "training_diagnostics": pd.DataFrame(diagnostic_rows),
        "comparison": comparison_table(performance),
    }


__all__ = ["run_dl_recipe_comparison"]
