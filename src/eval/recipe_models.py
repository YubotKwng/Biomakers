"""Fold-locked SRM and patient-adaptive evaluation for feature recipes.

The helpers in this module consume feature sets that were already selected
inside each persisted outer-training fold.  They never reselect imaging
features.  Model tuning uses FRDA training data only; fitted preprocessing and
weights are then applied unchanged to held-out FRDA and controls.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import replace

import numpy as np
import pandas as pd

from ..config import Config
from ..data.mri_qc import site_effect_screen
from ..data.qc import standardize_train_test
from ..models.interaction import InteractionLinearComposite
from ..models.srm_global import SRMGlobalLinear, srm_global_loocv
from ..reporting.experiment_artifacts import (
    COEFFICIENT_COLUMNS,
    OOF_VISIT_COLUMNS,
    PERFORMANCE_COLUMNS,
    split_participants_for_fold,
    validate_feature_recipe,
    validate_fold_manifest,
    validate_oof_uniqueness,
)
from .cv import interaction_loocv
from .intervals import (
    adjacent_pair_interval_effect_summary,
    annual_tuning_diagnostics,
    pooled_adjacent_pair_effect_summary,
)
from .metrics import bootstrap_paired_metric, paired_cohens_dz
from .model_selection import select_hierarchical_candidate


DEFAULT_MODULATOR_SETS = (
    ("age",),
    ("disease_duration",),
    ("gaa_1",),
    ("age", "disease_duration"),
    ("age", "gaa_1"),
    ("disease_duration", "gaa_1"),
    ("age", "disease_duration", "gaa_1"),
)


def selected_features_for_fold(
    recipe: pd.DataFrame,
    strategy: str,
    outer_fold: int,
) -> list[str]:
    """Return the persisted selected features in rank order."""
    validate_feature_recipe(recipe)
    rows = recipe.loc[
        recipe["strategy"].astype(str).eq(str(strategy))
        & pd.to_numeric(recipe["outer_fold"], errors="coerce").eq(int(outer_fold))
        & recipe["selected"].astype(bool)
    ].sort_values("rank", kind="mergesort")
    if rows.empty:
        raise ValueError(f"no selected features for {strategy!r}, fold {outer_fold}")
    return rows["feature"].astype(str).tolist()


def _complete_rows(
    frame: pd.DataFrame,
    required_values: Sequence[str],
    *,
    pair_col: str,
    participant_col: str,
    visit_col: str,
) -> pd.DataFrame:
    required = [pair_col, participant_col, visit_col, *required_values]
    missing = [column for column in required if column not in frame]
    if missing:
        raise KeyError(f"frame is missing required columns: {missing}")
    # Preserve non-required metadata (notably observed control age) so a
    # frozen patient-adaptive model can use it when that modulator is chosen.
    work = frame.copy()
    work = work.dropna(subset=required)
    work[visit_col] = pd.to_numeric(work[visit_col], errors="coerce")
    work = work[work[visit_col].isin([1, 2])].copy()
    complete = work.groupby(pair_col)[visit_col].agg(
        lambda values: {1, 2}.issubset(set(values.astype(int)))
    )
    return work.loc[work[pair_col].isin(complete.index[complete])].reset_index(drop=True)


def _subset_participants(
    frame: pd.DataFrame,
    participant_ids: set[str],
    *,
    participant_col: str,
) -> pd.DataFrame:
    return frame.loc[frame[participant_col].astype(str).isin(participant_ids)].copy()


def _recipe_strategies(recipe: pd.DataFrame) -> list[str]:
    return list(dict.fromkeys(recipe["strategy"].astype(str)))


def _score_metadata(
    frame: pd.DataFrame,
    scores: np.ndarray,
    *,
    run_id: str,
    model: str,
    strategy: str,
    cohort: str,
    outer_fold: int,
    feature_count: int,
    modulator_recipe: str,
    pair_col: str,
    participant_col: str,
    visit_col: str,
) -> pd.DataFrame:
    interval = (
        frame["interval"].astype(str)
        if "interval" in frame
        else frame[pair_col].astype(str).str.upper().str.extract(r"(V\d+V\d+)", expand=False)
        .map({"V1V2": "V1->V2", "V2V3": "V2->V3"})
    )
    site = frame["site"] if "site" in frame else np.nan
    out = pd.DataFrame(
        {
            "run_id": str(run_id),
            "model": str(model),
            "selection_strategy": str(strategy),
            "cohort": str(cohort),
            "outer_fold": int(outer_fold),
            "participant_id": frame[participant_col].astype(str).to_numpy(),
            "pair_id": frame[pair_col].astype(str).to_numpy(),
            "visit": frame[visit_col].astype(int).to_numpy(),
            "interval": interval.to_numpy(),
            "score": np.asarray(scores, dtype=float),
            "site": np.asarray(site),
            "feature_count": int(feature_count),
            "modulator_recipe": str(modulator_recipe),
        },
        columns=OOF_VISIT_COLUMNS,
    )
    return out


def _pair_deltas(scores: pd.DataFrame) -> pd.DataFrame:
    metadata = scores[["pair_id", "participant_id", "interval", "site"]].drop_duplicates("pair_id")
    paired = scores.pivot_table(index="pair_id", columns="visit", values="score", aggfunc="mean")
    if 1 not in paired or 2 not in paired:
        return pd.DataFrame(columns=["pair_id", "participant_id", "interval", "site", "delta"])
    out = paired[[1, 2]].dropna().reset_index()
    out["delta"] = out[2] - out[1]
    return metadata.merge(out[["pair_id", "delta"]], on="pair_id", how="inner")


def summarise_oof_performance(
    scores: pd.DataFrame,
    *,
    n_boot: int = 1000,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return schema-valid effect rows and descriptive site diagnostics."""
    validate_oof_uniqueness(scores)
    performance_rows: list[dict[str, object]] = []
    site_rows: list[pd.DataFrame] = []
    group_cols = ["run_id", "model", "selection_strategy", "cohort"]
    for keys, group in scores.groupby(group_cols, sort=False):
        run_id, model, strategy, cohort = keys
        deltas = _pair_deltas(group)
        interval_groups = [("pooled annual", deltas)] + [
            (label, deltas.loc[deltas["interval"].eq(label)])
            for label in ("V1->V2", "V2->V3")
        ]
        for offset, (label, part) in enumerate(interval_groups):
            values = pd.to_numeric(part["delta"], errors="coerce").dropna().to_numpy(dtype=float)
            effect = paired_cohens_dz(values) if len(values) >= 2 else np.nan
            boot = bootstrap_paired_metric(
                values,
                paired_cohens_dz,
                n_boot=int(n_boot),
                seed=int(seed) + offset,
            )
            performance_rows.append(
                {
                    "run_id": run_id,
                    "model": model,
                    "selection_strategy": strategy,
                    "cohort": cohort,
                    "interval": label,
                    "n_participants": int(part["participant_id"].nunique()),
                    "n_pairs": int(len(part)),
                    "d_z": effect,
                    "mean_delta": float(np.mean(values)) if len(values) else np.nan,
                    "sd_delta": float(np.std(values, ddof=1)) if len(values) >= 2 else np.nan,
                    "ci_low": boot["ci_low"],
                    "ci_high": boot["ci_high"],
                    "p_delta_gt_0": float(np.mean(values > 0)) if len(values) else np.nan,
                }
            )
        if "site" in deltas and deltas["site"].notna().any():
            site = site_effect_screen(deltas, ["delta"], site_col="site", covariates=[])
            if not site.empty:
                site.insert(0, "run_id", run_id)
                site.insert(1, "model", model)
                site.insert(2, "selection_strategy", strategy)
                site.insert(3, "cohort", cohort)
                site_rows.append(site)
    performance = pd.DataFrame(performance_rows, columns=PERFORMANCE_COLUMNS)
    sites = pd.concat(site_rows, ignore_index=True) if site_rows else pd.DataFrame()
    return performance, sites


def comparison_table(performance: pd.DataFrame) -> pd.DataFrame:
    """Place FRDA and control metrics side by side for notebook display."""
    index = ["model", "selection_strategy"]
    fields = ["d_z", "ci_low", "ci_high", "p_delta_gt_0", "n_participants", "n_pairs"]
    pieces = []
    for cohort in ("FRDA", "Control"):
        part = performance.loc[performance["cohort"].eq(cohort)].pivot_table(
            index=index,
            columns="interval",
            values=fields,
            aggfunc="first",
        )
        part.columns = [
            f"{cohort.lower()}_{str(interval).lower().replace(' ', '_').replace('->', '_')}_{field}"
            for field, interval in part.columns
        ]
        pieces.append(part)
    out = pd.concat(pieces, axis=1).reset_index()
    frda = "frda_pooled_annual_d_z"
    control = "control_pooled_annual_d_z"
    if frda in out and control in out:
        out["signed_frda_control_contrast"] = out[frda] - out[control]
        out["absolute_control_d_z"] = out[control].abs()
    if "frda_v1_v2_d_z" in out and "frda_v2_v3_d_z" in out:
        out["frda_interval_gap"] = (out["frda_v1_v2_d_z"] - out["frda_v2_v3_d_z"]).abs()
    return out


def frda_control_contrast_table(
    scores: pd.DataFrame,
    *,
    n_boot: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """Bootstrap the difference between FRDA and control paired effect sizes.

    Participants are resampled independently within each cohort, preserving
    both annual intervals from a participant when available.
    """
    validate_oof_uniqueness(scores)
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(seed)
    for keys, group in scores.groupby(["run_id", "model", "selection_strategy"], sort=False):
        run_id, model, strategy = keys
        cohort_deltas = {
            cohort: _pair_deltas(group.loc[group["cohort"].eq(cohort)])
            for cohort in ("FRDA", "Control")
        }
        if any(table.empty for table in cohort_deltas.values()):
            continue
        point = {
            cohort: paired_cohens_dz(table["delta"].to_numpy(dtype=float))
            for cohort, table in cohort_deltas.items()
        }
        grouped = {
            cohort: {
                participant: part["delta"].to_numpy(dtype=float)
                for participant, part in table.groupby("participant_id", sort=False)
            }
            for cohort, table in cohort_deltas.items()
        }
        boot = []
        for _ in range(int(n_boot)):
            estimates = {}
            for cohort in ("FRDA", "Control"):
                participants = np.asarray(list(grouped[cohort]), dtype=object)
                sampled = rng.choice(participants, size=len(participants), replace=True)
                values = np.concatenate([grouped[cohort][participant] for participant in sampled])
                estimates[cohort] = paired_cohens_dz(values)
            if np.isfinite(estimates["FRDA"]) and np.isfinite(estimates["Control"]):
                boot.append(float(estimates["FRDA"] - estimates["Control"]))
        boot_values = np.asarray(boot, dtype=float)
        rows.append(
            {
                "run_id": run_id,
                "model": model,
                "selection_strategy": strategy,
                "frda_d_z": point["FRDA"],
                "control_d_z": point["Control"],
                "signed_frda_control_contrast": point["FRDA"] - point["Control"],
                "contrast_ci_low": float(np.percentile(boot_values, 2.5)) if len(boot_values) >= 10 else np.nan,
                "contrast_ci_high": float(np.percentile(boot_values, 97.5)) if len(boot_values) >= 10 else np.nan,
                "p_contrast_gt_0": float(np.mean(boot_values > 0)) if len(boot_values) else np.nan,
                "frda_n_participants": int(cohort_deltas["FRDA"]["participant_id"].nunique()),
                "frda_n_pairs": int(len(cohort_deltas["FRDA"])),
                "control_n_participants": int(cohort_deltas["Control"]["participant_id"].nunique()),
                "control_n_pairs": int(len(cohort_deltas["Control"])),
            }
        )
    return pd.DataFrame(rows)


def _fit_srm(
    train: pd.DataFrame,
    frames: Mapping[str, pd.DataFrame],
    features: Sequence[str],
    *,
    ridge: float,
    covariance_shrinkage: float,
    z_clip: float | None,
    pair_col: str,
    visit_col: str,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    feature_list = list(features)
    x_train = train[feature_list].to_numpy(dtype=float)
    names = list(frames)
    lengths = [len(frames[name]) for name in names]
    x_score = np.vstack([frames[name][feature_list].to_numpy(dtype=float) for name in names])
    x_train_s, x_score_s, center, scale = standardize_train_test(x_train, x_score)
    if z_clip is not None:
        x_train_s = np.clip(x_train_s, -float(z_clip), float(z_clip))
        x_score_s = np.clip(x_score_s, -float(z_clip), float(z_clip))
    fitted = SRMGlobalLinear(
        ridge=float(ridge),
        covariance_shrinkage=float(covariance_shrinkage),
    ).fit(x_train_s, train[pair_col].to_numpy(), train[visit_col].to_numpy())
    scored: dict[str, np.ndarray] = {}
    start = 0
    for name, length in zip(names, lengths):
        scored[name] = fitted.score(x_score_s[start : start + length])
        start += length
    return scored, {
        "model": fitted,
        "features": feature_list,
        "center": np.asarray(center, dtype=float),
        "scale": np.asarray(scale, dtype=float),
        "coef": np.asarray(fitted.coef_, dtype=float),
        "ridge": float(ridge),
        "covariance_shrinkage": float(covariance_shrinkage),
        "z_clip": z_clip,
    }


def _select_srm_settings(
    train: pd.DataFrame,
    features: Sequence[str],
    *,
    candidates: Sequence[Mapping[str, object]],
    pair_col: str,
    participant_col: str,
    visit_col: str,
    inner_folds: int,
    seed: int,
) -> tuple[dict[str, object], pd.DataFrame]:
    rows = []
    for candidate_index, candidate_raw in enumerate(candidates, start=1):
        candidate = dict(candidate_raw)
        result = srm_global_loocv(
            train,
            features,
            pair_col,
            visit_col,
            selection_method="none",
            k=len(features),
            ridge=float(candidate.get("ridge", 0.0)),
            covariance_shrinkage=float(candidate.get("covariance_shrinkage", 0.45)),
            z_clip=candidate.get("z_clip"),
            cv_n_splits=int(inner_folds),
            random_seed=int(seed),
            compute_ci=False,
            split_group_col=participant_col,
        )
        intervals = adjacent_pair_interval_effect_summary(
            result["oof_df"], pair_col=pair_col, visit_col=visit_col, n_boot=0, seed=seed
        )
        diag = annual_tuning_diagnostics(intervals)
        rows.append(
            {
                "candidate_index": candidate_index,
                **candidate,
                "mean_validation_annual_dz": diag["mean_validation_annual_dz"],
                "dz_v1_v2": diag["dz_v1_v2"],
                "dz_v2_v3": diag["dz_v2_v3"],
                "annual_interval_gap": diag["annual_interval_gap"],
                "p_progression": diag["p_progression"],
                "feature_count": len(features),
                "se_validation_dz": 0.0,
            }
        )
    table = pd.DataFrame(rows)
    choice = select_hierarchical_candidate(table)
    return {
        "ridge": float(choice.get("ridge", 0.0)),
        "covariance_shrinkage": float(choice.get("covariance_shrinkage", 0.45)),
        "z_clip": None if pd.isna(choice.get("z_clip")) else float(choice.get("z_clip")),
    }, table


def run_srm_recipe_comparison(
    frda_long: pd.DataFrame,
    control_long: pd.DataFrame,
    folds: pd.DataFrame,
    recipe: pd.DataFrame,
    *,
    run_id: str,
    candidates: Sequence[Mapping[str, object]] | None = None,
    inner_folds: int = 3,
    seed: int = 42,
    n_boot: int = 1000,
    pair_col: str = "pair_id",
    participant_col: str = "subject",
    visit_col: str = "visit",
) -> dict[str, pd.DataFrame]:
    """Evaluate global SRM on both persisted feature-selection strategies."""
    validate_fold_manifest(folds)
    validate_feature_recipe(recipe)
    grid = list(candidates or (
        {"ridge": 0.0, "covariance_shrinkage": 0.35, "z_clip": None},
        {"ridge": 0.0, "covariance_shrinkage": 0.45, "z_clip": None},
        {"ridge": 0.0, "covariance_shrinkage": 0.45, "z_clip": 2.75},
    ))
    oof_parts: list[pd.DataFrame] = []
    parameter_rows: list[dict[str, object]] = []
    coefficient_rows: list[dict[str, object]] = []
    tuning_parts: list[pd.DataFrame] = []
    outer_folds = sorted(pd.to_numeric(folds["outer_fold"], errors="coerce").dropna().astype(int).unique())
    for strategy in _recipe_strategies(recipe):
        for outer_fold in outer_folds:
            features = selected_features_for_fold(recipe, strategy, outer_fold)
            frda_train_ids, frda_test_ids = split_participants_for_fold(
                folds, cohort="FRDA", outer_fold=outer_fold
            )
            _, control_test_ids = split_participants_for_fold(
                folds, cohort="Control", outer_fold=outer_fold
            )
            frda_train = _complete_rows(
                _subset_participants(frda_long, frda_train_ids, participant_col=participant_col),
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
            settings, tuning = _select_srm_settings(
                frda_train,
                features,
                candidates=grid,
                pair_col=pair_col,
                participant_col=participant_col,
                visit_col=visit_col,
                inner_folds=inner_folds,
                seed=seed + outer_fold,
            )
            tuning.insert(0, "selection_strategy", strategy)
            tuning.insert(1, "outer_fold", outer_fold)
            tuning_parts.append(tuning)
            scored, fitted = _fit_srm(
                frda_train,
                {"FRDA": frda_test, "Control": control_test},
                features,
                pair_col=pair_col,
                visit_col=visit_col,
                **settings,
            )
            for cohort, frame in (("FRDA", frda_test), ("Control", control_test)):
                oof_parts.append(
                    _score_metadata(
                        frame,
                        scored[cohort],
                        run_id=run_id,
                        model="srm_global_linear",
                        strategy=strategy,
                        cohort=cohort,
                        outer_fold=outer_fold,
                        feature_count=len(features),
                        modulator_recipe="none",
                        pair_col=pair_col,
                        participant_col=participant_col,
                        visit_col=visit_col,
                    )
                )
            parameter_rows.append(
                {
                    "model": "srm_global_linear",
                    "selection_strategy": strategy,
                    "outer_fold": outer_fold,
                    "train_frda_participants": frda_train[participant_col].nunique(),
                    "train_frda_pairs": frda_train[pair_col].nunique(),
                    "test_frda_participants": frda_test[participant_col].nunique(),
                    "test_control_participants": control_test[participant_col].nunique(),
                    "feature_count": len(features),
                    "features": "|".join(features),
                    **settings,
                }
            )
            for feature, center, scale, coefficient in zip(
                features, fitted["center"], fitted["scale"], fitted["coef"]
            ):
                coefficient_rows.append(
                    {
                        "run_id": run_id,
                        "model": "srm_global_linear",
                        "selection_strategy": strategy,
                        "fit_scope": "outer_fold",
                        "outer_fold": outer_fold,
                        "feature": feature,
                        "coefficient": coefficient,
                        "training_mean": center,
                        "training_sd": scale,
                        "z_clip": settings["z_clip"],
                        "modulator_reference_profile": "none",
                        "selection_frequency": np.nan,
                    }
                )
    oof = pd.concat(oof_parts, ignore_index=True)
    validate_oof_uniqueness(oof)
    performance, sites = summarise_oof_performance(oof, n_boot=n_boot, seed=seed)
    return {
        "oof_visit_scores": oof,
        "performance": performance,
        "site_diagnostics": sites,
        "fold_parameters": pd.DataFrame(parameter_rows),
        "coefficients": pd.DataFrame(coefficient_rows, columns=COEFFICIENT_COLUMNS),
        "tuning": pd.concat(tuning_parts, ignore_index=True),
        "comparison": comparison_table(performance),
    }


def _available_modulators(frame: pd.DataFrame, candidates: Sequence[Sequence[str]]) -> list[tuple[str, ...]]:
    available = set(frame)
    return [tuple(group) for group in candidates if group and set(group) <= available]


def _adaptive_inner_score(
    train: pd.DataFrame,
    features: Sequence[str],
    modulators: Sequence[str],
    *,
    pair_col: str,
    participant_col: str,
    visit_col: str,
    inner_folds: int,
    seed: int,
    config: Config,
) -> tuple[dict[str, object], pd.DataFrame]:
    result = interaction_loocv(
        train,
        features,
        modulators,
        subject_col=pair_col,
        visit_col=visit_col,
        selection_method="none",
        k=len(features),
        cv_n_splits=inner_folds,
        random_seed=seed,
        config=config,
        compute_ci=False,
        split_group_col=participant_col,
    )
    intervals = adjacent_pair_interval_effect_summary(
        result["oof_df"], pair_col=pair_col, visit_col=visit_col, n_boot=0, seed=seed
    )
    diag = annual_tuning_diagnostics(intervals)
    row = {
        "modulators": ",".join(modulators),
        "modulator_count": len(modulators),
        "mean_validation_annual_dz": diag["mean_validation_annual_dz"],
        "dz_v1_v2": diag["dz_v1_v2"],
        "dz_v2_v3": diag["dz_v2_v3"],
        "annual_interval_gap": diag["annual_interval_gap"],
        "p_progression": diag["p_progression"],
        "feature_count": len(features),
        "se_validation_dz": 0.0,
        "n_participants": train[participant_col].nunique(),
        "n_pairs": train[pair_col].nunique(),
    }
    return row, result.get("chosen_params_df", pd.DataFrame())


def choose_modulator_from_inner_scores(table: pd.DataFrame) -> pd.Series:
    """Choose a modulator only from inner-validation columns."""
    forbidden = [column for column in table if column.startswith("outer_") or column.startswith("test_")]
    if forbidden:
        raise ValueError(f"modulator selection table contains outer/test columns: {forbidden}")
    required = {"modulators", "mean_validation_annual_dz", "annual_interval_gap", "feature_count"}
    missing = required - set(table)
    if missing:
        raise KeyError(f"modulator selection table missing columns: {sorted(missing)}")
    work = table.copy()
    if "se_validation_dz" not in work:
        work["se_validation_dz"] = 0.0
    return select_hierarchical_candidate(work)


def _reference_control_modulators(
    control: pd.DataFrame,
    train: pd.DataFrame,
    modulators: Sequence[str],
) -> tuple[pd.DataFrame, str]:
    values = pd.DataFrame(index=control.index)
    policies = []
    for modulator in modulators:
        if modulator == "age" and modulator in control and control[modulator].notna().all():
            values[modulator] = pd.to_numeric(control[modulator], errors="coerce")
            policies.append("observed age")
        else:
            mean_value = float(pd.to_numeric(train[modulator], errors="coerce").mean())
            values[modulator] = mean_value
            policies.append(f"{modulator}=FRDA-training mean")
    return values, "; ".join(policies)


def run_adaptive_recipe_comparison(
    frda_long: pd.DataFrame,
    control_long: pd.DataFrame,
    folds: pd.DataFrame,
    recipe: pd.DataFrame,
    *,
    run_id: str,
    modulator_sets: Sequence[Sequence[str]] = DEFAULT_MODULATOR_SETS,
    inner_folds: int = 3,
    seed: int = 42,
    n_boot: int = 1000,
    config: Config | None = None,
    pair_col: str = "pair_id",
    participant_col: str = "subject",
    visit_col: str = "visit",
) -> dict[str, pd.DataFrame]:
    """Tune modulators inside outer FRDA training folds and evaluate both recipes."""
    validate_fold_manifest(folds)
    validate_feature_recipe(recipe)
    base_config = config or Config(
        random_state=seed,
        interaction_en_alpha_grid=(0.03, 0.1, 0.3, 1.0, 3.0),
        interaction_en_l1_ratio_grid=(0.0,),
        interaction_inner_cv_splits=3,
        interaction_z_clip=2.75,
        interaction_tune_inner_cv=True,
    )
    strategies = _recipe_strategies(recipe)
    candidates = _available_modulators(frda_long, modulator_sets)
    if not candidates:
        raise ValueError("no candidate modulators are available")
    outer_folds = sorted(pd.to_numeric(folds["outer_fold"], errors="coerce").dropna().astype(int).unique())
    inner_rows: list[dict[str, object]] = []
    inner_details: dict[tuple[str, int, str], pd.DataFrame] = {}

    # All candidate ranking is completed before any held-out row is scored.
    for strategy in strategies:
        for outer_fold in outer_folds:
            features = selected_features_for_fold(recipe, strategy, outer_fold)
            train_ids, _ = split_participants_for_fold(folds, cohort="FRDA", outer_fold=outer_fold)
            raw_train = _subset_participants(frda_long, train_ids, participant_col=participant_col)
            for modulators in candidates:
                train = _complete_rows(
                    raw_train,
                    [*features, *modulators],
                    pair_col=pair_col,
                    participant_col=participant_col,
                    visit_col=visit_col,
                )
                row, detail = _adaptive_inner_score(
                    train,
                    features,
                    modulators,
                    pair_col=pair_col,
                    participant_col=participant_col,
                    visit_col=visit_col,
                    inner_folds=inner_folds,
                    seed=seed + outer_fold,
                    config=replace(base_config, random_state=seed + outer_fold),
                )
                row.update({"selection_strategy": strategy, "outer_fold": outer_fold})
                inner_rows.append(row)
                inner_details[(strategy, outer_fold, ",".join(modulators))] = detail
    inner_table = pd.DataFrame(inner_rows)

    strategy_choices: dict[tuple[str, int], tuple[str, ...]] = {}
    common_choices: dict[int, tuple[str, ...]] = {}
    choice_rows: list[dict[str, object]] = []
    for outer_fold in outer_folds:
        fold_table = inner_table.loc[inner_table["outer_fold"].eq(outer_fold)]
        for strategy in strategies:
            part = fold_table.loc[fold_table["selection_strategy"].eq(strategy)].drop(
                columns=["selection_strategy", "outer_fold"]
            )
            choice = choose_modulator_from_inner_scores(part)
            selected = tuple(str(choice["modulators"]).split(","))
            strategy_choices[(strategy, outer_fold)] = selected
            choice_rows.append(
                {
                    "comparison_mode": "strategy_specific",
                    "selection_strategy": strategy,
                    "outer_fold": outer_fold,
                    **choice.to_dict(),
                }
            )
        common = (
            fold_table.groupby("modulators", as_index=False)
            .agg(
                mean_validation_annual_dz=("mean_validation_annual_dz", "mean"),
                dz_v1_v2=("dz_v1_v2", "mean"),
                dz_v2_v3=("dz_v2_v3", "mean"),
                annual_interval_gap=("annual_interval_gap", "mean"),
                p_progression=("p_progression", "mean"),
                feature_count=("feature_count", "mean"),
                modulator_count=("modulator_count", "first"),
                n_participants=("n_participants", "min"),
                n_pairs=("n_pairs", "min"),
            )
        )
        common_choice = choose_modulator_from_inner_scores(common)
        common_selected = tuple(str(common_choice["modulators"]).split(","))
        common_choices[outer_fold] = common_selected
        for strategy in strategies:
            choice_rows.append(
                {
                    "comparison_mode": "common_modulator",
                    "selection_strategy": strategy,
                    "outer_fold": outer_fold,
                    **common_choice.to_dict(),
                }
            )

    oof_parts: list[pd.DataFrame] = []
    parameter_rows: list[dict[str, object]] = []
    coefficient_rows: list[dict[str, object]] = []
    for mode in ("strategy_specific", "common_modulator"):
        for strategy in strategies:
            for outer_fold in outer_folds:
                features = selected_features_for_fold(recipe, strategy, outer_fold)
                modulators = (
                    strategy_choices[(strategy, outer_fold)]
                    if mode == "strategy_specific"
                    else common_choices[outer_fold]
                )
                frda_train_ids, frda_test_ids = split_participants_for_fold(
                    folds, cohort="FRDA", outer_fold=outer_fold
                )
                _, control_test_ids = split_participants_for_fold(
                    folds, cohort="Control", outer_fold=outer_fold
                )
                raw_train = _subset_participants(frda_long, frda_train_ids, participant_col=participant_col)
                frda_train = _complete_rows(
                    raw_train, [*features, *modulators], pair_col=pair_col,
                    participant_col=participant_col, visit_col=visit_col,
                )
                frda_test = _complete_rows(
                    _subset_participants(frda_long, frda_test_ids, participant_col=participant_col),
                    [*features, *modulators], pair_col=pair_col,
                    participant_col=participant_col, visit_col=visit_col,
                )
                control_test = _complete_rows(
                    _subset_participants(control_long, control_test_ids, participant_col=participant_col),
                    features, pair_col=pair_col, participant_col=participant_col, visit_col=visit_col,
                )
                fold_config = replace(base_config, random_state=seed + outer_fold)
                fitted = InteractionLinearComposite(config=fold_config).fit(
                    frda_train[features],
                    frda_train[list(modulators)],
                    frda_train[pair_col].to_numpy(),
                    frda_train[visit_col].to_numpy(),
                    cv_group_id=frda_train[participant_col].to_numpy(),
                )
                frda_scores = fitted.score(frda_test[features], frda_test[list(modulators)])
                control_modulators, control_policy = _reference_control_modulators(
                    control_test, frda_train, modulators
                )
                control_scores = fitted.score(control_test[features], control_modulators)
                model_name = f"patient_adaptive_{mode}"
                for cohort, frame, values in (
                    ("FRDA", frda_test, frda_scores),
                    ("Control", control_test, control_scores),
                ):
                    oof_parts.append(
                        _score_metadata(
                            frame,
                            values,
                            run_id=run_id,
                            model=model_name,
                            strategy=strategy,
                            cohort=cohort,
                            outer_fold=outer_fold,
                            feature_count=len(features),
                            modulator_recipe=",".join(modulators),
                            pair_col=pair_col,
                            participant_col=participant_col,
                            visit_col=visit_col,
                        )
                    )
                parameter_rows.append(
                    {
                        "model": model_name,
                        "selection_strategy": strategy,
                        "outer_fold": outer_fold,
                        "modulators": ",".join(modulators),
                        "control_modulator_policy": control_policy,
                        "feature_count": len(features),
                        "features": "|".join(features),
                        "train_frda_participants": frda_train[participant_col].nunique(),
                        "train_frda_pairs": frda_train[pair_col].nunique(),
                        **(fitted.best_params_ or {}),
                    }
                )
                z_profile = {modulator: float(frda_train[modulator].mean()) for modulator in modulators}
                effective = fitted.imaging_weights(z_profile)
                for feature in features:
                    coefficient_rows.append(
                        {
                            "run_id": run_id,
                            "model": model_name,
                            "selection_strategy": strategy,
                            "fit_scope": "outer_fold",
                            "outer_fold": outer_fold,
                            "feature": feature,
                            "coefficient": float(effective[feature]),
                            "training_mean": float(fitted.x_mean_[feature]),
                            "training_sd": float(fitted.x_sd_[feature]),
                            "z_clip": fold_config.interaction_z_clip,
                            "modulator_reference_profile": "; ".join(
                                f"{key}={value:.6g}" for key, value in z_profile.items()
                            ),
                            "selection_frequency": np.nan,
                        }
                    )
    oof = pd.concat(oof_parts, ignore_index=True)
    validate_oof_uniqueness(oof)
    performance, sites = summarise_oof_performance(oof, n_boot=n_boot, seed=seed)
    choices = pd.DataFrame(choice_rows)
    frequency = (
        choices.groupby(["comparison_mode", "selection_strategy", "modulators"], as_index=False)
        .size()
        .rename(columns={"size": "folds_selected"})
    )
    frequency["selection_frequency"] = frequency["folds_selected"] / len(outer_folds)
    return {
        "oof_visit_scores": oof,
        "performance": performance,
        "site_diagnostics": sites,
        "modulator_inner_scores": inner_table,
        "modulator_choices": choices,
        "modulator_frequency": frequency,
        "fold_parameters": pd.DataFrame(parameter_rows),
        "coefficients": pd.DataFrame(coefficient_rows, columns=COEFFICIENT_COLUMNS),
        "comparison": comparison_table(performance),
    }


__all__ = [
    "DEFAULT_MODULATOR_SETS",
    "choose_modulator_from_inner_scores",
    "comparison_table",
    "frda_control_contrast_table",
    "run_adaptive_recipe_comparison",
    "run_srm_recipe_comparison",
    "selected_features_for_fold",
    "summarise_oof_performance",
]
