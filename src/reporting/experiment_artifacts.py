"""Validated artifacts for cross-notebook TRACK-FA experiments.

The notebook pipeline uses these helpers to share participant folds, feature
recipes, and model results without relying on notebook execution state.  Every
table artifact has a JSON sidecar containing its schema, content hash, and the
experiment hashes it must match.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


FOLD_COLUMNS = (
    "cohort",
    "participant_id",
    "outer_fold",
    "seed",
    "n_splits",
)

FEATURE_RECIPE_COLUMNS = (
    "strategy",
    "outer_fold",
    "feature",
    "rank",
    "selected",
    "selection_score",
    "frda_pooled_d_z",
    "frda_v1_v2_d_z",
    "frda_v2_v3_d_z",
    "control_pooled_d_z",
    "control_v1_v2_d_z",
    "control_v2_v3_d_z",
    "train_frda_participants",
    "train_frda_pairs",
    "train_control_participants",
    "train_control_pairs",
)

OOF_VISIT_COLUMNS = (
    "run_id",
    "model",
    "selection_strategy",
    "cohort",
    "outer_fold",
    "participant_id",
    "pair_id",
    "visit",
    "interval",
    "score",
    "site",
    "feature_count",
    "modulator_recipe",
)

PERFORMANCE_COLUMNS = (
    "run_id",
    "model",
    "selection_strategy",
    "cohort",
    "interval",
    "n_participants",
    "n_pairs",
    "d_z",
    "mean_delta",
    "sd_delta",
    "ci_low",
    "ci_high",
    "p_delta_gt_0",
)

COEFFICIENT_COLUMNS = (
    "run_id",
    "model",
    "selection_strategy",
    "fit_scope",
    "outer_fold",
    "feature",
    "coefficient",
    "training_mean",
    "training_sd",
    "z_clip",
    "modulator_reference_profile",
    "selection_frequency",
)

SCHEMAS: dict[str, tuple[str, ...]] = {
    "folds": FOLD_COLUMNS,
    "feature_recipe": FEATURE_RECIPE_COLUMNS,
    "oof_visit_scores": OOF_VISIT_COLUMNS,
    "performance": PERFORMANCE_COLUMNS,
    "coefficients": COEFFICIENT_COLUMNS,
}


class ArtifactValidationError(ValueError):
    """Raised when an experiment artifact is incomplete or incompatible."""


def sha256_file(path: str | Path) -> str:
    """Return the SHA256 digest of a file without loading it all into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_values(values: Sequence[object]) -> str:
    """Hash an ordered sequence using a stable JSON representation."""
    payload = json.dumps([str(value) for value in values], ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sha256_frame(frame: pd.DataFrame) -> str:
    """Hash table values and column order independently of row order."""
    columns = sorted(map(str, frame.columns))
    canonical = frame.loc[:, columns].copy()
    for column in columns:
        canonical[column] = canonical[column].map(_canonical_scalar)
    if columns:
        canonical = canonical.sort_values(columns, kind="mergesort", na_position="first")
    payload = canonical.to_csv(index=False, lineterminator="\n")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _canonical_scalar(value: object) -> str:
    if pd.isna(value):
        return "<NA>"
    if isinstance(value, (float, np.floating)):
        return format(float(value), ".17g")
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (bool, np.bool_)):
        return "true" if bool(value) else "false"
    return str(value)


def _cohort_seed(seed: int, cohort: str) -> int:
    suffix = int.from_bytes(hashlib.sha256(cohort.encode("utf-8")).digest()[:4], "big")
    return (int(seed) + suffix) % (2**32)


def make_participant_folds(
    cohort_participants: Mapping[str, Sequence[object]],
    *,
    n_splits: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """Create deterministic, balanced outer-test folds for each cohort."""
    if int(n_splits) < 2:
        raise ValueError("n_splits must be at least 2")
    if not cohort_participants:
        raise ValueError("cohort_participants must not be empty")

    rows: list[dict[str, object]] = []
    global_ids: dict[str, str] = {}
    for cohort in sorted(map(str, cohort_participants)):
        participants = [str(value) for value in cohort_participants[cohort]]
        if len(participants) != len(set(participants)):
            raise ArtifactValidationError(f"duplicate participant ID in cohort {cohort!r}")
        if len(participants) < int(n_splits):
            raise ValueError(
                f"cohort {cohort!r} has {len(participants)} participants, fewer than n_splits={n_splits}"
            )
        for participant in participants:
            previous = global_ids.setdefault(participant, cohort)
            if previous != cohort:
                raise ArtifactValidationError(
                    f"participant {participant!r} appears in both {previous!r} and {cohort!r}"
                )

        rng = np.random.default_rng(_cohort_seed(seed, cohort))
        shuffled = rng.permutation(np.asarray(sorted(participants), dtype=object))
        for fold_index, fold_participants in enumerate(
            np.array_split(shuffled, int(n_splits)), start=1
        ):
            rows.extend(
                {
                    "cohort": cohort,
                    "participant_id": str(participant),
                    "outer_fold": int(fold_index),
                    "seed": int(seed),
                    "n_splits": int(n_splits),
                }
                for participant in fold_participants
            )
    result = pd.DataFrame(rows, columns=FOLD_COLUMNS)
    validate_fold_manifest(result)
    return result.sort_values(
        ["cohort", "outer_fold", "participant_id"], kind="mergesort"
    ).reset_index(drop=True)


def validate_columns(frame: pd.DataFrame, schema: str) -> None:
    """Require all columns defined by a named artifact schema."""
    if schema not in SCHEMAS:
        raise KeyError(f"unknown artifact schema: {schema!r}")
    missing = [column for column in SCHEMAS[schema] if column not in frame.columns]
    if missing:
        raise ArtifactValidationError(f"{schema} artifact missing columns: {missing}")


def validate_fold_manifest(folds: pd.DataFrame) -> None:
    validate_columns(folds, "folds")
    if folds.empty:
        raise ArtifactValidationError("fold manifest is empty")
    if folds[["cohort", "participant_id"]].isna().any().any():
        raise ArtifactValidationError("fold manifest has missing cohort or participant IDs")
    if folds.duplicated(["cohort", "participant_id"]).any():
        raise ArtifactValidationError("a participant is assigned to more than one outer fold")
    cross_cohort = folds.groupby("participant_id")["cohort"].nunique()
    if (cross_cohort > 1).any():
        offenders = cross_cohort[cross_cohort > 1].index.astype(str).tolist()[:10]
        raise ArtifactValidationError(f"participants occur in multiple cohorts: {offenders}")
    for cohort, group in folds.groupby("cohort", sort=False):
        split_values = sorted(pd.to_numeric(group["outer_fold"], errors="coerce").dropna().astype(int).unique())
        expected_n = pd.to_numeric(group["n_splits"], errors="coerce").dropna().astype(int).unique()
        if len(expected_n) != 1:
            raise ArtifactValidationError(f"cohort {cohort!r} has inconsistent n_splits")
        if split_values != list(range(1, int(expected_n[0]) + 1)):
            raise ArtifactValidationError(
                f"cohort {cohort!r} outer folds are {split_values}, expected consecutive folds"
            )


def split_participants_for_fold(
    folds: pd.DataFrame,
    *,
    cohort: str,
    outer_fold: int,
) -> tuple[set[str], set[str]]:
    """Return disjoint train and test participant IDs from a persisted manifest."""
    validate_fold_manifest(folds)
    cohort_rows = folds.loc[folds["cohort"].astype(str).eq(str(cohort))]
    if cohort_rows.empty:
        raise ArtifactValidationError(f"cohort {cohort!r} is absent from fold manifest")
    test_mask = pd.to_numeric(cohort_rows["outer_fold"], errors="coerce").eq(int(outer_fold))
    test_ids = set(cohort_rows.loc[test_mask, "participant_id"].astype(str))
    train_ids = set(cohort_rows.loc[~test_mask, "participant_id"].astype(str))
    if not test_ids:
        raise ArtifactValidationError(f"outer fold {outer_fold} is absent for cohort {cohort!r}")
    if train_ids & test_ids:
        raise ArtifactValidationError("train and test participants overlap")
    return train_ids, test_ids


def validate_feature_recipe(recipe: pd.DataFrame, *, panel: Sequence[str] | None = None) -> None:
    validate_columns(recipe, "feature_recipe")
    if recipe.empty:
        raise ArtifactValidationError("feature recipe is empty")
    if recipe[["strategy", "outer_fold", "feature", "rank", "selected"]].isna().any().any():
        raise ArtifactValidationError("feature recipe has missing identity fields")
    if recipe.duplicated(["strategy", "outer_fold", "feature"]).any():
        raise ArtifactValidationError("feature recipe contains duplicate strategy/fold/features")
    if panel is not None:
        outside = sorted(set(recipe["feature"].astype(str)) - set(map(str, panel)))
        if outside:
            raise ArtifactValidationError(f"feature recipe contains features outside the panel: {outside[:10]}")
    selected = recipe["selected"]
    if not selected.map(lambda value: isinstance(value, (bool, np.bool_))).all():
        raise ArtifactValidationError("feature recipe selected column must contain booleans")
    selected_counts = recipe.loc[selected].groupby(["strategy", "outer_fold"]).size()
    all_groups = recipe.groupby(["strategy", "outer_fold"]).size().index
    if set(selected_counts.index) != set(all_groups):
        raise ArtifactValidationError("every strategy/fold must select at least one feature")


def validate_oof_uniqueness(scores: pd.DataFrame) -> None:
    validate_columns(scores, "oof_visit_scores")
    if scores.empty:
        raise ArtifactValidationError("OOF score artifact is empty")
    identity = [
        "run_id",
        "model",
        "selection_strategy",
        "cohort",
        "participant_id",
        "pair_id",
        "visit",
    ]
    if scores[identity].isna().any().any():
        raise ArtifactValidationError("OOF score artifact has missing identity fields")
    if scores.duplicated(identity).any():
        raise ArtifactValidationError("OOF score artifact contains duplicate held-out visits")
    fold_counts = scores.groupby(
        ["run_id", "model", "selection_strategy", "cohort", "participant_id"]
    )["outer_fold"].nunique()
    if (fold_counts > 1).any():
        raise ArtifactValidationError("a participant appears in multiple OOF folds")


def validate_artifact_frame(frame: pd.DataFrame, schema: str, **kwargs: Any) -> None:
    if schema == "folds":
        validate_fold_manifest(frame)
    elif schema == "feature_recipe":
        validate_feature_recipe(frame, panel=kwargs.get("panel"))
    elif schema == "oof_visit_scores":
        validate_oof_uniqueness(frame)
    else:
        validate_columns(frame, schema)
        if frame.empty:
            raise ArtifactValidationError(f"{schema} artifact is empty")


def build_experiment_manifest(
    *,
    run_id: str,
    data_path: str | Path,
    feature_panel: Sequence[str],
    folds: pd.DataFrame,
    seed: int,
    outer_splits: int,
    inner_splits: int,
    source_notebooks: Sequence[str],
    strategy_definitions: Mapping[str, object],
    clinical_heads_enabled: bool = False,
) -> dict[str, Any]:
    """Build the immutable provenance block shared by all experiment artifacts."""
    validate_fold_manifest(folds)
    path = Path(data_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    panel = [str(feature) for feature in feature_panel]
    if not panel or len(panel) != len(set(panel)):
        raise ArtifactValidationError("feature_panel must be non-empty and unique")
    return {
        "schema_version": 1,
        "run_id": str(run_id),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "data_path": str(path),
        "data_sha256": sha256_file(path),
        "feature_panel_sha256": sha256_values(panel),
        "feature_panel_count": len(panel),
        "fold_sha256": sha256_frame(folds),
        "seed": int(seed),
        "outer_splits": int(outer_splits),
        "inner_splits": int(inner_splits),
        "source_notebooks": [str(value) for value in source_notebooks],
        "strategy_definitions": dict(strategy_definitions),
        "clinical_heads_enabled": bool(clinical_heads_enabled),
    }


def validate_manifest(manifest: Mapping[str, Any], *, folds: pd.DataFrame | None = None) -> None:
    required = {
        "schema_version",
        "run_id",
        "data_path",
        "data_sha256",
        "feature_panel_sha256",
        "fold_sha256",
        "seed",
        "outer_splits",
        "inner_splits",
        "strategy_definitions",
        "clinical_heads_enabled",
    }
    missing = sorted(required - set(manifest))
    if missing:
        raise ArtifactValidationError(f"experiment manifest missing fields: {missing}")
    if int(manifest["schema_version"]) != 1:
        raise ArtifactValidationError(f"unsupported manifest schema: {manifest['schema_version']!r}")
    if folds is not None:
        validate_fold_manifest(folds)
        actual = sha256_frame(folds)
        if actual != manifest["fold_sha256"]:
            raise ArtifactValidationError(
                f"fold hash mismatch: expected {manifest['fold_sha256']}, found {actual}"
            )


def write_experiment_contract(
    output_dir: str | Path,
    *,
    manifest: Mapping[str, Any],
    folds: pd.DataFrame,
) -> dict[str, Path]:
    """Persist and validate the top-level manifest and participant folds."""
    validate_manifest(manifest, folds=folds)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    folds_path = output / "folds.csv"
    manifest_path.write_text(json.dumps(dict(manifest), indent=2, sort_keys=True) + "\n")
    folds.to_csv(folds_path, index=False)
    return {"manifest": manifest_path, "folds": folds_path}


def read_experiment_contract(
    output_dir: str | Path,
    *,
    verify_data_file: bool = True,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Load a contract and fail loudly if its folds or source data changed."""
    output = Path(output_dir)
    manifest = json.loads((output / "manifest.json").read_text())
    folds = pd.read_csv(output / "folds.csv")
    validate_manifest(manifest, folds=folds)
    if verify_data_file:
        data_path = Path(manifest["data_path"])
        if not data_path.is_file():
            raise ArtifactValidationError(f"experiment data file is missing: {data_path}")
        actual = sha256_file(data_path)
        if actual != manifest["data_sha256"]:
            raise ArtifactValidationError(
                f"data hash mismatch: expected {manifest['data_sha256']}, found {actual}"
            )
    return manifest, folds


def write_table_artifact(
    path: str | Path,
    frame: pd.DataFrame,
    *,
    schema: str,
    manifest: Mapping[str, Any],
    metadata: Mapping[str, Any] | None = None,
    **validation_kwargs: Any,
) -> tuple[Path, Path]:
    """Write a validated CSV and provenance sidecar."""
    validate_manifest(manifest)
    validate_artifact_frame(frame, schema, **validation_kwargs)
    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_path, index=False)
    sidecar_path = csv_path.with_suffix(csv_path.suffix + ".meta.json")
    payload = {
        "schema": schema,
        "schema_version": 1,
        "run_id": manifest["run_id"],
        "data_sha256": manifest["data_sha256"],
        "feature_panel_sha256": manifest["feature_panel_sha256"],
        "fold_sha256": manifest["fold_sha256"],
        # Hash the serialized artifact, not the in-memory frame. CSV parsing can
        # round the final binary digit of a float while preserving its value.
        "table_sha256": sha256_file(csv_path),
        "metadata": dict(metadata or {}),
    }
    sidecar_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return csv_path, sidecar_path


def read_table_artifact(
    path: str | Path,
    *,
    schema: str,
    manifest: Mapping[str, Any],
    **validation_kwargs: Any,
) -> pd.DataFrame:
    """Read a CSV only when its schema, hashes, and experiment identity match."""
    validate_manifest(manifest)
    csv_path = Path(path)
    sidecar_path = csv_path.with_suffix(csv_path.suffix + ".meta.json")
    frame = pd.read_csv(csv_path)
    sidecar = json.loads(sidecar_path.read_text())
    if sidecar.get("schema") != schema:
        raise ArtifactValidationError(
            f"artifact schema mismatch: expected {schema!r}, found {sidecar.get('schema')!r}"
        )
    for field in ("run_id", "data_sha256", "feature_panel_sha256", "fold_sha256"):
        if sidecar.get(field) != manifest.get(field):
            raise ArtifactValidationError(
                f"artifact {field} mismatch: expected {manifest.get(field)!r}, found {sidecar.get(field)!r}"
            )
    actual = sha256_file(csv_path)
    if actual != sidecar.get("table_sha256"):
        raise ArtifactValidationError(
            f"table hash mismatch: expected {sidecar.get('table_sha256')}, found {actual}"
        )
    validate_artifact_frame(frame, schema, **validation_kwargs)
    return frame


__all__ = [
    "ArtifactValidationError",
    "COEFFICIENT_COLUMNS",
    "FEATURE_RECIPE_COLUMNS",
    "FOLD_COLUMNS",
    "OOF_VISIT_COLUMNS",
    "PERFORMANCE_COLUMNS",
    "SCHEMAS",
    "build_experiment_manifest",
    "make_participant_folds",
    "read_experiment_contract",
    "read_table_artifact",
    "sha256_file",
    "sha256_frame",
    "sha256_values",
    "split_participants_for_fold",
    "validate_artifact_frame",
    "validate_feature_recipe",
    "validate_fold_manifest",
    "validate_manifest",
    "validate_oof_uniqueness",
    "write_experiment_contract",
    "write_table_artifact",
]
