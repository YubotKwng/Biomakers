"""Leakage-safe, dual-cohort feature selection for longitudinal SRM models.

Controls may inform feature ranking and candidate choice, but never the
standardisation statistics or SRM coefficients.  Both cohorts are evaluated
with participant-grouped nested cross-validation so every reported score is
from a participant that was absent from ranking, tuning, scaling, and fitting.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, Mapping, Sequence

import numpy as np
import pandas as pd

from ..data.qc import standardize_train_test
from ..features.selection import progression_univariate_effects
from ..models.srm_global import SRMGlobalLinear
from .cv import group_kfold_indices


SelectorName = Literal["full_70", "frda_progression", "control_aware"]


@dataclass(frozen=True)
class DualCohortCandidate:
    """One SRM feature-selection and regularisation candidate."""

    label: str
    selector: SelectorName
    k: int
    control_penalty: float = 0.0
    ridge: float = 0.0
    covariance_shrinkage: float = 0.45
    z_clip: float | None = None

    def __post_init__(self) -> None:
        if self.selector not in {"full_70", "frda_progression", "control_aware"}:
            raise ValueError(f"Unknown selector: {self.selector!r}")
        if int(self.k) <= 0:
            raise ValueError("k must be positive")
        if float(self.control_penalty) < 0:
            raise ValueError("control_penalty must be non-negative")
        if not 0.0 <= float(self.covariance_shrinkage) <= 1.0:
            raise ValueError("covariance_shrinkage must be between 0 and 1")
        if self.z_clip is not None and float(self.z_clip) <= 0:
            raise ValueError("z_clip must be positive when supplied")


def _normalise_subject(value: object) -> str:
    subject = str(value)
    return subject[len("TRACKFA_") :] if subject.upper().startswith("TRACKFA_") else subject


def _visit_feature_column(columns: pd.Index, feature: str, visit: int) -> str | None:
    candidates = (
        f"{feature}_v{visit}",
        f"{feature}_V{visit}",
        f"{feature}_visit{visit}",
        f"{feature}_{visit}",
    )
    return next((name for name in candidates if name in columns), None)


def wide_cohort_to_pair_long(
    wide_df: pd.DataFrame,
    feature_cols: Sequence[str],
    *,
    cohort_value: object,
    id_col: str = "ID",
    cohort_col: str = "study_group",
    site_col: str = "site",
) -> pd.DataFrame:
    """Convert visit-suffixed wide rows into local annual-pair long rows.

    Visit columns are expected in the TRACK-FA form ``feature_v1`` through
    ``feature_v3``.  The returned ``visit`` is local to an annual pair: both
    V1V2 and V2V3 are encoded as visits 1 and 2, while ``source_visit`` retains
    the original visit number.
    """
    features = list(dict.fromkeys(feature_cols))
    missing_base = [c for c in (id_col, cohort_col) if c not in wide_df.columns]
    if missing_base:
        raise KeyError(f"wide_df missing required columns: {missing_base}")
    visit_columns = {
        (feature, visit): _visit_feature_column(wide_df.columns, feature, visit)
        for feature in features
        for visit in (1, 2, 3)
    }
    absent = [f"{feature}_v{visit}" for (feature, visit), col in visit_columns.items() if col is None]
    if absent:
        raise KeyError(f"wide_df missing visit feature columns: {absent[:10]}")

    cohort_numeric = pd.to_numeric(wide_df[cohort_col], errors="coerce")
    target_numeric = pd.to_numeric(pd.Series([cohort_value]), errors="coerce").iloc[0]
    if pd.notna(target_numeric):
        cohort_mask = cohort_numeric.eq(target_numeric)
    else:
        cohort_mask = wide_df[cohort_col].astype(str).eq(str(cohort_value))

    rows: list[dict] = []
    for _, source in wide_df.loc[cohort_mask].iterrows():
        subject = _normalise_subject(source[id_col])
        site = source.get(site_col, np.nan)
        for start, end, interval in ((1, 2, "V1->V2"), (2, 3, "V2->V3")):
            pair_id = f"{subject}_{interval.replace('->', '')}"
            for local_visit, source_visit in ((1, start), (2, end)):
                row = {
                    "pair_id": pair_id,
                    "subject": subject,
                    "visit": local_visit,
                    "source_visit": source_visit,
                    "interval": interval,
                    "site": site,
                    "cohort": cohort_value,
                    cohort_col: source[cohort_col],
                }
                for feature in features:
                    row[feature] = source[visit_columns[(feature, source_visit)]]
                rows.append(row)
    return pd.DataFrame(rows)


def _cohens_dz(values: Sequence[float] | pd.Series) -> float:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
    if arr.size < 2:
        return np.nan
    sd = float(np.std(arr, ddof=1))
    if not np.isfinite(sd) or sd == 0.0:
        return np.nan
    return float(np.mean(arr) / sd)


def _interval_from_pair_id(values: pd.Series) -> pd.Series:
    return (
        values.astype(str)
        .str.upper()
        .str.extract(r"(V\d+V\d+)", expand=False)
        .map({"V1V2": "V1->V2", "V2V3": "V2->V3"})
    )


def _pair_delta_table(
    pair_long: pd.DataFrame,
    value_col: str,
    *,
    pair_col: str = "pair_id",
    participant_col: str = "subject",
    visit_col: str = "visit",
) -> pd.DataFrame:
    required = {pair_col, participant_col, visit_col, value_col}
    missing = required - set(pair_long.columns)
    if missing:
        raise KeyError(f"pair_long missing required columns: {sorted(missing)}")
    work = pair_long[[pair_col, participant_col, visit_col, value_col]].copy()
    work[visit_col] = pd.to_numeric(work[visit_col], errors="coerce")
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    work = work.dropna(subset=[pair_col, participant_col, visit_col, value_col])
    wide = work.pivot_table(
        index=[pair_col, participant_col],
        columns=visit_col,
        values=value_col,
        aggfunc="mean",
    )
    if 1 not in wide.columns or 2 not in wide.columns:
        return pd.DataFrame(columns=[pair_col, participant_col, "interval", "delta"])
    out = wide[[1, 2]].dropna().reset_index()
    out["delta"] = out[2] - out[1]
    out["interval"] = _interval_from_pair_id(out[pair_col]).fillna("annual")
    return out[[pair_col, participant_col, "interval", "delta"]]


def _effect_fields(deltas: pd.DataFrame, prefix: str) -> dict[str, float | int]:
    values = deltas["delta"] if not deltas.empty else pd.Series(dtype=float)
    return {
        f"{prefix}_n_pairs": int(len(deltas)),
        f"{prefix}_n_participants": int(deltas["subject"].nunique()) if "subject" in deltas else 0,
        f"{prefix}_d_z": _cohens_dz(values),
    }


def single_feature_effect_table(
    pair_long: pd.DataFrame,
    features: Sequence[str],
    *,
    cohort: str,
    pair_col: str = "pair_id",
    participant_col: str = "subject",
    visit_col: str = "visit",
) -> pd.DataFrame:
    """Return marginal pooled and interval-specific paired effects by feature."""
    rows: list[dict] = []
    for feature in [f for f in dict.fromkeys(features) if f in pair_long.columns]:
        deltas = _pair_delta_table(
            pair_long,
            feature,
            pair_col=pair_col,
            participant_col=participant_col,
            visit_col=visit_col,
        ).rename(columns={participant_col: "subject"})
        row: dict = {"cohort": str(cohort), "feature": feature}
        row.update(_effect_fields(deltas, "pooled"))
        for interval, prefix in (("V1->V2", "v1_v2"), ("V2->V3", "v2_v3")):
            row.update(_effect_fields(deltas.loc[deltas["interval"].eq(interval)], prefix))
        d12 = row["v1_v2_d_z"]
        d23 = row["v2_v3_d_z"]
        row["interval_gap"] = (
            float(abs(d12 - d23)) if np.isfinite(d12) and np.isfinite(d23) else np.nan
        )
        row["intervals_same_direction"] = bool(
            np.isfinite(d12) and np.isfinite(d23) and np.sign(d12) == np.sign(d23)
        )
        rows.append(row)
    return pd.DataFrame(rows)


def control_aware_feature_ranking(
    frda_train: pd.DataFrame,
    control_train: pd.DataFrame,
    features: Sequence[str],
    *,
    control_penalty: float,
    pair_col: str = "pair_id",
    participant_col: str = "subject",
    visit_col: str = "visit",
) -> pd.DataFrame:
    """Rank strong, interval-consistent FRDA changes with a control penalty."""
    if float(control_penalty) < 0:
        raise ValueError("control_penalty must be non-negative")
    feature_list = [
        f for f in dict.fromkeys(features) if f in frda_train.columns and f in control_train.columns
    ]
    frda = single_feature_effect_table(
        frda_train,
        feature_list,
        cohort="FRDA",
        pair_col=pair_col,
        participant_col=participant_col,
        visit_col=visit_col,
    ).set_index("feature")
    control = single_feature_effect_table(
        control_train,
        feature_list,
        cohort="Control",
        pair_col=pair_col,
        participant_col=participant_col,
        visit_col=visit_col,
    ).set_index("feature")

    rows: list[dict] = []
    for feature in feature_list:
        d_pool = float(frda.at[feature, "pooled_d_z"])
        d12 = float(frda.at[feature, "v1_v2_d_z"])
        d23 = float(frda.at[feature, "v2_v3_d_z"])
        d_control = float(control.at[feature, "pooled_d_z"])
        direction = float(np.sign(d_pool)) if np.isfinite(d_pool) else np.nan
        oriented_12 = direction * d12 if np.isfinite(direction) and np.isfinite(d12) else np.nan
        oriented_23 = direction * d23 if np.isfinite(direction) and np.isfinite(d23) else np.nan
        consistency = (
            float(min(oriented_12, oriented_23))
            if np.isfinite(oriented_12) and np.isfinite(oriented_23)
            else np.nan
        )
        same_direction = (
            float(max(0.0, direction * d_control))
            if np.isfinite(direction) and np.isfinite(d_control)
            else np.nan
        )
        eligible = bool(
            np.isfinite(direction)
            and direction != 0
            and np.isfinite(consistency)
            and consistency > 0
            and np.isfinite(same_direction)
        )
        score = consistency - float(control_penalty) * same_direction if eligible else -np.inf
        rows.append(
            {
                "feature": feature,
                "frda_pooled_d_z": d_pool,
                "frda_v1_v2_d_z": d12,
                "frda_v2_v3_d_z": d23,
                "frda_direction": direction,
                "frda_consistency": consistency,
                "control_pooled_d_z": d_control,
                "control_v1_v2_d_z": float(control.at[feature, "v1_v2_d_z"]),
                "control_v2_v3_d_z": float(control.at[feature, "v2_v3_d_z"]),
                "same_direction_control_penalty": same_direction,
                "control_aware_score": float(score),
                "eligible": eligible,
                "frda_pooled_n": int(frda.at[feature, "pooled_n_pairs"]),
                "control_pooled_n": int(control.at[feature, "pooled_n_pairs"]),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["_abs_control"] = out["control_pooled_d_z"].abs()
    out = out.sort_values(
        [
            "control_aware_score",
            "frda_consistency",
            "same_direction_control_penalty",
            "_abs_control",
            "feature",
        ],
        ascending=[False, False, True, True, True],
        kind="mergesort",
    ).drop(columns="_abs_control").reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out


def _complete_pair_frame(
    frame: pd.DataFrame,
    feature_cols: Sequence[str],
    *,
    pair_col: str,
    participant_col: str,
    visit_col: str,
) -> pd.DataFrame:
    features = list(dict.fromkeys(feature_cols))
    required = {pair_col, participant_col, visit_col, *features}
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"frame missing required columns: {sorted(missing)}")
    optional = [c for c in ("interval", "site") if c in frame.columns]
    work = frame[[pair_col, participant_col, visit_col, *optional, *features]].copy()
    work[visit_col] = pd.to_numeric(work[visit_col], errors="coerce")
    for feature in features:
        work[feature] = pd.to_numeric(work[feature], errors="coerce")
    work = work.dropna(subset=[pair_col, participant_col, visit_col, *features])
    work = work[work[visit_col].isin([1, 2])].copy()
    complete = work.groupby(pair_col)[visit_col].agg(
        lambda values: {1, 2}.issubset(set(values.astype(int)))
    )
    work = work[work[pair_col].isin(complete.index[complete])].copy()
    if "interval" not in work:
        work["interval"] = _interval_from_pair_id(work[pair_col]).fillna("annual")
    return work.reset_index(drop=True)


def _candidate_features(
    candidate: DualCohortCandidate,
    frda_train: pd.DataFrame,
    control_train: pd.DataFrame,
    feature_cols: Sequence[str],
    *,
    pair_col: str,
    participant_col: str,
    visit_col: str,
) -> tuple[list[str], pd.DataFrame]:
    features = list(feature_cols)
    if candidate.selector == "full_70":
        ranking = pd.DataFrame({"rank": np.arange(1, len(features) + 1), "feature": features})
        selected = features
    elif candidate.selector == "frda_progression":
        ranking = progression_univariate_effects(
            frda_train,
            features,
            subject_col=pair_col,
            visit_col=visit_col,
        ).rename(columns={"abs_mean_annual_dz": "selection_score"})
        selected = ranking.head(min(int(candidate.k), len(ranking)))["feature"].tolist()
    else:
        ranking = control_aware_feature_ranking(
            frda_train,
            control_train,
            features,
            control_penalty=float(candidate.control_penalty),
            pair_col=pair_col,
            participant_col=participant_col,
            visit_col=visit_col,
        )
        eligible = ranking.loc[ranking["eligible"]]
        selected = eligible.head(min(int(candidate.k), len(eligible)))["feature"].tolist()
    ranking = ranking.copy()
    ranking["selected"] = ranking["feature"].isin(selected)
    return selected, ranking


def _fit_and_score(
    frda_train: pd.DataFrame,
    frames_to_score: Mapping[str, pd.DataFrame],
    features: Sequence[str],
    candidate: DualCohortCandidate,
    *,
    pair_col: str,
    visit_col: str,
) -> tuple[dict[str, pd.DataFrame], dict]:
    selected = list(features)
    if not selected:
        raise ValueError("candidate selected no eligible features")
    x_train = frda_train[selected].to_numpy(dtype=float)
    score_names = list(frames_to_score)
    score_frames = [frames_to_score[name] for name in score_names]
    lengths = [len(frame) for frame in score_frames]
    x_score = np.vstack([frame[selected].to_numpy(dtype=float) for frame in score_frames])
    x_train_s, x_score_s, center, scale = standardize_train_test(x_train, x_score)
    if candidate.z_clip is not None:
        clip = float(candidate.z_clip)
        x_train_s = np.clip(x_train_s, -clip, clip)
        x_score_s = np.clip(x_score_s, -clip, clip)
    model = SRMGlobalLinear(
        ridge=float(candidate.ridge),
        covariance_shrinkage=float(candidate.covariance_shrinkage),
    ).fit(
        x_train_s,
        frda_train[pair_col].to_numpy(),
        frda_train[visit_col].to_numpy(),
    )
    scored: dict[str, pd.DataFrame] = {}
    start = 0
    for name, frame, length in zip(score_names, score_frames, lengths):
        part = frame.copy()
        part["score"] = model.score(x_score_s[start : start + length])
        scored[name] = part
        start += length
    return scored, {
        "model": model,
        "center": np.asarray(center, dtype=float),
        "scale": np.asarray(scale, dtype=float),
        "coef": np.asarray(model.coef_, dtype=float),
        "features": selected,
    }


def _summary_values(scores: pd.DataFrame, *, pair_col: str, participant_col: str) -> dict:
    deltas = _pair_delta_table(scores, "score", pair_col=pair_col, participant_col=participant_col)
    out = {
        "pooled_d_z": _cohens_dz(deltas["delta"]),
        "pooled_n": int(len(deltas)),
        "pooled_n_participants": int(deltas[participant_col].nunique()) if not deltas.empty else 0,
        "p_delta_gt_0": float((deltas["delta"] > 0).mean()) if not deltas.empty else np.nan,
    }
    for interval, prefix in (("V1->V2", "v1_v2"), ("V2->V3", "v2_v3")):
        values = deltas.loc[deltas["interval"].eq(interval), "delta"]
        out[f"{prefix}_d_z"] = _cohens_dz(values)
        out[f"{prefix}_n"] = int(len(values))
    d12, d23 = out["v1_v2_d_z"], out["v2_v3_d_z"]
    out["annual_interval_gap"] = (
        float(abs(d12 - d23)) if np.isfinite(d12) and np.isfinite(d23) else np.nan
    )
    return out


def _cluster_bootstrap_ci(
    scores: pd.DataFrame,
    *,
    pair_col: str,
    participant_col: str,
    n_boot: int,
    seed: int,
) -> tuple[float, float]:
    deltas = _pair_delta_table(scores, "score", pair_col=pair_col, participant_col=participant_col)
    participants = np.asarray(sorted(deltas[participant_col].astype(str).unique()))
    if len(participants) < 2 or int(n_boot) <= 0:
        return np.nan, np.nan
    grouped = {
        participant: deltas.loc[deltas[participant_col].astype(str).eq(participant), "delta"].to_numpy()
        for participant in participants
    }
    rng = np.random.default_rng(seed)
    estimates: list[float] = []
    for _ in range(int(n_boot)):
        sample = rng.choice(participants, size=len(participants), replace=True)
        values = np.concatenate([grouped[participant] for participant in sample])
        estimate = _cohens_dz(values)
        if np.isfinite(estimate):
            estimates.append(float(estimate))
    if len(estimates) < 2:
        return np.nan, np.nan
    low, high = np.percentile(estimates, [2.5, 97.5])
    return float(low), float(high)


def _candidate_objective(
    frda_scores: pd.DataFrame,
    control_scores: pd.DataFrame,
    *,
    pair_col: str,
    participant_col: str,
    use_control: bool,
) -> tuple[float, dict, dict]:
    frda = _summary_values(frda_scores, pair_col=pair_col, participant_col=participant_col)
    control = _summary_values(control_scores, pair_col=pair_col, participant_col=participant_col)
    interval_values = [frda["v1_v2_d_z"], frda["v2_v3_d_z"]]
    if all(np.isfinite(value) for value in interval_values):
        frda_consistency = float(min(interval_values))
    else:
        frda_consistency = float(frda["pooled_d_z"])
    control_same_direction = 0.0
    if use_control:
        control_same_direction = (
            float(max(0.0, control["pooled_d_z"]))
            if np.isfinite(control["pooled_d_z"])
            else np.inf
        )
    objective = frda_consistency - control_same_direction
    if not np.isfinite(frda_consistency) or frda_consistency <= 0:
        objective = -np.inf
    return float(objective), frda, control


def _split_frame(frame: pd.DataFrame, participant_col: str, n_splits: int, seed: int):
    groups = frame[participant_col].to_numpy()
    unique = pd.unique(groups)
    if len(unique) < 2:
        raise ValueError("at least two participants per cohort are required")
    return list(group_kfold_indices(groups, n_splits=min(int(n_splits), len(unique)), seed=seed))


def _participant_string(frame: pd.DataFrame, participant_col: str) -> str:
    return "|".join(sorted(frame[participant_col].astype(str).unique()))


def _decorate_scores(
    scores: pd.DataFrame,
    *,
    strategy: str,
    candidate: DualCohortCandidate,
    cohort: str,
    outer_fold: int,
    pair_col: str,
) -> pd.DataFrame:
    keep = [c for c in (pair_col, "subject", "visit", "interval", "site", "score") if c in scores]
    out = scores[keep].copy()
    out.insert(0, "strategy", strategy)
    out.insert(1, "candidate_label", candidate.label)
    out.insert(2, "cohort", cohort)
    out.insert(3, "outer_fold", int(outer_fold))
    return out


def _fold_stability_table(fold_models: list[dict], strategy: str) -> pd.DataFrame:
    if not fold_models:
        return pd.DataFrame()
    n_folds = len(fold_models)
    all_features = sorted({feature for fit in fold_models for feature in fit["features"]})
    rows = []
    for feature in all_features:
        coefficients = [
            float(fit["coef"][fit["features"].index(feature)])
            for fit in fold_models
            if feature in fit["features"]
        ]
        signs = np.sign(coefficients)
        sign_frequency = (
            float(max(np.mean(signs > 0), np.mean(signs < 0))) if coefficients else np.nan
        )
        rows.append(
            {
                "strategy": strategy,
                "feature": feature,
                "folds_selected": int(len(coefficients)),
                "selection_frequency": float(len(coefficients) / n_folds),
                "coefficient_sign_frequency": sign_frequency,
                "coefficient_median": float(np.median(coefficients)),
                "coefficient_iqr": float(np.percentile(coefficients, 75) - np.percentile(coefficients, 25)),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["selection_frequency", "feature"], ascending=[False, True], kind="mergesort"
    ).reset_index(drop=True)


def evaluate_dual_cohort_strategy(
    frda_long: pd.DataFrame,
    control_long: pd.DataFrame,
    feature_cols: Sequence[str],
    candidates: Sequence[DualCohortCandidate],
    *,
    outer_folds: int = 5,
    inner_folds: int = 5,
    seed: int = 42,
    n_boot: int = 1000,
    strategy_label: str = "dual_cohort",
    pair_col: str = "pair_id",
    participant_col: str = "subject",
    visit_col: str = "visit",
) -> dict[str, pd.DataFrame]:
    """Nested participant-grouped evaluation of one dual-cohort strategy grid."""
    candidate_list = list(candidates)
    if not candidate_list:
        raise ValueError("candidates must not be empty")
    features = list(dict.fromkeys(feature_cols))
    if not features:
        raise ValueError("feature_cols must not be empty")
    frda = _complete_pair_frame(
        frda_long, features, pair_col=pair_col, participant_col=participant_col, visit_col=visit_col
    )
    control = _complete_pair_frame(
        control_long, features, pair_col=pair_col, participant_col=participant_col, visit_col=visit_col
    )
    frda_splits = _split_frame(frda, participant_col, outer_folds, seed)
    control_splits = _split_frame(control, participant_col, outer_folds, seed + 10_000)
    n_outer = min(len(frda_splits), len(control_splits))

    oof_parts: list[pd.DataFrame] = []
    choice_rows: list[dict] = []
    ranking_parts: list[pd.DataFrame] = []
    fold_models: list[dict] = []

    for outer_index in range(n_outer):
        outer_fold = outer_index + 1
        f_train_idx, f_test_idx = frda_splits[outer_index]
        c_train_idx, c_test_idx = control_splits[outer_index]
        f_train, f_test = frda.iloc[f_train_idx].copy(), frda.iloc[f_test_idx].copy()
        c_train, c_test = control.iloc[c_train_idx].copy(), control.iloc[c_test_idx].copy()

        for train_part, test_part, name in (
            (f_train, f_test, "FRDA"),
            (c_train, c_test, "Control"),
        ):
            overlap = set(train_part[participant_col]) & set(test_part[participant_col])
            if overlap:
                raise AssertionError(f"{name} outer fold {outer_fold} participant overlap: {overlap}")

        f_inner_splits = _split_frame(f_train, participant_col, inner_folds, seed + outer_fold)
        c_inner_splits = _split_frame(c_train, participant_col, inner_folds, seed + 20_000 + outer_fold)
        n_inner = min(len(f_inner_splits), len(c_inner_splits))
        candidate_rows: list[dict] = []
        for candidate in candidate_list:
            f_inner_oof: list[pd.DataFrame] = []
            c_inner_oof: list[pd.DataFrame] = []
            selected_counts: list[int] = []
            valid = True
            for inner_index in range(n_inner):
                fit_idx, fval_idx = f_inner_splits[inner_index]
                cit_idx, cval_idx = c_inner_splits[inner_index]
                fit, fval = f_train.iloc[fit_idx], f_train.iloc[fval_idx]
                cit, cval = c_train.iloc[cit_idx], c_train.iloc[cval_idx]
                selected, _ = _candidate_features(
                    candidate,
                    fit,
                    cit,
                    features,
                    pair_col=pair_col,
                    participant_col=participant_col,
                    visit_col=visit_col,
                )
                if not selected:
                    valid = False
                    break
                selected_counts.append(len(selected))
                try:
                    scored, _ = _fit_and_score(
                        fit,
                        {"frda": fval, "control": cval},
                        selected,
                        candidate,
                        pair_col=pair_col,
                        visit_col=visit_col,
                    )
                except (ValueError, np.linalg.LinAlgError):
                    valid = False
                    break
                f_inner_oof.append(scored["frda"])
                c_inner_oof.append(scored["control"])
            if valid and f_inner_oof and c_inner_oof:
                objective, f_metrics, c_metrics = _candidate_objective(
                    pd.concat(f_inner_oof, ignore_index=True),
                    pd.concat(c_inner_oof, ignore_index=True),
                    pair_col=pair_col,
                    participant_col=participant_col,
                    use_control=candidate.selector == "control_aware",
                )
            else:
                objective, f_metrics, c_metrics = -np.inf, {}, {}
            candidate_rows.append(
                {
                    "candidate": candidate,
                    "inner_objective": float(objective),
                    "inner_frda_pooled_d_z": f_metrics.get("pooled_d_z", np.nan),
                    "inner_frda_v1_v2_d_z": f_metrics.get("v1_v2_d_z", np.nan),
                    "inner_frda_v2_v3_d_z": f_metrics.get("v2_v3_d_z", np.nan),
                    "inner_control_pooled_d_z": c_metrics.get("pooled_d_z", np.nan),
                    "control_tuning_tiebreak": (
                        c_metrics.get("pooled_d_z", np.nan)
                        if candidate.selector == "control_aware"
                        else 0.0
                    ),
                    "median_selected_features": (
                        float(np.median(selected_counts)) if selected_counts else np.inf
                    ),
                }
            )
        candidate_table = pd.DataFrame(candidate_rows)
        finite = candidate_table[np.isfinite(candidate_table["inner_objective"])]
        if finite.empty:
            raise RuntimeError(f"no valid candidate in outer fold {outer_fold}")
        finite = finite.assign(_label=finite["candidate"].map(lambda item: item.label))
        best_row = finite.sort_values(
            ["inner_objective", "median_selected_features", "control_tuning_tiebreak", "_label"],
            ascending=[False, True, True, True],
            kind="mergesort",
        ).iloc[0]
        best: DualCohortCandidate = best_row["candidate"]
        selected, ranking = _candidate_features(
            best,
            f_train,
            c_train,
            features,
            pair_col=pair_col,
            participant_col=participant_col,
            visit_col=visit_col,
        )
        scored, fitted = _fit_and_score(
            f_train,
            {"frda": f_test, "control": c_test},
            selected,
            best,
            pair_col=pair_col,
            visit_col=visit_col,
        )
        fitted["outer_fold"] = outer_fold
        fold_models.append(fitted)
        oof_parts.extend(
            [
                _decorate_scores(
                    scored["frda"], strategy=strategy_label, candidate=best, cohort="FRDA",
                    outer_fold=outer_fold, pair_col=pair_col,
                ),
                _decorate_scores(
                    scored["control"], strategy=strategy_label, candidate=best, cohort="Control",
                    outer_fold=outer_fold, pair_col=pair_col,
                ),
            ]
        )
        ranking = ranking.copy()
        ranking.insert(0, "strategy", strategy_label)
        ranking.insert(1, "outer_fold", outer_fold)
        ranking.insert(2, "candidate_label", best.label)
        ranking_parts.append(ranking)
        choice_rows.append(
            {
                "strategy": strategy_label,
                "outer_fold": outer_fold,
                **asdict(best),
                "inner_objective": float(best_row["inner_objective"]),
                "inner_frda_pooled_d_z": best_row["inner_frda_pooled_d_z"],
                "inner_frda_v1_v2_d_z": best_row["inner_frda_v1_v2_d_z"],
                "inner_frda_v2_v3_d_z": best_row["inner_frda_v2_v3_d_z"],
                "inner_control_pooled_d_z": best_row["inner_control_pooled_d_z"],
                "n_features": len(selected),
                "selected_features": "|".join(selected),
                "frda_train_subjects": _participant_string(f_train, participant_col),
                "frda_test_subjects": _participant_string(f_test, participant_col),
                "control_train_subjects": _participant_string(c_train, participant_col),
                "control_test_subjects": _participant_string(c_test, participant_col),
                "scaler_means": "|".join(format(value, ".17g") for value in fitted["center"]),
                "scaler_sds": "|".join(format(value, ".17g") for value in fitted["scale"]),
            }
        )

    oof = pd.concat(oof_parts, ignore_index=True)
    performance_rows = []
    for cohort, cohort_scores in oof.groupby("cohort", sort=False):
        summary = _summary_values(cohort_scores, pair_col=pair_col, participant_col=participant_col)
        ci_low, ci_high = _cluster_bootstrap_ci(
            cohort_scores,
            pair_col=pair_col,
            participant_col=participant_col,
            n_boot=n_boot,
            seed=seed + (0 if cohort == "FRDA" else 1),
        )
        performance_rows.append(
            {
                "strategy": strategy_label,
                "cohort": cohort,
                **summary,
                "pooled_ci_low": ci_low,
                "pooled_ci_high": ci_high,
            }
        )
    return {
        "performance": pd.DataFrame(performance_rows),
        "oof_scores": oof,
        "outer_fold_choices": pd.DataFrame(choice_rows),
        "fold_stability": _fold_stability_table(fold_models, strategy_label),
        "feature_rankings": pd.concat(ranking_parts, ignore_index=True),
    }


def run_control_aware_comparison(
    frda_long: pd.DataFrame,
    control_long: pd.DataFrame,
    feature_cols: Sequence[str],
    *,
    strategy_grids: Mapping[str, Sequence[DualCohortCandidate]],
    outer_folds: int = 5,
    inner_folds: int = 5,
    seed: int = 42,
    n_boot: int = 1000,
    pair_col: str = "pair_id",
    participant_col: str = "subject",
    visit_col: str = "visit",
) -> dict[str, pd.DataFrame]:
    """Evaluate full-panel, FRDA-only, and control-aware grids separately."""
    combined: dict[str, list[pd.DataFrame]] = {}
    for strategy, candidates in strategy_grids.items():
        result = evaluate_dual_cohort_strategy(
            frda_long,
            control_long,
            feature_cols,
            candidates,
            outer_folds=outer_folds,
            inner_folds=inner_folds,
            # Identical outer/inner participant assignments make strategy
            # comparisons paired. Only the control-aware selector may use
            # control validation outcomes when choosing a candidate.
            seed=seed,
            n_boot=n_boot,
            strategy_label=strategy,
            pair_col=pair_col,
            participant_col=participant_col,
            visit_col=visit_col,
        )
        for name, table in result.items():
            combined.setdefault(name, []).append(table)
    return {
        name: pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
        for name, tables in combined.items()
    }


def fit_final_interpretation_model(
    frda_long: pd.DataFrame,
    control_long: pd.DataFrame,
    feature_cols: Sequence[str],
    candidate: DualCohortCandidate,
    *,
    selection_frequency: pd.DataFrame | Mapping[str, float] | None = None,
    pair_col: str = "pair_id",
    participant_col: str = "subject",
    visit_col: str = "visit",
) -> dict[str, pd.DataFrame]:
    """Fit one locked full-data model and report deployable scoring parameters."""
    features = list(dict.fromkeys(feature_cols))
    frda = _complete_pair_frame(
        frda_long, features, pair_col=pair_col, participant_col=participant_col, visit_col=visit_col
    )
    control = _complete_pair_frame(
        control_long, features, pair_col=pair_col, participant_col=participant_col, visit_col=visit_col
    )
    selected, ranking = _candidate_features(
        candidate,
        frda,
        control,
        features,
        pair_col=pair_col,
        participant_col=participant_col,
        visit_col=visit_col,
    )
    scored, fitted = _fit_and_score(
        frda,
        {"frda": frda, "control": control},
        selected,
        candidate,
        pair_col=pair_col,
        visit_col=visit_col,
    )
    scoring_parameters = pd.DataFrame(
        {
            "feature": selected,
            "training_mean": fitted["center"],
            "training_sd": fitted["scale"],
            "standardised_coefficient": fitted["coef"],
        }
    )
    scoring_parameters["absolute_coefficient"] = scoring_parameters[
        "standardised_coefficient"
    ].abs()
    scoring_parameters = scoring_parameters.sort_values(
        ["absolute_coefficient", "feature"], ascending=[False, True], kind="mergesort"
    ).reset_index(drop=True)
    scoring_parameters.insert(0, "coefficient_rank", np.arange(1, len(scoring_parameters) + 1))

    frda_effect = single_feature_effect_table(
        frda, selected, cohort="FRDA", pair_col=pair_col,
        participant_col=participant_col, visit_col=visit_col,
    ).drop(columns="cohort")
    control_effect = single_feature_effect_table(
        control, selected, cohort="Control", pair_col=pair_col,
        participant_col=participant_col, visit_col=visit_col,
    ).drop(columns="cohort")
    frda_effect = frda_effect.rename(
        columns={column: f"frda_{column}" for column in frda_effect if column != "feature"}
    )
    control_effect = control_effect.rename(
        columns={column: f"control_{column}" for column in control_effect if column != "feature"}
    )
    comparison = scoring_parameters.merge(frda_effect, on="feature", how="left").merge(
        control_effect, on="feature", how="left"
    )
    comparison["coefficient_sign"] = np.sign(comparison["standardised_coefficient"]).astype(int)
    comparison["frda_control_direction_relation"] = np.where(
        np.sign(comparison["frda_pooled_d_z"]) == np.sign(comparison["control_pooled_d_z"]),
        "same",
        "opposite",
    )
    if selection_frequency is not None:
        if isinstance(selection_frequency, pd.DataFrame):
            frequency = selection_frequency[["feature", "selection_frequency"]].drop_duplicates("feature")
        else:
            frequency = pd.DataFrame(
                {"feature": list(selection_frequency), "selection_frequency": list(selection_frequency.values())}
            )
        comparison = comparison.merge(frequency, on="feature", how="left")

    scores = pd.concat(
        [
            _decorate_scores(
                scored["frda"], strategy=candidate.label, candidate=candidate,
                cohort="FRDA", outer_fold=0, pair_col=pair_col,
            ),
            _decorate_scores(
                scored["control"], strategy=candidate.label, candidate=candidate,
                cohort="Control", outer_fold=0, pair_col=pair_col,
            ),
        ],
        ignore_index=True,
    )
    return {
        "feature_ranking": ranking,
        "scoring_parameters": scoring_parameters,
        "coefficient_single_feature_comparison": comparison,
        "scores": scores,
    }


__all__ = [
    "DualCohortCandidate",
    "control_aware_feature_ranking",
    "evaluate_dual_cohort_strategy",
    "fit_final_interpretation_model",
    "run_control_aware_comparison",
    "single_feature_effect_table",
    "wide_cohort_to_pair_long",
]
