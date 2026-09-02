"""Fold-level train/test clinical benchmark tables for supervisor review."""
from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

from ..data.trackfa_pairs import parse_trackfa_patient_id
from ..eval.cv import group_kfold_indices, resolve_split_group_col
from ..eval.metrics import compute_cohens_d


def _pair_type_from_id(values: pd.Series) -> pd.Series:
    return (
        values.astype(str)
        .str.upper()
        .str.extract(r"(V\d+V\d+)", expand=False)
        .fillna("")
    )


def _clinical_delta_col(scale: str) -> str | None:
    key = str(scale).strip().lower()
    mapping = {
        "fars": "delta_mfars_total",
        "mfars": "delta_mfars_total",
        "mfars_total": "delta_mfars_total",
        "sara": "delta_sara_total",
        "sara_total": "delta_sara_total",
        "adl": "delta_adl_total",
        "adl_total": "delta_adl_total",
    }
    return mapping.get(key)


def _effect(values: pd.Series) -> dict[str, float]:
    deltas = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    out = compute_cohens_d(deltas)
    return {
        "d": out["d"],
        "mean": out["mean"],
        "sd": out["sd"],
        "n": out["n"],
    }


def _model_ready_frame(
    long_df: pd.DataFrame,
    feature_cols: Sequence[str],
    *,
    subject_col: str,
    visit_col: str,
    split_group_col: str | None,
    start_visit: int,
    end_visit: int,
) -> tuple[pd.DataFrame, str]:
    feats = [f for f in feature_cols if f in long_df.columns]
    resolved_split_group_col = resolve_split_group_col(long_df, subject_col, split_group_col)
    cols = [subject_col, visit_col] + feats
    if resolved_split_group_col not in cols:
        cols.append(resolved_split_group_col)
    sub = long_df[cols].dropna().copy()
    interval_visits = {int(start_visit), int(end_visit)}
    visit_int = pd.to_numeric(sub[visit_col], errors="coerce").astype("Int64")
    sub = sub[visit_int.isin(interval_visits)].copy()
    counts = sub.groupby(subject_col)[visit_col].agg(
        lambda s: interval_visits.issubset(
            set(pd.to_numeric(s, errors="coerce").dropna().astype(int))
        )
    )
    return sub[sub[subject_col].isin(counts[counts].index)].copy(), resolved_split_group_col


def fold_train_test_clinical_benchmark_table(
    long_df: pd.DataFrame,
    pairs_df: pd.DataFrame,
    feature_cols: Sequence[str],
    *,
    subject_col: str = "pair_id",
    visit_col: str = "visit",
    split_group_col: str | None = "subject",
    cv_n_splits: int = 5,
    random_seed: int = 42,
    clinical_scales: Iterable[str] = ("FARS", "SARA"),
    pair_types: Iterable[str] = ("V1V2", "V2V3"),
    start_visit: int = 1,
    end_visit: int = 2,
) -> pd.DataFrame:
    """Compute clinical train/test d values on the exact outer CV folds.

    The output is fold-level and subject-count aware so review tables can show
    that FARS/SARA benchmarks were calculated from the same held-out subjects
    used for composite testing.
    """
    if "patient_id" not in pairs_df.columns:
        raise KeyError("pairs_df must contain the TRACK-FA patient_id column")

    sub, resolved_split_group_col = _model_ready_frame(
        long_df,
        feature_cols,
        subject_col=subject_col,
        visit_col=visit_col,
        split_group_col=split_group_col,
        start_visit=start_visit,
        end_visit=end_visit,
    )
    groups = sub[resolved_split_group_col].values
    split_groups = np.asarray(sub[resolved_split_group_col].unique())
    use_kfold = cv_n_splits is not None and 1 < int(cv_n_splits) < len(split_groups)
    splits = (
        group_kfold_indices(groups, n_splits=int(cv_n_splits), seed=random_seed)
        if use_kfold
        else (
            (np.where(groups != sid)[0], np.where(groups == sid)[0])
            for sid in split_groups
        )
    )

    pairs = pairs_df.copy()
    pairs["_subject"] = pairs["patient_id"].map(parse_trackfa_patient_id)
    pairs["_pair_type"] = _pair_type_from_id(pairs["patient_id"])
    allowed_pair_types = {str(p).upper() for p in pair_types}
    pairs = pairs[pairs["_pair_type"].isin(allowed_pair_types)].copy()

    rows: list[dict] = []
    for fold, (train_idx, test_idx) in enumerate(splits, start=1):
        train_subjects = set(sub.iloc[train_idx][resolved_split_group_col].astype(str))
        test_subjects = set(sub.iloc[test_idx][resolved_split_group_col].astype(str))
        train_pairs = pairs[pairs["_subject"].astype(str).isin(train_subjects)]
        test_pairs = pairs[pairs["_subject"].astype(str).isin(test_subjects)]

        for scale in clinical_scales:
            delta_col = _clinical_delta_col(scale)
            if delta_col is None or delta_col not in pairs.columns:
                continue
            train_eff = _effect(train_pairs[delta_col])
            test_eff = _effect(test_pairs[delta_col])
            rows.append(
                {
                    "fold": int(fold),
                    "clinical_scale": str(scale),
                    "train_n_subjects": int(len(train_subjects)),
                    "test_n_subjects": int(len(test_subjects)),
                    "train_n_pairs": int(len(train_pairs)),
                    "test_n_pairs": int(len(test_pairs)),
                    "clinical_train_d": train_eff["d"],
                    "clinical_test_d": test_eff["d"],
                    "clinical_train_mean_delta": train_eff["mean"],
                    "clinical_test_mean_delta": test_eff["mean"],
                    "clinical_train_sd_delta": train_eff["sd"],
                    "clinical_test_sd_delta": test_eff["sd"],
                    "clinical_train_n_pairs": train_eff["n"],
                    "clinical_test_n_pairs": test_eff["n"],
                    "clinical_train_minus_test_d": train_eff["d"] - test_eff["d"],
                    "pair_types": ",".join(sorted(allowed_pair_types)),
                    "split_group_col": resolved_split_group_col,
                    "cv_mode": "group_kfold" if use_kfold else "loo",
                    "cv_n_splits": int(cv_n_splits) if use_kfold else int(len(split_groups)),
                }
            )
    return pd.DataFrame(rows)


__all__ = ["fold_train_test_clinical_benchmark_table"]
