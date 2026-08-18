"""Longitudinal interval summaries for composite and baseline scores."""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from .metrics import (
    bootstrap_paired_metric,
    compute_cohens_d,
    compute_longitudinal_deltas,
    paired_cohens_dz,
    probability_positive_change,
)


INTERVAL_SUMMARY_COLUMNS = [
    "interval",
    "n_pairs",
    "mean_change",
    "sd_change",
    "d_z",
    "d_z_ci_low",
    "d_z_ci_high",
    "p_delta_positive",
]

DEFAULT_INTERVALS = (
    ("V1", "V3", "V1->V3", False),
    ("V1", "V2", "V1->V2", True),
    ("V2", "V3", "V2->V3", True),
)


def interval_effect_summary(
    scores: pd.DataFrame,
    *,
    subject_col: str = "subject_id",
    visit_col: str = "visit",
    score_col: str = "score",
    time_col: str | None = "time_years",
    intervals: Iterable[tuple] = DEFAULT_INTERVALS,
    n_boot: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """Summarise paired d_z and P(delta>0) separately by visit interval."""
    rows = []
    for item in intervals:
        if len(item) == 3:
            start, end, label = item
            annualise = False
        else:
            start, end, label, annualise = item
        deltas_df = compute_longitudinal_deltas(
            scores,
            start,
            end,
            subject_col=subject_col,
            visit_col=visit_col,
            score_col=score_col,
            time_col=time_col,
            annualise=bool(annualise),
        )
        value_col = "annualised_delta" if annualise else "delta"
        deltas = pd.to_numeric(deltas_df[value_col], errors="coerce").dropna().to_numpy(dtype=float)
        effect = compute_cohens_d(deltas)
        boot = bootstrap_paired_metric(deltas, paired_cohens_dz, n_boot=n_boot, seed=seed)
        rows.append({
            "interval": label,
            "start_visit": str(start),
            "end_visit": str(end),
            "annualised": bool(annualise),
            "n_pairs": int(effect["n"]),
            "mean_change": effect["mean"],
            "sd_change": effect["sd"],
            "d_z": effect["d"],
            "d_z_ci_low": boot["ci_low"],
            "d_z_ci_high": boot["ci_high"],
            "p_delta_positive": probability_positive_change(deltas),
        })
    return pd.DataFrame(rows)


def adjacent_pair_interval_effect_summary(
    scores: pd.DataFrame,
    *,
    pair_col: str = "pair_id",
    visit_col: str = "visit",
    score_col: str = "score",
    n_boot: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """Summarise annual V1->V2 and V2->V3 d_z from pair-table OOF scores.

    TRACK-FA pair-table models represent annual intervals as pair identifiers
    such as ``AAN001_V1V2`` and ``AAN001_V2V3`` with visits ``1`` and ``2``
    inside each pair. This helper keeps those annual windows separate rather
    than pooling them into one d_z.
    """
    required = {pair_col, visit_col, score_col}
    missing = required - set(scores.columns)
    if missing:
        raise KeyError(f"scores missing required columns: {sorted(missing)}")

    tmp = scores[[pair_col, visit_col, score_col]].dropna().copy()
    if tmp.empty:
        return pd.DataFrame(columns=INTERVAL_SUMMARY_COLUMNS)
    tmp["_interval"] = (
        tmp[pair_col].astype(str).str.upper().str.extract(r"(V\d+V\d+)", expand=False)
        .map({"V1V2": "V1->V2", "V2V3": "V2->V3"})
    )
    tmp = tmp.dropna(subset=["_interval"])
    rows = []
    for interval, group in tmp.groupby("_interval", sort=True):
        paired = (
            group.sort_values([pair_col, visit_col])
            .groupby(pair_col)[score_col]
            .apply(list)
        )
        paired = paired[paired.map(len) == 2]
        deltas = paired.map(lambda x: float(x[1] - x[0])).to_numpy(dtype=float)
        effect = compute_cohens_d(deltas)
        boot = bootstrap_paired_metric(deltas, paired_cohens_dz, n_boot=n_boot, seed=seed)
        rows.append({
            "interval": interval,
            "n_pairs": effect["n"],
            "mean_change": effect["mean"],
            "sd_change": effect["sd"],
            "d_z": effect["d"],
            "d_z_ci_low": boot["ci_low"],
            "d_z_ci_high": boot["ci_high"],
            "p_delta_positive": probability_positive_change(deltas),
        })
    return pd.DataFrame(rows)


def pooled_adjacent_pair_effect_summary(
    scores: pd.DataFrame,
    *,
    pair_col: str = "pair_id",
    visit_col: str = "visit",
    score_col: str = "score",
    interval_label: str = "V1->V2 + V2->V3",
    n_boot: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """Summarise pooled annual d_z across V1V2 and V2V3 OOF pair deltas.

    This is an evaluation-only pooled annual metric. It does not alter the CV
    split: upstream OOF predictions must still be generated with participant
    grouping so all annual intervals from the same participant stay on the same
    side of each train/test split.
    """
    required = {pair_col, visit_col, score_col}
    missing = required - set(scores.columns)
    if missing:
        raise KeyError(f"scores missing required columns: {sorted(missing)}")

    tmp = scores[[pair_col, visit_col, score_col]].dropna().copy()
    if tmp.empty:
        return pd.DataFrame(columns=INTERVAL_SUMMARY_COLUMNS)
    tmp["_interval"] = (
        tmp[pair_col]
        .astype(str)
        .str.upper()
        .str.extract(r"(V\d+V\d+)", expand=False)
    )
    tmp = tmp[tmp["_interval"].isin(["V1V2", "V2V3"])].copy()
    paired = (
        tmp.sort_values([pair_col, visit_col])
        .groupby(pair_col)[score_col]
        .apply(list)
    )
    paired = paired[paired.map(len) == 2]
    deltas = paired.map(lambda x: float(x[1] - x[0])).to_numpy(dtype=float)
    effect = compute_cohens_d(deltas)
    boot = bootstrap_paired_metric(deltas, paired_cohens_dz, n_boot=n_boot, seed=seed)
    return pd.DataFrame([{
        "interval": interval_label,
        "n_pairs": effect["n"],
        "mean_change": effect["mean"],
        "sd_change": effect["sd"],
        "d_z": effect["d"],
        "d_z_ci_low": boot["ci_low"],
        "d_z_ci_high": boot["ci_high"],
        "p_delta_positive": probability_positive_change(deltas),
    }])


def annual_tuning_diagnostics(
    interval_summary: pd.DataFrame,
    *,
    interval_col: str = "interval",
    dz_col: str = "d_z",
    p_col: str = "p_delta_positive",
    require_both: bool = True,
) -> dict:
    """Return mean annual d_z and consistency diagnostics from V1V2/V2V3."""
    if interval_summary is None or interval_summary.empty:
        return {
            "dz_v1_v2": np.nan,
            "dz_v2_v3": np.nan,
            "mean_validation_annual_dz": np.nan,
            "annual_interval_gap": np.nan,
            "p_progression": np.nan,
        }
    work = interval_summary.copy()
    key = work[interval_col].astype(str).str.upper()

    def get(interval: str, col: str):
        rows = work[key == interval.upper()]
        if rows.empty or col not in rows.columns:
            return np.nan
        return rows.iloc[0][col]

    d12 = pd.to_numeric(pd.Series([get("V1->V2", dz_col)]), errors="coerce").iloc[0]
    d23 = pd.to_numeric(pd.Series([get("V2->V3", dz_col)]), errors="coerce").iloc[0]
    p12 = pd.to_numeric(pd.Series([get("V1->V2", p_col)]), errors="coerce").iloc[0]
    p23 = pd.to_numeric(pd.Series([get("V2->V3", p_col)]), errors="coerce").iloc[0]
    if require_both and not (np.isfinite(d12) and np.isfinite(d23)):
        return {
            "dz_v1_v2": float(d12) if np.isfinite(d12) else np.nan,
            "dz_v2_v3": float(d23) if np.isfinite(d23) else np.nan,
            "mean_validation_annual_dz": np.nan,
            "annual_interval_gap": np.nan,
            "p_progression": np.nan,
        }
    annual_vals = [v for v in (d12, d23) if np.isfinite(v)]
    p_vals = [v for v in (p12, p23) if np.isfinite(v)]
    return {
        "dz_v1_v2": float(d12) if np.isfinite(d12) else np.nan,
        "dz_v2_v3": float(d23) if np.isfinite(d23) else np.nan,
        "mean_validation_annual_dz": float(np.mean(annual_vals)) if annual_vals else np.nan,
        "annual_interval_gap": (
            float(abs(d12 - d23))
            if np.isfinite(d12) and np.isfinite(d23)
            else np.nan
        ),
        "p_progression": float(np.mean(p_vals)) if p_vals else np.nan,
    }


__all__ = [
    "DEFAULT_INTERVALS",
    "INTERVAL_SUMMARY_COLUMNS",
    "adjacent_pair_interval_effect_summary",
    "annual_tuning_diagnostics",
    "interval_effect_summary",
    "pooled_adjacent_pair_effect_summary",
]
