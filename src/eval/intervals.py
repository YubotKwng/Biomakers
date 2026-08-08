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


DEFAULT_INTERVALS = (
    ("V1", "V3", "V1->V3", True),
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


__all__ = ["DEFAULT_INTERVALS", "interval_effect_summary"]
