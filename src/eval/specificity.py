"""Healthy-control specificity analysis for locked composite models."""
from __future__ import annotations

import pandas as pd

from .metrics import (
    compute_cohens_d,
    compute_longitudinal_deltas,
    probability_positive_change,
)


def evaluate_locked_model_specificity(
    frda_scores: pd.DataFrame,
    control_scores: pd.DataFrame,
    *,
    subject_col: str = "subject_id",
    visit_col: str = "visit",
    score_col: str = "score",
    start_visit="V1",
    end_visit="V3",
) -> pd.DataFrame:
    """Compare FRDA and control longitudinal change after model locking.

    Inputs must already be scores from the same FRDA-trained preprocessing and
    composite model. This helper deliberately does not fit or tune anything.
    """
    rows = []
    for cohort, scores in (("FRDA", frda_scores), ("Control", control_scores)):
        deltas = compute_longitudinal_deltas(
            scores,
            start_visit,
            end_visit,
            subject_col=subject_col,
            visit_col=visit_col,
            score_col=score_col,
            annualise=False,
        )
        d_out = compute_cohens_d(deltas["delta"])
        rows.append({
            "cohort": cohort,
            "interval": f"{start_visit}->{end_visit}",
            "n_pairs": d_out["n"],
            "mean_change": d_out["mean"],
            "sd_change": d_out["sd"],
            "cohens_dz": d_out["d"],
            "p_delta_positive": probability_positive_change(deltas["delta"]),
        })
    return pd.DataFrame(rows)


__all__ = ["evaluate_locked_model_specificity"]
