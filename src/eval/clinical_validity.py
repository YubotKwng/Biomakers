"""Clinical validation for locked OOF composite scores."""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .metrics import compute_longitudinal_deltas


def _spearman_with_n(x, y) -> dict:
    frame = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(frame) < 3 or frame["x"].nunique() < 2 or frame["y"].nunique() < 2:
        return {"rho": np.nan, "p_value": np.nan, "n": int(len(frame))}
    rho, p = spearmanr(frame["x"], frame["y"])
    return {"rho": float(rho), "p_value": float(p), "n": int(len(frame))}


def clinical_validity(
    oof_scores: pd.DataFrame,
    clinical_data: pd.DataFrame,
    *,
    subject_col: str = "subject_id",
    visit_col: str = "visit",
    score_col: str = "score",
    clinical_variables: Iterable[str] = ("FARS", "SARA"),
    start_visit="V1",
    end_visit="V3",
) -> pd.DataFrame:
    """Calculate post-lock clinical associations for OOF composite scores."""
    merge_cols = [subject_col, visit_col]
    score_df = oof_scores[merge_cols + [score_col]].copy()
    clin_vars = [v for v in clinical_variables if v in clinical_data.columns]
    clinical_df = clinical_data[merge_cols + clin_vars].copy()
    merged = score_df.merge(clinical_df, on=merge_cols, how="inner")

    rows: list[dict] = []
    for var in clin_vars:
        cross = _spearman_with_n(merged[score_col], merged[var])
        rows.append({"analysis": "cross_sectional", "clinical_variable": var, **cross})

        deltas_score = compute_longitudinal_deltas(
            merged[[subject_col, visit_col, score_col]],
            start_visit,
            end_visit,
            subject_col=subject_col,
            visit_col=visit_col,
            score_col=score_col,
            annualise=False,
        )[[subject_col, "delta"]].rename(columns={"delta": "delta_score"})
        deltas_clin = compute_longitudinal_deltas(
            merged[[subject_col, visit_col, var]].rename(columns={var: "_clinical"}),
            start_visit,
            end_visit,
            subject_col=subject_col,
            visit_col=visit_col,
            score_col="_clinical",
            annualise=False,
        )[[subject_col, "delta"]].rename(columns={"delta": f"delta_{var}"})
        delta_merged = deltas_score.merge(deltas_clin, on=subject_col, how="inner")
        longi = _spearman_with_n(delta_merged["delta_score"], delta_merged[f"delta_{var}"])
        rows.append({"analysis": "longitudinal_delta", "clinical_variable": var, **longi})
    return pd.DataFrame(rows)


__all__ = ["clinical_validity"]
