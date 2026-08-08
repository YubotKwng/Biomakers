"""Single-feature longitudinal baselines."""
from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from .intervals import DEFAULT_INTERVALS, interval_effect_summary


def single_feature_interval_baselines(
    long_df: pd.DataFrame,
    features: Iterable[str],
    *,
    subject_col: str = "subject_id",
    visit_col: str = "visit",
    time_col: str | None = "time_years",
    intervals=DEFAULT_INTERVALS,
    n_boot: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """Compute V1->V3, V1->V2, and V2->V3 d_z for each MRI feature."""
    rows = []
    for feature in [f for f in features if f in long_df.columns]:
        summary = interval_effect_summary(
            long_df.dropna(subset=[feature]),
            subject_col=subject_col,
            visit_col=visit_col,
            score_col=feature,
            time_col=time_col,
            intervals=intervals,
            n_boot=n_boot,
            seed=seed,
        )
        if summary.empty:
            continue
        summary = summary.assign(feature=feature, kind="single_mri_feature")
        rows.append(summary)
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if not out.empty:
        out = out.sort_values(["interval", "d_z"], ascending=[True, False], key=lambda s: s.abs() if s.name == "d_z" else s, kind="mergesort")
    return out.reset_index(drop=True)


__all__ = ["single_feature_interval_baselines"]
