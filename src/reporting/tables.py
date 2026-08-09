"""Notebook-ready summary tables for final biomarker reporting."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _lookup_interval(interval_table: pd.DataFrame | None, interval: str, metric: str):
    if interval_table is None or interval_table.empty or metric not in interval_table.columns:
        return np.nan
    rows = interval_table[interval_table["interval"].astype(str).str.upper() == interval.upper()]
    if rows.empty:
        return np.nan
    return rows.iloc[0][metric]


def final_model_performance_matrix(
    *,
    composite_intervals: pd.DataFrame | None = None,
    clinical_intervals: pd.DataFrame | None = None,
    single_feature_intervals: pd.DataFrame | None = None,
    clinical_validity: pd.DataFrame | None = None,
    specificity: pd.DataFrame | None = None,
    stability: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Assemble the final evaluation checklist required by the project plan."""
    best_single = np.nan
    if single_feature_intervals is not None and not single_feature_intervals.empty:
        rows = single_feature_intervals[single_feature_intervals["interval"].astype(str).str.upper() == "V1->V3"]
        if rows.empty:
            rows = single_feature_intervals[single_feature_intervals["interval"].astype(str).str.upper() == "V1->V3".upper()]
        if not rows.empty:
            best_single = rows.iloc[rows["d_z"].abs().argmax()]["d_z"]

    clin_ref = np.nan
    if clinical_intervals is not None and not clinical_intervals.empty:
        rows = clinical_intervals[clinical_intervals["interval"].astype(str).str.upper() == "V1->V3"]
        if not rows.empty and "d_z" in rows:
            clin_ref = rows.iloc[rows["d_z"].abs().argmax()]["d_z"]

    rows = [
        {
            "question": "12-month sensitivity V1->V2",
            "metric": "V1->V2 paired d_z",
            "role": "Primary",
            "value": _lookup_interval(composite_intervals, "V1->V2", "d_z"),
        },
        {
            "question": "12-month sensitivity V2->V3",
            "metric": "V2->V3 paired d_z",
            "role": "Primary temporal replication",
            "value": _lookup_interval(composite_intervals, "V2->V3", "d_z"),
        },
        {
            "question": "24-month cumulative sensitivity",
            "metric": "V1->V3 paired d_z",
            "role": "Secondary",
            "value": _lookup_interval(composite_intervals, "V1->V3", "d_z"),
        },
        {
            "question": "Direction consistency",
            "metric": "P(delta > 0)",
            "role": "Secondary",
            "value": f"{_lookup_interval(composite_intervals, 'V1->V2', 'p_delta_positive')}; {_lookup_interval(composite_intervals, 'V2->V3', 'p_delta_positive')}",
        },
        {
            "question": "Robustness",
            "metric": "Bootstrap CI for d_z",
            "role": "Primary uncertainty",
            "value": f"V1->V2 [{_lookup_interval(composite_intervals, 'V1->V2', 'd_z_ci_low')}, {_lookup_interval(composite_intervals, 'V1->V2', 'd_z_ci_high')}]; V2->V3 [{_lookup_interval(composite_intervals, 'V2->V3', 'd_z_ci_low')}, {_lookup_interval(composite_intervals, 'V2->V3', 'd_z_ci_high')}]",
        },
        {
            "question": "Better than clinical scale?",
            "metric": "d_z composite vs FARS/SARA",
            "role": "RQ1",
            "value": f"{_lookup_interval(composite_intervals, 'V1->V3', 'd_z')} vs {clin_ref}",
        },
        {
            "question": "Better than MRI alone?",
            "metric": "vs strongest individual MRI feature",
            "role": "RQ1",
            "value": f"{_lookup_interval(composite_intervals, 'V1->V3', 'd_z')} vs {best_single}",
        },
        {
            "question": "Disease specific?",
            "metric": "FRDA vs control change",
            "role": "Specificity",
            "value": _first_value(specificity),
        },
        {
            "question": "Clinically meaningful?",
            "metric": "Spearman Z vs FARS/SARA",
            "role": "RQ3",
            "value": _first_value(clinical_validity, contains="cross"),
        },
        {
            "question": "Tracks clinical change?",
            "metric": "Spearman delta Z vs delta FARS/SARA",
            "role": "Strong RQ3",
            "value": _first_value(clinical_validity, contains="delta"),
        },
        {
            "question": "Feature robustness",
            "metric": "Coefficient/sign/Jaccard stability",
            "role": "Robustness",
            "value": _first_value(stability),
        },
    ]
    return pd.DataFrame(rows)


def _first_value(table: pd.DataFrame | None, *, contains: str | None = None):
    if table is None or table.empty:
        return np.nan
    work = table
    if contains is not None:
        text = table.astype(str).agg(" ".join, axis=1).str.lower()
        work = table[text.str.contains(contains.lower(), na=False)]
        if work.empty:
            return np.nan
    for col in ("value", "rho", "effect", "d_z", "mean_jaccard", "sign_stability"):
        if col in work.columns:
            return work.iloc[0][col]
    return work.iloc[0].to_dict()


__all__ = ["final_model_performance_matrix"]
