"""Data audit helpers for longitudinal TRACK-FA tables."""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pandas as pd


VISIT_TIME: dict[str, float] = {"V1": 0.0, "V2": 1.0, "V3": 2.0}


def normalise_visit_label(value: Any) -> str:
    """Return canonical visit labels such as ``V1`` from common inputs."""
    if pd.isna(value):
        return ""
    text = str(value).strip().upper()
    if text.startswith("V"):
        return text
    try:
        return f"V{int(float(text))}"
    except ValueError:
        return text


def add_visit_time(
    df: pd.DataFrame,
    *,
    visit_col: str = "visit",
    output_col: str = "time_years",
    visit_time: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Attach canonical planned visit time in years without changing visits."""
    mapping = dict(visit_time or VISIT_TIME)
    out = df.copy()
    out[output_col] = out[visit_col].map(lambda v: mapping.get(normalise_visit_label(v), pd.NA))
    return out


def audit_visit_patterns(
    df: pd.DataFrame,
    subject_col: str,
    visit_col: str,
    *,
    visits: Iterable[str] = ("V1", "V2", "V3"),
) -> dict[str, Any]:
    """Summarise observed longitudinal visit availability per subject.

    The returned dictionary is machine-readable and includes the binary visit
    patterns requested in the implementation guide plus common pair counts.
    """
    required = {subject_col, visit_col}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"df missing required columns: {sorted(missing)}")

    visit_labels = [normalise_visit_label(v) for v in visits]
    present = (
        df[[subject_col, visit_col]]
        .dropna(subset=[subject_col])
        .assign(_visit=lambda x: x[visit_col].map(normalise_visit_label))
        .drop_duplicates([subject_col, "_visit"])
    )
    cross = pd.crosstab(present[subject_col], present["_visit"])
    for visit in visit_labels:
        if visit not in cross.columns:
            cross[visit] = 0
    cross = cross[visit_labels].astype(bool)

    pattern_series = cross.apply(lambda r: "".join("1" if bool(r[v]) else "0" for v in visit_labels), axis=1)
    pattern_order = ["111", "110", "101", "011", "100", "010", "001"]
    pattern_counts = {p: int((pattern_series == p).sum()) for p in pattern_order}

    def count_has(*needed: str) -> int:
        return int(cross[list(needed)].all(axis=1).sum())

    summary = {
        "n_subjects": int(cross.shape[0]),
        "patterns": pattern_counts,
        "n_v1": count_has("V1"),
        "n_v2": count_has("V2"),
        "n_v3": count_has("V3"),
        "n_v1_v2": count_has("V1", "V2"),
        "n_v2_v3": count_has("V2", "V3"),
        "n_v1_v3": count_has("V1", "V3"),
        "n_complete": count_has("V1", "V2", "V3"),
    }
    return summary


__all__ = ["VISIT_TIME", "add_visit_time", "audit_visit_patterns", "normalise_visit_label"]
