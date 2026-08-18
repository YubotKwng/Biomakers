"""Data audit helpers for longitudinal TRACK-FA tables."""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pandas as pd


VISIT_TIME: dict[str, float] = {"V1": 0.0, "V2": 1.0, "V3": 2.0}
VISIT_PATTERN_MEANINGS: dict[str, str] = {
    "111": "V1, V2, and V3 available",
    "110": "V1 and V2 available; V3 missing",
    "101": "V1 and V3 available; V2 missing",
    "011": "V2 and V3 available; V1 missing",
    "100": "Only V1 available",
    "010": "Only V2 available",
    "001": "Only V3 available",
    "000": "No requested visits available",
}


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

    pattern_series = cross.apply(
        lambda row: "".join("1" if bool(row[visit]) else "0" for visit in visit_labels),
        axis=1,
    )
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


def visit_pattern_table(audit: dict[str, Any]) -> pd.DataFrame:
    """Return notebook-ready visit-pattern counts with plain-language meaning."""
    patterns = dict(audit.get("patterns", {}))
    n_subjects = int(audit.get("n_subjects", sum(patterns.values())))
    rows = []
    for pattern, meaning in VISIT_PATTERN_MEANINGS.items():
        n = int(patterns.get(pattern, 0))
        if n == 0 and pattern == "000":
            continue
        rows.append({
            "pattern": pattern,
            "meaning": meaning,
            "n_subjects": n,
            "percent_subjects": (100.0 * n / n_subjects) if n_subjects else pd.NA,
        })
    return pd.DataFrame(rows)


def analysis_population_counts(
    df: pd.DataFrame,
    subject_col: str,
    visit_col: str = "visit",
) -> pd.DataFrame:
    """Count the analysis populations used by the longitudinal pipeline."""
    audit = audit_visit_patterns(df, subject_col=subject_col, visit_col=visit_col)
    rows = [
        {
            "population": "V1-V3 primary cohort",
            "required_visits": "V1,V3",
            "n_subjects": audit["n_v1_v3"],
            "role": "Primary 24-month annualised paired change",
        },
        {
            "population": "V1-V2 cohort",
            "required_visits": "V1,V2",
            "n_subjects": audit["n_v1_v2"],
            "role": "Secondary 12-month interval",
        },
        {
            "population": "V2-V3 cohort",
            "required_visits": "V2,V3",
            "n_subjects": audit["n_v2_v3"],
            "role": "Secondary 12-month interval",
        },
        {
            "population": "Complete V1-V2-V3 cohort",
            "required_visits": "V1,V2,V3",
            "n_subjects": audit["n_complete"],
            "role": "Consistency and trajectory checks",
        },
        {
            "population": "All longitudinal with any adjacent pair",
            "required_visits": "V1,V2 or V2,V3",
            "n_subjects": int(audit["n_v1_v2"] + audit["n_v2_v3"] - audit["n_complete"]),
            "role": "Maximum adjacent-interval sensitivity check",
        },
    ]
    return pd.DataFrame(rows)


def modelling_pair_count_table(
    pairs: pd.DataFrame,
    *,
    patient_col: str = "patient_id",
    expected: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Return canonical modelling-cohort counts from annual pair identifiers.

    ``trackfa_pairs_drop3poms.csv`` stores adjacent intervals as identifiers
    such as ``AAN001_V1V2`` and ``AAN001_V2V3``. The 24-month V1->V3 count is
    therefore the number of subjects with both annual rows, not a separate row
    count in the paired CSV.
    """
    if patient_col not in pairs.columns:
        raise KeyError(f"pairs missing required column {patient_col!r}")
    parsed = pairs[patient_col].astype(str).str.extract(
        r"(?P<subject>.+)_(?P<pair_type>V1V2|V2V3)$"
    )
    if parsed.isna().any().any():
        examples = pairs.loc[parsed.isna().any(axis=1), patient_col].head(10).tolist()
        raise ValueError(f"Could not parse annual pair identifiers: {examples}")

    subjects_v12 = set(parsed.loc[parsed["pair_type"].eq("V1V2"), "subject"])
    subjects_v23 = set(parsed.loc[parsed["pair_type"].eq("V2V3"), "subject"])
    subjects_both = subjects_v12 & subjects_v23
    counts = {
        "N12": len(subjects_v12),
        "N23": len(subjects_v23),
        "N13": len(subjects_both),
        "N123": len(subjects_both),
    }
    rows = [
        {
            "count": "N12",
            "interval": "V1->V2",
            "n": counts["N12"],
            "definition": "subjects with a V1V2 annual pair row",
        },
        {
            "count": "N23",
            "interval": "V2->V3",
            "n": counts["N23"],
            "definition": "subjects with a V2V3 annual pair row",
        },
        {
            "count": "N13",
            "interval": "V1->V3",
            "n": counts["N13"],
            "definition": "subjects with both V1V2 and V2V3 annual pair rows",
        },
        {
            "count": "N123",
            "interval": "V1,V2,V3",
            "n": counts["N123"],
            "definition": "subjects represented across all three visits via both annual pair rows",
        },
    ]
    out = pd.DataFrame(rows)
    if expected is not None:
        expected = dict(expected)
        mismatches = {
            key: (counts.get(key), value)
            for key, value in expected.items()
            if counts.get(key) != value
        }
        if mismatches:
            raise ValueError(f"Modelling pair-count mismatch: {mismatches}")
    return out


__all__ = [
    "VISIT_TIME",
    "VISIT_PATTERN_MEANINGS",
    "add_visit_time",
    "analysis_population_counts",
    "audit_visit_patterns",
    "modelling_pair_count_table",
    "normalise_visit_label",
    "visit_pattern_table",
]
