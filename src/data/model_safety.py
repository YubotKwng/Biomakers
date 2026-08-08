"""Guardrails that prevent clinical-score/control leakage into model fitting."""
from __future__ import annotations

import re
from typing import Any, Iterable, Sequence

import pandas as pd


CONTROL_GROUP_COLUMNS = ("study_group", "group", "cohort", "diagnosis", "dx")
CONTROL_LABELS = {
    "1",
    "control",
    "controls",
    "healthy_control",
    "healthy controls",
    "healthy control",
    "hc",
    "unaffected",
}

CLINICAL_SCORE_BASES = {
    "fars",
    "fars1",
    "fars2",
    "dfars",
    "mfars_total",
    "sara",
    "sara_total",
    "adl",
    "adl_total",
    "lcslc_total",
    "functional_staging_score",
    "hpt_dom_av",
    "hpt_ndom_av",
}


def _normalise_token(value: Any) -> str:
    """Normalise free-text labels before leakage/control matching."""
    return str(value).strip().lower().replace("-", "_")


def _strip_model_suffixes(name: str) -> str:
    """Remove visit/delta suffixes so clinical score families are recognised."""
    s = str(name).strip()
    if s.startswith("delta_"):
        s = s[len("delta_") :]
    for suffix in (
        "_baseline",
        "_followup",
        "_v1",
        "_v2",
        "_v3",
        "1",
        "2",
    ):
        if s.endswith(suffix):
            s = s[: -len(suffix)]
            break
    return s


def is_clinical_score_column(name: str) -> bool:
    """Return True for clinical scale columns in raw, long, or pair tables."""
    s = _normalise_token(name)
    base = _normalise_token(_strip_model_suffixes(s))
    if base in CLINICAL_SCORE_BASES:
        return True
    return bool(re.match(r"^(delta_)?(m?fars|sara|adl|lcslc)(_|$)", s))


def clinical_score_columns(cols: Iterable[str]) -> list[str]:
    """Clinical-score columns, preserving caller order."""
    return [str(c) for c in cols if is_clinical_score_column(str(c))]


def assert_no_clinical_score_features(feature_cols: Sequence[str]) -> None:
    """Fail if any model input feature is a clinical score."""
    leaked = clinical_score_columns(feature_cols)
    if leaked:
        raise ValueError(
            "Clinical-score columns cannot be used as model features: "
            + ", ".join(leaked[:20])
        )


def assert_no_clinical_score_target(target_col: str | None) -> None:
    """Fail if model training uses a clinical score as the supervised target."""
    if target_col is not None and is_clinical_score_column(target_col):
        raise ValueError(f"Clinical score {target_col!r} cannot be used as a model-training target.")


def control_mask(df: pd.DataFrame) -> pd.Series:
    """Identify rows that look like healthy/control participants."""
    mask = pd.Series(False, index=df.index)
    for col in CONTROL_GROUP_COLUMNS:
        if col not in df.columns:
            continue
        values = df[col].map(_normalise_token)
        mask |= values.isin(CONTROL_LABELS)
    return mask


def drop_control_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with control participants removed when group columns exist."""
    mask = control_mask(df)
    if not mask.any():
        return df.copy()
    return df.loc[~mask].copy()


def assert_no_control_rows(df: pd.DataFrame) -> None:
    """Fail if a model-training frame still contains controls."""
    mask = control_mask(df)
    if mask.any():
        examples = df.loc[mask, [c for c in CONTROL_GROUP_COLUMNS if c in df.columns]].head(5)
        raise ValueError(
            "Control rows are not allowed in model training data. Examples:\n"
            + examples.to_string(index=False)
        )


def assert_training_frame_is_patient_only(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    *,
    target_col: str | None = None,
    allow_clinical_target: bool = False,
) -> None:
    """Validate the two leakage rules before fitting a model.

    The progression models should learn from patient imaging rows only.
    Clinical scores are reserved for benchmark tables unless a caller
    explicitly opts into a clinical-target reference path.
    """
    assert_no_control_rows(df)
    assert_no_clinical_score_features(feature_cols)
    if not allow_clinical_target:
        assert_no_clinical_score_target(target_col)


__all__ = [
    "assert_no_clinical_score_features",
    "assert_no_clinical_score_target",
    "assert_no_control_rows",
    "assert_training_frame_is_patient_only",
    "clinical_score_columns",
    "control_mask",
    "drop_control_rows",
    "is_clinical_score_column",
]
