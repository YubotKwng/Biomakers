"""Wide → long reshape utilities.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Canonical implementation
# ---------------------------------------------------------------------------
def wide_to_long(
    df: pd.DataFrame,
    subject_col: str,
    suffixes: Tuple[str, str] = ("_v1", "_v2"),
    extra_targets: Optional[Sequence[Tuple[str, str, str]]] = None,
) -> pd.DataFrame:
    """Generic wide→long reshape that auto-pairs ``*<sfx1>`` / ``*<sfx2>``
    columns.

    Parameters
    ----------
    df : DataFrame
        Wide-format frame with one row per subject.
    subject_col : str
        Subject id column.
    suffixes : (str, str), default ``("_v1", "_v2")``
        The visit suffixes to detect. Bases are extracted by stripping
        ``suffixes[0]`` / ``suffixes[1]``.
    extra_targets : optional sequence of ``(base, v1_col, v2_col)`` tuples
        Targets that don't follow the suffix convention (e.g.
        ``("FARS", "FARS1", "FARS2")``). They are added as ``base`` to each
        long row.

    Returns
    -------
    DataFrame with columns ``[subject_col, 'visit'] + bases + extras``.
    """
    sfx1, sfx2 = suffixes
    v1_cols = [c for c in df.columns if c.endswith(sfx1)]
    v2_cols = [c for c in df.columns if c.endswith(sfx2)]
    bases = sorted({c[: -len(sfx1)] for c in v1_cols} & {c[: -len(sfx2)] for c in v2_cols})

    rows: list[dict] = []
    extra_targets = list(extra_targets or [])

    for _, r in df.iterrows():
        sid = r[subject_col]

        row1 = {subject_col: sid, "visit": 1}
        for b in bases:
            row1[b] = r.get(b + sfx1)
        for base, v1_col, _v2_col in extra_targets:
            row1[base] = r.get(v1_col)
        rows.append(row1)

        row2 = {subject_col: sid, "visit": 2}
        for b in bases:
            row2[b] = r.get(b + sfx2)
        for base, _v1_col, v2_col in extra_targets:
            row2[base] = r.get(v2_col)
        rows.append(row2)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Domain-specific reshaping helpers
# ---------------------------------------------------------------------------
def build_long_format(
    df: pd.DataFrame,
    id_col: str,
    feature_cols_v1: Sequence[str],
    feature_cols_v2: Sequence[str],
    base_cols: Sequence[str],
    target_v1: str = "FARS1",
    target_v2: str = "FARS2",
    target_name: str = "FARS",
) -> pd.DataFrame:
    """Build a two-visit long table from explicit visit-specific columns.

    The returned frame is suitable for paired progression metrics and
    subject-level validation because each subject contributes one row per
    visit with a common target name.
    """
    df_v1 = df[[id_col] + list(feature_cols_v1) + [target_v1]].copy()
    df_v1.columns = [id_col] + list(base_cols) + [target_name]
    df_v1["visit"] = 1

    df_v2 = df[[id_col] + list(feature_cols_v2) + [target_v2]].copy()
    df_v2.columns = [id_col] + list(base_cols) + [target_name]
    df_v2["visit"] = 2

    return pd.concat([df_v1, df_v2], ignore_index=True)


def build_long_format_fars(
    df: pd.DataFrame,
    id_col: str,
    feature_cols_v1: Sequence[str],
    feature_cols_v2: Sequence[str],
    base_cols: Sequence[str],
) -> pd.DataFrame:
    """Build a FARS long table from visit-specific FARS columns."""
    return build_long_format(
        df,
        id_col,
        feature_cols_v1,
        feature_cols_v2,
        base_cols,
        target_v1="FARS1",
        target_v2="FARS2",
        target_name="FARS",
    )


def build_long_format_sara(
    df: pd.DataFrame,
    id_col: str,
    feature_cols_v1: Sequence[str],
    feature_cols_v2: Sequence[str],
    base_cols: Sequence[str],
) -> pd.DataFrame:
    """Build a SARA long table from visit-specific SARA columns."""
    return build_long_format(
        df,
        id_col,
        feature_cols_v1,
        feature_cols_v2,
        base_cols,
        target_v1="SARA1",
        target_v2="SARA2",
        target_name="SARA",
    )


def build_long_from_wide(
    df: pd.DataFrame,
    subject_col: str,
    structural_ext: Sequence[str],
    diffusion: Sequence[str],
    fars_v1_col: str = "FARS1",
    fars_v2_col: str = "FARS2",
    sara_v1_col: str = "SARA1",
    sara_v2_col: str = "SARA2",
) -> pd.DataFrame:
    """Build visit-level rows from a wide two-visit FRDA table.

    Demographic and genetic covariates are copied to each visit, clinical
    targets are mapped to ``FARS`` and ``SARA``, and available imaging
    features are carried through by base name.
    """
    rows: list[dict] = []
    cols = set(df.columns)

    for _, r in df.iterrows():
        sid = r[subject_col]

        # visit 1
        row1 = {subject_col: sid, "visit": 1}
        row1["age"] = r.get("age1")
        row1["age_at_onset"] = r.get("age_at_onset")
        row1["dur"] = r.get("dur1")
        row1["sex"] = r.get("sex")
        row1["GAA1"] = r.get("GAA1")
        row1["GAA2"] = r.get("GAA2")
        row1["FARS"] = r.get(fars_v1_col)
        row1["SARA"] = r.get(sara_v1_col)

        for f in list(structural_ext) + list(diffusion):
            col = f + "_v1"
            if col in cols:
                row1[f] = r.get(col)

        # visit 2
        row2 = {subject_col: sid, "visit": 2}
        row2["age"] = r.get("age2")
        row2["age_at_onset"] = r.get("age_at_onset")
        row2["dur"] = r.get("dur2")
        row2["sex"] = r.get("sex")
        row2["GAA1"] = r.get("GAA1")
        row2["GAA2"] = r.get("GAA2")
        row2["FARS"] = r.get(fars_v2_col)
        row2["SARA"] = r.get(sara_v2_col)

        for f in list(structural_ext) + list(diffusion):
            col = f + "_v2"
            if col in cols:
                row2[f] = r.get(col)

        rows.append(row1)
        rows.append(row2)

    return pd.DataFrame(rows)


__all__ = [
    "wide_to_long",
    "build_long_format",
    "build_long_format_fars",
    "build_long_format_sara",
    "build_long_from_wide",
]
