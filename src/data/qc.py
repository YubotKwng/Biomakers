"""Quality-control + data-conditioning helpers.
"""
from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
import pandas as pd


def _as_float_array(x) -> np.ndarray:
    """Coerce input to a one- or two-dimensional float array."""
    return np.asarray(x, dtype=float)


def standardize_train_test(
    X_train, X_test
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Z-score train and test arrays using statistics from training data.

    Returns ``(X_train_s, X_test_s, mu, sd)``. Zero-variance columns are
    assigned a scale of 1.0 so downstream models remain numerically stable.
    """
    X_train = _as_float_array(X_train)
    X_test = _as_float_array(X_test)
    mu = np.nanmean(X_train, axis=0)
    sd = np.nanstd(X_train, axis=0, ddof=0)
    sd = np.where(sd == 0, 1.0, sd)
    return (X_train - mu) / sd, (X_test - mu) / sd, mu, sd


def missingness_summary(df: pd.DataFrame, cols: Sequence[str]) -> pd.Series:
    """Return per-column missing-value counts sorted from high to low.

    Missing requested columns are reported so data-availability problems are
    visible before model fitting.
    """
    present = [c for c in cols if c in df.columns]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        print("WARNING: missing columns:", missing)
    return df[present].isna().sum().sort_values(ascending=False)


def tukey_outliers_mask(
    df: pd.DataFrame, cols: Sequence[str], k: float = 3.0
) -> pd.Series:
    """Return rows flagged by Tukey's fence rule for any selected column.

    A row is flagged if at least one feature falls outside
    ``[Q1 - k*IQR, Q3 + k*IQR]``.
    """
    mask = pd.Series(False, index=df.index)
    for col in cols:
        if col not in df.columns:
            continue
        series = df[col].dropna()
        if series.empty:
            continue
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lo = q1 - k * iqr
        hi = q3 + k * iqr
        mask |= (df[col] < lo) | (df[col] > hi)
    return mask


def filter_complete_pairs(
    df: pd.DataFrame, subject_col: str, feature_cols: Sequence[str]
) -> pd.DataFrame:
    """Keep subjects with both visits and complete selected features."""
    cols = [subject_col, "visit"] + list(feature_cols) + ["FARS", "SARA"]
    if "subject" in df.columns and "subject" not in cols:
        cols.append("subject")
    sub = df[cols].copy()
    sub = sub.dropna(subset=list(feature_cols))
    counts = sub.groupby(subject_col)["visit"].nunique()
    valid = counts[counts == 2].index
    return sub[sub[subject_col].isin(valid)].copy()


__all__ = [
    "_as_float_array",
    "standardize_train_test",
    "missingness_summary",
    "tukey_outliers_mask",
    "filter_complete_pairs",
]
