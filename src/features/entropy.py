"""Mutual-information feature ranking

Discretization-based MI: features (and continuous targets) are binned
into deciles via ``pd.qcut`` (with fallbacks for low-cardinality inputs);
binary labels are kept as integers. Mutual information is computed from
the empirical joint contingency table.

Functions
---------
* ``_discretize_series(x, n_bins=10)`` — quantile binning with fallbacks.
* ``mutual_information_discrete(x_disc, y_disc)`` — MI from a contingency
  table, returns ``(mi, n_pairs)``.
* ``mi_feature_vs_binary_label(x, y, is_discrete=False, n_bins=10)``
* ``mi_feature_vs_continuous_target(x, y, is_discrete_x=False, n_bins=10)``
* ``rank_features_by_mi(df, features, y, mi_kind, feature_groups=None,
  n_bins=10)`` — returns a per-feature MI ranking dataframe.

"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _discretize_series(x, n_bins=10):
    """Quantile-bin ``x`` with deterministic fallbacks for sparse values."""
    x = pd.Series(x).astype(float)
    x = x.replace([np.inf, -np.inf], np.nan)
    x = x.dropna()
    if x.empty:
        return None
    # Use quantile bins with fallbacks
    try:
        bins = pd.qcut(x, q=min(n_bins, x.nunique()), duplicates="drop")
        return bins
    except Exception:
        # fallback to uniform bins
        bins = pd.cut(x, bins=min(n_bins, x.nunique()))
        return bins


def mutual_information_discrete(x_disc, y_disc):
    """MI of two discrete-valued vectors, contingency-table based.

    Returns the mutual information estimate and the number of paired values.
    """
    x = pd.Series(x_disc)
    y = pd.Series(y_disc)
    df = pd.DataFrame({"x": x, "y": y}).dropna()
    if df.empty:
        return np.nan, 0
    ct = pd.crosstab(df["x"], df["y"])
    n = ct.to_numpy().sum()
    if n == 0:
        return np.nan, 0
    pxy = ct.to_numpy(dtype=float) / n
    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(pxy > 0, pxy / (px @ py), 1.0)
        mi = np.where(pxy > 0, pxy * np.log(ratio), 0.0).sum()
    return float(mi), int(n)


def mi_feature_vs_binary_label(x, y, is_discrete=False, n_bins=10):
    """Estimate MI between a feature and a binary label."""
    y = pd.Series(y).astype(float)
    y = y.replace([np.inf, -np.inf], np.nan)
    mask = np.isfinite(y)
    y = y[mask]
    x = pd.Series(x).iloc[mask.to_numpy()]

    df = pd.DataFrame({"x": x, "y": y}).dropna()
    if df.shape[0] < 3:
        return np.nan, int(df.shape[0])

    if is_discrete:
        x_disc = df["x"].astype(str)
    else:
        bins = _discretize_series(df["x"], n_bins=n_bins)
        if bins is None:
            return np.nan, int(df.shape[0])
        x_disc = bins

    y_disc = df["y"].astype(int)
    return mutual_information_discrete(x_disc, y_disc)


def mi_feature_vs_continuous_target(x, y, is_discrete_x=False, n_bins=10):
    """Estimate MI between a feature and a quantile-binned target."""
    y = pd.Series(y).astype(float)
    y = y.replace([np.inf, -np.inf], np.nan)
    mask = np.isfinite(y)
    y = y[mask]
    x = pd.Series(x).iloc[mask.to_numpy()]

    df = pd.DataFrame({"x": x, "y": y}).dropna()
    if df.shape[0] < 3:
        return np.nan, int(df.shape[0])

    if is_discrete_x:
        x_disc = df["x"].astype(str)
    else:
        x_disc = _discretize_series(df["x"], n_bins=n_bins)
        if x_disc is None:
            return np.nan, int(df.shape[0])

    y_disc = _discretize_series(df["y"], n_bins=n_bins)
    if y_disc is None:
        return np.nan, int(df.shape[0])

    return mutual_information_discrete(x_disc, y_disc)


def rank_features_by_mi(df, features, y, mi_kind, feature_groups=None, n_bins=10):
    """Return a per-feature MI ranking dataframe.

    ``mi_kind='mi_visit'`` uses the binary visit-label path; other values use
    the continuous-target path. The ``sex`` feature is treated as discrete.
    """
    rows = []
    for f in features:
        is_discrete = (f == "sex")
        x = df[f] if f in df.columns else pd.Series([np.nan] * len(df))
        if mi_kind == "mi_visit":
            mi, n = mi_feature_vs_binary_label(x, y, is_discrete=is_discrete, n_bins=n_bins)
        else:
            mi, n = mi_feature_vs_continuous_target(x, y, is_discrete_x=is_discrete, n_bins=n_bins)
        rows.append({"feature": f, "mi": mi, "n": n})

    out = pd.DataFrame(rows)
    out["mi"] = out["mi"].fillna(-np.inf)
    out = out.sort_values("mi", ascending=False)
    if feature_groups is not None:
        out["group"] = out["feature"].map(feature_groups)
    return out


__all__ = [
    "_discretize_series",
    "mutual_information_discrete",
    "mi_feature_vs_binary_label",
    "mi_feature_vs_continuous_target",
    "rank_features_by_mi",
]
