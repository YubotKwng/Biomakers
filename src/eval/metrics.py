"""Canonical metric implementations.

Background
----------
Drift findings (verified numerically on a fixed-seed synthetic fixture in
``tests/test_metrics.py``):

* All three implementations agree on the value of *d* (they share the same
  mean / SD-with-ddof=1 formula). The differences are purely structural:

  - ``compute_paired_d`` works on a long-format dataframe and returns
    ``(d, mean, sd, n)``.
  - ``compute_cohens_d`` works on the already-paired delta vector and
    returns a dict including ``n``.
  - ``compute_cohens_d_from_oof`` is the "scalar-only" variant of
    ``compute_paired_d`` and uses ``pred`` rather than ``score`` as the
    default value column.

* All three return NaN when fewer than 2 paired subjects are available.
* ``compute_cohens_d`` additionally drops non-finite deltas before counting,
  which the other two do not. This is documented behaviour, not a bug.

The canonical ``paired_cohens_d`` is bit-exact with
the paired progression effect-size calculation used by the modelling pipeline.
"""
from __future__ import annotations

from typing import Iterable, Mapping, Tuple

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from sklearn.metrics import mean_squared_error


# ---------------------------------------------------------------------------
# Canonical Cohen's d (paired, from a long-format OOF dataframe)
# ---------------------------------------------------------------------------
def paired_cohens_d(
    oof_df: pd.DataFrame,
    subject_col: str,
    visit_col: str = "visit",
    score_col: str = "score",
) -> Tuple[float, float, float, int]:
    """Paired Cohen's d, equivalent to SRM, on long-format OOF predictions.

    Returns
    -------
    (d, mean_diff, sd_diff, n_pairs) : tuple
        ``d`` is ``mean(diff) / sd(diff, ddof=1)``; NaN if SD is zero or
        fewer than two paired subjects are available.
    """
    # Sort within each subject before differencing so visit2 - visit1 is
    # applied consistently even if the OOF rows arrive in fold order.
    paired = (oof_df.sort_values([subject_col, visit_col])
                .groupby(subject_col)[score_col]
                .apply(list))
    paired = paired[paired.map(len) == 2]
    if len(paired) < 2:
        return np.nan, np.nan, np.nan, 0
    diffs = paired.map(lambda x: x[1] - x[0]).astype(float)
    sd = diffs.std(ddof=1)
    d = np.nan if sd == 0 else diffs.mean() / sd
    return d, diffs.mean(), sd, len(diffs)


# Compatibility alias for callers using the concise metric name.
compute_paired_d = paired_cohens_d


# ---------------------------------------------------------------------------
# SRM (paired) — numerically identical to paired Cohen's d.
# ---------------------------------------------------------------------------
def srm(
    oof_df: pd.DataFrame,
    subject_col: str,
    visit_col: str = "visit",
    score_col: str = "score",
) -> float:
    """Standardised Response Mean — equal to ``paired_cohens_d`` for paired
    measurements."""
    d, _, _, _ = paired_cohens_d(oof_df, subject_col, visit_col, score_col)
    return d


# ---------------------------------------------------------------------------
# Paired t-test (visit-2 vs visit-1)
# ---------------------------------------------------------------------------
def paired_ttest(
    oof_df: pd.DataFrame,
    subject_col: str,
    visit_col: str = "visit",
    score_col: str = "score",
) -> float:
    """Two-sided paired t-test p-value (visit-2 minus visit-1).

    Returns the paired t-test statistic and p-value for visit-level scores.
    """
    paired = (oof_df.sort_values([subject_col, visit_col])
                .groupby(subject_col)[score_col]
                .apply(list))
    paired = paired[paired.map(len) == 2]
    if len(paired) < 2:
        return np.nan
    v1 = paired.map(lambda x: x[0]).astype(float)
    v2 = paired.map(lambda x: x[1]).astype(float)
    return ttest_rel(v2, v1).pvalue


# ---------------------------------------------------------------------------
# Delta-vector metric helpers.
# ---------------------------------------------------------------------------
def _as_float_array(x) -> np.ndarray:
    """Coerce input to a one-dimensional float array."""
    return np.asarray(x, dtype=float)


def paired_deltas_from_long(
    oof_df: pd.DataFrame,
    subject_col: str,
    visit_col: str,
    value_col: str,
) -> pd.Series:
    """Convert long-format OOF rows into a per-subject visit-2 minus visit-1
    delta series."""
    required = [subject_col, visit_col, value_col]
    if oof_df is None or any(c not in oof_df.columns for c in required):
        return pd.Series(dtype=float)
    tmp = oof_df[required].dropna().copy()
    if tmp.empty:
        return pd.Series(dtype=float)
    # The modelling notebooks report progression as follow-up minus baseline.
    # Keeping this helper central avoids sign drift between model families.
    paired = (tmp.sort_values([subject_col, visit_col])
                .groupby(subject_col)[value_col]
                .apply(list))
    paired = paired[paired.map(len) == 2]
    if len(paired) == 0:
        return pd.Series(dtype=float)
    deltas = paired.map(lambda x: float(x[1] - x[0]))
    return deltas


def compute_cohens_d(deltas) -> dict:
    """Return Cohen's d on an already-paired delta vector."""
    deltas = _as_float_array(deltas)
    deltas = deltas[np.isfinite(deltas)]
    n = int(len(deltas))
    if n < 2:
        return {"d": np.nan, "n": n, "mean": np.nan, "sd": np.nan}
    mean = float(np.mean(deltas))
    sd = float(np.std(deltas, ddof=1))
    d = np.nan if sd == 0 else float(mean / sd)
    return {"d": d, "n": n, "mean": mean, "sd": sd}


def compute_srm(deltas) -> dict:
    """SRM on a delta vector — same formula as Cohen's d for paired data."""
    out = compute_cohens_d(deltas)
    return {"srm": out["d"], "n": out["n"], "mean": out["mean"], "sd": out["sd"]}


def bootstrap_ci_d(
    oof_df: pd.DataFrame,
    subject_col: str,
    visit_col: str,
    value_col: str,
    n_boot: int = 1000,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Bootstrap mean / 2.5% / 97.5% percentile of Cohen's d.

    Deterministic for a fixed bootstrap seed.
    """
    rs = np.random.RandomState(seed)
    deltas = paired_deltas_from_long(oof_df, subject_col, visit_col, value_col)
    if deltas.empty or deltas.shape[0] < 2:
        return (np.nan, np.nan, np.nan)
    subjects = deltas.index.to_numpy()
    vals = deltas.to_numpy(dtype=float)
    boot = []
    for _ in range(int(n_boot)):
        idx = rs.randint(0, len(subjects), size=len(subjects))
        sample = vals[idx]
        out = compute_cohens_d(sample)
        boot.append(out["d"])
    boot = np.asarray(boot, dtype=float)
    boot = boot[np.isfinite(boot)]
    if len(boot) < 10:
        return (float(np.mean(boot)) if len(boot) else np.nan, np.nan, np.nan)
    return (
        float(np.mean(boot)),
        float(np.percentile(boot, 2.5)),
        float(np.percentile(boot, 97.5)),
    )


def _infer_pair_type(df: pd.DataFrame) -> pd.Series:
    """Infer adjacent clinical interval labels such as V1V2 and V2V3."""
    if "pair" in df.columns:
        pair = df["pair"].astype(str).str.upper()
        hit = pair.str.extract(r"(V\d+V\d+)", expand=False)
        return hit
    for col in ("patient_id", "pair_id", "ID", "subject_id"):
        if col in df.columns:
            hit = df[col].astype(str).str.upper().str.extract(r"(V\d+V\d+)", expand=False)
            if hit.notna().any():
                return hit
    return pd.Series(np.nan, index=df.index, dtype=object)


def _clinical_delta_col(scale: str) -> str | None:
    key = str(scale).strip().lower()
    mapping = {
        "fars": "delta_mfars_total",
        "mfars": "delta_mfars_total",
        "mfars_total": "delta_mfars_total",
        "sara": "delta_sara_total",
        "sara_total": "delta_sara_total",
        "adl": "delta_adl_total",
        "adl_total": "delta_adl_total",
        "lcslc": "delta_lcslc_total",
        "lcslc_total": "delta_lcslc_total",
    }
    return mapping.get(key)


def clinical_change_effect_sizes(
    pairs_df: pd.DataFrame,
    scale_cols: Iterable[str] = ("FARS", "SARA"),
    *,
    pair_types: Iterable[str] = ("V1V2", "V2V3"),
    scale_to_delta_col: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Clinical benchmark d from adjacent visit changes only.

    This uses explicit change columns from the paired TRACK-FA table, e.g.
    ``delta_mfars_total`` and ``delta_sara_total``. Rows are restricted to
    adjacent intervals such as ``V1V2`` and ``V2V3``, so the benchmark is
    computed from FARS2-FARS1 and FARS3-FARS2, never from non-adjacent or
    visit-level raw values.
    """
    if pairs_df is None or pairs_df.empty:
        return pd.DataFrame(columns=[
            "feature", "kind", "d", "mean_diff", "sd_diff", "n_pairs",
            "source_delta_col", "pair_types",
        ])

    # Clinical benchmarks come from already-computed adjacent deltas in the
    # paired table, not from raw visit-level clinical values.
    pair_type = _infer_pair_type(pairs_df)
    allowed = {str(p).upper() for p in pair_types}
    pair_mask = pair_type.isin(allowed)

    scale_to_delta_col = dict(scale_to_delta_col or {})
    rows = []
    for scale in list(scale_cols):
        delta_col = scale_to_delta_col.get(scale) or _clinical_delta_col(scale)
        if delta_col is None or delta_col not in pairs_df.columns:
            continue
        deltas = pd.to_numeric(pairs_df.loc[pair_mask, delta_col], errors="coerce").dropna()
        out = compute_cohens_d(deltas.to_numpy(dtype=float))
        rows.append({
            "feature": scale,
            "kind": "scale",
            "d": out["d"],
            "mean_diff": out["mean"],
            "sd_diff": out["sd"],
            "n_pairs": out["n"],
            "source_delta_col": delta_col,
            "pair_types": ",".join(sorted(allowed)),
        })

    out_df = pd.DataFrame(rows)
    if not out_df.empty:
        out_df = out_df.reindex(out_df["d"].abs().sort_values(ascending=False).index)
        out_df = out_df.reset_index(drop=True)
    return out_df


# ---------------------------------------------------------------------------
# Long-format out-of-fold paired effect size helper.
# ---------------------------------------------------------------------------
def compute_cohens_d_from_oof(
    oof_df: pd.DataFrame,
    subject_col: str = "subject",
    pred_col: str = "pred",
    visit_col: str = "visit",
) -> float:
    """Return scalar paired Cohen's d from held-out visit predictions."""
    paired = (oof_df.sort_values([subject_col, visit_col])
              .groupby(subject_col)[pred_col]
              .apply(list))
    paired = paired[paired.map(len) == 2]
    if len(paired) < 2:
        return np.nan
    diffs = paired.map(lambda x: x[1] - x[0]).astype(float).to_numpy()
    diffs = diffs[np.isfinite(diffs)]
    if len(diffs) < 2:
        return np.nan
    sd = float(np.std(diffs, ddof=1))
    if sd == 0:
        return np.nan
    return float(np.mean(diffs) / sd)


# ---------------------------------------------------------------------------
# Pointwise regression metrics.
# ---------------------------------------------------------------------------
def rmse(y_true, y_pred) -> float:
    """Return root mean squared error while ignoring non-finite entries."""
    y_true = _as_float_array(y_true)
    y_pred = _as_float_array(y_pred)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() == 0:
        return np.nan
    return float(np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2)))


def r2(y_true, y_pred) -> float:
    """Coefficient of determination, ignoring non-finite entries.

    Compute R-squared from explicit true and predicted values.
    """
    y_true = _as_float_array(y_true)
    y_pred = _as_float_array(y_pred)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 2:
        return np.nan
    yt = y_true[mask]
    yp = y_pred[mask]
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
    return np.nan if ss_tot == 0 else float(1.0 - ss_res / ss_tot)


# ---------------------------------------------------------------------------
# Per-feature reference effect sizes (interaction-term benchmarking)
# ---------------------------------------------------------------------------
def reference_effect_sizes(
    long_df: pd.DataFrame,
    imaging_cols: Iterable[str],
    scale_cols: Iterable[str] = ("FARS", "SARA"),
    subject_col: str = "ID",
    visit_col: str = "visit",
) -> pd.DataFrame:
    """Per-feature paired Cohen's d for benchmarking the composite.

    Compute benchmark progression effect sizes for candidate biomarkers.
    Computes the paired ``mean / sd(ddof=1)`` effect size for every imaging
    feature and clinical scale supplied, returning one row per feature
    sorted by ``|d|`` descending. These benchmarks anchor the composite
    against the strongest single imaging feature and clinical FARS/SARA
    progression signals.

    Parameters
    ----------
    long_df : pd.DataFrame
        Long-format table with one row per (subject, visit).
    imaging_cols : iterable of str
        Imaging feature column names.
    scale_cols : iterable of str
        Clinical scales (e.g. FARS, SARA) to also benchmark.
    subject_col, visit_col : str
        Column names.

    Returns
    -------
    pd.DataFrame
        Columns: ``feature, kind, d, mean_diff, sd_diff, n_pairs``.
        Sorted by ``|d|`` descending. ``kind`` ∈ {"imaging", "scale"}.
    """
    rows = []
    for c in list(imaging_cols):
        if c not in long_df.columns:
            continue
        d, m, s, n = paired_cohens_d(
            long_df.dropna(subset=[c]),
            subject_col=subject_col,
            visit_col=visit_col,
            score_col=c,
        )
        rows.append({"feature": c, "kind": "imaging",
                     "d": d, "mean_diff": m, "sd_diff": s, "n_pairs": n})
    for c in list(scale_cols):
        if c not in long_df.columns:
            continue
        d, m, s, n = paired_cohens_d(
            long_df.dropna(subset=[c]),
            subject_col=subject_col,
            visit_col=visit_col,
            score_col=c,
        )
        rows.append({"feature": c, "kind": "scale",
                     "d": d, "mean_diff": m, "sd_diff": s, "n_pairs": n})
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.reindex(out["d"].abs().sort_values(ascending=False).index)
        out = out.reset_index(drop=True)
    return out


__all__ = [
    "paired_cohens_d",
    "compute_paired_d",
    "srm",
    "rmse",
    "r2",
    "paired_ttest",
    "compute_cohens_d",
    "compute_srm",
    "paired_deltas_from_long",
    "bootstrap_ci_d",
    "clinical_change_effect_sizes",
    "compute_cohens_d_from_oof",
    "reference_effect_sizes",
    "_as_float_array",
]
