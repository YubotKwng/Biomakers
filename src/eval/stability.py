"""Model-stability diagnostics for nested TRACK-FA tuning."""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def selected_feature_jaccard(feature_sets: Iterable[Iterable[str]]) -> dict:
    """Return mean pairwise Jaccard stability for selected feature sets."""
    sets = [set(s) for s in feature_sets]
    vals = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            a, b = sets[i], sets[j]
            if not a and not b:
                vals.append(1.0)
            elif not a or not b:
                vals.append(0.0)
            else:
                vals.append(len(a & b) / len(a | b))
    arr = np.asarray(vals, dtype=float)
    return {
        "n_sets": len(sets),
        "mean_jaccard": float(np.mean(arr)) if len(arr) else (1.0 if sets else np.nan),
        "median_jaccard": float(np.median(arr)) if len(arr) else np.nan,
        "iqr_jaccard": float(np.percentile(arr, 75) - np.percentile(arr, 25)) if len(arr) else np.nan,
        "min_jaccard": float(np.min(arr)) if len(arr) else np.nan,
        "max_jaccard": float(np.max(arr)) if len(arr) else np.nan,
    }


def coefficient_sign_stability(coef_samples, feature_names: Sequence[str]) -> pd.DataFrame:
    """Report dominant coefficient sign stability feature-by-feature."""
    coefs = np.asarray(coef_samples, dtype=float)
    if coefs.ndim == 1:
        coefs = coefs.reshape(1, -1)
    names = list(feature_names)
    if coefs.shape[1] != len(names):
        raise ValueError("coef_samples columns must match feature_names")
    rows = []
    for j, name in enumerate(names):
        signs = np.sign(coefs[:, j])
        signs = signs[np.isfinite(signs)]
        if len(signs) == 0:
            rows.append({
                "feature": name,
                "dominant_sign": np.nan,
                "sign_stability": np.nan,
                "positive_frequency": np.nan,
                "negative_frequency": np.nan,
                "zero_frequency": np.nan,
            })
            continue
        pos = float(np.mean(signs > 0))
        neg = float(np.mean(signs < 0))
        zero = float(np.mean(signs == 0))
        freqs = {1: pos, -1: neg, 0: zero}
        dominant = max(freqs, key=freqs.get)
        rows.append({
            "feature": name,
            "dominant_sign": int(dominant),
            "sign_stability": float(freqs[dominant]),
            "positive_frequency": pos,
            "negative_frequency": neg,
            "zero_frequency": zero,
        })
    return pd.DataFrame(rows).sort_values("sign_stability", ascending=False, kind="mergesort").reset_index(drop=True)


def score_ranking_stability(score_samples) -> dict:
    """Compute pairwise Spearman stability of patient score rankings."""
    if isinstance(score_samples, pd.DataFrame):
        mat = score_samples.to_numpy(dtype=float)
    else:
        mat = np.asarray(score_samples, dtype=float)
    if mat.ndim == 1:
        mat = mat.reshape(-1, 1)
    vals = []
    for i in range(mat.shape[1]):
        for j in range(i + 1, mat.shape[1]):
            mask = np.isfinite(mat[:, i]) & np.isfinite(mat[:, j])
            if int(mask.sum()) < 3:
                continue
            rho = spearmanr(mat[mask, i], mat[mask, j]).correlation
            if np.isfinite(rho):
                vals.append(float(rho))
    return {
        "n_rankings": int(mat.shape[1]),
        "n_pairs": int(len(vals)),
        "mean_spearman": float(np.mean(vals)) if vals else np.nan,
        "min_spearman": float(np.min(vals)) if vals else np.nan,
    }


def candidate_stability_summary(candidates: Mapping[str, Mapping]) -> pd.DataFrame:
    """Collect available stability diagnostics for model-selection candidates."""
    rows = []
    for name, payload in candidates.items():
        row = {"candidate": name}
        if "feature_sets" in payload:
            row["jaccard_stability"] = selected_feature_jaccard(payload["feature_sets"])["mean_jaccard"]
        if "coef_samples" in payload and "feature_names" in payload:
            sign_df = coefficient_sign_stability(payload["coef_samples"], payload["feature_names"])
            row["sign_stability"] = float(sign_df["sign_stability"].mean()) if not sign_df.empty else np.nan
        if "score_samples" in payload:
            row["score_ranking_stability"] = score_ranking_stability(payload["score_samples"])["mean_spearman"]
        rows.append(row)
    return pd.DataFrame(rows)


__all__ = [
    "candidate_stability_summary",
    "coefficient_sign_stability",
    "score_ranking_stability",
    "selected_feature_jaccard",
]
