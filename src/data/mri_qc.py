"""MRI quality-control helpers for notebook-first TRACK-FA audits."""
from __future__ import annotations

from collections.abc import Iterable, Sequence
import os
import tempfile

import numpy as np
import pandas as pd
from scipy import stats


def _present_columns(df: pd.DataFrame, candidates: Iterable[str]) -> list[str]:
    return [c for c in candidates if c in df.columns]


def _design_matrix(df: pd.DataFrame, terms: Sequence[str]) -> pd.DataFrame:
    parts = [pd.Series(1.0, index=df.index, name="Intercept")]
    for term in terms:
        if term not in df.columns:
            continue
        s = df[term]
        if pd.api.types.is_numeric_dtype(s):
            parts.append(pd.to_numeric(s, errors="coerce").rename(term))
        else:
            dummies = pd.get_dummies(s.astype("category"), prefix=term, drop_first=True, dtype=float)
            if not dummies.empty:
                parts.append(dummies)
    if len(parts) == 1:
        return pd.concat(parts, axis=1)
    return pd.concat(parts, axis=1)


def _rss(y: np.ndarray, X: np.ndarray) -> tuple[float, int]:
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    return float(np.sum(resid ** 2)), int(np.linalg.matrix_rank(X))


def site_effect_screen(
    df: pd.DataFrame,
    features: Iterable[str],
    *,
    site_col: str = "site",
    covariates: Sequence[str] | None = None,
    disease_severity_candidates: Sequence[str] = ("mfars_total", "FARS", "sara_total", "SARA"),
) -> pd.DataFrame:
    """Screen each MRI feature for residual site association.

    Fits a reduced model ``Feature ~ Age + Sex + DiseaseSeverity`` and a full
    model adding ``Site``. Parameters are estimated on whichever dataframe the
    caller supplies, so use this on training subjects inside CV when it informs
    harmonisation/model decisions.
    """
    if site_col not in df.columns:
        raise KeyError(f"df missing required column: {site_col}")

    feature_list = [f for f in features if f in df.columns]
    if covariates is None:
        severity = _present_columns(df, disease_severity_candidates)[:1]
        covariates = [*_present_columns(df, ("age", "Age")), *_present_columns(df, ("sex", "gender", "Sex")), *severity]
    covariates = list(dict.fromkeys(covariates))

    rows = []
    for feature in feature_list:
        cols = [feature, site_col, *covariates]
        tmp = df[cols].replace([np.inf, -np.inf], np.nan).dropna(subset=[feature, site_col]).copy()
        if tmp[site_col].nunique(dropna=True) < 2 or len(tmp) < 5:
            rows.append({
                "feature": feature,
                "n": int(len(tmp)),
                "site_levels": int(tmp[site_col].nunique(dropna=True)),
                "site_r2_delta": np.nan,
                "site_f": np.nan,
                "site_p_value": np.nan,
                "covariates_used": ",".join(covariates),
            })
            continue

        design_cols = [feature, site_col, *covariates]
        tmp = tmp[design_cols].dropna()
        y = pd.to_numeric(tmp[feature], errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(y)
        tmp = tmp.loc[ok].copy()
        y = y[ok]
        if len(y) < 5:
            continue
        X_reduced = _design_matrix(tmp, covariates).to_numpy(dtype=float)
        X_full = _design_matrix(tmp, [*covariates, site_col]).to_numpy(dtype=float)
        rss_reduced, rank_reduced = _rss(y, X_reduced)
        rss_full, rank_full = _rss(y, X_full)
        total = float(np.sum((y - np.mean(y)) ** 2))
        df_num = max(rank_full - rank_reduced, 0)
        df_den = max(len(y) - rank_full, 0)
        if df_num == 0 or df_den == 0 or rss_full <= 0:
            f_stat = np.nan
            p_value = np.nan
        else:
            f_stat = ((rss_reduced - rss_full) / df_num) / (rss_full / df_den)
            p_value = float(stats.f.sf(f_stat, df_num, df_den))
        rows.append({
            "feature": feature,
            "n": int(len(y)),
            "site_levels": int(tmp[site_col].nunique(dropna=True)),
            "site_r2_delta": float((rss_reduced - rss_full) / total) if total > 0 else np.nan,
            "site_f": float(f_stat) if np.isfinite(f_stat) else np.nan,
            "site_p_value": p_value,
            "covariates_used": ",".join(covariates),
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["site_p_value", "site_r2_delta"], ascending=[True, False], kind="mergesort")
    return out.reset_index(drop=True)


def mri_outlier_table(
    df: pd.DataFrame,
    features: Iterable[str],
    *,
    group_cols: Sequence[str] = ("visit", "site"),
    iqr_multiplier: float = 3.0,
) -> pd.DataFrame:
    """Flag feature values outside group-specific IQR fences for review."""
    feature_list = [f for f in features if f in df.columns]
    groups = [c for c in group_cols if c in df.columns]
    rows = []
    iterable = df.groupby(groups, dropna=False) if groups else [((), df)]
    for key, group in iterable:
        key_vals = key if isinstance(key, tuple) else (key,)
        key_map = dict(zip(groups, key_vals))
        for feature in feature_list:
            vals = pd.to_numeric(group[feature], errors="coerce")
            vals = vals[np.isfinite(vals)]
            if len(vals) < 4:
                continue
            q1, q3 = np.percentile(vals, [25, 75])
            iqr = q3 - q1
            lower = q1 - iqr_multiplier * iqr
            upper = q3 + iqr_multiplier * iqr
            flagged = vals[(vals < lower) | (vals > upper)]
            if len(flagged):
                rows.append({
                    **key_map,
                    "feature": feature,
                    "n": int(len(vals)),
                    "n_outliers": int(len(flagged)),
                    "outlier_pct": float(len(flagged) / len(vals)),
                    "lower_fence": float(lower),
                    "upper_fence": float(upper),
                    "min_flagged": float(flagged.min()),
                    "max_flagged": float(flagged.max()),
                })
    columns = [
        *groups,
        "feature",
        "n",
        "n_outliers",
        "outlier_pct",
        "lower_fence",
        "upper_fence",
        "min_flagged",
        "max_flagged",
    ]
    out = pd.DataFrame(rows, columns=columns)
    if out.empty:
        return out
    return out.sort_values("n_outliers", ascending=False, kind="mergesort").reset_index(drop=True)


def plot_feature_distributions_by_site(
    df: pd.DataFrame,
    features: Iterable[str],
    *,
    site_col: str = "site",
    max_features: int = 12,
):
    """Build box/strip distribution plots for MRI features by site."""
    cache_dir = os.path.join(tempfile.gettempdir(), "biomarkers-matplotlib-cache")
    fontconfig_dir = os.path.join(cache_dir, "fontconfig")
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(fontconfig_dir, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", cache_dir)
    os.environ.setdefault("XDG_CACHE_HOME", cache_dir)
    os.environ.setdefault("FONTCONFIG_PATH", fontconfig_dir)
    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib.pyplot as plt

    feature_list = [f for f in features if f in df.columns][: int(max_features)]
    if site_col not in df.columns or not feature_list:
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.axis("off")
        ax.text(0.01, 0.5, "No site column or MRI features available for QC distribution plot.")
        return fig

    n = len(feature_list)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.2 * nrows), squeeze=False)
    for ax, feature in zip(axes.ravel(), feature_list):
        tmp = df[[site_col, feature]].replace([np.inf, -np.inf], np.nan).dropna()
        if tmp.empty:
            ax.axis("off")
            ax.set_title(feature)
            continue
        sites = sorted(tmp[site_col].dropna().astype(str).unique())
        values = [
            pd.to_numeric(tmp.loc[tmp[site_col].astype(str) == site, feature], errors="coerce").dropna().to_numpy(dtype=float)
            for site in sites
        ]
        positions = np.arange(1, len(sites) + 1)
        ax.boxplot(
            values,
            positions=positions,
            widths=0.55,
            patch_artist=True,
            showfliers=False,
            boxprops={"facecolor": "#d8e2dc", "edgecolor": "#52796f"},
            medianprops={"color": "#1b4332", "linewidth": 1.5},
            whiskerprops={"color": "#52796f"},
            capprops={"color": "#52796f"},
        )
        rng = np.random.default_rng(42)
        for pos, vals in zip(positions, values):
            if len(vals) == 0:
                continue
            jitter = rng.normal(0, 0.045, size=len(vals))
            ax.scatter(np.full(len(vals), pos) + jitter, vals, s=8, alpha=0.45, color="#264653", linewidths=0)
        ax.set_xticks(positions)
        ax.set_xticklabels(sites)
        ax.set_xlabel(site_col)
        ax.set_ylabel(feature)
        ax.set_title(feature)
        ax.tick_params(axis="x", rotation=35)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.tight_layout()
    return fig


def harmonisation_leakage_policy() -> pd.DataFrame:
    """Document the fold-specific harmonisation guardrails for notebooks."""
    return pd.DataFrame(
        [
            {
                "rule": "Estimate harmonisation parameters inside each outer training fold only.",
                "reason": "Using held-out subjects to estimate site/scanner adjustment leaks test distribution information.",
            },
            {
                "rule": "Apply learned parameters unchanged to validation/test subjects.",
                "reason": "The outer test fold must remain untouched until score generation.",
            },
            {
                "rule": "Preserve within-subject longitudinal change.",
                "reason": "The biomarker objective is progression sensitivity; harmonisation must not remove true V1->V3 change.",
            },
            {
                "rule": "Report pre/post site-effect diagnostics if ComBat or longitudinal ComBat is enabled.",
                "reason": "A site correction is only defensible if it reduces nuisance site signal without flattening progression.",
            },
        ]
    )


__all__ = [
    "harmonisation_leakage_policy",
    "mri_outlier_table",
    "plot_feature_distributions_by_site",
    "site_effect_screen",
]
