"""Human-readable hyperparameter tuning review tables."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..eval.model_selection import select_hierarchical_candidate


def prepare_tuning_review_table(
    tuning_log: pd.DataFrame,
    *,
    mean_col: str = "mean_validation_annual_dz",
    se_col: str = "se_validation_dz",
) -> pd.DataFrame:
    """Return a sorted table with the columns needed for manual review."""
    if tuning_log is None or tuning_log.empty:
        return pd.DataFrame()
    out = tuning_log.copy()
    if se_col not in out.columns:
        if {"d_ci_low", "d_ci_high"} <= set(out.columns):
            out[se_col] = (
                pd.to_numeric(out["d_ci_high"], errors="coerce")
                - pd.to_numeric(out["d_ci_low"], errors="coerce")
            ) / (2 * 1.96)
        else:
            out[se_col] = np.nan
    if "feature_count" not in out.columns:
        if "param_k" in out.columns:
            out["feature_count"] = out["param_k"]
        elif "n_features" in out.columns:
            out["feature_count"] = out["n_features"]
        else:
            out["feature_count"] = np.nan
    for col in (
        "dz_v1_v2",
        "dz_v2_v3",
        mean_col,
        "annual_interval_gap",
        "p_progression",
        "feature_count",
        "jaccard_stability",
        "sign_stability",
        "coefficient_sign_stability",
        se_col,
    ):
        if col not in out.columns:
            out[col] = np.nan
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["raw_rank"] = out[mean_col].rank(ascending=False, method="min")
    param_cols = [c for c in out.columns if c.startswith("param_")]
    review_cols = [
        *param_cols,
        "dz_v1_v2",
        "dz_v2_v3",
        mean_col,
        "annual_interval_gap",
        "p_progression",
        "feature_count",
        "jaccard_stability",
        "sign_stability",
        "coefficient_sign_stability",
        se_col,
        "raw_rank",
    ]
    keep = [c for c in review_cols if c in out.columns]
    return out.sort_values(
        [mean_col, "annual_interval_gap", "p_progression", "feature_count"],
        ascending=[False, True, False, True],
        kind="mergesort",
    )[keep]


def tuning_recommendation(
    tuning_log: pd.DataFrame,
    *,
    mean_col: str = "mean_validation_annual_dz",
    se_col: str = "se_validation_dz",
) -> dict:
    """Return raw-best, one-SE set, implemented recommendation, and rationale."""
    review = prepare_tuning_review_table(tuning_log, mean_col=mean_col, se_col=se_col)
    if review.empty:
        return {
            "review_table": review,
            "raw_best": pd.Series(dtype=object),
            "near_optimal": review,
            "recommended": pd.Series(dtype=object),
            "summary": "No tuning candidates were available.",
        }
    raw_best = review.iloc[0]
    threshold = float(raw_best[mean_col]) - float(0.0 if pd.isna(raw_best[se_col]) else raw_best[se_col])
    near = review[pd.to_numeric(review[mean_col], errors="coerce") >= threshold].copy()
    recommendation_input = review.copy()
    recommended = select_hierarchical_candidate(recommendation_input, mean_col=mean_col, se_col=se_col)
    delta_perf = float(raw_best[mean_col]) - float(recommended[mean_col])
    reason = (
        f"Recommended candidate is within one SE of the raw best "
        f"(performance difference {delta_perf:.4g}) and is preferred by the implemented hierarchy: "
        f"smaller annual interval gap first, then higher P(delta>0), fewer features, and available stability diagnostics."
    )
    if int(recommended.name) == int(raw_best.name):
        reason = (
            "Recommended candidate is also the raw best by mean annual validation d_z; "
            "the consistency, progression-probability, simplicity, and stability tie-breakers did not select a different row."
        )
    return {
        "review_table": review,
        "raw_best": raw_best,
        "near_optimal": near,
        "recommended": recommended,
        "summary": reason,
    }


def tuning_verification_summary(
    recommendation: dict,
    *,
    mean_col: str = "mean_validation_annual_dz",
) -> pd.DataFrame:
    """Build the concise final human-verification summary table."""
    raw = recommendation.get("raw_best", pd.Series(dtype=object))
    rec = recommendation.get("recommended", pd.Series(dtype=object))
    if raw.empty or rec.empty:
        return pd.DataFrame([{"item": "Warning", "value": "No tuning recommendation available."}])
    param_cols = [c for c in raw.index if str(c).startswith("param_")]
    raw_params = {c.replace("param_", ""): raw[c] for c in param_cols}
    rec_params = {c.replace("param_", ""): rec[c] for c in param_cols}
    diff = float(raw.get(mean_col, np.nan)) - float(rec.get(mean_col, np.nan))
    warning = ""
    if pd.notna(rec.get("annual_interval_gap", np.nan)) and rec.get("annual_interval_gap", 0) > 0.5:
        warning = "Annual d12/d23 gap is large; inspect temporal consistency before accepting."
    return pd.DataFrame(
        [
            {"item": "Best raw-performance parameters", "value": raw_params},
            {"item": "Recommended parameters", "value": rec_params},
            {"item": "Difference in performance", "value": diff},
            {"item": "Reason for recommendation", "value": recommendation.get("summary", "")},
            {"item": "Any instability/warning", "value": warning or "No automatic warning."},
        ]
    )


def plot_tuning_review(
    review_table: pd.DataFrame,
    *,
    title: str = "Tuning Review",
    mean_col: str = "mean_validation_annual_dz",
):
    """Plot annual performance, interval replication, and feature count."""
    import matplotlib.pyplot as plt

    if review_table is None or review_table.empty:
        return None
    table = review_table.reset_index(drop=True).copy()
    labels = []
    param_cols = [c for c in table.columns if str(c).startswith("param_")]
    for _, row in table.iterrows():
        bits = []
        for col in param_cols[:3]:
            val = row.get(col)
            bits.append(f"{col.replace('param_', '')}={val}")
        labels.append("\n".join(bits) if bits else str(len(labels) + 1))
    x = np.arange(len(table))
    fig, axes = plt.subplots(1, 3, figsize=(max(12, len(table) * 0.75), 4))
    axes[0].plot(x, table[mean_col].astype(float), marker="o")
    axes[0].set_title("Mean Annual d_z")
    axes[0].set_ylabel("Validation d_z")
    width = 0.38
    axes[1].bar(x - width / 2, table["dz_v1_v2"].astype(float), width=width, label="V1->V2")
    axes[1].bar(x + width / 2, table["dz_v2_v3"].astype(float), width=width, label="V2->V3")
    axes[1].set_title("Annual Intervals")
    axes[1].legend()
    axes[2].scatter(table["feature_count"].astype(float), table[mean_col].astype(float))
    axes[2].set_title("Performance vs Features")
    axes[2].set_xlabel("Selected features")
    axes[2].set_ylabel("Mean annual d_z")
    for ax in axes[:2]:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


__all__ = [
    "plot_tuning_review",
    "prepare_tuning_review_table",
    "tuning_recommendation",
    "tuning_verification_summary",
]
