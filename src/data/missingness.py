"""Missingness audits for longitudinal feature tables."""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .audit import normalise_visit_label


def _percent_missing(frame: pd.DataFrame, features: list[str]) -> pd.Series:
    if frame.empty:
        return pd.Series({f: np.nan for f in features}, dtype=float)
    return frame[features].isna().mean().astype(float)


def _concentration_flag(values: pd.Series, *, spread_threshold: float) -> bool:
    vals = values.dropna().astype(float)
    if vals.empty:
        return False
    return bool(vals.max() - vals.min() >= spread_threshold)


def feature_missingness_report(
    df: pd.DataFrame,
    features: Iterable[str],
    by: tuple[str, ...] = ("visit", "site"),
    *,
    concentration_spread: float = 0.25,
) -> dict[str, pd.DataFrame]:
    """Report feature missingness globally and by requested grouping columns.

    Features are never deleted by this helper. Concentration flags simply mark
    features whose missingness varies sharply across visit or site strata.
    """
    feature_list = [f for f in features if f in df.columns]
    missing_features = [f for f in features if f not in df.columns]
    if not feature_list:
        empty = pd.DataFrame(columns=["feature", "missing_pct"])
        return {
            "global": empty,
            "by_visit": empty,
            "by_site": empty,
            "by_visit_site": empty,
            "flags": pd.DataFrame(columns=["feature", "reason"]),
            "missing_features": pd.DataFrame({"feature": missing_features}),
        }

    global_df = (
        _percent_missing(df, feature_list)
        .rename("missing_pct")
        .reset_index()
        .rename(columns={"index": "feature"})
        .sort_values("missing_pct", ascending=False, kind="mergesort")
        .reset_index(drop=True)
    )

    outputs: dict[str, pd.DataFrame] = {"global": global_df}
    grouped_tables: dict[str, pd.DataFrame] = {}
    group_specs = {
        "by_visit": ("visit",),
        "by_site": ("site",),
        "by_visit_site": ("visit", "site"),
    }
    for name, cols in group_specs.items():
        if not set(cols).issubset(df.columns) or not set(cols).issubset(set(by) | set(cols)):
            outputs[name] = pd.DataFrame()
            continue
        tmp = df.copy()
        if "visit" in cols:
            tmp["visit"] = tmp["visit"].map(normalise_visit_label)
        table = (
            tmp.groupby(list(cols), dropna=False)[feature_list]
            .apply(lambda g: g.isna().mean())
            .reset_index()
            .melt(id_vars=list(cols), var_name="feature", value_name="missing_pct")
            .sort_values(["feature", *cols], kind="mergesort")
            .reset_index(drop=True)
        )
        outputs[name] = table
        grouped_tables[name] = table

    flags: list[dict[str, Any]] = []
    for table_name, table in grouped_tables.items():
        if table.empty:
            continue
        for feature, rows in table.groupby("feature"):
            if _concentration_flag(rows["missing_pct"], spread_threshold=concentration_spread):
                flags.append({
                    "feature": feature,
                    "reason": f"missingness spread >= {concentration_spread:.2f} in {table_name}",
                    "max_missing_pct": float(rows["missing_pct"].max()),
                    "min_missing_pct": float(rows["missing_pct"].min()),
                })
    outputs["flags"] = pd.DataFrame(flags).drop_duplicates().reset_index(drop=True)
    outputs["missing_features"] = pd.DataFrame({"feature": missing_features})
    return outputs


def followup_missingness_analysis(
    df: pd.DataFrame,
    *,
    subject_col: str,
    visit_col: str = "visit",
    variables: Iterable[str] = (
        "mfars_total",
        "sara_total",
        "gaa_1",
        "gaa_2",
        "age",
        "onset_age",
        "disease_duration",
        "site",
        "gender",
        "sex",
    ),
    baseline_visit: str = "V1",
    followup_visit: str = "V3",
    fit_logistic: bool = False,
) -> dict[str, Any]:
    """Compare complete V1-to-V3 subjects with subjects missing V3."""
    required = {subject_col, visit_col}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"df missing required columns: {sorted(missing)}")

    tmp = df.copy()
    tmp["_visit_label"] = tmp[visit_col].map(normalise_visit_label)
    baseline = tmp[tmp["_visit_label"] == baseline_visit].drop_duplicates(subject_col)
    have_followup = set(tmp.loc[tmp["_visit_label"] == followup_visit, subject_col])
    baseline = baseline.copy()
    baseline["has_followup"] = baseline[subject_col].isin(have_followup)
    vars_present = [v for v in variables if v in baseline.columns]

    rows: list[dict[str, Any]] = []
    for var in vars_present:
        s = baseline[var]
        if pd.api.types.is_numeric_dtype(s):
            for label, grp in baseline.groupby("has_followup"):
                vals = pd.to_numeric(grp[var], errors="coerce").dropna()
                rows.append({
                    "variable": var,
                    "group": "complete_v1_v3" if label else "missing_v3",
                    "n": int(vals.shape[0]),
                    "mean": float(vals.mean()) if len(vals) else np.nan,
                    "sd": float(vals.std(ddof=1)) if len(vals) > 1 else np.nan,
                    "missing_pct": float(grp[var].isna().mean()),
                })
        else:
            counts = baseline.groupby(["has_followup", var], dropna=False).size().reset_index(name="n")
            for _, row in counts.iterrows():
                rows.append({
                    "variable": var,
                    "group": "complete_v1_v3" if bool(row["has_followup"]) else "missing_v3",
                    "level": row[var],
                    "n": int(row["n"]),
                })

    logistic_summary: dict[str, Any] | None = None
    if fit_logistic:
        numeric = [v for v in vars_present if pd.api.types.is_numeric_dtype(baseline[v])]
        model_df = baseline[[*numeric, "has_followup"]].dropna()
        if len(numeric) and model_df["has_followup"].nunique() == 2 and len(model_df) >= 8:
            model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
            X = model_df[numeric]
            y = model_df["has_followup"].astype(int)
            model.fit(X, y)
            logistic_summary = {
                "n": int(len(model_df)),
                "variables": numeric,
                "training_accuracy": float(model.score(X, y)),
            }

    return {
        "n_baseline_subjects": int(baseline[subject_col].nunique()),
        "n_complete_v1_v3": int(baseline["has_followup"].sum()),
        "n_missing_v3": int((~baseline["has_followup"]).sum()),
        "comparison": pd.DataFrame(rows),
        "logistic": logistic_summary,
    }


__all__ = ["feature_missingness_report", "followup_missingness_analysis"]
