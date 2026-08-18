"""Consolidated model-performance reporting for notebook display."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PERFORMANCE_SPEC = pd.DataFrame(
    [
        {
            "question": "12-month sensitivity V1->V2",
            "metric": "V1->V2 paired d_z, CI, N, P(delta>0)",
            "role": "Primary",
        },
        {
            "question": "12-month sensitivity V2->V3",
            "metric": "V2->V3 paired d_z, CI, N, P(delta>0)",
            "role": "Primary temporal replication",
        },
        {
            "question": "12-month pooled annual sensitivity",
            "metric": "Pooled V1->V2 + V2->V3 paired d_z, CI, N, P(delta>0)",
            "role": "Pooled annual diagnostic",
        },
        {
            "question": "24-month cumulative sensitivity",
            "metric": "V1->V3 paired d_z",
            "role": "Secondary",
        },
        {"question": "Direction consistency", "metric": "P(delta > 0)", "role": "Secondary"},
        {"question": "Robustness", "metric": "bootstrap CI for d_z", "role": "Primary uncertainty"},
        {
            "question": "Better than clinical scale?",
            "metric": "d_z composite vs FARS/SARA",
            "role": "RQ1",
        },
        {
            "question": "Better than MRI alone?",
            "metric": "vs strongest individual MRI feature",
            "role": "RQ1",
        },
        {
            "question": "Disease specific?",
            "metric": "FRDA vs control change",
            "role": "Specificity",
        },
        {"question": "Clinically meaningful?", "metric": "Spearman Z vs FARS/SARA", "role": "RQ3"},
        {
            "question": "Tracks clinical change?",
            "metric": "Spearman delta Z vs delta FARS/SARA",
            "role": "Strong RQ3",
        },
        {
            "question": "Feature robustness",
            "metric": "coefficient/sign/Jaccard stability",
            "role": "Robustness",
        },
    ]
)

CANONICAL_PAIR_COUNTS = {
    "V1->V2": 108,
    "V2->V3": 99,
    "V1->V3": 90,
    "pooled_annual": 207,
}


def cv_contract_table(
    *,
    outer_cv: str = "repeated subject-level 5-fold CV",
    inner_cv: str = "grouped subject-level CV inside outer training subjects",
    grouping_unit: str = "subject_id / participant group",
) -> pd.DataFrame:
    """Return the common audit/QC/CV contract every model report must follow."""
    return pd.DataFrame(
        [
            {
                "component": "Data audit",
                "requirement": (
                    "Visit pattern, analysis population, and Feature x Visit x Site "
                    "missingness displayed before modelling."
                ),
            },
            {
                "component": "MRI QC",
                "requirement": (
                    "Distribution by site, site-effect screen, outlier review, "
                    "and harmonisation leakage policy displayed before modelling."
                ),
            },
            {"component": "Outer CV", "requirement": outer_cv},
            {"component": "Inner CV", "requirement": inner_cv},
            {
                "component": "Grouping",
                "requirement": (
                    "All visits/intervals for one participant stay together: "
                    f"{grouping_unit}."
                ),
            },
            {
                "component": "Leakage policy",
                "requirement": (
                    "Imputation, scaling, feature selection, harmonisation, and "
                    "tuning are fit on training subjects only."
                ),
            },
        ]
    )


def best_model_rows_from_logs(log_paths) -> pd.DataFrame:
    """Load existing optimization logs and keep the best row per model name."""
    frames = []
    for path in [Path(p) for p in log_paths]:
        if path.exists():
            try:
                frame = pd.read_csv(path)
            except pd.errors.EmptyDataError:
                continue
            if frame.empty:
                continue
            frame["source_log"] = path.name
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    logs = pd.concat(frames, ignore_index=True, sort=False)
    if "model" not in logs or "d_score" not in logs:
        return pd.DataFrame()
    logs["d_score"] = pd.to_numeric(logs["d_score"], errors="coerce")
    if "mean_validation_annual_dz" in logs.columns:
        logs["mean_validation_annual_dz"] = pd.to_numeric(
            logs["mean_validation_annual_dz"],
            errors="coerce",
        )
    else:
        logs["mean_validation_annual_dz"] = np.nan

    if "annual_interval_gap" in logs.columns:
        logs["annual_interval_gap"] = pd.to_numeric(logs["annual_interval_gap"], errors="coerce")
    else:
        logs["annual_interval_gap"] = np.nan
    logs = logs.dropna(subset=["model"]).copy()
    logs["_sort_annual"] = logs["mean_validation_annual_dz"].fillna(logs["d_score"])
    logs["_sort_gap"] = logs["annual_interval_gap"].fillna(np.inf)
    logs = logs.sort_values(
        ["_sort_annual", "_sort_gap", "d_score"],
        ascending=[False, True, False],
        kind="mergesort",
    )
    return logs.groupby("model", as_index=False, sort=False).head(1).reset_index(drop=True)


def assemble_performance_rows(
    model_name: str,
    *,
    composite_intervals: pd.DataFrame | None = None,
    clinical_intervals: pd.DataFrame | None = None,
    single_feature_intervals: pd.DataFrame | None = None,
    clinical_validity: pd.DataFrame | None = None,
    specificity: pd.DataFrame | None = None,
    stability: pd.DataFrame | None = None,
    source: str = "notebook",
    cv_mode: str | None = None,
) -> pd.DataFrame:
    """Build one model's required performance rows."""
    rows = []
    for _, spec in PERFORMANCE_SPEC.iterrows():
        value, n, status, evidence = _resolve_value(
            spec["question"],
            composite_intervals=composite_intervals,
            clinical_intervals=clinical_intervals,
            single_feature_intervals=single_feature_intervals,
            clinical_validity=clinical_validity,
            specificity=specificity,
            stability=stability,
        )
        rows.append({
            "model": model_name,
            "question": spec["question"],
            "metric": spec["metric"],
            "role": spec["role"],
            "value": value,
            "n": n,
            "status": status,
            "evidence": evidence,
            "cv_mode": cv_mode,
            "source": source,
        })
    return pd.DataFrame(rows)


def append_log_model_summaries(performance: pd.DataFrame, log_models: pd.DataFrame) -> pd.DataFrame:
    """Append required-schema rows for models only available in source logs."""
    if log_models is None or log_models.empty:
        return performance
    existing = set(performance["model"].astype(str)) if not performance.empty else set()
    rows = []
    for _, row in log_models.iterrows():
        model = str(row["model"])
        if model in existing:
            continue

        for _, spec in PERFORMANCE_SPEC.iterrows():
            value = np.nan
            n = np.nan
            status = "not_available_in_existing_log"
            evidence = row.get("notes", "")
            if spec["question"] == "12-month sensitivity V1->V2":
                value = row.get("dz_v1_v2", np.nan)
                n = CANONICAL_PAIR_COUNTS["V1->V2"]
                status = _availability_status(value)
            elif spec["question"] == "12-month sensitivity V2->V3":
                value = row.get("dz_v2_v3", np.nan)
                n = CANONICAL_PAIR_COUNTS["V2->V3"]
                status = _availability_status(value)
            elif spec["question"] == "12-month pooled annual sensitivity":
                has_annual_context = (
                    pd.notna(row.get("dz_v1_v2", np.nan))
                    or pd.notna(row.get("dz_v2_v3", np.nan))
                    or pd.notna(row.get("mean_validation_annual_dz", np.nan))
                )
                if has_annual_context:
                    value = row.get("d_score", np.nan)
                    n = CANONICAL_PAIR_COUNTS["pooled_annual"]
                    status = _availability_status(value)
            elif spec["question"] == "24-month cumulative sensitivity":
                value = row.get("dz_v1_v3", np.nan)
                n = CANONICAL_PAIR_COUNTS["V1->V3"]
                status = _availability_status(value)
            elif spec["question"] == "Direction consistency":
                value = row.get("p_progression", np.nan)
                n = CANONICAL_PAIR_COUNTS["pooled_annual"]
                status = _availability_status(value)
            elif spec["question"] == "Robustness":
                lo = row.get("d_ci_low", np.nan)
                hi = row.get("d_ci_high", np.nan)
                if pd.notna(lo) or pd.notna(hi):
                    value = f"{lo}, {hi}"
                    n = CANONICAL_PAIR_COUNTS["pooled_annual"]
                    status = "available_from_existing_log"
            rows.append({
                "model": model,
                "question": spec["question"],
                "metric": spec["metric"],
                "role": spec["role"],
                "value": value,
                "n": n,
                "status": status,
                "evidence": evidence,
                "cv_mode": row.get("cv_mode", np.nan),
                "source": row.get("source_log", np.nan),
            })
    if not rows:
        return performance
    return pd.concat([performance, pd.DataFrame(rows)], ignore_index=True, sort=False)


def save_one_performance_csv(performance: pd.DataFrame, path: str | Path) -> Path:
    """Save the single consolidated performance CSV."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    performance.to_csv(out, index=False)
    return out


def _interval_row(table: pd.DataFrame | None, interval: str) -> pd.Series | None:
    if table is None or table.empty or "interval" not in table.columns:
        return None
    mask = table["interval"].astype(str).str.upper().eq(interval.upper())
    if not mask.any():
        return None
    return table.loc[mask].iloc[0]


def _availability_status(value) -> str:
    return "available_from_existing_log" if pd.notna(value) else "not_available_in_existing_log"


def _best_abs(
    table: pd.DataFrame | None,
    interval: str,
    value_col: str = "d_z",
) -> pd.Series | None:
    if (
        table is None
        or table.empty
        or value_col not in table.columns
        or "interval" not in table.columns
    ):
        return None
    rows = table[table["interval"].astype(str).str.upper().eq(interval.upper())].copy()
    if rows.empty:
        return None
    vals = pd.to_numeric(rows[value_col], errors="coerce").abs()
    if vals.notna().sum() == 0:
        return None
    return rows.loc[vals.idxmax()]


def _resolve_value(question: str, **tables):
    composite = tables["composite_intervals"]
    clinical = tables["clinical_intervals"]
    single = tables["single_feature_intervals"]
    validity = tables["clinical_validity"]
    specificity = tables["specificity"]
    stability = tables["stability"]

    if question == "12-month sensitivity V1->V2":
        row = _interval_row(composite, "V1->V2")
        if row is None:
            return np.nan, np.nan, "missing", "composite V1->V2"
        return (
            _format_effect_value(row),
            row.get("n_pairs", np.nan),
            "computed",
            "composite V1->V2 OOF annual interval",
        )
    if question == "12-month sensitivity V2->V3":
        row = _interval_row(composite, "V2->V3")
        if row is None:
            return np.nan, np.nan, "missing", "composite V2->V3"
        return (
            _format_effect_value(row),
            row.get("n_pairs", np.nan),
            "computed",
            "composite V2->V3 OOF annual interval",
        )
    if question == "12-month pooled annual sensitivity":
        row = _interval_row(composite, "V1->V2 + V2->V3")
        if row is None:
            return np.nan, np.nan, "missing", "pooled annual V1->V2 + V2->V3"
        return (
            _format_effect_value(row),
            row.get("n_pairs", np.nan),
            "computed",
            "pooled OOF annual pair deltas; participant-grouped splitting is preserved upstream",
        )
    if question == "24-month cumulative sensitivity":
        row = _interval_row(composite, "V1->V3")
        return _value(row, "d_z", "computed", "composite V1->V3 cumulative")
    if question == "Direction consistency":
        r12 = _interval_row(composite, "V1->V2")
        r23 = _interval_row(composite, "V2->V3")
        value = f"{_maybe(r12, 'p_delta_positive')}; {_maybe(r23, 'p_delta_positive')}"
        status = "computed" if r12 is not None and r23 is not None else "missing"
        return value, _maybe(r12, "n_pairs"), status, "annual V1->V2 and V2->V3 P(delta>0)"
    if question == "Robustness":
        r12 = _interval_row(composite, "V1->V2")
        r23 = _interval_row(composite, "V2->V3")
        if r12 is None and r23 is None:
            return np.nan, np.nan, "missing", "annual bootstrap CI"
        value = (
            f"V1->V2 [{_maybe(r12, 'd_z_ci_low')}, {_maybe(r12, 'd_z_ci_high')}]; "
            f"V2->V3 [{_maybe(r23, 'd_z_ci_low')}, {_maybe(r23, 'd_z_ci_high')}]"
        )
        return value, _maybe(r12, "n_pairs"), "computed", "annual interval bootstrap CI"
    if question == "Better than clinical scale?":
        comp = _interval_row(composite, "V1->V3")
        clin = _best_abs(clinical, "V1->V3")
        status = "computed" if comp is not None and clin is not None else "missing_reference"
        return (
            f"{_maybe(comp, 'd_z')} vs {_maybe(clin, 'd_z')}",
            _maybe(comp, "n_pairs"),
            status,
            "clinical interval benchmark",
        )
    if question == "Better than MRI alone?":
        comp = _interval_row(composite, "V1->V3")
        base = _best_abs(single, "V1->V3")
        feature = _maybe(base, "feature")
        status = "computed" if comp is not None and base is not None else "missing_reference"
        return (
            f"{_maybe(comp, 'd_z')} vs {feature}: {_maybe(base, 'd_z')}",
            _maybe(comp, "n_pairs"),
            status,
            "strongest single MRI feature",
        )
    if question == "Disease specific?":
        row = specificity.iloc[0] if specificity is not None and not specificity.empty else None
        return _value(row, "value", "computed", "FRDA vs control change")
    if question == "Clinically meaningful?":
        row = _validity_row(validity, "cross_sectional")
        return _value(row, "rho", "computed", "cross-sectional Spearman")
    if question == "Tracks clinical change?":
        row = _validity_row(validity, "longitudinal_delta")
        return _value(row, "rho", "computed", "delta Spearman")
    if question == "Feature robustness":
        if stability is None or stability.empty:
            return np.nan, np.nan, "missing", "stability diagnostics"
        value_cols = [
            c
            for c in ("mean_jaccard", "sign_stability", "score_ranking_stability")
            if c in stability.columns
        ]
        return (
            "; ".join(f"{c}={stability.iloc[0][c]}" for c in value_cols),
            np.nan,
            "computed",
            "feature/sign/ranking stability",
        )
    return np.nan, np.nan, "missing", ""


def _validity_row(table: pd.DataFrame | None, analysis: str):
    if table is None or table.empty or "analysis" not in table.columns:
        return None
    rows = table[table["analysis"].astype(str).eq(analysis)]
    return rows.iloc[0] if not rows.empty else None


def _maybe(row, col):
    if row is None:
        return np.nan
    return row.get(col, np.nan)


def _format_effect_value(row) -> str:
    return (
        f"{row.get('d_z', np.nan)} "
        f"[{row.get('d_z_ci_low', np.nan)}, {row.get('d_z_ci_high', np.nan)}]; "
        f"P={row.get('p_delta_positive', np.nan)}"
    )


def _value(row, col, ok_status, evidence):
    if row is None:
        return np.nan, np.nan, "missing", evidence
    return row.get(col, np.nan), row.get("n_pairs", row.get("n", np.nan)), ok_status, evidence


__all__ = [
    "PERFORMANCE_SPEC",
    "CANONICAL_PAIR_COUNTS",
    "append_log_model_summaries",
    "assemble_performance_rows",
    "best_model_rows_from_logs",
    "cv_contract_table",
    "save_one_performance_csv",
]
