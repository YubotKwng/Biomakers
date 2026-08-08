"""Consolidated model-performance reporting for notebook display."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PERFORMANCE_SPEC = pd.DataFrame(
    [
        {"question": "2-year disease sensitivity", "metric": "V1->V3 paired d_z", "role": "Primary"},
        {"question": "Annual sensitivity", "metric": "V1->V2 & V2->V3 d_z", "role": "Secondary"},
        {"question": "Direction consistency", "metric": "P(delta > 0)", "role": "Secondary"},
        {"question": "Robustness", "metric": "bootstrap CI for d_z", "role": "Primary uncertainty"},
        {"question": "Better than clinical scale?", "metric": "d_z composite vs FARS/SARA", "role": "RQ1"},
        {"question": "Better than MRI alone?", "metric": "vs strongest individual MRI feature", "role": "RQ1"},
        {"question": "Disease specific?", "metric": "FRDA vs control change", "role": "Specificity"},
        {"question": "Clinically meaningful?", "metric": "Spearman Z vs FARS/SARA", "role": "RQ3"},
        {"question": "Tracks clinical change?", "metric": "Spearman delta Z vs delta FARS/SARA", "role": "Strong RQ3"},
        {"question": "Feature robustness", "metric": "coefficient/sign/Jaccard stability", "role": "Robustness"},
    ]
)


def cv_contract_table(
    *,
    outer_cv: str = "repeated subject-level 5-fold CV",
    inner_cv: str = "grouped subject-level CV inside outer training subjects",
    grouping_unit: str = "subject_id / participant group",
) -> pd.DataFrame:
    """Return the common audit/QC/CV contract every model report must follow."""
    return pd.DataFrame(
        [
            {"component": "Data audit", "requirement": "Visit pattern, analysis population, and Feature x Visit x Site missingness displayed before modelling."},
            {"component": "MRI QC", "requirement": "Distribution by site, site-effect screen, outlier review, and harmonisation leakage policy displayed before modelling."},
            {"component": "Outer CV", "requirement": outer_cv},
            {"component": "Inner CV", "requirement": inner_cv},
            {"component": "Grouping", "requirement": f"All visits/intervals for one participant stay together: {grouping_unit}."},
            {"component": "Leakage policy", "requirement": "Imputation, scaling, feature selection, harmonisation, and tuning are fit on training subjects only."},
        ]
    )


def best_model_rows_from_logs(log_paths) -> pd.DataFrame:
    """Load existing optimization logs and keep the best row per model name."""
    frames = []
    for path in [Path(p) for p in log_paths]:
        if path.exists():
            frame = pd.read_csv(path)
            frame["source_log"] = path.name
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    logs = pd.concat(frames, ignore_index=True, sort=False)
    if "model" not in logs or "d_score" not in logs:
        return pd.DataFrame()
    logs["d_score"] = pd.to_numeric(logs["d_score"], errors="coerce")
    logs = logs.dropna(subset=["model", "d_score"]).copy()
    logs = logs.sort_values("d_score", ascending=False, kind="mergesort")
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
            if spec["question"] == "2-year disease sensitivity":
                value = row.get("d_score", np.nan)
                n = row.get("n_subjects", np.nan)
                status = "available_from_existing_log_check_interval_before_claiming_v1_v3"
            elif spec["question"] == "Robustness":
                lo = row.get("d_ci_low", np.nan)
                hi = row.get("d_ci_high", np.nan)
                if pd.notna(lo) or pd.notna(hi):
                    value = f"{lo}, {hi}"
                    n = row.get("n_subjects", np.nan)
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


def _best_abs(table: pd.DataFrame | None, interval: str, value_col: str = "d_z") -> pd.Series | None:
    if table is None or table.empty or value_col not in table.columns or "interval" not in table.columns:
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

    if question == "2-year disease sensitivity":
        row = _interval_row(composite, "V1->V3")
        return _value(row, "d_z", "computed", "composite V1->V3")
    if question == "Annual sensitivity":
        r12 = _interval_row(composite, "V1->V2")
        r23 = _interval_row(composite, "V2->V3")
        value = f"{_maybe(r12, 'd_z')}; {_maybe(r23, 'd_z')}"
        status = "computed" if r12 is not None and r23 is not None else "missing"
        return value, _maybe(r12, "n_pairs"), status, "composite V1->V2 and V2->V3"
    if question == "Direction consistency":
        row = _interval_row(composite, "V1->V3")
        return _value(row, "p_delta_positive", "computed", "composite V1->V3")
    if question == "Robustness":
        row = _interval_row(composite, "V1->V3")
        if row is None:
            return np.nan, np.nan, "missing", "composite V1->V3 bootstrap CI"
        return f"{row.get('d_z_ci_low', np.nan)}, {row.get('d_z_ci_high', np.nan)}", row.get("n_pairs", np.nan), "computed", "composite V1->V3 bootstrap CI"
    if question == "Better than clinical scale?":
        comp = _interval_row(composite, "V1->V3")
        clin = _best_abs(clinical, "V1->V3")
        return f"{_maybe(comp, 'd_z')} vs {_maybe(clin, 'd_z')}", _maybe(comp, "n_pairs"), "computed" if comp is not None and clin is not None else "missing_reference", "clinical interval benchmark"
    if question == "Better than MRI alone?":
        comp = _interval_row(composite, "V1->V3")
        base = _best_abs(single, "V1->V3")
        feature = _maybe(base, "feature")
        return f"{_maybe(comp, 'd_z')} vs {feature}: {_maybe(base, 'd_z')}", _maybe(comp, "n_pairs"), "computed" if comp is not None and base is not None else "missing_reference", "strongest single MRI feature"
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
        value_cols = [c for c in ("mean_jaccard", "sign_stability", "score_ranking_stability") if c in stability.columns]
        return "; ".join(f"{c}={stability.iloc[0][c]}" for c in value_cols), np.nan, "computed", "feature/sign/ranking stability"
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


def _value(row, col, ok_status, evidence):
    if row is None:
        return np.nan, np.nan, "missing", evidence
    return row.get(col, np.nan), row.get("n_pairs", row.get("n", np.nan)), ok_status, evidence


__all__ = [
    "PERFORMANCE_SPEC",
    "append_log_model_summaries",
    "assemble_performance_rows",
    "best_model_rows_from_logs",
    "cv_contract_table",
    "save_one_performance_csv",
]
