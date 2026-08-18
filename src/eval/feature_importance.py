"""Feature-importance helpers for the locked SRM Global Linear model."""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from ..data.qc import standardize_train_test
from ..features.selection import select_features
from ..models.srm_global import SRMGlobalLinear, srm_global_loocv
from .intervals import adjacent_pair_interval_effect_summary, annual_tuning_diagnostics
from .stability import selected_feature_jaccard


def selected_srm_config_from_log(
    log: pd.DataFrame,
    model_name: str = "SRM Global Linear exploratory",
) -> dict:
    """Return the currently best SRM configuration from an optimization log."""
    if log is None or log.empty:
        raise ValueError("optimization log is empty")
    if "model" not in log.columns or "mean_validation_annual_dz" not in log.columns:
        raise KeyError("log must include model and mean_validation_annual_dz columns")
    rows = log[log["model"].astype(str).eq(model_name)].copy()
    if rows.empty:
        rows = log[
            log["model"].astype(str).str.contains("SRM Global Linear", na=False)
        ].copy()
    if rows.empty:
        raise ValueError("no SRM Global Linear rows found in optimization log")
    rows["mean_validation_annual_dz"] = pd.to_numeric(
        rows["mean_validation_annual_dz"],
        errors="coerce",
    )
    rows["annual_interval_gap"] = pd.to_numeric(
        rows.get("annual_interval_gap", np.nan),
        errors="coerce",
    )
    rows = rows.sort_values(
        ["mean_validation_annual_dz", "annual_interval_gap"],
        ascending=[False, True],
        kind="mergesort",
    )
    best = rows.iloc[0]

    def _none_if_nan(value):
        if pd.isna(value):
            return None
        return value

    return {
        "model": str(best.get("model", "SRM Global Linear")),
        "ridge": _float_or_default(best, "param_ridge", 0.0),
        "covariance_shrinkage": float(best.get("param_covariance_shrinkage", 0.0))
        if pd.notna(best.get("param_covariance_shrinkage", np.nan))
        else 0.0,
        "z_clip": _float_or_none(best, "param_z_clip"),
        "selection_method": str(best.get("param_selection_method", "none"))
        if pd.notna(best.get("param_selection_method", np.nan))
        else "none",
        "k": int(best.get("param_k", 8)) if pd.notna(best.get("param_k", np.nan)) else 8,
        "tuning_metric": "mean_validation_annual_dz",
        "source_row": best.to_dict(),
        "regularisation": {
            "ridge": _float_or_default(best, "param_ridge", 0.0),
            "covariance_shrinkage": float(best.get("param_covariance_shrinkage", 0.0))
            if pd.notna(best.get("param_covariance_shrinkage", np.nan))
            else 0.0,
            "z_clip": _none_if_nan(best.get("param_z_clip", np.nan)),
        },
    }


def _float_or_default(row: pd.Series, key: str, default: float) -> float:
    value = row.get(key, default)
    return float(value) if pd.notna(value) else float(default)


def _float_or_none(row: pd.Series, key: str) -> float | None:
    value = row.get(key, np.nan)
    return None if pd.isna(value) else float(value)


def eligible_interval_frame(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    *,
    pair_col: str = "pair_id",
    visit_col: str = "visit",
    split_group_col: str = "subject",
) -> pd.DataFrame:
    """Return complete-case annual pair rows for selected features."""
    cols = [pair_col, visit_col, split_group_col] + [
        f for f in feature_cols if f in df.columns
    ]
    work = df[cols].dropna().copy()
    work[visit_col] = pd.to_numeric(work[visit_col], errors="coerce").astype("Int64")
    work = work[work[visit_col].isin([1, 2])].copy()
    complete = work.groupby(pair_col)[visit_col].agg(
        lambda series: {1, 2}.issubset(set(series.dropna().astype(int)))
    )
    complete_pairs = complete.index[complete.to_numpy(dtype=bool)]
    return work[work[pair_col].isin(complete_pairs)].copy()


def locked_selected_features(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    *,
    selection_method: str = "none",
    k: int = 8,
    visit_col: str = "visit",
) -> list[str]:
    """Apply the locked feature-selection rule on the supplied frame."""
    feats_present = [f for f in feature_cols if f in df.columns]
    if not feats_present:
        return []
    y_select = (pd.to_numeric(df[visit_col], errors="coerce").values == 2).astype(int)
    return list(
        select_features(
            selection_method,
            df[feats_present],
            y_select,
            feats_present,
            k=int(k),
            train_frame=df,
            subject_col="pair_id" if "pair_id" in df.columns else None,
            visit_col=visit_col,
        )
    )


def fit_locked_srm_full_data(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    *,
    pair_col: str = "pair_id",
    visit_col: str = "visit",
    split_group_col: str = "subject",
    selection_method: str = "none",
    k: int = 8,
    ridge: float = 0.0,
    covariance_shrinkage: float = 0.0,
    z_clip: float | None = None,
) -> dict:
    """Fit the locked SRM model on all eligible FRDA rows for interpretation."""
    sub = eligible_interval_frame(
        df,
        feature_cols,
        pair_col=pair_col,
        visit_col=visit_col,
        split_group_col=split_group_col,
    )
    feats = locked_selected_features(
        sub,
        feature_cols,
        selection_method=selection_method,
        k=k,
        visit_col=visit_col,
    )
    X = sub[feats].to_numpy(dtype=float) if feats else np.zeros((len(sub), 0))
    X_std, _, center, scale = standardize_train_test(X, X)
    if z_clip is not None and X_std.shape[1] > 0:
        X_std = np.clip(X_std, -float(z_clip), float(z_clip))
    model = SRMGlobalLinear(
        ridge=float(ridge),
        covariance_shrinkage=float(covariance_shrinkage),
        start_visit=1,
        end_visit=2,
    ).fit(X_std, sub[pair_col].values, sub[visit_col].values)
    scored = sub[[pair_col, visit_col, split_group_col]].copy()
    scored["score"] = model.score(X_std)
    return {
        "model": model,
        "feature_names": feats,
        "coef": model.coef_.copy() if model.coef_ is not None else np.zeros(len(feats)),
        "center": np.asarray(center, dtype=float),
        "scale": np.asarray(scale, dtype=float),
        "standardized_X": pd.DataFrame(X_std, columns=feats, index=sub.index),
        "frame": sub,
        "scores": scored,
    }


def coefficient_importance_table(
    feature_names: Sequence[str],
    coef: Sequence[float],
) -> pd.DataFrame:
    """Return standardised coefficient magnitude and sign table."""
    out = pd.DataFrame(
        {
            "feature": list(feature_names),
            "standardised_coefficient": np.asarray(coef, dtype=float),
        }
    )
    out["absolute_coefficient"] = out["standardised_coefficient"].abs()
    out["coefficient_sign"] = np.sign(out["standardised_coefficient"]).astype(int)
    out = out.sort_values(
        "absolute_coefficient",
        ascending=False,
        kind="mergesort",
    ).reset_index(drop=True)
    out["rank_by_absolute_coefficient"] = np.arange(1, len(out) + 1)
    return out


def bootstrap_srm_coefficients(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    *,
    pair_col: str = "pair_id",
    visit_col: str = "visit",
    split_group_col: str = "subject",
    selection_method: str = "none",
    k: int = 8,
    ridge: float = 0.0,
    covariance_shrinkage: float = 0.0,
    z_clip: float | None = None,
    n_boot: int = 1000,
    random_seed: int = 42,
    locked_feature_names: Sequence[str] | None = None,
    full_coef: Sequence[float] | None = None,
) -> dict:
    """Subject-bootstrap locked SRM coefficients without hyperparameter retuning."""
    base = eligible_interval_frame(
        df,
        feature_cols,
        pair_col=pair_col,
        visit_col=visit_col,
        split_group_col=split_group_col,
    )
    feature_names = list(
        locked_feature_names
        or locked_selected_features(
            base,
            feature_cols,
            selection_method=selection_method,
            k=k,
            visit_col=visit_col,
        )
    )
    rng = np.random.default_rng(random_seed)
    subjects = np.asarray(sorted(base[split_group_col].dropna().unique()))
    subject_rows = {s: base.index[base[split_group_col].eq(s)].to_numpy() for s in subjects}
    coef_rows: list[np.ndarray] = []
    selected_sets: list[list[str]] = []
    kept_boot = 0
    for b in range(int(n_boot)):
        sampled = rng.choice(subjects, size=len(subjects), replace=True)
        parts = []
        for draw_idx, subj in enumerate(sampled):
            part = base.loc[subject_rows[subj]].copy()
            part[split_group_col] = f"{subj}__boot{b}_{draw_idx}"
            part[pair_col] = part[pair_col].astype(str) + f"__boot{b}_{draw_idx}"
            parts.append(part)
        boot = pd.concat(parts, ignore_index=True)
        try:
            fit = fit_locked_srm_full_data(
                boot,
                feature_cols,
                pair_col=pair_col,
                visit_col=visit_col,
                split_group_col=split_group_col,
                selection_method=selection_method,
                k=k,
                ridge=ridge,
                covariance_shrinkage=covariance_shrinkage,
                z_clip=z_clip,
            )
        except Exception:
            continue
        selected = list(fit["feature_names"])
        selected_sets.append(selected)
        aligned = np.zeros(len(feature_names), dtype=float)
        coef_map = dict(zip(selected, fit["coef"]))
        for j, feat in enumerate(feature_names):
            aligned[j] = float(coef_map.get(feat, 0.0))
        coef_rows.append(aligned)
        kept_boot += 1
    coef_mat = np.vstack(coef_rows) if coef_rows else np.zeros((0, len(feature_names)))
    summary = _bootstrap_coef_summary(
        feature_names,
        coef_mat,
        selected_sets,
        full_coef=full_coef,
    )
    return {
        "summary": summary,
        "coef_samples": coef_mat,
        "selected_feature_sets": selected_sets,
        "n_boot_kept": kept_boot,
    }


def _bootstrap_coef_summary(
    feature_names: Sequence[str],
    coef_mat: np.ndarray,
    selected_sets: Sequence[Sequence[str]],
    *,
    full_coef: Sequence[float] | None = None,
) -> pd.DataFrame:
    rows = []
    full = (
        np.asarray(full_coef, dtype=float)
        if full_coef is not None
        else np.full(len(feature_names), np.nan)
    )
    selected_sets_as_sets = [set(s) for s in selected_sets]
    for j, feat in enumerate(feature_names):
        vals = coef_mat[:, j] if coef_mat.size else np.asarray([], dtype=float)
        vals = vals[np.isfinite(vals)]
        pos = float(np.mean(vals > 0)) if len(vals) else np.nan
        neg = float(np.mean(vals < 0)) if len(vals) else np.nan
        sign_consistency = float(max(pos, neg)) if len(vals) else np.nan
        rows.append(
            {
                "feature": feat,
                "full_data_coefficient": full[j] if j < len(full) else np.nan,
                "bootstrap_median_coefficient": (
                    float(np.median(vals)) if len(vals) else np.nan
                ),
                "bootstrap_ci_low": (
                    float(np.percentile(vals, 2.5)) if len(vals) else np.nan
                ),
                "bootstrap_ci_high": (
                    float(np.percentile(vals, 97.5)) if len(vals) else np.nan
                ),
                "selection_frequency": (
                    float(np.mean([feat in s for s in selected_sets_as_sets]))
                    if selected_sets_as_sets
                    else np.nan
                ),
                "positive_sign_frequency": pos,
                "negative_sign_frequency": neg,
                "sign_consistency": sign_consistency,
                "crosses_zero": (
                    bool(np.nanmin(vals) <= 0 <= np.nanmax(vals))
                    if len(vals)
                    else True
                ),
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values(
        "sign_consistency",
        ascending=False,
        kind="mergesort",
    ).reset_index(drop=True)


def annual_feature_contributions(
    fit_result: Mapping,
    *,
    pair_col: str = "pair_id",
    visit_col: str = "visit",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Decompose annual composite deltas into feature-level contributions."""
    frame = fit_result["frame"]
    X = fit_result["standardized_X"].copy()
    coef = np.asarray(fit_result["coef"], dtype=float)
    features = list(fit_result["feature_names"])
    records = []
    pair_meta = frame[[pair_col, visit_col]].copy()
    pair_meta["_interval"] = (
        pair_meta[pair_col].astype(str).str.upper().str.extract(r"(V\d+V\d+)", expand=False)
        .map({"V1V2": "V1->V2", "V2V3": "V2->V3"})
    )
    for pid, rows in pair_meta.dropna(subset=["_interval"]).groupby(pair_col, sort=False):
        ordered = rows.sort_values(visit_col)
        if len(ordered) != 2:
            continue
        idx0, idx1 = ordered.index[0], ordered.index[1]
        delta_x = (
            X.loc[idx1, features].to_numpy(dtype=float)
            - X.loc[idx0, features].to_numpy(dtype=float)
        )
        contrib = coef * delta_x
        interval = str(ordered["_interval"].iloc[0])
        for feat, dx, c in zip(features, delta_x, contrib):
            records.append({
                pair_col: pid,
                "interval": interval,
                "feature": feat,
                "standardized_feature_change": float(dx),
                "contribution": float(c),
            })
    long = pd.DataFrame(records)
    if long.empty:
        return long, pd.DataFrame()
    summary = (
        long.groupby(["feature", "interval"], as_index=False)
        .agg(
            mean_standardized_feature_change=("standardized_feature_change", "mean"),
            mean_contribution=("contribution", "mean"),
            n_pairs=("contribution", "size"),
        )
    )
    wide = summary.pivot(index="feature", columns="interval")
    out = pd.DataFrame({"feature": wide.index})

    def _wide_values(metric: str, interval: str):
        key = (metric, interval)
        if key in wide.columns:
            return wide[key].to_numpy()
        return np.full(len(wide.index), np.nan)

    for interval in ("V1->V2", "V2->V3"):
        out[f"mean_standardized_feature_change_{interval}"] = _wide_values(
            "mean_standardized_feature_change",
            interval,
        )
        out[f"mean_contribution_{interval}"] = _wide_values("mean_contribution", interval)
    out["contribution_gap"] = out["mean_contribution_V1->V2"] - out["mean_contribution_V2->V3"]
    out["absolute_annual_contribution"] = (
        out[["mean_contribution_V1->V2", "mean_contribution_V2->V3"]].abs().mean(axis=1)
    )
    out = out.sort_values(
        "absolute_annual_contribution",
        ascending=False,
        kind="mergesort",
    ).reset_index(drop=True)
    return long, out


def lofo_srm_importance(
    df: pd.DataFrame,
    feature_names: Sequence[str],
    *,
    pair_col: str = "pair_id",
    visit_col: str = "visit",
    split_group_col: str = "subject",
    selection_method: str = "none",
    k: int = 8,
    ridge: float = 0.0,
    covariance_shrinkage: float = 0.0,
    z_clip: float | None = None,
    cv_n_splits: int = 5,
    random_seed: int = 42,
    n_boot: int = 200,
) -> pd.DataFrame:
    """Leave-one-feature-out OOF annual sensitivity under fixed hyperparameters."""
    features = list(feature_names)
    full = _annual_oof_diagnostics(
        df,
        features,
        pair_col=pair_col,
        visit_col=visit_col,
        split_group_col=split_group_col,
        selection_method=selection_method,
        k=k,
        ridge=ridge,
        covariance_shrinkage=covariance_shrinkage,
        z_clip=z_clip,
        cv_n_splits=cv_n_splits,
        random_seed=random_seed,
        n_boot=n_boot,
    )
    rows = []
    for feat in features:
        reduced = [f for f in features if f != feat]
        diag = _annual_oof_diagnostics(
            df,
            reduced,
            pair_col=pair_col,
            visit_col=visit_col,
            split_group_col=split_group_col,
            selection_method=selection_method,
            k=min(k, len(reduced)) if selection_method != "none" else k,
            ridge=ridge,
            covariance_shrinkage=covariance_shrinkage,
            z_clip=z_clip,
            cv_n_splits=cv_n_splits,
            random_seed=random_seed,
            n_boot=n_boot,
        )
        rows.append({
            "removed_feature": feat,
            "full_d12": full["dz_v1_v2"],
            "lofo_d12": diag["dz_v1_v2"],
            "d12_loss": full["dz_v1_v2"] - diag["dz_v1_v2"],
            "full_d23": full["dz_v2_v3"],
            "lofo_d23": diag["dz_v2_v3"],
            "d23_loss": full["dz_v2_v3"] - diag["dz_v2_v3"],
            "full_mean_annual_dz": full["mean_validation_annual_dz"],
            "lofo_mean_annual_dz": diag["mean_validation_annual_dz"],
            "annual_performance_loss": (
                full["mean_validation_annual_dz"] - diag["mean_validation_annual_dz"]
            ),
        })
    return pd.DataFrame(rows).sort_values(
        "annual_performance_loss",
        ascending=False,
        kind="mergesort",
    ).reset_index(drop=True)


def _annual_oof_diagnostics(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    *,
    pair_col: str,
    visit_col: str,
    split_group_col: str,
    selection_method: str,
    k: int,
    ridge: float,
    covariance_shrinkage: float,
    z_clip: float | None,
    cv_n_splits: int,
    random_seed: int,
    n_boot: int,
) -> dict:
    res = srm_global_loocv(
        df,
        feature_cols,
        subject_col=pair_col,
        visit_col=visit_col,
        selection_method=selection_method,
        k=k,
        ridge=ridge,
        covariance_shrinkage=covariance_shrinkage,
        z_clip=z_clip,
        cv_n_splits=cv_n_splits,
        random_seed=random_seed,
        compute_ci=False,
        split_group_col=split_group_col,
        start_visit=1,
        end_visit=2,
    )
    summary = adjacent_pair_interval_effect_summary(
        res["oof_df"],
        pair_col=pair_col,
        visit_col=visit_col,
        score_col="score",
        n_boot=n_boot,
        seed=random_seed,
    )
    return annual_tuning_diagnostics(summary)


def correlation_redundancy(
    standardized_x: pd.DataFrame,
    *,
    threshold: float = 0.7,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Return correlation matrix, high-correlation pairs, and per-feature flags."""
    corr = standardized_x.corr()
    pairs = []
    cols = list(corr.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            r = corr.loc[a, b]
            if pd.notna(r) and abs(float(r)) > float(threshold):
                pairs.append({"feature_a": a, "feature_b": b, "correlation": float(r)})
    if pairs:
        pair_df = pd.DataFrame(pairs).sort_values(
            "correlation",
            key=lambda series: series.abs(),
            ascending=False,
            kind="mergesort",
        )
    else:
        pair_df = pd.DataFrame(columns=["feature_a", "feature_b", "correlation"])
    flags = pd.Series(False, index=cols)
    for _, row in pair_df.iterrows():
        flags.loc[row["feature_a"]] = True
        flags.loc[row["feature_b"]] = True
    return corr, pair_df.reset_index(drop=True), flags


def infer_feature_domains(feature_names: Sequence[str], groups=None) -> pd.DataFrame:
    """Derive conservative MRI domains from existing feature-group names."""
    poms = set(getattr(groups, "poms", []) or [])
    dti = set(getattr(groups, "braindti", []) or [])
    morph = set(getattr(groups, "brainspinemorph", []) or [])
    rows = []
    for feat in feature_names:
        low = feat.lower()
        if feat in {
            "csa_c1c2",
            "Cereb_vol",
            "SCP_vol",
            "TotalBrainGMVol_nocereb",
            "TotalBrainWMVol_nocereb",
            "TotalBrainVol_nocereb",
            "eTIV",
        }:
            if "scp" in low:
                domain = "SCP morphometry"
            elif "csa" in low:
                domain = "Spinal cord structural"
            else:
                domain = "Brain morphometry"
        elif feat in {"sFA_c3c5", "sMD_c3c5", "sRD_c3c5", "sAD_c3c5"}:
            domain = "Spinal cord diffusion"
        elif "scp" in low and any(
            low.startswith(prefix) for prefix in ("fa_", "md_", "rd_", "ad_")
        ):
            domain = "SCP diffusion"
        elif feat in dti or any(low.startswith(prefix) for prefix in ("fa_", "md_", "rd_", "ad_")):
            domain = "Brain DTI"
        elif "qsm" in low or "suscept" in low or "dentate" in low:
            domain = "Dentate/QSM or deep grey matter"
        elif feat in poms or feat in morph:
            domain = "Other MRI"
        else:
            domain = "Unmapped MRI"
        rows.append({"feature": feat, "domain": domain})
    return pd.DataFrame(rows)


def domain_contributions(
    contribution_table: pd.DataFrame,
    domain_map: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate feature contributions by MRI domain."""
    if contribution_table.empty:
        return pd.DataFrame()
    merged = contribution_table.merge(domain_map, on="feature", how="left")
    merged["domain"] = merged["domain"].fillna("Unmapped MRI")
    out = (
        merged.groupby("domain", as_index=False)
        .agg(
            v1_v2_total_contribution=("mean_contribution_V1->V2", "sum"),
            v2_v3_total_contribution=("mean_contribution_V2->V3", "sum"),
        )
    )
    out["difference"] = out["v1_v2_total_contribution"] - out["v2_v3_total_contribution"]
    out["absolute_mean_annual_contribution"] = (
        out[["v1_v2_total_contribution", "v2_v3_total_contribution"]]
        .abs()
        .mean(axis=1)
    )
    return out.sort_values(
        "absolute_mean_annual_contribution",
        ascending=False,
        kind="mergesort",
    ).reset_index(drop=True)


def selection_stability_summary(
    selected_sets: Sequence[Sequence[str]],
    feature_names: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return feature selection frequency and Jaccard-distribution summaries."""
    sets = [set(s) for s in selected_sets]
    rows = []
    for feat in feature_names:
        rows.append(
            {
                "feature": feat,
                "selection_frequency": (
                    float(np.mean([feat in s for s in sets])) if sets else np.nan
                ),
            }
        )
    feature_df = pd.DataFrame(rows).sort_values(
        "selection_frequency",
        ascending=False,
        kind="mergesort",
    )
    vals = []
    for i, a in enumerate(sets):
        for b in sets[i + 1:]:
            if not a and not b:
                vals.append(1.0)
            elif not a or not b:
                vals.append(0.0)
            else:
                vals.append(len(a & b) / len(a | b))
    base = selected_feature_jaccard(selected_sets)
    summary = pd.DataFrame(
        [
            {
                **base,
                "median_jaccard": float(np.median(vals)) if vals else np.nan,
                "iqr_jaccard": (
                    float(np.percentile(vals, 75) - np.percentile(vals, 25))
                    if vals
                    else np.nan
                ),
            }
        ]
    )
    return feature_df.reset_index(drop=True), summary


def integrated_feature_importance(
    coefficient_table: pd.DataFrame,
    bootstrap_table: pd.DataFrame,
    contribution_table: pd.DataFrame,
    lofo_table: pd.DataFrame,
    redundancy_flags: pd.Series,
) -> pd.DataFrame:
    """Combine evidence streams without forming an arbitrary weighted score."""
    out = coefficient_table.merge(bootstrap_table, on="feature", how="left")
    out = out.merge(contribution_table, on="feature", how="left")
    lofo = lofo_table.rename(columns={"removed_feature": "feature"})
    out = out.merge(
        lofo[["feature", "d12_loss", "d23_loss", "annual_performance_loss"]],
        on="feature",
        how="left",
    )
    out["strong_correlation_redundancy_flag"] = (
        out["feature"].map(redundancy_flags).fillna(False).astype(bool)
    )
    out["interpretation_label"] = qualitative_importance_labels(out)
    keep = [
        "feature",
        "standardised_coefficient",
        "absolute_coefficient",
        "bootstrap_ci_low",
        "bootstrap_ci_high",
        "selection_frequency",
        "sign_consistency",
        "mean_contribution_V1->V2",
        "mean_contribution_V2->V3",
        "d12_loss",
        "d23_loss",
        "annual_performance_loss",
        "strong_correlation_redundancy_flag",
        "interpretation_label",
    ]
    return out[[c for c in keep if c in out.columns]].sort_values(
        ["annual_performance_loss", "absolute_coefficient"],
        ascending=[False, False],
        kind="mergesort",
    ).reset_index(drop=True)


def qualitative_importance_labels(table: pd.DataFrame) -> pd.Series:
    """Assign transparent qualitative labels from displayed evidence."""
    abs_contrib = (
        table[["mean_contribution_V1->V2", "mean_contribution_V2->V3"]]
        .abs()
        .mean(axis=1)
    )
    contrib_cut = abs_contrib.quantile(0.75) if abs_contrib.notna().any() else np.inf
    lofo = pd.to_numeric(table.get("annual_performance_loss", np.nan), errors="coerce")
    lofo_cut = lofo.quantile(0.75) if lofo.notna().any() else np.inf
    labels = []
    for idx, row in table.iterrows():
        ci_crosses_zero = (
            pd.notna(row.get("bootstrap_ci_low"))
            and pd.notna(row.get("bootstrap_ci_high"))
            and row.get("bootstrap_ci_low") <= 0 <= row.get("bootstrap_ci_high")
        )
        sign_ok = (
            pd.notna(row.get("sign_consistency"))
            and float(row.get("sign_consistency")) >= 0.8
        )
        high_contrib = pd.notna(abs_contrib.loc[idx]) and abs_contrib.loc[idx] >= contrib_cut
        high_lofo = pd.notna(lofo.loc[idx]) and lofo.loc[idx] >= lofo_cut and lofo.loc[idx] > 0
        interval_specific = (
            pd.notna(row.get("mean_contribution_V1->V2"))
            and pd.notna(row.get("mean_contribution_V2->V3"))
            and abs(row.get("mean_contribution_V1->V2") - row.get("mean_contribution_V2->V3"))
            > abs(row.get("mean_contribution_V1->V2") + row.get("mean_contribution_V2->V3")) / 2
        )
        if (not sign_ok) or ci_crosses_zero:
            labels.append("Unstable / uncertain")
        elif (
            bool(row.get("strong_correlation_redundancy_flag", False))
            and high_contrib
            and not high_lofo
        ):
            labels.append("Potentially redundant contributor")
        elif high_contrib and high_lofo and sign_ok:
            labels.append("Strong robust contributor")
        elif interval_specific and high_contrib:
            labels.append("Interval-specific contributor")
        elif high_contrib or high_lofo:
            labels.append("Moderate contributor")
        else:
            labels.append("Minimal contribution")
    return pd.Series(labels, index=table.index)


__all__ = [
    "annual_feature_contributions",
    "bootstrap_srm_coefficients",
    "coefficient_importance_table",
    "correlation_redundancy",
    "domain_contributions",
    "eligible_interval_frame",
    "fit_locked_srm_full_data",
    "infer_feature_domains",
    "integrated_feature_importance",
    "locked_selected_features",
    "lofo_srm_importance",
    "qualitative_importance_labels",
    "selected_srm_config_from_log",
    "selection_stability_summary",
]
