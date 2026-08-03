"""TRACK-FA composite biomarker modelling pipeline (wide CSV → pair table → SRM).

In the `biomarkers/` repo, TRACK-FA preprocessing produces a **wide** table
(`cfg.trackfa_processed_csv`) with one row per participant and visit-suffixed
columns (`*_v1`, `*_v2`, `*_v3`). This module builds an in-memory consecutive
pair table and runs feature selection + subject-level LOO-CV experiments.

Primary metric everywhere: paired Cohen's d (SRM) = mean(delta) / std(delta, ddof=1).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning

from ..config import Config, DEFAULT_CONFIG
from .metrics import compute_cohens_d


# ---------------------------------------------------------------------------
# Constants (kept local to avoid cross-repo coupling to baseline_composite_biomakers/)
# ---------------------------------------------------------------------------
CLINICAL_SCORES = ("mfars_total", "adl_total", "sara_total", "lcslc_total")
VISIT_PAIRS = ((1, 2, "V1V2"), (2, 3, "V2V3"))

# POMs list from the project spec (used for feature-group experiments).
_POMS_BASES: set[str] = {
    "csa_c1c2",
    "Cereb_vol",
    "SCP_vol",
    "sFA_c3c5",
    "sMD_c3c5",
    "sRD_c3c5",
    "sAD_c3c5",
    "DN_vol",
    "DN_suscept",
    "FA_SCP",
    "MD_SCP",
    "RD_SCP",
    "AD_SCP",
    "TotalBrainGMVol_nocereb",
    "TotalBrainWMVol_nocereb",
    "TotalBrainVol_nocereb",
    "eTIV",
}

# Wide-table clinical columns beyond the four benchmark scales (exclude from "imaging").
_NON_IMAGING_BASE_PREFIXES: tuple[str, ...] = (
    "bmi",
    "functional_staging_score",
    "hpt_dom_av",
    "hpt_ndom_av",
    "scan_date_crf",
    "vygr_nfl_conc_",
)


def _delta_col(base: str) -> str:
    return f"delta_{base}"


def _is_missing(x: Any) -> bool:
    if x is None:
        return True
    if isinstance(x, float) and np.isnan(x):
        return True
    if isinstance(x, str) and x.strip() == "":
        return True
    return False


def _infer_visit_bases(wide_df: pd.DataFrame, visit: int) -> set[str]:
    sfx = f"_v{visit}"
    out = set()
    for c in wide_df.columns:
        if c.endswith(sfx):
            out.add(c[: -len(sfx)])
    return out


def _infer_imaging_bases(wide_df: pd.DataFrame) -> list[str]:
    """Infer imaging bases by excluding known clinical prefixes + benchmark scales."""
    bases = set.intersection(
        _infer_visit_bases(wide_df, 1),
        _infer_visit_bases(wide_df, 2),
        _infer_visit_bases(wide_df, 3),
    )

    def is_non_imaging(b: str) -> bool:
        if b in CLINICAL_SCORES:
            return True
        for p in _NON_IMAGING_BASE_PREFIXES:
            if b.startswith(p):
                return True
        return False

    imaging = [b for b in bases if not is_non_imaging(b)]
    return sorted(imaging)


def _has_any_imaging(row: pd.Series, bases: list[str], visit: int) -> bool:
    cols = [f"{b}_v{visit}" for b in bases if f"{b}_v{visit}" in row.index]
    if not cols:
        return False
    return row[cols].notna().any()


def build_trackfa_pairs_from_wide(
    wide_df: pd.DataFrame,
    *,
    subject_col: str = "ID",
) -> pd.DataFrame:
    """Build a consecutive-pair table from TRACK-FA wide format.

    Output columns:
      - identity: ID, pair
      - static: study_group, age, gender, gaa_1, gaa_2, onset_age, disease_duration, upenn_fxn_total
      - per-feature: {feat}_baseline, {feat}_followup, delta_{feat}

    Pair inclusion rule matches the prior spec: a pair is dropped if imaging is
    missing at either visit (defined as "no imaging feature present" at that visit).
    """
    if subject_col not in wide_df.columns:
        raise KeyError(f"wide_df missing subject column {subject_col!r}")

    imaging_bases = _infer_imaging_bases(wide_df)
    if not imaging_bases:
        raise ValueError("Could not infer imaging feature bases from wide_df.")

    static_cols = [
        "study_group",
        "age",
        "gender",
        "gaa_1",
        "gaa_2",
        "onset_age",
        "disease_duration",
        "upenn_fxn_total",
    ]
    static_cols = [c for c in static_cols if c in wide_df.columns]

    feature_bases = list(CLINICAL_SCORES) + imaging_bases

    rows: list[dict[str, Any]] = []
    for _, r in wide_df.iterrows():
        sid = r[subject_col]
        if _is_missing(sid):
            continue

        for v_base, v_follow, label in VISIT_PAIRS:
            if not (_has_any_imaging(r, imaging_bases, v_base) and _has_any_imaging(r, imaging_bases, v_follow)):
                continue

            out: dict[str, Any] = {subject_col: sid, "pair": label}
            for c in static_cols:
                out[c] = r.get(c)

            for b in feature_bases:
                cb = f"{b}_v{v_base}"
                cf = f"{b}_v{v_follow}"
                vb = r.get(cb, np.nan)
                vf = r.get(cf, np.nan)
                out[f"{b}_baseline"] = vb
                out[f"{b}_followup"] = vf
                try:
                    out[_delta_col(b)] = float(vf) - float(vb)
                except Exception:
                    out[_delta_col(b)] = np.nan

            rows.append(out)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Step 1 — Clinical score benchmarks
# ---------------------------------------------------------------------------
def compute_clinical_benchmarks(pairs_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute paired Cohen's d (SRM) for each clinical score.
    Report separately for V1V2 pairs, V2V3 pairs, and combined.

    Clinical scores: mfars_total, adl_total, sara_total, lcslc_total
    Formula: d = mean(delta_{score}) / std(delta_{score}, ddof=1)

    Returns DataFrame with columns: score, pair_type, n, cohens_d
    These are the benchmark targets the imaging composite must beat.
    """
    rows: list[dict[str, Any]] = []
    for score in CLINICAL_SCORES:
        dc = _delta_col(score)
        if dc not in pairs_df.columns:
            continue

        for pair_type, mask in [
            ("V1V2", pairs_df["pair"].eq("V1V2") if "pair" in pairs_df.columns else pd.Series(False, index=pairs_df.index)),
            ("V2V3", pairs_df["pair"].eq("V2V3") if "pair" in pairs_df.columns else pd.Series(False, index=pairs_df.index)),
            ("both", pd.Series(True, index=pairs_df.index)),
        ]:
            deltas = pairs_df.loc[mask, dc].to_numpy(dtype=float)
            out = compute_cohens_d(deltas)
            rows.append(
                {
                    "score": score,
                    "pair_type": pair_type,
                    "n": int(out["n"]),
                    "cohens_d": float(out["d"]),
                }
            )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["score", "pair_type"]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Step 2 — Feature selection
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class _OlsFit:
    coef: np.ndarray
    intercept: float
    sse: float
    n: int


def _fit_ols_aic_bic(X: np.ndarray, y: np.ndarray) -> tuple[float, float, _OlsFit]:
    """Fit OLS y ~ 1 + X and return (AIC, BIC, fit).

    Log-likelihood assumes Gaussian i.i.d. residuals with sigma^2 = SSE / n (MLE).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n = int(y.shape[0])
    if n < 3:
        return float("inf"), float("inf"), _OlsFit(np.zeros(X.shape[1]), 0.0, float("inf"), n)

    X1 = np.column_stack([np.ones((n, 1), dtype=float), X])
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    resid = y - (X1 @ beta)
    sse = float(np.sum(resid ** 2))
    sigma2 = sse / n if n > 0 else float("inf")
    if not np.isfinite(sigma2) or sigma2 <= 0:
        return float("inf"), float("inf"), _OlsFit(beta[1:], float(beta[0]), sse, n)

    logL = -0.5 * n * (np.log(2.0 * np.pi * sigma2) + 1.0)
    k = int(X1.shape[1])
    aic = float(-2.0 * logL + 2.0 * k)
    bic = float(-2.0 * logL + k * np.log(n))
    return aic, bic, _OlsFit(beta[1:], float(beta[0]), sse, n)


def _imaging_delta_cols(pairs_df: pd.DataFrame) -> list[str]:
    """delta_* columns excluding clinical deltas."""
    cols = []
    for c in pairs_df.columns:
        if not c.startswith("delta_"):
            continue
        base = c[len("delta_") :]
        if base in CLINICAL_SCORES:
            continue
        cols.append(c)
    return sorted(cols)


def select_features_aic_bic(
    pairs_df: pd.DataFrame,
    target_col: str = "delta_mfars_total",
    max_features: int = 30,
    *,
    cfg: Config = DEFAULT_CONFIG,
) -> list[str]:
    """
    AIC/BIC-penalised forward selection on delta_* imaging columns.
    Fit OLS on delta features predicting target_col.
    AIC = -2*log(L) + 2k,  BIC = -2*log(L) + k*log(n).
    Return ranked list of selected imaging feature base names.
    """
    if target_col not in pairs_df.columns:
        raise KeyError(f"Missing target column: {target_col}")

    candidate_cols = _imaging_delta_cols(pairs_df)
    if not candidate_cols:
        return []

    work = pairs_df[[target_col] + candidate_cols].replace([np.inf, -np.inf], np.nan).dropna(subset=[target_col]).copy()
    y = work[target_col].to_numpy(dtype=float)
    X = work[candidate_cols]
    X_imp = SimpleImputer(strategy="median").fit_transform(X)

    selected: list[int] = []
    remaining = list(range(X_imp.shape[1]))
    best_bic = float("inf")
    records: list[dict[str, Any]] = []

    for _ in range(min(int(max_features), len(remaining))):
        best_idx = None
        best_step_bic = float("inf")
        best_step_aic = float("inf")

        for j in remaining:
            idxs = selected + [j]
            aic, bic, _fit = _fit_ols_aic_bic(X_imp[:, idxs], y)
            if bic < best_step_bic:
                best_step_bic = bic
                best_step_aic = aic
                best_idx = j

        if best_idx is None:
            break
        if best_step_bic >= best_bic:
            break

        best_bic = best_step_bic
        selected.append(best_idx)
        remaining.remove(best_idx)

        base = candidate_cols[best_idx][len("delta_") :]
        records.append(
            {
                "rank": len(selected),
                "feature": base,
                "criterion": "BIC-forward",
                "aic": best_step_aic,
                "bic": best_step_bic,
                "n": int(len(y)),
            }
        )

    out_bases = [candidate_cols[i][len("delta_") :] for i in selected]

    if records:
        cfg.results_dir.mkdir(parents=True, exist_ok=True)
        path = cfg.results_dir / "trackfa_feature_selection.csv"
        prev = None
        if path.exists():
            try:
                prev = pd.read_csv(path)
            except Exception:
                prev = None
        new = pd.DataFrame(records)
        pd.concat([p for p in [prev, new] if p is not None], ignore_index=True).to_csv(path, index=False)

    return out_bases


def select_features_entropy(
    pairs_df: pd.DataFrame,
    target_col: str = "delta_mfars_total",
    k: int = 20,
    *,
    cfg: Config = DEFAULT_CONFIG,
) -> list[str]:
    """
    Rank imaging features by mutual information with target_col.
    Use sklearn mutual_info_regression on delta_* columns.
    Return top-k feature base names.
    """
    if target_col not in pairs_df.columns:
        raise KeyError(f"Missing target column: {target_col}")

    cols = _imaging_delta_cols(pairs_df)
    if not cols:
        return []

    work = pairs_df[[target_col] + cols].replace([np.inf, -np.inf], np.nan).dropna(subset=[target_col]).copy()
    y = work[target_col].to_numpy(dtype=float)
    X = SimpleImputer(strategy="median").fit_transform(work[cols])

    mi = mutual_info_regression(X, y, random_state=42)
    order = np.argsort(mi)[::-1]
    top = order[: max(0, int(k))]
    out_bases = [cols[i][len("delta_") :] for i in top]

    if len(out_bases) > 0:
        cfg.results_dir.mkdir(parents=True, exist_ok=True)
        path = cfg.results_dir / "trackfa_feature_selection.csv"
        prev = None
        if path.exists():
            try:
                prev = pd.read_csv(path)
            except Exception:
                prev = None
        new = pd.DataFrame(
            [
                {
                    "rank": r + 1,
                    "feature": cols[i][len("delta_") :],
                    "criterion": "MI-entropy",
                    "mi": float(mi[i]),
                    "n": int(len(y)),
                }
                for r, i in enumerate(top)
            ]
        )
        pd.concat([p for p in [prev, new] if p is not None], ignore_index=True).to_csv(path, index=False)

    return out_bases


# ---------------------------------------------------------------------------
# Step 3 — Subject-level LOO-CV
# ---------------------------------------------------------------------------
def subject_loo_cv(
    pairs_df: pd.DataFrame,
    feature_cols: list[str],
    model_type: str = "lda",
    pair_type: str | None = None,
    *,
    subject_col: str = "ID",
) -> dict:
    """
    Subject-level Leave-One-Out cross-validation.

    CRITICAL: Both pairs of the same subject (V1V2 and V2V3) must be held
    out together. Uses LeaveOneGroupOut(groups=pairs_df[subject_col]).

    For each fold:
      - Fit median imputer + StandardScaler on train only; apply to test.
      - Train model on train delta features → composite score.
      - Apply to held-out subject's delta features → predicted composite delta.

    After all folds, compute Cohen's d on the full vector of predicted deltas.

    Models:
      lda:        LinearDiscriminantAnalysis; label V1V2=0, V2V3=1.
                  Uses LDA decision-function score as the composite.
      elasticnet: ElasticNetCV predicting delta_mfars_total.
                  Uses regression predictions as the composite score.
    """
    if subject_col not in pairs_df.columns:
        raise KeyError(f"pairs_df missing subject column {subject_col!r}")
    if "pair" not in pairs_df.columns:
        raise KeyError("pairs_df must include 'pair'")

    work = pairs_df.copy()
    if pair_type is not None:
        work = work[work["pair"] == pair_type].copy()

    if work.empty:
        return {"cohens_d": float("nan"), "n_pairs": 0, "predicted_deltas": np.array([]), "fold_details": []}

    # For regression we must drop rows with missing target (and keep X aligned).
    if model_type == "elasticnet":
        target = "delta_mfars_total"
        if target not in work.columns:
            raise KeyError(f"pairs_df missing {target!r} needed for elasticnet target")
        work = work.replace([np.inf, -np.inf], np.nan).dropna(subset=[target]).copy()
        if work.empty:
            return {"cohens_d": float("nan"), "n_pairs": 0, "predicted_deltas": np.array([]), "fold_details": []}

    missing = [c for c in feature_cols if c not in work.columns]
    if missing:
        raise KeyError(f"Missing feature columns: {missing[:5]}{'...' if len(missing) > 5 else ''}")

    X = work[feature_cols].replace([np.inf, -np.inf], np.nan)
    groups = work[subject_col].to_numpy()
    preds = np.full((len(work),), np.nan, dtype=float)
    fold_details: list[dict[str, Any]] = []

    splitter = LeaveOneGroupOut()

    if model_type not in {"lda", "elasticnet"}:
        raise ValueError(f"Unknown model_type: {model_type}")

    if model_type == "lda":
        if pair_type is not None:
            return {"cohens_d": float("nan"), "n_pairs": int(len(work)), "predicted_deltas": np.full((len(work),), np.nan), "fold_details": []}

        y = work["pair"].map({"V1V2": 0, "V2V3": 1}).to_numpy()
        if np.unique(y).size < 2:
            return {"cohens_d": float("nan"), "n_pairs": int(len(work)), "predicted_deltas": np.full((len(work),), np.nan), "fold_details": []}

        for fold_i, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups=groups)):
            pre = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
            X_train = pre.fit_transform(X.iloc[train_idx])
            X_test = pre.transform(X.iloc[test_idx])
            model = LinearDiscriminantAnalysis()
            model.fit(X_train, y[train_idx])
            score = np.asarray(model.decision_function(X_test)).reshape(-1)
            preds[test_idx] = score
            fold_details.append(
                {
                    "fold": fold_i,
                    "heldout_subject": str(work.iloc[test_idx[0]][subject_col]),
                    "n_train": int(len(train_idx)),
                    "n_test": int(len(test_idx)),
                }
            )

    if model_type == "elasticnet":
        target = "delta_mfars_total"
        if target not in work.columns:
            raise KeyError(f"pairs_df missing {target!r} needed for elasticnet target")
        y = work[target].to_numpy(dtype=float)

        import warnings

        for fold_i, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups=groups)):
            # leakage-safe preprocessing
            imputer = SimpleImputer(strategy="median")
            scaler = StandardScaler()
            X_train = scaler.fit_transform(imputer.fit_transform(X.iloc[train_idx]))
            X_test = scaler.transform(imputer.transform(X.iloc[test_idx]))

            model = ElasticNetCV(
                l1_ratio=[0.1, 0.5, 0.9],
                cv=5,
                random_state=42,
                max_iter=20000,
            )
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                model.fit(X_train, y[train_idx])
            preds[test_idx] = model.predict(X_test)
            fold_details.append(
                {
                    "fold": fold_i,
                    "heldout_subject": str(work.iloc[test_idx[0]][subject_col]),
                    "n_train": int(len(train_idx)),
                    "n_test": int(len(test_idx)),
                    "alpha": float(model.alpha_),
                    "l1_ratio": float(model.l1_ratio_),
                }
            )

    out = compute_cohens_d(preds)
    return {
        "cohens_d": float(out["d"]),
        "n_pairs": int(out["n"]),
        "predicted_deltas": preds,
        "fold_details": fold_details,
    }


# ---------------------------------------------------------------------------
# Step 4 — Feature group experiments
# ---------------------------------------------------------------------------
def run_feature_group_experiments(
    pairs_df: pd.DataFrame,
    selected_features: list[str],
) -> pd.DataFrame:
    """LOO-CV across feature group × model × pair_type combinations.

    Groups tested:
      imaging_only         selected imaging delta features
      imaging+demographics add age, gaa_1, gaa_2, disease_duration
      poms_only            delta features from POMs list only
      dti_only             delta features with base starting FA_/MD_/RD_/AD_
      morph_only           all other imaging delta features
    """
    imaging_cols = _imaging_delta_cols(pairs_df)
    imaging_bases = [c[len("delta_") :] for c in imaging_cols]
    imaging_base_set = set(imaging_bases)

    selected_bases = [b for b in selected_features if b in imaging_base_set]
    selected_delta_cols = [_delta_col(b) for b in selected_bases if _delta_col(b) in pairs_df.columns]

    poms_delta_cols = [_delta_col(b) for b in sorted(set(selected_bases).intersection(_POMS_BASES)) if _delta_col(b) in pairs_df.columns]
    dti_delta_cols = [c for c in imaging_cols if c[len("delta_") :].startswith(("FA_", "MD_", "RD_", "AD_"))]
    morph_delta_cols = [c for c in imaging_cols if c not in set(poms_delta_cols).union(dti_delta_cols)]

    demo_covars = [c for c in ["age", "gaa_1", "gaa_2", "disease_duration"] if c in pairs_df.columns]

    groups: dict[str, list[str]] = {
        "imaging_only": selected_delta_cols,
        "imaging+demographics": selected_delta_cols + demo_covars,
        "poms_only": poms_delta_cols,
        "dti_only": dti_delta_cols,
        "morph_only": morph_delta_cols,
    }

    rows: list[dict[str, Any]] = []
    for feature_group, cols in groups.items():
        for model in ("lda", "elasticnet"):
            for pair_type in ("V1V2", "V2V3", "both"):
                pt = None if pair_type == "both" else pair_type
                try:
                    res = subject_loo_cv(pairs_df, cols, model_type=model, pair_type=pt)
                    d = float(res["cohens_d"])
                    n = int(res["n_pairs"])
                except Exception:
                    d = float("nan")
                    n = 0
                rows.append(
                    {
                        "feature_group": feature_group,
                        "model": model,
                        "pair_type": pair_type,
                        "cohens_d": d,
                        "n": n,
                        "n_features": int(len(cols)),
                    }
                )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["feature_group", "model", "pair_type"]).reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# Step 5 — Results reporting (CSV + minimal PNG bar chart)
# ---------------------------------------------------------------------------
def _crc32(data: bytes) -> int:
    import zlib

    return zlib.crc32(data) & 0xFFFFFFFF


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    import struct

    length = struct.pack(">I", len(data))
    crc = struct.pack(">I", _crc32(chunk_type + data))
    return length + chunk_type + data + crc


def _write_simple_bar_png(
    df: pd.DataFrame,
    out_path: Path,
    *,
    value_col: str = "cohens_d",
    label_cols: tuple[str, str, str] = ("feature_group", "model", "pair_type"),
) -> None:
    """Write a minimal PNG bar chart without matplotlib/plotly dependencies."""
    import struct
    import zlib

    if df.empty or value_col not in df.columns:
        return

    values = df[value_col].to_numpy(dtype=float)
    if not np.isfinite(values).any():
        return

    width = 1200
    bar_h = 18
    top = 30
    left = 240
    right = 20
    bottom = 20
    height = max(80, top + bottom + bar_h * len(values))

    img = np.ones((height, width, 4), dtype=np.uint8) * 255

    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))
    span = max(abs(vmin), abs(vmax), 1e-6)
    x0 = int(left + (0.0 + span) / (2.0 * span) * (width - left - right))
    img[:, x0 : x0 + 2, :] = np.array([0, 0, 0, 255], dtype=np.uint8)

    for i, v in enumerate(values):
        if not np.isfinite(v):
            continue
        y1 = top + i * bar_h + 3
        y2 = y1 + bar_h - 6
        x1 = x0
        x2 = int(left + (v + span) / (2.0 * span) * (width - left - right))
        if x2 < x1:
            x1, x2 = x2, x1
        x1 = max(left, min(width - right, x1))
        x2 = max(left, min(width - right, x2))
        color = np.array([40, 120, 220, 255], dtype=np.uint8) if v >= 0 else np.array([220, 80, 80, 255], dtype=np.uint8)
        img[y1:y2, x1:x2, :] = color
        img[y1:y2, 0:left, :] = np.array([245, 245, 245, 255], dtype=np.uint8)

    raw = b"".join([b"\x00" + img[y].tobytes() for y in range(img.shape[0])])
    compressed = zlib.compress(raw, level=6)

    signature = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    png = signature + _png_chunk(b"IHDR", ihdr) + _png_chunk(b"IDAT", compressed) + _png_chunk(b"IEND", b"")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(png)


def report_results(
    benchmarks: pd.DataFrame,
    experiments: pd.DataFrame,
    save_path: Path | None = None,
    *,
    cfg: Config = DEFAULT_CONFIG,
) -> None:
    """
    Print formatted comparison table:
      - Clinical score benchmarks (targets to beat)
      - Imaging composite Cohen's d per group/model/pair_type
      - Flag combinations that beat the best clinical score benchmark
    Save to CSV (+ minimal PNG bar chart).
    """
    cfg.results_dir.mkdir(parents=True, exist_ok=True)

    bench_path = cfg.results_dir / "trackfa_clinical_benchmarks.csv"
    benchmarks.to_csv(bench_path, index=False)

    if save_path is None:
        save_path = cfg.results_dir / "trackfa_pipeline_results.csv"

    exp = experiments.copy()

    best_by_pair = (
        benchmarks.groupby("pair_type", dropna=False)["cohens_d"].max().to_dict()
        if not benchmarks.empty
        else {}
    )
    best_overall = float(best_by_pair.get("both", np.nan))

    def _flag(row: pd.Series) -> bool:
        target = best_by_pair.get(row["pair_type"], best_overall)
        try:
            return bool(np.isfinite(row["cohens_d"]) and np.isfinite(target) and row["cohens_d"] > target)
        except Exception:
            return False

    exp["beats_best_clinical"] = exp.apply(_flag, axis=1) if (not exp.empty and best_by_pair) else False
    exp.to_csv(save_path, index=False)

    _write_simple_bar_png(
        exp.sort_values(["pair_type", "cohens_d"], ascending=[True, False]).reset_index(drop=True),
        cfg.results_dir / "trackfa_feature_group_comparison.png",
    )

    print("── Clinical score benchmarks (SRM) ───────────────────────────────")
    if benchmarks.empty:
        print("No clinical benchmarks computed.")
    else:
        best_rows = (
            benchmarks.sort_values("cohens_d", ascending=False)
            .groupby("pair_type", as_index=False)
            .head(1)
            .sort_values("pair_type")
        )
        for _, r in best_rows.iterrows():
            print(f"{r['pair_type']:>4} | best: {r['score']:<12}  d={float(r['cohens_d']):.3f}  (n={int(r['n'])})")

    print("\n── Imaging composite experiments (SRM) ────────────────────────────")
    if exp.empty:
        print("No experiments run.")
        return
    best_exp = exp.sort_values("cohens_d", ascending=False).head(10)
    for _, r in best_exp.iterrows():
        flag = "YES" if bool(r.get("beats_best_clinical", False)) else "no"
        print(
            f"{r['pair_type']:>4} | {r['feature_group']:<20} | {r['model']:<10}"
            f" d={float(r['cohens_d']): .3f} (n={int(r['n'])})  beats_best={flag}"
        )


__all__ = [
    "build_trackfa_pairs_from_wide",
    "compute_clinical_benchmarks",
    "select_features_aic_bic",
    "select_features_entropy",
    "subject_loo_cv",
    "run_feature_group_experiments",
    "report_results",
]
