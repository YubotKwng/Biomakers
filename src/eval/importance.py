"""Variable-importance bootstraps.

The two share the "resample subjects, refit, summarise" skeleton but differ
in shape: paper bootstrap reports one row per feature, subgroup bootstrap
reports one row per (feature, subgroup) cell. Keeping them side-by-side is
intentional — the subgroup variant is bound to the interaction-term
composite (``models/interaction.py``) and uses ``imaging_weights(z*)``,
which has no equivalent on sklearn ElasticNet.
"""
from __future__ import annotations

from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler

from ..data.qc import tukey_outliers_mask


def bootstrap_importance(
    df: pd.DataFrame,
    target_col: str,
    combo: dict,
    n_boot: int = 201,
    *,
    alpha: float = 1.3,
    l1_ratio: float = 0.0,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Subject-resampled bootstrap of ElasticNet coefficients.

    Fits the model repeatedly on subject-resampled data and summarizes the
    coefficient distribution for each feature.

    Notes
    -----
    The bootstrap draws subjects (groups) with replacement, then keeps every
    row of every drawn subject. Each draw fits a fresh ``StandardScaler`` on
    the bootstrap sample (X and y) and a fresh ``ElasticNet`` on the
    standardised data. ``coef_`` is collected per draw and summarised as
    mean / 2.5 / 97.5 percentile / fraction non-zero.
    """
    feature_cols: list[str] = []
    for domain in combo["domains"]:
        feature_cols.extend(domain)

    sub = df.dropna(subset=feature_cols + [target_col]).copy()
    sub = sub[~tukey_outliers_mask(sub, feature_cols, k=3.0)].copy()

    group_col = (
        "subject" if "subject" in sub.columns else (
            "melb_id" if "melb_id" in sub.columns else (
                "ID" if "ID" in sub.columns else None)))
    if group_col is None:
        raise KeyError("No subject/group id column found for bootstrap")
    groups = sub[group_col].values
    unique_groups = np.unique(groups)

    X = sub[feature_cols].values
    y = sub[target_col].values

    coef_list = []

    for _ in range(n_boot):
        sampled_groups = np.random.choice(unique_groups, size=len(unique_groups), replace=True)
        mask = np.isin(groups, sampled_groups)
        Xb, yb = X[mask], y[mask]

        xs = StandardScaler()
        ys = StandardScaler()
        Xs = xs.fit_transform(Xb)
        ys_ = ys.fit_transform(yb.reshape(-1, 1)).ravel()

        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=5000, random_state=random_seed)
        model.fit(Xs, ys_)
        coef_list.append(model.coef_)

    coef_arr = np.vstack(coef_list)
    summary = {
        "feature": feature_cols,
        "coef_mean": coef_arr.mean(axis=0),
        "coef_p2_5": np.percentile(coef_arr, 2.5, axis=0),
        "coef_p97_5": np.percentile(coef_arr, 97.5, axis=0),
        "pct_nonzero": (coef_arr != 0).mean(axis=0),
    }
    return pd.DataFrame(summary)


def bootstrap_subgroup_importance(
    model_factory: Callable[[], "object"],
    X: pd.DataFrame,
    Z: pd.DataFrame,
    subject_id: pd.Series,
    visit: pd.Series,
    subgroups: Mapping[str, Mapping[str, float]],
    n_boot: int = 201,
    random_state: int = 42,
) -> pd.DataFrame:
    """Bootstrap β(z*) for every (feature, subgroup) cell.

    Estimate subgroup-specific imaging weights by subject bootstrap.
    For each bootstrap b: (1) resample SUBJECTS (not rows) with replacement,
    (2) refit a fresh ``model_factory()`` on all rows of the resampled
    subjects, (3) for every named subgroup z*, evaluate β(z*) via
    ``model.imaging_weights(z*)``. Aggregates across bootstraps to a tidy
    DataFrame with mean / 2.5 / 97.5 percentile / sign-consistency.

    The model_factory abstraction lets this work with any class exposing
    ``.fit(X, Z, subject_id, visit)`` and ``.imaging_weights(z_star)`` —
    today that's ``models.interaction.InteractionLinearComposite``.

    Parameters
    ----------
    model_factory : callable returning a fresh model instance
        Each call must return a fresh, unfit model.
    X, Z : pd.DataFrame
        Imaging and modulator feature tables (raw scale, long format).
    subject_id : pd.Series
        Subject id per row.
    visit : pd.Series
        Integer visit per row.
    subgroups : mapping[str, mapping[str, float]]
        ``{subgroup_name: {modulator_name: raw_value}}``.
    n_boot : int, default 201
        Number of bootstrap resamples.
    random_state : int, default 42
        Master seed for the bootstrap RNG.

    Returns
    -------
    pd.DataFrame
        Columns: ``feature, subgroup, mean, lo, hi, sign_consistency``.
    """
    rng = np.random.default_rng(random_state)
    subject_id = pd.Series(subject_id).reset_index(drop=True)
    visit = pd.Series(visit).reset_index(drop=True)
    X = X.reset_index(drop=True)
    Z = Z.reset_index(drop=True)

    unique_subjects = subject_id.unique()
    subject_to_rows: dict = {
        s: np.where(subject_id.values == s)[0] for s in unique_subjects
    }

    records: list[dict] = []

    for b in range(n_boot):
        sampled = rng.choice(unique_subjects, size=len(unique_subjects), replace=True)
        rows = np.concatenate([subject_to_rows[s] for s in sampled])
        # Bootstrapped IDs need to be unique per draw (a subject sampled
        # twice must look like two different subjects). Suffix the draw
        # index to enforce uniqueness while still keeping each subject's
        # two visits paired.
        boot_subject_id = []
        for k, s in enumerate(sampled):
            n_rows = len(subject_to_rows[s])
            boot_subject_id.extend([f"{s}__b{b}_k{k}"] * n_rows)
        boot_subject_id = pd.Series(boot_subject_id)
        boot_visit = visit.iloc[rows].reset_index(drop=True)
        boot_X = X.iloc[rows].reset_index(drop=True)
        boot_Z = Z.iloc[rows].reset_index(drop=True)

        model = model_factory()
        try:
            model.fit(boot_X, boot_Z, boot_subject_id, boot_visit)
        except Exception:
            # Skip pathological resamples (e.g. singular ridge system).
            continue

        for sg_name, z_star in subgroups.items():
            w = model.imaging_weights(dict(z_star))
            for feat, val in w.items():
                records.append({
                    "boot": b,
                    "feature": feat,
                    "subgroup": sg_name,
                    "beta": float(val),
                })

    if not records:
        return pd.DataFrame(
            columns=["feature", "subgroup", "mean", "lo", "hi", "sign_consistency"]
        )

    long = pd.DataFrame.from_records(records)
    grouped = long.groupby(["feature", "subgroup"])["beta"]
    summary = grouped.agg(
        mean="mean",
        lo=lambda s: float(np.percentile(s, 2.5)),
        hi=lambda s: float(np.percentile(s, 97.5)),
    ).reset_index()

    def _sign_consistency(s: pd.Series) -> float:
        med = float(np.median(s))
        if med == 0:
            return float("nan")
        med_sign = np.sign(med)
        return float((np.sign(s) == med_sign).mean())

    consistency = (
        grouped.apply(_sign_consistency)
        .rename("sign_consistency")
        .reset_index()
    )
    out = summary.merge(consistency, on=["feature", "subgroup"], how="left")
    return out


__all__ = ["bootstrap_importance", "bootstrap_subgroup_importance"]
