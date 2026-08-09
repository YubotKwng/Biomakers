"""Helpers for logging model configurations optimized on paired d_z."""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


def optimization_row(
    *,
    model: str,
    params: Mapping | None,
    result: Mapping,
    objective: str = "d_score",
    feature_pool: str = "all_imaging",
    notes: str = "",
    runtime_sec: float | None = None,
) -> dict:
    """Build one tidy optimization-log row from a model result dictionary."""
    params = dict(params or {})
    row = {
        "model": model,
        "feature_pool": feature_pool,
        "objective": objective,
        "d_score": float(result.get("d_score", np.nan)),
        "dz_v1_v2": float(result.get("dz_v1_v2", np.nan)),
        "dz_v2_v3": float(result.get("dz_v2_v3", np.nan)),
        "mean_validation_annual_dz": float(result.get("mean_validation_annual_dz", np.nan)),
        "annual_interval_gap": float(result.get("annual_interval_gap", np.nan)),
        "p_progression": float(result.get("p_progression", np.nan)),
        "d_ci_low": float(result.get("d_ci_low", np.nan)),
        "d_ci_high": float(result.get("d_ci_high", np.nan)),
        "n_subjects": int(result.get("n_subjects", 0) or 0),
        "cv_mode": result.get("cv_mode", result.get("method", np.nan)),
        "cv_n_splits": result.get("cv_n_splits", np.nan),
        "split_group_col": result.get("split_group_col", np.nan),
        "n_split_groups": result.get("n_split_groups", np.nan),
        "runtime_sec": runtime_sec if runtime_sec is not None else result.get("runtime_sec", np.nan),
        "notes": notes,
    }
    for key, value in params.items():
        row[f"param_{key}"] = value
    return row


def optimization_log(rows: Sequence[Mapping], *, sort_by: str = "d_score") -> pd.DataFrame:
    """Return a sorted DataFrame of tried model configurations."""
    df = pd.DataFrame(list(rows))
    if df.empty or sort_by not in df.columns:
        return df
    return df.sort_values(sort_by, ascending=False).reset_index(drop=True)


def save_optimization_log(df: pd.DataFrame, path: str | Path) -> Path:
    """Persist an optimization log as CSV, creating parent directories."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return out


__all__ = ["optimization_row", "optimization_log", "save_optimization_log"]
