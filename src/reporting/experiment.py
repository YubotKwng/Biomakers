"""Persist reproducible experiment artifacts."""
from __future__ import annotations

import json
import platform
import subprocess
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in ("numpy", "pandas", "sklearn", "scipy", "torch"):
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", None)
        except Exception:
            versions[name] = None
    return versions


def _git_commit(repo_root: Path | None) -> str | None:
    if repo_root is None:
        return None
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def save_experiment_artifacts(
    output_dir: str | Path,
    *,
    config: Mapping | None = None,
    fold_assignments: pd.DataFrame | None = None,
    hyperparameter_results: pd.DataFrame | None = None,
    oof_predictions: pd.DataFrame | None = None,
    selected_features: pd.DataFrame | None = None,
    bootstrap_results: pd.DataFrame | None = None,
    metrics: Mapping | None = None,
    repo_root: str | Path | None = None,
) -> dict[str, Path]:
    """Save the standard artifact set required by the framework."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    repo_path = Path(repo_root) if repo_root is not None else None
    config_payload = dict(config or {})
    config_payload.setdefault("runtime", {})
    config_payload["runtime"].update({
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": _package_versions(),
        "git_commit": _git_commit(repo_path),
    })
    path = out / "config.json"
    path.write_text(json.dumps(config_payload, indent=2, default=_json_default) + "\n")
    paths["config"] = path

    frames = {
        "fold_assignments": fold_assignments,
        "hyperparameter_results": hyperparameter_results,
        "oof_predictions": oof_predictions,
        "selected_features": selected_features,
        "bootstrap_results": bootstrap_results,
    }
    for name, frame in frames.items():
        if frame is None:
            frame = pd.DataFrame()
        path = out / f"{name}.csv"
        frame.to_csv(path, index=False)
        paths[name] = path

    path = out / "metrics.json"
    path.write_text(json.dumps(dict(metrics or {}), indent=2, default=_json_default) + "\n")
    paths["metrics"] = path
    return paths


__all__ = ["save_experiment_artifacts"]
