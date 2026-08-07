"""Project-wide configuration for the FRDA biomarker pipeline.
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Resolve once at import time. Paths are relative to the project root
# (``biomarkers/``).

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Config:
    """Runtime settings for data loading, modelling, evaluation, and outputs.

    Defaults follow the repository layout. Callers can construct a fresh
    ``Config(...)`` when an experiment needs alternative paths, targets, or
    validation settings.
    """

    # ------------------------------------------------------------------
    # Filesystem
    # ------------------------------------------------------------------
    repo_root: Path = field(default_factory=lambda: _PROJECT_ROOT)
    raw_data_dir: Path = field(default_factory=lambda: _PROJECT_ROOT / "data" / "raw")
    processed_data_dir: Path = field(
        default_factory=lambda: _PROJECT_ROOT / "data" / "processed"
    )
    master_csv: Path = field(
        default_factory=lambda: _PROJECT_ROOT
        / "data"
        / "processed"
        / "melbourne_frda_merged.csv"
    )
    results_dir: Path = field(default_factory=lambda: _PROJECT_ROOT / "results")

    # ------------------------------------------------------------------
    # TRACK-FA inputs + outputs
    # ------------------------------------------------------------------
    trackfa_dir: Path = field(default_factory=lambda: _PROJECT_ROOT / "data" / "TRACK-FA")
    trackfa_redcap_csv: Path = field(
        default_factory=lambda: _PROJECT_ROOT
        / "data"
        / "TRACK-FA"
        / "ANaturalHistoryStudy-DataExportForYuboWan_DATA_2026-05-14_1551.csv"
    )
    trackfa_masterfile_xlsx: Path = field(
        default_factory=lambda: _PROJECT_ROOT
        / "data"
        / "TRACK-FA"
        / "MasterFile.v11b.POMs.BrainSpineMorph_ForYubo_18May2026.xlsx"
    )
    trackfa_masterfile_sheet: str | None = None
    trackfa_processed_dir: Path = field(
        default_factory=lambda: _PROJECT_ROOT / "data" / "processed"
    )
    trackfa_processed_csv: Path = field(
        default_factory=lambda: _PROJECT_ROOT
        / "data"
        / "processed"
        / "trackfa_merged_wide.csv"
    )
    trackfa_audit_csv: Path = field(
        default_factory=lambda: _PROJECT_ROOT
        / "data"
        / "processed"
        / "trackfa_merge_audit.csv"
    )
    trackfa_metadata_json: Path = field(
        default_factory=lambda: _PROJECT_ROOT
        / "data"
        / "processed"
        / "trackfa_merge_metadata.json"
    )

    # ------------------------------------------------------------------
    # Identifier columns / data conventions
    # ------------------------------------------------------------------
    subject_col: str = "ID"
    visit_col: str = "visit"
    visit_suffixes: tuple[str, str] = ("_v1", "_v2")

    # ------------------------------------------------------------------
    # RNG seeds for reproducible data processing, validation, and modelling.
    # ------------------------------------------------------------------
    random_state: int = 42
    n_boot: int = 201            # Bootstrap draws for coefficient summaries.
    n_boot_ci: int = 1000        # entropy bootstrap CI default

    # ------------------------------------------------------------------
    # Cross-validation
    # ------------------------------------------------------------------
    cv_n_splits: int = 5
    val_fraction: float = 0.2

    # ------------------------------------------------------------------
    # ElasticNet defaults for composite biomarker models.
    # ------------------------------------------------------------------
    en_alpha: float = 1.3
    en_l1_ratio: float = 0.0
    en_alpha_grid: tuple[float, ...] = (0.1, 0.3, 0.5, 0.7, 1.0, 1.3, 1.7, 2.0)
    en_l1_ratio_grid: tuple[float, ...] = (0.0, 0.2, 0.5, 0.8, 1.0)

    # ------------------------------------------------------------------
    # ElasticNet backend toggle for sklearn or local coordinate descent.
    # ------------------------------------------------------------------
    elasticnet_backend: str = "sklearn"  # {"sklearn", "cd"}

    # ------------------------------------------------------------------
    # Deep-learning defaults for progression-first neural models.
    # ------------------------------------------------------------------
    dl_max_epochs: int = 200
    dl_lr: float = 1e-3
    dl_weight_decay: float = 1e-4
    dl_grad_clip_norm: float = 1.0
    dl_eps: float = 1e-6

    # ------------------------------------------------------------------
    # Interaction-term composite for patient-adaptive weighting.
    # Modulators enter the design ONLY as multiplicative interaction terms
    # with imaging features (no raw main effects, by construction). Fit is
    # sparse ElasticNet on paired differences (Δdesign → 1).
    # ------------------------------------------------------------------
    modulators: list[str] = field(
        default_factory=lambda: ["GAA1", "age_at_onset", "dur"]
    )
    exclude_imaging: list[str] = field(
        default_factory=lambda: ["CSA_C1", "CSA_C2", "ECC_C1", "ECC_C2"]
    )
    interaction_en_alpha: float = 0.3       # ElasticNet alpha (λ) for Δdesign fit
    interaction_en_l1_ratio: float = 0.8    # ElasticNet l1_ratio (1.0=Lasso)
    interaction_en_alpha_grid: tuple[float, ...] = (
        0.01, 0.03, 0.1, 0.3, 0.7, 1.0, 1.3, 1.7, 2.0
    )
    interaction_en_l1_ratio_grid: tuple[float, ...] = (0.2, 0.5, 0.8, 1.0)
    interaction_tune_inner_cv: bool = True
    interaction_inner_cv_splits: int = 5
    interaction_z_clip: float | None = None  # Optional clipping after fold-local standardisation.
    interaction_eps: float = 1e-6           # SD floor in standardiser + loss
    interaction_n_boot: int = 201           # subgroup-importance resamples
    subgroups: dict[str, dict[str, float]] = field(
        default_factory=lambda: {
            "paediatric_lowGAA":  {"age_at_onset": 10.0, "GAA1": 500.0, "dur": 5.0},
            "paediatric_highGAA": {"age_at_onset": 10.0, "GAA1": 900.0, "dur": 5.0},
            "adult_lowGAA":       {"age_at_onset": 25.0, "GAA1": 500.0, "dur": 5.0},
            "adult_highGAA":      {"age_at_onset": 25.0, "GAA1": 900.0, "dur": 5.0},
        }
    )

    # ------------------------------------------------------------------
    # Bookkeeping
    # ------------------------------------------------------------------
    progress_verbose: bool = True


DEFAULT_CONFIG = Config()


# ---------------------------------------------------------------------------
# Determinism harness - call before stochastic processing or model fitting.
# ---------------------------------------------------------------------------
def set_global_seeds(seed: int = 42) -> None:
    """Seed the random-number generators used by the pipeline.

    Parameters
    ----------
    seed : int, default 42
        Master seed passed to Python, NumPy, and PyTorch when available.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch  # local import — torch is heavy and not always needed

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # ``use_deterministic_algorithms`` is a no-op on CPU but enforces
        # determinism on GPU paths where possible.
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:  # pragma: no cover — older torch versions
            pass
    except ImportError:
        pass
