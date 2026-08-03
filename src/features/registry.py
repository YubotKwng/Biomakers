"""Feature-group registry for multimodal FRDA biomarker models.

The groups separate background covariates, structural MRI measures, diffusion
MRI measures, and optional QSM features so experiments can compare single-
modality and multimodal composites consistently.
"""
from __future__ import annotations

from typing import Dict, List


# ---------------------------------------------------------------------------
# Base-name registry for long-format visit-level tables.
# ---------------------------------------------------------------------------
background: List[str] = [
    "age",
    "age_at_onset",
    "dur",
    "sex",
    "GAA1",
    "GAA2",
]

structural: List[str] = [
    "SCP", "MCP", "ICP",
    "Midbrain", "Pons", "Medulla",
    "AntCBLM", "SupPostCBLM", "InfPostCBLM",
    "FlocCBLM", "VermisCBLM",
]

structural_ext: List[str] = structural + ["CSA_C1", "CSA_C2", "ECC_C1", "ECC_C2"]

diffusion: List[str] = [
    "FASCP", "FAMCP", "FAICP",
    "MDSCP", "MDMCP", "MDICP",
    "ADSCP", "ADMCP", "ADICP",
    "RDSCP", "RDMCP", "RDICP",
]

qsm: List[str] = []  # Optional modality; empty when no QSM features are available.


FEATURE_GROUPS: Dict[str, List[str]] = {
    "background": list(background),
    "structural": list(structural),
    "structural_ext": list(structural_ext),
    "diffusion": list(diffusion),
}

# Flat combinations used by feature-selection and progression models.
FEATURE_COMBOS: Dict[str, List[str]] = {
    "background": list(background),
    "structural": list(structural),
    "structural_ext": list(structural_ext),
    "diffusion": list(diffusion),
    "background_diffusion": background + diffusion,
    "background_structural": background + structural,
    "background_structural_ext": background + structural_ext,
    "structural_diffusion": structural + diffusion,
    "structural_ext_diffusion": structural_ext + diffusion,
    "background_structural_diffusion": background + structural + diffusion,
    "background_structural_ext": background + structural_ext,
    "background_structural_ext_diffusion": background + structural_ext + diffusion,
}

# Global pool used for entropy-selected subsets.
ALL_FEATURES: List[str] = sorted(set(background + structural_ext + diffusion))


# ---------------------------------------------------------------------------
# Visit-1-suffix registry for wide baseline prediction tables.
# ---------------------------------------------------------------------------
background_v1: List[str] = [
    "age1",
    "age2",
    "age_at_onset",
    "dur1",
    "dur2",
    "sex",
    "GAA1",
    "GAA2",
]

structural_v1: List[str] = [
    "SCP_v1", "MCP_v1", "ICP_v1",
    "Midbrain_v1", "Pons_v1", "Medulla_v1",
    "AntCBLM_v1", "SupPostCBLM_v1", "InfPostCBLM_v1",
    "FlocCBLM_v1", "VermisCBLM_v1",
]

diffusion_v1: List[str] = [
    "FASCP_v1", "FAMCP_v1", "FAICP_v1",
    "MDSCP_v1", "MDMCP_v1", "MDICP_v1",
    "ADSCP_v1", "ADMCP_v1", "ADICP_v1",
    "RDSCP_v1", "RDMCP_v1", "RDICP_v1",
]

FEATURE_GROUPS_V1: Dict[str, List[str]] = {
    "background": list(background_v1),
    "structural": list(structural_v1),
    "diffusion": list(diffusion_v1),
    "qsm": list(qsm),
}


PAPER_COMBINATIONS = [
    {"name": "background_only",                   "domains": [background_v1],                                            "skip": False},
    {"name": "structural_only",                   "domains": [structural_v1],                                            "skip": False},
    {"name": "diffusion_only",                    "domains": [diffusion_v1],                                             "skip": False},
    {"name": "qsm_only",                          "domains": [qsm],                                                       "skip": True},
    {"name": "all_neuroimaging",                  "domains": [structural_v1, diffusion_v1],                              "skip": False},
    {"name": "background_structural",             "domains": [background_v1, structural_v1],                             "skip": False},
    {"name": "background_diffusion",              "domains": [background_v1, diffusion_v1],                              "skip": False},
    {"name": "background_qsm",                    "domains": [background_v1, qsm],                                       "skip": True},
    {"name": "background_structural_diffusion",   "domains": [background_v1, structural_v1, diffusion_v1],               "skip": False},
    {"name": "background_all_neuroimaging",       "domains": [background_v1, structural_v1, diffusion_v1, qsm],          "skip": True},
]


__all__ = [
    "background",
    "structural",
    "structural_ext",
    "diffusion",
    "qsm",
    "FEATURE_GROUPS",
    "FEATURE_COMBOS",
    "ALL_FEATURES",
    "background_v1",
    "structural_v1",
    "diffusion_v1",
    "FEATURE_GROUPS_V1",
    "PAPER_COMBINATIONS",
]
