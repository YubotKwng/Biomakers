"""Domain-defined feature panels for TRACK-FA biomarker modelling."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, List, Sequence

import pandas as pd


APRIORI_STRUCTURAL_BRAIN_FEATURES: List[str] = [
    "Cerebellum_WM_CerebNet",
    "Cerebellum_Cortex_CerebNet",
    "SCP",
    "Medulla",
    "Pons",
    "Midbrain",
    "TotalBrainGMVol_nocereb",
    "TotalBrainWMVol_nocereb",
    "Lateral_Ventricle",
    "Caudate",
    "Pallidum",
    "Putamen",
    "Thalamus",
]

APRIORI_SPINAL_STRUCTURAL_CANDIDATES: List[str] = [
    "bt1CSA_C12_UMN",
    "bt2CSA_C12_UMN",
    "sCSA_C12_UMN",
]

# Provisional choice until PG/supervisor confirmation. This keeps the panel
# usable while preserving an explicit audit trail for the unresolved item.
DEFAULT_APRIORI_SPINAL_STRUCTURAL_FEATURE = "sCSA_C12_UMN"

APRIORI_BRAIN_DTI_TRACTS: List[str] = [
    "ACR",
    "ALIC",
    "CP",
    "CST",
    "Cing",
    "Cing_h",
    "EC",
    "Fx",
    "Fx_ST",
    "ICP",
    "ILF_IFOF",
    "MCP",
    "PCR",
    "PCT",
    "PLIC",
    "PTR",
    "RLIC",
    "SCP",
    "SCR",
    "SFOF",
    "SLF",
    "Tap",
    "UNC",
    "bCC",
    "gCC",
    "mLEM",
    "sCC",
]

APRIORI_DTI_TRACT_GROUPS: dict[str, List[str]] = {
    "diffusion_cerebellum_brainstem": ["SCP", "MCP", "ICP", "CP", "mLEM", "PCT"],
    "diffusion_projection": ["ACR", "SCR", "PCR", "ALIC", "PLIC", "RLIC", "PTR", "CST"],
    "diffusion_association": [
        "Cing",
        "Cing_h",
        "EC",
        "Fx",
        "Fx_ST",
        "ILF_IFOF",
        "SFOF",
        "SLF",
        "UNC",
    ],
    "diffusion_commissural": ["bCC", "gCC", "sCC", "Tap"],
}

APRIORI_SPINAL_DTI_FEATURES: List[str] = ["sFA_c3c5", "sRD_c3c5"]

APRIORI_STRUCTURAL_PANEL_FEATURES: dict[str, List[str]] = {
    "structural_cerebellum_brainstem": [
        "Cerebellum_WM_CerebNet",
        "Cerebellum_Cortex_CerebNet",
        "SCP",
        "Medulla",
        "Pons",
        "Midbrain",
    ],
    "structural_cerebrum": [
        "TotalBrainGMVol_nocereb",
        "TotalBrainWMVol_nocereb",
        "Lateral_Ventricle",
        "Caudate",
        "Pallidum",
        "Putamen",
        "Thalamus",
    ],
}


@dataclass(frozen=True)
class FeaturePanel:
    """Resolved feature panel plus a dataset-column audit."""

    name: str
    features: List[str]
    audit: pd.DataFrame

    @property
    def missing_features(self) -> List[str]:
        return self.audit.loc[~self.audit["present"], "feature"].tolist()


def _available_long_features(columns: Iterable[str]) -> set[str]:
    """Return base feature names available in a wide or long TRACK-FA table."""
    available = set(columns)
    for col in columns:
        if col.endswith("_baseline"):
            available.add(col[: -len("_baseline")])
        elif col.endswith("_followup"):
            available.add(col[: -len("_followup")])
        elif col.startswith("delta_"):
            available.add(col[len("delta_") :])
    return available


def a_priori_70_feature_names(
    *,
    spinal_structural_feature: str = DEFAULT_APRIORI_SPINAL_STRUCTURAL_FEATURE,
) -> List[str]:
    """Return Ian/Susmita's 70-feature a priori TRACK-FA panel.

    The panel is 13 brain structural features, one provisional spinal
    structural feature, 27 FA brain DTI tracts, 27 RD brain DTI tracts, and
    two spinal DTI features.
    """
    if spinal_structural_feature not in APRIORI_SPINAL_STRUCTURAL_CANDIDATES:
        raise ValueError(
            "spinal_structural_feature must be one of "
            f"{APRIORI_SPINAL_STRUCTURAL_CANDIDATES!r}; got {spinal_structural_feature!r}"
        )

    brain_fa = [f"FA_{tract}" for tract in APRIORI_BRAIN_DTI_TRACTS]
    brain_rd = [f"RD_{tract}" for tract in APRIORI_BRAIN_DTI_TRACTS]
    features = (
        list(APRIORI_STRUCTURAL_BRAIN_FEATURES)
        + [spinal_structural_feature]
        + brain_fa
        + brain_rd
        + list(APRIORI_SPINAL_DTI_FEATURES)
    )
    if len(features) != 70:
        raise AssertionError(f"A priori panel should contain 70 features, got {len(features)}")
    if len(set(features)) != len(features):
        raise AssertionError("A priori panel contains duplicate feature names")
    return features


def a_priori_70_named_panels(
    *,
    spinal_structural_feature: str = DEFAULT_APRIORI_SPINAL_STRUCTURAL_FEATURE,
) -> dict[str, List[str]]:
    """Return the 70 features split into supervisor-facing MRI panels."""
    panels = {k: list(v) for k, v in APRIORI_STRUCTURAL_PANEL_FEATURES.items()}
    panels["structural_spinal"] = [spinal_structural_feature]
    for name, tracts in APRIORI_DTI_TRACT_GROUPS.items():
        panels[name] = [f"FA_{tract}" for tract in tracts] + [f"RD_{tract}" for tract in tracts]
    panels["diffusion_spinal"] = list(APRIORI_SPINAL_DTI_FEATURES)

    flat = [feature for features in panels.values() for feature in features]
    expected = a_priori_70_feature_names(spinal_structural_feature=spinal_structural_feature)
    if set(flat) != set(expected) or len(flat) != len(expected):
        raise AssertionError("Named a priori panels do not match the 70-feature panel")
    return panels


def a_priori_70_modality_panels(
    *,
    spinal_structural_feature: str = DEFAULT_APRIORI_SPINAL_STRUCTURAL_FEATURE,
) -> dict[str, List[str]]:
    """Return broad modality panels for 70-feature combination searches."""
    return {
        "brain_structural": list(APRIORI_STRUCTURAL_BRAIN_FEATURES),
        "spinal_structural": [spinal_structural_feature],
        "brain_diffusion_fa": [f"FA_{tract}" for tract in APRIORI_BRAIN_DTI_TRACTS],
        "brain_diffusion_rd": [f"RD_{tract}" for tract in APRIORI_BRAIN_DTI_TRACTS],
        "spinal_diffusion_fa": ["sFA_c3c5"],
        "spinal_diffusion_rd": ["sRD_c3c5"],
    }


def a_priori_70_panels(
    *,
    family: str = "anatomical",
    spinal_structural_feature: str = DEFAULT_APRIORI_SPINAL_STRUCTURAL_FEATURE,
) -> dict[str, List[str]]:
    """Return named 70-feature panels for a requested grouping family."""
    family = str(family).lower()
    if family == "anatomical":
        return a_priori_70_named_panels(
            spinal_structural_feature=spinal_structural_feature
        )
    if family == "modality":
        return a_priori_70_modality_panels(
            spinal_structural_feature=spinal_structural_feature
        )
    raise ValueError("family must be one of {'anatomical', 'modality'}")


def a_priori_70_panel_table(
    columns: Sequence[str],
    *,
    family: str = "anatomical",
    spinal_structural_feature: str = DEFAULT_APRIORI_SPINAL_STRUCTURAL_FEATURE,
) -> pd.DataFrame:
    """Audit named panel sizes and dataset availability."""
    available = _available_long_features(columns)
    rows = []
    for panel_name, features in a_priori_70_panels(
        family=family,
        spinal_structural_feature=spinal_structural_feature
    ).items():
        present = [f for f in features if f in available]
        missing = [f for f in features if f not in available]
        rows.append(
            {
                "panel": panel_name,
                "n_features": len(features),
                "n_present": len(present),
                "n_missing": len(missing),
                "missing_features": ", ".join(missing),
                "features": ", ".join(features),
            }
        )
    return pd.DataFrame(rows)


def a_priori_70_panel_combinations(
    *,
    family: str = "anatomical",
    spinal_structural_feature: str = DEFAULT_APRIORI_SPINAL_STRUCTURAL_FEATURE,
    min_size: int = 1,
    max_size: int | None = None,
) -> list[dict]:
    """Generate every non-empty named-panel combination from the 70-feature panel."""
    panels = a_priori_70_panels(
        family=family,
        spinal_structural_feature=spinal_structural_feature,
    )
    names = list(panels)
    max_k = len(names) if max_size is None else min(int(max_size), len(names))
    combos = []
    for k in range(int(min_size), max_k + 1):
        for picked in combinations(names, k):
            features = sorted({feat for panel in picked for feat in panels[panel]})
            combos.append(
                {
                    "panel_combo": "+".join(picked),
                    "panel_family": family,
                    "panel_count": len(picked),
                    "panels": tuple(picked),
                    "features": features,
                    "feature_count": len(features),
                }
            )
    return combos


def resolve_a_priori_70_panel(
    columns: Sequence[str],
    *,
    spinal_structural_feature: str = DEFAULT_APRIORI_SPINAL_STRUCTURAL_FEATURE,
) -> FeaturePanel:
    """Resolve the 70-feature a priori panel against a TRACK-FA dataframe."""
    features = a_priori_70_feature_names(spinal_structural_feature=spinal_structural_feature)
    available = _available_long_features(columns)
    rows = []
    for feature in features:
        if feature in APRIORI_STRUCTURAL_BRAIN_FEATURES:
            group = "structural_brain"
            status = "confirmed"
        elif feature == spinal_structural_feature:
            group = "structural_spinal"
            status = "provisional_pg_confirmation_needed"
        elif feature in APRIORI_SPINAL_DTI_FEATURES:
            group = "diffusion_spinal"
            status = "confirmed"
        elif feature.startswith("FA_"):
            group = "diffusion_brain_fa"
            status = "confirmed"
        elif feature.startswith("RD_"):
            group = "diffusion_brain_rd"
            status = "confirmed"
        else:
            group = "unknown"
            status = "review"
        rows.append(
            {
                "feature": feature,
                "group": group,
                "present": feature in available,
                "status": status,
            }
        )

    audit = pd.DataFrame(rows)
    panel = FeaturePanel(name="a_priori_70", features=features, audit=audit)
    return panel


def spinal_structural_candidate_audit(columns: Sequence[str]) -> pd.DataFrame:
    """Show the unresolved spinal structural options from Ian's email."""
    available = _available_long_features(columns)
    return pd.DataFrame(
        [
            {
                "feature": feature,
                "present": feature in available,
                "used_in_a_priori_70": feature == DEFAULT_APRIORI_SPINAL_STRUCTURAL_FEATURE,
            }
            for feature in APRIORI_SPINAL_STRUCTURAL_CANDIDATES
        ]
    )


__all__ = [
    "APRIORI_BRAIN_DTI_TRACTS",
    "APRIORI_DTI_TRACT_GROUPS",
    "APRIORI_SPINAL_DTI_FEATURES",
    "APRIORI_SPINAL_STRUCTURAL_CANDIDATES",
    "APRIORI_STRUCTURAL_PANEL_FEATURES",
    "APRIORI_STRUCTURAL_BRAIN_FEATURES",
    "DEFAULT_APRIORI_SPINAL_STRUCTURAL_FEATURE",
    "FeaturePanel",
    "a_priori_70_modality_panels",
    "a_priori_70_named_panels",
    "a_priori_70_panels",
    "a_priori_70_panel_combinations",
    "a_priori_70_panel_table",
    "a_priori_70_feature_names",
    "resolve_a_priori_70_panel",
    "spinal_structural_candidate_audit",
]
