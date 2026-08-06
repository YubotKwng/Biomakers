"""Public model API for FRDA biomarker analyses.

The package exposes model components for clinical-score prediction, visit
separation, progression-sensitive composites, and patient-adaptive weighting.
"""
from .interaction import InteractionLinearComposite
from .srm_global import SRMGlobalLinear, srm_global_loocv

__all__ = ["InteractionLinearComposite", "SRMGlobalLinear", "srm_global_loocv"]
