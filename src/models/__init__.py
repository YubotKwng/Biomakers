"""Public model API for FRDA biomarker analyses.

The package exposes model components for clinical-score prediction, visit
separation, progression-sensitive composites, and patient-adaptive weighting.
"""
from .interaction import InteractionLinearComposite

__all__ = ["InteractionLinearComposite"]
