"""Opt-in interfaces for future ensemble experiments.

Nothing in this package is enabled by the standard first nnU-Net baseline.
"""

from .base import EnsembleResult, ModelProbabilities, ProbabilityEnsembler
from .probability_mean import ProbabilityMeanEnsembler
from .region_weighted import RegionWeightedEnsembler, nested_region_masks

__all__ = [
    "EnsembleResult",
    "ModelProbabilities",
    "ProbabilityEnsembler",
    "ProbabilityMeanEnsembler",
    "RegionWeightedEnsembler",
    "nested_region_masks",
]
