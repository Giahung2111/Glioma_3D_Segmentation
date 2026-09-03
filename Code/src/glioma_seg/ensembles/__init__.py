"""Opt-in interfaces for future ensemble experiments.

Nothing in this package is enabled by the standard first nnU-Net baseline.
"""

from .base import EnsembleResult, ModelProbabilities, ProbabilityEnsembler
from .nnunet_probabilities import (
    NNUNET_REGION_CHANNEL_ORDER,
    NNUNET_TO_CANONICAL_INDICES,
    load_nnunet_region_probabilities,
    validate_brats_region_probability_contract,
)
from .probability_mean import ProbabilityMeanEnsembler
from .region_weighted import RegionWeightedEnsembler, nested_region_masks

__all__ = [
    "EnsembleResult",
    "ModelProbabilities",
    "NNUNET_REGION_CHANNEL_ORDER",
    "NNUNET_TO_CANONICAL_INDICES",
    "ProbabilityEnsembler",
    "ProbabilityMeanEnsembler",
    "RegionWeightedEnsembler",
    "load_nnunet_region_probabilities",
    "nested_region_masks",
    "validate_brats_region_probability_contract",
]
