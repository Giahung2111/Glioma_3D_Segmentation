"""Opt-in region-specific probability ensemble for future experiments."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from glioma_seg.evaluation.regions import REGION_ORDER

from .base import EnsembleResult, ModelProbabilities, ProbabilityEnsembler, validate_members


class RegionWeightedEnsembler(ProbabilityEnsembler):
    """Apply a separately declared member-weight vector to ET, TC and WT.

    Members must expose channels in canonical ``(ET, TC, WT)`` order.  The
    class is never selected automatically by baseline orchestration.
    """

    def __init__(self, weights: Mapping[str, Mapping[str, float]]) -> None:
        self._weights = {
            str(region).upper(): {
                str(model): float(weight) for model, weight in model_weights.items()
            }
            for region, model_weights in weights.items()
        }

    def combine(self, members: Sequence[ModelProbabilities]) -> EnsembleResult:
        arrays, channel_names = validate_members(members)
        if channel_names != REGION_ORDER:
            raise ValueError(
                f"Region-weighted ensemble requires channel order {REGION_ORDER}, "
                f"got {channel_names}"
            )
        member_index = {member.model_id: index for index, member in enumerate(members)}
        combined = np.zeros_like(arrays[0], dtype=np.float32)
        recorded_weights: dict[str, dict[str, float]] = {}
        for channel, region in enumerate(REGION_ORDER):
            if region not in self._weights:
                raise ValueError(f"Missing weights for region {region}")
            unknown = sorted(set(self._weights[region]) - set(member_index))
            missing = sorted(set(member_index) - set(self._weights[region]))
            if unknown or missing:
                raise ValueError(
                    f"Weights for {region} do not match member IDs; "
                    f"missing={missing}, unknown={unknown}"
                )
            weights = np.asarray(
                [self._weights[region][member.model_id] for member in members], dtype=np.float64
            )
            if (
                not np.all(np.isfinite(weights))
                or np.any(weights < 0)
                or float(np.sum(weights)) <= 0
            ):
                raise ValueError(
                    f"Weights for {region} must be finite, non-negative, and sum above zero"
                )
            weights /= np.sum(weights)
            for weight, array in zip(weights, arrays, strict=False):
                combined[channel] += np.float32(weight) * array[channel]
            recorded_weights[region] = {
                member.model_id: float(weight)
                for member, weight in zip(members, weights, strict=False)
            }
        return EnsembleResult(
            probabilities=combined,
            channel_names=REGION_ORDER,
            member_ids=tuple(member.model_id for member in members),
            method="region_weighted_probability_mean",
            metadata={"normalized_region_weights": recorded_weights, "opt_in_experiment": True},
        )


def nested_region_masks(
    region_probabilities: ArrayLike, *, threshold: float = 0.5
) -> dict[str, NDArray[np.bool_]]:
    """Threshold ET/TC/WT probabilities and enforce ET subset TC subset WT."""

    probabilities = np.asarray(region_probabilities)
    if probabilities.ndim < 2 or probabilities.shape[0] != len(REGION_ORDER):
        raise ValueError(
            f"Expected probabilities shaped (3, ...), channel order {REGION_ORDER}; "
            f"got {probabilities.shape}"
        )
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Region probabilities contain NaN or infinite values")
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")
    et = probabilities[0] >= threshold
    tc = (probabilities[1] >= threshold) | et
    wt = (probabilities[2] >= threshold) | tc
    return {"ET": et, "TC": tc, "WT": wt}
