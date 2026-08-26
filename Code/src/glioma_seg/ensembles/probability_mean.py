"""Opt-in arithmetic probability averaging for future experiments."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .base import EnsembleResult, ModelProbabilities, ProbabilityEnsembler, validate_members


class ProbabilityMeanEnsembler(ProbabilityEnsembler):
    """Combine aligned members with equal or explicitly supplied weights."""

    def __init__(self, weights: Sequence[float] | None = None) -> None:
        self._weights = None if weights is None else tuple(float(value) for value in weights)

    def combine(self, members: Sequence[ModelProbabilities]) -> EnsembleResult:
        arrays, channel_names = validate_members(members)
        if self._weights is None:
            weights = np.ones(len(arrays), dtype=np.float64)
        else:
            if len(self._weights) != len(arrays):
                raise ValueError(
                    f"Received {len(self._weights)} weights for {len(arrays)} ensemble members"
                )
            weights = np.asarray(self._weights, dtype=np.float64)
        if not np.all(np.isfinite(weights)) or np.any(weights < 0) or float(np.sum(weights)) <= 0:
            raise ValueError(
                "Ensemble weights must be finite, non-negative, and have a positive sum"
            )
        normalized = weights / np.sum(weights)
        combined = np.zeros_like(arrays[0], dtype=np.float32)
        for weight, array in zip(normalized, arrays, strict=False):
            combined += np.float32(weight) * array
        return EnsembleResult(
            probabilities=combined,
            channel_names=channel_names,
            member_ids=tuple(member.model_id for member in members),
            method="weighted_probability_mean" if self._weights is not None else "probability_mean",
            metadata={"normalized_weights": normalized.tolist(), "opt_in_experiment": True},
        )
