"""Framework-neutral, explicitly invoked probability-ensemble interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class ModelProbabilities:
    """Named model/fold probabilities with a declared leading-channel order."""

    model_id: str
    probabilities: ArrayLike
    channel_names: tuple[str, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def validated(self) -> NDArray[np.floating]:
        array = np.asarray(self.probabilities, dtype=np.float32)
        if not self.model_id.strip():
            raise ValueError("model_id must be non-empty")
        if array.ndim < 2:
            raise ValueError(
                "Probabilities must have a leading channel axis and spatial dimensions"
            )
        if len(self.channel_names) != array.shape[0]:
            raise ValueError(
                f"channel_names has {len(self.channel_names)} entries but probabilities has "
                f"{array.shape[0]} channels"
            )
        if len(set(self.channel_names)) != len(self.channel_names):
            raise ValueError("channel_names must be unique")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"Probabilities for {self.model_id} contain NaN or infinite values")
        if np.any(array < 0) or np.any(array > 1):
            raise ValueError(f"Probabilities for {self.model_id} must lie in [0, 1]")
        return array


@dataclass(frozen=True)
class EnsembleResult:
    probabilities: NDArray[np.floating]
    channel_names: tuple[str, ...]
    member_ids: tuple[str, ...]
    method: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


class ProbabilityEnsembler(ABC):
    """Base class for future, opt-in probability combination experiments."""

    @abstractmethod
    def combine(self, members: Sequence[ModelProbabilities]) -> EnsembleResult:
        """Combine already aligned probabilities; implementations must validate inputs."""


def validate_members(
    members: Sequence[ModelProbabilities],
) -> tuple[list[NDArray[np.floating]], tuple[str, ...]]:
    if not members:
        raise ValueError("At least one ensemble member is required")
    arrays = [member.validated() for member in members]
    expected_shape = arrays[0].shape
    expected_channels = members[0].channel_names
    seen_ids: set[str] = set()
    for member, array in zip(members, arrays, strict=False):
        if member.model_id in seen_ids:
            raise ValueError(f"Duplicate ensemble member ID: {member.model_id}")
        seen_ids.add(member.model_id)
        if array.shape != expected_shape:
            raise ValueError(
                f"Probability shape mismatch for {member.model_id}: "
                f"{array.shape} vs {expected_shape}"
            )
        if member.channel_names != expected_channels:
            raise ValueError(
                f"Channel order mismatch for {member.model_id}: {member.channel_names} "
                f"vs {expected_channels}"
            )
    return arrays, expected_channels
