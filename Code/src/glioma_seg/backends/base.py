"""Stable interface shared by present and future segmentation backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BackendArtifacts:
    """Files produced by a backend operation.

    Paths are deliberately backend-neutral so evaluation, reporting, and future
    ensemble code do not need to know an nnU-Net directory convention.
    """

    experiment_id: str
    checkpoint_paths: tuple[Path, ...] = ()
    prediction_dir: Path | None = None
    probability_dir: Path | None = None
    log_paths: tuple[Path, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


class SegmentationBackend(ABC):
    """Contract implemented by model-specific orchestration adapters."""

    @abstractmethod
    def prepare_dataset(self, *args: Any, **kwargs: Any) -> Any:
        """Prepare backend-specific input without mutating source data."""

    @abstractmethod
    def preprocess(self, *args: Any, **kwargs: Any) -> Any:
        """Run model-specific preprocessing."""

    @abstractmethod
    def train(self, *args: Any, **kwargs: Any) -> Any:
        """Train one fold/configuration."""

    @abstractmethod
    def predict(self, *args: Any, **kwargs: Any) -> Any:
        """Create predictions from an explicit trained fold or fold set."""

    @abstractmethod
    def evaluate(self, *args: Any, **kwargs: Any) -> Any:
        """Evaluate predictions through the common project evaluator."""

    @abstractmethod
    def get_artifacts(self, experiment_id: str) -> BackendArtifacts:
        """Return artifacts for reporting and downstream experiments."""

    @staticmethod
    def validate_folds(folds: Sequence[int]) -> tuple[int, ...]:
        normalized = tuple(int(fold) for fold in folds)
        if not normalized or any(fold not in range(5) for fold in normalized):
            raise ValueError("folds must be a non-empty subset of {0, 1, 2, 3, 4}")
        if len(set(normalized)) != len(normalized):
            raise ValueError("folds must not contain duplicates")
        return normalized
