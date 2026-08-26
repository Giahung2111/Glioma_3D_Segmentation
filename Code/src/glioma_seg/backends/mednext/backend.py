"""Interface placeholder; MedNeXt is intentionally not part of baseline one."""

from __future__ import annotations

from typing import Any, NoReturn

from glioma_seg.backends.base import BackendArtifacts, SegmentationBackend


class MedNeXtBackend(SegmentationBackend):
    """Reserve the common API without silently enabling research behavior."""

    _MESSAGE = (
        "MedNeXt is a future, separate baseline. It is not implemented or enabled "
        "for the standard nnU-Net preliminary experiment."
    )

    def _unavailable(self) -> NoReturn:
        raise NotImplementedError(self._MESSAGE)

    def prepare_dataset(self, *args: Any, **kwargs: Any) -> Any:
        self._unavailable()

    def preprocess(self, *args: Any, **kwargs: Any) -> Any:
        self._unavailable()

    def train(self, *args: Any, **kwargs: Any) -> Any:
        self._unavailable()

    def predict(self, *args: Any, **kwargs: Any) -> Any:
        self._unavailable()

    def evaluate(self, *args: Any, **kwargs: Any) -> Any:
        self._unavailable()

    def get_artifacts(self, experiment_id: str) -> BackendArtifacts:
        self._unavailable()
