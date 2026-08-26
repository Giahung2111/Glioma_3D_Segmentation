"""Official nnU-Net v2 command-line backend."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from glioma_seg.backends.nnunet.backend import NNUNetV2Backend

__all__ = ["NNUNetV2Backend"]


def __getattr__(name: str) -> Any:
    # Avoid importing the CLI module while ``python -m ...backend`` initializes.
    if name == "NNUNetV2Backend":
        from glioma_seg.backends.nnunet.backend import NNUNetV2Backend

        return NNUNetV2Backend
    raise AttributeError(name)
