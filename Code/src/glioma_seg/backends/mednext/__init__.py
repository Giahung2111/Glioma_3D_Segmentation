"""Source-pinned MedNeXt v1 project backend."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from glioma_seg.backends.mednext.backend import MedNeXtBackend

__all__ = ["MedNeXtBackend"]


def __getattr__(name: str) -> Any:
    if name == "MedNeXtBackend":
        from glioma_seg.backends.mednext.backend import MedNeXtBackend

        return MedNeXtBackend
    raise AttributeError(name)
