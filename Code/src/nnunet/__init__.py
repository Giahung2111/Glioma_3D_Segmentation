"""Compatibility alias for the source-pinned MedNeXt nnU-Net v1 fork.

MedNeXt installs its fork as ``nnunet_mednext`` but several unchanged upstream
modules still use legacy dynamic import strings rooted at ``nnunet``.  This
project-owned alias lets spawned Windows preprocessing workers import those
modules without editing the official MedNeXt checkout.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_upstream: Any = import_module("nnunet_mednext")

globals()["__path__"] = _upstream.__path__


def __getattr__(name: str) -> Any:
    return getattr(_upstream, name)
