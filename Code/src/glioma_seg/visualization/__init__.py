"""T1c/FLAIR failure-case visualization utilities."""

from typing import Any

__all__ = [
    "create_failure_figure",
    "create_failure_figure_from_nifti",
    "overlay_labels",
    "select_informative_slices",
]


def __getattr__(name: str) -> Any:
    """Keep the matplotlib-backed executable module lazy."""

    if name == "select_informative_slices":
        from .slices import select_informative_slices

        return select_informative_slices
    if name in {"create_failure_figure", "create_failure_figure_from_nifti", "overlay_labels"}:
        from . import overlays

        return getattr(overlays, name)
    raise AttributeError(name)
