"""Tumor-extent-aware slice selection."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from glioma_seg.evaluation.regions import validate_brats_labels


def select_informative_slices(
    gt_labels: ArrayLike,
    pred_labels: ArrayLike,
    *,
    axis: int = 2,
    n_slices: int = 3,
) -> tuple[int, ...]:
    """Select spatially distributed tumor-containing slices.

    The maximum tumor-area slice is always included.  Remaining slices are
    sampled across the union of GT and prediction extent and then filled by
    descending tumor area.  If both maps are empty, the central slice is used.
    """

    gt = validate_brats_labels(gt_labels, name="ground truth")
    pred = validate_brats_labels(pred_labels, name="prediction")
    if gt.shape != pred.shape:
        raise ValueError(f"Label shapes differ: GT={gt.shape}, prediction={pred.shape}")
    if n_slices < 1:
        raise ValueError("n_slices must be positive")
    normalized_axis = axis % gt.ndim
    tumor = (gt != 0) | (pred != 0)
    reduction_axes = tuple(index for index in range(tumor.ndim) if index != normalized_axis)
    counts = np.count_nonzero(tumor, axis=reduction_axes)
    extent = np.flatnonzero(counts)
    if extent.size == 0:
        return (gt.shape[normalized_axis] // 2,)

    requested = min(n_slices, int(extent.size))
    selected: set[int] = {int(np.argmax(counts))}
    if requested > 1:
        targets = np.linspace(float(extent[0]), float(extent[-1]), requested)
        for target in targets:
            nearest = int(extent[np.argmin(np.abs(extent - target))])
            selected.add(nearest)
            if len(selected) >= requested:
                break
    if len(selected) < requested:
        for index in extent[np.argsort(counts[extent])[::-1]]:
            selected.add(int(index))
            if len(selected) >= requested:
                break
    return tuple(sorted(selected))
