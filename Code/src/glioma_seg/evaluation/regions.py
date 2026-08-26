"""BraTS 2023 GLI label and nested-region conversions.

BraTS 2023 uses integer labels 0/1/2/3.  In particular, enhancing tumor is
label 3, not the label 4 used by older BraTS releases.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

REGION_ORDER: tuple[str, ...] = ("ET", "TC", "WT")
"""Canonical metric/report order used throughout this project."""

REGION_LABELS: Mapping[str, tuple[int, ...]] = MappingProxyType(
    {
        "ET": (3,),
        "TC": (1, 3),
        "WT": (1, 2, 3),
    }
)
VALID_BRATS_LABELS = frozenset({0, 1, 2, 3})


def validate_brats_labels(labels: ArrayLike, *, name: str = "labels") -> NDArray[np.integer]:
    """Validate and return an integer BraTS 2023 label array.

    Floating arrays are accepted only when every finite value is exactly an
    integer.  NaN/inf and the legacy ET label 4 are rejected.
    """

    array = np.asarray(labels)
    if array.ndim < 1:
        raise ValueError(f"{name} must have at least one dimension")
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{name} must be numeric, got {array.dtype}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinite values")
    if not np.issubdtype(array.dtype, np.integer) and not np.all(array == np.rint(array)):
        raise ValueError(f"{name} contains non-integer label values")

    integer = array.astype(np.int16, copy=False)
    observed = {int(value) for value in np.unique(integer)}
    unexpected = sorted(observed - VALID_BRATS_LABELS)
    if unexpected:
        raise ValueError(f"{name} contains labels outside BraTS 2023 {{0,1,2,3}}: {unexpected}")
    return integer


def regions_from_labels(labels: ArrayLike) -> dict[str, NDArray[np.bool_]]:
    """Convert a BraTS 2023 integer label map to ET, TC and WT masks."""

    integer = validate_brats_labels(labels)
    return {
        region: np.isin(integer, region_labels) for region, region_labels in REGION_LABELS.items()
    }


def assert_nested_regions(regions: Mapping[str, ArrayLike]) -> None:
    """Raise when masks do not satisfy ET subset TC subset WT."""

    missing = [region for region in REGION_ORDER if region not in regions]
    if missing:
        raise KeyError(f"Missing region masks: {missing}")
    et = np.asarray(regions["ET"], dtype=bool)
    tc = np.asarray(regions["TC"], dtype=bool)
    wt = np.asarray(regions["WT"], dtype=bool)
    if et.shape != tc.shape or tc.shape != wt.shape:
        raise ValueError(f"Region shapes differ: ET={et.shape}, TC={tc.shape}, WT={wt.shape}")
    et_outside_tc = int(np.count_nonzero(et & ~tc))
    tc_outside_wt = int(np.count_nonzero(tc & ~wt))
    if et_outside_tc or tc_outside_wt:
        raise ValueError(
            "Nested-region invariant violated: "
            f"ET outside TC={et_outside_tc} voxels, TC outside WT={tc_outside_wt} voxels"
        )


def _as_mask(values: ArrayLike, threshold: float) -> NDArray[np.bool_]:
    array = np.asarray(values)
    if not np.issubdtype(array.dtype, np.number) and array.dtype != np.bool_:
        raise TypeError(f"Region mask/probability must be numeric or bool, got {array.dtype}")
    if not np.all(np.isfinite(array)):
        raise ValueError("Region mask/probability contains NaN or infinite values")
    if array.dtype == np.bool_:
        return cast(NDArray[np.bool_], np.asarray(array, dtype=np.bool_).copy())
    return array >= threshold


def regions_to_brats(
    wt: ArrayLike,
    tc: ArrayLike,
    et: ArrayLike,
    *,
    threshold: float = 0.5,
    enforce_nested: bool = True,
    dtype: np.dtype | type = np.uint8,
) -> NDArray[np.integer]:
    """Reconstruct BraTS 2023 labels from nnU-Net region outputs.

    nnU-Net's region channel order is WT, TC, ET and the corresponding class
    order is 2, 1, 3.  Reconstruction therefore writes WT as ED (2), then TC
    as NCR (1), and finally ET (3).  By default, union operations repair
    threshold-induced nesting violations before reconstruction.
    """

    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")
    wt_mask = _as_mask(wt, threshold)
    tc_mask = _as_mask(tc, threshold)
    et_mask = _as_mask(et, threshold)
    if wt_mask.shape != tc_mask.shape or tc_mask.shape != et_mask.shape:
        raise ValueError(
            f"Region shapes differ: WT={wt_mask.shape}, TC={tc_mask.shape}, ET={et_mask.shape}"
        )

    if enforce_nested:
        tc_mask |= et_mask
        wt_mask |= tc_mask
    else:
        assert_nested_regions({"ET": et_mask, "TC": tc_mask, "WT": wt_mask})

    labels = np.zeros(wt_mask.shape, dtype=dtype)
    labels[wt_mask] = 2
    labels[tc_mask] = 1
    labels[et_mask] = 3
    return labels


def stacked_regions_from_labels(
    labels: ArrayLike, *, order: Sequence[str] = REGION_ORDER
) -> NDArray[np.bool_]:
    """Return region masks stacked along a leading channel axis."""

    regions = regions_from_labels(labels)
    invalid = [name for name in order if name not in REGION_LABELS]
    if invalid:
        raise KeyError(f"Unknown regions: {invalid}")
    return np.stack([regions[name] for name in order], axis=0)
