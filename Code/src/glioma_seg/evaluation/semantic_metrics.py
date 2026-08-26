"""Standard (non-lesion-wise) BraTS ET/TC/WT metrics.

HD95 is computed from bidirectional surface distances using SciPy's tested
Euclidean distance transform and the supplied physical voxel spacing.  These
metrics must not be described as the official BraTS lesion-wise metrics.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from math import prod
from typing import Any, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import ndimage  # type: ignore[import-untyped]

from .regions import REGION_ORDER, regions_from_labels, validate_brats_labels

EMPTY_BOTH = "both_empty"
GT_EMPTY = "gt_empty_pred_present"
PRED_EMPTY = "gt_present_pred_empty"
PRESENT_BOTH = "both_present"


@dataclass(frozen=True)
class RegionMetrics:
    """Metrics and explicit validity state for one region in one case."""

    dice: float
    hd95_mm: float
    gt_voxels: int
    pred_voxels: int
    gt_present: bool
    pred_present: bool
    gt_volume_mm3: float
    pred_volume_mm3: float
    empty_state: str
    dice_status: str
    hd95_status: str
    failure_type: str

    def as_flat_dict(self, region: str) -> dict[str, Any]:
        prefix = region.lower()
        return {
            f"dice_{prefix}": self.dice,
            f"hd95_{prefix}_mm": self.hd95_mm,
            f"gt_{prefix}_voxels": self.gt_voxels,
            f"pred_{prefix}_voxels": self.pred_voxels,
            f"{prefix}_gt_present": self.gt_present,
            f"{prefix}_pred_present": self.pred_present,
            f"gt_{prefix}_volume_mm3": self.gt_volume_mm3,
            f"pred_{prefix}_volume_mm3": self.pred_volume_mm3,
            f"{prefix}_empty_state": self.empty_state,
            f"dice_{prefix}_status": self.dice_status,
            f"hd95_{prefix}_status": self.hd95_status,
            f"{prefix}_failure_type": self.failure_type,
        }


@dataclass(frozen=True)
class CaseMetrics:
    """All standard region-wise metrics for one case."""

    case_id: str
    spacing_mm: tuple[float, ...]
    regions: Mapping[str, RegionMetrics]

    def __post_init__(self) -> None:
        missing = [region for region in REGION_ORDER if region not in self.regions]
        if missing:
            raise ValueError(f"Missing metrics for regions: {missing}")

    def as_flat_dict(self) -> dict[str, Any]:
        row: dict[str, Any] = {"case_id": self.case_id}
        for region in REGION_ORDER:
            row.update(self.regions[region].as_flat_dict(region))
        return row


def _binary(mask: ArrayLike, *, name: str) -> NDArray[np.bool_]:
    array = np.asarray(mask)
    if array.ndim < 1:
        raise ValueError(f"{name} must have at least one dimension")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return np.asarray(array, dtype=np.bool_)


def _validate_pair(gt: ArrayLike, pred: ArrayLike) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    gt_mask = _binary(gt, name="ground-truth mask")
    pred_mask = _binary(pred, name="prediction mask")
    if gt_mask.shape != pred_mask.shape:
        raise ValueError(
            f"Mask shapes differ: ground truth={gt_mask.shape}, prediction={pred_mask.shape}"
        )
    return gt_mask, pred_mask


def _validate_spacing(spacing_mm: Sequence[float], ndim: int) -> tuple[float, ...]:
    spacing = tuple(float(value) for value in spacing_mm)
    if len(spacing) != ndim:
        raise ValueError(f"Expected {ndim} spacing values, got {len(spacing)}")
    if not all(np.isfinite(value) and value > 0 for value in spacing):
        raise ValueError(f"Spacing must contain finite positive values, got {spacing}")
    return spacing


def dice_score(gt: ArrayLike, pred: ArrayLike) -> float:
    """Return Dice, using NaN when both masks are empty.

    A one-sided empty mask has Dice 0.  A both-empty region has no observed
    positive class and is excluded from aggregate means through NaN.
    """

    gt_mask, pred_mask = _validate_pair(gt, pred)
    gt_count = int(np.count_nonzero(gt_mask))
    pred_count = int(np.count_nonzero(pred_mask))
    denominator = gt_count + pred_count
    if denominator == 0:
        return float("nan")
    intersection = int(np.count_nonzero(gt_mask & pred_mask))
    return float(2.0 * intersection / denominator)


def _surface(mask: NDArray[np.bool_]) -> NDArray[np.bool_]:
    structure = ndimage.generate_binary_structure(mask.ndim, 1)
    eroded = ndimage.binary_erosion(mask, structure=structure, border_value=0)
    return cast(NDArray[np.bool_], mask & ~eroded)


def hd95_mm(gt: ArrayLike, pred: ArrayLike, spacing_mm: Sequence[float]) -> float:
    """Return symmetric 95th-percentile surface Hausdorff distance in mm.

    The result is NaN whenever either mask is empty; callers can use
    :func:`compute_region_metrics` to retain the corresponding failure state.
    """

    gt_mask, pred_mask = _validate_pair(gt, pred)
    spacing = _validate_spacing(spacing_mm, gt_mask.ndim)
    if not np.any(gt_mask) or not np.any(pred_mask):
        return float("nan")

    gt_surface = _surface(gt_mask)
    pred_surface = _surface(pred_mask)
    # EDT gives distance to the nearest zero.  Inverting a surface therefore
    # yields the distance to that surface in physical units.
    distance_to_gt = ndimage.distance_transform_edt(~gt_surface, sampling=spacing)
    distance_to_pred = ndimage.distance_transform_edt(~pred_surface, sampling=spacing)
    pred_to_gt = distance_to_gt[pred_surface]
    gt_to_pred = distance_to_pred[gt_surface]
    distances = np.concatenate((pred_to_gt, gt_to_pred))
    return float(np.percentile(distances, 95))


def _presence_state(gt_present: bool, pred_present: bool) -> tuple[str, str, str, str]:
    if gt_present and pred_present:
        return PRESENT_BOTH, "defined", "defined", "none_observed"
    if not gt_present and not pred_present:
        return EMPTY_BOTH, "undefined_both_empty", "undefined_both_empty", "both_empty"
    if not gt_present:
        return GT_EMPTY, "defined_zero", "undefined_gt_empty", "false_positive"
    return PRED_EMPTY, "defined_zero", "undefined_pred_empty", "false_negative"


def compute_region_metrics(
    gt: ArrayLike, pred: ArrayLike, spacing_mm: Sequence[float]
) -> RegionMetrics:
    """Compute one region's semantic metrics and empty-mask evidence."""

    gt_mask, pred_mask = _validate_pair(gt, pred)
    spacing = _validate_spacing(spacing_mm, gt_mask.ndim)
    gt_voxels = int(np.count_nonzero(gt_mask))
    pred_voxels = int(np.count_nonzero(pred_mask))
    gt_present = gt_voxels > 0
    pred_present = pred_voxels > 0
    empty_state, dice_status, hd95_status, failure_type = _presence_state(gt_present, pred_present)
    voxel_volume_mm3 = float(prod(spacing))
    return RegionMetrics(
        dice=dice_score(gt_mask, pred_mask),
        hd95_mm=hd95_mm(gt_mask, pred_mask, spacing),
        gt_voxels=gt_voxels,
        pred_voxels=pred_voxels,
        gt_present=gt_present,
        pred_present=pred_present,
        gt_volume_mm3=gt_voxels * voxel_volume_mm3,
        pred_volume_mm3=pred_voxels * voxel_volume_mm3,
        empty_state=empty_state,
        dice_status=dice_status,
        hd95_status=hd95_status,
        failure_type=failure_type,
    )


def evaluate_case(
    gt_labels: ArrayLike,
    pred_labels: ArrayLike,
    spacing_mm: Sequence[float],
    *,
    case_id: str,
) -> CaseMetrics:
    """Compute standard semantic ET/TC/WT metrics for one BraTS case."""

    gt_integer = validate_brats_labels(gt_labels, name="ground truth")
    pred_integer = validate_brats_labels(pred_labels, name="prediction")
    if gt_integer.shape != pred_integer.shape:
        raise ValueError(
            f"Label shapes differ for {case_id}: ground truth={gt_integer.shape}, "
            f"prediction={pred_integer.shape}"
        )
    spacing = _validate_spacing(spacing_mm, gt_integer.ndim)
    gt_regions = regions_from_labels(gt_integer)
    pred_regions = regions_from_labels(pred_integer)
    metrics = {
        region: compute_region_metrics(gt_regions[region], pred_regions[region], spacing)
        for region in REGION_ORDER
    }
    return CaseMetrics(case_id=case_id, spacing_mm=spacing, regions=metrics)


def _finite_mean(values: Iterable[float]) -> tuple[float, int, int]:
    array = np.asarray(list(values), dtype=float)
    finite = np.isfinite(array)
    valid = int(np.count_nonzero(finite))
    total = int(array.size)
    mean = float(np.mean(array[finite])) if valid else float("nan")
    return mean, valid, total


def summarize_cases(cases: Sequence[CaseMetrics]) -> list[dict[str, Any]]:
    """Aggregate cases with an explicit finite-value denominator per region."""

    rows: list[dict[str, Any]] = []
    for metric_name, attribute, unit, direction in (
        ("Dice", "dice", "", "higher_is_better"),
        ("HD95", "hd95_mm", "mm", "lower_is_better"),
    ):
        row: dict[str, Any] = {
            "metric": metric_name,
            "unit": unit,
            "direction": direction,
            "total_cases": len(cases),
        }
        for region in REGION_ORDER:
            mean, valid, total = _finite_mean(
                getattr(case.regions[region], attribute) for case in cases
            )
            row[region] = mean
            row[f"{region}_n_valid"] = valid
            row[f"{region}_n_excluded"] = total - valid
        rows.append(row)
    return rows


def per_case_fieldnames() -> list[str]:
    """Return the stable metrics_per_case.csv schema."""

    fields = ["case_id"]
    for stem in ("dice",):
        fields.extend(f"{stem}_{region.lower()}" for region in REGION_ORDER)
    fields.extend(f"hd95_{region.lower()}_mm" for region in REGION_ORDER)
    fields.extend(f"gt_{region.lower()}_voxels" for region in REGION_ORDER)
    fields.extend(f"pred_{region.lower()}_voxels" for region in REGION_ORDER)
    fields.extend(f"{region.lower()}_gt_present" for region in REGION_ORDER)
    fields.extend(f"{region.lower()}_pred_present" for region in REGION_ORDER)
    fields.extend(f"gt_{region.lower()}_volume_mm3" for region in REGION_ORDER)
    fields.extend(f"pred_{region.lower()}_volume_mm3" for region in REGION_ORDER)
    fields.extend(f"{region.lower()}_empty_state" for region in REGION_ORDER)
    fields.extend(f"dice_{region.lower()}_status" for region in REGION_ORDER)
    fields.extend(f"hd95_{region.lower()}_status" for region in REGION_ORDER)
    fields.extend(f"{region.lower()}_failure_type" for region in REGION_ORDER)
    return fields
