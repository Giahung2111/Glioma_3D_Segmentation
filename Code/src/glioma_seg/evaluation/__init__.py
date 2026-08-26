"""BraTS label conversion and evaluation utilities."""

from .regions import (
    REGION_LABELS,
    REGION_ORDER,
    assert_nested_regions,
    regions_from_labels,
    regions_to_brats,
    validate_brats_labels,
)
from .semantic_metrics import (
    CaseMetrics,
    RegionMetrics,
    dice_score,
    evaluate_case,
    hd95_mm,
    summarize_cases,
)

__all__ = [
    "CaseMetrics",
    "REGION_LABELS",
    "REGION_ORDER",
    "RegionMetrics",
    "assert_nested_regions",
    "dice_score",
    "evaluate_case",
    "hd95_mm",
    "regions_from_labels",
    "regions_to_brats",
    "summarize_cases",
    "validate_brats_labels",
]
