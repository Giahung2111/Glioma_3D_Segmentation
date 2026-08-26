"""Observable, threshold-documented BraTS failure-pattern classification."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from math import prod
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import ndimage  # type: ignore[import-untyped]

from glioma_seg.evaluation.regions import REGION_ORDER, regions_from_labels, validate_brats_labels
from glioma_seg.evaluation.semantic_metrics import CaseMetrics, evaluate_case

from .case_ranking import load_metrics_csv, rank_worst_cases, select_representative_cases


@dataclass(frozen=True)
class FailureAnalysisConfig:
    """Operational (not clinical) thresholds used to flag observed patterns."""

    small_et_max_volume_mm3: float = 1000.0
    material_error_fraction: float = 0.25
    boundary_dice_ceiling: float = 0.90
    boundary_hd95_floor_mm: float = 3.0
    remote_component_distance_mm: float = 10.0
    min_component_volume_mm3: float = 10.0
    fragmentation_excess_components: int = 2

    def __post_init__(self) -> None:
        positive = (
            self.small_et_max_volume_mm3,
            self.material_error_fraction,
            self.boundary_hd95_floor_mm,
            self.remote_component_distance_mm,
            self.min_component_volume_mm3,
        )
        if not all(np.isfinite(value) and value > 0 for value in positive):
            raise ValueError("Failure-analysis thresholds must be finite and positive")
        if not 0 < self.boundary_dice_ceiling <= 1:
            raise ValueError("boundary_dice_ceiling must be in (0, 1]")
        if self.fragmentation_excess_components < 1:
            raise ValueError("fragmentation_excess_components must be positive")


@dataclass(frozen=True)
class FailureEvidence:
    case_id: str
    region: str
    failure_type: str
    observation: str
    possible_explanation: str
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "region": self.region,
            "failure_type": self.failure_type,
            "observation": self.observation,
            "possible_explanation": self.possible_explanation,
            **{f"evidence_{key}": value for key, value in self.evidence.items()},
        }


def _component_count(mask: NDArray[np.bool_]) -> tuple[int, NDArray[np.int32]]:
    structure = ndimage.generate_binary_structure(mask.ndim, 1)
    labeled, count = ndimage.label(mask, structure=structure)
    return int(count), labeled


def _add(
    failures: list[FailureEvidence],
    *,
    case_id: str,
    region: str,
    failure_type: str,
    observation: str,
    evidence: Mapping[str, Any],
) -> None:
    if any(item.region == region and item.failure_type == failure_type for item in failures):
        return
    failures.append(
        FailureEvidence(
            case_id=case_id,
            region=region,
            failure_type=failure_type,
            observation=f"Observed pattern: {observation}",
            possible_explanation=(
                "Possible explanation: this is an image-level error pattern; "
                "causality requires review of the source MRI and training evidence."
            ),
            evidence=dict(evidence),
        )
    )


def _one_sided_and_volume_errors(
    failures: list[FailureEvidence],
    case_id: str,
    region: str,
    gt: NDArray[np.bool_],
    pred: NDArray[np.bool_],
    voxel_volume_mm3: float,
    config: FailureAnalysisConfig,
) -> None:
    gt_count = int(np.count_nonzero(gt))
    pred_count = int(np.count_nonzero(pred))
    if gt_count and not pred_count:
        failure_type = (
            "small ET missed"
            if (region == "ET" and gt_count * voxel_volume_mm3 <= config.small_et_max_volume_mm3)
            else f"{region} false negative"
        )
        _add(
            failures,
            case_id=case_id,
            region=region,
            failure_type=failure_type,
            observation=(
                f"GT has {gt_count} voxels "
                f"({gt_count * voxel_volume_mm3:.1f} mm3) and prediction is empty"
            ),
            evidence={"gt_voxels": gt_count, "pred_voxels": pred_count},
        )
        return
    if pred_count and not gt_count:
        _add(
            failures,
            case_id=case_id,
            region=region,
            failure_type=f"{region} false positive",
            observation=(
                f"GT is empty and prediction has {pred_count} voxels "
                f"({pred_count * voxel_volume_mm3:.1f} mm3)"
            ),
            evidence={"gt_voxels": gt_count, "pred_voxels": pred_count},
        )
        return
    if not gt_count:
        return

    false_negative = int(np.count_nonzero(gt & ~pred))
    false_positive = int(np.count_nonzero(pred & ~gt))
    missed_fraction = false_negative / gt_count
    extra_fraction = false_positive / pred_count if pred_count else 0.0
    evidence = {
        "false_negative_voxels": false_negative,
        "false_positive_voxels": false_positive,
        "missed_gt_fraction": round(missed_fraction, 6),
        "extra_pred_fraction": round(extra_fraction, 6),
    }
    if missed_fraction >= config.material_error_fraction and false_negative > false_positive:
        _add(
            failures,
            case_id=case_id,
            region=region,
            failure_type=f"{region} under-segmentation",
            observation=(
                f"missed GT fraction {missed_fraction:.1%} exceeds the operational "
                f"{config.material_error_fraction:.0%} threshold and FN voxels exceed FP voxels"
            ),
            evidence=evidence,
        )
    if extra_fraction >= config.material_error_fraction and false_positive > false_negative:
        _add(
            failures,
            case_id=case_id,
            region=region,
            failure_type=f"{region} over-segmentation",
            observation=(
                f"extra prediction fraction {extra_fraction:.1%} exceeds the operational "
                f"{config.material_error_fraction:.0%} threshold and FP voxels exceed FN voxels"
            ),
            evidence=evidence,
        )


def _component_failures(
    failures: list[FailureEvidence],
    case_id: str,
    gt_wt: NDArray[np.bool_],
    pred_wt: NDArray[np.bool_],
    spacing_mm: tuple[float, ...],
    config: FailureAnalysisConfig,
) -> None:
    voxel_volume = float(prod(spacing_mm))
    gt_components, _ = _component_count(gt_wt)
    pred_components, pred_labeled = _component_count(pred_wt)
    if pred_components >= gt_components + config.fragmentation_excess_components:
        _add(
            failures,
            case_id=case_id,
            region="WT",
            failure_type="prediction fragmentation",
            observation=f"prediction has {pred_components} components versus {gt_components} in GT",
            evidence={"gt_components": gt_components, "pred_components": pred_components},
        )
    if not pred_components:
        return

    distance_to_gt = None
    if np.any(gt_wt):
        distance_to_gt = ndimage.distance_transform_edt(~gt_wt, sampling=spacing_mm)
    for component_id in range(1, pred_components + 1):
        component = pred_labeled == component_id
        component_voxels = int(np.count_nonzero(component))
        component_volume = component_voxels * voxel_volume
        if component_volume < config.min_component_volume_mm3 or np.any(component & gt_wt):
            continue
        min_distance = (
            float(np.min(distance_to_gt[component])) if distance_to_gt is not None else float("nan")
        )
        _add(
            failures,
            case_id=case_id,
            region="WT",
            failure_type="isolated false positive component",
            observation=(
                f"a disconnected prediction-only component has {component_voxels} voxels "
                f"({component_volume:.1f} mm3)"
            ),
            evidence={"component_voxels": component_voxels, "minimum_gt_distance_mm": min_distance},
        )
        if np.isfinite(min_distance) and min_distance >= config.remote_component_distance_mm:
            _add(
                failures,
                case_id=case_id,
                region="WT",
                failure_type="large Hausdorff caused by remote component",
                observation=(
                    f"a prediction-only component is at least {min_distance:.1f} mm from GT, "
                    f"above the operational {config.remote_component_distance_mm:.1f} mm threshold"
                ),
                evidence={
                    "minimum_gt_distance_mm": min_distance,
                    "component_voxels": component_voxels,
                },
            )


def _nested_inconsistency(
    failures: list[FailureEvidence],
    case_id: str,
    predicted_regions: Mapping[str, ArrayLike] | None,
) -> None:
    if predicted_regions is None:
        return
    missing = [region for region in REGION_ORDER if region not in predicted_regions]
    if missing:
        raise KeyError(f"Missing predicted region masks: {missing}")
    et = np.asarray(predicted_regions["ET"], dtype=bool)
    tc = np.asarray(predicted_regions["TC"], dtype=bool)
    wt = np.asarray(predicted_regions["WT"], dtype=bool)
    if et.shape != tc.shape or tc.shape != wt.shape:
        raise ValueError("Predicted region shapes differ")
    et_outside = int(np.count_nonzero(et & ~tc))
    tc_outside = int(np.count_nonzero(tc & ~wt))
    if et_outside or tc_outside:
        _add(
            failures,
            case_id=case_id,
            region="ET/TC/WT",
            failure_type="nested-region inconsistency",
            observation=f"ET outside TC={et_outside} voxels and TC outside WT={tc_outside} voxels",
            evidence={"et_outside_tc_voxels": et_outside, "tc_outside_wt_voxels": tc_outside},
        )


def classify_case_failures(
    gt_labels: ArrayLike,
    pred_labels: ArrayLike,
    spacing_mm: Sequence[float],
    *,
    case_id: str,
    metrics: CaseMetrics | None = None,
    predicted_regions: Mapping[str, ArrayLike] | None = None,
    config: FailureAnalysisConfig | None = None,
) -> list[FailureEvidence]:
    """Classify observable patterns; does not assert a medical cause."""

    settings = config or FailureAnalysisConfig()
    gt_array = validate_brats_labels(gt_labels, name="ground truth")
    pred_array = validate_brats_labels(pred_labels, name="prediction")
    if gt_array.shape != pred_array.shape:
        raise ValueError(f"Label shapes differ: GT={gt_array.shape}, prediction={pred_array.shape}")
    spacing = tuple(float(value) for value in spacing_mm)
    if len(spacing) != gt_array.ndim or not all(
        np.isfinite(value) and value > 0 for value in spacing
    ):
        raise ValueError(f"Invalid spacing {spacing} for shape {gt_array.shape}")
    case_metrics = metrics or evaluate_case(gt_array, pred_array, spacing, case_id=case_id)
    gt_regions = regions_from_labels(gt_array)
    pred_regions = regions_from_labels(pred_array)
    voxel_volume = float(prod(spacing))
    failures: list[FailureEvidence] = []

    for region in REGION_ORDER:
        _one_sided_and_volume_errors(
            failures,
            case_id,
            region,
            gt_regions[region],
            pred_regions[region],
            voxel_volume,
            settings,
        )
        region_metrics = case_metrics.regions[region]
        if (
            region_metrics.gt_present
            and region_metrics.pred_present
            and np.isfinite(region_metrics.hd95_mm)
            and region_metrics.dice < settings.boundary_dice_ceiling
            and region_metrics.hd95_mm >= settings.boundary_hd95_floor_mm
        ):
            volume_ratio = region_metrics.pred_voxels / region_metrics.gt_voxels
            if 0.67 <= volume_ratio <= 1.5:
                _add(
                    failures,
                    case_id=case_id,
                    region=region,
                    failure_type="boundary error",
                    observation=(
                        f"Dice={region_metrics.dice:.3f}, HD95={region_metrics.hd95_mm:.2f} mm, "
                        f"and prediction/GT volume ratio={volume_ratio:.2f}"
                    ),
                    evidence={
                        "dice": region_metrics.dice,
                        "hd95_mm": region_metrics.hd95_mm,
                        "pred_to_gt_volume_ratio": volume_ratio,
                    },
                )

    _component_failures(failures, case_id, gt_regions["WT"], pred_regions["WT"], spacing, settings)
    _nested_inconsistency(failures, case_id, predicted_regions)
    return failures


def _find_case_nifti(directory: Path, case_id: str) -> Path:
    candidates = (directory / f"{case_id}.nii.gz", directory / f"{case_id}.nii")
    existing = [path for path in candidates if path.is_file()]
    if len(existing) != 1:
        raise FileNotFoundError(
            f"Expected exactly one NIfTI for {case_id} in {directory}; found {existing}"
        )
    return existing[0]


def _load_label_pair(
    gt_path: Path, pred_path: Path, case_id: str
) -> tuple[np.ndarray, np.ndarray, tuple[float, ...]]:
    try:
        import nibabel as nib
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("nibabel is required for failure analysis") from exc
    gt_image: Any = nib.load(str(gt_path))
    pred_image: Any = nib.load(str(pred_path))
    gt = np.asanyarray(gt_image.dataobj)
    pred = np.asanyarray(pred_image.dataobj)
    if gt.shape != pred.shape:
        raise ValueError(f"Shape mismatch for {case_id}: GT={gt.shape}, prediction={pred.shape}")
    if not np.allclose(gt_image.affine, pred_image.affine, rtol=0.0, atol=1e-4):
        raise ValueError(f"Affine mismatch for {case_id}")
    spacing = tuple(float(value) for value in gt_image.header.get_zooms()[: gt.ndim])
    return gt, pred, spacing


def analyze_failure_directories(
    *,
    ground_truth_dir: str | Path,
    prediction_dir: str | Path,
    metrics_per_case_csv: str | Path,
    output_dir: str | Path,
    top_n: int = 5,
    max_cases: int = 15,
    config: FailureAnalysisConfig | None = None,
) -> tuple[Path, Path]:
    """Rank, deduplicate, classify and write failure-analysis artifacts."""

    metric_rows = load_metrics_csv(metrics_per_case_csv)
    metrics_by_case = {str(row["case_id"]): row for row in metric_rows}
    if len(metrics_by_case) != len(metric_rows):
        raise ValueError("metrics_per_case.csv contains duplicate case IDs")
    rankings = rank_worst_cases(metric_rows, n_per_metric=top_n)
    representatives = select_representative_cases(rankings, max_cases=max_cases)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    rankings_path = destination / "failure_rankings.csv"
    with rankings_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["case_id", "region", "criterion", "value", "status", "failure_type", "rank"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for ranked_list in rankings.values():
            writer.writerows(item.as_dict() for item in ranked_list)

    failure_rows: list[dict[str, Any]] = []
    gt_directory = Path(ground_truth_dir)
    pred_directory = Path(prediction_dir)
    for representative in representatives:
        case_id = str(representative["case_id"])
        gt, pred, spacing = _load_label_pair(
            _find_case_nifti(gt_directory, case_id),
            _find_case_nifti(pred_directory, case_id),
            case_id,
        )
        case_metrics = evaluate_case(gt, pred, spacing, case_id=case_id)
        failures = classify_case_failures(
            gt,
            pred,
            spacing,
            case_id=case_id,
            metrics=case_metrics,
            config=config,
        )
        if not failures:
            failures = [
                FailureEvidence(
                    case_id=case_id,
                    region=str(representative["primary_region"]),
                    failure_type="ranked metric failure; pattern not classified",
                    observation=(
                        "Observed pattern: selected by "
                        f"{representative['selection_reasons']}; no configured "
                        "mask-pattern threshold fired"
                    ),
                    possible_explanation=(
                        "Possible explanation: inspect the source MRI and overlays; "
                        "no cause is inferred."
                    ),
                    evidence={},
                )
            ]
        metric_row = metrics_by_case[case_id]
        for failure in failures:
            failure_rows.append(
                {
                    "case_id": case_id,
                    "selection_reasons": representative["selection_reasons"],
                    "region": failure.region,
                    "failure_type": failure.failure_type,
                    "observation": failure.observation,
                    "possible_explanation": failure.possible_explanation,
                    "evidence_json": json.dumps(failure.evidence, sort_keys=True),
                    "dice_et": metric_row.get("dice_et", ""),
                    "dice_tc": metric_row.get("dice_tc", ""),
                    "dice_wt": metric_row.get("dice_wt", ""),
                    "hd95_et_mm": metric_row.get("hd95_et_mm", ""),
                    "hd95_tc_mm": metric_row.get("hd95_tc_mm", ""),
                    "hd95_wt_mm": metric_row.get("hd95_wt_mm", ""),
                }
            )
    failure_path = destination / "failure_cases.csv"
    failure_fields = [
        "case_id",
        "selection_reasons",
        "region",
        "failure_type",
        "observation",
        "possible_explanation",
        "evidence_json",
        "dice_et",
        "dice_tc",
        "dice_wt",
        "hd95_et_mm",
        "hd95_tc_mm",
        "hd95_wt_mm",
    ]
    with failure_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=failure_fields)
        writer.writeheader()
        writer.writerows(failure_rows)
    return rankings_path, failure_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rank and classify preliminary BraTS failures.")
    parser.add_argument("--ground-truth-dir", type=Path, required=True)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--metrics-per-case-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--max-cases", type=int, default=15)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    rankings, failures = analyze_failure_directories(
        ground_truth_dir=args.ground_truth_dir,
        prediction_dir=args.prediction_dir,
        metrics_per_case_csv=args.metrics_per_case_csv,
        output_dir=args.output_dir,
        top_n=args.top_n,
        max_cases=args.max_cases,
    )
    print(
        json.dumps(
            {"failure_rankings": str(rankings.resolve()), "failure_cases": str(failures.resolve())},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
