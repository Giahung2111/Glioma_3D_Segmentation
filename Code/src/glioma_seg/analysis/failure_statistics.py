"""Backend-neutral failure statistics for completed out-of-fold predictions.

The analysis consumes ground-truth masks, prediction masks, the project's
``metrics_per_case.csv`` artifact, and a validated cross-validation integrity
record. It does not import or invoke model training, inference, preprocessing,
or backend orchestration code.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import nibabel as nib
import numpy as np
from numpy.typing import NDArray
from scipy import ndimage  # type: ignore[import-untyped]

REGIONS = ("ET", "TC", "WT")
DEFAULT_EXPECTED_CASE_COUNT = 1251
MATERIAL_ERROR_FRACTION = 0.25
HD95_LARGE_MM = 10.0
MIN_COMPONENT_VOLUME_MM3 = 10.0
REMOTE_COMPONENT_DISTANCE_MM = 10.0
FRAGMENTATION_EXCESS_COMPONENTS = 2
STRUCTURE = ndimage.generate_binary_structure(3, 1)

FLAG_ORDER = (
    "complete_false_positive",
    "complete_false_negative",
    "under_segmentation",
    "over_segmentation",
    "mixed_error",
    "large_hd95",
    "prediction_fragmentation",
    "isolated_false_positive_component",
    "remote_false_positive_component",
)

FLAG_LABELS = {
    "complete_false_positive": "Complete false positive",
    "complete_false_negative": "Complete false negative",
    "under_segmentation": "Under-segmentation",
    "over_segmentation": "Over-segmentation",
    "mixed_error": "Mixed under/over-segmentation",
    "large_hd95": "Large HD95 (>10 mm)",
    "prediction_fragmentation": "Prediction fragmentation",
    "isolated_false_positive_component": "Isolated false-positive component",
    "remote_false_positive_component": "Remote false-positive component (>=10 mm)",
}


def _default_worker_count() -> int:
    return max(1, min(4, os.cpu_count() or 1))


@dataclass(frozen=True)
class CaseInput:
    """Paths and retained metric evidence required to audit one case."""

    case_id: str
    gt_path: str
    pred_path: str
    metrics_row: dict[str, str]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _case_id(path: Path) -> str:
    name = path.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    raise ValueError(f"Not a NIfTI path: {path}")


def _inventory(directory: Path) -> dict[str, Path]:
    paths = sorted([*directory.glob("*.nii.gz"), *directory.glob("*.nii")])
    inventory = {_case_id(path): path for path in paths}
    if len(inventory) != len(paths):
        raise ValueError(f"Duplicate NIfTI case IDs in {directory}")
    return inventory


def _load_metrics(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result = {str(row["case_id"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError("metrics_per_case.csv contains duplicate case IDs")
    return result


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"Invalid boolean value {value!r}")


def _regions(labels: NDArray[np.integer[Any]]) -> dict[str, NDArray[np.bool_]]:
    unique = set(int(value) for value in np.unique(labels))
    if not unique.issubset({0, 1, 2, 3}):
        raise ValueError(f"Unexpected BraTS label values: {sorted(unique)}")
    return {
        "ET": np.asarray(labels == 3, dtype=bool),
        "TC": np.asarray((labels == 1) | (labels == 3), dtype=bool),
        "WT": np.asarray(labels > 0, dtype=bool),
    }


def _component_evidence(
    gt: NDArray[np.bool_],
    pred: NDArray[np.bool_],
    spacing: tuple[float, float, float],
) -> dict[str, Any]:
    gt_labeled, gt_count = ndimage.label(gt, structure=STRUCTURE)
    del gt_labeled
    pred_labeled, pred_count = ndimage.label(pred, structure=STRUCTURE)
    fragmentation = int(pred_count) >= int(gt_count) + FRAGMENTATION_EXCESS_COMPONENTS

    isolated_ids: list[int] = []
    farthest_isolated_distance = math.nan
    if pred_count:
        component_sizes = np.bincount(pred_labeled.ravel())
        overlapping_ids = set(int(value) for value in np.unique(pred_labeled[gt & pred]))
        voxel_volume = float(np.prod(spacing))
        isolated_ids = [
            component_id
            for component_id in range(1, int(pred_count) + 1)
            if component_sizes[component_id] * voxel_volume >= MIN_COMPONENT_VOLUME_MM3
            and component_id not in overlapping_ids
        ]
        if isolated_ids and np.any(gt):
            distance_to_gt = ndimage.distance_transform_edt(~gt, sampling=spacing)
            minima = ndimage.minimum(
                distance_to_gt,
                labels=pred_labeled,
                index=np.asarray(isolated_ids, dtype=np.int32),
            )
            minima_array = np.atleast_1d(np.asarray(minima, dtype=float))
            farthest_isolated_distance = float(np.max(minima_array))

    remote = bool(
        isolated_ids
        and math.isfinite(farthest_isolated_distance)
        and farthest_isolated_distance >= REMOTE_COMPONENT_DISTANCE_MM
    )
    return {
        "gt_components": int(gt_count),
        "pred_components": int(pred_count),
        "prediction_fragmentation": bool(fragmentation),
        "isolated_false_positive_component": bool(isolated_ids),
        "isolated_component_count": len(isolated_ids),
        "remote_false_positive_component": remote,
        "farthest_isolated_component_min_distance_mm": farthest_isolated_distance,
    }


def _analyze_case(case: CaseInput) -> list[dict[str, Any]]:
    gt_image = cast(nib.Nifti1Image, nib.load(case.gt_path))
    pred_image = cast(nib.Nifti1Image, nib.load(case.pred_path))
    if gt_image.shape != pred_image.shape:
        raise ValueError(
            f"Shape mismatch for {case.case_id}: GT={gt_image.shape}, pred={pred_image.shape}"
        )
    if len(gt_image.shape) != 3:
        raise ValueError(f"Expected 3D masks for {case.case_id}, got {gt_image.shape}")
    if not np.allclose(gt_image.affine, pred_image.affine, rtol=0.0, atol=1e-4):
        raise ValueError(f"Affine mismatch for {case.case_id}")
    gt_zooms = gt_image.header.get_zooms()
    pred_zooms = pred_image.header.get_zooms()
    gt_spacing = (float(gt_zooms[0]), float(gt_zooms[1]), float(gt_zooms[2]))
    pred_spacing = (float(pred_zooms[0]), float(pred_zooms[1]), float(pred_zooms[2]))
    if not np.allclose(gt_spacing, pred_spacing, rtol=0.0, atol=1e-6):
        raise ValueError(
            f"Spacing mismatch for {case.case_id}: GT={gt_spacing}, pred={pred_spacing}"
        )
    if not all(math.isfinite(value) and value > 0 for value in gt_spacing):
        raise ValueError(f"Invalid spacing for {case.case_id}: {gt_spacing}")

    gt_labels = np.asanyarray(gt_image.dataobj)
    pred_labels = np.asanyarray(pred_image.dataobj)
    gt_regions = _regions(gt_labels)
    pred_regions = _regions(pred_labels)
    rows: list[dict[str, Any]] = []
    for region in REGIONS:
        key = region.lower()
        gt = gt_regions[region]
        pred = pred_regions[region]
        gt_voxels = int(np.count_nonzero(gt))
        pred_voxels = int(np.count_nonzero(pred))
        gt_present = gt_voxels > 0
        pred_present = pred_voxels > 0
        both_present = gt_present and pred_present
        intersection = int(np.count_nonzero(gt & pred))
        false_negative_voxels = gt_voxels - intersection
        false_positive_voxels = pred_voxels - intersection
        missed_fraction = false_negative_voxels / gt_voxels if gt_voxels else math.nan
        extra_fraction = false_positive_voxels / pred_voxels if pred_voxels else math.nan

        complete_fp = not gt_present and pred_present
        complete_fn = gt_present and not pred_present
        under = bool(both_present and missed_fraction >= MATERIAL_ERROR_FRACTION)
        over = bool(both_present and extra_fraction >= MATERIAL_ERROR_FRACTION)
        mixed = under and over
        hd95 = float(case.metrics_row[f"hd95_{key}_mm"])
        large_hd95 = bool(math.isfinite(hd95) and hd95 > HD95_LARGE_MM)
        components = _component_evidence(gt, pred, gt_spacing)

        expected_gt_voxels = int(case.metrics_row[f"gt_{key}_voxels"])
        expected_pred_voxels = int(case.metrics_row[f"pred_{key}_voxels"])
        if gt_voxels != expected_gt_voxels or pred_voxels != expected_pred_voxels:
            raise ValueError(
                f"Voxel-count mismatch for {case.case_id} {region}: "
                f"scan=({gt_voxels},{pred_voxels}), "
                f"metrics=({expected_gt_voxels},{expected_pred_voxels})"
            )
        if gt_present != _parse_bool(case.metrics_row[f"{key}_gt_present"]):
            raise ValueError(f"GT-presence mismatch for {case.case_id} {region}")
        if pred_present != _parse_bool(case.metrics_row[f"{key}_pred_present"]):
            raise ValueError(f"Prediction-presence mismatch for {case.case_id} {region}")
        expected_dice = float(case.metrics_row[f"dice_{key}"])
        denominator = gt_voxels + pred_voxels
        scanned_dice = 2.0 * intersection / denominator if denominator else math.nan
        if math.isfinite(expected_dice) != math.isfinite(scanned_dice) or (
            math.isfinite(expected_dice)
            and not math.isclose(expected_dice, scanned_dice, rel_tol=0.0, abs_tol=1e-12)
        ):
            raise ValueError(
                f"Dice mismatch for {case.case_id} {region}: "
                f"scan={scanned_dice}, metrics={expected_dice}"
            )

        flags = {
            "complete_false_positive": complete_fp,
            "complete_false_negative": complete_fn,
            "under_segmentation": under,
            "over_segmentation": over,
            "mixed_error": mixed,
            "large_hd95": large_hd95,
            "prediction_fragmentation": components["prediction_fragmentation"],
            "isolated_false_positive_component": components[
                "isolated_false_positive_component"
            ],
            "remote_false_positive_component": components["remote_false_positive_component"],
        }
        rows.append(
            {
                "case_id": case.case_id,
                "region": region,
                "gt_voxels": gt_voxels,
                "pred_voxels": pred_voxels,
                "intersection_voxels": intersection,
                "false_negative_voxels": false_negative_voxels,
                "false_positive_voxels": false_positive_voxels,
                "gt_present": gt_present,
                "pred_present": pred_present,
                "both_present": both_present,
                "missed_gt_fraction": missed_fraction,
                "extra_pred_fraction": extra_fraction,
                "dice": scanned_dice,
                "hd95_mm": hd95,
                **components,
                **flags,
                "any_major_error": any(flags.values()),
            }
        )
    return rows


def _percent(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return math.nan
    return 100.0 * numerator / denominator


def _conditional_denominator(flag: str, rows: list[dict[str, Any]]) -> int:
    if flag == "complete_false_positive":
        return sum(not bool(row["gt_present"]) for row in rows)
    if flag == "complete_false_negative":
        return sum(bool(row["gt_present"]) for row in rows)
    if flag in {"under_segmentation", "over_segmentation", "mixed_error", "large_hd95"}:
        return sum(bool(row["both_present"]) for row in rows)
    if flag == "isolated_false_positive_component":
        return sum(bool(row["pred_present"]) for row in rows)
    if flag == "remote_false_positive_component":
        return sum(bool(row["both_present"]) for row in rows)
    return len(rows)


def _json_clean(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_clean(item) for item in value]
    return value


def _analyze_inputs(inputs: list[CaseInput], workers: int) -> list[dict[str, Any]]:
    all_rows: list[dict[str, Any]] = []
    if workers == 1:
        for completed, item in enumerate(inputs, start=1):
            all_rows.extend(_analyze_case(item))
            if completed % 25 == 0 or completed == len(inputs):
                print(
                    f"[failure-statistics] {completed}/{len(inputs)} cases complete",
                    flush=True,
                )
        return all_rows

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_analyze_case, item): item.case_id for item in inputs}
        for completed, future in enumerate(as_completed(futures), start=1):
            case_id = futures[future]
            try:
                all_rows.extend(future.result())
            except Exception as exc:
                raise RuntimeError(f"Analysis failed for {case_id}") from exc
            if completed % 25 == 0 or completed == len(inputs):
                print(
                    f"[failure-statistics] {completed}/{len(inputs)} cases complete",
                    flush=True,
                )
    return all_rows


def _analysis_scope_from_integrity(
    integrity: dict[str, Any],
    *,
    expected_case_count: int,
    expected_case_ids: set[str],
) -> str:
    """Return a truthful, closed-set scope derived from validated evidence."""

    smoke_scope = "real_data_smoke_test_not_full_cross_validation"
    if integrity.get("scope") == smoke_scope:
        case_ids = {str(value) for value in integrity.get("case_ids", [])}
        if case_ids != expected_case_ids:
            raise ValueError("Smoke integrity case IDs do not match the analyzed inventory")
        if integrity.get("each_case_validated_once") is not True:
            raise ValueError("Smoke integrity does not confirm one prediction per case")
        return (
            "real-data smoke subset; NOT full cross-validation; failure-statistics "
            "computation only, with no training or inference in this analysis stage"
        )

    if integrity.get("evaluation_scope") == "five_fold_out_of_fold":
        if expected_case_count != DEFAULT_EXPECTED_CASE_COUNT:
            raise ValueError(
                "Five-fold OOF failure statistics require the complete 1,251-case cohort"
            )
        if integrity.get("each_case_validated_once") is not True:
            raise ValueError("Five-fold integrity does not confirm one prediction per case")
        if [int(value) for value in integrity.get("folds", [])] != [0, 1, 2, 3, 4]:
            raise ValueError("Five-fold integrity must contain folds 0,1,2,3,4")
        return "completed five-fold out-of-fold predictions; no training or inference"

    raise ValueError("Unsupported or missing evaluation scope in crossval_integrity.json")


def _stage_file(target: Path, writer: Callable[[Path], None]) -> Path:
    """Fully write and fsync a temporary sibling before publication."""

    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        writer(temporary)
        # Windows requires a writable descriptor for fsync; the staged file is
        # already closed by the writer before it is reopened here.
        with temporary.open("r+b") as handle:
            os.fsync(handle.fileno())
        return temporary
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _publish_outputs(
    *,
    output_json: Path,
    output_csv: Path,
    result: dict[str, Any],
    rows: list[dict[str, Any]],
) -> None:
    if output_json == output_csv:
        raise ValueError("output_json and output_csv must be different paths")

    def write_json(path: Path) -> None:
        path.write_text(
            json.dumps(_json_clean(result), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    def write_csv(path: Path) -> None:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    staged_json: Path | None = None
    staged_csv: Path | None = None
    try:
        staged_json = _stage_file(output_json, write_json)
        staged_csv = _stage_file(output_csv, write_csv)
        os.replace(staged_json, output_json)
        staged_json = None
        os.replace(staged_csv, output_csv)
        staged_csv = None
    finally:
        if staged_json is not None:
            staged_json.unlink(missing_ok=True)
        if staged_csv is not None:
            staged_csv.unlink(missing_ok=True)


def compute_failure_statistics(
    *,
    ground_truth_dir: str | Path,
    prediction_dir: str | Path,
    metrics_csv: str | Path,
    integrity_json: str | Path,
    output_json: str | Path,
    output_csv: str | Path,
    expected_case_count: int = DEFAULT_EXPECTED_CASE_COUNT,
    workers: int | None = None,
) -> dict[str, Any]:
    """Validate a complete cohort, compute statistics, and atomically publish artifacts."""

    if expected_case_count < 1:
        raise ValueError("expected_case_count must be positive")
    if workers is None:
        workers = _default_worker_count()
    if workers < 1:
        raise ValueError("workers must be positive")

    gt_directory = Path(ground_truth_dir).resolve()
    prediction_directory = Path(prediction_dir).resolve()
    metrics_path = Path(metrics_csv).resolve()
    integrity_path = Path(integrity_json).resolve()
    output_json_path = Path(output_json).resolve()
    output_csv_path = Path(output_csv).resolve()

    gt_inventory = _inventory(gt_directory)
    pred_inventory = _inventory(prediction_directory)
    metrics = _load_metrics(metrics_path)
    expected_ids = set(gt_inventory)
    if len(expected_ids) != expected_case_count:
        raise ValueError(
            f"Expected {expected_case_count} GT cases, found {len(expected_ids)}"
        )
    if set(pred_inventory) != expected_ids:
        raise ValueError(
            "Prediction inventory differs from GT: "
            f"missing={sorted(expected_ids - set(pred_inventory))[:10]}, "
            f"extra={sorted(set(pred_inventory) - expected_ids)[:10]}"
        )
    if set(metrics) != expected_ids:
        raise ValueError(
            "Metrics inventory differs from GT: "
            f"missing={sorted(expected_ids - set(metrics))[:10]}, "
            f"extra={sorted(set(metrics) - expected_ids)[:10]}"
        )
    integrity = json.loads(integrity_path.read_text(encoding="utf-8"))
    if not integrity.get("valid") or int(integrity.get("total_cases", -1)) != len(
        expected_ids
    ):
        raise ValueError("crossval_integrity.json is not valid for the expected cohort")
    analysis_scope = _analysis_scope_from_integrity(
        integrity,
        expected_case_count=expected_case_count,
        expected_case_ids=expected_ids,
    )

    inputs = [
        CaseInput(
            case_id=case_id,
            gt_path=str(gt_inventory[case_id]),
            pred_path=str(pred_inventory[case_id]),
            metrics_row=metrics[case_id],
        )
        for case_id in sorted(expected_ids)
    ]
    all_rows = _analyze_inputs(inputs, workers)
    region_index = {region: index for index, region in enumerate(REGIONS)}
    all_rows.sort(key=lambda row: (str(row["case_id"]), region_index[str(row["region"])]))
    if len(all_rows) != len(inputs) * len(REGIONS):
        raise ValueError(
            f"Expected {len(inputs) * len(REGIONS)} case-region rows, got {len(all_rows)}"
        )

    rows_by_region = {
        region: [row for row in all_rows if row["region"] == region] for region in REGIONS
    }
    table1: dict[str, Any] = {}
    table2: dict[str, Any] = {}
    for region in REGIONS:
        rows = rows_by_region[region]
        if len(rows) != expected_case_count:
            raise ValueError(f"Region {region} has {len(rows)} rows")
        gt_positive = sum(bool(row["gt_present"]) for row in rows)
        pred_positive = sum(bool(row["pred_present"]) for row in rows)
        table1[region] = {
            "total_cases": len(rows),
            "gt_positive": gt_positive,
            "gt_negative": len(rows) - gt_positive,
            "prediction_positive": pred_positive,
            "prediction_negative": len(rows) - pred_positive,
        }
        region_stats: dict[str, Any] = {}
        for flag in FLAG_ORDER:
            count = sum(bool(row[flag]) for row in rows)
            conditional_denominator = _conditional_denominator(flag, rows)
            region_stats[flag] = {
                "count": count,
                "total_denominator": len(rows),
                "percent_total": _percent(count, len(rows)),
                "conditional_denominator": conditional_denominator,
                "percent_conditional": _percent(count, conditional_denominator),
            }
        any_count = sum(bool(row["any_major_error"]) for row in rows)
        region_stats["any_major_error"] = {
            "count": any_count,
            "total_denominator": len(rows),
            "percent_total": _percent(any_count, len(rows)),
            "conditional_denominator": len(rows),
            "percent_conditional": _percent(any_count, len(rows)),
        }
        table2[region] = region_stats

    type_case_sets: dict[str, set[str]] = defaultdict(set)
    for row in all_rows:
        for flag in FLAG_ORDER:
            if row[flag]:
                type_case_sets[flag].add(str(row["case_id"]))
    ranked_flags = sorted(
        FLAG_ORDER,
        key=lambda flag: (-len(type_case_sets[flag]), FLAG_ORDER.index(flag)),
    )
    table4: list[dict[str, Any]] = []
    for rank, flag in enumerate(ranked_flags[:3], start=1):
        unique_count = len(type_case_sets[flag])
        table4.append(
            {
                "rank": rank,
                "error_type": flag,
                "display_name": FLAG_LABELS[flag],
                "ET_count": table2["ET"][flag]["count"],
                "TC_count": table2["TC"][flag]["count"],
                "WT_count": table2["WT"][flag]["count"],
                "region_occurrences": sum(
                    table2[region][flag]["count"] for region in REGIONS
                ),
                "unique_cases": unique_count,
                "unique_case_denominator": expected_case_count,
                "percent_unique_cases": _percent(unique_count, expected_case_count),
            }
        )

    result = {
        "analysis_scope": analysis_scope,
        "case_count": expected_case_count,
        "case_region_count": len(all_rows),
        "regions": list(REGIONS),
        "sources": {
            "ground_truth_dir": str(gt_directory),
            "prediction_dir": str(prediction_directory),
            "metrics_per_case_csv": str(metrics_path),
            "metrics_per_case_sha256": _sha256(metrics_path),
            "crossval_integrity_json": str(integrity_path),
            "crossval_integrity_sha256": _sha256(integrity_path),
        },
        "definitions": {
            "regions": {"ET": [3], "TC": [1, 3], "WT": [1, 2, 3]},
            "component_connectivity": "6-neighbor in 3D",
            "complete_false_positive": "GT empty and prediction nonempty",
            "complete_false_negative": "GT nonempty and prediction empty",
            "under_segmentation": (
                "both masks nonempty and false-negative voxels / GT voxels >= 0.25"
            ),
            "over_segmentation": (
                "both masks nonempty and false-positive voxels / prediction voxels >= 0.25"
            ),
            "mixed_error": "both under_segmentation and over_segmentation are true",
            "large_hd95": "finite semantic HD95 > 10.0 mm",
            "prediction_fragmentation": (
                "prediction component count >= GT component count + 2"
            ),
            "isolated_false_positive_component": (
                "at least one prediction component >= 10.0 mm3 with zero GT overlap"
            ),
            "remote_false_positive_component": (
                "an eligible isolated component whose minimum Euclidean distance to GT >= 10.0 mm"
            ),
            "any_major_error": (
                "union of all independent flags in Table 2 except the redundant union itself"
            ),
            "table4_ranking": (
                "descending unique patient IDs affected anywhere in ET/TC/WT; "
                "one patient counted once per error type"
            ),
        },
        "thresholds": {
            "material_error_fraction": MATERIAL_ERROR_FRACTION,
            "large_hd95_mm_strictly_greater_than": HD95_LARGE_MM,
            "minimum_isolated_component_volume_mm3": MIN_COMPONENT_VOLUME_MM3,
            "remote_component_distance_mm": REMOTE_COMPONENT_DISTANCE_MM,
            "fragmentation_excess_components": FRAGMENTATION_EXCESS_COMPONENTS,
        },
        "table1": table1,
        "table2": table2,
        "table4": table4,
        "all_error_type_unique_case_counts": {
            flag: {
                "unique_cases": len(type_case_sets[flag]),
                "percent_unique_cases": _percent(
                    len(type_case_sets[flag]), expected_case_count
                ),
            }
            for flag in ranked_flags
        },
        "validation": {
            "ground_truth_inventory_matches_predictions": True,
            "ground_truth_inventory_matches_metrics": True,
            "each_case_has_three_region_rows": True,
            "mask_voxel_counts_match_metrics_csv": True,
            "mask_presence_states_match_metrics_csv": True,
            "dice_recomputed_from_masks_matches_metrics_csv_at_1e-12": True,
            "all_shapes_affines_and_spacings_match": True,
        },
    }
    _publish_outputs(
        output_json=output_json_path,
        output_csv=output_csv_path,
        result=result,
        rows=all_rows,
    )
    return _json_clean(result)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the backend-neutral failure-statistics CLI parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth-dir", type=Path, required=True)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--metrics-csv", type=Path, required=True)
    parser.add_argument("--integrity-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument(
        "--expected-case-count",
        type=int,
        default=DEFAULT_EXPECTED_CASE_COUNT,
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=_default_worker_count(),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = compute_failure_statistics(
        ground_truth_dir=args.ground_truth_dir,
        prediction_dir=args.prediction_dir,
        metrics_csv=args.metrics_csv,
        integrity_json=args.integrity_json,
        output_json=args.output_json,
        output_csv=args.output_csv,
        expected_case_count=args.expected_case_count,
        workers=args.workers,
    )
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_csv}")
    print(
        json.dumps(
            {"table1": result["table1"], "table2": result["table2"], "table4": result["table4"]},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
