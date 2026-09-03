from __future__ import annotations

import csv
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from glioma_seg.analysis.failure_statistics import (
    DEFAULT_EXPECTED_CASE_COUNT,
    build_arg_parser,
    compute_failure_statistics,
)

REGIONS = {
    "et": lambda labels: labels == 3,
    "tc": lambda labels: (labels == 1) | (labels == 3),
    "wt": lambda labels: labels > 0,
}


def _write_nifti(path: Path, labels: np.ndarray) -> None:
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    image = nib.Nifti1Image(labels.astype(np.uint8), affine)  # type: ignore[no-untyped-call]
    nib.save(image, path)  # type: ignore[no-untyped-call]


def _metric_row(
    case_id: str,
    gt_labels: np.ndarray,
    pred_labels: np.ndarray,
    *,
    hd95_mm: float,
) -> dict[str, str]:
    row = {"case_id": case_id}
    for region, select in REGIONS.items():
        gt = select(gt_labels)
        pred = select(pred_labels)
        gt_voxels = int(np.count_nonzero(gt))
        pred_voxels = int(np.count_nonzero(pred))
        intersection = int(np.count_nonzero(gt & pred))
        denominator = gt_voxels + pred_voxels
        dice = 2.0 * intersection / denominator if denominator else float("nan")
        row.update(
            {
                f"gt_{region}_voxels": str(gt_voxels),
                f"pred_{region}_voxels": str(pred_voxels),
                f"{region}_gt_present": str(gt_voxels > 0),
                f"{region}_pred_present": str(pred_voxels > 0),
                f"dice_{region}": repr(dice),
                f"hd95_{region}_mm": repr(hd95_mm),
            }
        )
    return row


def _make_inputs(tmp_path: Path) -> dict[str, Path]:
    ground_truth = tmp_path / "ground_truth"
    predictions = tmp_path / "predictions"
    ground_truth.mkdir()
    predictions.mkdir()

    perfect_gt = np.zeros((10, 10, 10), dtype=np.uint8)
    perfect_gt[1, 1, 1:3] = 3
    perfect_pred = perfect_gt.copy()

    error_gt = perfect_gt.copy()
    error_pred = error_gt.copy()
    error_pred[7, 7, 7:9] = 3
    error_pred[7, 2, 7:9] = 3

    cases = {
        "case-perfect": (perfect_gt, perfect_pred, 0.0),
        "case-remote-fragmented": (error_gt, error_pred, 12.0),
    }
    metric_rows: list[dict[str, str]] = []
    for case_id, (gt_labels, pred_labels, hd95) in cases.items():
        _write_nifti(ground_truth / f"{case_id}.nii.gz", gt_labels)
        _write_nifti(predictions / f"{case_id}.nii.gz", pred_labels)
        metric_rows.append(
            _metric_row(case_id, gt_labels, pred_labels, hd95_mm=hd95)
        )

    metrics = tmp_path / "metrics_per_case.csv"
    with metrics.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metric_rows[0]))
        writer.writeheader()
        writer.writerows(metric_rows)

    integrity = tmp_path / "crossval_integrity.json"
    integrity.write_text(
        json.dumps(
            {
                "valid": True,
                "scope": "real_data_smoke_test_not_full_cross_validation",
                "total_cases": len(cases),
                "case_ids": list(cases),
                "each_case_validated_once": True,
            }
        ),
        encoding="utf-8",
    )
    return {
        "ground_truth": ground_truth,
        "predictions": predictions,
        "metrics": metrics,
        "integrity": integrity,
        "output_json": tmp_path / "failure_statistics.json",
        "output_csv": tmp_path / "failure_statistics_per_case_region.csv",
    }


def test_failure_statistics_preserves_definitions_schema_and_unique_case_ranking(
    tmp_path: Path,
) -> None:
    paths = _make_inputs(tmp_path)

    result = compute_failure_statistics(
        ground_truth_dir=paths["ground_truth"],
        prediction_dir=paths["predictions"],
        metrics_csv=paths["metrics"],
        integrity_json=paths["integrity"],
        output_json=paths["output_json"],
        output_csv=paths["output_csv"],
        expected_case_count=2,
        workers=1,
    )

    assert result["case_count"] == 2
    assert result["case_region_count"] == 6
    assert result["analysis_scope"].startswith("real-data smoke subset; NOT full")
    assert result["thresholds"] == {
        "material_error_fraction": 0.25,
        "large_hd95_mm_strictly_greater_than": 10.0,
        "minimum_isolated_component_volume_mm3": 10.0,
        "remote_component_distance_mm": 10.0,
        "fragmentation_excess_components": 2,
    }
    for region in ("ET", "TC", "WT"):
        assert result["table1"][region] == {
            "total_cases": 2,
            "gt_positive": 2,
            "gt_negative": 0,
            "prediction_positive": 2,
            "prediction_negative": 0,
        }
        stats = result["table2"][region]
        assert stats["over_segmentation"]["count"] == 1
        assert stats["large_hd95"]["count"] == 1
        assert stats["prediction_fragmentation"]["count"] == 1
        assert stats["isolated_false_positive_component"]["count"] == 1
        assert stats["remote_false_positive_component"]["count"] == 1
        assert stats["under_segmentation"]["count"] == 0
        assert stats["mixed_error"]["count"] == 0
        assert stats["any_major_error"]["count"] == 1
        assert stats["over_segmentation"]["conditional_denominator"] == 2

    assert [row["error_type"] for row in result["table4"]] == [
        "over_segmentation",
        "large_hd95",
        "prediction_fragmentation",
    ]
    assert all(row["unique_cases"] == 1 for row in result["table4"])
    assert all(row["unique_case_denominator"] == 2 for row in result["table4"])

    assert json.loads(paths["output_json"].read_text(encoding="utf-8")) == result
    with paths["output_csv"].open("r", encoding="utf-8", newline="") as handle:
        output_rows = list(csv.DictReader(handle))
    assert len(output_rows) == 6
    assert [(row["case_id"], row["region"]) for row in output_rows] == [
        ("case-perfect", "ET"),
        ("case-perfect", "TC"),
        ("case-perfect", "WT"),
        ("case-remote-fragmented", "ET"),
        ("case-remote-fragmented", "TC"),
        ("case-remote-fragmented", "WT"),
    ]
    assert not list(tmp_path.glob(".*.tmp"))


def test_failure_statistics_rejects_wrong_expected_count_before_publication(
    tmp_path: Path,
) -> None:
    paths = _make_inputs(tmp_path)

    with pytest.raises(ValueError, match="Expected 3 GT cases, found 2"):
        compute_failure_statistics(
            ground_truth_dir=paths["ground_truth"],
            prediction_dir=paths["predictions"],
            metrics_csv=paths["metrics"],
            integrity_json=paths["integrity"],
            output_json=paths["output_json"],
            output_csv=paths["output_csv"],
            expected_case_count=3,
            workers=1,
        )

    assert not paths["output_json"].exists()
    assert not paths["output_csv"].exists()
    assert not list(tmp_path.glob(".*.tmp"))


def test_failure_statistics_cli_expected_count_defaults_to_full_brats_cohort() -> None:
    args = build_arg_parser().parse_args(
        [
            "--ground-truth-dir",
            "gt",
            "--prediction-dir",
            "pred",
            "--metrics-csv",
            "metrics.csv",
            "--integrity-json",
            "integrity.json",
            "--output-json",
            "statistics.json",
            "--output-csv",
            "statistics.csv",
        ]
    )

    assert args.expected_case_count == DEFAULT_EXPECTED_CASE_COUNT == 1251
