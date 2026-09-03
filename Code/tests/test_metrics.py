from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from glioma_seg.evaluation.brats2023_official import evaluate_official_brats2023
from glioma_seg.evaluation.regions import (
    REGION_ORDER,
    assert_nested_regions,
    regions_from_labels,
    regions_to_brats,
    validate_brats_labels,
)
from glioma_seg.evaluation.semantic_metrics import (
    compute_region_metrics,
    dice_score,
    evaluate_case,
    hd95_mm,
    per_case_fieldnames,
    summarize_cases,
)


def test_brats_2023_region_definitions_and_nesting() -> None:
    labels = np.array([0, 1, 2, 3], dtype=np.uint8)

    regions = regions_from_labels(labels)

    assert REGION_ORDER == ("ET", "TC", "WT")
    np.testing.assert_array_equal(regions["ET"], [False, False, False, True])
    np.testing.assert_array_equal(regions["TC"], [False, True, False, True])
    np.testing.assert_array_equal(regions["WT"], [False, True, True, True])
    assert_nested_regions(regions)


def test_region_outputs_reconstruct_raw_brats_labels() -> None:
    # WT only -> ED=2; WT+TC -> NCR=1; WT+TC+ET -> ET=3.
    wt = np.array([False, True, True, True])
    tc = np.array([False, False, True, True])
    et = np.array([False, False, False, True])

    reconstructed = regions_to_brats(wt, tc, et)

    np.testing.assert_array_equal(reconstructed, [0, 2, 1, 3])


def test_region_reconstruction_can_enforce_nesting() -> None:
    wt = np.zeros((2, 2), dtype=bool)
    tc = np.zeros((2, 2), dtype=bool)
    et = np.zeros((2, 2), dtype=bool)
    et[0, 0] = True

    reconstructed = regions_to_brats(wt, tc, et, enforce_nested=True)

    assert reconstructed[0, 0] == 3
    with pytest.raises(ValueError, match="Nested-region invariant"):
        regions_to_brats(wt, tc, et, enforce_nested=False)


def test_legacy_label_four_is_rejected() -> None:
    with pytest.raises(ValueError, match=r"\{0,1,2,3\}"):
        validate_brats_labels(np.array([0, 4], dtype=np.uint8))


def test_perfect_prediction_has_dice_one_and_hd95_zero() -> None:
    mask = np.zeros((8, 8, 8), dtype=bool)
    mask[2:6, 2:6, 2:6] = True

    assert dice_score(mask, mask) == pytest.approx(1.0)
    assert hd95_mm(mask, mask, (1.0, 1.0, 1.0)) == pytest.approx(0.0)


def test_shifted_prediction_has_lower_dice() -> None:
    gt = np.zeros((8, 8, 8), dtype=bool)
    pred = np.zeros_like(gt)
    gt[1:4, 1:4, 1:4] = True
    pred[2:5, 1:4, 1:4] = True

    assert 0.0 < dice_score(gt, pred) < 1.0


def test_hd95_uses_physical_spacing() -> None:
    gt = np.zeros((5, 5, 5), dtype=bool)
    pred = np.zeros_like(gt)
    gt[1, 2, 2] = True
    pred[2, 2, 2] = True

    assert hd95_mm(gt, pred, (2.5, 1.0, 1.0)) == pytest.approx(2.5)


def test_both_empty_metrics_are_explicitly_undefined() -> None:
    empty = np.zeros((4, 4, 4), dtype=bool)

    metrics = compute_region_metrics(empty, empty, (1.0, 1.0, 1.0))

    assert np.isnan(metrics.dice)
    assert np.isnan(metrics.hd95_mm)
    assert metrics.empty_state == "both_empty"
    assert metrics.dice_status == "undefined_both_empty"
    assert metrics.hd95_status == "undefined_both_empty"
    assert metrics.failure_type == "both_empty"


def test_one_sided_empty_is_zero_dice_but_undefined_hd95() -> None:
    gt = np.zeros((4, 4, 4), dtype=bool)
    gt[1, 1, 1] = True
    pred = np.zeros_like(gt)

    metrics = compute_region_metrics(gt, pred, (1.0, 1.0, 1.0))

    assert metrics.dice == 0.0
    assert np.isnan(metrics.hd95_mm)
    assert metrics.empty_state == "gt_present_pred_empty"
    assert metrics.dice_status == "defined_zero"
    assert metrics.hd95_status == "undefined_pred_empty"
    assert metrics.failure_type == "false_negative"


def test_volume_uses_voxel_spacing() -> None:
    mask = np.zeros((3, 3, 3), dtype=bool)
    mask[1, 1, 1] = True

    metrics = compute_region_metrics(mask, mask, (2.0, 3.0, 4.0))

    assert metrics.gt_volume_mm3 == pytest.approx(24.0)
    assert metrics.pred_volume_mm3 == pytest.approx(24.0)


def test_summary_reports_finite_denominators() -> None:
    empty = np.zeros((3, 3, 3), dtype=np.uint8)
    et = empty.copy()
    et[1, 1, 1] = 3
    both_empty = evaluate_case(empty, empty, (1, 1, 1), case_id="empty")
    perfect = evaluate_case(et, et, (1, 1, 1), case_id="present")

    summary = summarize_cases([both_empty, perfect])

    dice = summary[0]
    hd95 = summary[1]
    assert dice["ET"] == pytest.approx(1.0)
    assert dice["ET_n_valid"] == 1
    assert dice["ET_n_excluded"] == 1
    assert hd95["ET"] == pytest.approx(0.0)
    assert hd95["ET_n_valid"] == 1
    assert hd95["ET_n_excluded"] == 1


def test_per_case_schema_uses_et_tc_wt_order() -> None:
    fields = per_case_fieldnames()

    assert fields[1:4] == ["dice_et", "dice_tc", "dice_wt"]
    assert fields[4:7] == ["hd95_et_mm", "hd95_tc_mm", "hd95_wt_mm"]
    assert fields[7:10] == ["gt_et_voxels", "gt_tc_voxels", "gt_wt_voxels"]


def test_official_adapter_never_falls_back_to_semantic_metrics(tmp_path: Path) -> None:
    status = evaluate_official_brats2023(
        tmp_path / "unused_gt",
        tmp_path / "unused_prediction",
        tmp_path / "report",
        adapter=None,
    )

    assert status.available is False
    assert "no pinned official" in status.reason
    assert not (tmp_path / "report" / "official_lesionwise_metrics_summary.csv").exists()
    recorded = json.loads(
        (tmp_path / "report" / "official_brats_metrics_status.json").read_text(encoding="utf-8")
    )
    assert recorded["available"] is False


def test_evaluation_analysis_visualization_and_reporting_integration(tmp_path: Path) -> None:
    import nibabel as nib

    from glioma_seg.analysis.failure_analysis import analyze_failure_directories
    from glioma_seg.evaluation.brats2023_official import mark_official_metrics_unavailable
    from glioma_seg.evaluation.evaluate import evaluate_directories
    from glioma_seg.reporting.report import ReportInputs, generate_reports
    from glioma_seg.visualization.overlays import generate_failure_figures

    case_id = "BraTS-GLI-TEST-001"
    shape = (8, 8, 8)
    gt = np.zeros(shape, dtype=np.uint8)
    gt[2:6, 2:6, 2:6] = 2
    gt[3:5, 3:5, 3:5] = 1
    gt[4, 4, 4] = 3
    pred = np.zeros_like(gt)
    pred[1:5, 2:6, 2:6] = 2
    pred[2:4, 3:5, 3:5] = 1
    pred[3, 4, 4] = 3
    affine = np.diag([2.0, 1.0, 1.0, 1.0])

    gt_dir = tmp_path / "labelsTr"
    pred_dir = tmp_path / "predictions"
    raw_dir = tmp_path / "raw" / case_id
    report_dir = tmp_path / "report"
    for directory in (gt_dir, pred_dir, raw_dir):
        directory.mkdir(parents=True)
    nib.save(
        nib.Nifti1Image(gt, affine),  # type: ignore[no-untyped-call]
        gt_dir / f"{case_id}.nii.gz",
    )
    nib.save(
        nib.Nifti1Image(pred, affine),  # type: ignore[no-untyped-call]
        pred_dir / f"{case_id}.nii.gz",
    )
    coordinates = np.indices(shape, dtype=np.float32)
    t1c = coordinates[0] + coordinates[1]
    flair = coordinates[1] + coordinates[2]
    nib.save(
        nib.Nifti1Image(t1c, affine),  # type: ignore[no-untyped-call]
        raw_dir / f"{case_id}-t1c.nii.gz",
    )
    nib.save(
        nib.Nifti1Image(flair, affine),  # type: ignore[no-untyped-call]
        raw_dir / f"{case_id}-t2f.nii.gz",
    )

    evaluated = evaluate_directories(
        gt_dir,
        pred_dir,
        report_dir,
        prediction_provenance="nnU-Net perform_actual_validation output (fold_0/validation)",
        prediction_tta_state="DEFAULT_MIRRORING",
    )
    _, failure_cases = analyze_failure_directories(
        ground_truth_dir=gt_dir,
        prediction_dir=pred_dir,
        metrics_per_case_csv=evaluated.per_case_csv,
        output_dir=report_dir,
        top_n=5,
        max_cases=15,
    )
    manifest = generate_failure_figures(
        raw_training_dir=tmp_path / "raw",
        ground_truth_dir=gt_dir,
        prediction_dir=pred_dir,
        failure_cases_csv=failure_cases,
        metrics_per_case_csv=evaluated.per_case_csv,
        output_dir=report_dir / "figures",
        n_slices=1,
    )
    official_status = mark_official_metrics_unavailable(
        report_dir, "Pinned official BraTS 2023 evaluator is not installed in this test environment"
    )
    assert official_status.available is False
    experiment_path = report_dir / "experiment.json"
    experiment_path.write_text(
        json.dumps(
            {
                "experiment_id": "integration-test",
                "dataset": "BraTS 2023 Adult Glioma Pre-Treatment",
                "dataset_id": 501,
                "dataset_case_count": 1,
                "framework": "nnU-Net v2",
                "configuration": "3d_fullres",
                "fold": 0,
                "trainer": "nnUNetTrainer_20epochs",
                "epochs": 20,
                "architecture": "recorded test architecture",
                "TTA_state": "OFF",
            }
        ),
        encoding="utf-8",
    )
    runtime_path = report_dir / "runtime.json"
    runtime_path.write_text(
        json.dumps({"total_seconds": 123.0, "average_seconds_per_epoch": 6.15}),
        encoding="utf-8",
    )
    gpu_path = report_dir / "gpu_summary.json"
    gpu_path.write_text(
        json.dumps(
            {
                "peak_memory_used_mb": 4321.0,
                "mean_gpu_utilization_percent": 87.5,
            }
        ),
        encoding="utf-8",
    )
    inference_runtime_path = report_dir / "inference_runtime.json"
    inference_runtime_path.write_text(
        json.dumps(
            {
                "mean_seconds_per_case": 2.5,
                "number_of_cases": 10,
                "timing_scope": "fresh_complete_run",
                "tta_state": "OFF",
            }
        ),
        encoding="utf-8",
    )
    summary, weekly = generate_reports(
        ReportInputs(
            output_dir=report_dir,
            experiment_json=experiment_path,
            metrics_summary_csv=evaluated.summary_csv,
            runtime_json=runtime_path,
            inference_runtime_json=inference_runtime_path,
            gpu_summary_json=gpu_path,
            official_status_json=report_dir / "official_brats_metrics_status.json",
            evaluation_protocol_json=evaluated.protocol_json,
            failure_cases_csv=failure_cases,
            figures_dir=report_dir / "figures",
            figures_manifest_csv=manifest,
        )
    )

    assert manifest.is_file()
    with manifest.open("r", encoding="utf-8", newline="") as handle:
        manifest_rows = list(csv.DictReader(handle))
    assert manifest_rows[0]["figure_path"] == f"{case_id}_t1c_flair_gt_pred.png"
    assert not Path(manifest_rows[0]["figure_path"]).is_absolute()
    assert (report_dir / "figures" / f"{case_id}_t1c_flair_gt_pred.png").stat().st_size > 0
    assert summary.is_file() and weekly.is_file()
    summary_text = summary.read_text(encoding="utf-8")
    assert "| Metric | ET | TC | WT |" in summary_text
    assert "NOT AVAILABLE" in summary_text
    assert "..." not in summary_text
    assert "DEFAULT_MIRRORING" in summary_text
    assert "Primary timing inference TTA state" in summary_text
    assert "OFF" in summary_text
    assert f"figures/{case_id}_t1c_flair_gt_pred.png" in summary_text
    assert "| Training time (s) | 123.0 |" in summary_text
    assert "| Peak dedicated VRAM (MB) | 4321.0 |" in summary_text
    assert "| Mean GPU utilization (%) | 87.5 |" in summary_text
    assert "| Inference time (s/case) | 2.5 |" in summary_text
    assert "| Inference cases timed | 10 |" in summary_text
    assert "| Inference timing scope | fresh_complete_run |" in summary_text
    protocol = json.loads(evaluated.protocol_json.read_text(encoding="utf-8"))
    assert protocol["region_order"] == ["ET", "TC", "WT"]
    assert protocol["regions"] == {"ET": [3], "TC": [1, 3], "WT": [1, 2, 3]}
    assert protocol["prediction_tta_state"] == "DEFAULT_MIRRORING"
    assert protocol["case_ids"] == [case_id]


def test_evaluation_requires_real_ground_truth_and_writes_no_metrics_on_missing_gt(
    tmp_path: Path,
) -> None:
    from glioma_seg.evaluation.evaluate import evaluate_directories

    prediction_dir = tmp_path / "official-validation-predictions"
    prediction_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="NIfTI directory does not exist"):
        evaluate_directories(
            tmp_path / "official-validation-has-no-labels",
            prediction_dir,
            tmp_path / "report",
        )
    assert not (tmp_path / "report" / "metrics_summary.csv").exists()


def test_primary_inference_contract_is_tta_off_and_exactly_matches_cases(
    tmp_path: Path,
) -> None:
    from glioma_seg.backends.nnunet.commands import build_predict
    from glioma_seg.evaluation.inference_audit import audit_inference_timing

    input_dir = tmp_path / "input"
    prediction_dir = tmp_path / "predictions"
    input_dir.mkdir()
    prediction_dir.mkdir()
    case_ids = ("BraTS-GLI-A", "BraTS-GLI-B")
    for case_id in case_ids:
        for channel in range(4):
            (input_dir / f"{case_id}_{channel:04d}.nii.gz").write_bytes(b"image")
        (prediction_dir / f"{case_id}.nii.gz").write_bytes(b"prediction")
    runtime_path = tmp_path / "inference_runtime.json"
    runtime_path.write_text(
        json.dumps(
            {
                "tta_state": "OFF",
                "total_seconds": 8.0,
                "number_of_cases": 2,
                "mean_seconds_per_case": 4.0,
                "output_dir": str(prediction_dir.resolve()),
            }
        ),
        encoding="utf-8",
    )

    command = build_predict(
        501,
        "3d_fullres",
        (0,),
        input_dir,
        prediction_dir,
        trainer="nnUNetTrainer_20epochs",
    )
    assert "--disable_tta" in command.arguments
    recorded = audit_inference_timing(
        input_dir=input_dir,
        prediction_dir=prediction_dir,
        runtime_json=runtime_path,
        finalize_fresh_run=True,
    )
    assert recorded["timing_scope"] == "fresh_complete_run"
    assert recorded["timing_comparable"] is True
    assert recorded["case_ids"] == list(case_ids)

    (prediction_dir / "BraTS-GLI-OFFICIAL-VALIDATION.nii.gz").write_bytes(b"extra")
    with pytest.raises(ValueError, match="output inventory mismatch"):
        audit_inference_timing(
            input_dir=input_dir,
            prediction_dir=prediction_dir,
            runtime_json=runtime_path,
        )


def test_official_result_flattening_and_summary_keep_et_tc_wt_order() -> None:
    from glioma_seg.evaluation.official_runner import (
        PER_CASE_FIELDS,
        flatten_official_case,
        summarize_official_cases,
    )

    columns = {
        "Num_TP": 1.0,
        "Num_FP": 0.0,
        "Num_FN": 0.0,
        "Sensitivity": 1.0,
        "Specificity": 1.0,
        "Legacy_Dice": 1.0,
        "Legacy_HD95": 0.0,
        "GT_Complete_Volume": 100.0,
        "LesionWise_Score_Dice": 1.0,
        "LesionWise_Score_HD95": 0.0,
    }
    flattened = flatten_official_case(
        {
            "case_id": "case-1",
            # Upstream returns WT, TC, ET; project output must still be ET, TC, WT.
            "regions": {"WT": dict(columns), "TC": dict(columns), "ET": dict(columns)},
        }
    )
    summary = summarize_official_cases([flattened])

    assert PER_CASE_FIELDS[1:4] == ("dice_et", "dice_tc", "dice_wt")
    assert PER_CASE_FIELDS[4:7] == ("hd95_et_mm", "hd95_tc_mm", "hd95_wt_mm")
    assert summary[0]["metric"] == "Dice"
    assert summary[1]["metric"] == "HD95"
    assert [summary[0][region] for region in REGION_ORDER] == [1.0, 1.0, 1.0]
    assert [summary[1][region] for region in REGION_ORDER] == [0.0, 0.0, 0.0]


def test_official_runner_failure_publishes_status_but_no_metric_tables(tmp_path: Path) -> None:
    from glioma_seg.evaluation.official_runner import (
        OFFICIAL_PER_CASE_FILENAME,
        OFFICIAL_SUMMARY_FILENAME,
        OfficialEvaluationError,
        run_official_evaluation,
    )

    output = tmp_path / "official-output"
    with pytest.raises(OfficialEvaluationError, match="official lesion-wise metric unavailable"):
        run_official_evaluation(
            ground_truth_dir=tmp_path / "missing-gt",
            prediction_dir=tmp_path / "missing-pred",
            output_dir=output,
            official_root=tmp_path / "missing-official-repository",
            python_executable=tmp_path / "missing-python",
        )

    status = json.loads((output / "official_brats_metrics_status.json").read_text(encoding="utf-8"))
    assert status["available"] is False
    assert not (output / OFFICIAL_PER_CASE_FILENAME).exists()
    assert not (output / OFFICIAL_SUMMARY_FILENAME).exists()
