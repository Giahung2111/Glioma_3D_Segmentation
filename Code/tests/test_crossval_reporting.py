from __future__ import annotations

import json
from pathlib import Path

import pytest

from glioma_seg.reporting.crossval import aggregate_crossval_telemetry
from glioma_seg.reporting.report import ReportInputs, generate_reports


def test_aggregate_crossval_telemetry_preserves_fold_evidence(tmp_path: Path) -> None:
    experiment = tmp_path / "experiment"
    experiment.mkdir()
    (experiment / "experiment.json").write_text(
        json.dumps({"experiment_id": "cv", "trainer": "nnUNetTrainer_100epochs"}),
        encoding="utf-8",
    )
    for fold in range(5):
        fold_dir = experiment / "folds" / f"fold_{fold}"
        fold_dir.mkdir(parents=True)
        (fold_dir / "runtime.json").write_text(
            json.dumps(
                {
                    "stage": "train",
                    "total_seconds": 3600.0 + fold,
                    "number_of_epochs": 100,
                    "average_seconds_per_epoch": 100.0 + fold,
                    "epoch_seconds_min": 90.0 + fold,
                    "epoch_seconds_median": 99.0 + fold,
                    "epoch_seconds_max": 120.0 + fold,
                }
            ),
            encoding="utf-8",
        )
        (fold_dir / "gpu_summary.json").write_text(
            json.dumps(
                {
                    "samples": 10 + fold,
                    "peak_memory_used_mb": 10_000.0 + fold,
                    "dedicated_memory_total_mb": 11_264.0,
                    "mean_gpu_utilization_percent": 80.0 + fold,
                    "peak_temperature_c": 82.0 + fold,
                    "mean_power_w": 180.0 + fold,
                    "errors": [],
                }
            ),
            encoding="utf-8",
        )

    runtime_path, gpu_path, table_path = aggregate_crossval_telemetry(experiment)

    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    gpu = json.loads(gpu_path.read_text(encoding="utf-8"))
    manifest = json.loads((experiment / "experiment.json").read_text(encoding="utf-8"))
    assert runtime["number_of_epochs"] == 500
    assert runtime["total_seconds"] == pytest.approx(sum(3600.0 + fold for fold in range(5)))
    assert len(runtime["fold_records"]) == 5
    assert gpu["peak_memory_used_mb"] == 10_004.0
    assert gpu["peak_temperature_c"] == 86.0
    assert table_path.is_file()
    assert manifest["epochs_per_fold"] == 100
    assert manifest["total_training_epochs"] == 500


def test_report_switches_to_verified_compute_limited_full_cv(tmp_path: Path) -> None:
    output = tmp_path / "report"
    output.mkdir()
    split_path = tmp_path / "splits_final.json"
    split_path.write_text("[]", encoding="utf-8")
    experiment_path = output / "experiment.json"
    experiment_path.write_text(
        json.dumps(
            {
                "experiment_id": "full-cv",
                "experiment_kind": "fullcv",
                "baseline_classification": "compute_limited_cross_validation",
                "dataset": "Dataset501_BraTS2023GLI",
                "dataset_id": 501,
                "trainer": "nnUNetTrainer_100epochs",
                "epochs": 100,
                "configuration": "3d_fullres",
            }
        ),
        encoding="utf-8",
    )
    metrics_path = output / "metrics_summary.csv"
    metrics_path.write_text(
        "metric,ET,TC,WT,total_cases\nDice,0.8,0.9,0.95,1251\nHD95,4,5,6,1251\n",
        encoding="utf-8",
    )
    crossval_path = output / "crossval_summary.json"
    per_fold = [
        {
            "fold": fold,
            "case_count": 251 if fold == 0 else 250,
            "metrics": {
                "Dice": {"ET": 0.8, "TC": 0.9, "WT": 0.95},
                "HD95": {"ET": 4.0, "TC": 5.0, "WT": 6.0},
            },
        }
        for fold in range(5)
    ]
    crossval_path.write_text(
        json.dumps(
            {
                "valid": True,
                "folds": [0, 1, 2, 3, 4],
                "total_cases": 1251,
                "each_case_validated_once": True,
                "split_source": str(split_path),
                "validation_case_counts": [251, 250, 250, 250, 250],
                "probabilities_retained": True,
                "probability_source_channel_order": ["WT", "TC", "ET"],
                "probability_canonical_order": ["ET", "TC", "WT"],
                "per_fold": per_fold,
            }
        ),
        encoding="utf-8",
    )
    preprocessing_path = output / "preprocessing_artifacts.json"
    preprocessing_path.write_text(
        json.dumps(
            {
                "valid": True,
                "details": {"splits_file": str(split_path), "splits_created": True},
            }
        ),
        encoding="utf-8",
    )

    summary, weekly = generate_reports(
        ReportInputs(
            output_dir=output,
            experiment_json=experiment_path,
            metrics_summary_csv=metrics_path,
            preprocessing_artifacts_json=preprocessing_path,
            crossval_summary_json=crossval_path,
        )
    )

    summary_text = summary.read_text(encoding="utf-8")
    weekly_text = weekly.read_text(encoding="utf-8")
    assert "5-Fold nnU-Net v2 Cross-Validation" in summary_text
    assert "all 1,251 cases appear in validation exactly once (verified)" in summary_text
    assert "| 4 | 250 |" in summary_text
    assert "not the standard 1,000-epoch" in summary_text
    assert "source order WT,TC,ET" in summary_text
    assert "5-Fold BraTS 2023 GLI Cross-Validation" in weekly_text


def test_declared_fullcv_report_refuses_missing_crossval_evidence(tmp_path: Path) -> None:
    output = tmp_path / "report"
    output.mkdir()
    experiment_path = output / "experiment.json"
    experiment_path.write_text(
        json.dumps(
            {
                "experiment_id": "full-cv-without-evidence",
                "experiment_kind": "fullcv",
                "dataset": "Dataset501_BraTS2023GLI",
                "dataset_id": 501,
                "trainer": "nnUNetTrainer_100epochs",
                "epochs": 100,
                "configuration": "3d_fullres",
            }
        ),
        encoding="utf-8",
    )
    metrics_path = output / "metrics_summary.csv"
    metrics_path.write_text(
        "metric,ET,TC,WT,total_cases\nDice,0.8,0.9,0.95,1251\nHD95,4,5,6,1251\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="requires a verified five-fold"):
        generate_reports(
            ReportInputs(
                output_dir=output,
                experiment_json=experiment_path,
                metrics_summary_csv=metrics_path,
            )
        )
