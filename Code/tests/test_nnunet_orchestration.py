import json
from pathlib import Path

import pytest

from glioma_seg.backends.nnunet.commands import (
    build_accumulate_cross_validation,
    build_benchmark,
    build_train,
)
from glioma_seg.backends.nnunet.parser import summarize_benchmark
from glioma_seg.monitoring.process_monitor import NNUNetProcessMonitor


def test_official_v281_benchmark_and_resume_commands() -> None:
    assert build_benchmark(501).argv == (
        "nnUNetv2_train",
        "501",
        "3d_fullres",
        "0",
        "-tr",
        "nnUNetTrainerBenchmark_5epochs",
    )
    assert build_train(
        501,
        "3d_fullres",
        0,
        trainer="nnUNetTrainer_20epochs",
        continue_training=True,
    ).argv == (
        "nnUNetv2_train",
        "501",
        "3d_fullres",
        "0",
        "-tr",
        "nnUNetTrainer_20epochs",
        "--npz",
        "--c",
    )


def test_official_100epoch_cv_commands_keep_the_same_trainer(tmp_path: Path) -> None:
    assert build_train(
        501,
        "3d_fullres",
        4,
        trainer="nnUNetTrainer_100epochs",
        save_probabilities=True,
    ).argv == (
        "nnUNetv2_train",
        "501",
        "3d_fullres",
        "4",
        "-tr",
        "nnUNetTrainer_100epochs",
        "--npz",
    )
    assert build_accumulate_cross_validation(
        501,
        "3d_fullres",
        tmp_path / "crossval",
        trainer="nnUNetTrainer_100epochs",
    ).argv == (
        "nnUNetv2_accumulate_crossval_results",
        "501",
        "-c",
        "3d_fullres",
        "-o",
        str(tmp_path / "crossval"),
        "-f",
        "0",
        "1",
        "2",
        "3",
        "4",
        "-tr",
        "nnUNetTrainer_100epochs",
    )


class _ProgressGPU:
    class _Snapshot:
        gpu_utilization_percent = 91.0
        memory_used_mb = 8000.0
        memory_total_mb = 11264.0
        temperature_c = 82.0
        power_w = 205.0

    latest = _Snapshot()


def test_progress_monitor_shows_target_eta_power_and_final_validation() -> None:
    monitor = NNUNetProcessMonitor(
        "cv100",
        2,
        "nnUNetTrainer_100epochs",
        _ProgressGPU(),  # type: ignore[arg-type]
        target_epochs=100,
        expected_validation_cases=251,
    )
    monitor.consume_line("stdout", "Epoch 9")
    monitor.consume_line("stdout", "Epoch time: 120.0 s")
    status = monitor.status_line()
    assert "Epoch=10/100" in status
    assert "Progress=10.0%" in status
    assert "TrainETA=3.00h" in status
    assert "Power=205W" in status

    monitor.consume_line("stdout", "predicting BraTS-GLI-00000-000")
    validation_status = monitor.status_line()
    assert "Phase=FINAL_VALIDATION" in validation_status
    assert "Cases=1/251" in validation_status


def test_benchmark_parser_selects_current_environment_and_observed_mean(
    tmp_path: Path,
) -> None:
    benchmark_path = tmp_path / "benchmark_result.json"
    benchmark_path.write_text(
        json.dumps(
            {
                "old-host": {
                    "torch_version": "2.7.0",
                    "cudnn_version": 90000,
                    "gpu_name": "Other GPU",
                    "num_gpus": 1,
                    "fastest_epoch": 1.0,
                },
                "current-host": {
                    "torch_version": "2.8.0+cu126",
                    "cudnn_version": 91002,
                    "gpu_name": "NVIDIA GeForce RTX 2080 Ti",
                    "num_gpus": 1,
                    "fastest_epoch": 190.0,
                },
            }
        ),
        encoding="utf-8",
    )

    summary = summarize_benchmark(
        benchmark_path,
        measured_wall_seconds=2_000.0,
        observed_mean_epoch_seconds=200.0,
        expected_record={
            "torch_version": "2.8.0+cu126",
            "cudnn_version": 91002,
            "gpu_name": "NVIDIA GeForce RTX 2080 Ti",
            "num_gpus": 1,
        },
    )

    assert summary["official_record_key"] == "current-host"
    assert summary["runtime_estimate_basis_seconds_per_epoch"] == 200.0
    assert summary["wall_seconds_divided_by_five"] == 400.0
    assert summary["linear_runtime_estimates_seconds"]["50"] == 10_000.0
    assert summary["recommended_preliminary_trainer"] == "nnUNetTrainer_50epochs"


def test_benchmark_parser_rejects_ambiguous_environment(tmp_path: Path) -> None:
    benchmark_path = tmp_path / "benchmark_result.json"
    record = {
        "torch_version": "2.8.0+cu126",
        "cudnn_version": 91002,
        "gpu_name": "NVIDIA GeForce RTX 2080 Ti",
        "num_gpus": 1,
        "fastest_epoch": 190.0,
    }
    benchmark_path.write_text(
        json.dumps({"duplicate-a": record, "duplicate-b": record}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="exactly one benchmark record"):
        summarize_benchmark(
            benchmark_path,
            expected_record={
                "torch_version": "2.8.0+cu126",
                "cudnn_version": 91002,
                "gpu_name": "NVIDIA GeForce RTX 2080 Ti",
                "num_gpus": 1,
            },
        )
