import json
from pathlib import Path

import pytest

from glioma_seg.backends.nnunet.commands import build_benchmark, build_train
from glioma_seg.backends.nnunet.parser import summarize_benchmark


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
