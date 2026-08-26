from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

import glioma_seg.backends.nnunet.backend as backend_module
from glioma_seg.backends.nnunet.backend import NNUNetV2Backend
from glioma_seg.backends.nnunet.commands import CommandSpec
from glioma_seg.utils.subprocess import LiveCommandResult

GPU_SUMMARY: dict[str, Any] = {
    "samples": 3,
    "peak_memory_used_mb": 4321.5,
    "dedicated_memory_total_mb": 11264.0,
    "mean_gpu_utilization_percent": 87.25,
    "peak_temperature_c": 82.0,
    "mean_power_w": 211.0,
    "backend": "test",
    "errors": [],
}


class FakeGPUSummary:
    def to_dict(self) -> dict[str, Any]:
        return dict(GPU_SUMMARY)


class FakeGPUMonitor:
    latest = None

    def __init__(self, output_path: Path, *, interval_seconds: float) -> None:
        self.output_path = output_path
        self.interval_seconds = interval_seconds

    def start(self) -> FakeGPUMonitor:
        return self

    def stop(self) -> FakeGPUSummary:
        return FakeGPUSummary()


def fake_run_live_command(
    argv: Sequence[str | Path],
    *,
    log_path: Path,
    **_: Any,
) -> LiveCommandResult:
    return LiveCommandResult(
        argv=tuple(str(part) for part in argv),
        returncode=0,
        started_at="2026-08-26T00:00:00+00:00",
        ended_at="2026-08-26T00:00:12+00:00",
        elapsed_seconds=12.0,
        log_path=log_path,
        stdout_tail=(),
        stderr_tail=(),
    )


def configured_backend(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> NNUNetV2Backend:
    backend = NNUNetV2Backend(project_root=tmp_path)
    monkeypatch.setattr(backend_module, "GPUMonitor", FakeGPUMonitor)
    monkeypatch.setattr(backend_module, "run_live_command", fake_run_live_command)
    monkeypatch.setattr(backend, "_official_command", lambda spec: spec)
    return backend


@pytest.mark.parametrize("stage", ["predict", "benchmark"])
def test_nontraining_telemetry_preserves_training_aliases_and_records_stage_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
) -> None:
    backend = configured_backend(tmp_path, monkeypatch)
    experiment_id = backend.initialize_experiment(
        "shared_experiment",
        kind="prelim",
        fold=0,
        trainer="nnUNetTrainer_50epochs",
    )
    backend._update_manifest(
        experiment_id,
        {
            "peak_vram_mb": 10001.0,
            "mean_gpu_utilization": 64.5,
        },
    )

    backend._execute(
        CommandSpec("fake_nnunet", (), stage),
        experiment_id=experiment_id,
        fold=0,
        trainer="nnUNetTrainer_50epochs",
        monitor_gpu=True,
    )

    experiment_dir = backend.paths.reports / experiment_id
    artifact_path = experiment_dir / f"{stage}_gpu_summary.json"
    manifest = json.loads((experiment_dir / "experiment.json").read_text(encoding="utf-8"))
    assert manifest["peak_vram_mb"] == 10001.0
    assert manifest["mean_gpu_utilization"] == 64.5
    assert manifest[f"{stage}_peak_vram_mb"] == GPU_SUMMARY["peak_memory_used_mb"]
    assert manifest[f"{stage}_mean_gpu_utilization"] == GPU_SUMMARY["mean_gpu_utilization_percent"]
    assert manifest[f"{stage}_gpu_telemetry_file"] == str(artifact_path)
    assert json.loads(artifact_path.read_text(encoding="utf-8")) == GPU_SUMMARY


def test_training_telemetry_updates_legacy_aliases_and_stage_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = configured_backend(tmp_path, monkeypatch)
    experiment_id = backend.initialize_experiment(
        "training_experiment",
        kind="prelim",
        fold=0,
        trainer="nnUNetTrainer_50epochs",
    )

    backend._execute(
        CommandSpec("fake_nnunet", (), "train"),
        experiment_id=experiment_id,
        fold=0,
        trainer="nnUNetTrainer_50epochs",
        monitor_gpu=True,
        number_of_epochs=50,
    )

    experiment_dir = backend.paths.reports / experiment_id
    artifact_path = experiment_dir / "gpu_summary.json"
    manifest = json.loads((experiment_dir / "experiment.json").read_text(encoding="utf-8"))
    assert manifest["peak_vram_mb"] == GPU_SUMMARY["peak_memory_used_mb"]
    assert manifest["mean_gpu_utilization"] == GPU_SUMMARY["mean_gpu_utilization_percent"]
    assert manifest["train_peak_vram_mb"] == manifest["peak_vram_mb"]
    assert manifest["train_mean_gpu_utilization"] == manifest["mean_gpu_utilization"]
    assert manifest["train_gpu_telemetry_file"] == str(artifact_path)
    assert json.loads(artifact_path.read_text(encoding="utf-8")) == GPU_SUMMARY


def test_reconcile_telemetry_restores_training_aliases_after_predict_overwrite(
    tmp_path: Path,
) -> None:
    backend = NNUNetV2Backend(project_root=tmp_path)
    experiment_id = backend.initialize_experiment(
        "existing_experiment",
        kind="prelim",
        fold=0,
        trainer="nnUNetTrainer_50epochs",
    )
    experiment_dir = backend.paths.reports / experiment_id
    training_summary = dict(GPU_SUMMARY)
    prediction_summary = {
        **GPU_SUMMARY,
        "peak_memory_used_mb": 3000.0,
        "mean_gpu_utilization_percent": 41.0,
    }
    (experiment_dir / "gpu_summary.json").write_text(json.dumps(training_summary), encoding="utf-8")
    (experiment_dir / "predict_gpu_summary.json").write_text(
        json.dumps(prediction_summary), encoding="utf-8"
    )
    backend._update_manifest(
        experiment_id,
        {
            "peak_vram_mb": prediction_summary["peak_memory_used_mb"],
            "mean_gpu_utilization": prediction_summary["mean_gpu_utilization_percent"],
            "unrelated_provenance": "preserved",
        },
    )

    updates = backend.reconcile_telemetry(experiment_id)

    manifest = json.loads((experiment_dir / "experiment.json").read_text(encoding="utf-8"))
    assert updates["peak_vram_mb"] == training_summary["peak_memory_used_mb"]
    assert updates["mean_gpu_utilization"] == training_summary["mean_gpu_utilization_percent"]
    assert manifest["peak_vram_mb"] == training_summary["peak_memory_used_mb"]
    assert manifest["mean_gpu_utilization"] == training_summary["mean_gpu_utilization_percent"]
    assert manifest["train_peak_vram_mb"] == training_summary["peak_memory_used_mb"]
    assert manifest["predict_peak_vram_mb"] == prediction_summary["peak_memory_used_mb"]
    assert (
        manifest["predict_mean_gpu_utilization"]
        == prediction_summary["mean_gpu_utilization_percent"]
    )
    assert manifest["train_gpu_telemetry_file"] == str(
        (experiment_dir / "gpu_summary.json").resolve()
    )
    assert manifest["predict_gpu_telemetry_file"] == str(
        (experiment_dir / "predict_gpu_summary.json").resolve()
    )
    assert manifest["unrelated_provenance"] == "preserved"

    reconciled_manifest = (experiment_dir / "experiment.json").read_bytes()
    assert backend.reconcile_telemetry(experiment_id) == updates
    assert (experiment_dir / "experiment.json").read_bytes() == reconciled_manifest


@pytest.mark.parametrize(
    ("field_name", "invalid_value", "message"),
    [
        ("samples", 0, "samples must be a positive integer"),
        ("peak_memory_used_mb", "invalid", "must be numeric"),
        ("mean_gpu_utilization_percent", float("nan"), "must be finite"),
    ],
)
def test_reconcile_telemetry_rejects_invalid_summary_without_changing_manifest(
    tmp_path: Path,
    field_name: str,
    invalid_value: Any,
    message: str,
) -> None:
    backend = NNUNetV2Backend(project_root=tmp_path)
    experiment_id = backend.initialize_experiment(
        "invalid_telemetry",
        kind="prelim",
        fold=0,
        trainer="nnUNetTrainer_50epochs",
    )
    experiment_dir = backend.paths.reports / experiment_id
    invalid_summary = dict(GPU_SUMMARY)
    invalid_summary[field_name] = invalid_value
    (experiment_dir / "gpu_summary.json").write_text(json.dumps(invalid_summary), encoding="utf-8")
    manifest_path = experiment_dir / "experiment.json"
    original_manifest = manifest_path.read_bytes()

    with pytest.raises(ValueError, match=message):
        backend.reconcile_telemetry(experiment_id)

    assert manifest_path.read_bytes() == original_manifest


def test_reconcile_telemetry_cli_prints_returned_updates(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    backend = NNUNetV2Backend(project_root=tmp_path)
    experiment_id = backend.initialize_experiment(
        "cli_experiment",
        kind="prelim",
        fold=0,
        trainer="nnUNetTrainer_50epochs",
    )
    experiment_dir = backend.paths.reports / experiment_id
    (experiment_dir / "gpu_summary.json").write_text(json.dumps(GPU_SUMMARY), encoding="utf-8")

    exit_code = backend_module.main(
        [
            "--project-root",
            str(tmp_path),
            "reconcile-telemetry",
            "--experiment-id",
            experiment_id,
        ]
    )

    printed = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert printed["peak_vram_mb"] == GPU_SUMMARY["peak_memory_used_mb"]
    assert printed["train_peak_vram_mb"] == GPU_SUMMARY["peak_memory_used_mb"]
    assert printed["train_gpu_telemetry_file"] == str(
        (experiment_dir / "gpu_summary.json").resolve()
    )
