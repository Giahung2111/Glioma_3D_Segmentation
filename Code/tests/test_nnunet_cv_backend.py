from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

import glioma_seg.backends.nnunet.backend as backend_module
from glioma_seg.backends.nnunet.backend import (
    BENCHMARK_TRAINER,
    NNUNetV2Backend,
    ReadinessReport,
)
from glioma_seg.backends.nnunet.commands import CommandSpec
from glioma_seg.utils.subprocess import LiveCommandResult

GPU_SUMMARY = {
    "samples": 2,
    "peak_memory_used_mb": 9000.0,
    "dedicated_memory_total_mb": 11264.0,
    "mean_gpu_utilization_percent": 90.0,
    "peak_temperature_c": 83.0,
    "mean_power_w": 200.0,
    "backend": "test",
    "errors": [],
}


def test_benchmark_stage_does_not_lock_or_replace_the_scientific_trainer(
    tmp_path: Path,
) -> None:
    backend = NNUNetV2Backend(project_root=tmp_path)
    experiment_id = backend.initialize_experiment(
        "benchmark_then_cv", kind="fullcv", fold=0, trainer=None
    )
    benchmark_summary = {
        "trainer": BENCHMARK_TRAINER,
        "fold": 0,
        "fastest_epoch_seconds": 120.0,
    }

    backend._record_benchmark_manifest(experiment_id, benchmark_summary)
    manifest_path = backend.paths.reports / experiment_id / "experiment.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["trainer"] is None
    assert manifest["benchmark_trainer"] == BENCHMARK_TRAINER
    backend._assert_experiment_compatible(experiment_id, "nnUNetTrainer_100epochs")

    backend._update_manifest(experiment_id, {"trainer": "nnUNetTrainer_100epochs"})
    backend._record_benchmark_manifest(experiment_id, benchmark_summary)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["trainer"] == "nnUNetTrainer_100epochs"
    assert manifest["benchmark_summary"]["trainer"] == BENCHMARK_TRAINER


def test_legacy_benchmark_only_manifest_can_transition_once_to_cv_trainer(
    tmp_path: Path,
) -> None:
    backend = NNUNetV2Backend(project_root=tmp_path)
    experiment_id = backend.initialize_experiment(
        "legacy_benchmark", kind="benchmark", fold=0, trainer=BENCHMARK_TRAINER
    )
    backend._assert_experiment_compatible(experiment_id, "nnUNetTrainer_100epochs")


class _FakeGPUSummary:
    def to_dict(self) -> dict[str, Any]:
        return dict(GPU_SUMMARY)


class _FakeGPUMonitor:
    latest = None

    def __init__(self, output_path: Path, *, interval_seconds: float) -> None:
        self.output_path = output_path
        self.interval_seconds = interval_seconds

    def start(self) -> _FakeGPUMonitor:
        return self

    def stop(self) -> _FakeGPUSummary:
        return _FakeGPUSummary()


def _successful_command(
    argv: Sequence[str | Path],
    *,
    log_path: Path,
    **_: Any,
) -> LiveCommandResult:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("successful attempt\n", encoding="utf-8")
    return LiveCommandResult(
        argv=tuple(str(part) for part in argv),
        returncode=0,
        started_at="2026-08-29T00:00:00+00:00",
        ended_at="2026-08-29T00:00:10+00:00",
        elapsed_seconds=10.0,
        log_path=log_path,
        stdout_tail=(),
        stderr_tail=(),
    )


def _backend_with_split(tmp_path: Path) -> tuple[NNUNetV2Backend, list[list[str]]]:
    backend = NNUNetV2Backend(project_root=tmp_path)
    case_ids = [f"BraTS-GLI-{index:05d}-000" for index in range(10)]
    validation_folds = [case_ids[index * 2 : index * 2 + 2] for index in range(5)]
    splits = [
        {
            "train": [case for case in case_ids if case not in validation_ids],
            "val": validation_ids,
        }
        for validation_ids in validation_folds
    ]
    backend.dataset_preprocessed_dir.mkdir(parents=True)
    (backend.dataset_preprocessed_dir / "splits_final.json").write_text(
        json.dumps(splits), encoding="utf-8"
    )
    return backend, validation_folds


def _materialize_complete_fold(
    backend: NNUNetV2Backend,
    experiment_id: str,
    trainer: str,
    fold: int,
    case_ids: list[str],
) -> None:
    output = backend._model_output_folder(trainer) / f"fold_{fold}"
    validation = output / "validation"
    validation.mkdir(parents=True)
    (output / "checkpoint_final.pth").write_bytes(b"checkpoint")
    (output / "glioma_experiment_owner.json").write_text(
        json.dumps(backend._model_owner(experiment_id, trainer, fold)), encoding="utf-8"
    )
    for case_id in case_ids:
        (validation / f"{case_id}.nii.gz").write_bytes(b"segmentation")
        (validation / f"{case_id}.npz").write_bytes(b"probabilities")
        (validation / f"{case_id}.pkl").write_bytes(b"properties")
    (validation / "summary.json").write_text(
        json.dumps(
            {
                "metric_per_case": [
                    {"prediction_file": str(validation / f"{case_id}.nii.gz")}
                    for case_id in case_ids
                ]
            }
        ),
        encoding="utf-8",
    )


def _materialize_checkpointless_owned_fold(
    backend: NNUNetV2Backend,
    experiment_id: str,
    trainer: str,
    fold: int,
) -> Path:
    output = backend._model_output_folder(trainer) / f"fold_{fold}"
    output.mkdir(parents=True)
    (output / "glioma_experiment_owner.json").write_text(
        json.dumps(backend._model_owner(experiment_id, trainer, fold)), encoding="utf-8"
    )
    (output / "training_log_2026_08_29.txt").write_text(
        "interrupted before first checkpoint\n", encoding="utf-8"
    )
    return output


def test_strict_fold_audit_records_complete_and_detects_missing_probability(
    tmp_path: Path,
) -> None:
    backend, validation_folds = _backend_with_split(tmp_path)
    trainer = "nnUNetTrainer_100epochs"
    experiment_id = backend.initialize_experiment(
        "cv100", kind="compute_limited_cross_validation", fold=0, trainer=trainer
    )
    _materialize_complete_fold(
        backend, experiment_id, trainer, 0, validation_folds[0]
    )

    complete = backend.audit_fold_artifacts(
        experiment_id=experiment_id,
        trainer=trainer,
        fold=0,
        require_probabilities=True,
        record=True,
    )
    assert complete["complete"] is True
    assert complete["valid"] is True
    assert complete["safe_to_resume"] is True
    manifest = json.loads(
        (backend.paths.reports / experiment_id / "experiment.json").read_text(encoding="utf-8")
    )
    assert manifest["fold_runs"]["0"]["artifact_status"] == "COMPLETE"

    missing_case = validation_folds[0][0]
    (
        backend._model_output_folder(trainer)
        / "fold_0"
        / "validation"
        / f"{missing_case}.npz"
    ).unlink()
    incomplete = backend.audit_fold_artifacts(
        experiment_id=experiment_id,
        trainer=trainer,
        fold=0,
        require_probabilities=True,
    )
    assert incomplete["complete"] is False
    assert incomplete["safe_to_resume"] is True
    assert incomplete["inventories"][".npz"]["missing"] == [missing_case]


def test_audit_fold_cli_writes_output_and_uses_completion_exit_code(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    backend, validation_folds = _backend_with_split(tmp_path)
    trainer = "nnUNetTrainer_100epochs"
    experiment_id = backend.initialize_experiment(
        "cv_cli", kind="compute_limited_cross_validation", fold=0, trainer=trainer
    )
    _materialize_complete_fold(
        backend, experiment_id, trainer, 0, validation_folds[0]
    )
    output = tmp_path / "audit.json"

    exit_code = backend_module.main(
        [
            "--project-root",
            str(tmp_path),
            "audit-fold",
            "--experiment-id",
            experiment_id,
            "--fold",
            "0",
            "--trainer",
            trainer,
            "--require-probabilities",
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    assert json.loads(output.read_text(encoding="utf-8"))["complete"] is True
    assert json.loads(capsys.readouterr().out)["valid"] is True


def test_archive_restart_fold_cli_preserves_owned_checkpointless_output(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    backend, _ = _backend_with_split(tmp_path)
    trainer = "nnUNetTrainer_100epochs"
    experiment_id = backend.initialize_experiment(
        "cv_restart", kind="fullcv", fold=0, trainer=trainer
    )
    fold_output = _materialize_checkpointless_owned_fold(
        backend, experiment_id, trainer, 0
    )
    audit = backend.audit_fold_artifacts(
        experiment_id=experiment_id,
        trainer=trainer,
        fold=0,
    )
    assert audit["safe_to_resume"] is False
    assert audit["safe_to_restart"] is True

    cli_output = tmp_path / "restart_archive.json"
    exit_code = backend_module.main(
        [
            "--project-root",
            str(tmp_path),
            "archive-restart-fold",
            "--experiment-id",
            experiment_id,
            "--fold",
            "0",
            "--trainer",
            trainer,
            "--output",
            str(cli_output),
        ]
    )

    assert exit_code == 0
    archive_record = json.loads(cli_output.read_text(encoding="utf-8"))
    archive_path = Path(archive_record["archive"])
    assert archive_record["safe_to_start_fresh"] is True
    assert not fold_output.exists()
    assert archive_path.parent == backend._model_output_folder(trainer).resolve()
    assert (archive_path / "training_log_2026_08_29.txt").is_file()
    assert (archive_path / "glioma_experiment_owner.json").is_file()
    assert (archive_path / "glioma_restart_archive.json").is_file()
    manifest = json.loads(
        (backend.paths.reports / experiment_id / "experiment.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["fold_runs"]["0"]["restart_archives"][0]["archive"] == str(
        archive_path
    )
    assert json.loads(capsys.readouterr().out)["safe_to_start_fresh"] is True


@pytest.mark.parametrize("unsafe_state", ["checkpoint", "foreign_owner"])
def test_archive_restart_fold_refuses_unsafe_output_without_moving_it(
    tmp_path: Path,
    unsafe_state: str,
) -> None:
    backend, _ = _backend_with_split(tmp_path)
    trainer = "nnUNetTrainer_100epochs"
    experiment_id = backend.initialize_experiment(
        "cv_restart_refusal", kind="fullcv", fold=0, trainer=trainer
    )
    fold_output = _materialize_checkpointless_owned_fold(
        backend, experiment_id, trainer, 0
    )
    if unsafe_state == "checkpoint":
        nested = fold_output / "unexpected"
        nested.mkdir()
        (nested / "checkpoint_surprise.pth").write_bytes(b"checkpoint")
    else:
        (fold_output / "glioma_experiment_owner.json").write_text(
            json.dumps(backend._model_owner("different_experiment", trainer, 0)),
            encoding="utf-8",
        )

    audit = backend.audit_fold_artifacts(
        experiment_id=experiment_id,
        trainer=trainer,
        fold=0,
    )
    assert audit["safe_to_restart"] is False
    with pytest.raises(RuntimeError, match="cannot be archived"):
        backend.archive_restartable_fold(
            experiment_id=experiment_id,
            trainer=trainer,
            fold=0,
        )

    assert fold_output.is_dir()
    assert not list(fold_output.parent.glob("glioma_arch_f0_*"))


def test_cv_execution_keeps_fold_logs_runtime_gpu_and_manifest_separate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = NNUNetV2Backend(project_root=tmp_path)
    trainer = "nnUNetTrainer_100epochs"
    experiment_id = backend.initialize_experiment(
        "cv_scoped", kind="compute_limited_cross_validation", fold=0, trainer=trainer
    )
    monkeypatch.setattr(backend_module, "GPUMonitor", _FakeGPUMonitor)
    monkeypatch.setattr(backend_module, "run_live_command", _successful_command)
    monkeypatch.setattr(backend, "_official_command", lambda spec: spec)

    for fold in (0, 1):
        backend._execute(
            CommandSpec("fake_nnunet", (), "train"),
            experiment_id=experiment_id,
            fold=fold,
            trainer=trainer,
            monitor_gpu=True,
            number_of_epochs=100,
            fold_scoped=True,
            expected_validation_cases=2,
        )

    experiment_dir = backend.paths.reports / experiment_id
    assert (experiment_dir / "logs" / "train_fold_0.log").parent.is_dir()
    assert (experiment_dir / "folds" / "fold_0" / "runtime.json").is_file()
    assert (experiment_dir / "folds" / "fold_1" / "runtime.json").is_file()
    assert (experiment_dir / "folds" / "fold_0" / "gpu_summary.json").is_file()
    assert (experiment_dir / "folds" / "fold_1" / "gpu_summary.json").is_file()
    assert not (experiment_dir / "runtime.json").exists()
    assert not (experiment_dir / "gpu_summary.json").exists()
    manifest = json.loads((experiment_dir / "experiment.json").read_text(encoding="utf-8"))
    assert set(manifest["fold_runs"]) == {"0", "1"}
    assert (
        manifest["fold_runs"]["0"]["gpu_telemetry_csv"]
        != manifest["fold_runs"]["1"]["gpu_telemetry_csv"]
    )


def test_cv_keyboard_interrupt_records_fold_status_without_assertion_masking(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = NNUNetV2Backend(project_root=tmp_path)
    trainer = "nnUNetTrainer_100epochs"
    experiment_id = backend.initialize_experiment(
        "cv_interrupt", kind="compute_limited_cross_validation", fold=0, trainer=trainer
    )
    monkeypatch.setattr(backend_module, "GPUMonitor", _FakeGPUMonitor)
    monkeypatch.setattr(backend, "_official_command", lambda spec: spec)

    def interrupt(*_: Any, **__: Any) -> LiveCommandResult:
        raise KeyboardInterrupt

    monkeypatch.setattr(backend_module, "run_live_command", interrupt)
    with pytest.raises(KeyboardInterrupt):
        backend._execute(
            CommandSpec("fake_nnunet", (), "train"),
            experiment_id=experiment_id,
            fold=3,
            trainer=trainer,
            monitor_gpu=True,
            number_of_epochs=100,
            fold_scoped=True,
        )

    fold_dir = backend.paths.reports / experiment_id / "folds" / "fold_3"
    assert (fold_dir / "runtime.json").is_file()
    assert (fold_dir / "gpu_summary.json").is_file()
    manifest = json.loads(
        (backend.paths.reports / experiment_id / "experiment.json").read_text(encoding="utf-8")
    )
    assert manifest["fold_runs"]["3"]["stage_status"] == "INTERRUPTED"
    assert manifest["fold_runs"]["3"]["failure_type"] == "KeyboardInterrupt"


def test_interrupted_then_resumed_fold_preserves_attempts_and_cumulative_telemetry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = NNUNetV2Backend(project_root=tmp_path)
    trainer = "nnUNetTrainer_100epochs"
    experiment_id = backend.initialize_experiment(
        "cv_attempts", kind="fullcv", fold=0, trainer=trainer
    )
    monkeypatch.setattr(backend_module, "GPUMonitor", _FakeGPUMonitor)
    monkeypatch.setattr(backend, "_official_command", lambda spec: spec)

    def interrupt(*_: Any, log_path: Path, **__: Any) -> LiveCommandResult:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("interrupted attempt\n", encoding="utf-8")
        raise KeyboardInterrupt

    monkeypatch.setattr(backend_module, "run_live_command", interrupt)
    with pytest.raises(KeyboardInterrupt):
        backend._execute(
            CommandSpec("fake_nnunet", (), "train"),
            experiment_id=experiment_id,
            fold=0,
            trainer=trainer,
            monitor_gpu=True,
            number_of_epochs=100,
            fold_scoped=True,
        )

    fold_dir = backend.paths.reports / experiment_id / "folds" / "fold_0"
    attempt_1_runtime = json.loads(
        (fold_dir / "attempts" / "attempt_001" / "runtime.json").read_text(
            encoding="utf-8"
        )
    )
    monkeypatch.setattr(backend_module, "run_live_command", _successful_command)
    backend._execute(
        CommandSpec("fake_nnunet", (), "train"),
        experiment_id=experiment_id,
        fold=0,
        trainer=trainer,
        monitor_gpu=True,
        number_of_epochs=100,
        fold_scoped=True,
        resume_attempt=True,
    )

    attempt_2_runtime = json.loads(
        (fold_dir / "attempts" / "attempt_002" / "runtime.json").read_text(
            encoding="utf-8"
        )
    )
    cumulative_runtime = json.loads((fold_dir / "runtime.json").read_text(encoding="utf-8"))
    cumulative_gpu = json.loads((fold_dir / "gpu_summary.json").read_text(encoding="utf-8"))
    assert cumulative_runtime["attempt_count"] == 2
    assert cumulative_runtime["total_seconds"] == pytest.approx(
        attempt_1_runtime["total_seconds"] + attempt_2_runtime["total_seconds"]
    )
    assert cumulative_runtime["number_of_epochs"] == 100
    assert cumulative_runtime["epochs_observed_across_attempts"] == 0
    assert cumulative_gpu["attempt_count"] == 2
    assert cumulative_gpu["samples"] == 4
    assert cumulative_gpu["estimated_energy_wh"] == pytest.approx(
        200.0 * cumulative_runtime["total_seconds"] / 3600.0
    )

    manifest = json.loads(
        (backend.paths.reports / experiment_id / "experiment.json").read_text(encoding="utf-8")
    )
    attempts = manifest["fold_runs"]["0"]["attempts"]
    assert [attempt["status"] for attempt in attempts] == ["INTERRUPTED", "DONE"]
    assert attempts[0]["runtime_file"] != attempts[1]["runtime_file"]
    assert attempts[0]["gpu_telemetry_csv"] != attempts[1]["gpu_telemetry_csv"]
    combined_log = backend.paths.reports / experiment_id / "logs" / "train_fold_0.log"
    combined_text = combined_log.read_text(encoding="utf-8")
    assert "interrupted attempt" in combined_text
    assert "successful attempt" in combined_text


def test_100epoch_training_is_compute_limited_and_fold_scoped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = NNUNetV2Backend(project_root=tmp_path)
    trainer = "nnUNetTrainer_100epochs"
    experiment_id = backend.initialize_experiment(
        "cv_protocol", kind="compute_limited_cross_validation", fold=0, trainer=trainer
    )
    readiness = ReadinessReport(
        experiment_id=experiment_id,
        dataset=backend.dataset_name,
        dataset_id=backend.dataset_id,
        fold=2,
        configuration=backend.configuration,
        trainer=trainer,
        details={"fold_train_cases": 8, "fold_val_cases": 2},
    )
    captured: dict[str, Any] = {}

    monkeypatch.setattr(backend, "check_readiness", lambda **_: readiness)
    monkeypatch.setattr(backend, "_snapshot_reproducibility", lambda *_: None)
    monkeypatch.setattr(backend_module, "collect_system_report", lambda *_: {})
    monkeypatch.setattr(
        backend,
        "audit_fold_artifacts",
        lambda **_: {"complete": True, "checks": []},
    )
    monkeypatch.setattr(backend, "_write_cv_aggregate_telemetry", lambda *_: None)

    def execute(spec: CommandSpec, **kwargs: Any) -> LiveCommandResult:
        captured["argv"] = spec.argv
        captured.update(kwargs)
        return _successful_command(spec.argv, log_path=tmp_path / "train.log")

    monkeypatch.setattr(backend, "_execute", execute)
    backend.train(fold=2, trainer=trainer, experiment_id=experiment_id)

    manifest = json.loads(
        (backend.paths.reports / experiment_id / "experiment.json").read_text(encoding="utf-8")
    )
    assert manifest["experiment_kind"] == "fullcv"
    assert manifest["baseline_classification"] == "compute_limited_cross_validation"
    assert manifest["folds"] == [0, 1, 2, 3, 4]
    assert captured["fold_scoped"] is True
    assert captured["number_of_epochs"] == 100
    assert "nnUNetTrainer_100epochs" in captured["argv"]
