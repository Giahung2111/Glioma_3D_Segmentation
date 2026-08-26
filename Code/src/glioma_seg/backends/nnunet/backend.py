"""Project orchestration around the unmodified official nnU-Net v2 CLI.

The baseline boundary is intentional: this module builds and executes official
console commands. Dataset conversion, evaluation, reports, and experiments stay
in ``glioma_seg``; model/trainer implementations stay in ``External/nnUNet``.
"""

from __future__ import annotations

import argparse
import ctypes
import datetime as dt
import importlib.metadata
import json
import math
import os
import platform
import secrets
import shutil
import subprocess
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from glioma_seg.backends.base import BackendArtifacts, SegmentationBackend
from glioma_seg.backends.nnunet.artifacts import (
    PreprocessingArtifactReport,
    validate_preprocessing_artifacts,
)
from glioma_seg.backends.nnunet.commands import (
    CommandSpec,
    build_accumulate_cross_validation,
    build_benchmark,
    build_plan_and_preprocess,
    build_predict,
    build_train,
)
from glioma_seg.backends.nnunet.parser import (
    load_json,
    summarize_benchmark,
    summarize_fingerprint,
    summarize_plans,
)
from glioma_seg.monitoring.gpu_monitor import GPUMonitor, query_gpu_once
from glioma_seg.monitoring.process_monitor import NNUNetProcessMonitor
from glioma_seg.monitoring.timing import build_runtime_record, write_json_atomic
from glioma_seg.utils.subprocess import (
    LiveCommandError,
    LiveCommandResult,
    format_command,
    run_live_command,
)

DATASET_ID = 501
DATASET_NAME = "Dataset501_BraTS2023GLI"
CONFIGURATION = "3d_fullres"
PLANS_NAME = "nnUNetPlans"
EXPECTED_UPSTREAM_COMMIT = "0e495086eb108ff79afe106291e8c15bd2f2bc3a"
EXPECTED_UPSTREAM_VERSION = "2.8.1"
PRELIMINARY_TRAINERS = {"nnUNetTrainer_20epochs": 20, "nnUNetTrainer_50epochs": 50}


@dataclass(frozen=True)
class NNUNetPaths:
    project_root: Path
    code_root: Path
    workspace: Path
    raw: Path
    preprocessed: Path
    results: Path
    reports: Path
    telemetry: Path
    predictions: Path
    external_nnunet: Path

    @classmethod
    def create(cls, project_root: Path | None = None) -> NNUNetPaths:
        if project_root is None:
            # backend.py -> nnunet -> backends -> glioma_seg -> src -> Code -> project
            project_root = Path(__file__).resolve().parents[5]
        root = project_root.resolve()
        if root.name.casefold() == "code":
            root = root.parent
        code_root = root / "Code"
        workspace = root / "Workspace"
        return cls(
            project_root=root,
            code_root=code_root,
            workspace=workspace,
            raw=workspace / "nnUNet_raw",
            preprocessed=workspace / "nnUNet_preprocessed",
            results=workspace / "nnUNet_results",
            reports=workspace / "reports",
            telemetry=workspace / "telemetry",
            predictions=workspace / "predictions",
            external_nnunet=root / "External" / "nnUNet",
        )

    def environment(self, *, augmentation_workers: int | None = None) -> dict[str, str]:
        values = {
            "nnUNet_raw": str(self.raw),
            "nnUNet_preprocessed": str(self.preprocessed),
            "nnUNet_results": str(self.results),
        }
        if augmentation_workers is not None:
            values["nnUNet_n_proc_DA"] = str(augmentation_workers)
        return values

    def ensure_output_directories(self) -> None:
        for path in (
            self.raw,
            self.preprocessed,
            self.results,
            self.reports,
            self.telemetry,
            self.predictions,
        ):
            path.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class ReadinessCheck:
    name: str
    ok: bool
    critical: bool
    detail: str


@dataclass
class ReadinessReport:
    experiment_id: str
    dataset: str
    dataset_id: int
    fold: int
    configuration: str
    trainer: str
    checks: list[ReadinessCheck] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def ready(self) -> bool:
        return all(check.ok or not check.critical for check in self.checks)

    def add(self, name: str, ok: bool, detail: str, *, critical: bool = True) -> None:
        self.checks.append(ReadinessCheck(name, ok, critical, detail))

    def to_dict(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "experiment_id": self.experiment_id,
            "dataset": self.dataset,
            "dataset_id": self.dataset_id,
            "fold": self.fold,
            "configuration": self.configuration,
            "trainer": self.trainer,
            "checks": [asdict(check) for check in self.checks],
            "details": self.details,
        }


def create_experiment_id(kind: str = "prelim", *, fold: int = 0) -> str:
    now = dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    nonce = secrets.token_hex(3)
    return f"nnunetv2_3dfullres_fold{fold}_{kind}_{now}_{nonce}"


def _memory_status() -> tuple[int | None, int | None]:
    try:
        import psutil  # type: ignore[import-untyped]

        memory = psutil.virtual_memory()
        return int(memory.total), int(memory.available)
    except Exception:
        if os.name != "nt":
            return None, None

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MEMORYSTATUSEX()
        status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.ullTotalPhys), int(status.ullAvailPhys)
        return None, None


def _cpu_counts() -> tuple[int | None, int | None]:
    logical = os.cpu_count()
    try:
        import psutil

        physical = psutil.cpu_count(logical=False)
    except Exception:
        physical = None
    return physical, logical


def recommend_augmentation_workers() -> int:
    """Conservative host-aware value; leaves cores and RAM for Windows."""

    physical, logical = _cpu_counts()
    usable_cores = physical or max(2, (logical or 4) // 2)
    _, available_memory = _memory_status()
    # A conservative upper bound of one worker per ~2.5 GiB available RAM.
    memory_bound = max(2, int(available_memory / (2.5 * 1024**3))) if available_memory else 12
    return max(2, min(12, max(2, usable_cores - 2), memory_bound))


def _run_metadata_command(argv: Sequence[str], cwd: Path | None = None) -> str | None:
    try:
        completed = subprocess.run(
            argv,
            cwd=str(cwd) if cwd else None,
            check=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=30,
        )
        return completed.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _distribution_metadata() -> dict[str, Any]:
    try:
        distribution = importlib.metadata.distribution("nnunetv2")
    except importlib.metadata.PackageNotFoundError:
        return {"installed": False, "version": None, "editable": False, "source": None}
    direct_url_raw = distribution.read_text("direct_url.json")
    direct_url = json.loads(direct_url_raw) if direct_url_raw else {}
    return {
        "installed": True,
        "version": distribution.version,
        "editable": bool(direct_url.get("dir_info", {}).get("editable")),
        "source": direct_url.get("url"),
    }


def _cudnn_version(torch_module: Any) -> int | None:
    """Read the partially typed torch cuDNN API behind one narrow boundary."""

    version = torch_module.backends.cudnn.version()
    return int(version) if version is not None else None


def collect_system_report(paths: NNUNetPaths) -> dict[str, Any]:
    physical, logical = _cpu_counts()
    total_ram, available_ram = _memory_status()
    disk = shutil.disk_usage(paths.project_root.anchor or paths.project_root)
    try:
        gpu = asdict(query_gpu_once())
    except Exception as exc:
        gpu = {"available": False, "error": f"{type(exc).__name__}: {exc}"}

    torch_info: dict[str, Any]
    try:
        import torch

        torch_info = {
            "installed": True,
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_runtime": torch.version.cuda,
            "cudnn": _cudnn_version(torch),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    except Exception as exc:
        torch_info = {"installed": False, "error": f"{type(exc).__name__}: {exc}"}

    upstream_commit = _run_metadata_command(["git", "rev-parse", "HEAD"], cwd=paths.external_nnunet)
    upstream_status = _run_metadata_command(
        ["git", "status", "--porcelain", "--untracked-files=all"], cwd=paths.external_nnunet
    )
    project_commit = _run_metadata_command(["git", "rev-parse", "HEAD"], cwd=paths.code_root)
    driver = _run_metadata_command(
        [
            "nvidia-smi",
            "--query-gpu=driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    gpu_name = gpu.get("gpu_name")
    gpu_vram_mb = gpu.get("memory_total_mb")
    torch_version = torch_info.get("version")
    report: dict[str, Any] = {
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "platform": platform.platform(),
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
        },
        "python": platform.python_version(),
        "python_version": platform.python_version(),
        "python_details": {"version": platform.python_version(), "executable": sys.executable},
        "cpu": {
            "processor": platform.processor(),
            "physical_cores": physical,
            "logical_cores": logical,
        },
        "ram": {"total_bytes": total_ram, "available_bytes": available_ram},
        "disk": {
            "root": paths.project_root.anchor,
            "total_bytes": disk.total,
            "free_bytes": disk.free,
        },
        "gpu_details": gpu,
        "gpu_name": gpu_name,
        "gpu_vram_mb": gpu_vram_mb,
        "nvidia_driver": driver.splitlines()[0] if driver else None,
        "torch": torch_version,
        "torch_version": torch_version,
        "torch_details": torch_info,
        "cuda": torch_info.get("cuda_runtime"),
        "cuda_runtime": torch_info.get("cuda_runtime"),
        "cudnn": torch_info.get("cudnn"),
        "cudnn_version": torch_info.get("cudnn"),
        "nnunet": _distribution_metadata(),
        "upstream": {
            "path": str(paths.external_nnunet),
            "commit": upstream_commit,
            "expected_commit": EXPECTED_UPSTREAM_COMMIT,
            "working_tree_clean": upstream_status == "",
            "working_tree_status": upstream_status.splitlines() if upstream_status else [],
        },
        "project_git_commit": project_commit,
        "nnunet_environment": paths.environment(),
        "recommended_nnUNet_n_proc_DA": recommend_augmentation_workers(),
    }
    torch_version_text = str(torch_version or "")
    report["nnUNet_version"] = report["nnunet"].get("version")
    report["safety_checks"] = {
        "torch_2_9_rejected": torch_version_text.startswith("2.9"),
        "official_upstream_commit": upstream_commit == EXPECTED_UPSTREAM_COMMIT,
        "upstream_unmodified": upstream_status == "",
        "editable_nnunet_install": bool(report["nnunet"].get("editable")),
    }
    return report


class NNUNetV2Backend(SegmentationBackend):
    """Typed, monitored adapter for official nnU-Net v2 console entrypoints."""

    def __init__(
        self,
        *,
        project_root: Path | None = None,
        dataset_id: int = DATASET_ID,
        dataset_name: str = DATASET_NAME,
        configuration: str = CONFIGURATION,
        plans_name: str = PLANS_NAME,
    ) -> None:
        self.paths = NNUNetPaths.create(project_root)
        self.dataset_id = int(dataset_id)
        self.dataset_name = dataset_name
        self.configuration = configuration
        self.plans_name = plans_name
        self.paths.ensure_output_directories()

    @property
    def dataset_raw_dir(self) -> Path:
        return self.paths.raw / self.dataset_name

    @property
    def dataset_preprocessed_dir(self) -> Path:
        return self.paths.preprocessed / self.dataset_name

    def _resolve_cli(self, executable: str) -> Path:
        scripts_dir = Path(sys.executable).resolve().parent
        console_dirs = (scripts_dir, scripts_dir / "Scripts")
        candidates = [
            candidate
            for console_dir in console_dirs
            for candidate in (
                console_dir / f"{executable}.exe",
                console_dir / f"{executable}.cmd",
                console_dir / executable,
            )
        ]
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        discovered = shutil.which(executable)
        if discovered:
            return Path(discovered).resolve()
        raise FileNotFoundError(
            f"Official entrypoint {executable!r} is unavailable below {scripts_dir}. "
            f"Activate/install Code/.venv and run `pip install -e "
            f"{self.paths.external_nnunet}`."
        )

    def _official_command(self, spec: CommandSpec) -> CommandSpec:
        return spec.with_executable(self._resolve_cli(spec.executable))

    def _experiment_dir(self, experiment_id: str) -> Path:
        return self.paths.reports / experiment_id

    def initialize_experiment(
        self,
        experiment_id: str | None,
        *,
        kind: str,
        fold: int,
        trainer: str | None = None,
    ) -> str:
        identifier = experiment_id or create_experiment_id(kind, fold=fold)
        allowed_characters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
        if any(character not in allowed_characters for character in identifier):
            raise ValueError(
                "experiment_id may contain only letters, digits, underscore, and hyphen"
            )
        experiment_dir = self._experiment_dir(identifier)
        experiment_dir.mkdir(parents=True, exist_ok=True)
        for child in ("logs", "config_snapshot", "figures"):
            (experiment_dir / child).mkdir(exist_ok=True)
        manifest_path = experiment_dir / "experiment.json"
        if not manifest_path.exists():
            write_json_atomic(
                manifest_path,
                {
                    "experiment_id": identifier,
                    "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
                    "experiment_kind": kind,
                    "baseline_classification": (
                        "standard_reference_baseline"
                        if kind in {"prelim", "fullcv", "benchmark"}
                        else "custom_experiment"
                    ),
                    "upstream_source_modified": False,
                    "dataset": self.dataset_name,
                    "dataset_id": self.dataset_id,
                    "fold": fold,
                    "configuration": self.configuration,
                    "framework": "nnU-Net v2",
                    "model": "nnU-Net v2",
                    "nnUNet_version": _distribution_metadata().get("version"),
                    "trainer": trainer,
                    "command_history": [],
                    "notes": [],
                },
            )
        return identifier

    def _update_manifest(self, experiment_id: str, updates: Mapping[str, Any]) -> None:
        manifest_path = self._experiment_dir(experiment_id) / "experiment.json"
        manifest: dict[str, Any] = (
            load_json(manifest_path) if manifest_path.exists() else {"experiment_id": experiment_id}
        )
        for key, value in updates.items():
            if key == "command" and value is not None:
                history = list(manifest.get("command_history", []))
                history.append(value)
                manifest["command_history"] = history
            manifest[key] = value
        write_json_atomic(manifest_path, manifest)

    def _snapshot_reproducibility(
        self,
        experiment_id: str,
        config_path: Path | None = None,
    ) -> None:
        destination = self._experiment_dir(experiment_id) / "config_snapshot"
        sources = [
            self.dataset_raw_dir / "dataset.json",
            self.dataset_preprocessed_dir / "nnUNetPlans.json",
            self.dataset_preprocessed_dir / "dataset_fingerprint.json",
            self.dataset_preprocessed_dir / "splits_final.json",
        ]
        if config_path:
            sources.append(config_path)
        for source in sources:
            if source.is_file():
                shutil.copy2(source, destination / source.name)

    def _model_output_folder(self, trainer: str) -> Path:
        return (
            self.paths.results
            / self.dataset_name
            / f"{trainer}__{self.plans_name}__{self.configuration}"
        )

    def _model_owner_path(self, trainer: str, fold: int) -> Path:
        return self._model_output_folder(trainer) / f"fold_{fold}" / "glioma_experiment_owner.json"

    def _model_owner(self, experiment_id: str, trainer: str, fold: int) -> dict[str, Any]:
        return {
            "experiment_id": experiment_id,
            "dataset_id": self.dataset_id,
            "dataset_name": self.dataset_name,
            "configuration": self.configuration,
            "plans_name": self.plans_name,
            "trainer": trainer,
            "fold": fold,
        }

    def _benchmark_result_path(self) -> Path:
        return (
            self._model_output_folder("nnUNetTrainerBenchmark_5epochs")
            / "fold_0"
            / "benchmark_result.json"
        )

    def _preprocessing_artifacts(self, *, ensure_splits: bool) -> PreprocessingArtifactReport:
        """Validate the exact v2.8.1 output and its deterministic official split."""

        return validate_preprocessing_artifacts(
            raw_dataset_dir=self.dataset_raw_dir,
            preprocessed_dataset_dir=self.dataset_preprocessed_dir,
            dataset_name=self.dataset_name,
            configuration=self.configuration,
            plans_name=self.plans_name,
            expected_case_count=1251,
            ensure_splits=ensure_splits,
        )

    def _conversion_manifest_candidates(self, experiment_id: str) -> tuple[Path, ...]:
        return (
            self._experiment_dir(experiment_id) / "nnunet_conversion.json",
            self.paths.raw / ".glioma_manifests" / f"{self.dataset_name}_conversion.json",
            self.paths.reports / "nnunet_conversion.json",
        )

    def _validate_conversion_manifest(self, experiment_id: str) -> tuple[bool, str, dict[str, Any]]:
        """Cross-check conversion provenance against the exact training inventory."""

        manifest_path = next(
            (
                candidate
                for candidate in self._conversion_manifest_candidates(experiment_id)
                if candidate.is_file()
            ),
            None,
        )
        if manifest_path is None:
            searched = [str(path) for path in self._conversion_manifest_candidates(experiment_id)]
            return False, f"Conversion manifest missing; searched={searched}", {}

        payload = load_json(manifest_path)
        files = payload.get("files")
        methods = payload.get("method_counts")
        if not isinstance(files, list) or not isinstance(methods, Mapping):
            return False, f"Invalid conversion manifest schema: {manifest_path}", {}

        expected_case_ids = {
            path.name.removesuffix("_0000.nii.gz")
            for path in (self.dataset_raw_dir / "imagesTr").glob("*_0000.nii.gz")
        }
        expected_pairs = {
            (case_id, role)
            for case_id in expected_case_ids
            for role in ("t1n", "t1c", "t2w", "t2f", "seg")
        }
        training_entries: list[Mapping[str, Any]] = []
        for entry in files:
            if not isinstance(entry, Mapping):
                continue
            destination = str(entry.get("destination", "")).replace("/", "\\").casefold()
            if "\\imagestr\\" in destination or "\\labelstr\\" in destination:
                training_entries.append(entry)
        actual_pairs = {
            (str(entry.get("case_id")), str(entry.get("role")).casefold())
            for entry in training_entries
        }
        methods_valid = all(
            isinstance(name, str) and isinstance(value, int) and value >= 0
            for name, value in methods.items()
        )
        method_total = sum(
            value for value in methods.values() if isinstance(value, int) and value >= 0
        )
        dataset_directory = payload.get("dataset_directory")
        directory_matches = (
            isinstance(dataset_directory, str)
            and str(Path(dataset_directory).resolve()).casefold()
            == str(self.dataset_raw_dir.resolve()).casefold()
        )
        ok = (
            payload.get("dataset_id") == self.dataset_id
            and payload.get("dataset_name") == self.dataset_name
            and payload.get("training_cases") == 1251
            and len(expected_case_ids) == 1251
            and len(training_entries) == 1251 * 5
            and actual_pairs == expected_pairs
            and all(isinstance(entry, Mapping) for entry in files)
            and methods_valid
            and method_total == len(files)
            and directory_matches
        )
        summary: dict[str, Any] = {
            "path": str(manifest_path),
            "experiment_specific": manifest_path.parent == self._experiment_dir(experiment_id),
            "dataset_id": payload.get("dataset_id"),
            "dataset_name": payload.get("dataset_name"),
            "dataset_directory": dataset_directory,
            "training_cases": payload.get("training_cases"),
            "training_files": len(training_entries),
            "total_files": len(files),
            "method_counts": dict(methods),
        }
        detail = (
            f"path={manifest_path}, training_cases={payload.get('training_cases')}, "
            f"training_files={len(training_entries)}, total_files={len(files)}, "
            f"experiment_specific={summary['experiment_specific']}"
        )
        return ok, detail, summary

    def _execute(
        self,
        spec: CommandSpec,
        *,
        experiment_id: str,
        fold: int,
        trainer: str,
        monitor_gpu: bool,
        number_of_epochs: int | None = None,
    ) -> LiveCommandResult:
        command = self._official_command(spec)
        experiment_dir = self._experiment_dir(experiment_id)
        log_path = experiment_dir / "logs" / f"{spec.stage}.log"
        workers = recommend_augmentation_workers()
        environment = self.paths.environment(augmentation_workers=workers)
        progress: NNUNetProcessMonitor | None = None
        gpu_monitor: GPUMonitor | None = None
        gpu_summary: dict[str, Any] | None = None
        gpu_summary_path: Path | None = None
        if monitor_gpu:
            telemetry_name = (
                f"{experiment_id}_gpu.csv"
                if spec.stage == "train"
                else f"{experiment_id}_{spec.stage}_gpu.csv"
            )
            gpu_monitor = GPUMonitor(self.paths.telemetry / telemetry_name, interval_seconds=2.0)
            gpu_monitor.start()
        if spec.stage in {"train", "benchmark"}:
            progress = NNUNetProcessMonitor(experiment_id, fold, trainer, gpu_monitor)

        result: LiveCommandResult | None = None
        failure: LiveCommandError | None = None
        try:
            result = run_live_command(
                command.argv,
                log_path=log_path,
                cwd=self.paths.code_root,
                env=environment,
                stage=spec.stage.upper(),
                line_callback=progress.consume_line if progress else None,
                heartbeat_callback=progress.status_line if progress else None,
                heartbeat_interval_seconds=20.0,
            )
        except LiveCommandError as exc:
            result = exc.result
            failure = exc
        finally:
            if gpu_monitor:
                gpu_summary = gpu_monitor.stop().to_dict()
                summary_name = (
                    "gpu_summary.json"
                    if spec.stage == "train"
                    else f"{spec.stage}_gpu_summary.json"
                )
                gpu_summary_path = experiment_dir / summary_name
                write_json_atomic(gpu_summary_path, gpu_summary)

        assert result is not None
        epoch_durations = progress.progress.epoch_durations_seconds if progress else []
        runtime = build_runtime_record(
            stage=spec.stage,
            started_at=result.started_at,
            ended_at=result.ended_at,
            total_seconds=result.elapsed_seconds,
            number_of_epochs=number_of_epochs,
            epoch_durations_seconds=epoch_durations or (),
        )
        runtime_name = "runtime.json" if spec.stage == "train" else f"runtime_{spec.stage}.json"
        write_json_atomic(experiment_dir / runtime_name, runtime.to_dict())
        stage_aliases: dict[str, Any] = {}
        if spec.stage == "train":
            stage_aliases = {
                "training_start": result.started_at,
                "training_end": result.ended_at,
                "training_seconds": result.elapsed_seconds,
                "average_epoch_seconds": runtime.average_seconds_per_epoch,
            }
        telemetry_updates: dict[str, Any] = {}
        if gpu_summary is not None:
            assert gpu_summary_path is not None
            telemetry_updates = {
                f"{spec.stage}_peak_vram_mb": gpu_summary.get("peak_memory_used_mb"),
                f"{spec.stage}_mean_gpu_utilization": gpu_summary.get(
                    "mean_gpu_utilization_percent"
                ),
                f"{spec.stage}_gpu_telemetry_file": str(gpu_summary_path),
            }
            if spec.stage == "train":
                telemetry_updates.update(
                    {
                        "peak_vram_mb": gpu_summary.get("peak_memory_used_mb"),
                        "mean_gpu_utilization": gpu_summary.get("mean_gpu_utilization_percent"),
                    }
                )
        self._update_manifest(
            experiment_id,
            {
                "command": format_command(command.argv),
                "nnUNet_n_proc_DA": workers,
                f"{spec.stage}_start": result.started_at,
                f"{spec.stage}_end": result.ended_at,
                f"{spec.stage}_seconds": result.elapsed_seconds,
                "stage_status": "FAILED" if failure else "DONE",
                **stage_aliases,
                **telemetry_updates,
            },
        )
        if failure:
            raise failure
        return result

    @staticmethod
    def _validated_gpu_metrics(summary_path: Path) -> tuple[float, float]:
        summary = load_json(summary_path)
        samples = summary.get("samples")
        if isinstance(samples, bool) or not isinstance(samples, int) or samples < 1:
            raise ValueError(
                f"GPU telemetry samples must be a positive integer in {summary_path}: {samples!r}"
            )

        numeric_bounds: dict[str, tuple[float, float | None, bool]] = {
            "peak_memory_used_mb": (0.0, None, False),
            "dedicated_memory_total_mb": (0.0, None, True),
            "mean_gpu_utilization_percent": (0.0, 100.0, False),
            "peak_temperature_c": (0.0, None, True),
            "mean_power_w": (0.0, None, True),
        }
        validated: dict[str, float] = {}
        for field_name, (minimum, maximum, optional) in numeric_bounds.items():
            value = summary.get(field_name)
            if value is None and optional:
                continue
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise ValueError(
                    f"GPU telemetry field {field_name!r} must be numeric in "
                    f"{summary_path}: {value!r}"
                )
            numeric_value = float(value)
            if (
                not math.isfinite(numeric_value)
                or numeric_value < minimum
                or (maximum is not None and numeric_value > maximum)
            ):
                constraint = f"[{minimum}, {maximum}]" if maximum is not None else f">= {minimum}"
                raise ValueError(
                    f"GPU telemetry field {field_name!r} must be finite and {constraint} in "
                    f"{summary_path}: {value!r}"
                )
            validated[field_name] = numeric_value

        return (
            validated["peak_memory_used_mb"],
            validated["mean_gpu_utilization_percent"],
        )

    def reconcile_telemetry(self, experiment_id: str) -> dict[str, Any]:
        """Repair manifest GPU provenance from validated stage summary artifacts."""

        allowed_characters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
        if not experiment_id or any(
            character not in allowed_characters for character in experiment_id
        ):
            raise ValueError(
                "experiment_id may contain only letters, digits, underscore, and hyphen"
            )

        experiment_dir = self._experiment_dir(experiment_id)
        manifest_path = experiment_dir / "experiment.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Experiment manifest does not exist: {manifest_path}")

        stage_summaries: dict[str, Path] = {}
        training_summary = experiment_dir / "gpu_summary.json"
        if training_summary.is_file():
            stage_summaries["train"] = training_summary
        for summary_path in sorted(
            experiment_dir.glob("*_gpu_summary.json"), key=lambda path: path.name
        ):
            stage = summary_path.name.removesuffix("_gpu_summary.json")
            if not stage or any(
                character not in "abcdefghijklmnopqrstuvwxyz0123456789_" for character in stage
            ):
                raise ValueError(f"Invalid telemetry stage filename: {summary_path}")
            if stage in stage_summaries:
                raise ValueError(
                    f"Multiple GPU telemetry summaries resolve to stage {stage!r}: "
                    f"{stage_summaries[stage]} and {summary_path}"
                )
            stage_summaries[stage] = summary_path
        if not stage_summaries:
            raise FileNotFoundError(
                f"No gpu_summary.json or <stage>_gpu_summary.json found in {experiment_dir}"
            )

        updates: dict[str, Any] = {}
        ordered_stages = sorted(stage_summaries, key=lambda stage: (stage != "train", stage))
        for stage in ordered_stages:
            summary_path = stage_summaries[stage].resolve()
            peak_vram_mb, mean_gpu_utilization = self._validated_gpu_metrics(summary_path)
            updates.update(
                {
                    f"{stage}_peak_vram_mb": peak_vram_mb,
                    f"{stage}_mean_gpu_utilization": mean_gpu_utilization,
                    f"{stage}_gpu_telemetry_file": str(summary_path),
                }
            )
            if stage == "train":
                updates.update(
                    {
                        "peak_vram_mb": peak_vram_mb,
                        "mean_gpu_utilization": mean_gpu_utilization,
                    }
                )

        self._update_manifest(experiment_id, updates)
        return updates

    def prepare_dataset(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(
            "nnU-Net dataset conversion belongs to glioma_seg.data.nnunet_conversion; "
            "the backend intentionally does not reimplement or modify upstream code."
        )

    def preprocess(self, *, experiment_id: str | None = None) -> LiveCommandResult:
        identifier = self.initialize_experiment(
            experiment_id, kind="preprocess", fold=0, trainer=None
        )
        spec = build_plan_and_preprocess(
            self.dataset_id,
            self.configuration,
            verify_dataset_integrity=True,
        )
        result = self._execute(
            spec,
            experiment_id=identifier,
            fold=0,
            trainer="none",
            monitor_gpu=False,
        )
        artifact_report = self._preprocessing_artifacts(ensure_splits=True)
        artifact_report_path = self._experiment_dir(identifier) / "preprocessing_artifacts.json"
        write_json_atomic(artifact_report_path, artifact_report.to_dict())
        artifact_report.require_valid()
        plans_path = self.dataset_preprocessed_dir / "nnUNetPlans.json"
        fingerprint_path = self.dataset_preprocessed_dir / "dataset_fingerprint.json"
        plans_summary = summarize_plans(plans_path, self.configuration)
        summary: dict[str, Any] = {
            "brats_provided_preprocessing": [
                "co-registration",
                "atlas alignment/common space",
                "approximately 1 mm isotropic resampling",
                "skull stripping",
            ],
            "local_nnunet_preprocessing": [
                "automatic dataset fingerprinting",
                "experiment planning",
                "normalization",
                "model-specific resampling/preparation",
                "patch configuration",
            ],
            "plans": plans_summary,
            "fingerprint": summarize_fingerprint(fingerprint_path),
            "artifact_validation": str(artifact_report_path),
            "official_split": artifact_report.details.get("splits_file"),
            "official_split_created": artifact_report.details.get("splits_created"),
        }
        write_json_atomic(self._experiment_dir(identifier) / "preprocessing_summary.json", summary)
        self._snapshot_reproducibility(identifier)
        self._update_manifest(identifier, plans_summary)
        return result

    def benchmark(self, *, experiment_id: str | None = None) -> dict[str, Any]:
        identifier = self.initialize_experiment(
            experiment_id,
            kind="benchmark",
            fold=0,
            trainer="nnUNetTrainerBenchmark_5epochs",
        )
        self._assert_preprocessing_available(identifier)
        benchmark_path = self._benchmark_result_path()
        previous_mtime_ns = benchmark_path.stat().st_mtime_ns if benchmark_path.is_file() else None
        result = self._execute(
            build_benchmark(self.dataset_id, self.configuration, 0),
            experiment_id=identifier,
            fold=0,
            trainer="nnUNetTrainerBenchmark_5epochs",
            monitor_gpu=True,
            number_of_epochs=5,
        )
        if not benchmark_path.is_file():
            raise FileNotFoundError(
                "Official benchmark completed but benchmark_result.json was not found "
                f"at the expected Fold-0 path: {benchmark_path}"
            )
        if previous_mtime_ns is not None and benchmark_path.stat().st_mtime_ns <= previous_mtime_ns:
            raise RuntimeError(
                "Official benchmark did not refresh its exact benchmark_result.json; "
                "refusing to reuse stale timing metadata"
            )
        runtime_path = self._experiment_dir(identifier) / "runtime_benchmark.json"
        runtime = load_json(runtime_path)
        try:
            import torch

            expected_record = {
                "torch_version": str(torch.__version__),
                "cudnn_version": _cudnn_version(torch),
                "gpu_name": torch.cuda.get_device_name(0),
                "num_gpus": 1,
            }
        except Exception as exc:
            raise RuntimeError(
                f"Unable to identify the current benchmark environment: {exc}"
            ) from exc
        summary = summarize_benchmark(
            benchmark_path,
            measured_wall_seconds=result.elapsed_seconds,
            observed_mean_epoch_seconds=runtime.get("average_seconds_per_epoch"),
            expected_record=expected_record,
        )
        gpu_summary_path = self._experiment_dir(identifier) / "benchmark_gpu_summary.json"
        gpu_summary = load_json(gpu_summary_path)
        summary.update(
            {
                "experiment_id": identifier,
                "dataset_id": self.dataset_id,
                "dataset_name": self.dataset_name,
                "configuration": self.configuration,
                "fold": 0,
                "trainer": "nnUNetTrainerBenchmark_5epochs",
                "gpu_telemetry_file": str(gpu_summary_path),
                "gpu_samples": gpu_summary.get("samples"),
            }
        )
        write_json_atomic(self._experiment_dir(identifier) / "benchmark_summary.json", summary)
        self._snapshot_reproducibility(identifier)
        self._update_manifest(identifier, summary)
        if summary.get("recommended_preliminary_trainer") not in PRELIMINARY_TRAINERS:
            raise RuntimeError(
                "Official benchmark did not produce a finite epoch time; inspect "
                f"{benchmark_path} and the benchmark log before training"
            )
        if not isinstance(gpu_summary.get("samples"), int) or gpu_summary["samples"] < 1:
            raise RuntimeError(
                f"Benchmark completed without GPU telemetry samples: {gpu_summary_path}"
            )
        return summary

    def train(
        self,
        *,
        fold: int = 0,
        trainer: str,
        experiment_id: str | None = None,
        continue_training: bool = False,
        config_path: Path | None = None,
        allow_low_gpu_utilization: bool = False,
    ) -> LiveCommandResult:
        if trainer not in PRELIMINARY_TRAINERS and trainer != "nnUNetTrainer":
            raise ValueError(
                "Baseline training only accepts the official default trainer or official "
                "20/50-epoch validation trainers. Custom trainers require a separately "
                "documented custom experiment."
            )
        kind = "fullcv" if trainer == "nnUNetTrainer" else "prelim"
        identifier = self.initialize_experiment(
            experiment_id, kind=kind, fold=fold, trainer=trainer
        )
        self._update_manifest(
            identifier,
            {
                "experiment_kind": kind,
                "baseline_classification": "standard_reference_baseline",
                "fold": fold,
                "trainer": trainer,
            },
        )
        readiness = self.check_readiness(
            experiment_id=identifier,
            fold=fold,
            trainer=trainer,
            continue_training=continue_training,
            allow_low_gpu_utilization=allow_low_gpu_utilization,
        )
        readiness_path = self._experiment_dir(identifier) / "readiness.json"
        write_json_atomic(readiness_path, readiness.to_dict())
        self.print_readiness(readiness)
        if not readiness.ready:
            raise RuntimeError(
                f"Readiness gate failed; training was not started. See {readiness_path}"
            )

        environment = collect_system_report(self.paths)
        self._update_manifest(
            identifier,
            {
                "dataset_path": str(self.dataset_raw_dir),
                "dataset_case_count": readiness.details.get("dataset_case_count"),
                "GPU": environment.get("gpu_name"),
                "gpu_vram_mb": environment.get("gpu_vram_mb"),
                "torch": environment.get("torch"),
                "cuda": environment.get("cuda"),
                "cudnn": environment.get("cudnn"),
                "python": environment.get("python"),
                "nnUNet_version": environment.get("nnUNet_version"),
                "git_commit": environment.get("project_git_commit"),
                "upstream_commit": environment.get("upstream", {}).get("commit"),
                "patch_size": readiness.details.get("patch_size"),
                "batch_size": readiness.details.get("batch_size"),
                "target_spacing": readiness.details.get("target_spacing"),
                "architecture": readiness.details.get("architecture"),
                "split": {
                    "source": str(self.dataset_preprocessed_dir / "splits_final.json"),
                    "fold": fold,
                    "train_cases": readiness.details.get("fold_train_cases"),
                    "validation_cases": readiness.details.get("fold_val_cases"),
                },
            },
        )

        epochs = PRELIMINARY_TRAINERS.get(trainer, 1000)
        self._snapshot_reproducibility(identifier, config_path)
        owner_path = self._model_owner_path(trainer, fold)
        owner_path.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(owner_path, self._model_owner(identifier, trainer, fold))
        self._update_manifest(identifier, {"trainer_output_owner": str(owner_path)})
        result = self._execute(
            build_train(
                self.dataset_id,
                self.configuration,
                fold,
                trainer=trainer,
                plans=self.plans_name,
                save_probabilities=True,
                continue_training=continue_training,
            ),
            experiment_id=identifier,
            fold=fold,
            trainer=trainer,
            monitor_gpu=True,
            number_of_epochs=epochs,
        )
        output_folder = self._model_output_folder(trainer) / f"fold_{fold}"
        checkpoints = [str(path) for path in sorted(output_folder.glob("checkpoint_*.pth"))]
        self._update_manifest(
            identifier,
            {
                "epochs": epochs,
                "checkpoint_paths": checkpoints,
                "trainer_output": str(output_folder),
                "resume": continue_training,
            },
        )
        return result

    def accumulate_cross_validation(
        self,
        *,
        output_dir: Path,
        experiment_id: str | None = None,
    ) -> LiveCommandResult:
        """Accumulate official default-trainer validation predictions for all five folds."""

        identifier = self.initialize_experiment(
            experiment_id,
            kind="fullcv",
            fold=0,
            trainer="nnUNetTrainer",
        )
        model_folder = self._model_output_folder("nnUNetTrainer")
        missing = [
            model_folder / f"fold_{fold}" / "validation"
            for fold in range(5)
            if not (model_folder / f"fold_{fold}" / "validation").is_dir()
        ]
        if missing:
            raise FileNotFoundError(
                "Cannot accumulate CV; validation folder(s) missing: "
                + ", ".join(str(path) for path in missing)
            )
        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError(
                f"Refusing to overwrite non-empty accumulated CV output: {output_dir}"
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        result = self._execute(
            build_accumulate_cross_validation(
                self.dataset_id,
                self.configuration,
                output_dir,
                trainer="nnUNetTrainer",
                plans=self.plans_name,
            ),
            experiment_id=identifier,
            fold=0,
            trainer="nnUNetTrainer",
            monitor_gpu=False,
        )
        self._update_manifest(identifier, {"crossval_predictions": str(output_dir.resolve())})
        return result

    def predict(
        self,
        *,
        input_dir: Path,
        output_dir: Path,
        folds: Iterable[int] = (0,),
        trainer: str,
        experiment_id: str | None = None,
        disable_tta: bool = True,
        save_probabilities: bool = False,
        continue_prediction: bool = False,
    ) -> LiveCommandResult:
        normalized_folds = self.validate_folds(tuple(folds))
        identifier = self.initialize_experiment(
            experiment_id,
            kind="inference",
            fold=normalized_folds[0],
            trainer=trainer,
        )
        if not input_dir.is_dir():
            raise FileNotFoundError(f"Inference input directory does not exist: {input_dir}")
        model_folder = self._model_output_folder(trainer)
        missing_checkpoints = [
            model_folder / f"fold_{fold}" / "checkpoint_final.pth"
            for fold in normalized_folds
            if not (model_folder / f"fold_{fold}" / "checkpoint_final.pth").is_file()
        ]
        if missing_checkpoints:
            raise FileNotFoundError(
                "Required checkpoint(s) missing: "
                + ", ".join(str(path) for path in missing_checkpoints)
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        result = self._execute(
            build_predict(
                self.dataset_id,
                self.configuration,
                normalized_folds,
                input_dir,
                output_dir,
                trainer=trainer,
                plans=self.plans_name,
                disable_tta=disable_tta,
                save_probabilities=save_probabilities,
                continue_prediction=continue_prediction,
            ),
            experiment_id=identifier,
            fold=normalized_folds[0],
            trainer=trainer,
            monitor_gpu=True,
        )
        prediction_count = len(list(output_dir.glob("*.nii.gz")))
        inference_summary = {
            "folds": list(normalized_folds),
            "tta_state": "OFF" if disable_tta else "DEFAULT MIRRORING",
            "total_seconds": result.elapsed_seconds,
            "number_of_cases": prediction_count,
            "mean_seconds_per_case": (
                result.elapsed_seconds / prediction_count if prediction_count else None
            ),
            "output_dir": str(output_dir.resolve()),
        }
        write_json_atomic(
            self._experiment_dir(identifier) / "inference_runtime.json",
            inference_summary,
        )
        self._update_manifest(
            identifier,
            {
                "TTA_state": inference_summary["tta_state"],
                "inference_seconds_per_case": inference_summary["mean_seconds_per_case"],
                "prediction_dir": str(output_dir.resolve()),
            },
        )
        return result

    def evaluate(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(
            "Evaluation is backend-neutral and belongs to glioma_seg.evaluation; "
            "official validation data without public labels must never be scored locally."
        )

    def get_artifacts(self, experiment_id: str) -> BackendArtifacts:
        experiment_dir = self._experiment_dir(experiment_id)
        manifest_path = experiment_dir / "experiment.json"
        manifest = load_json(manifest_path)
        return BackendArtifacts(
            experiment_id=experiment_id,
            checkpoint_paths=tuple(Path(path) for path in manifest.get("checkpoint_paths", [])),
            prediction_dir=(
                Path(manifest["prediction_dir"]) if manifest.get("prediction_dir") else None
            ),
            log_paths=tuple(sorted((experiment_dir / "logs").glob("*.log"))),
            metadata=manifest,
        )

    def record_artifacts(
        self,
        experiment_id: str,
        artifacts: Mapping[str, Path],
    ) -> None:
        """Attach verified project artifacts to the reproducibility manifest."""

        manifest_path = self._experiment_dir(experiment_id) / "experiment.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Experiment manifest does not exist: {manifest_path}")
        manifest = load_json(manifest_path)
        recorded = dict(manifest.get("artifacts", {}))
        for name, path in artifacts.items():
            if not name.replace("_", "").isalnum():
                raise ValueError(f"Invalid artifact name: {name!r}")
            resolved = path.resolve()
            if not resolved.exists():
                raise FileNotFoundError(f"Artifact does not exist: {resolved}")
            recorded[name] = str(resolved)
        metric_files = [
            value for key, value in recorded.items() if "metric" in key or key == "official_status"
        ]
        self._update_manifest(
            experiment_id,
            {"artifacts": recorded, "metrics_files": metric_files},
        )

    def _assert_preprocessing_available(self, experiment_id: str) -> None:
        report = self._preprocessing_artifacts(ensure_splits=True)
        report.require_valid()
        conversion_ok, conversion_detail, _ = self._validate_conversion_manifest(experiment_id)
        if not conversion_ok:
            message = "Conversion provenance validation failed before GPU work: "
            raise RuntimeError(message + conversion_detail)

    def check_readiness(
        self,
        *,
        experiment_id: str,
        fold: int,
        trainer: str,
        continue_training: bool = False,
        allow_low_gpu_utilization: bool = False,
    ) -> ReadinessReport:
        report = ReadinessReport(
            experiment_id=experiment_id,
            dataset=self.dataset_name,
            dataset_id=self.dataset_id,
            fold=fold,
            configuration=self.configuration,
            trainer=trainer,
        )
        if fold not in range(5):
            report.add("fold", False, f"Invalid fold: {fold}")
        else:
            report.add("fold", True, str(fold))

        distribution = _distribution_metadata()
        source = str(distribution.get("source") or "")
        expected_source_token = str(self.paths.external_nnunet).replace("\\", "/").casefold()
        source_matches = expected_source_token in source.replace("\\", "/").casefold()
        report.add(
            "official editable nnU-Net",
            bool(distribution.get("installed"))
            and bool(distribution.get("editable"))
            and source_matches,
            json.dumps(distribution, ensure_ascii=False),
        )
        report.add(
            "nnU-Net version",
            distribution.get("version") == EXPECTED_UPSTREAM_VERSION,
            f"installed={distribution.get('version')} expected={EXPECTED_UPSTREAM_VERSION}",
        )

        upstream_commit = _run_metadata_command(
            ["git", "rev-parse", "HEAD"], cwd=self.paths.external_nnunet
        )
        upstream_status = _run_metadata_command(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=self.paths.external_nnunet,
        )
        report.add(
            "upstream pinned commit",
            upstream_commit == EXPECTED_UPSTREAM_COMMIT,
            f"actual={upstream_commit} expected={EXPECTED_UPSTREAM_COMMIT}",
        )
        report.add(
            "upstream source unmodified",
            upstream_status == "",
            "clean" if upstream_status == "" else (upstream_status or "repository unavailable"),
        )

        for executable in ("nnUNetv2_train", "nnUNetv2_predict", "nnUNetv2_plan_and_preprocess"):
            try:
                resolved = self._resolve_cli(executable)
                report.add(executable, True, str(resolved))
            except FileNotFoundError as exc:
                report.add(executable, False, str(exc))

        try:
            import torch

            torch_version = str(torch.__version__)
            report.add("PyTorch installed", True, torch_version)
            report.add(
                "PyTorch 2.9 safety guard",
                not torch_version.startswith("2.9"),
                torch_version,
            )
            report.add("CUDA available", torch.cuda.is_available(), str(torch.cuda.is_available()))
            report.details.update(
                {
                    "torch": torch_version,
                    "cuda": torch.version.cuda,
                    "cudnn": _cudnn_version(torch),
                }
            )
        except Exception as exc:
            report.add("PyTorch/CUDA", False, f"{type(exc).__name__}: {exc}")

        try:
            gpu = query_gpu_once()
            report.add(
                "dedicated GPU memory",
                gpu.memory_total_mb >= 10_000,
                f"{gpu.gpu_name}: {gpu.memory_total_mb:.0f} MiB dedicated",
            )
            report.details["gpu"] = asdict(gpu)
        except Exception as exc:
            report.add("GPU telemetry", False, f"{type(exc).__name__}: {exc}")

        dataset_json_path = self.dataset_raw_dir / "dataset.json"
        if dataset_json_path.is_file():
            dataset_json = load_json(dataset_json_path)
            count = dataset_json.get("numTraining")
            report.add("dataset.json", True, str(dataset_json_path))
            report.add(
                "training case count",
                count == 1251,
                f"actual={count}, expected=1251",
            )
            expected_channels = {"0": "T1n", "1": "T1c", "2": "T2w", "3": "T2F"}
            channels = dataset_json.get("channel_names")
            channel_ok = isinstance(channels, Mapping) and {
                str(key): str(value).casefold() for key, value in channels.items()
            } == {key: value.casefold() for key, value in expected_channels.items()}
            report.add("channel mapping", channel_ok, json.dumps(channels, ensure_ascii=False))
            labels = dataset_json.get("labels")
            expected_labels = {
                "background": 0,
                "whole_tumor": [1, 2, 3],
                "tumor_core": [1, 3],
                "enhancing_tumor": 3,
            }
            report.add("BraTS 2023 regions", labels == expected_labels, json.dumps(labels))
            region_order = list(labels) if isinstance(labels, Mapping) else []
            report.add(
                "region dictionary order",
                region_order == list(expected_labels),
                str(region_order),
            )
            class_order = dataset_json.get("regions_class_order")
            report.add("regions_class_order", class_order == [2, 1, 3], str(class_order))
            images = list((self.dataset_raw_dir / "imagesTr").glob("*_0000.nii.gz"))
            labels_tr = list((self.dataset_raw_dir / "labelsTr").glob("*.nii.gz"))
            counts_match = (
                isinstance(count, int) and len(images) == count and len(labels_tr) == count
            )
            report.add(
                "converted files match manifest",
                counts_match,
                f"dataset.json={count}, imagesTr cases={len(images)}, labelsTr={len(labels_tr)}",
            )
            report.details["dataset_case_count"] = count
        else:
            report.add("dataset.json", False, f"Missing: {dataset_json_path}")

        validation_path = self.paths.reports / "data_validation.json"
        if validation_path.is_file():
            try:
                validation = load_json(validation_path)
                validated_ids = {
                    str(case.get("case_id"))
                    for case in validation.get("cases", [])
                    if isinstance(case, Mapping)
                }
                converted_ids = {
                    path.name.removesuffix("_0000.nii.gz")
                    for path in (self.dataset_raw_dir / "imagesTr").glob("*_0000.nii.gz")
                }
                validation_ok = (
                    validation.get("valid") is True
                    and validation.get("dataset_kind") == "training"
                    and validation.get("actual_case_count") == 1251
                    and validated_ids == converted_ids
                )
                report.add(
                    "read-only source validation manifest",
                    validation_ok,
                    (
                        f"valid={validation.get('valid')}, "
                        f"validated={len(validated_ids)}, converted={len(converted_ids)}"
                    ),
                )
            except (OSError, ValueError, TypeError) as exc:
                report.add("read-only source validation manifest", False, str(exc))
        else:
            report.add(
                "read-only source validation manifest",
                False,
                f"Missing: {validation_path}",
            )

        try:
            conversion_ok, conversion_detail, conversion_summary = (
                self._validate_conversion_manifest(experiment_id)
            )
            report.add("nnU-Net conversion provenance", conversion_ok, conversion_detail)
            report.details["conversion_manifest"] = conversion_summary
        except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
            report.add("nnU-Net conversion provenance", False, str(exc))

        plans_path = self.dataset_preprocessed_dir / "nnUNetPlans.json"
        fingerprint_path = self.dataset_preprocessed_dir / "dataset_fingerprint.json"
        if plans_path.is_file():
            try:
                plans_summary = summarize_plans(plans_path, self.configuration)
                report.add("nnU-Net plans", True, str(plans_path))
                report.details.update(plans_summary)
            except (KeyError, ValueError) as exc:
                report.add("nnU-Net plans", False, str(exc))
        else:
            report.add("nnU-Net plans", False, f"Missing: {plans_path}")
        report.add(
            "dataset fingerprint",
            fingerprint_path.is_file(),
            str(fingerprint_path),
        )
        try:
            artifact_report = self._preprocessing_artifacts(ensure_splits=True)
            failed_artifact_checks = [
                check.name for check in artifact_report.checks if not check.ok
            ]
            report.add(
                "complete official preprocessing artifacts",
                artifact_report.valid,
                (
                    f"valid={artifact_report.valid}, "
                    f"failed_checks={failed_artifact_checks or 'none'}, "
                    f"directory={self.dataset_preprocessed_dir}"
                ),
            )
            report.details["preprocessing_artifacts"] = artifact_report.to_dict()
        except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
            report.add("complete official preprocessing artifacts", False, str(exc))

        converted_case_ids = {
            path.name.removesuffix("_0000.nii.gz")
            for path in (self.dataset_raw_dir / "imagesTr").glob("*_0000.nii.gz")
        }
        splits_path = self.dataset_preprocessed_dir / "splits_final.json"
        if splits_path.is_file():
            try:
                splits = json.loads(splits_path.read_text(encoding="utf-8"))
                fold_entry = (
                    splits[fold]
                    if isinstance(splits, list) and fold in range(len(splits))
                    else None
                )
                if isinstance(fold_entry, Mapping):
                    train_cases = set(fold_entry.get("train", []))
                    val_cases = set(fold_entry.get("val", []))
                    split_ok = (
                        bool(train_cases)
                        and bool(val_cases)
                        and train_cases.isdisjoint(val_cases)
                        and train_cases | val_cases == converted_case_ids
                    )
                    report.details["fold_train_cases"] = len(train_cases)
                    report.details["fold_val_cases"] = len(val_cases)
                    detail = (
                        f"train={len(train_cases)}, val={len(val_cases)}, "
                        f"disjoint={train_cases.isdisjoint(val_cases)}, "
                        f"complete_partition={train_cases | val_cases == converted_case_ids}"
                    )
                else:
                    detail = f"Fold {fold} missing or invalid"
                report.add("case-level fold split", split_ok, detail)
            except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
                report.add("case-level fold split", False, str(exc))
        else:
            report.add("case-level fold split", False, f"Missing: {splits_path}")

        output_fold = self._model_output_folder(trainer) / f"fold_{fold}"
        checkpoints = list(output_fold.glob("checkpoint_*.pth")) if output_fold.exists() else []
        output_entries = list(output_fold.iterdir()) if output_fold.is_dir() else []
        if continue_training:
            final_checkpoint = output_fold / "checkpoint_final.pth"
            resume_checkpoint = output_fold / "checkpoint_latest.pth"
            best_checkpoint = output_fold / "checkpoint_best.pth"
            selected_checkpoint = next(
                (
                    path
                    for path in (final_checkpoint, resume_checkpoint, best_checkpoint)
                    if path.is_file()
                ),
                None,
            )
            report.add(
                "resume checkpoint",
                selected_checkpoint is not None,
                (
                    f"Selected by official nnU-Net continuation order: {selected_checkpoint}"
                    if selected_checkpoint
                    else (
                        "No checkpoint_final.pth, checkpoint_latest.pth, or "
                        f"checkpoint_best.pth in {output_fold}"
                    )
                ),
            )
            report.details["resume_checkpoint"] = (
                str(selected_checkpoint) if selected_checkpoint else None
            )
            owner_path = self._model_owner_path(trainer, fold)
            try:
                actual_owner = load_json(owner_path)
                expected_owner = self._model_owner(experiment_id, trainer, fold)
                report.add(
                    "resume experiment ownership",
                    actual_owner == expected_owner,
                    f"owner={owner_path}, experiment={actual_owner.get('experiment_id')}",
                )
            except (OSError, ValueError, TypeError) as exc:
                report.add("resume experiment ownership", False, str(exc))
        else:
            report.add(
                "no implicit overwrite",
                not output_entries,
                (
                    f"Output directory is absent or empty: {output_fold}"
                    if not output_entries
                    else (
                        "Existing trainer output requires explicit resume or archival: "
                        f"{len(output_entries)} entries, {len(checkpoints)} checkpoints"
                    )
                ),
            )

        workers = recommend_augmentation_workers()
        report.details["nnUNet_n_proc_DA"] = workers
        report.details["output"] = str(output_fold)
        report.details["regions"] = {"WT": [1, 2, 3], "TC": [1, 3], "ET": [3]}

        if trainer in PRELIMINARY_TRAINERS:
            benchmark_path = self._experiment_dir(experiment_id) / "benchmark_summary.json"
            gpu_summary_path = self._experiment_dir(experiment_id) / "benchmark_gpu_summary.json"
            if benchmark_path.is_file():
                benchmark = load_json(benchmark_path)
                recommended = benchmark.get("recommended_preliminary_trainer")
                report.add(
                    "benchmark trainer selection",
                    recommended == trainer
                    and benchmark.get("experiment_id") == experiment_id
                    and benchmark.get("dataset_id") == self.dataset_id
                    and benchmark.get("configuration") == self.configuration
                    and benchmark.get("fold") == 0,
                    (
                        f"recommended={recommended}, requested={trainer}, "
                        f"experiment={benchmark.get('experiment_id')}, "
                        f"dataset={benchmark.get('dataset_id')}, "
                        f"configuration={benchmark.get('configuration')}, "
                        f"fold={benchmark.get('fold')}"
                    ),
                )
                estimates = benchmark.get("linear_runtime_estimates_seconds", {})
                epochs = PRELIMINARY_TRAINERS[trainer]
                report.details["estimated_runtime_seconds"] = estimates.get(str(epochs))
                report.details["runtime_estimate_note"] = benchmark.get("estimate_note")
            else:
                report.add("official five-epoch benchmark", False, f"Missing: {benchmark_path}")
            if gpu_summary_path.is_file():
                benchmark_gpu = load_json(gpu_summary_path)
                samples = benchmark_gpu.get("samples")
                mean_utilization = benchmark_gpu.get("mean_gpu_utilization_percent")
                telemetry_measured = (
                    isinstance(samples, int)
                    and samples > 0
                    and isinstance(mean_utilization, (int, float))
                )
                utilization_ok = bool(
                    telemetry_measured
                    and isinstance(mean_utilization, (int, float))
                    and (mean_utilization >= 20.0 or allow_low_gpu_utilization)
                )
                report.add(
                    "benchmark GPU telemetry",
                    utilization_ok,
                    (
                        f"samples={samples}, mean_utilization={mean_utilization}%, "
                        f"reviewed_override={allow_low_gpu_utilization}"
                    ),
                )
                report.details["low_gpu_utilization_reviewed_override"] = allow_low_gpu_utilization
            else:
                report.add("benchmark GPU telemetry", False, f"Missing: {gpu_summary_path}")
        return report

    @staticmethod
    def print_readiness(report: ReadinessReport) -> None:
        details = report.details
        print("=" * 40)
        print("READY TO TRAIN" if report.ready else "NOT READY TO TRAIN")
        print("=" * 40)
        print(f"Experiment: {report.experiment_id}")
        print(f"Dataset: {report.dataset}")
        print(f"Validated training cases: {details.get('dataset_case_count', 'NOT AVAILABLE')}")
        print(f"Fold: {report.fold}")
        print(f"Fold train cases: {details.get('fold_train_cases', 'NOT AVAILABLE')}")
        print(f"Fold val cases: {details.get('fold_val_cases', 'NOT AVAILABLE')}")
        print("Channels: 0000 T1n | 0001 T1c | 0002 T2w | 0003 T2F")
        print("Regions: WT [1,2,3] | TC [1,3] | ET [3]")
        print(f"Configuration: {report.configuration}")
        print(f"Architecture: {details.get('architecture_class', 'NOT AVAILABLE')}")
        print(f"Patch: {details.get('patch_size', 'NOT AVAILABLE')}")
        print(f"Batch: {details.get('batch_size', 'NOT AVAILABLE')}")
        print(f"Target spacing: {details.get('target_spacing', 'NOT AVAILABLE')}")
        print(f"Trainer: {report.trainer}")
        gpu = details.get("gpu", {})
        print(f"GPU: {gpu.get('gpu_name', 'NOT AVAILABLE')}")
        print(f"VRAM: {gpu.get('memory_total_mb', 'NOT AVAILABLE')} MiB dedicated")
        estimated_runtime = details.get("estimated_runtime_seconds")
        print(
            "Estimated runtime: "
            + (
                f"{estimated_runtime / 3600:.2f} h (linear rough estimate)"
                if isinstance(estimated_runtime, (int, float))
                else "NOT AVAILABLE"
            )
        )
        print(f"nnUNet_n_proc_DA: {details.get('nnUNet_n_proc_DA')}")
        print(f"Output: {details.get('output')}")
        failures = [check for check in report.checks if check.critical and not check.ok]
        if failures:
            print("Critical failures:")
            for check in failures:
                print(f"  - {check.name}: {check.detail}")
        print("=" * 40)


def _write_system_outputs(backend: NNUNetV2Backend, output: Path, lock_output: Path) -> None:
    report = collect_system_report(backend.paths)
    write_json_atomic(output, report)
    freeze = _run_metadata_command([sys.executable, "-m", "pip", "freeze"])
    if freeze is None:
        raise RuntimeError("Unable to record installed dependency versions with pip freeze")
    lock_output.parent.mkdir(parents=True, exist_ok=True)
    lock_output.write_text(freeze + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Environment report: {output}")
    print(f"Dependency lock: {lock_output}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--dataset-id", type=int, default=DATASET_ID)
    parser.add_argument("--dataset-name", default=DATASET_NAME)
    parser.add_argument("--configuration", default=CONFIGURATION)
    subparsers = parser.add_subparsers(dest="action", required=True)

    system = subparsers.add_parser("system-check")
    system.add_argument("--output", type=Path, required=True)
    system.add_argument("--lock-output", type=Path, required=True)

    new_experiment = subparsers.add_parser("new-experiment")
    new_experiment.add_argument("--kind", default="prelim")
    new_experiment.add_argument("--fold", type=int, default=0)

    preprocess = subparsers.add_parser("preprocess")
    preprocess.add_argument("--experiment-id", default=None)

    benchmark = subparsers.add_parser("benchmark")
    benchmark.add_argument("--experiment-id", default=None)

    readiness = subparsers.add_parser("readiness")
    readiness.add_argument("--experiment-id", required=True)
    readiness.add_argument("--fold", type=int, default=0)
    readiness.add_argument("--trainer", required=True)
    readiness.add_argument("--continue", dest="continue_training", action="store_true")
    readiness.add_argument("--allow-low-gpu-utilization", action="store_true")
    readiness.add_argument("--output", type=Path, default=None)

    train = subparsers.add_parser("train")
    train.add_argument("--experiment-id", default=None)
    train.add_argument("--fold", type=int, default=0)
    train.add_argument("--trainer", required=True)
    train.add_argument("--continue", dest="continue_training", action="store_true")
    train.add_argument("--allow-low-gpu-utilization", action="store_true")
    train.add_argument("--config", type=Path, default=None)

    predict = subparsers.add_parser("predict")
    predict.add_argument("--experiment-id", default=None)
    predict.add_argument("--input-dir", type=Path, required=True)
    predict.add_argument("--output-dir", type=Path, required=True)
    predict.add_argument("--fold", type=int, action="append", default=None)
    predict.add_argument("--trainer", required=True)
    predict.add_argument("--tta", action="store_true", help="Enable default mirroring TTA")
    predict.add_argument("--save-probabilities", action="store_true")
    predict.add_argument("--continue-prediction", action="store_true")

    accumulate = subparsers.add_parser("accumulate-crossval")
    accumulate.add_argument("--experiment-id", default=None)
    accumulate.add_argument("--output-dir", type=Path, required=True)

    record = subparsers.add_parser("record-artifacts")
    record.add_argument("--experiment-id", required=True)
    record.add_argument(
        "--artifact",
        action="append",
        required=True,
        help="Existing artifact as NAME=PATH; repeat for multiple files/directories",
    )

    reconcile = subparsers.add_parser("reconcile-telemetry")
    reconcile.add_argument("--experiment-id", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    backend = NNUNetV2Backend(
        project_root=args.project_root,
        dataset_id=args.dataset_id,
        dataset_name=args.dataset_name,
        configuration=args.configuration,
    )
    if args.action == "system-check":
        _write_system_outputs(backend, args.output.resolve(), args.lock_output.resolve())
    elif args.action == "new-experiment":
        identifier = backend.initialize_experiment(None, kind=args.kind, fold=args.fold)
        print(identifier)
    elif args.action == "preprocess":
        backend.preprocess(experiment_id=args.experiment_id)
    elif args.action == "benchmark":
        summary = backend.benchmark(experiment_id=args.experiment_id)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    elif args.action == "readiness":
        report = backend.check_readiness(
            experiment_id=args.experiment_id,
            fold=args.fold,
            trainer=args.trainer,
            continue_training=args.continue_training,
            allow_low_gpu_utilization=args.allow_low_gpu_utilization,
        )
        backend.print_readiness(report)
        if args.output:
            write_json_atomic(args.output.resolve(), report.to_dict())
        return 0 if report.ready else 2
    elif args.action == "train":
        backend.train(
            fold=args.fold,
            trainer=args.trainer,
            experiment_id=args.experiment_id,
            continue_training=args.continue_training,
            config_path=args.config,
            allow_low_gpu_utilization=args.allow_low_gpu_utilization,
        )
    elif args.action == "predict":
        backend.predict(
            input_dir=args.input_dir.resolve(),
            output_dir=args.output_dir.resolve(),
            folds=args.fold or [0],
            trainer=args.trainer,
            experiment_id=args.experiment_id,
            disable_tta=not args.tta,
            save_probabilities=args.save_probabilities,
            continue_prediction=args.continue_prediction,
        )
    elif args.action == "accumulate-crossval":
        backend.accumulate_cross_validation(
            output_dir=args.output_dir.resolve(),
            experiment_id=args.experiment_id,
        )
    elif args.action == "record-artifacts":
        artifacts: dict[str, Path] = {}
        for item in args.artifact:
            if "=" not in item:
                raise ValueError(f"Artifact must use NAME=PATH syntax: {item!r}")
            name, raw_path = item.split("=", 1)
            artifacts[name] = Path(raw_path)
        backend.record_artifacts(args.experiment_id, artifacts)
    elif args.action == "reconcile-telemetry":
        updates = backend.reconcile_telemetry(args.experiment_id)
        print(json.dumps(updates, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
