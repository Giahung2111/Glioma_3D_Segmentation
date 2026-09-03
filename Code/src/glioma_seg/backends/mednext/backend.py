"""CLI and artifact lifecycle for the source-pinned official MedNeXt v1 baseline."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import importlib.metadata
import json
import os
import platform
import secrets
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

from glioma_seg.backends.base import BackendArtifacts, SegmentationBackend
from glioma_seg.backends.mednext.config import (
    EXPECTED_MEDNEXT_COMMIT,
    EXPECTED_PLANNER,
    MedNeXtRecipe,
    load_recipe,
)
from glioma_seg.backends.mednext.dataset import (
    EXPECTED_CASE_COUNT,
    EXPECTED_FOLD_VALIDATION_COUNTS,
    inventory_dataset,
    load_canonical_splits,
    prepare_v1_adapter,
)
from glioma_seg.backends.mednext.trainer import (
    MedNeXtFoldSpec,
    audit_preprocessed_task,
    memory_preflight,
    normalized_spec_hash,
    run_fold,
    write_fold_manifest,
)
from glioma_seg.ensembles.canonical_probabilities import (
    MEDNEXT_MULTICLASS_CONVERSION,
    MEDNEXT_MULTICLASS_ORDER,
    validate_canonical_probability_npz,
)
from glioma_seg.monitoring.gpu_monitor import query_gpu_once
from glioma_seg.monitoring.timing import write_json_atomic
from glioma_seg.utils.hashing import sha256_file

DATASET_NAME = "Dataset501_BraTS2023GLI"
FOLDS = (0, 1, 2, 3, 4)
MODEL_ID = "mednext_v1_s_kernel3"


def _project_root(value: Path | None = None) -> Path:
    if value is not None:
        root = value.resolve()
        return root.parent if root.name.casefold() == "code" else root
    return Path(__file__).resolve().parents[5]


def _paths(root: Path) -> dict[str, Path]:
    workspace = root / "Workspace"
    mednext_workspace = workspace / "mednext_v1"
    return {
        "root": root,
        "code": root / "Code",
        "workspace": workspace,
        "raw_dataset": workspace / "nnUNet_raw" / DATASET_NAME,
        "split": workspace / "nnUNet_preprocessed" / DATASET_NAME / "splits_final.json",
        "reports": workspace / "reports",
        "results": workspace / "model_results" / "mednext",
        "mednext_workspace": mednext_workspace,
        "raw_base": mednext_workspace / "raw_base",
        "preprocessed": mednext_workspace / "preprocessed",
        "runtime_results": mednext_workspace / "runtime_results",
        "upstream": root / "External" / "MedNeXt",
        "model_config": root / "Code" / "configs" / "models" / "mednext.yaml",
        "experiment_config": root
        / "Code"
        / "configs"
        / "experiments"
        / "mednext_100epoch_cv.yaml",
    }


def configure_mednext_environment(paths: Mapping[str, Path]) -> dict[str, str]:
    """Set the three official v1 variables before importing any upstream module."""

    environment = {
        "nnUNet_raw_data_base": str(paths["raw_base"].resolve()),
        "nnUNet_preprocessed": str(paths["preprocessed"].resolve()),
        "RESULTS_FOLDER": str(paths["runtime_results"].resolve()),
        "nnunet_use_progress_bar": "0",
    }
    for name, value in environment.items():
        os.environ[name] = value
    for directory in (paths["raw_base"], paths["preprocessed"], paths["runtime_results"]):
        directory.mkdir(parents=True, exist_ok=True)
    return environment


def _git_output(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        timeout=30,
    )
    return completed.stdout.strip()


def _check_upstream(repository: Path) -> dict[str, Any]:
    if not (repository / ".git").is_dir():
        raise FileNotFoundError(f"Official MedNeXt repository is missing: {repository}")
    commit = _git_output(repository, "rev-parse", "HEAD")
    if commit != EXPECTED_MEDNEXT_COMMIT:
        raise RuntimeError(
            f"MedNeXt commit mismatch: expected={EXPECTED_MEDNEXT_COMMIT}, actual={commit}"
        )
    status = _git_output(repository, "status", "--porcelain", "--untracked-files=no")
    if status:
        raise RuntimeError("Official MedNeXt source contains tracked modifications")
    return {
        "name": "MedNeXt",
        "repository": "https://github.com/MIC-DKFZ/MedNeXt.git",
        "path": str(repository.resolve()),
        "commit": commit,
        "tracked_clean": True,
    }


def _copy_config_snapshot(paths: Mapping[str, Path], report_dir: Path) -> None:
    destination = report_dir / "config_snapshot"
    destination.mkdir(parents=True, exist_ok=True)
    sources = {
        "project_model_config.yaml": paths["model_config"],
        "project_experiment_config.yaml": paths["experiment_config"],
        "official_trainer.py": paths["upstream"]
        / "nnunet_mednext"
        / "training"
        / "network_training"
        / "MedNeXt"
        / "nnUNetTrainerV2_MedNeXt.py",
        "official_planner.py": paths["upstream"]
        / "nnunet_mednext"
        / "experiment_planning"
        / "alternative_experiment_planning"
        / "target_spacing"
        / "experiment_planner_v21_isotropic1mm.py",
        "official_readme.md": paths["upstream"] / "README.md",
    }
    for name, source in sources.items():
        if not source.is_file():
            raise FileNotFoundError(f"MedNeXt provenance source is missing: {source}")
        target = destination / name
        if target.exists() and sha256_file(target) != sha256_file(source):
            raise FileExistsError(f"Conflicting config snapshot exists: {target}")
        if not target.exists():
            shutil.copy2(source, target)


def new_experiment_id(kind: str) -> str:
    if kind not in {"fullcv", "smoke"}:
        raise ValueError("kind must be fullcv or smoke")
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return f"mednext_s_k3_{kind}_{timestamp}_{secrets.token_hex(3)}"


def initialize_experiment(root: Path, experiment_id: str, *, kind: str) -> dict[str, Any]:
    if kind not in {"fullcv", "smoke"}:
        raise ValueError("kind must be fullcv or smoke")
    paths = _paths(root)
    recipe = load_recipe(paths["model_config"])
    cases = inventory_dataset(paths["raw_dataset"])
    splits, split_sha = load_canonical_splits(paths["split"], set(cases))
    report_dir = paths["reports"] / experiment_id
    result_dir = paths["results"] / experiment_id
    manifest_path = report_dir / "experiment.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("experiment_id") != experiment_id or existing.get(
            "backend"
        ) != "mednext":
            raise ValueError(f"Experiment manifest ownership mismatch: {manifest_path}")
        if existing.get("experiment_kind") != kind:
            raise ValueError(f"Experiment kind mismatch: {manifest_path}")
        return existing
    if report_dir.exists() and any(report_dir.iterdir()):
        raise FileExistsError(f"Non-empty report directory has no owned manifest: {report_dir}")
    report_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    _copy_config_snapshot(paths, report_dir)
    manifest: dict[str, Any] = {
        "schema": "glioma_model_experiment_v1",
        "experiment_id": experiment_id,
        "experiment_kind": kind,
        "backend": "mednext",
        "framework": "official MedNeXt v1 nnU-Net v1 training pipeline",
        "model": "MedNeXt-S (3x3x3)",
        "model_id": recipe.model_id,
        "classification": (
            "100-epoch compute-limited comparison using the unchanged official "
            "MedNeXt-S-k3 architecture and recipe"
        ),
        "dataset": DATASET_NAME,
        "dataset_name": DATASET_NAME,
        "dataset_id": 501,
        "dataset_case_count": EXPECTED_CASE_COUNT,
        "folds": list(FOLDS),
        "fold_validation_counts": list(EXPECTED_FOLD_VALIDATION_COUNTS),
        "epochs": 3 if kind == "smoke" else recipe.epochs,
        "epochs_per_fold": 3 if kind == "smoke" else recipe.epochs,
        "original_recipe_epochs": recipe.original_recipe_epochs,
        "configuration": "3d_fullres",
        "trainer": recipe.payload["framework"]["trainer"],
        "planner": recipe.payload["framework"]["planner_3d"],
        "plans_identifier": recipe.payload["framework"]["plans_identifier"],
        "patch_size": list(recipe.patch_size),
        "target_spacing": [1.0, 1.0, 1.0],
        "TTA_state": "OFF",
        "postprocessing_state": "OFF",
        "split": {
            "source": str(paths["split"].resolve()),
            "sha256": split_sha,
            "folds": [
                {
                    "fold": fold,
                    "train_cases": len(item["train"]),
                    "validation_cases": len(item["val"]),
                }
                for fold, item in enumerate(splits)
            ],
        },
        "upstream": {
            "repository": recipe.payload["upstream"]["repository"],
            "commit": EXPECTED_MEDNEXT_COMMIT,
            "package_version": "1.7.0",
            "source_modified": False,
        },
        "project_adapters": {
            "v2_to_v1_layout": "hardlink/copy with byte verification",
            "windows_compatibility": "path parsing only; no numerical/model code changed",
            "duration_override": "100 epochs instead of official trainer default 1000",
        },
        "probabilities": {
            "native_channel_order": list(MEDNEXT_MULTICLASS_ORDER),
            "activation": "softmax",
            "conversion": MEDNEXT_MULTICLASS_CONVERSION,
            "canonical_channel_order": ["ET", "TC", "WT"],
        },
        "model_config": str(recipe.source_path),
        "model_config_sha256": recipe.source_sha256,
        "report_dir": str(report_dir.resolve()),
        "result_dir": str(result_dir.resolve()),
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "ensemble": False,
        "external_data": False,
    }
    write_json_atomic(manifest_path, manifest)
    return manifest


def system_check(root: Path, experiment_id: str, *, output: Path) -> dict[str, Any]:
    paths = _paths(root)
    kind = "smoke" if "_smoke_" in experiment_id else "fullcv"
    manifest = initialize_experiment(root, experiment_id, kind=kind)
    configure_mednext_environment(paths)
    upstream = _check_upstream(paths["upstream"])
    import torch

    package_version = importlib.metadata.version("mednextv1")
    if package_version != "1.7.0":
        raise RuntimeError(f"Pinned mednextv1 1.7.0 is required, found {package_version}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the research-model environment")
    gpu = query_gpu_once()
    trainer_source = (
        paths["upstream"]
        / "nnunet_mednext"
        / "training"
        / "network_training"
        / "MedNeXt"
        / "nnUNetTrainerV2_MedNeXt.py"
    )
    report = {
        "valid": True,
        "experiment_id": experiment_id,
        "backend": "mednext",
        "python": sys.version,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_version": torch.__version__,
        "cuda": torch.version.cuda,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "mednextv1": package_version,
        "numpy": np.__version__,
        "nibabel": importlib.metadata.version("nibabel"),
        "gpu": gpu.gpu_name,
        "gpu_name": gpu.gpu_name,
        "gpu_vram_mb": gpu.memory_total_mb,
        "upstream": upstream,
        "official_trainer_source": str(trainer_source.resolve()),
        "official_trainer_source_sha256": sha256_file(trainer_source),
        "split_sha256": manifest["split"]["sha256"],
        "environment_variables": configure_mednext_environment(paths),
        "checked_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    write_json_atomic(output, report)
    return report


def prepare_dataset(
    root: Path,
    experiment_id: str,
    *,
    smoke: bool,
    quick_train_cases: int = 8,
    quick_validation_cases: int = 2,
) -> dict[str, Any]:
    paths = _paths(root)
    initialize_experiment(root, experiment_id, kind="smoke" if smoke else "fullcv")
    recipe = load_recipe(paths["model_config"])
    configure_mednext_environment(paths)
    _, provenance = prepare_v1_adapter(
        source_dataset=paths["raw_dataset"],
        source_split=paths["split"],
        raw_base=paths["raw_base"],
        preprocessed_root=paths["preprocessed"],
        full_task_name=recipe.task_full,
        smoke_task_name=recipe.task_smoke,
        smoke=smoke,
        quick_train_cases=quick_train_cases,
        quick_validation_cases=quick_validation_cases,
    )
    report_path = paths["reports"] / experiment_id / "mednext_v1_adapter.json"
    if report_path.exists():
        existing = json.loads(report_path.read_text(encoding="utf-8"))
        stable_keys = (
            "task_name",
            "smoke",
            "source_split_sha256",
            "case_count",
            "train_case_ids_by_fold",
            "validation_case_ids_by_fold",
        )
        if any(existing.get(key) != provenance.get(key) for key in stable_keys):
            raise ValueError(f"MedNeXt adapter provenance collision: {report_path}")
    write_json_atomic(report_path, provenance)
    return provenance


def preprocess_dataset(
    root: Path,
    experiment_id: str,
    *,
    smoke: bool,
    threads: int = 8,
    quick_train_cases: int = 8,
    quick_validation_cases: int = 2,
) -> dict[str, Any]:
    if threads < 1:
        raise ValueError("threads must be positive")
    paths = _paths(root)
    recipe = load_recipe(paths["model_config"])
    environment = configure_mednext_environment(paths)
    provenance = prepare_dataset(
        root,
        experiment_id,
        smoke=smoke,
        quick_train_cases=quick_train_cases,
        quick_validation_cases=quick_validation_cases,
    )
    task_name = recipe.task_smoke if smoke else recipe.task_full
    task_id = 951 if smoke else 501
    expected_ids = sorted(
        {
            str(value)
            for group in (
                provenance["train_case_ids_by_fold"]
                + provenance["validation_case_ids_by_fold"]
            )
            for value in group
        }
    )
    task_directory = paths["preprocessed"] / task_name
    try:
        audit = audit_preprocessed_task(task_directory, recipe, expected_case_ids=expected_ids)
        audit["reused"] = True
    except (FileNotFoundError, ValueError):
        command = [
            sys.executable,
            "-m",
            "glioma_seg.backends.mednext.compat",
            "preprocess",
            "--task-id",
            str(task_id),
            "--planner-3d",
            EXPECTED_PLANNER,
            "--threads",
            str(threads),
            "--verify-dataset-integrity",
        ]
        completed = subprocess.run(
            command,
            check=False,
            env={**os.environ, **environment},
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"Official MedNeXt preprocessing failed with exit code {completed.returncode}"
            ) from None
        audit = audit_preprocessed_task(task_directory, recipe, expected_case_ids=expected_ids)
        audit["reused"] = False
    audit.update(
        {
            "backend": "mednext",
            "task_name": task_name,
            "official_planner": EXPECTED_PLANNER,
            "source_split_sha256": provenance["source_split_sha256"],
            "windows_path_compatibility": "project-owned path parsing shim only",
        }
    )
    write_json_atomic(paths["reports"] / experiment_id / "preprocessing_artifacts.json", audit)
    return audit


def _fold_spec(
    root: Path,
    experiment_id: str,
    *,
    fold: int,
    smoke: bool,
    epochs: int,
    quick_train_cases: int,
    quick_validation_cases: int,
) -> tuple[MedNeXtRecipe, MedNeXtFoldSpec]:
    if fold not in FOLDS:
        raise ValueError("fold must be one of 0,1,2,3,4")
    if smoke and fold != 0:
        raise ValueError("The isolated MedNeXt smoke task contains only canonical source fold 0")
    paths = _paths(root)
    configure_mednext_environment(paths)
    recipe = load_recipe(paths["model_config"])
    labels = inventory_dataset(paths["raw_dataset"])
    splits, split_sha = load_canonical_splits(paths["split"], set(labels))
    source = splits[fold]
    train_ids = source["train"]
    validation_ids = source["val"]
    task_name = recipe.task_full
    if smoke:
        if epochs != 3:
            raise ValueError("Mandatory MedNeXt smoke training must use exactly 3 epochs")
        train_ids = train_ids[:quick_train_cases]
        validation_ids = validation_ids[:quick_validation_cases]
        task_name = recipe.task_smoke
        native_fold = 0
    else:
        if epochs != recipe.epochs:
            raise ValueError(f"Full MedNeXt CV must use exactly {recipe.epochs} epochs")
        native_fold = fold
    task_dir = paths["preprocessed"] / task_name
    expected_task_ids = (
        tuple(dict.fromkeys((*train_ids, *validation_ids)))
        if smoke
        else tuple(sorted(labels))
    )
    audit_preprocessed_task(task_dir, recipe, expected_case_ids=expected_task_ids)
    digest = normalized_spec_hash(
        recipe,
        experiment_id=experiment_id,
        fold=native_fold,
        epochs=epochs,
        task_name=task_name,
        train_case_ids=train_ids,
        validation_case_ids=validation_ids,
        split_sha256=split_sha,
        smoke=smoke,
    )
    spec = MedNeXtFoldSpec(
        experiment_id=experiment_id,
        fold=native_fold,
        epochs=epochs,
        task_name=task_name,
        preprocessed_task_directory=task_dir,
        output_root=paths["results"] / experiment_id,
        train_case_ids=tuple(train_ids),
        validation_case_ids=tuple(validation_ids),
        reference_labels=labels,
        split_sha256=split_sha,
        config_sha256=digest,
        smoke=smoke,
    )
    return recipe, spec


def audit_fold(fold_dir: Path, *, raw_labels: Path | None = None) -> dict[str, Any]:
    manifest_path = fold_dir / "fold_manifest.json"
    owner_path = fold_dir / "checkpoint_owner.json"
    if not manifest_path.is_file() or not owner_path.is_file():
        return {
            "valid": False,
            "complete": False,
            "safe_to_resume": False,
            "reason": "missing fold manifest or checkpoint owner",
        }
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    owner = json.loads(owner_path.read_text(encoding="utf-8"))
    expected_owner = {
        "backend": "mednext",
        "model_id": manifest["model_id"],
        "experiment_id": manifest["experiment_id"],
        "fold": manifest["fold"],
        "target_epochs": manifest["target_epochs"],
        "config_sha256": manifest["config_sha256"],
        "split_sha256": manifest["split_sha256"],
    }
    if any(owner.get(key) != value for key, value in expected_owner.items()):
        return {
            "valid": False,
            "complete": False,
            "safe_to_resume": False,
            "reason": "checkpoint ownership mismatch",
        }
    final_checkpoint = fold_dir / "model_final_checkpoint.model"
    resume_checkpoint = fold_dir / "model_resume_checkpoint.model"
    latest_checkpoint = fold_dir / "model_latest.model"
    summary_path = fold_dir / "validation_summary.json"
    safe_to_resume = bool(resume_checkpoint.is_file() or latest_checkpoint.is_file())
    if not final_checkpoint.is_file() or not summary_path.is_file():
        return {
            "valid": False,
            "complete": False,
            "safe_to_resume": safe_to_resume or final_checkpoint.is_file(),
            "reason": "final checkpoint or validation summary missing",
        }
    history_path = fold_dir / "train_history.json"
    if not history_path.is_file():
        return {
            "valid": False,
            "complete": False,
            "safe_to_resume": False,
            "reason": "training history missing",
        }
    history = json.loads(history_path.read_text(encoding="utf-8"))
    if len(history) != int(manifest["target_epochs"]):
        return {
            "valid": False,
            "complete": False,
            "safe_to_resume": False,
            "reason": "checkpoint epoch/history count mismatch",
        }
    try:
        if [int(row.get("epoch", -1)) for row in history] != list(
            range(1, int(manifest["target_epochs"]) + 1)
        ):
            raise ValueError("training history epoch sequence is invalid")
        if any(not np.isfinite(float(row.get("train_loss", np.nan))) for row in history):
            raise ValueError("training history contains a non-finite train loss")
        validation_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if validation_summary.get("valid") is not True:
            raise ValueError("validation summary is not valid")
        if int(validation_summary.get("fold", -1)) != int(manifest["fold"]):
            raise ValueError("validation summary fold disagrees")
        if int(validation_summary.get("case_count", -1)) != len(
            manifest["validation_case_ids"]
        ):
            raise ValueError("validation summary case count disagrees")
        if validation_summary.get("native_channel_order") != list(
            MEDNEXT_MULTICLASS_ORDER
        ):
            raise ValueError("validation summary native channel order disagrees")
        if validation_summary.get("canonical_channel_order") != ["ET", "TC", "WT"]:
            raise ValueError("validation summary canonical channel order disagrees")
        if validation_summary.get("probability_conversion") != MEDNEXT_MULTICLASS_CONVERSION:
            raise ValueError("validation summary probability conversion disagrees")
        inference_total = float(validation_summary.get("inference_total_seconds", -1.0))
        if not np.isfinite(inference_total) or inference_total <= 0:
            raise ValueError("validation summary inference timing is invalid")
    except Exception as exc:
        return {
            "valid": False,
            "complete": False,
            "safe_to_resume": True,
            "reason": f"validation/training summary is invalid: {type(exc).__name__}: {exc}",
        }
    runtime_path = fold_dir / "runtime.json"
    gpu_summary_path = fold_dir / "gpu_summary.json"
    gpu_samples_path = fold_dir / "gpu_samples.csv"
    telemetry_segments = list(fold_dir.glob("gpu_samples_segment_*.csv"))
    supporting = (runtime_path, gpu_summary_path, gpu_samples_path)
    missing_supporting = [path.name for path in supporting if not path.is_file()]
    if missing_supporting:
        gpu_missing = any(
            name in {"gpu_summary.json", "gpu_samples.csv"}
            for name in missing_supporting
        )
        return {
            "valid": False,
            "complete": False,
            "safe_to_resume": not gpu_missing or bool(telemetry_segments),
            "reason": f"supporting fold evidence is missing: {missing_supporting}",
        }
    try:
        runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
        gpu_summary = json.loads(gpu_summary_path.read_text(encoding="utf-8"))
        target_epochs = int(manifest["target_epochs"])
        if int(runtime.get("number_of_epochs", -1)) != target_epochs:
            raise ValueError("runtime epoch count does not equal target epochs")
        if int(runtime.get("target_epochs", -1)) != target_epochs:
            raise ValueError("runtime target epoch count disagrees")
        if runtime.get("stopped_for_resume_test") is not False:
            raise ValueError("runtime still claims an interrupted resume-test leg")
        total_seconds = float(runtime.get("total_seconds", np.nan))
        average_seconds = float(runtime.get("average_seconds_per_epoch", np.nan))
        if (
            not np.isfinite(total_seconds)
            or total_seconds <= 0
            or not np.isfinite(average_seconds)
            or average_seconds <= 0
        ):
            raise ValueError("runtime timing is invalid")
        if int(gpu_summary.get("samples", 0)) < 1:
            raise ValueError("GPU summary contains no samples")
        if gpu_summary.get("includes_all_owner_matched_invocations") is not True:
            raise ValueError("GPU summary is not cumulative across resume invocations")
        expected_segment_count = 2 if bool(manifest.get("smoke")) else 1
        if int(gpu_summary.get("segments", 0)) < expected_segment_count:
            raise ValueError(
                f"GPU summary requires at least {expected_segment_count} telemetry segments"
            )
        recorded_segments = {str(value) for value in gpu_summary.get("segment_files", [])}
        actual_segments = {path.name for path in telemetry_segments}
        if recorded_segments != actual_segments:
            raise ValueError("GPU summary segment inventory disagrees")
        with gpu_samples_path.open("r", encoding="utf-8", newline="") as handle:
            sample_rows = list(csv.DictReader(handle))
        if len(sample_rows) != int(gpu_summary["samples"]):
            raise ValueError("GPU samples row count disagrees with summary")
    except Exception as exc:
        gpu_problem = "GPU" in str(exc) or "gpu" in str(exc)
        return {
            "valid": False,
            "complete": False,
            "safe_to_resume": not gpu_problem or bool(telemetry_segments),
            "reason": f"supporting fold evidence is invalid: {type(exc).__name__}: {exc}",
        }
    import torch

    checkpoint = torch.load(final_checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("glioma_project_owner") != owner or checkpoint.get("epoch") != int(
        manifest["target_epochs"]
    ):
        return {
            "valid": False,
            "complete": False,
            "safe_to_resume": False,
            "reason": "embedded checkpoint owner or completed epoch mismatch",
        }
    expected_ids = set(str(value) for value in manifest["validation_case_ids"])
    predictions = fold_dir / "predictions"
    native = fold_dir / "probabilities" / "native"
    canonical = fold_dir / "probabilities" / "canonical"
    inventories = (
        {path.name.removesuffix(".nii.gz") for path in predictions.glob("*.nii.gz")},
        {path.stem for path in native.glob("*.npz")},
        {path.stem for path in canonical.glob("*.npz")},
    )
    if any(inventory != expected_ids for inventory in inventories):
        return {
            "valid": False,
            "complete": False,
            "safe_to_resume": False,
            "reason": "prediction/probability inventory mismatch",
        }
    labels_dir = raw_labels or _paths(_project_root())["raw_dataset"] / "labelsTr"
    for case_id in sorted(expected_ids):
        reference = labels_dir / f"{case_id}.nii.gz"
        ground_truth: Any = nib.load(str(reference))
        prediction: Any = nib.load(str(predictions / f"{case_id}.nii.gz"))
        if prediction.shape != ground_truth.shape or not np.allclose(
            prediction.affine, ground_truth.affine, rtol=0.0, atol=1e-4
        ):
            raise ValueError(f"MedNeXt prediction geometry mismatch for {case_id}")
        values = set(int(value) for value in np.unique(np.asanyarray(prediction.dataobj)))
        if not values.issubset({0, 1, 2, 3}):
            raise ValueError(f"MedNeXt prediction labels are invalid for {case_id}: {values}")
        with np.load(native / f"{case_id}.npz", allow_pickle=False) as archive:
            if set(archive.files) != {"softmax"}:
                raise ValueError(f"MedNeXt native NPZ keys are invalid for {case_id}")
            probabilities = np.asarray(archive["softmax"])
            spatial = tuple(int(value) for value in probabilities.shape[1:])
            valid_spatial = {
                tuple(int(value) for value in ground_truth.shape),
                tuple(reversed(tuple(int(value) for value in ground_truth.shape))),
            }
            if (
                probabilities.shape[0] != 4
                or spatial not in valid_spatial
                or probabilities.dtype != np.dtype(np.float16)
                or not np.all(np.isfinite(probabilities))
            ):
                raise ValueError(f"MedNeXt native probabilities are invalid for {case_id}")
            if not np.allclose(
                np.sum(probabilities.astype(np.float32), axis=0),
                1.0,
                rtol=0.0,
                atol=2e-3,
            ):
                raise ValueError(f"MedNeXt native softmax simplex is invalid for {case_id}")
        validate_canonical_probability_npz(
            canonical / f"{case_id}.npz",
            reference_nifti=reference,
            expected_case_id=case_id,
            expected_native_channel_order=MEDNEXT_MULTICLASS_ORDER,
            expected_conversion=MEDNEXT_MULTICLASS_CONVERSION,
        )
    return {
        "valid": True,
        "complete": True,
        "safe_to_resume": False,
        "fold": manifest["fold"],
        "experiment_id": manifest["experiment_id"],
        "expected_validation_cases": len(expected_ids),
        "validation_case_count": len(expected_ids),
        "checkpoint_final": str(final_checkpoint.resolve()),
        "prediction_dir": str(predictions.resolve()),
        "canonical_probability_dir": str(canonical.resolve()),
    }


def assemble_oof(root: Path, experiment_id: str, *, folds: Sequence[int]) -> dict[str, Any]:
    normalized_folds = SegmentationBackend.validate_folds(folds)
    paths = _paths(root)
    experiment_root = paths["results"] / experiment_id
    destination = experiment_root / "oof"
    case_to_fold: dict[str, int] = {}
    manifests: list[dict[str, Any]] = []
    for fold in normalized_folds:
        fold_dir = experiment_root / f"fold_{fold}"
        audit = audit_fold(fold_dir, raw_labels=paths["raw_dataset"] / "labelsTr")
        if not audit.get("valid") or not audit.get("complete"):
            raise RuntimeError(f"MedNeXt fold {fold} is incomplete: {audit}")
        manifest = json.loads((fold_dir / "fold_manifest.json").read_text(encoding="utf-8"))
        manifests.append(manifest)
        for case_id in manifest["validation_case_ids"]:
            if case_id in case_to_fold:
                raise ValueError(f"Case {case_id} is present in more than one OOF fold")
            case_to_fold[case_id] = fold
    if len(normalized_folds) == 5 and len(case_to_fold) != EXPECTED_CASE_COUNT:
        raise ValueError(f"Full OOF assembly expected 1251 cases, got {len(case_to_fold)}")
    if destination.exists():
        masks = {
            path.name.removesuffix(".nii.gz")
            for path in (destination / "predictions").glob("*.nii.gz")
        }
        probabilities = {
            path.stem
            for path in (destination / "probabilities" / "canonical").glob("*.npz")
        }
        if masks == set(case_to_fold) and probabilities == set(case_to_fold):
            return {
                "valid": True,
                "reused": True,
                "case_count": len(case_to_fold),
                "oof_dir": str(destination.resolve()),
            }
        raise FileExistsError(f"Conflicting/incomplete MedNeXt OOF directory: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".mednext_oof_stage_", dir=destination.parent))
    try:
        for name in ("predictions", "probabilities/native", "probabilities/canonical"):
            (staging / name).mkdir(parents=True, exist_ok=True)
        for manifest in manifests:
            fold_dir = Path(manifest["output_dir"])
            for case_id in manifest["validation_case_ids"]:
                for source, target in (
                    (
                        fold_dir / "predictions" / f"{case_id}.nii.gz",
                        staging / "predictions" / f"{case_id}.nii.gz",
                    ),
                    (
                        fold_dir / "probabilities" / "native" / f"{case_id}.npz",
                        staging / "probabilities" / "native" / f"{case_id}.npz",
                    ),
                    (
                        fold_dir / "probabilities" / "canonical" / f"{case_id}.npz",
                        staging / "probabilities" / "canonical" / f"{case_id}.npz",
                    ),
                ):
                    try:
                        os.link(source, target)
                    except OSError:
                        shutil.copy2(source, target)
        write_json_atomic(
            staging / "oof_manifest.json",
            {
                "valid": True,
                "backend": "mednext",
                "model_id": MODEL_ID,
                "experiment_id": experiment_id,
                "folds": list(normalized_folds),
                "case_count": len(case_to_fold),
                "each_case_once": True,
                "case_to_fold": case_to_fold,
                "native_channel_order": list(MEDNEXT_MULTICLASS_ORDER),
                "canonical_channel_order": ["ET", "TC", "WT"],
                "conversion": MEDNEXT_MULTICLASS_CONVERSION,
            },
        )
        os.replace(staging, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return {
        "valid": True,
        "reused": False,
        "case_count": len(case_to_fold),
        "oof_dir": str(destination.resolve()),
    }


def write_evaluation_manifest(root: Path, experiment_id: str, *, folds: Sequence[int]) -> Path:
    normalized_folds = SegmentationBackend.validate_folds(folds)
    paths = _paths(root)
    experiment_root = paths["results"] / experiment_id
    payload = {
        "schema": "glioma_model_crossval_artifacts_v1",
        "backend": "mednext",
        "model_id": MODEL_ID,
        "model_provenance": {
            "implementation_repository": "https://github.com/MIC-DKFZ/MedNeXt.git",
            "implementation_commit": EXPECTED_MEDNEXT_COMMIT,
            "official_trainer": "nnUNetTrainerV2_MedNeXt_S_kernel3",
            "official_plans_identifier": "nnUNetPlansv2.1_trgSp_1x1x1",
            "training_duration_classification": "compute-limited 100-epoch comparison",
        },
        "prediction_tta_state": "OFF",
        "folds": [
            {
                "fold": fold,
                "prediction_dir": str(
                    (experiment_root / f"fold_{fold}" / "predictions").resolve()
                ),
                "probability_dir": str(
                    (
                        experiment_root
                        / f"fold_{fold}"
                        / "probabilities"
                        / "canonical"
                    ).resolve()
                ),
            }
            for fold in normalized_folds
        ],
        "pooled_prediction_dir": str((experiment_root / "oof" / "predictions").resolve()),
        "pooled_probability_dir": str(
            (experiment_root / "oof" / "probabilities" / "canonical").resolve()
        ),
        "probability_contract": {
            "required": True,
            "native_channel_order": list(MEDNEXT_MULTICLASS_ORDER),
            "canonical_channel_order": ["ET", "TC", "WT"],
            "conversion": MEDNEXT_MULTICLASS_CONVERSION,
            "schema": "glioma_canonical_probabilities_v1",
        },
    }
    destination = paths["reports"] / experiment_id / "crossval_artifact_manifest.json"
    write_json_atomic(destination, payload)
    return destination


class MedNeXtBackend(SegmentationBackend):
    """Programmatic facade over the same project-owned MedNeXt lifecycle."""

    def __init__(self, project_root: Path | None = None) -> None:
        self.root = _project_root(project_root)

    def prepare_dataset(self, experiment_id: str, **kwargs: Any) -> dict[str, Any]:
        return prepare_dataset(self.root, experiment_id, **kwargs)

    def preprocess(self, experiment_id: str, **kwargs: Any) -> dict[str, Any]:
        return preprocess_dataset(self.root, experiment_id, **kwargs)

    def train(
        self,
        experiment_id: str,
        *,
        fold: int,
        smoke: bool,
        epochs: int,
        quick_train_cases: int = 8,
        quick_validation_cases: int = 2,
        resume: bool = False,
        stop_after_epoch: int | None = None,
    ) -> dict[str, Any]:
        recipe, spec = _fold_spec(
            self.root,
            experiment_id,
            fold=fold,
            smoke=smoke,
            epochs=epochs,
            quick_train_cases=quick_train_cases,
            quick_validation_cases=quick_validation_cases,
        )
        return run_fold(
            recipe,
            spec,
            resume=resume,
            stop_after_epoch=stop_after_epoch,
        )

    def predict(self, experiment_id: str, *, fold: int) -> dict[str, Any]:
        """Return the integrity audit for prediction integrated into ``train-fold``."""

        fold_dir = _paths(self.root)["results"] / experiment_id / f"fold_{fold}"
        return audit_fold(fold_dir)

    def evaluate(self, experiment_id: str, *, folds: Sequence[int]) -> dict[str, Any]:
        """Assemble verified OOF artifacts and publish the common evaluator manifest."""

        oof = assemble_oof(self.root, experiment_id, folds=folds)
        manifest = write_evaluation_manifest(self.root, experiment_id, folds=folds)
        return {**oof, "artifact_manifest": str(manifest.resolve())}

    def get_artifacts(self, experiment_id: str) -> BackendArtifacts:
        experiment = _paths(self.root)["results"] / experiment_id
        checkpoints = tuple(sorted(experiment.glob("fold_*/model_final_checkpoint.model")))
        return BackendArtifacts(
            experiment_id=experiment_id,
            checkpoint_paths=checkpoints,
            prediction_dir=experiment / "oof" / "predictions",
            probability_dir=experiment / "oof" / "probabilities" / "canonical",
            metadata={"backend": "mednext", "model_id": MODEL_ID},
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path)
    subparsers = parser.add_subparsers(dest="command", required=True)
    allocate = subparsers.add_parser("new-experiment")
    allocate.add_argument("--kind", choices=("fullcv", "smoke"), required=True)
    initialize = subparsers.add_parser("initialize")
    initialize.add_argument("--experiment-id", required=True)
    initialize.add_argument("--kind", choices=("fullcv", "smoke"), required=True)
    check = subparsers.add_parser("system-check")
    check.add_argument("--experiment-id", required=True)
    check.add_argument("--output", type=Path, required=True)
    prepare = subparsers.add_parser("prepare-dataset")
    prepare.add_argument("--experiment-id", required=True)
    prepare.add_argument("--smoke", action="store_true")
    prepare.add_argument("--quick-train-cases", type=int, default=8)
    prepare.add_argument("--quick-validation-cases", type=int, default=2)
    preprocess = subparsers.add_parser("preprocess")
    preprocess.add_argument("--experiment-id", required=True)
    preprocess.add_argument("--smoke", action="store_true")
    preprocess.add_argument("--threads", type=int, default=8)
    preprocess.add_argument("--quick-train-cases", type=int, default=8)
    preprocess.add_argument("--quick-validation-cases", type=int, default=2)
    memory = subparsers.add_parser("memory-preflight")
    memory.add_argument("--experiment-id", required=True)
    memory.add_argument("--output", type=Path, required=True)
    train = subparsers.add_parser("train-fold")
    train.add_argument("--experiment-id", required=True)
    train.add_argument("--fold", type=int, required=True)
    train.add_argument("--epochs", type=int, default=100)
    train.add_argument("--smoke", action="store_true")
    train.add_argument("--quick-train-cases", type=int, default=8)
    train.add_argument("--quick-validation-cases", type=int, default=2)
    train.add_argument("--resume", action="store_true")
    train.add_argument("--stop-after-epoch", type=int)
    audit = subparsers.add_parser("audit-fold")
    audit.add_argument("--fold-dir", type=Path, required=True)
    audit.add_argument("--output", type=Path)
    oof = subparsers.add_parser("assemble-oof")
    oof.add_argument("--experiment-id", required=True)
    oof.add_argument("--fold", action="append", type=int, required=True)
    evaluation = subparsers.add_parser("write-evaluation-manifest")
    evaluation.add_argument("--experiment-id", required=True)
    evaluation.add_argument("--fold", action="append", type=int, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    root = _project_root(args.project_root)
    paths = _paths(root)
    configure_mednext_environment(paths)
    if args.command == "new-experiment":
        print(new_experiment_id(args.kind))
        return 0
    if args.command == "initialize":
        result = initialize_experiment(root, args.experiment_id, kind=args.kind)
    elif args.command == "system-check":
        result = system_check(root, args.experiment_id, output=args.output)
    elif args.command == "prepare-dataset":
        result = prepare_dataset(
            root,
            args.experiment_id,
            smoke=args.smoke,
            quick_train_cases=args.quick_train_cases,
            quick_validation_cases=args.quick_validation_cases,
        )
    elif args.command == "preprocess":
        result = preprocess_dataset(
            root,
            args.experiment_id,
            smoke=args.smoke,
            threads=args.threads,
            quick_train_cases=args.quick_train_cases,
            quick_validation_cases=args.quick_validation_cases,
        )
    elif args.command == "memory-preflight":
        recipe, spec = _fold_spec(
            root,
            args.experiment_id,
            fold=0,
            smoke=True,
            epochs=3,
            quick_train_cases=8,
            quick_validation_cases=2,
        )
        write_fold_manifest(spec, recipe)
        result = memory_preflight(recipe, spec, output_json=args.output)
    elif args.command == "train-fold":
        recipe, spec = _fold_spec(
            root,
            args.experiment_id,
            fold=args.fold,
            smoke=args.smoke,
            epochs=args.epochs,
            quick_train_cases=args.quick_train_cases,
            quick_validation_cases=args.quick_validation_cases,
        )
        result = run_fold(
            recipe,
            spec,
            resume=args.resume,
            stop_after_epoch=args.stop_after_epoch,
        )
    elif args.command == "audit-fold":
        result = audit_fold(
            args.fold_dir.resolve(), raw_labels=paths["raw_dataset"] / "labelsTr"
        )
        if args.output:
            write_json_atomic(args.output, result)
    elif args.command == "assemble-oof":
        result = assemble_oof(root, args.experiment_id, folds=args.fold)
    elif args.command == "write-evaluation-manifest":
        manifest = write_evaluation_manifest(root, args.experiment_id, folds=args.fold)
        result = {"valid": True, "manifest": str(manifest.resolve())}
    else:  # pragma: no cover
        raise AssertionError(args.command)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result.get("valid", True) else 2


__all__ = [
    "DATASET_NAME",
    "FOLDS",
    "MODEL_ID",
    "MedNeXtBackend",
    "assemble_oof",
    "audit_fold",
    "build_arg_parser",
    "configure_mednext_environment",
    "initialize_experiment",
    "new_experiment_id",
    "prepare_dataset",
    "preprocess_dataset",
    "system_check",
    "write_evaluation_manifest",
]


if __name__ == "__main__":
    raise SystemExit(main())
