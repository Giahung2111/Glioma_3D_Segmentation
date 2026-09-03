"""CLI and artifact lifecycle for the pinned official MONAI SegResNet recipe."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import importlib
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
from typing import Any, cast

import nibabel as nib
import numpy as np

from glioma_seg.backends.segresnet.config import (
    EXPECTED_MODEL_ZOO_COMMIT,
    EXPECTED_MONAI_COMMIT,
    SegResNetRecipe,
    load_recipe,
)
from glioma_seg.backends.segresnet.trainer import (
    CasePaths,
    FoldRunSpec,
    config_hash,
    fold_manifest,
    memory_preflight,
    run_fold,
)
from glioma_seg.ensembles.canonical_probabilities import (
    SEGRESNET_REGION_CONVERSION,
    SEGRESNET_REGION_ORDER,
    validate_canonical_probability_npz,
)
from glioma_seg.monitoring.gpu_monitor import query_gpu_once
from glioma_seg.monitoring.timing import write_json_atomic
from glioma_seg.utils.hashing import sha256_file

DATASET_NAME = "Dataset501_BraTS2023GLI"
EXPECTED_CASE_COUNT = 1251
EXPECTED_FOLD_VALIDATION_COUNTS = (251, 250, 250, 250, 250)
FOLDS = (0, 1, 2, 3, 4)


def _project_root(value: Path | None = None) -> Path:
    if value is not None:
        root = value.resolve()
        return root.parent if root.name.casefold() == "code" else root
    return Path(__file__).resolve().parents[5]


def _paths(root: Path) -> dict[str, Path]:
    workspace = root / "Workspace"
    return {
        "root": root,
        "code": root / "Code",
        "workspace": workspace,
        "raw_dataset": workspace / "nnUNet_raw" / DATASET_NAME,
        "split": workspace / "nnUNet_preprocessed" / DATASET_NAME / "splits_final.json",
        "reports": workspace / "reports",
        "results": workspace / "model_results" / "segresnet",
        "mednext": root / "External" / "MedNeXt",
        "monai": root / "External" / "MONAI",
        "model_zoo": root / "External" / "MONAI-model-zoo",
        "official_metrics": root / "External" / "BraTS-2023-Metrics",
        "model_config": root / "Code" / "configs" / "models" / "segresnet_monai_bundle.yaml",
        "experiment_config": root
        / "Code"
        / "configs"
        / "experiments"
        / "segresnet_100epoch_cv.yaml",
    }


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


def _check_repository(repository: Path, expected_commit: str, name: str) -> dict[str, Any]:
    if not (repository / ".git").is_dir():
        raise FileNotFoundError(f"{name} repository is missing: {repository}")
    commit = _git_output(repository, "rev-parse", "HEAD")
    if commit != expected_commit:
        raise RuntimeError(f"{name} commit mismatch: expected={expected_commit}, actual={commit}")
    tracked_status = _git_output(repository, "status", "--porcelain", "--untracked-files=no")
    if tracked_status:
        raise RuntimeError(f"{name} contains tracked changes; upstream must remain unchanged")
    return {"name": name, "path": str(repository), "commit": commit, "tracked_clean": True}


def _load_splits(path: Path, case_ids: set[str]) -> tuple[list[dict[str, list[str]]], str]:
    if not path.is_file():
        raise FileNotFoundError(f"Canonical five-fold split is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or len(payload) != 5:
        raise ValueError("Canonical split must contain exactly five folds")
    normalized: list[dict[str, list[str]]] = []
    seen_validation: set[str] = set()
    for fold, expected_validation_count in enumerate(EXPECTED_FOLD_VALIDATION_COUNTS):
        item = payload[fold]
        if not isinstance(item, dict):
            raise ValueError(f"Fold {fold} must be a JSON object")
        train = [str(value) for value in item.get("train", [])]
        validation = [str(value) for value in item.get("val", [])]
        if len(train) != len(set(train)) or len(validation) != len(set(validation)):
            raise ValueError(f"Fold {fold} contains duplicate IDs")
        if set(train) & set(validation):
            raise ValueError(f"Fold {fold} has train/validation leakage")
        if set(train) | set(validation) != case_ids:
            raise ValueError(f"Fold {fold} does not partition all {len(case_ids)} cases")
        if len(validation) != expected_validation_count:
            raise ValueError(
                f"Fold {fold} validation count must be {expected_validation_count}, "
                f"got {len(validation)}"
            )
        overlap = seen_validation & set(validation)
        if overlap:
            raise ValueError(f"Cases appear in validation more than once: {sorted(overlap)[:5]}")
        seen_validation.update(validation)
        normalized.append({"train": train, "val": validation})
    if seen_validation != case_ids:
        raise ValueError("Five-fold validation union is not the complete cohort")
    return normalized, sha256_file(path)


def _inventory_cases(raw_dataset: Path) -> dict[str, CasePaths]:
    images = raw_dataset / "imagesTr"
    labels = raw_dataset / "labelsTr"
    if not images.is_dir() or not labels.is_dir():
        raise FileNotFoundError(f"Converted BraTS dataset is missing: {raw_dataset}")
    cases: dict[str, CasePaths] = {}
    for label in sorted(labels.glob("*.nii.gz")):
        case_id = label.name.removesuffix(".nii.gz")
        # Official MONAI bundle order is T1c,T1,T2,FLAIR.  Project suffixes
        # 0001,0000,0002,0003 are explicitly mapped and never alphabetized.
        modalities = tuple(
            images / f"{case_id}_{suffix}.nii.gz"
            for suffix in ("0001", "0000", "0002", "0003")
        )
        if any(not path.is_file() for path in modalities):
            raise FileNotFoundError(f"One or more modalities are missing for {case_id}")
        cases[case_id] = CasePaths(case_id, modalities, label.resolve())  # type: ignore[arg-type]
    if len(cases) != EXPECTED_CASE_COUNT:
        raise ValueError(f"Expected {EXPECTED_CASE_COUNT} BraTS cases, found {len(cases)}")
    return cases


def _copy_config_snapshot(paths: Mapping[str, Path], report_dir: Path) -> None:
    destination = report_dir / "config_snapshot"
    destination.mkdir(parents=True, exist_ok=True)
    sources = {
        "project_model_config.yaml": paths["model_config"],
        "project_experiment_config.yaml": paths["experiment_config"],
        "official_monai_train.json": paths["model_zoo"]
        / "models"
        / "brats_mri_segmentation"
        / "configs"
        / "train.json",
        "official_monai_metadata.json": paths["model_zoo"]
        / "models"
        / "brats_mri_segmentation"
        / "configs"
        / "metadata.json",
    }
    for name, source in sources.items():
        if not source.is_file():
            raise FileNotFoundError(f"Configuration provenance file is missing: {source}")
        target = destination / name
        if target.exists() and sha256_file(target) != sha256_file(source):
            raise FileExistsError(f"Conflicting config snapshot exists: {target}")
        if not target.exists():
            shutil.copy2(source, target)


def new_experiment_id(kind: str) -> str:
    if kind not in {"fullcv", "smoke"}:
        raise ValueError("kind must be fullcv or smoke")
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return f"segresnet_monai_{kind}_{timestamp}_{secrets.token_hex(3)}"


def initialize_experiment(root: Path, experiment_id: str, *, kind: str) -> dict[str, Any]:
    paths = _paths(root)
    recipe = load_recipe(paths["model_config"])
    cases = _inventory_cases(paths["raw_dataset"])
    splits, split_sha = _load_splits(paths["split"], set(cases))
    report_dir = paths["reports"] / experiment_id
    result_dir = paths["results"] / experiment_id
    manifest_path = report_dir / "experiment.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("experiment_id") != experiment_id or existing.get("backend") != "segresnet":
            raise ValueError(f"Experiment manifest ownership mismatch: {manifest_path}")
        return existing
    if report_dir.exists() and any(report_dir.iterdir()):
        raise FileExistsError(f"Non-empty report directory has no owned manifest: {report_dir}")
    report_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    _copy_config_snapshot(paths, report_dir)
    split_record: dict[str, Any] = {
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
    }
    if kind == "smoke":
        split_record.update(
            {
                "fold": 0,
                "train_cases": 8,
                "validation_cases": 2,
                "selection": (
                    "deterministic first 8 training IDs and first 2 validation IDs "
                    "from canonical fold 0; smoke only"
                ),
            }
        )
    manifest: dict[str, Any] = {
        "schema": "glioma_model_experiment_v1",
        "experiment_id": experiment_id,
        "experiment_kind": kind,
        "backend": "segresnet",
        "framework": "MONAI",
        "model": "SegResNet",
        "model_id": recipe.model_id,
        "classification": "100-epoch compute-limited single SegResNet comparison",
        "dataset": DATASET_NAME,
        "dataset_name": DATASET_NAME,
        "dataset_id": 501,
        "dataset_case_count": EXPECTED_CASE_COUNT,
        "folds": list(FOLDS),
        "fold_validation_counts": list(EXPECTED_FOLD_VALIDATION_COUNTS),
        "epochs": 3 if kind == "smoke" else recipe.epochs,
        "epochs_per_fold": 3 if kind == "smoke" else recipe.epochs,
        "original_recipe_epochs": recipe.original_recipe_epochs,
        "configuration": "MONAI Model Zoo brats_mri_segmentation bundle 0.5.4",
        "trainer": "project full-state loop using official MONAI components",
        "architecture": "monai.networks.nets.SegResNet",
        "patch_size": list(recipe.crop_size),
        "batch_size": recipe.batch_size,
        "target_spacing": [1.0, 1.0, 1.0],
        "TTA_state": "OFF",
        "split": split_record,
        "upstream": {
            "monai_repository": "https://github.com/Project-MONAI/MONAI.git",
            "monai_commit": EXPECTED_MONAI_COMMIT,
            "monai_version": "1.4.0",
            "model_zoo_repository": "https://github.com/Project-MONAI/model-zoo.git",
            "model_zoo_commit": EXPECTED_MODEL_ZOO_COMMIT,
            "bundle_version": "0.5.4",
            "bundle_metadata_pytorch": "2.4.0",
            "runtime_pytorch": "2.5.1+cu121",
            "runtime_deviation": (
                "Windows compatibility: torch 2.4.0+cu121 imports an unbundled "
                "libomp140.x86_64.dll on this host; model recipe unchanged"
            ),
        },
        "probabilities": {
            "native_channel_order": list(SEGRESNET_REGION_ORDER),
            "activation": "sigmoid",
            "conversion": SEGRESNET_REGION_CONVERSION,
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
    manifest = initialize_experiment(
        root,
        experiment_id,
        kind="smoke" if "_smoke_" in experiment_id else "fullcv",
    )
    repositories = [
        _check_repository(paths["monai"], EXPECTED_MONAI_COMMIT, "MONAI"),
        _check_repository(paths["model_zoo"], EXPECTED_MODEL_ZOO_COMMIT, "MONAI Model Zoo"),
    ]
    import torch

    monai = importlib.import_module("monai")
    if monai.__version__ != "1.4.0":
        raise RuntimeError(f"Pinned MONAI 1.4.0 is required, found {monai.__version__}")
    if not torch.__version__.startswith("2.5.1+cu121"):
        raise RuntimeError(f"Pinned torch 2.5.1+cu121 is required, found {torch.__version__}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the research-model environment")
    gpu = query_gpu_once()
    report = {
        "valid": True,
        "experiment_id": experiment_id,
        "backend": "segresnet",
        "python": sys.version,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_version": torch.__version__,
        "cuda": torch.version.cuda,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "monai": monai.__version__,
        "monai_version": monai.__version__,
        "numpy": np.__version__,
        "nibabel": importlib.metadata.version("nibabel"),
        "gpu": gpu.gpu_name,
        "gpu_name": gpu.gpu_name,
        "gpu_vram_mb": gpu.memory_total_mb,
        "repositories": repositories,
        "split_sha256": manifest["split"]["sha256"],
        "checked_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    write_json_atomic(output, report)
    return report


def _fold_spec(
    root: Path,
    experiment_id: str,
    *,
    fold: int,
    smoke: bool,
    epochs: int,
    quick_train_cases: int,
    quick_validation_cases: int,
) -> tuple[SegResNetRecipe, FoldRunSpec]:
    if fold not in FOLDS:
        raise ValueError("fold must be one of 0,1,2,3,4")
    paths = _paths(root)
    recipe = load_recipe(paths["model_config"])
    cases = _inventory_cases(paths["raw_dataset"])
    splits, split_sha = _load_splits(paths["split"], set(cases))
    split = splits[fold]
    train_ids = split["train"]
    validation_ids = split["val"]
    if smoke:
        if quick_train_cases < 1 or quick_validation_cases < 1:
            raise ValueError("Smoke subsets must contain at least one training and validation case")
        if quick_train_cases != 8 or quick_validation_cases != 2:
            raise ValueError(
                "The pinned SegResNet smoke protocol requires exactly 8 train/2 validation"
            )
        train_ids = train_ids[:quick_train_cases]
        validation_ids = validation_ids[:quick_validation_cases]
    elif epochs != recipe.epochs:
        raise ValueError(f"Full CV must use exactly {recipe.epochs} epochs")
    output_dir = paths["results"] / experiment_id / f"fold_{fold}"
    digest = config_hash(
        recipe,
        experiment_id=experiment_id,
        fold=fold,
        epochs=epochs,
        train_ids=train_ids,
        validation_ids=validation_ids,
        smoke=smoke,
    )
    spec = FoldRunSpec(
        experiment_id=experiment_id,
        fold=fold,
        epochs=epochs,
        train_cases=tuple(cases[case_id] for case_id in train_ids),
        validation_cases=tuple(cases[case_id] for case_id in validation_ids),
        output_dir=output_dir,
        config_hash=digest,
        split_sha256=split_sha,
        smoke=smoke,
    )
    return recipe, spec


def write_fold_manifest(spec: FoldRunSpec, recipe: SegResNetRecipe) -> Path:
    path = spec.output_dir / "fold_manifest.json"
    payload = fold_manifest(spec, recipe)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise ValueError(f"Fold manifest collision: {path}")
    else:
        write_json_atomic(path, payload)
    return path


def audit_fold(fold_dir: Path) -> dict[str, Any]:
    manifest_path = fold_dir / "fold_manifest.json"
    if not manifest_path.is_file():
        return {
            "valid": False,
            "complete": False,
            "safe_to_resume": False,
            "reason": "missing fold manifest",
        }
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_ids = set(str(value) for value in manifest["validation_case_ids"])
    final_checkpoint = fold_dir / "checkpoint_final.pth"
    latest_checkpoint = fold_dir / "checkpoint_latest.pth"
    summary_path = fold_dir / "validation_summary.json"
    checkpoint_path = (
        final_checkpoint
        if final_checkpoint.is_file()
        else latest_checkpoint
        if latest_checkpoint.is_file()
        else None
    )
    if checkpoint_path is None:
        return {
            "valid": False,
            "complete": False,
            "safe_to_resume": False,
            "reason": "no checkpoint is available",
        }
    import torch

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception as exc:
        return {
            "valid": False,
            "complete": False,
            "safe_to_resume": False,
            "reason": f"checkpoint cannot be loaded: {type(exc).__name__}: {exc}",
        }
    expected_checkpoint = {
        "experiment_id": manifest["experiment_id"],
        "fold": manifest["fold"],
        "config_sha256": manifest["config_sha256"],
        "split_sha256": manifest["split_sha256"],
        "target_epochs": manifest["target_epochs"],
    }
    mismatch = {
        key: (checkpoint.get(key), value)
        for key, value in expected_checkpoint.items()
        if checkpoint.get(key) != value
    }
    completed_epoch = int(checkpoint.get("completed_epoch", -1))
    target_epochs = int(manifest["target_epochs"])
    if mismatch or completed_epoch < 1 or completed_epoch > target_epochs:
        return {
            "valid": False,
            "complete": False,
            "safe_to_resume": False,
            "reason": f"checkpoint mismatch: {mismatch}",
        }
    if not final_checkpoint.is_file():
        return {
            "valid": False,
            "complete": False,
            "safe_to_resume": True,
            "reason": (
                f"owner-matched checkpoint is complete through epoch {completed_epoch}; "
                "final publication/export may resume"
            ),
        }
    if completed_epoch != target_epochs:
        return {
            "valid": False,
            "complete": False,
            "safe_to_resume": False,
            "reason": "final checkpoint epoch does not equal target epochs",
        }
    if not summary_path.is_file():
        return {
            "valid": False,
            "complete": False,
            "safe_to_resume": True,
            "reason": "owner-matched final checkpoint exists; validation export is incomplete",
        }
    try:
        validation_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if validation_summary.get("valid") is not True:
            raise ValueError("validation summary is not valid")
        if int(validation_summary.get("fold", -1)) != int(manifest["fold"]):
            raise ValueError("validation summary fold disagrees")
        if int(validation_summary.get("case_count", -1)) != len(expected_ids):
            raise ValueError("validation summary case count disagrees")
        summary_case_ids = [str(value) for value in validation_summary.get("case_ids", [])]
        if summary_case_ids != [str(value) for value in manifest["validation_case_ids"]]:
            raise ValueError("validation summary case IDs disagree")
        if validation_summary.get("native_channel_order") != list(SEGRESNET_REGION_ORDER):
            raise ValueError("validation summary native channel order disagrees")
        if validation_summary.get("canonical_channel_order") != ["ET", "TC", "WT"]:
            raise ValueError("validation summary canonical channel order disagrees")
        if validation_summary.get("probability_conversion") != SEGRESNET_REGION_CONVERSION:
            raise ValueError("validation summary probability conversion disagrees")
        inference_total = float(validation_summary.get("inference_total_seconds", -1.0))
        if not np.isfinite(inference_total) or inference_total <= 0:
            raise ValueError("validation summary inference timing is invalid")
    except Exception as exc:
        return {
            "valid": False,
            "complete": False,
            "safe_to_resume": True,
            "reason": f"validation summary is invalid: {type(exc).__name__}: {exc}",
        }
    training_log = fold_dir / "train_history.json"
    runtime_path = fold_dir / "runtime.json"
    gpu_summary_path = fold_dir / "gpu_summary.json"
    gpu_samples_path = fold_dir / "gpu_samples.csv"
    supporting = (training_log, runtime_path, gpu_summary_path, gpu_samples_path)
    missing_supporting = [path.name for path in supporting if not path.is_file()]
    telemetry_segments = list(fold_dir.glob("gpu_samples_segment_*.csv"))
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
        training_history = json.loads(training_log.read_text(encoding="utf-8"))
        runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
        gpu_summary = json.loads(gpu_summary_path.read_text(encoding="utf-8"))
        if not isinstance(training_history, list) or len(training_history) != target_epochs:
            raise ValueError("training history length does not equal target epochs")
        if [int(row.get("epoch", -1)) for row in training_history] != list(
            range(1, target_epochs + 1)
        ):
            raise ValueError("training history epoch sequence is invalid")
        if any(
            not np.isfinite(float(row.get("train_loss", np.nan)))
            for row in training_history
        ):
            raise ValueError("training history contains a non-finite loss")
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
    predictions = fold_dir / "predictions"
    canonical = fold_dir / "probabilities" / "canonical"
    native = fold_dir / "probabilities" / "native"
    prediction_ids = {path.name.removesuffix(".nii.gz") for path in predictions.glob("*.nii.gz")}
    canonical_ids = {path.stem for path in canonical.glob("*.npz")}
    native_ids = {path.stem for path in native.glob("*.npz")}
    if (
        prediction_ids != expected_ids
        or canonical_ids != expected_ids
        or native_ids != expected_ids
    ):
        extra_ids = (prediction_ids | canonical_ids | native_ids) - expected_ids
        return {
            "valid": False,
            "complete": False,
            "safe_to_resume": not extra_ids,
            "reason": "prediction/probability inventory mismatch",
        }
    raw_labels = _paths(_project_root())["raw_dataset"] / "labelsTr"
    try:
        for case_id in sorted(expected_ids):
            reference = raw_labels / f"{case_id}.nii.gz"
            if not reference.is_file():
                raise FileNotFoundError(f"Ground truth is missing for {case_id}: {reference}")
            prediction = cast(
                nib.Nifti1Image, nib.load(str(predictions / f"{case_id}.nii.gz"))
            )
            ground_truth = cast(nib.Nifti1Image, nib.load(str(reference)))
            if prediction.shape != ground_truth.shape or not np.allclose(
                prediction.affine, ground_truth.affine, rtol=0.0, atol=1e-4
            ):
                raise ValueError(f"Prediction geometry mismatch for {case_id}")
            labels = set(int(value) for value in np.unique(np.asanyarray(prediction.dataobj)))
            if not labels.issubset({0, 1, 2, 3}):
                raise ValueError(
                    f"Prediction labels are invalid for {case_id}: {sorted(labels)}"
                )
            validate_canonical_probability_npz(
                canonical / f"{case_id}.npz",
                reference_nifti=reference,
                expected_case_id=case_id,
                expected_native_channel_order=SEGRESNET_REGION_ORDER,
                expected_conversion=SEGRESNET_REGION_CONVERSION,
            )
    except FileNotFoundError:
        raise
    except Exception as exc:
        return {
            "valid": False,
            "complete": False,
            "safe_to_resume": True,
            "reason": f"derived validation artifact is corrupt: {type(exc).__name__}: {exc}",
        }
    return {
        "valid": True,
        "complete": True,
        "safe_to_resume": False,
        "fold": manifest["fold"],
        "experiment_id": manifest["experiment_id"],
        "expected_validation_cases": len(expected_ids),
        "validation_case_count": len(prediction_ids),
        "checkpoint_final": str(final_checkpoint.resolve()),
        "prediction_dir": str(predictions.resolve()),
        "canonical_probability_dir": str(canonical.resolve()),
    }


def assemble_oof(root: Path, experiment_id: str, *, folds: Sequence[int]) -> dict[str, Any]:
    paths = _paths(root)
    experiment_root = paths["results"] / experiment_id
    destination = experiment_root / "oof"
    expected: dict[str, int] = {}
    fold_manifests: list[dict[str, Any]] = []
    for fold in folds:
        fold_dir = experiment_root / f"fold_{fold}"
        audit = audit_fold(fold_dir)
        if not audit.get("complete") or not audit.get("valid"):
            raise RuntimeError(f"Fold {fold} is not complete: {audit}")
        manifest = json.loads((fold_dir / "fold_manifest.json").read_text(encoding="utf-8"))
        fold_manifests.append(manifest)
        for case_id in manifest["validation_case_ids"]:
            if case_id in expected:
                raise ValueError(f"Case {case_id} appears in folds {expected[case_id]} and {fold}")
            expected[case_id] = fold
    if len(folds) == 5 and len(expected) != EXPECTED_CASE_COUNT:
        raise ValueError(
            f"Full OOF assembly expected {EXPECTED_CASE_COUNT} cases, got {len(expected)}"
        )
    if destination.exists():
        masks = {
            path.name.removesuffix(".nii.gz")
            for path in (destination / "predictions").glob("*.nii.gz")
        }
        probabilities = {
            path.stem
            for path in (destination / "probabilities" / "canonical").glob("*.npz")
        }
        if masks == set(expected) and probabilities == set(expected):
            return {
                "valid": True,
                "reused": True,
                "case_count": len(expected),
                "oof_dir": str(destination.resolve()),
            }
        raise FileExistsError(f"Incomplete/conflicting OOF directory exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".oof_stage_", dir=destination.parent))
    try:
        for subdirectory in ("predictions", "probabilities/native", "probabilities/canonical"):
            (staging / subdirectory).mkdir(parents=True, exist_ok=True)
        for manifest in fold_manifests:
            fold_dir = Path(manifest["output_dir"])
            for case_id in manifest["validation_case_ids"]:
                sources = (
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
                )
                for source, target in sources:
                    try:
                        os.link(source, target)
                    except OSError:
                        shutil.copy2(source, target)
        summary = {
            "valid": True,
            "backend": "segresnet",
            "model_id": "monai_model_zoo_brats_seg_resnet",
            "experiment_id": experiment_id,
            "folds": list(folds),
            "case_count": len(expected),
            "each_case_once": True,
            "case_to_fold": expected,
            "native_channel_order": list(SEGRESNET_REGION_ORDER),
            "canonical_channel_order": ["ET", "TC", "WT"],
            "conversion": SEGRESNET_REGION_CONVERSION,
        }
        write_json_atomic(staging / "oof_manifest.json", summary)
        os.replace(staging, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return {
        "valid": True,
        "reused": False,
        "case_count": len(expected),
        "oof_dir": str(destination.resolve()),
    }


def write_evaluation_manifest(root: Path, experiment_id: str, *, folds: Sequence[int]) -> Path:
    paths = _paths(root)
    experiment_root = paths["results"] / experiment_id
    oof = experiment_root / "oof"
    records = []
    for fold in folds:
        fold_dir = experiment_root / f"fold_{fold}"
        records.append(
            {
                "fold": fold,
                "prediction_dir": str((fold_dir / "predictions").resolve()),
                "probability_dir": str(
                    (fold_dir / "probabilities" / "canonical").resolve()
                ),
            }
        )
    payload = {
        "schema": "glioma_model_crossval_artifacts_v1",
        "backend": "segresnet",
        "model_id": "monai_model_zoo_brats_seg_resnet",
        "model_provenance": {
            "implementation_repository": "https://github.com/Project-MONAI/MONAI.git",
            "implementation_commit": EXPECTED_MONAI_COMMIT,
            "recipe_repository": "https://github.com/Project-MONAI/model-zoo.git",
            "recipe_commit": EXPECTED_MODEL_ZOO_COMMIT,
        },
        "prediction_tta_state": "OFF",
        "folds": records,
        "pooled_prediction_dir": str((oof / "predictions").resolve()),
        "pooled_probability_dir": str((oof / "probabilities" / "canonical").resolve()),
        "probability_contract": {
            "required": True,
            "native_channel_order": list(SEGRESNET_REGION_ORDER),
            "canonical_channel_order": ["ET", "TC", "WT"],
            "conversion": SEGRESNET_REGION_CONVERSION,
            "schema": "glioma_canonical_probabilities_v1",
        },
    }
    destination = paths["reports"] / experiment_id / "crossval_artifact_manifest.json"
    write_json_atomic(destination, payload)
    return destination


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

    memory = subparsers.add_parser("memory-preflight")
    memory.add_argument("--experiment-id", required=True)
    memory.add_argument("--fold", type=int, default=0)
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
    if args.command == "new-experiment":
        print(new_experiment_id(args.kind))
        return 0
    if args.command == "initialize":
        result = initialize_experiment(root, args.experiment_id, kind=args.kind)
    elif args.command == "system-check":
        result = system_check(root, args.experiment_id, output=args.output)
    elif args.command == "memory-preflight":
        recipe, spec = _fold_spec(
            root,
            args.experiment_id,
            fold=args.fold,
            smoke=True,
            epochs=3,
            quick_train_cases=8,
            quick_validation_cases=2,
        )
        write_fold_manifest(spec, recipe)
        result = memory_preflight(recipe, spec.train_cases[0], output_json=args.output)
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
        write_fold_manifest(spec, recipe)
        result = run_fold(
            recipe,
            spec,
            resume=args.resume,
            stop_after_epoch=args.stop_after_epoch,
        )
    elif args.command == "audit-fold":
        result = audit_fold(args.fold_dir.resolve())
        if args.output:
            write_json_atomic(args.output, result)
    elif args.command == "assemble-oof":
        result = assemble_oof(root, args.experiment_id, folds=args.fold)
    elif args.command == "write-evaluation-manifest":
        path = write_evaluation_manifest(root, args.experiment_id, folds=args.fold)
        result = {"manifest": str(path.resolve())}
    else:  # pragma: no cover
        raise AssertionError(args.command)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result.get("valid", True) else 2


if __name__ == "__main__":
    raise SystemExit(main())
