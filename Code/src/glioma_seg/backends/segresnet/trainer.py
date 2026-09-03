"""Project-owned orchestration around the unmodified MONAI SegResNet class.

The network, loss, optimizer, augmentations, and inference window are copied
from the pinned official MONAI Model Zoo BraTS bundle.  This module adds only
BraTS-2023 label/modality adapters, exact project folds, crash-safe checkpoint
state, monitoring, and original-grid artifact export.
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import importlib
import json
import math
import os
import random
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import nibabel as nib
import numpy as np

from glioma_seg.backends.segresnet.config import SegResNetRecipe, normalized_recipe_json
from glioma_seg.ensembles.canonical_probabilities import (
    SEGRESNET_REGION_CONVERSION,
    SEGRESNET_REGION_ORDER,
    segresnet_regions_to_canonical,
    write_canonical_probability_npz,
)
from glioma_seg.monitoring.gpu_monitor import GPUMonitor
from glioma_seg.monitoring.timing import write_json_atomic
from glioma_seg.utils.hashing import sha256_file


@dataclass(frozen=True)
class CasePaths:
    case_id: str
    images: tuple[Path, Path, Path, Path]
    label: Path


@dataclass(frozen=True)
class FoldRunSpec:
    experiment_id: str
    fold: int
    epochs: int
    train_cases: tuple[CasePaths, ...]
    validation_cases: tuple[CasePaths, ...]
    output_dir: Path
    config_hash: str
    split_sha256: str
    smoke: bool


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _torch() -> Any:
    import torch

    return torch


def _monai_imports() -> dict[str, Any]:
    """Import MONAI lazily so the project remains importable outside the model env."""

    data = importlib.import_module("monai.data")
    inferers = importlib.import_module("monai.inferers")
    losses = importlib.import_module("monai.losses")
    nets = importlib.import_module("monai.networks.nets")
    transforms = importlib.import_module("monai.transforms")
    utilities = importlib.import_module("monai.utils")
    return {
        "Compose": transforms.Compose,
        "ConvertToMultiChannelBasedOnBratsClassesd": (
            transforms.ConvertToMultiChannelBasedOnBratsClassesd
        ),
        "DataLoader": data.DataLoader,
        "Dataset": data.Dataset,
        "DiceLoss": losses.DiceLoss,
        "Lambdad": transforms.Lambdad,
        "LoadImaged": transforms.LoadImaged,
        "NormalizeIntensityd": transforms.NormalizeIntensityd,
        "RandFlipd": transforms.RandFlipd,
        "RandScaleIntensityd": transforms.RandScaleIntensityd,
        "RandShiftIntensityd": transforms.RandShiftIntensityd,
        "RandSpatialCropd": transforms.RandSpatialCropd,
        "SegResNet": nets.SegResNet,
        "set_determinism": utilities.set_determinism,
        "sliding_window_inference": inferers.sliding_window_inference,
    }


def _remap_brats2023_et_to_legacy(label: Any) -> Any:
    """Map ET 3 to legacy bundle ET 4 without changing NCR/ED/background."""

    torch = _torch()
    if isinstance(label, torch.Tensor):
        return torch.where(label == 3, torch.as_tensor(4, device=label.device), label)
    array = np.asarray(label)
    return np.where(array == 3, 4, array)


def _case_entry(case: CasePaths) -> dict[str, Any]:
    return {
        "case_id": case.case_id,
        "image": [str(path) for path in case.images],
        "label": str(case.label),
    }


def build_transforms(recipe: SegResNetRecipe, *, training: bool) -> Any:
    monai = _monai_imports()
    transforms: list[Any] = [
        monai["LoadImaged"](keys=["image", "label"], image_only=False),
        monai["Lambdad"](keys="label", func=_remap_brats2023_et_to_legacy),
        monai["ConvertToMultiChannelBasedOnBratsClassesd"](keys="label"),
        monai["NormalizeIntensityd"](
            keys="image",
            nonzero=True,
            channel_wise=True,
        ),
    ]
    if training:
        augmentation = recipe.payload["augmentation"]
        transforms.extend(
            [
                monai["RandSpatialCropd"](
                    keys=["image", "label"],
                    roi_size=recipe.crop_size,
                    random_size=False,
                ),
                monai["RandFlipd"](
                    keys=["image", "label"],
                    prob=float(augmentation["flip_probability_per_axis"]),
                    spatial_axis=0,
                ),
                monai["RandFlipd"](
                    keys=["image", "label"],
                    prob=float(augmentation["flip_probability_per_axis"]),
                    spatial_axis=1,
                ),
                monai["RandFlipd"](
                    keys=["image", "label"],
                    prob=float(augmentation["flip_probability_per_axis"]),
                    spatial_axis=2,
                ),
                monai["RandScaleIntensityd"](
                    keys="image",
                    factors=float(augmentation["intensity_scale_factor"]),
                    prob=float(augmentation["intensity_scale_probability"]),
                ),
                monai["RandShiftIntensityd"](
                    keys="image",
                    offsets=float(augmentation["intensity_shift_offset"]),
                    prob=float(augmentation["intensity_shift_probability"]),
                ),
            ]
        )
    return monai["Compose"](transforms)


def build_model(recipe: SegResNetRecipe) -> Any:
    monai = _monai_imports()
    architecture = recipe.payload["architecture"]
    return monai["SegResNet"](
        blocks_down=tuple(int(value) for value in architecture["blocks_down"]),
        blocks_up=tuple(int(value) for value in architecture["blocks_up"]),
        init_filters=int(architecture["init_filters"]),
        in_channels=int(architecture["in_channels"]),
        out_channels=int(architecture["out_channels"]),
        dropout_prob=float(architecture["dropout_prob"]),
    )


def _loss(recipe: SegResNetRecipe) -> Any:
    monai = _monai_imports()
    training = recipe.payload["training"]
    return monai["DiceLoss"](
        smooth_nr=float(training["loss_smooth_nr"]),
        smooth_dr=float(training["loss_smooth_dr"]),
        squared_pred=bool(training["squared_pred"]),
        to_onehot_y=False,
        sigmoid=bool(training["sigmoid"]),
    )


def _dataloader(
    cases: Sequence[CasePaths],
    recipe: SegResNetRecipe,
    *,
    training: bool,
    generator: Any,
    workers: int | None = None,
) -> Any:
    monai = _monai_imports()
    dataset = monai["Dataset"](
        data=[_case_entry(case) for case in cases],
        transform=build_transforms(recipe, training=training),
    )
    return monai["DataLoader"](
        dataset,
        batch_size=recipe.batch_size,
        shuffle=training,
        num_workers=recipe.num_workers if workers is None else workers,
        generator=generator,
    )


def _rng_state(generator: Any) -> dict[str, Any]:
    torch = _torch()
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all(),
        "data_loader_generator": generator.get_state(),
    }


def _restore_rng_state(state: Mapping[str, Any], generator: Any) -> None:
    torch = _torch()
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state_all(state["torch_cuda"])
    generator.set_state(state["data_loader_generator"])


def _save_torch_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    torch = _torch()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(dict(payload), temporary)
        # Windows requires a writable file descriptor for fsync.
        with temporary.open("r+b") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _checkpoint_payload(
    *,
    spec: FoldRunSpec,
    model: Any,
    optimizer: Any,
    scheduler: Any,
    scaler: Any,
    generator: Any,
    completed_epoch: int,
    best_validation_dice: float,
    history: Sequence[Mapping[str, Any]],
    cumulative_training_seconds: float,
) -> dict[str, Any]:
    return {
        "schema": "glioma_segresnet_checkpoint_v1",
        "experiment_id": spec.experiment_id,
        "backend": "segresnet",
        "model_id": "monai_model_zoo_brats_seg_resnet",
        "fold": spec.fold,
        "target_epochs": spec.epochs,
        "completed_epoch": completed_epoch,
        "config_sha256": spec.config_hash,
        "split_sha256": spec.split_sha256,
        "smoke": spec.smoke,
        "saved_at": _utc_now(),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "grad_scaler_state": scaler.state_dict(),
        "rng_state": _rng_state(generator),
        "best_validation_dice": best_validation_dice,
        "cumulative_training_seconds": cumulative_training_seconds,
        "history": list(history),
    }


def _load_checkpoint(
    path: Path,
    *,
    spec: FoldRunSpec,
    model: Any,
    optimizer: Any,
    scheduler: Any,
    scaler: Any,
    generator: Any,
) -> tuple[int, float, list[dict[str, Any]], float]:
    torch = _torch()
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise ValueError(f"Unable to load SegResNet checkpoint {path}: {exc}") from exc
    expected = {
        "schema": "glioma_segresnet_checkpoint_v1",
        "experiment_id": spec.experiment_id,
        "backend": "segresnet",
        "fold": spec.fold,
        "config_sha256": spec.config_hash,
        "split_sha256": spec.split_sha256,
        "smoke": spec.smoke,
    }
    mismatches = {
        key: (payload.get(key), value)
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Checkpoint ownership/config mismatch: {mismatches}")
    completed = int(payload.get("completed_epoch", -1))
    if completed < 0 or completed > spec.epochs:
        raise ValueError(f"Invalid completed_epoch={completed} in {path}")
    model.load_state_dict(payload["model_state"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state"])
    scheduler.load_state_dict(payload["scheduler_state"])
    scaler.load_state_dict(payload["grad_scaler_state"])
    _restore_rng_state(payload["rng_state"], generator)
    history = payload.get("history")
    if not isinstance(history, list) or len(history) != completed:
        raise ValueError("Checkpoint history length does not match completed_epoch")
    cumulative_training_seconds = float(
        payload.get(
            "cumulative_training_seconds",
            sum(float(row.get("epoch_seconds", 0.0)) for row in history),
        )
    )
    if not math.isfinite(cumulative_training_seconds) or cumulative_training_seconds < 0:
        raise ValueError("Checkpoint cumulative_training_seconds is invalid")
    return (
        completed,
        float(payload.get("best_validation_dice", -math.inf)),
        history,
        cumulative_training_seconds,
    )


def _region_dice(probabilities: Any, target: Any, threshold: float) -> tuple[float, float, float]:
    torch = _torch()
    prediction = probabilities >= threshold
    truth = target > 0.5
    values: list[float] = []
    for channel in range(3):
        pred_channel = prediction[:, channel]
        truth_channel = truth[:, channel]
        intersection = torch.count_nonzero(pred_channel & truth_channel).item()
        denominator = torch.count_nonzero(pred_channel).item() + torch.count_nonzero(
            truth_channel
        ).item()
        values.append(float(2.0 * intersection / denominator) if denominator else math.nan)
    return tuple(values)  # type: ignore[return-value]


def _mean_finite(values: Sequence[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return float(np.mean(finite)) if finite else math.nan


def validate_model(
    model: Any,
    loader: Any,
    recipe: SegResNetRecipe,
    *,
    device: Any,
    export_cases: Mapping[str, CasePaths] | None = None,
    prediction_dir: Path | None = None,
    native_probability_dir: Path | None = None,
    canonical_probability_dir: Path | None = None,
    overwrite: bool = False,
) -> tuple[float, dict[str, float], list[float]]:
    torch = _torch()
    monai = _monai_imports()
    inference = recipe.payload["inference"]
    threshold = float(inference["threshold"])
    model.eval()
    region_values: list[list[float]] = [[], [], []]
    inference_seconds: list[float] = []
    with torch.no_grad():
        for batch in loader:
            case_ids = [str(value) for value in batch["case_id"]]
            images = batch["image"].to(device, non_blocking=False)
            targets = batch["label"].to(device, non_blocking=False)
            started = time.monotonic()
            with torch.cuda.amp.autocast(enabled=bool(recipe.payload["training"]["amp"])):
                logits = monai["sliding_window_inference"](
                    images,
                    roi_size=recipe.validation_roi_size,
                    sw_batch_size=int(inference["sliding_window_batch_size"]),
                    predictor=model,
                    overlap=float(inference["overlap"]),
                )
            probabilities = torch.sigmoid(logits)
            torch.cuda.synchronize()
            per_case_seconds = (time.monotonic() - started) / len(case_ids)
            inference_seconds.extend([per_case_seconds] * len(case_ids))
            if not torch.isfinite(probabilities).all():
                raise FloatingPointError("SegResNet validation probabilities contain NaN/Inf")
            for channel, value in enumerate(_region_dice(probabilities, targets, threshold)):
                if math.isfinite(value):
                    region_values[channel].append(value)

            if export_cases is not None:
                if not all(
                    path is not None
                    for path in (
                        prediction_dir,
                        native_probability_dir,
                        canonical_probability_dir,
                    )
                ):
                    raise ValueError(
                        "All prediction/probability directories are required for export"
                    )
                probability_arrays = probabilities.detach().cpu().numpy().astype(np.float32)
                for item_index, case_id in enumerate(case_ids):
                    case = export_cases.get(case_id)
                    if case is None:
                        raise ValueError(f"Validation loader returned unknown case {case_id}")
                    _export_case(
                        case,
                        probability_arrays[item_index],
                        threshold=threshold,
                        prediction_dir=prediction_dir,  # type: ignore[arg-type]
                        native_probability_dir=native_probability_dir,  # type: ignore[arg-type]
                        canonical_probability_dir=canonical_probability_dir,  # type: ignore[arg-type]
                        overwrite=overwrite,
                    )
    per_region = {
        "TC": _mean_finite(region_values[0]),
        "WT": _mean_finite(region_values[1]),
        "ET": _mean_finite(region_values[2]),
    }
    return _mean_finite(list(per_region.values())), per_region, inference_seconds


def _write_native_npz(
    path: Path,
    *,
    case: CasePaths,
    probabilities: np.ndarray,
    overwrite: bool,
) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite native probabilities: {path}")
    reference = cast(nib.Nifti1Image, nib.load(str(case.label)))
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(
                handle,
                probabilities=np.asarray(probabilities, dtype=np.float32),
                channel_order=np.asarray(SEGRESNET_REGION_ORDER),
                activation=np.asarray("sigmoid"),
                affine=np.asarray(reference.affine, dtype=np.float64),
                spacing_mm=np.asarray(reference.header.get_zooms()[:3], dtype=np.float64),
                case_id=np.asarray(case.case_id),
                reference_sha256=np.asarray(sha256_file(case.label)),
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_nifti_atomic(
    path: Path,
    labels: np.ndarray,
    reference_path: Path,
    overwrite: bool,
) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite prediction: {path}")
    reference = cast(nib.Nifti1Image, nib.load(str(reference_path)))
    header = reference.header.copy()
    header.set_data_dtype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".nii.gz", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        nib.save(
            nib.Nifti1Image(np.asarray(labels, dtype=np.uint8), reference.affine, header),
            str(temporary),
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _export_case(
    case: CasePaths,
    native_probabilities: np.ndarray,
    *,
    threshold: float,
    prediction_dir: Path,
    native_probability_dir: Path,
    canonical_probability_dir: Path,
    overwrite: bool,
) -> None:
    if native_probabilities.shape[0] != 3 or not np.all(np.isfinite(native_probabilities)):
        raise ValueError(f"Invalid SegResNet probability tensor for {case.case_id}")
    tc = native_probabilities[0] >= threshold
    wt = native_probabilities[1] >= threshold
    et = native_probabilities[2] >= threshold
    # BraTS regions are nested.  Independent sigmoid outputs are closed into a
    # valid hard-label map; probabilities remain untouched for future ensemble.
    tc = tc | et
    wt = wt | tc
    labels = np.zeros(tc.shape, dtype=np.uint8)
    labels[wt] = 2
    labels[tc] = 1
    labels[et] = 3
    _write_nifti_atomic(
        prediction_dir / f"{case.case_id}.nii.gz", labels, case.label, overwrite
    )
    _write_native_npz(
        native_probability_dir / f"{case.case_id}.npz",
        case=case,
        probabilities=native_probabilities,
        overwrite=overwrite,
    )
    canonical = segresnet_regions_to_canonical(native_probabilities)
    write_canonical_probability_npz(
        canonical_probability_dir / f"{case.case_id}.npz",
        case_id=case.case_id,
        probabilities=canonical,
        native_channel_order=SEGRESNET_REGION_ORDER,
        conversion=SEGRESNET_REGION_CONVERSION,
        reference_nifti=case.label,
        overwrite=overwrite,
    )


def memory_preflight(
    recipe: SegResNetRecipe,
    case: CasePaths,
    *,
    output_json: Path,
) -> dict[str, Any]:
    torch = _torch()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the SegResNet memory preflight")
    monai = _monai_imports()
    monai["set_determinism"](seed=recipe.seed)
    device = torch.device("cuda:0")
    result: dict[str, Any] = {
        "valid": False,
        "model": recipe.model_id,
        "case_id": case.case_id,
        "crop_size": list(recipe.crop_size),
        "batch_size": recipe.batch_size,
        "amp": bool(recipe.payload["training"]["amp"]),
        "started_at": _utc_now(),
    }
    try:
        transformed = build_transforms(recipe, training=True)(_case_entry(case))
        images = transformed["image"].unsqueeze(0).to(device)
        targets = transformed["label"].unsqueeze(0).to(device)
        if tuple(images.shape) != (1, 4, *recipe.crop_size):
            raise ValueError(f"Unexpected memory-probe image shape: {tuple(images.shape)}")
        if tuple(targets.shape) != (1, 3, *recipe.crop_size):
            raise ValueError(f"Unexpected memory-probe label shape: {tuple(targets.shape)}")
        model = build_model(recipe).to(device)
        loss_function = _loss(recipe)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=recipe.learning_rate, weight_decay=recipe.weight_decay
        )
        scaler = torch.cuda.amp.GradScaler(enabled=bool(recipe.payload["training"]["amp"]))
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=bool(recipe.payload["training"]["amp"])):
            output = model(images)
            loss = loss_function(output, targets)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite memory-probe loss: {loss.item()}")
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize()
        total = int(torch.cuda.get_device_properties(0).total_memory)
        peak_allocated = int(torch.cuda.max_memory_allocated())
        peak_reserved = int(torch.cuda.max_memory_reserved())
        result.update(
            {
                "valid": True,
                "loss": float(loss.detach().cpu()),
                "output_shape": list(output.shape),
                "peak_allocated_mb": peak_allocated / 1024**2,
                "peak_reserved_mb": peak_reserved / 1024**2,
                "total_vram_mb": total / 1024**2,
                "reserved_fraction": peak_reserved / total,
            }
        )
    except torch.cuda.OutOfMemoryError as exc:
        result.update({"error_type": "CUDAOutOfMemoryError", "error": str(exc)})
        result["ended_at"] = _utc_now()
        write_json_atomic(output_json, result)
        raise RuntimeError(
            "The unchanged official MONAI SegResNet BraTS crop (224x224x144, batch 1) "
            "does not fit this GPU. No smaller crop was selected silently."
        ) from exc
    except Exception as exc:
        result.update({"error_type": type(exc).__name__, "error": str(exc)})
        result["ended_at"] = _utc_now()
        write_json_atomic(output_json, result)
        raise
    finally:
        if "ended_at" not in result:
            result["ended_at"] = _utc_now()
    write_json_atomic(output_json, result)
    return result


def _aggregate_gpu_telemetry(
    output_dir: Path,
    *,
    merged_csv: Path,
    summary_json: Path,
) -> dict[str, Any]:
    """Merge immutable per-invocation samples without losing resume evidence."""

    fields = (
        "timestamp",
        "elapsed_seconds",
        "gpu_name",
        "gpu_utilization_percent",
        "memory_used_mb",
        "memory_total_mb",
        "temperature_c",
        "power_w",
    )
    segments = sorted(output_dir.glob("gpu_samples_segment_*.csv"))
    if not segments:
        if merged_csv.is_file() and summary_json.is_file():
            return json.loads(summary_json.read_text(encoding="utf-8"))
        raise RuntimeError("No SegResNet training GPU telemetry segment is available")

    merged_rows: list[dict[str, str]] = []
    elapsed_offset = 0.0
    errors: list[str] = []
    backends: set[str] = set()
    for segment in segments:
        with segment.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != fields:
                raise ValueError(f"GPU telemetry columns are invalid in {segment}")
            rows = list(reader)
        segment_max = 0.0
        for row in rows:
            local_elapsed = float(row["elapsed_seconds"])
            segment_max = max(segment_max, local_elapsed)
            row["elapsed_seconds"] = repr(elapsed_offset + local_elapsed)
            merged_rows.append(row)
        elapsed_offset += segment_max + (0.001 if rows else 0.0)
        segment_summary = segment.with_suffix(".summary.json")
        if segment_summary.is_file():
            evidence = json.loads(segment_summary.read_text(encoding="utf-8"))
            backends.add(str(evidence.get("backend", "unknown")))
            errors.extend(str(value) for value in evidence.get("errors", []))

    if not merged_rows:
        raise RuntimeError("SegResNet GPU telemetry segments contain no samples")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{merged_csv.name}.", suffix=".tmp", dir=merged_csv.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(merged_rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, merged_csv)
    finally:
        temporary.unlink(missing_ok=True)

    def values(name: str) -> list[float]:
        return [float(row[name]) for row in merged_rows if row[name] not in {"", "None"}]

    utilizations = values("gpu_utilization_percent")
    memories = values("memory_used_mb")
    totals = values("memory_total_mb")
    temperatures = values("temperature_c")
    powers = values("power_w")
    summary = {
        "samples": len(merged_rows),
        "segments": len(segments),
        "peak_memory_used_mb": max(memories),
        "dedicated_memory_total_mb": max(totals),
        "mean_gpu_utilization_percent": float(np.mean(utilizations)),
        "peak_temperature_c": max(temperatures),
        "mean_power_w": float(np.mean(powers)) if powers else None,
        "backend": "aggregated:" + ",".join(sorted(backends or {"unknown"})),
        "errors": errors,
        "includes_all_owner_matched_invocations": True,
        "segment_files": [path.name for path in segments],
    }
    write_json_atomic(summary_json, summary)
    return summary


def run_fold(
    recipe: SegResNetRecipe,
    spec: FoldRunSpec,
    *,
    resume: bool,
    stop_after_epoch: int | None = None,
) -> dict[str, Any]:
    torch = _torch()
    monai = _monai_imports()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for SegResNet training")
    if spec.epochs < 1 or spec.epochs > recipe.epochs:
        raise ValueError(f"epochs must be in 1..{recipe.epochs}, got {spec.epochs}")
    if stop_after_epoch is not None and not (1 <= stop_after_epoch <= spec.epochs):
        raise ValueError("stop_after_epoch must be within the run epoch range")
    output = spec.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_latest = output / "checkpoint_latest.pth"
    checkpoint_best = output / "checkpoint_best.pth"
    checkpoint_final = output / "checkpoint_final.pth"
    training_log = output / "train_history.json"
    runtime_path = output / "runtime.json"
    gpu_summary_path = output / "gpu_summary.json"
    gpu_samples = output / "gpu_samples.csv"
    predictions = output / "predictions"
    native_probabilities = output / "probabilities" / "native"
    canonical_probabilities = output / "probabilities" / "canonical"

    monai["set_determinism"](seed=recipe.seed + spec.fold)
    generator = torch.Generator()
    generator.manual_seed(recipe.seed + spec.fold)
    device = torch.device("cuda:0")
    model = build_model(recipe).to(device)
    loss_function = _loss(recipe)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=recipe.learning_rate, weight_decay=recipe.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=spec.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(recipe.payload["training"]["amp"]))
    start_epoch = 0
    best_validation_dice = -math.inf
    history: list[dict[str, Any]] = []
    prior_training_seconds = 0.0
    if resume:
        resume_path = checkpoint_final if checkpoint_final.is_file() else checkpoint_latest
        if not resume_path.is_file():
            raise FileNotFoundError(f"No owner-matched SegResNet checkpoint to resume: {output}")
        (
            start_epoch,
            best_validation_dice,
            history,
            prior_training_seconds,
        ) = _load_checkpoint(
            resume_path,
            spec=spec,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            generator=generator,
        )
    elif any(path.exists() for path in (checkpoint_latest, checkpoint_best, checkpoint_final)):
        raise FileExistsError(
            "Fold output already contains checkpoints; use verified resume or a new "
            f"experiment: {output}"
        )

    training_required = start_epoch < spec.epochs
    train_loader = (
        _dataloader(spec.train_cases, recipe, training=True, generator=generator)
        if training_required
        else None
    )
    validation_loader = (
        _dataloader(spec.validation_cases, recipe, training=False, generator=generator)
        if training_required
        else None
    )
    segment_stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    segment_csv = output / (
        f"gpu_samples_segment_epoch_{start_epoch:04d}_{segment_stamp}.csv"
    )
    monitor = (
        GPUMonitor(segment_csv, interval_seconds=2.0).start()
        if training_required
        else None
    )
    started_at = _utc_now()
    started_monotonic = time.monotonic()
    stopped_for_resume_test = False
    try:
        for epoch_index in range(start_epoch, spec.epochs):
            if train_loader is None or validation_loader is None or monitor is None:
                raise AssertionError("Training loaders/monitor were not initialized")
            epoch_started = time.monotonic()
            model.train()
            losses: list[float] = []
            for batch in train_loader:
                images = batch["image"].to(device, non_blocking=False)
                targets = batch["label"].to(device, non_blocking=False)
                if images.shape[1] != 4 or targets.shape[1] != 3:
                    raise ValueError(
                        f"Wrong channel count: image={tuple(images.shape)}, "
                        f"label={tuple(targets.shape)}"
                    )
                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=bool(recipe.payload["training"]["amp"])):
                    logits = model(images)
                    loss = loss_function(logits, targets)
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"NaN/Inf training loss at fold={spec.fold}, epoch={epoch_index + 1}"
                    )
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                losses.append(float(loss.detach().cpu()))
            scheduler.step()
            validation_mean, validation_regions, _ = validate_model(
                model, validation_loader, recipe, device=device
            )
            epoch_seconds = time.monotonic() - epoch_started
            row = {
                "epoch": epoch_index + 1,
                "train_loss": float(np.mean(losses)),
                "validation_mean_dice": validation_mean,
                "validation_dice_native_order": validation_regions,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "epoch_seconds": epoch_seconds,
                "completed_at": _utc_now(),
            }
            history.append(row)
            completed_epoch = epoch_index + 1
            payload = _checkpoint_payload(
                spec=spec,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                generator=generator,
                completed_epoch=completed_epoch,
                best_validation_dice=max(best_validation_dice, validation_mean),
                history=history,
                cumulative_training_seconds=(
                    prior_training_seconds + time.monotonic() - started_monotonic
                ),
            )
            _save_torch_atomic(checkpoint_latest, payload)
            if validation_mean > best_validation_dice:
                best_validation_dice = validation_mean
                _save_torch_atomic(checkpoint_best, payload)
            write_json_atomic(training_log, history)
            elapsed = time.monotonic() - started_monotonic
            average_epoch = elapsed / max(1, completed_epoch - start_epoch)
            eta = average_epoch * (spec.epochs - completed_epoch)
            snapshot = monitor.latest
            gpu_text = ""
            if snapshot is not None:
                gpu_text = (
                    f" GPU={snapshot.gpu_utilization_percent:.0f}%"
                    f" VRAM={snapshot.memory_used_mb:.0f}/{snapshot.memory_total_mb:.0f}MiB"
                    f" Temp={snapshot.temperature_c:.0f}C"
                )
            print(
                f"[SEGRESNET] Experiment={spec.experiment_id} Fold={spec.fold} "
                f"Epoch={completed_epoch}/{spec.epochs} Progress={completed_epoch/spec.epochs:.1%} "
                f"TrainLoss={row['train_loss']:.6f} ValDice={validation_mean:.6f} "
                f"Elapsed={elapsed/60:.1f}m ETA={eta/3600:.2f}h{gpu_text}",
                flush=True,
            )
            if stop_after_epoch is not None and completed_epoch == stop_after_epoch:
                stopped_for_resume_test = True
                break
    finally:
        if monitor is not None:
            segment_summary = monitor.stop().to_dict()
            write_json_atomic(segment_csv.with_suffix(".summary.json"), segment_summary)
        _aggregate_gpu_telemetry(
            output,
            merged_csv=gpu_samples,
            summary_json=gpu_summary_path,
        )

    ended_at = _utc_now()
    invocation_seconds = time.monotonic() - started_monotonic
    total_seconds = prior_training_seconds + invocation_seconds
    runtime = {
        "stage": "segresnet_fold_training",
        "fold": spec.fold,
        "started_at": started_at,
        "ended_at": ended_at,
        "total_seconds": total_seconds,
        "total_hours": total_seconds / 3600.0,
        "invocation_seconds": invocation_seconds,
        "resumed_from_epoch": start_epoch,
        "timing_scope": "cumulative owner-matched training invocations",
        "number_of_epochs": len(history),
        "target_epochs": spec.epochs,
        "average_seconds_per_epoch": (
            total_seconds / len(history) if history else None
        ),
        "stopped_for_resume_test": stopped_for_resume_test,
    }
    write_json_atomic(training_log, history)
    write_json_atomic(runtime_path, runtime)
    if stopped_for_resume_test:
        return {"complete": False, "resume_checkpoint": str(checkpoint_latest), **runtime}

    if len(history) != spec.epochs:
        raise RuntimeError(f"Training ended at {len(history)}/{spec.epochs} epochs")
    final_payload = _checkpoint_payload(
        spec=spec,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        generator=generator,
        completed_epoch=spec.epochs,
        best_validation_dice=best_validation_dice,
        history=history,
        cumulative_training_seconds=total_seconds,
    )
    _save_torch_atomic(checkpoint_final, final_payload)
    export_loader = _dataloader(
        spec.validation_cases, recipe, training=False, generator=generator, workers=0
    )
    validation_mean, validation_regions, inference_seconds = validate_model(
        model,
        export_loader,
        recipe,
        device=device,
        export_cases={case.case_id: case for case in spec.validation_cases},
        prediction_dir=predictions,
        native_probability_dir=native_probabilities,
        canonical_probability_dir=canonical_probabilities,
        overwrite=resume,
    )
    prediction_summary = {
        "valid": True,
        "fold": spec.fold,
        "case_count": len(spec.validation_cases),
        "case_ids": [case.case_id for case in spec.validation_cases],
        "validation_mean_dice": validation_mean,
        "validation_dice_native_order": validation_regions,
        "inference_total_seconds": float(np.sum(inference_seconds)),
        "inference_mean_seconds_per_case": float(np.mean(inference_seconds)),
        "prediction_dir": str(predictions),
        "native_probability_dir": str(native_probabilities),
        "canonical_probability_dir": str(canonical_probabilities),
        "native_channel_order": list(SEGRESNET_REGION_ORDER),
        "canonical_channel_order": ["ET", "TC", "WT"],
        "probability_conversion": SEGRESNET_REGION_CONVERSION,
        "hard_label_rule": (
            "threshold 0.5; nested closure ET subset TC subset WT; "
            "ET=3, TC-only=1, WT-only=2"
        ),
    }
    write_json_atomic(output / "validation_summary.json", prediction_summary)
    return {
        "complete": True,
        "checkpoint_final": str(checkpoint_final),
        "runtime": runtime,
        "validation": prediction_summary,
    }


def config_hash(
    recipe: SegResNetRecipe,
    *,
    experiment_id: str,
    fold: int,
    epochs: int,
    train_ids: Sequence[str],
    validation_ids: Sequence[str],
    smoke: bool,
) -> str:
    payload = {
        "recipe": json.loads(normalized_recipe_json(recipe)),
        "experiment_id": experiment_id,
        "fold": fold,
        "epochs": epochs,
        "train_ids": list(train_ids),
        "validation_ids": list(validation_ids),
        "smoke": smoke,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def fold_manifest(spec: FoldRunSpec, recipe: SegResNetRecipe) -> dict[str, Any]:
    return {
        "schema": "glioma_model_fold_manifest_v1",
        "backend": "segresnet",
        "model_id": recipe.model_id,
        "experiment_id": spec.experiment_id,
        "fold": spec.fold,
        "target_epochs": spec.epochs,
        "smoke": spec.smoke,
        "train_case_count": len(spec.train_cases),
        "validation_case_count": len(spec.validation_cases),
        "train_case_ids": [case.case_id for case in spec.train_cases],
        "validation_case_ids": [case.case_id for case in spec.validation_cases],
        "config_sha256": spec.config_hash,
        "model_config_sha256": recipe.source_sha256,
        "split_sha256": spec.split_sha256,
        "output_dir": str(spec.output_dir.resolve()),
        "prediction_dir": str((spec.output_dir / "predictions").resolve()),
        "native_probability_dir": str(
            (spec.output_dir / "probabilities" / "native").resolve()
        ),
        "canonical_probability_dir": str(
            (spec.output_dir / "probabilities" / "canonical").resolve()
        ),
    }


__all__ = [
    "CasePaths",
    "FoldRunSpec",
    "build_model",
    "build_transforms",
    "config_hash",
    "fold_manifest",
    "memory_preflight",
    "run_fold",
    "validate_model",
]
