"""Project-owned launcher around the unchanged official MedNeXt v1 trainer."""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import importlib
import json
import math
import os
import pickle
import shutil
import tempfile
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

from glioma_seg.backends.mednext.compat import apply_windows_path_compatibility
from glioma_seg.backends.mednext.config import MedNeXtRecipe
from glioma_seg.ensembles.canonical_probabilities import (
    MEDNEXT_MULTICLASS_CONVERSION,
    MEDNEXT_MULTICLASS_ORDER,
    mednext_multiclass_to_canonical,
    write_canonical_probability_npz,
)
from glioma_seg.monitoring.gpu_monitor import GPUMonitor
from glioma_seg.monitoring.timing import write_json_atomic
from glioma_seg.utils.hashing import sha256_file


@dataclass(frozen=True, slots=True)
class MedNeXtFoldSpec:
    experiment_id: str
    fold: int
    epochs: int
    task_name: str
    preprocessed_task_directory: Path
    output_root: Path
    train_case_ids: tuple[str, ...]
    validation_case_ids: tuple[str, ...]
    reference_labels: Mapping[str, Path]
    split_sha256: str
    config_sha256: str
    smoke: bool

    @property
    def output_directory(self) -> Path:
        return self.output_root / f"fold_{self.fold}"


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def normalized_spec_hash(
    recipe: MedNeXtRecipe,
    *,
    experiment_id: str,
    fold: int,
    epochs: int,
    task_name: str,
    train_case_ids: Sequence[str],
    validation_case_ids: Sequence[str],
    split_sha256: str,
    smoke: bool,
) -> str:
    payload = {
        "recipe_sha256": recipe.source_sha256,
        "experiment_id": experiment_id,
        "fold": fold,
        "epochs": epochs,
        "task_name": task_name,
        "train_case_ids": list(train_case_ids),
        "validation_case_ids": list(validation_case_ids),
        "split_sha256": split_sha256,
        "smoke": smoke,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def fold_manifest(spec: MedNeXtFoldSpec, recipe: MedNeXtRecipe) -> dict[str, Any]:
    return {
        "schema": "glioma_model_fold_manifest_v1",
        "backend": "mednext",
        "model_id": recipe.model_id,
        "experiment_id": spec.experiment_id,
        "fold": spec.fold,
        "target_epochs": spec.epochs,
        "smoke": spec.smoke,
        "task_name": spec.task_name,
        "train_case_count": len(spec.train_case_ids),
        "validation_case_count": len(spec.validation_case_ids),
        "train_case_ids": list(spec.train_case_ids),
        "validation_case_ids": list(spec.validation_case_ids),
        "config_sha256": spec.config_sha256,
        "model_config_sha256": recipe.source_sha256,
        "split_sha256": spec.split_sha256,
        "output_dir": str(spec.output_directory.resolve()),
        "prediction_dir": str((spec.output_directory / "predictions").resolve()),
        "native_probability_dir": str(
            (spec.output_directory / "probabilities" / "native").resolve()
        ),
        "canonical_probability_dir": str(
            (spec.output_directory / "probabilities" / "canonical").resolve()
        ),
        "official_trainer": "nnUNetTrainerV2_MedNeXt_S_kernel3",
        "official_plans_identifier": "nnUNetPlansv2.1_trgSp_1x1x1",
        "project_duration_override_only": True,
    }


def write_fold_manifest(spec: MedNeXtFoldSpec, recipe: MedNeXtRecipe) -> Path:
    path = spec.output_directory / "fold_manifest.json"
    payload = fold_manifest(spec, recipe)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise ValueError(f"MedNeXt fold manifest collision: {path}")
    else:
        write_json_atomic(path, payload)
    return path


def _load_plans(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Official MedNeXt plan is missing: {path}")
    with path.open("rb") as handle:
        value = pickle.load(handle)  # noqa: S301 - locally generated official nnU-Net plan
    if not isinstance(value, dict):
        raise ValueError(f"MedNeXt plan must be a mapping: {path}")
    return value


def audit_preprocessed_task(
    task_directory: Path,
    recipe: MedNeXtRecipe,
    *,
    expected_case_ids: Sequence[str],
) -> dict[str, Any]:
    """Fail closed unless the official 1 mm / 128-cube plan and cases are complete."""

    plan_path = task_directory / "nnUNetPlansv2.1_trgSp_1x1x1_plans_3D.pkl"
    plans = _load_plans(plan_path)
    if plans.get("data_identifier") != "nnUNetData_plans_v2.1_trgSp_1x1x1":
        raise ValueError(f"Unexpected MedNeXt data identifier: {plans.get('data_identifier')!r}")
    stages = plans.get("plans_per_stage")
    if not isinstance(stages, dict) or not stages:
        raise ValueError("MedNeXt plan contains no 3D stage")
    stage_key = max(stages, key=lambda value: int(value))
    stage_index = int(stage_key)
    stage = stages[stage_key]
    patch = tuple(int(value) for value in stage["patch_size"])
    spacing = tuple(float(value) for value in stage["current_spacing"])
    if patch != recipe.patch_size:
        raise ValueError(f"Official MedNeXt patch must be {recipe.patch_size}, got {patch}")
    if not np.allclose(spacing, (1.0, 1.0, 1.0), rtol=0.0, atol=1e-8):
        raise ValueError(f"Official MedNeXt target spacing must be 1 mm, got {spacing}")
    data_dir = task_directory / f"{plans['data_identifier']}_stage{stage_index}"
    expected = set(expected_case_ids)
    found = {path.stem for path in data_dir.glob("*.npz")}
    if found != expected:
        raise ValueError(
            "MedNeXt preprocessed case inventory mismatch: "
            f"missing={sorted(expected - found)[:5]}, extra={sorted(found - expected)[:5]}"
        )
    split_path = task_directory / "splits_final.pkl"
    if not split_path.is_file():
        raise FileNotFoundError(f"Project-owned native split is missing: {split_path}")
    return {
        "valid": True,
        "task_directory": str(task_directory.resolve()),
        "plans_file": str(plan_path.resolve()),
        "plans_sha256": sha256_file(plan_path),
        "stage": stage_index,
        "patch_size": list(patch),
        "target_spacing_mm": list(spacing),
        "batch_size_from_official_planner": int(stage["batch_size"]),
        "batch_dice_from_official_configuration_rule": len(stages) > 1,
        "data_identifier": plans["data_identifier"],
        "case_count": len(found),
        "split_file": str(split_path.resolve()),
    }


def _trainer_class() -> type[Any]:
    """Build a Code-owned subclass while importing the official class unchanged."""

    apply_windows_path_compatibility()
    module = importlib.import_module(
        "nnunet_mednext.training.network_training.MedNeXt.nnUNetTrainerV2_MedNeXt"
    )
    official_class = module.nnUNetTrainerV2_MedNeXt_S_kernel3

    class ProjectComputeLimitedMedNeXt(official_class):  # type: ignore[misc,valid-type]
        def __init__(
            self,
            *args: Any,
            project_target_epochs: int,
            project_checkpoint_owner: Mapping[str, Any],
            project_stop_after_epoch: int | None = None,
            **kwargs: Any,
        ) -> None:
            super().__init__(*args, **kwargs)
            if project_target_epochs < 1:
                raise ValueError("project_target_epochs must be positive")
            if project_stop_after_epoch is not None and not (
                1 <= project_stop_after_epoch < project_target_epochs
            ):
                raise ValueError("stop-after epoch must be below the target epoch count")
            self.max_num_epochs = project_target_epochs
            self.save_every = 1
            self.save_latest_only = True
            self.save_intermediate_checkpoints = True
            self._project_target_epochs = project_target_epochs
            self._project_checkpoint_owner = dict(project_checkpoint_owner)
            self._project_stop_after_epoch = project_stop_after_epoch
            self._project_epoch_durations: list[float] = []
            self._project_epoch_started = time.monotonic()

        def save_checkpoint(self, fname: str, save_optimizer: bool = True) -> None:
            """Embed experiment ownership without changing upstream loader semantics."""

            super().save_checkpoint(fname, save_optimizer)
            import torch

            checkpoint = torch.load(fname, map_location="cpu", weights_only=False)
            checkpoint["glioma_project_owner"] = dict(self._project_checkpoint_owner)
            temporary = f"{fname}.project-owner.tmp"
            try:
                with open(temporary, "wb") as handle:
                    torch.save(checkpoint, handle)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, fname)
            finally:
                Path(temporary).unlink(missing_ok=True)

        def run_iteration(self, *args: Any, **kwargs: Any) -> Any:
            loss = super().run_iteration(*args, **kwargs)
            if not np.all(np.isfinite(np.asarray(loss))):
                raise FloatingPointError(
                    f"NaN/Inf MedNeXt loss at completed epoch count {self.epoch}"
                )
            return loss

        def on_epoch_end(self) -> bool:
            official_continue = bool(super().on_epoch_end())
            completed_epoch = int(self.epoch) + 1
            elapsed = time.monotonic() - self._project_epoch_started
            self._project_epoch_durations.append(elapsed)
            self._project_epoch_started = time.monotonic()
            train_loss = float(self.all_tr_losses[-1])
            validation_loss = float(self.all_val_losses[-1])
            if not math.isfinite(train_loss) or not math.isfinite(validation_loss):
                raise FloatingPointError(
                    f"NaN/Inf MedNeXt epoch summary at epoch {completed_epoch}"
                )
            eta = float(np.mean(self._project_epoch_durations)) * (
                self._project_target_epochs - completed_epoch
            )
            print(
                f"[MEDNEXT] Epoch={completed_epoch}/{self._project_target_epochs} "
                f"Progress={completed_epoch/self._project_target_epochs:.1%} "
                f"TrainLoss={train_loss:.6f} ValLoss={validation_loss:.6f} "
                f"EpochTime={elapsed:.1f}s ETA={eta/3600:.2f}h",
                flush=True,
            )
            if self._project_stop_after_epoch == completed_epoch:
                self.save_checkpoint(
                    os.path.join(self.output_folder, "model_resume_checkpoint.model")
                )
                return False
            return official_continue

        def run_training(self) -> Any:
            interrupted = self._project_stop_after_epoch is not None
            original_save_final = bool(self.save_final_checkpoint)  # type: ignore[has-type]
            if interrupted:
                self.save_final_checkpoint = False
            try:
                return super().run_training()
            finally:
                self.save_final_checkpoint = original_save_final

    ProjectComputeLimitedMedNeXt.__name__ = "ProjectComputeLimitedMedNeXt"
    return ProjectComputeLimitedMedNeXt


def _instantiate_trainer(
    spec: MedNeXtFoldSpec,
    recipe: MedNeXtRecipe,
    *,
    stop_after_epoch: int | None,
) -> Any:
    plans_file = (
        spec.preprocessed_task_directory
        / "nnUNetPlansv2.1_trgSp_1x1x1_plans_3D.pkl"
    )
    plans = _load_plans(plans_file)
    stages = plans["plans_per_stage"]
    stage_key = max(stages, key=lambda value: int(value))
    stage = int(stage_key)
    batch_dice = len(stages) > 1
    trainer_type = _trainer_class()
    return trainer_type(
        str(plans_file),
        spec.fold,
        output_folder=str(spec.output_root),
        dataset_directory=str(spec.preprocessed_task_directory),
        batch_dice=batch_dice,
        stage=stage,
        unpack_data=True,
        deterministic=bool(recipe.payload["training"]["deterministic"]),
        fp16=bool(recipe.payload["training"]["mixed_precision"]),
        project_target_epochs=spec.epochs,
        project_checkpoint_owner=_checkpoint_owner(spec, recipe),
        project_stop_after_epoch=stop_after_epoch,
    )


def _checkpoint_owner(spec: MedNeXtFoldSpec, recipe: MedNeXtRecipe) -> dict[str, Any]:
    return {
        "schema": "glioma_mednext_checkpoint_owner_v1",
        "backend": "mednext",
        "model_id": recipe.model_id,
        "experiment_id": spec.experiment_id,
        "fold": spec.fold,
        "target_epochs": spec.epochs,
        "config_sha256": spec.config_sha256,
        "split_sha256": spec.split_sha256,
        "official_trainer": recipe.payload["framework"]["trainer"],
        "official_plans_identifier": recipe.payload["framework"]["plans_identifier"],
    }


def _write_or_verify_owner(spec: MedNeXtFoldSpec, recipe: MedNeXtRecipe) -> Path:
    path = spec.output_directory / "checkpoint_owner.json"
    expected = _checkpoint_owner(spec, recipe)
    if path.exists():
        found = json.loads(path.read_text(encoding="utf-8"))
        if found != expected:
            raise ValueError(f"MedNeXt checkpoint ownership mismatch: {path}")
    else:
        write_json_atomic(path, expected)
    return path


def _verify_native_checkpoint_owner(path: Path, expected: Mapping[str, Any]) -> int:
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("glioma_project_owner") != dict(expected):
        raise ValueError(f"MedNeXt checkpoint embedded ownership mismatch: {path}")
    completed_epoch = checkpoint.get("epoch")
    if isinstance(completed_epoch, bool) or not isinstance(completed_epoch, int):
        raise ValueError(f"MedNeXt checkpoint epoch is invalid: {path}")
    return completed_epoch


def _atomic_copy(source: Path, destination: Path, *, overwrite: bool) -> None:
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite prediction artifact: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _link_native_probability(source: Path, destination: Path, *, overwrite: bool) -> None:
    """Retain the exact upstream float16 NPZ without storing a second array copy."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        same = False
        try:
            same = os.path.samefile(source, destination)
        except OSError:
            same = False
        if not same and sha256_file(source) != sha256_file(destination):
            if not overwrite:
                raise FileExistsError(
                    f"Conflicting native MedNeXt probabilities: {destination}"
                )
            destination.unlink()
        else:
            return
    try:
        os.link(source, destination)
    except OSError:
        _atomic_copy(source, destination, overwrite=overwrite)


def _native_xyz(softmax: np.ndarray, reference_shape: tuple[int, ...]) -> np.ndarray:
    if softmax.ndim != 4 or softmax.shape[0] != 4:
        raise ValueError(f"MedNeXt softmax must have shape (4,X,Y,Z), got {softmax.shape}")
    spatial = tuple(int(value) for value in softmax.shape[1:])
    if spatial == reference_shape:
        return np.asarray(softmax, dtype=np.float32)
    if spatial == tuple(reversed(reference_shape)):
        return np.asarray(softmax.transpose(0, 3, 2, 1), dtype=np.float32)
    raise ValueError(
        f"MedNeXt softmax geometry {spatial} cannot map to reference {reference_shape}"
    )


def normalize_validation_artifacts(
    spec: MedNeXtFoldSpec,
    *,
    overwrite: bool,
) -> dict[str, Any]:
    """Publish raw masks and native/canonical probabilities in common contracts."""

    upstream = spec.output_directory / "validation_raw"
    predictions = spec.output_directory / "predictions"
    native_dir = spec.output_directory / "probabilities" / "native"
    canonical_dir = spec.output_directory / "probabilities" / "canonical"
    missing: list[str] = []
    for case_id in spec.validation_case_ids:
        if not (upstream / f"{case_id}.nii.gz").is_file() or not (
            upstream / f"{case_id}.npz"
        ).is_file():
            missing.append(case_id)
    if missing:
        raise FileNotFoundError(f"Official MedNeXt validation artifacts missing: {missing[:5]}")
    for index, case_id in enumerate(spec.validation_case_ids, start=1):
        reference = spec.reference_labels[case_id]
        reference_image: Any = nib.load(str(reference))
        upstream_mask = upstream / f"{case_id}.nii.gz"
        mask_image: Any = nib.load(str(upstream_mask))
        if mask_image.shape != reference_image.shape or not np.allclose(
            mask_image.affine, reference_image.affine, rtol=0.0, atol=1e-4
        ):
            raise ValueError(f"Official MedNeXt mask geometry mismatch for {case_id}")
        labels = set(int(value) for value in np.unique(np.asanyarray(mask_image.dataobj)))
        if not labels.issubset({0, 1, 2, 3}):
            raise ValueError(f"Invalid MedNeXt labels for {case_id}: {sorted(labels)}")
        _atomic_copy(
            upstream_mask,
            predictions / f"{case_id}.nii.gz",
            overwrite=overwrite,
        )
        upstream_npz = upstream / f"{case_id}.npz"
        with np.load(upstream_npz, allow_pickle=False) as archive:
            if set(archive.files) != {"softmax"}:
                raise ValueError(
                    f"Unexpected official MedNeXt NPZ keys for {case_id}: {archive.files}"
                )
            native = _native_xyz(
                np.asarray(archive["softmax"]), tuple(int(v) for v in reference_image.shape)
            )
        canonical = mednext_multiclass_to_canonical(native, simplex_atol=2e-3)
        _link_native_probability(
            upstream_npz,
            native_dir / f"{case_id}.npz",
            overwrite=overwrite,
        )
        write_canonical_probability_npz(
            canonical_dir / f"{case_id}.npz",
            case_id=case_id,
            probabilities=canonical,
            native_channel_order=MEDNEXT_MULTICLASS_ORDER,
            conversion=MEDNEXT_MULTICLASS_CONVERSION,
            reference_nifti=reference,
            overwrite=overwrite,
        )
        print(
            f"[MEDNEXT-EXPORT] {index}/{len(spec.validation_case_ids)} {case_id}",
            flush=True,
        )
    write_json_atomic(
        native_dir / "native_probability_contract.json",
        {
            "schema": "glioma_mednext_native_probabilities_v1",
            "source": "unchanged official validation_raw NPZ (hardlink when supported)",
            "array_key": "softmax",
            "storage_dtype": "float16",
            "native_channel_order": list(MEDNEXT_MULTICLASS_ORDER),
            "activation": "softmax",
            "spatial_axis_order": "upstream SimpleITK ZYX; canonical export is nibabel XYZ",
            "float16_simplex_tolerance": 0.002,
        },
    )
    return {
        "valid": True,
        "case_count": len(spec.validation_case_ids),
        "case_ids": list(spec.validation_case_ids),
        "prediction_dir": str(predictions.resolve()),
        "native_probability_dir": str(native_dir.resolve()),
        "canonical_probability_dir": str(canonical_dir.resolve()),
        "native_channel_order": list(MEDNEXT_MULTICLASS_ORDER),
        "canonical_channel_order": ["ET", "TC", "WT"],
        "probability_conversion": MEDNEXT_MULTICLASS_CONVERSION,
        "upstream_softmax_storage_dtype": "float16",
        "canonical_storage_dtype": "float32",
        "upstream_float16_simplex_tolerance": 0.002,
        "tta_state": "OFF",
        "postprocessing": "OFF",
    }


def _history(trainer: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    durations = list(getattr(trainer, "_project_epoch_durations", []))
    for index, (train_loss, validation_loss) in enumerate(
        zip(trainer.all_tr_losses, trainer.all_val_losses, strict=True), start=1
    ):
        metric = trainer.all_val_eval_metrics[index - 1] if index <= len(
            trainer.all_val_eval_metrics
        ) else None
        rows.append(
            {
                "epoch": index,
                "train_loss": float(train_loss),
                "validation_loss": float(validation_loss),
                "validation_pseudo_dice": float(metric) if metric is not None else None,
                "epoch_seconds_current_process": (
                    float(durations[index - 1]) if index <= len(durations) else None
                ),
            }
        )
    return rows


def run_fold(
    recipe: MedNeXtRecipe,
    spec: MedNeXtFoldSpec,
    *,
    resume: bool,
    stop_after_epoch: int | None = None,
) -> dict[str, Any]:
    """Run official MedNeXt training/validation with owned safety metadata."""

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for MedNeXt training")
    if spec.epochs < 1 or spec.epochs > recipe.epochs:
        raise ValueError(f"epochs must be in 1..{recipe.epochs}, got {spec.epochs}")
    if not spec.smoke and spec.epochs != recipe.epochs:
        raise ValueError(f"Full MedNeXt CV must use exactly {recipe.epochs} epochs")
    write_fold_manifest(spec, recipe)
    _write_or_verify_owner(spec, recipe)
    fold_dir = spec.output_directory
    checkpoints = (
        fold_dir / "model_final_checkpoint.model",
        fold_dir / "model_latest.model",
        fold_dir / "model_resume_checkpoint.model",
    )
    if not resume and any(path.exists() for path in checkpoints):
        raise FileExistsError(
            f"MedNeXt fold already contains checkpoints; use verified resume: {fold_dir}"
        )
    trainer = _instantiate_trainer(spec, recipe, stop_after_epoch=stop_after_epoch)
    previous_runtime: dict[str, Any] = {}
    runtime_path = fold_dir / "runtime.json"
    if resume and runtime_path.is_file():
        loaded_runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
        if isinstance(loaded_runtime, dict):
            previous_runtime = loaded_runtime
    segment_stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    segment_index = len(list(fold_dir.glob("gpu_samples_segment_*.csv"))) + 1
    gpu_segment = (
        fold_dir / f"gpu_samples_segment_invocation_{segment_index:02d}_{segment_stamp}.csv"
    )
    monitor = GPUMonitor(gpu_segment, interval_seconds=2.0).start()
    heartbeat_stop = threading.Event()
    heartbeat_phase = {"name": "initializing"}

    def print_heartbeat() -> None:
        heartbeat_started = time.monotonic()
        while not heartbeat_stop.wait(30.0):
            snapshot = monitor.latest
            gpu_text = "GPU=waiting-for-first-sample"
            if snapshot is not None:
                power = "N/A" if snapshot.power_w is None else f"{snapshot.power_w:.0f}W"
                gpu_text = (
                    f"GPU={snapshot.gpu_utilization_percent:.0f}% "
                    f"VRAM={snapshot.memory_used_mb:.0f}/{snapshot.memory_total_mb:.0f}MiB "
                    f"Temp={snapshot.temperature_c:.0f}C Power={power}"
                )
            completed = len(getattr(trainer, "all_tr_losses", []))
            print(
                f"[MEDNEXT MONITOR] Experiment={spec.experiment_id} Fold={spec.fold} "
                f"Phase={heartbeat_phase['name']} CompletedEpochs={completed}/{spec.epochs} "
                f"Elapsed={(time.monotonic() - heartbeat_started) / 60.0:.1f}m {gpu_text}",
                flush=True,
            )

    heartbeat = threading.Thread(
        target=print_heartbeat,
        name="mednext-terminal-heartbeat",
        daemon=True,
    )
    heartbeat.start()
    started_at = _utc_now()
    training_started: float | None = None
    current_training_seconds = 0.0
    inference_seconds = 0.0
    try:
        trainer.initialize(True)
        if set(trainer.dataset_tr) != set(spec.train_case_ids) or set(trainer.dataset_val) != set(
            spec.validation_case_ids
        ):
            raise ValueError("Official MedNeXt trainer did not load the declared project split")
        if resume:
            resume_candidates = (
                fold_dir / "model_resume_checkpoint.model",
                fold_dir / "model_latest.model",
                fold_dir / "model_final_checkpoint.model",
            )
            resume_path = next((path for path in resume_candidates if path.is_file()), None)
            if resume_path is None:
                raise FileNotFoundError(
                    f"No owner-matched MedNeXt checkpoint to resume: {fold_dir}"
                )
            _verify_native_checkpoint_owner(
                resume_path, _checkpoint_owner(spec, recipe)
            )
            trainer.load_checkpoint(str(resume_path), train=True)
        heartbeat_phase["name"] = "training"
        training_started = time.monotonic()
        trainer.run_training()
        current_training_seconds = time.monotonic() - training_started
        history = _history(trainer)
        write_json_atomic(fold_dir / "train_history.json", history)
        interrupted = stop_after_epoch is not None
        if interrupted:
            return {
                "valid": True,
                "complete": False,
                "safe_to_resume": True,
                "completed_epochs": len(history),
                "resume_checkpoint": str(
                    (fold_dir / "model_resume_checkpoint.model").resolve()
                ),
            }
        final_checkpoint = fold_dir / "model_final_checkpoint.model"
        if not final_checkpoint.is_file() or len(history) != spec.epochs:
            raise RuntimeError(
                f"MedNeXt training ended at {len(history)}/{spec.epochs} epochs "
                "without final checkpoint"
            )
        heartbeat_phase["name"] = "validation-inference"
        inference_started = time.monotonic()
        trainer.network.eval()
        trainer.validate(
            do_mirroring=False,
            use_sliding_window=True,
            step_size=float(recipe.payload["inference"]["step_size"]),
            save_softmax=True,
            use_gaussian=True,
            overwrite=resume,
            validation_folder_name="validation_raw",
            run_postprocessing_on_folds=False,
        )
        inference_seconds = time.monotonic() - inference_started
        validation = normalize_validation_artifacts(spec, overwrite=resume)
        validation.update(
            {
                "fold": spec.fold,
                "inference_total_seconds": inference_seconds,
                "inference_mean_seconds_per_case": inference_seconds
                / len(spec.validation_case_ids),
                "timing_scope": (
                    "official sliding-window prediction, NIfTI/NPZ export, and native "
                    "validation evaluation"
                ),
            }
        )
        write_json_atomic(fold_dir / "validation_summary.json", validation)
    finally:
        heartbeat_stop.set()
        heartbeat.join(timeout=5.0)
        if training_started is not None and current_training_seconds == 0.0:
            current_training_seconds = time.monotonic() - training_started
        segment_summary = monitor.stop().to_dict()
        write_json_atomic(gpu_segment.with_suffix(".summary.json"), segment_summary)
        _aggregate_gpu_telemetry(
            fold_dir,
            merged_csv=fold_dir / "gpu_samples.csv",
            summary_json=fold_dir / "gpu_summary.json",
        )
        previous_training_seconds = float(previous_runtime.get("total_seconds", 0.0))
        total_seconds = previous_training_seconds + current_training_seconds
        previous_durations = [
            float(value)
            for value in previous_runtime.get("epoch_durations_seconds", [])
            if isinstance(value, (int, float)) and math.isfinite(float(value))
        ]
        current_durations = [
            float(value)
            for value in getattr(trainer, "_project_epoch_durations", [])
            if math.isfinite(float(value))
        ]
        epoch_durations = previous_durations + current_durations
        number_of_epochs = len(getattr(trainer, "all_tr_losses", []))
        runtime = {
            "stage": "mednext_fold_training",
            "fold": spec.fold,
            "started_at": started_at,
            "ended_at": _utc_now(),
            "total_seconds": total_seconds,
            "total_hours": total_seconds / 3600.0,
            "number_of_epochs": number_of_epochs,
            "target_epochs": spec.epochs,
            "average_seconds_per_epoch": (
                total_seconds / number_of_epochs if number_of_epochs else None
            ),
            "epoch_seconds_min": min(epoch_durations) if epoch_durations else None,
            "epoch_seconds_median": (
                float(np.median(epoch_durations)) if epoch_durations else None
            ),
            "epoch_seconds_max": max(epoch_durations) if epoch_durations else None,
            "epoch_durations_seconds": epoch_durations,
            "inference_seconds_excluded": inference_seconds,
            "stopped_for_resume_test": stop_after_epoch is not None,
        }
        write_json_atomic(runtime_path, runtime)
    return {
        "valid": True,
        "complete": True,
        "safe_to_resume": False,
        "checkpoint_final": str((fold_dir / "model_final_checkpoint.model").resolve()),
        "runtime": runtime,
        "validation": validation,
    }


def memory_preflight(
    recipe: MedNeXtRecipe,
    spec: MedNeXtFoldSpec,
    *,
    output_json: Path,
) -> dict[str, Any]:
    """Run one real official forward/loss/backward step on the exact smoke plan."""

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the MedNeXt memory preflight")
    trainer = _instantiate_trainer(spec, recipe, stop_after_epoch=None)
    result: dict[str, Any] = {
        "valid": False,
        "model": recipe.model_id,
        "fold": spec.fold,
        "patch_size": list(recipe.patch_size),
        "official_loss_and_augmentation": True,
        "started_at": _utc_now(),
    }
    try:
        trainer.initialize(True)
        trainer._maybe_init_amp()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        loss = trainer.run_iteration(trainer.tr_gen, True)
        torch.cuda.synchronize()
        total = int(torch.cuda.get_device_properties(0).total_memory)
        reserved = int(torch.cuda.max_memory_reserved())
        result.update(
            {
                "valid": True,
                "loss": float(loss),
                "peak_allocated_mb": int(torch.cuda.max_memory_allocated()) / 1024**2,
                "peak_reserved_mb": reserved / 1024**2,
                "total_vram_mb": total / 1024**2,
                "reserved_fraction": reserved / total,
                "dedicated_vram_fit": reserved <= total,
                "oversubscription_detected": reserved > total,
            }
        )
    except torch.cuda.OutOfMemoryError as exc:
        result.update({"error_type": "CUDAOutOfMemoryError", "error": str(exc)})
        raise RuntimeError(
            "The unchanged official MedNeXt S-k3 128x128x128 plan does not fit this GPU. "
            "No smaller patch or altered model was selected silently."
        ) from exc
    except Exception as exc:
        result.update({"error_type": type(exc).__name__, "error": str(exc)})
        raise
    finally:
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
        raise RuntimeError("No MedNeXt training GPU telemetry segment is available")

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
        raise RuntimeError("MedNeXt GPU telemetry segments contain no samples")
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


__all__ = [
    "MedNeXtFoldSpec",
    "audit_preprocessed_task",
    "fold_manifest",
    "memory_preflight",
    "normalize_validation_artifacts",
    "normalized_spec_hash",
    "run_fold",
    "write_fold_manifest",
]
