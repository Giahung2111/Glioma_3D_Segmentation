"""Pinned configuration validation for the official MONAI BraTS SegResNet bundle."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from glioma_seg.utils.hashing import sha256_file

EXPECTED_MONAI_COMMIT = "46a5272196a6c2590ca2589029eed8e4d56ff008"
EXPECTED_MODEL_ZOO_COMMIT = "b9e4d04bb2a073110bde9e5c05c9690241e938b6"


@dataclass(frozen=True)
class SegResNetRecipe:
    source_path: Path
    source_sha256: str
    epochs: int
    original_recipe_epochs: int
    batch_size: int
    crop_size: tuple[int, int, int]
    validation_roi_size: tuple[int, int, int]
    learning_rate: float
    weight_decay: float
    num_workers: int
    seed: int
    payload: dict[str, Any]

    @property
    def model_id(self) -> str:
        return str(self.payload["model_id"])


def _require_equal(value: Any, expected: Any, label: str) -> None:
    if value != expected:
        raise ValueError(f"Pinned SegResNet field {label} must be {expected!r}, got {value!r}")


def _require_mapping(payload: dict[str, Any], name: str) -> dict[str, Any]:
    value = payload.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"SegResNet config field {name!r} must be a mapping")
    return {str(key): item for key, item in value.items()}


def load_recipe(path: str | Path) -> SegResNetRecipe:
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"SegResNet model config is missing: {source}")
    loaded = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected a YAML mapping in {source}")
    payload: dict[str, Any] = {str(key): value for key, value in loaded.items()}
    upstream = _require_mapping(payload, "upstream")
    architecture = _require_mapping(payload, "architecture")
    training = _require_mapping(payload, "training")
    inference = _require_mapping(payload, "inference")
    model_input = _require_mapping(payload, "input")
    output = _require_mapping(payload, "output")

    _require_equal(upstream["commit"], EXPECTED_MONAI_COMMIT, "upstream.commit")
    _require_equal(upstream["version"], "1.4.0", "upstream.version")
    _require_equal(
        upstream["recipe_commit"], EXPECTED_MODEL_ZOO_COMMIT, "upstream.recipe_commit"
    )
    _require_equal(architecture["class"], "monai.networks.nets.SegResNet", "architecture.class")
    _require_equal(architecture["in_channels"], 4, "architecture.in_channels")
    _require_equal(architecture["out_channels"], 3, "architecture.out_channels")
    _require_equal(architecture["init_filters"], 16, "architecture.init_filters")
    _require_equal(architecture["blocks_down"], [1, 2, 2, 4], "architecture.blocks_down")
    _require_equal(architecture["blocks_up"], [1, 1, 1], "architecture.blocks_up")
    _require_equal(architecture["dropout_prob"], 0.2, "architecture.dropout_prob")
    _require_equal(
        model_input["native_channel_order"],
        ["T1c", "T1n", "T2w", "T2F"],
        "input.native_channel_order",
    )
    _require_equal(
        model_input["project_suffix_order"],
        ["0001", "0000", "0002", "0003"],
        "input.project_suffix_order",
    )
    _require_equal(output["native_channel_order"], ["TC", "WT", "ET"], "output.native")
    _require_equal(
        output["canonical_channel_order"], ["ET", "TC", "WT"], "output.canonical"
    )
    _require_equal(training["optimizer"], "Adam", "training.optimizer")
    _require_equal(training["loss"], "DiceLoss", "training.loss")
    _require_equal(training["scheduler"], "CosineAnnealingLR", "training.scheduler")
    _require_equal(training["batch_size"], 1, "training.batch_size")
    _require_equal(training["crop_size"], [224, 224, 144], "training.crop_size")
    _require_equal(training["original_recipe_epochs"], 300, "training.original_recipe_epochs")
    _require_equal(training["learning_rate"], 0.0001, "training.learning_rate")
    _require_equal(training["weight_decay"], 0.00001, "training.weight_decay")
    _require_equal(training["loss_smooth_nr"], 0.0, "training.loss_smooth_nr")
    _require_equal(training["loss_smooth_dr"], 0.00001, "training.loss_smooth_dr")
    _require_equal(training["squared_pred"], True, "training.squared_pred")
    _require_equal(training["sigmoid"], True, "training.sigmoid")
    _require_equal(training["amp"], True, "training.amp")
    _require_equal(training["validation_interval_epochs"], 1, "training.val_interval")
    _require_equal(training["num_workers"], 4, "training.num_workers")
    _require_equal(training["seed"], 12345, "training.seed")
    augmentation = _require_mapping(payload, "augmentation")
    _require_equal(augmentation["normalize_nonzero"], True, "augmentation.normalize_nonzero")
    _require_equal(
        augmentation["normalize_channel_wise"],
        True,
        "augmentation.normalize_channel_wise",
    )
    _require_equal(
        augmentation["flip_probability_per_axis"],
        0.5,
        "augmentation.flip_probability_per_axis",
    )
    _require_equal(
        augmentation["intensity_scale_factor"],
        0.1,
        "augmentation.intensity_scale_factor",
    )
    _require_equal(
        augmentation["intensity_scale_probability"],
        1.0,
        "augmentation.intensity_scale_probability",
    )
    _require_equal(
        augmentation["intensity_shift_offset"],
        0.1,
        "augmentation.intensity_shift_offset",
    )
    _require_equal(
        augmentation["intensity_shift_probability"],
        1.0,
        "augmentation.intensity_shift_probability",
    )
    _require_equal(inference["roi_size"], [240, 240, 160], "inference.roi_size")
    _require_equal(
        inference["sliding_window_batch_size"],
        1,
        "inference.sliding_window_batch_size",
    )
    _require_equal(inference["overlap"], 0.5, "inference.overlap")
    _require_equal(inference["threshold"], 0.5, "inference.threshold")
    epochs = int(training["epochs_per_fold"])
    if epochs != 100:
        raise ValueError("The full SegResNet compute-limited config must use exactly 100 epochs")
    crop_values = training["crop_size"]
    roi_values = inference["roi_size"]
    crop_size = (int(crop_values[0]), int(crop_values[1]), int(crop_values[2]))
    validation_roi_size = (int(roi_values[0]), int(roi_values[1]), int(roi_values[2]))
    return SegResNetRecipe(
        source_path=source,
        source_sha256=sha256_file(source),
        epochs=epochs,
        original_recipe_epochs=int(training["original_recipe_epochs"]),
        batch_size=int(training["batch_size"]),
        crop_size=crop_size,
        validation_roi_size=validation_roi_size,
        learning_rate=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        num_workers=int(training["num_workers"]),
        seed=int(training["seed"]),
        payload=payload,
    )


def normalized_recipe_json(recipe: SegResNetRecipe) -> str:
    return json.dumps(recipe.payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
