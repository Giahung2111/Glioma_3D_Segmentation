"""Strict, source-pinned MedNeXt v1 recipe configuration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from glioma_seg.utils.hashing import sha256_file

EXPECTED_MEDNEXT_COMMIT = "0b78ed869fbd1cc2fd38754d2f8519f1b72d43ba"
EXPECTED_TRAINER = "nnUNetTrainerV2_MedNeXt_S_kernel3"
EXPECTED_PLANNER = "ExperimentPlanner3D_v21_customTargetSpacing_1x1x1"
EXPECTED_PLANS = "nnUNetPlansv2.1_trgSp_1x1x1"
EXPECTED_DATA_IDENTIFIER = "nnUNetData_plans_v2.1_trgSp_1x1x1"

_EXPECTED: dict[str, Any] = {
    "schema": "glioma_mednext_recipe_v1",
    "backend": "mednext",
    "model_id": "mednext_v1_s_kernel3",
    "upstream": {
        "repository": "https://github.com/MIC-DKFZ/MedNeXt.git",
        "commit": EXPECTED_MEDNEXT_COMMIT,
        "package": "mednextv1",
        "package_version": "1.7.0",
    },
    "framework": {
        "name": "nnU-Net v1 (MedNeXt fork)",
        "task_full": "Task501_BraTS2023GLI",
        "task_smoke": "Task951_BraTS2023GLISmoke",
        "network": "3d_fullres",
        "trainer": EXPECTED_TRAINER,
        "planner_3d": EXPECTED_PLANNER,
        "plans_identifier": EXPECTED_PLANS,
        "data_identifier": EXPECTED_DATA_IDENTIFIER,
    },
    "architecture": {
        "variant": "S",
        "kernel_size": 3,
        "in_channels": 4,
        "classes_including_background": 4,
        "base_channels": 32,
        "expansion_ratio": 2,
        "block_counts": [2, 2, 2, 2, 2, 2, 2, 2, 2],
        "deep_supervision": True,
        "residual_blocks": True,
        "residual_up_down": True,
    },
    "planning": {
        "target_spacing_mm": [1.0, 1.0, 1.0],
        "patch_size": [128, 128, 128],
    },
    "training": {
        "original_recipe_epochs": 1000,
        "requested_epochs": 100,
        "classification": (
            "compute-limited comparison; architecture, loss, augmentation, optimizer, "
            "and poly schedule unchanged"
        ),
        "optimizer": "AdamW",
        "initial_learning_rate": 0.001,
        "optimizer_epsilon": 0.0001,
        "weight_decay": 0.00003,
        "lr_schedule": "polynomial power 0.9 over requested duration",
        "batches_per_epoch": 250,
        "validation_batches_per_epoch": 50,
        "mixed_precision": True,
        "deterministic": False,
        "checkpoint_every_completed_epochs": 1,
    },
    "inference": {
        "checkpoint": "final",
        "sliding_window": True,
        "step_size": 0.5,
        "gaussian_weighting": True,
        "tta": False,
        "fold_specific_postprocessing": False,
        "retain_native_softmax": True,
    },
    "probabilities": {
        "native_channel_order": ["background", "NCR", "ED", "ET"],
        "canonical_channel_order": ["ET", "TC", "WT"],
        "conversion": "ET=p(ET);TC=p(NCR)+p(ET);WT=p(NCR)+p(ED)+p(ET)",
    },
}


@dataclass(frozen=True, slots=True)
class MedNeXtRecipe:
    """Validated official architecture/recipe plus the explicit 100-epoch duration."""

    payload: dict[str, Any]
    source_path: Path
    source_sha256: str

    @property
    def model_id(self) -> str:
        return str(self.payload["model_id"])

    @property
    def epochs(self) -> int:
        return int(self.payload["training"]["requested_epochs"])

    @property
    def original_recipe_epochs(self) -> int:
        return int(self.payload["training"]["original_recipe_epochs"])

    @property
    def task_full(self) -> str:
        return str(self.payload["framework"]["task_full"])

    @property
    def task_smoke(self) -> str:
        return str(self.payload["framework"]["task_smoke"])

    @property
    def patch_size(self) -> tuple[int, int, int]:
        values = self.payload["planning"]["patch_size"]
        return int(values[0]), int(values[1]), int(values[2])


def _first_difference(expected: Any, actual: Any, path: str = "recipe") -> str | None:
    if isinstance(expected, dict) and isinstance(actual, dict):
        if set(expected) != set(actual):
            return f"{path} keys: expected={sorted(expected)}, actual={sorted(actual)}"
        for key in expected:
            difference = _first_difference(expected[key], actual[key], f"{path}.{key}")
            if difference is not None:
                return difference
        return None
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return f"{path} length: expected={len(expected)}, actual={len(actual)}"
        for index, (wanted, found) in enumerate(zip(expected, actual, strict=True)):
            difference = _first_difference(wanted, found, f"{path}[{index}]")
            if difference is not None:
                return difference
        return None
    if type(expected) is not type(actual) or expected != actual:
        return f"{path}: expected={expected!r}, actual={actual!r}"
    return None


def load_recipe(path: str | Path) -> MedNeXtRecipe:
    """Load the recipe and fail closed on any unreviewed parameter change."""

    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"MedNeXt recipe is missing: {source}")
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("MedNeXt recipe must be a YAML mapping")
    difference = _first_difference(_EXPECTED, payload)
    if difference is not None:
        raise ValueError(
            "MedNeXt recipe differs from the reviewed source-pinned contract: "
            f"{difference}"
        )
    normalized = json.loads(json.dumps(payload, sort_keys=True))
    return MedNeXtRecipe(normalized, source, sha256_file(source))


__all__ = [
    "EXPECTED_DATA_IDENTIFIER",
    "EXPECTED_MEDNEXT_COMMIT",
    "EXPECTED_PLANNER",
    "EXPECTED_PLANS",
    "EXPECTED_TRAINER",
    "MedNeXtRecipe",
    "load_recipe",
]
