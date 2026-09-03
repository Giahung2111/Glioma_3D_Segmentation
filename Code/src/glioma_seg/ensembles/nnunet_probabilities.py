"""Safe loading of nnU-Net region probabilities.

BraTS is configured for overlapping-region training in ``dataset.json``.  In
that file the regions are deliberately declared as WT, TC, ET, and nnU-Net
stores ``probabilities`` channels in exactly that order.  Project ensemble
code, on the other hand, uses the canonical scientific order ET, TC, WT.  This
module is the single explicit conversion boundary between those conventions.

The companion nnU-Net ``.pkl`` file is intentionally not opened here.  Pickle
is executable input and is not required to combine already aligned arrays.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from glioma_seg.evaluation.regions import REGION_ORDER

from .base import ModelProbabilities

NNUNET_REGION_CHANNEL_ORDER: tuple[str, ...] = ("WT", "TC", "ET")
NNUNET_TO_CANONICAL_INDICES: tuple[int, ...] = (2, 1, 0)


def _load_dataset_json(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"nnU-Net dataset.json does not exist: {source}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {source}")
    return payload


def validate_brats_region_probability_contract(dataset_json: str | Path) -> dict[str, Any]:
    """Verify the dataset declaration that determines nnU-Net channel order."""

    source = Path(dataset_json).resolve()
    payload = _load_dataset_json(source)
    labels = payload.get("labels")
    if not isinstance(labels, dict):
        raise ValueError(f"dataset.json has no labels object: {source}")
    expected_names = (
        "background",
        "whole_tumor",
        "tumor_core",
        "enhancing_tumor",
    )
    if tuple(labels) != expected_names:
        raise ValueError(
            "Unsafe BraTS region declaration/order in dataset.json: "
            f"expected {expected_names}, got {tuple(labels)}"
        )
    if labels.get("background") != 0:
        raise ValueError("BraTS background must be label 0")
    if tuple(labels.get("whole_tumor", ())) != (1, 2, 3):
        raise ValueError("BraTS whole_tumor must be [1, 2, 3]")
    if tuple(labels.get("tumor_core", ())) != (1, 3):
        raise ValueError("BraTS tumor_core must be [1, 3]")
    if labels.get("enhancing_tumor") != 3:
        raise ValueError("BraTS enhancing_tumor must be label 3")
    if tuple(payload.get("regions_class_order", ())) != (2, 1, 3):
        raise ValueError("BraTS regions_class_order must be [2, 1, 3]")
    return {
        "dataset_json": str(source),
        "source_channel_order": list(NNUNET_REGION_CHANNEL_ORDER),
        "canonical_channel_order": list(REGION_ORDER),
        "canonical_reorder_indices": list(NNUNET_TO_CANONICAL_INDICES),
        "regions_class_order": [2, 1, 3],
    }


def load_nnunet_region_probabilities(
    npz_path: str | Path,
    *,
    dataset_json: str | Path,
    model_id: str | None = None,
) -> ModelProbabilities:
    """Load one official nnU-Net ``.npz`` and return canonical ET/TC/WT channels.

    ``allow_pickle=False`` prevents object-array deserialization.  The input is
    accepted only when it has the official ``probabilities`` key, three
    numeric spatial channels, finite values, and values in ``[0, 1]``.
    """

    source = Path(npz_path).resolve()
    if source.suffix.lower() != ".npz" or not source.is_file():
        raise FileNotFoundError(f"nnU-Net probability .npz does not exist: {source}")
    provenance = validate_brats_region_probability_contract(dataset_json)
    try:
        with np.load(source, allow_pickle=False) as archive:
            if tuple(archive.files) != ("probabilities",):
                raise ValueError(
                    f"Expected only the official 'probabilities' array in {source}; "
                    f"found {archive.files}"
                )
            raw = np.asarray(archive["probabilities"])
    except (OSError, ValueError) as exc:
        raise ValueError(f"Unable to load safe nnU-Net probabilities from {source}: {exc}") from exc

    if raw.ndim != 4 or raw.shape[0] != len(NNUNET_REGION_CHANNEL_ORDER):
        raise ValueError(
            "Expected nnU-Net BraTS probabilities shaped (3, X, Y, Z), "
            f"got {raw.shape} in {source}"
        )
    if not np.issubdtype(raw.dtype, np.number) or np.issubdtype(raw.dtype, np.complexfloating):
        raise ValueError(f"Probability array must have a real numeric dtype, got {raw.dtype}")
    if not np.all(np.isfinite(raw)):
        raise ValueError(f"Probability array contains NaN or infinity: {source}")
    if np.any(raw < 0) or np.any(raw > 1):
        raise ValueError(f"Probability values must lie in [0, 1]: {source}")

    canonical = np.asarray(raw[list(NNUNET_TO_CANONICAL_INDICES)], dtype=np.float32)
    identifier = model_id or source.stem
    return ModelProbabilities(
        model_id=identifier,
        probabilities=canonical,
        channel_names=REGION_ORDER,
        metadata={
            **provenance,
            "source_npz": str(source),
            "npz_key": "probabilities",
            "source_dtype": str(raw.dtype),
            "source_shape": list(raw.shape),
        },
    )
