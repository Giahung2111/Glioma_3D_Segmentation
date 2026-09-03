"""Lossless BraTS-to-nnU-Net-v1 adapter owned by the project.

The official MedNeXt repository consumes the old nnU-Net v1 task layout.  This
module creates that layout in a separate workspace without changing either the
source Dataset501 files or upstream MedNeXt.
"""

from __future__ import annotations

import json
import os
import pickle
import shutil
import tempfile
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from glioma_seg.monitoring.timing import write_json_atomic
from glioma_seg.utils.hashing import sha256_file

EXPECTED_CASE_COUNT = 1251
EXPECTED_FOLD_VALIDATION_COUNTS = (251, 250, 250, 250, 250)
MODALITY_SUFFIXES = ("0000", "0001", "0002", "0003")


@dataclass(frozen=True, slots=True)
class V1AdapterLayout:
    task_name: str
    task_directory: Path
    preprocessed_task_directory: Path
    case_ids: tuple[str, ...]
    splits: tuple[dict[str, tuple[str, ...]], ...]
    source_split_sha256: str
    smoke: bool


def inventory_dataset(
    raw_dataset: Path, *, expected_count: int = EXPECTED_CASE_COUNT
) -> dict[str, Path]:
    """Return exact label references after validating all four modality files."""

    images = raw_dataset / "imagesTr"
    labels = raw_dataset / "labelsTr"
    if not images.is_dir() or not labels.is_dir():
        raise FileNotFoundError(f"Converted BraTS Dataset501 is missing: {raw_dataset}")
    cases: dict[str, Path] = {}
    for label in sorted(labels.glob("*.nii.gz")):
        case_id = label.name.removesuffix(".nii.gz")
        if case_id in cases:
            raise ValueError(f"Duplicate BraTS case ID: {case_id}")
        missing = [
            suffix
            for suffix in MODALITY_SUFFIXES
            if not (images / f"{case_id}_{suffix}.nii.gz").is_file()
        ]
        if missing:
            raise FileNotFoundError(f"Modalities {missing} are missing for {case_id}")
        cases[case_id] = label.resolve()
    if len(cases) != expected_count:
        raise ValueError(f"Expected {expected_count} BraTS cases, found {len(cases)}")
    image_ids = {
        path.name.rsplit("_", 1)[0]
        for path in images.glob("*.nii.gz")
        if "_" in path.name
    }
    if image_ids != set(cases):
        raise ValueError("Image and label case inventories differ")
    return cases


def load_canonical_splits(
    path: Path,
    case_ids: set[str],
) -> tuple[tuple[dict[str, tuple[str, ...]], ...], str]:
    """Validate the exact existing five-fold split; never generate a replacement."""

    if not path.is_file():
        raise FileNotFoundError(f"Canonical split is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or len(payload) != 5:
        raise ValueError("Canonical split must contain exactly five folds")
    normalized: list[dict[str, tuple[str, ...]]] = []
    validation_union: set[str] = set()
    for fold, expected_validation_count in enumerate(EXPECTED_FOLD_VALIDATION_COUNTS):
        item = payload[fold]
        if not isinstance(item, dict) or set(item) != {"train", "val"}:
            raise ValueError(f"Fold {fold} must contain only train and val")
        train = tuple(str(value) for value in item["train"])
        validation = tuple(str(value) for value in item["val"])
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
        overlap = validation_union & set(validation)
        if overlap:
            raise ValueError(f"Validation cases are repeated across folds: {sorted(overlap)[:5]}")
        validation_union.update(validation)
        normalized.append({"train": train, "val": validation})
    if validation_union != case_ids:
        raise ValueError("Five-fold validation union is not the complete cohort")
    return tuple(normalized), sha256_file(path)


def _same_content(first: Path, second: Path) -> bool:
    try:
        if os.path.samefile(first, second):
            return True
    except OSError:
        pass
    return (
        first.stat().st_size == second.stat().st_size
        and sha256_file(first) == sha256_file(second)
    )


def _link_or_copy_verified(source: Path, destination: Path) -> str:
    if destination.exists():
        if not destination.is_file() or not _same_content(source, destination):
            raise FileExistsError(f"Conflicting v1 adapter file exists: {destination}")
        try:
            return "hardlink" if os.path.samefile(source, destination) else "copy"
        except OSError:
            return "copy"
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
        method = "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        method = "copy"
    if not _same_content(source, destination):
        destination.unlink(missing_ok=True)
        raise OSError(f"v1 adapter copy verification failed: {source} -> {destination}")
    return method


def _write_json_owned(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise FileExistsError(f"Conflicting owned JSON exists: {path}")
        return
    write_json_atomic(path, payload)


def _write_pickle_owned(path: Path, payload: Any) -> None:
    if path.exists():
        with path.open("rb") as handle:
            existing = pickle.load(handle)  # noqa: S301 - local project-owned nnU-Net v1 artifact
        if len(existing) != len(payload):
            raise FileExistsError(f"Conflicting split pickle exists: {path}")
        for found, wanted in zip(existing, payload, strict=True):
            if list(found["train"]) != list(wanted["train"]) or list(found["val"]) != list(
                wanted["val"]
            ):
                raise FileExistsError(f"Conflicting split pickle exists: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _v1_dataset_json(task_name: str, case_ids: Sequence[str]) -> dict[str, Any]:
    return {
        "name": task_name,
        "description": "BraTS 2023 Adult Glioma; project-owned nnU-Net v1 adapter",
        "tensorImageSize": "4D",
        "reference": "BraTS 2023 Adult Glioma challenge",
        "licence": "Subject to the source BraTS data-use terms",
        "release": "2023",
        "modality": {"0": "T1", "1": "T1ce", "2": "T2", "3": "FLAIR"},
        "labels": {"0": "background", "1": "NCR", "2": "ED", "3": "ET"},
        "numTraining": len(case_ids),
        "numTest": 0,
        "training": [
            {
                "image": f"./imagesTr/{case_id}.nii.gz",
                "label": f"./labelsTr/{case_id}.nii.gz",
            }
            for case_id in case_ids
        ],
        "test": [],
    }


def _native_splits(
    splits: Sequence[Mapping[str, Sequence[str]]],
) -> list[OrderedDict[str, np.ndarray]]:
    native: list[OrderedDict[str, np.ndarray]] = []
    for split in splits:
        item: OrderedDict[str, np.ndarray] = OrderedDict()
        item["train"] = np.asarray(split["train"], dtype=str)
        item["val"] = np.asarray(split["val"], dtype=str)
        native.append(item)
    return native


def prepare_v1_adapter(
    *,
    source_dataset: Path,
    source_split: Path,
    raw_base: Path,
    preprocessed_root: Path,
    full_task_name: str,
    smoke_task_name: str,
    smoke: bool,
    smoke_fold: int = 0,
    quick_train_cases: int = 8,
    quick_validation_cases: int = 2,
) -> tuple[V1AdapterLayout, dict[str, Any]]:
    """Create or verify one complete/smoke v1 task and its exact split mapping."""

    if smoke_fold not in range(5):
        raise ValueError("smoke_fold must be in 0..4")
    if quick_train_cases < 1 or quick_validation_cases < 1:
        raise ValueError("Smoke train/validation subsets must be non-empty")
    labels = inventory_dataset(source_dataset)
    splits, split_sha = load_canonical_splits(source_split, set(labels))
    task_name = smoke_task_name if smoke else full_task_name
    if smoke:
        source_fold = splits[smoke_fold]
        train = source_fold["train"][:quick_train_cases]
        validation = source_fold["val"][:quick_validation_cases]
        selected_splits: tuple[dict[str, tuple[str, ...]], ...] = (
            {"train": train, "val": validation},
        )
        case_ids = tuple(dict.fromkeys((*train, *validation)))
    else:
        selected_splits = splits
        case_ids = tuple(sorted(labels))
    task_directory = raw_base / "nnUNet_raw_data" / task_name
    preprocessed_task = preprocessed_root / task_name
    images_source = source_dataset / "imagesTr"
    methods: dict[str, int] = {}
    for case_index, case_id in enumerate(case_ids, start=1):
        for suffix in MODALITY_SUFFIXES:
            source = images_source / f"{case_id}_{suffix}.nii.gz"
            destination = task_directory / "imagesTr" / source.name
            method = _link_or_copy_verified(source, destination)
            methods[method] = methods.get(method, 0) + 1
        method = _link_or_copy_verified(
            labels[case_id], task_directory / "labelsTr" / f"{case_id}.nii.gz"
        )
        methods[method] = methods.get(method, 0) + 1
        if case_index % 100 == 0 or case_index == len(case_ids):
            print(
                f"[MEDNEXT-ADAPTER] {case_index}/{len(case_ids)} cases verified",
                flush=True,
            )
    _write_json_owned(task_directory / "dataset.json", _v1_dataset_json(task_name, case_ids))
    _write_pickle_owned(preprocessed_task / "splits_final.pkl", _native_splits(selected_splits))
    provenance = {
        "valid": True,
        "schema": "glioma_mednext_v1_adapter_v1",
        "task_name": task_name,
        "smoke": smoke,
        "source_dataset": str(source_dataset.resolve()),
        "source_split": str(source_split.resolve()),
        "source_split_sha256": split_sha,
        "case_count": len(case_ids),
        "modality_file_count": len(case_ids) * len(MODALITY_SUFFIXES),
        "label_file_count": len(case_ids),
        "link_methods": methods,
        "smoke_source_fold": smoke_fold if smoke else None,
        "smoke_train_case_count": quick_train_cases if smoke else None,
        "smoke_validation_case_count": quick_validation_cases if smoke else None,
        "train_case_ids_by_fold": [list(item["train"]) for item in selected_splits],
        "validation_case_ids_by_fold": [list(item["val"]) for item in selected_splits],
        "raw_task_directory": str(task_directory.resolve()),
        "preprocessed_task_directory": str(preprocessed_task.resolve()),
        "source_files_unchanged": True,
    }
    _write_json_owned(task_directory / "project_adapter.json", provenance)
    return (
        V1AdapterLayout(
            task_name=task_name,
            task_directory=task_directory.resolve(),
            preprocessed_task_directory=preprocessed_task.resolve(),
            case_ids=case_ids,
            splits=selected_splits,
            source_split_sha256=split_sha,
            smoke=smoke,
        ),
        provenance,
    )


__all__ = [
    "EXPECTED_CASE_COUNT",
    "EXPECTED_FOLD_VALIDATION_COUNTS",
    "MODALITY_SUFFIXES",
    "V1AdapterLayout",
    "inventory_dataset",
    "load_canonical_splits",
    "prepare_v1_adapter",
]
