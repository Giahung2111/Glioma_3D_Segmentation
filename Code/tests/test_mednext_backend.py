from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from glioma_seg.backends.mednext import backend, dataset
from glioma_seg.backends.mednext.compat import _basename
from glioma_seg.backends.mednext.config import EXPECTED_MEDNEXT_COMMIT, load_recipe
from glioma_seg.backends.mednext.trainer import (
    _aggregate_gpu_telemetry,
    _native_xyz,
    _verify_native_checkpoint_owner,
)

ROOT = Path(__file__).resolve().parents[2]
MODEL_CONFIG = ROOT / "Code" / "configs" / "models" / "mednext.yaml"


def test_pinned_mednext_recipe_matches_official_s_k3_contract() -> None:
    recipe = load_recipe(MODEL_CONFIG)

    assert recipe.model_id == "mednext_v1_s_kernel3"
    assert recipe.epochs == 100
    assert recipe.original_recipe_epochs == 1000
    assert recipe.patch_size == (128, 128, 128)
    assert recipe.payload["upstream"]["commit"] == EXPECTED_MEDNEXT_COMMIT
    assert recipe.payload["framework"] == {
        "name": "nnU-Net v1 (MedNeXt fork)",
        "task_full": "Task501_BraTS2023GLI",
        "task_smoke": "Task951_BraTS2023GLISmoke",
        "network": "3d_fullres",
        "trainer": "nnUNetTrainerV2_MedNeXt_S_kernel3",
        "planner_3d": "ExperimentPlanner3D_v21_customTargetSpacing_1x1x1",
        "plans_identifier": "nnUNetPlansv2.1_trgSp_1x1x1",
        "data_identifier": "nnUNetData_plans_v2.1_trgSp_1x1x1",
    }
    assert recipe.payload["training"]["optimizer"] == "AdamW"
    assert recipe.payload["training"]["initial_learning_rate"] == pytest.approx(1e-3)
    assert recipe.payload["training"]["optimizer_epsilon"] == pytest.approx(1e-4)


def test_pinned_mednext_recipe_rejects_silent_parameter_change(tmp_path: Path) -> None:
    payload = yaml.safe_load(MODEL_CONFIG.read_text(encoding="utf-8"))
    payload["architecture"]["kernel_size"] = 5
    changed = tmp_path / "changed.yaml"
    changed.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=r"architecture\.kernel_size"):
        load_recipe(changed)


def test_canonical_split_validation_requires_exact_five_fold_partition(
    tmp_path: Path,
) -> None:
    case_ids = [f"case_{index:04d}" for index in range(1251)]
    splits: list[dict[str, list[str]]] = []
    start = 0
    for count in (251, 250, 250, 250, 250):
        validation = case_ids[start : start + count]
        validation_set = set(validation)
        splits.append(
            {
                "train": [case for case in case_ids if case not in validation_set],
                "val": validation,
            }
        )
        start += count
    path = tmp_path / "splits_final.json"
    path.write_text(json.dumps(splits), encoding="utf-8")

    normalized, digest = dataset.load_canonical_splits(path, set(case_ids))

    assert [len(split["val"]) for split in normalized] == [251, 250, 250, 250, 250]
    assert len(digest) == 64
    splits[1]["val"][0] = splits[0]["val"][0]
    path.write_text(json.dumps(splits), encoding="utf-8")
    with pytest.raises(ValueError):
        dataset.load_canonical_splits(path, set(case_ids))


def test_v1_adapter_is_source_preserving_and_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "Dataset501_BraTS2023GLI"
    (source / "imagesTr").mkdir(parents=True)
    (source / "labelsTr").mkdir()
    case_ids = ("case_a", "case_b", "case_c")
    labels: dict[str, Path] = {}
    for case_index, case_id in enumerate(case_ids):
        for suffix_index, suffix in enumerate(dataset.MODALITY_SUFFIXES):
            (source / "imagesTr" / f"{case_id}_{suffix}.nii.gz").write_bytes(
                bytes([case_index, suffix_index])
            )
        label = source / "labelsTr" / f"{case_id}.nii.gz"
        label.write_bytes(bytes([case_index, 9]))
        labels[case_id] = label.resolve()
    splits = (
        {"train": ("case_a", "case_b"), "val": ("case_c",)},
        {"train": ("case_a", "case_c"), "val": ("case_b",)},
        {"train": ("case_b", "case_c"), "val": ("case_a",)},
    )
    monkeypatch.setattr(dataset, "inventory_dataset", lambda _: labels)
    monkeypatch.setattr(
        dataset,
        "load_canonical_splits",
        lambda _path, _ids: (splits, "a" * 64),
    )
    split_path = tmp_path / "splits_final.json"
    split_path.write_text("[]", encoding="utf-8")

    layout, first = dataset.prepare_v1_adapter(
        source_dataset=source,
        source_split=split_path,
        raw_base=tmp_path / "mednext" / "raw_base",
        preprocessed_root=tmp_path / "mednext" / "preprocessed",
        full_task_name="Task501_BraTS2023GLI",
        smoke_task_name="Task951_BraTS2023GLISmoke",
        smoke=False,
    )
    _, second = dataset.prepare_v1_adapter(
        source_dataset=source,
        source_split=split_path,
        raw_base=tmp_path / "mednext" / "raw_base",
        preprocessed_root=tmp_path / "mednext" / "preprocessed",
        full_task_name="Task501_BraTS2023GLI",
        smoke_task_name="Task951_BraTS2023GLISmoke",
        smoke=False,
    )

    assert layout.case_ids == case_ids
    assert first["case_count"] == second["case_count"] == 3
    dataset_json = json.loads((layout.task_directory / "dataset.json").read_text())
    assert dataset_json["modality"] == {
        "0": "T1",
        "1": "T1ce",
        "2": "T2",
        "3": "FLAIR",
    }
    assert dataset_json["labels"] == {
        "0": "background",
        "1": "NCR",
        "2": "ED",
        "3": "ET",
    }
    assert (source / "labelsTr" / "case_a.nii.gz").read_bytes() == bytes([0, 9])


def test_windows_path_adapter_and_softmax_axis_mapping_are_explicit() -> None:
    assert _basename(r"C:\project\Task951\case_0000.nii.gz") == "case_0000.nii.gz"
    softmax_zyx = np.zeros((4, 3, 2, 1), dtype=np.float32)
    softmax_zyx[1, 2, 1, 0] = 1.0

    mapped = _native_xyz(softmax_zyx, (1, 2, 3))

    assert mapped.shape == (4, 1, 2, 3)
    assert mapped[1, 0, 1, 2] == pytest.approx(1.0)


def test_evaluation_manifest_declares_exact_probability_contract(tmp_path: Path) -> None:
    destination = backend.write_evaluation_manifest(
        tmp_path,
        "mednext_test",
        folds=(0, 1, 2, 3, 4),
    )
    payload = json.loads(destination.read_text(encoding="utf-8"))

    assert payload["backend"] == "mednext"
    assert payload["prediction_tta_state"] == "OFF"
    assert payload["probability_contract"] == {
        "required": True,
        "native_channel_order": ["background", "NCR", "ED", "ET"],
        "canonical_channel_order": ["ET", "TC", "WT"],
        "conversion": "ET=p(ET);TC=p(NCR)+p(ET);WT=p(NCR)+p(ED)+p(ET)",
        "schema": "glioma_canonical_probabilities_v1",
    }
    assert [record["fold"] for record in payload["folds"]] == [0, 1, 2, 3, 4]


def test_native_checkpoint_must_embed_exact_project_owner(tmp_path: Path) -> None:
    import torch

    owner = {
        "experiment_id": "mednext_smoke_test",
        "fold": 0,
        "config_sha256": "a" * 64,
    }
    checkpoint = tmp_path / "model_resume_checkpoint.model"
    torch.save({"epoch": 1, "glioma_project_owner": owner}, checkpoint)

    assert _verify_native_checkpoint_owner(checkpoint, owner) == 1
    with pytest.raises(ValueError, match="ownership mismatch"):
        _verify_native_checkpoint_owner(checkpoint, {**owner, "fold": 1})


def test_mednext_gpu_telemetry_aggregates_resume_segments(tmp_path: Path) -> None:
    fields = [
        "timestamp",
        "elapsed_seconds",
        "gpu_name",
        "gpu_utilization_percent",
        "memory_used_mb",
        "memory_total_mb",
        "temperature_c",
        "power_w",
    ]
    for index, utilization in enumerate((40.0, 90.0)):
        segment = tmp_path / f"gpu_samples_segment_invocation_{index + 1:02d}_test.csv"
        with segment.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerow(
                {
                    "timestamp": f"2026-01-01T00:00:0{index}+00:00",
                    "elapsed_seconds": "1.0",
                    "gpu_name": "GPU",
                    "gpu_utilization_percent": str(utilization),
                    "memory_used_mb": str(9000 + index),
                    "memory_total_mb": "11264",
                    "temperature_c": str(70 + index),
                    "power_w": str(180 + index),
                }
            )
        segment.with_suffix(".summary.json").write_text(
            json.dumps({"backend": "nvidia-smi", "errors": []}), encoding="utf-8"
        )

    summary = _aggregate_gpu_telemetry(
        tmp_path,
        merged_csv=tmp_path / "gpu_samples.csv",
        summary_json=tmp_path / "gpu_summary.json",
    )

    assert summary["samples"] == 2
    assert summary["segments"] == 2
    assert summary["mean_gpu_utilization_percent"] == pytest.approx(65.0)
    assert summary["peak_memory_used_mb"] == pytest.approx(9001.0)
    assert summary["includes_all_owner_matched_invocations"] is True
    with (tmp_path / "gpu_samples.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert float(rows[1]["elapsed_seconds"]) > float(rows[0]["elapsed_seconds"])
