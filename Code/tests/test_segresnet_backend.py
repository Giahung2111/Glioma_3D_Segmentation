from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from glioma_seg.backends.segresnet import backend
from glioma_seg.backends.segresnet.config import load_recipe
from glioma_seg.backends.segresnet.trainer import (
    _aggregate_gpu_telemetry,
    _remap_brats2023_et_to_legacy,
    _save_torch_atomic,
)

ROOT = Path(__file__).resolve().parents[2]
MODEL_CONFIG = ROOT / "Code" / "configs" / "models" / "segresnet_monai_bundle.yaml"


def test_pinned_segresnet_recipe_matches_official_bundle_contract() -> None:
    recipe = load_recipe(MODEL_CONFIG)

    assert recipe.epochs == 100
    assert recipe.original_recipe_epochs == 300
    assert recipe.batch_size == 1
    assert recipe.crop_size == (224, 224, 144)
    assert recipe.validation_roi_size == (240, 240, 160)
    assert recipe.learning_rate == pytest.approx(1e-4)
    assert recipe.weight_decay == pytest.approx(1e-5)
    assert recipe.payload["architecture"] == {
        "class": "monai.networks.nets.SegResNet",
        "in_channels": 4,
        "out_channels": 3,
        "init_filters": 16,
        "blocks_down": [1, 2, 2, 4],
        "blocks_up": [1, 1, 1],
        "dropout_prob": 0.2,
    }


def test_pinned_recipe_rejects_silent_parameter_change(tmp_path: Path) -> None:
    payload = yaml.safe_load(MODEL_CONFIG.read_text(encoding="utf-8"))
    payload["inference"]["overlap"] = 0.25
    changed = tmp_path / "changed.yaml"
    changed.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="inference.overlap"):
        load_recipe(changed)


def test_brats_2023_et_adapter_only_changes_label_three() -> None:
    labels = np.asarray([0, 1, 2, 3], dtype=np.uint8)

    remapped = _remap_brats2023_et_to_legacy(labels)

    np.testing.assert_array_equal(remapped, np.asarray([0, 1, 2, 4], dtype=np.uint8))


def test_checkpoint_atomic_publish_is_windows_safe(tmp_path: Path) -> None:
    import torch

    checkpoint = tmp_path / "checkpoint_latest.pth"
    _save_torch_atomic(checkpoint, {"epoch": 1, "tensor": torch.asarray([2.0])})

    loaded = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert loaded["epoch"] == 1
    assert loaded["tensor"].item() == pytest.approx(2.0)
    assert not checkpoint.with_suffix(".pth.tmp").exists()


def test_gpu_telemetry_aggregates_resume_segments_without_overwrite(tmp_path: Path) -> None:
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
    for index, utilization in enumerate((50.0, 100.0)):
        segment = tmp_path / f"gpu_samples_segment_epoch_{index:04d}_test.csv"
        with segment.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerow(
                {
                    "timestamp": f"2026-01-01T00:00:0{index}+00:00",
                    "elapsed_seconds": "1.0",
                    "gpu_name": "GPU",
                    "gpu_utilization_percent": str(utilization),
                    "memory_used_mb": str(1000 + index),
                    "memory_total_mb": "11000",
                    "temperature_c": str(60 + index),
                    "power_w": str(150 + index),
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
    assert summary["mean_gpu_utilization_percent"] == pytest.approx(75.0)
    assert summary["peak_memory_used_mb"] == pytest.approx(1001.0)
    with (tmp_path / "gpu_samples.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert float(rows[1]["elapsed_seconds"]) > float(rows[0]["elapsed_seconds"])


def _audit_manifest(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    fold_dir = tmp_path / "fold_0"
    fold_dir.mkdir()
    manifest: dict[str, object] = {
        "schema": "glioma_model_fold_manifest_v1",
        "experiment_id": "segresnet_smoke_test",
        "fold": 0,
        "config_sha256": "config-hash",
        "split_sha256": "split-hash",
        "target_epochs": 3,
        "validation_case_ids": ["case-a"],
    }
    (fold_dir / "fold_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return fold_dir, manifest


def test_audit_allows_owner_matched_final_checkpoint_to_resume_missing_export(
    tmp_path: Path,
) -> None:
    import torch

    fold_dir, manifest = _audit_manifest(tmp_path)
    torch.save(
        {
            "experiment_id": manifest["experiment_id"],
            "fold": 0,
            "config_sha256": manifest["config_sha256"],
            "split_sha256": manifest["split_sha256"],
            "target_epochs": 3,
            "completed_epoch": 3,
        },
        fold_dir / "checkpoint_final.pth",
    )

    result = backend.audit_fold(fold_dir)

    assert result["valid"] is False
    assert result["complete"] is False
    assert result["safe_to_resume"] is True
    assert "validation export is incomplete" in result["reason"]


def test_audit_rejects_foreign_final_checkpoint(tmp_path: Path) -> None:
    import torch

    fold_dir, manifest = _audit_manifest(tmp_path)
    torch.save(
        {
            "experiment_id": "foreign-experiment",
            "fold": 0,
            "config_sha256": manifest["config_sha256"],
            "split_sha256": manifest["split_sha256"],
            "target_epochs": 3,
            "completed_epoch": 3,
        },
        fold_dir / "checkpoint_final.pth",
    )

    result = backend.audit_fold(fold_dir)

    assert result["safe_to_resume"] is False
    assert "checkpoint mismatch" in result["reason"]


def test_audit_allows_latest_at_target_to_publish_final_artifacts(tmp_path: Path) -> None:
    import torch

    fold_dir, manifest = _audit_manifest(tmp_path)
    torch.save(
        {
            "experiment_id": manifest["experiment_id"],
            "fold": 0,
            "config_sha256": manifest["config_sha256"],
            "split_sha256": manifest["split_sha256"],
            "target_epochs": 3,
            "completed_epoch": 3,
        },
        fold_dir / "checkpoint_latest.pth",
    )

    result = backend.audit_fold(fold_dir)

    assert result["safe_to_resume"] is True
    assert "final publication/export may resume" in result["reason"]


def test_canonical_split_requires_exact_disjoint_five_fold_partition(
    tmp_path: Path,
) -> None:
    case_ids = [f"case_{index:04d}" for index in range(1251)]
    validation_counts = (251, 250, 250, 250, 250)
    splits: list[dict[str, list[str]]] = []
    start = 0
    for count in validation_counts:
        validation = case_ids[start : start + count]
        validation_set = set(validation)
        train = [case_id for case_id in case_ids if case_id not in validation_set]
        splits.append({"train": train, "val": validation})
        start += count
    split_path = tmp_path / "splits_final.json"
    split_path.write_text(json.dumps(splits), encoding="utf-8")

    normalized, digest = backend._load_splits(split_path, set(case_ids))

    assert [len(item["val"]) for item in normalized] == list(validation_counts)
    assert len(digest) == 64


def test_modality_inventory_uses_explicit_official_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = tmp_path / "Dataset501_BraTS2023GLI"
    images = raw / "imagesTr"
    labels = raw / "labelsTr"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    case_id = "BraTS-GLI-00001-000"
    (labels / f"{case_id}.nii.gz").touch()
    for suffix in ("0000", "0001", "0002", "0003"):
        (images / f"{case_id}_{suffix}.nii.gz").touch()
    monkeypatch.setattr(backend, "EXPECTED_CASE_COUNT", 1)

    inventory = backend._inventory_cases(raw)

    assert [path.name for path in inventory[case_id].images] == [
        f"{case_id}_0001.nii.gz",
        f"{case_id}_0000.nii.gz",
        f"{case_id}_0002.nii.gz",
        f"{case_id}_0003.nii.gz",
    ]
