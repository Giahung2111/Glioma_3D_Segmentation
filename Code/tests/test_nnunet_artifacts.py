import json
from pathlib import Path

import numpy as np

from glioma_seg.backends.nnunet.artifacts import validate_preprocessing_artifacts

DATASET_NAME = "Dataset501_BraTS2023GLI"


def _build_artifacts(tmp_path: Path) -> tuple[Path, Path, tuple[str, ...]]:
    raw = tmp_path / "nnUNet_raw" / DATASET_NAME
    preprocessed = tmp_path / "nnUNet_preprocessed" / DATASET_NAME
    images = raw / "imagesTr"
    labels = raw / "labelsTr"
    configuration = preprocessed / "nnUNetPlans_3d_fullres"
    ground_truth = preprocessed / "gt_segmentations"
    for directory in (images, labels, configuration, ground_truth):
        directory.mkdir(parents=True)

    case_ids = tuple(f"BraTS-GLI-{index:05d}-000" for index in range(5))
    from nnunetv2.training.dataloading.nnunet_dataset import (  # type: ignore[import-not-found]
        nnUNetDatasetBlosc2,
    )

    for case_id in case_ids:
        for channel in range(4):
            (images / f"{case_id}_{channel:04d}.nii.gz").write_bytes(b"raw-image")
        (labels / f"{case_id}.nii.gz").write_bytes(b"raw-label")
        nnUNetDatasetBlosc2.save_case(
            np.zeros((4, 8, 8, 8), dtype=np.float32),
            np.zeros((1, 8, 8, 8), dtype=np.int16),
            {
                "spacing": [1.0, 1.0, 1.0],
                "shape_before_cropping": [8, 8, 8],
                "bbox_used_for_cropping": [[0, 8], [0, 8], [0, 8]],
                "shape_after_cropping_and_before_resampling": [8, 8, 8],
                "class_locations": {},
            },
            str(configuration / case_id),
            chunks=(1, 8, 8, 8),
            blocks=(1, 4, 4, 4),
            chunks_seg=(1, 8, 8, 8),
            blocks_seg=(1, 4, 4, 4),
        )
        (ground_truth / f"{case_id}.nii.gz").write_bytes(b"gt-copy")

    dataset_json = {
        "channel_names": {"0": "T1n", "1": "T1c", "2": "T2w", "3": "T2F"},
        "labels": {
            "background": 0,
            "whole_tumor": [1, 2, 3],
            "tumor_core": [1, 3],
            "enhancing_tumor": 3,
        },
        "regions_class_order": [2, 1, 3],
        "numTraining": 5,
        "file_ending": ".nii.gz",
    }
    (raw / "dataset.json").write_text(json.dumps(dataset_json), encoding="utf-8")
    (preprocessed / "dataset.json").write_text(json.dumps(dataset_json), encoding="utf-8")
    plans = {
        "dataset_name": DATASET_NAME,
        "plans_name": "nnUNetPlans",
        "configurations": {
            "3d_fullres": {
                "data_identifier": "nnUNetPlans_3d_fullres",
                "preprocessor_name": "DefaultPreprocessor",
                "batch_size": 2,
                "patch_size": [128, 128, 128],
                "median_image_size_in_voxels": [155, 240, 240],
                "spacing": [1.0, 1.0, 1.0],
                "normalization_schemes": [
                    "ZScoreNormalization",
                    "ZScoreNormalization",
                    "ZScoreNormalization",
                    "ZScoreNormalization",
                ],
                "use_mask_for_norm": [True, True, True, True],
                "resampling_fn_data": "resample_data_or_seg_to_shape",
                "resampling_fn_seg": "resample_data_or_seg_to_shape",
                "resampling_fn_probabilities": "resample_data_or_seg_to_shape",
                "architecture": {
                    "network_class_name": "dynamic_network_architectures.PlainConvUNet",
                    "arch_kwargs": {"n_stages": 2, "features_per_stage": [32, 64]},
                },
            }
        },
    }
    (preprocessed / "nnUNetPlans.json").write_text(json.dumps(plans), encoding="utf-8")
    fingerprint = {
        "spacings": [[1.0, 1.0, 1.0] for _ in case_ids],
        "shapes_after_crop": [[155, 240, 240] for _ in case_ids],
        "foreground_intensity_properties_per_channel": {str(channel): {} for channel in range(4)},
    }
    (preprocessed / "dataset_fingerprint.json").write_text(
        json.dumps(fingerprint), encoding="utf-8"
    )
    return raw, preprocessed, case_ids


def test_complete_v281_inventory_creates_official_split(tmp_path: Path) -> None:
    raw, preprocessed, case_ids = _build_artifacts(tmp_path)

    report = validate_preprocessing_artifacts(
        raw_dataset_dir=raw,
        preprocessed_dataset_dir=preprocessed,
        dataset_name=DATASET_NAME,
        expected_case_count=5,
        ensure_splits=True,
    )

    assert report.valid, report.to_dict()
    assert report.details["preprocessed_format"] == "Blosc2"
    assert report.details["splits_created"] is True
    splits = json.loads((preprocessed / "splits_final.json").read_text(encoding="utf-8"))
    assert len(splits) == 5
    assert all(set(fold["train"]) | set(fold["val"]) == set(case_ids) for fold in splits)
    assert all(set(fold["train"]).isdisjoint(fold["val"]) for fold in splits)


def test_incomplete_preprocessing_never_creates_split(tmp_path: Path) -> None:
    raw, preprocessed, case_ids = _build_artifacts(tmp_path)
    (preprocessed / "nnUNetPlans_3d_fullres" / f"{case_ids[0]}_seg.b2nd").unlink()

    report = validate_preprocessing_artifacts(
        raw_dataset_dir=raw,
        preprocessed_dataset_dir=preprocessed,
        dataset_name=DATASET_NAME,
        expected_case_count=5,
        ensure_splits=True,
    )

    assert not report.valid
    assert not (preprocessed / "splits_final.json").exists()
    inventory = next(
        check for check in report.checks if check.name == "preprocessed Blosc2 case inventory"
    )
    assert not inventory.ok


def test_custom_or_corrupt_existing_split_is_not_overwritten(tmp_path: Path) -> None:
    raw, preprocessed, _case_ids = _build_artifacts(tmp_path)
    split_path = preprocessed / "splits_final.json"
    original = [{"train": ["not-a-case"], "val": []}]
    split_path.write_text(json.dumps(original), encoding="utf-8")

    report = validate_preprocessing_artifacts(
        raw_dataset_dir=raw,
        preprocessed_dataset_dir=preprocessed,
        dataset_name=DATASET_NAME,
        expected_case_count=5,
        ensure_splits=True,
    )

    assert not report.valid
    assert json.loads(split_path.read_text(encoding="utf-8")) == original
    split_check = next(check for check in report.checks if check.name == "official five-fold split")
    assert "not overwritten" in split_check.detail
