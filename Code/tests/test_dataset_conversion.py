import json
from dataclasses import replace
from pathlib import Path

import pytest

from glioma_seg.config.loader import load_dataset_config
from glioma_seg.config.schema import DatasetConfig
from glioma_seg.data.brats2023 import MODALITIES, nnunet_image_name, source_filename
from glioma_seg.data.nnunet_conversion import (
    ConversionError,
    build_dataset_json,
    convert_brats_to_nnunet,
)
from glioma_seg.data.validate import CaseValidation, NiftiFileValidation, ValidationReport
from glioma_seg.utils.hashing import fingerprint_file

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "datasets" / "brats2023_gli.yaml"
CASE_ID = "BraTS-GLI-00008-001"


def _small_spec() -> DatasetConfig:
    return replace(
        load_dataset_config(CONFIG_PATH),
        expected_training_cases=1,
        expected_validation_cases=0,
    )


def _training_report(raw_root: Path) -> ValidationReport:
    case_directory = raw_root / CASE_ID
    case_directory.mkdir(parents=True)
    records = []
    for role in (*MODALITIES, "seg"):
        path = case_directory / source_filename(CASE_ID, role)
        path.write_bytes(f"immutable-{role}".encode())
        records.append(
            NiftiFileValidation(
                role=role, path=str(path.resolve()), fingerprint=fingerprint_file(path)
            )
        )
    return ValidationReport(
        schema_version=1,
        created_at_utc="2026-01-01T00:00:00+00:00",
        dataset_root=str(raw_root.resolve()),
        dataset_kind="training",
        expected_case_count=1,
        cases=(
            CaseValidation(
                case_id=CASE_ID,
                directory=str(case_directory.resolve()),
                files=tuple(records),
            ),
        ),
    )


def test_dataset_json_has_required_region_order_and_class_order() -> None:
    value = build_dataset_json(_small_spec(), num_training=1)

    assert list(value) == [
        "channel_names",
        "labels",
        "regions_class_order",
        "numTraining",
        "file_ending",
    ]
    assert list(value["labels"]) == [
        "background",
        "whole_tumor",
        "tumor_core",
        "enhancing_tumor",
    ]
    assert value["labels"]["whole_tumor"] == [1, 2, 3]
    assert value["labels"]["tumor_core"] == [1, 3]
    assert value["labels"]["enhancing_tumor"] == 3
    assert value["regions_class_order"] == [2, 1, 3]


def test_conversion_is_copy_safe_and_resumable(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    output_root = tmp_path / "nnUNet_raw"
    report = _training_report(raw_root)
    before = {item.role: item.fingerprint for item in report.cases[0].files}

    first = convert_brats_to_nnunet(
        report,
        output_root,
        _small_spec(),
        prefer_hardlink=False,
    )
    second = convert_brats_to_nnunet(
        report,
        output_root,
        _small_spec(),
        prefer_hardlink=False,
    )

    assert first.method_counts == {"copy": 5}
    assert second.method_counts == {"reused_file": 5}
    assert all(fingerprint is not None and fingerprint.matches() for fingerprint in before.values())
    dataset_directory = output_root / "Dataset501_BraTS2023GLI"
    assert not (dataset_directory / "imagesTs").exists()
    for modality in MODALITIES:
        assert (dataset_directory / "imagesTr" / nnunet_image_name(CASE_ID, modality)).is_file()
    with (dataset_directory / "dataset.json").open(encoding="utf-8") as stream:
        dataset_json = json.load(stream)
    assert dataset_json["numTraining"] == 1
    assert dataset_json["regions_class_order"] == [2, 1, 3]


def test_default_conversion_prefers_hardlink_with_safe_fallback(tmp_path: Path) -> None:
    report = _training_report(tmp_path / "raw")

    result = convert_brats_to_nnunet(report, tmp_path / "out", _small_spec())

    assert set(result.method_counts) <= {"hardlink", "copy"}
    assert sum(result.method_counts.values()) == 5
    assert all(
        item.fingerprint is not None and item.fingerprint.matches()
        for item in report.cases[0].files
    )


def test_conflicting_destination_is_never_overwritten(tmp_path: Path) -> None:
    report = _training_report(tmp_path / "raw")
    output_root = tmp_path / "out"
    convert_brats_to_nnunet(
        report,
        output_root,
        _small_spec(),
        prefer_hardlink=False,
    )
    conflict = (
        output_root / "Dataset501_BraTS2023GLI" / "imagesTr" / nnunet_image_name(CASE_ID, "t1n")
    )
    conflict.write_bytes(b"conflicting-generated-content")

    with pytest.raises(ConversionError, match="Refusing to overwrite"):
        convert_brats_to_nnunet(
            report,
            output_root,
            _small_spec(),
            prefer_hardlink=False,
        )
    source_fingerprint = report.cases[0].file_map["t1n"].fingerprint
    assert source_fingerprint is not None
    assert source_fingerprint.matches()


def test_dataset_json_refuses_unvalidated_count() -> None:
    with pytest.raises(ConversionError, match="does not match expected"):
        build_dataset_json(_small_spec(), num_training=2)
