from pathlib import Path

import pytest

from glioma_seg.config.loader import load_base_config, load_dataset_config
from glioma_seg.config.schema import ConfigError, DatasetConfig

CONFIG_ROOT = Path(__file__).resolve().parents[1] / "configs"


def test_dataset_config_preserves_medical_region_order() -> None:
    config = load_dataset_config(CONFIG_ROOT / "datasets" / "brats2023_gli.yaml")

    assert config.dataset_id == 501
    assert config.modalities == (
        ("0000", "t1n"),
        ("0001", "t1c"),
        ("0002", "t2w"),
        ("0003", "t2f"),
    )
    assert list(config.nnunet_labels) == [
        "background",
        "whole_tumor",
        "tumor_core",
        "enhancing_tumor",
    ]
    assert config.regions_class_order == (2, 1, 3)


def test_dataset_schema_rejects_legacy_et_label() -> None:
    valid = load_dataset_config(CONFIG_ROOT / "datasets" / "brats2023_gli.yaml")
    value = {
        "schema_version": valid.schema_version,
        "dataset_id": valid.dataset_id,
        "dataset_name": valid.dataset_name,
        "description": valid.description,
        "expected_training_cases": valid.expected_training_cases,
        "expected_validation_cases": valid.expected_validation_cases,
        "case_id_pattern": valid.case_id_pattern,
        "file_ending": valid.file_ending,
        "modalities": dict(valid.modalities),
        "channel_names": dict(valid.channel_names),
        "raw_labels": {"background": 0, "NCR": 1, "ED": 2, "ET": 4},
        "regions": valid.nnunet_labels,
        "regions_class_order": list(valid.regions_class_order),
    }

    with pytest.raises(ConfigError, match="raw labels"):
        DatasetConfig.from_mapping(value)


def test_base_paths_are_resolved_relative_to_config() -> None:
    config = load_base_config(CONFIG_ROOT / "base.yaml")

    assert config.paths.project_root == CONFIG_ROOT.parents[1]
    assert config.paths.datasets_root == CONFIG_ROOT.parents[1] / "Datasets"
    assert config.paths.workspace_root == CONFIG_ROOT.parents[1] / "Workspace"
