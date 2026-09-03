from __future__ import annotations

import csv
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from glioma_seg.ensembles.nnunet_probabilities import (
    load_nnunet_region_probabilities,
    validate_brats_region_probability_contract,
)
from glioma_seg.evaluation.crossval import evaluate_five_fold_cross_validation


def _dataset_json(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "labels": {
                    "background": 0,
                    "whole_tumor": [1, 2, 3],
                    "tumor_core": [1, 3],
                    "enhancing_tumor": 3,
                },
                "regions_class_order": [2, 1, 3],
                "file_ending": ".nii.gz",
            }
        ),
        encoding="utf-8",
    )
    return path


def test_nnunet_probability_loader_reorders_wt_tc_et_to_et_tc_wt(tmp_path: Path) -> None:
    dataset = _dataset_json(tmp_path / "dataset.json")
    raw = np.stack(
        (
            np.full((2, 3, 4), 0.1, dtype=np.float32),  # WT
            np.full((2, 3, 4), 0.2, dtype=np.float32),  # TC
            np.full((2, 3, 4), 0.3, dtype=np.float32),  # ET
        )
    )
    probability_path = tmp_path / "case.npz"
    np.savez_compressed(probability_path, probabilities=raw)

    member = load_nnunet_region_probabilities(
        probability_path, dataset_json=dataset, model_id="nnunet-fold0"
    )

    assert member.channel_names == ("ET", "TC", "WT")
    assert member.model_id == "nnunet-fold0"
    np.testing.assert_allclose(member.probabilities[0], 0.3)
    np.testing.assert_allclose(member.probabilities[1], 0.2)
    np.testing.assert_allclose(member.probabilities[2], 0.1)
    assert member.metadata["source_channel_order"] == ["WT", "TC", "ET"]
    assert member.metadata["canonical_reorder_indices"] == [2, 1, 0]


def test_nnunet_probability_loader_rejects_pickle_object_array(tmp_path: Path) -> None:
    dataset = _dataset_json(tmp_path / "dataset.json")
    probability_path = tmp_path / "unsafe.npz"
    np.savez_compressed(probability_path, probabilities=np.asarray([object()], dtype=object))

    with pytest.raises(ValueError, match="Unable to load safe nnU-Net probabilities"):
        load_nnunet_region_probabilities(probability_path, dataset_json=dataset)


def _write_nifti(path: Path, value_shift: int = 0) -> None:
    labels = np.zeros((5, 5, 5), dtype=np.uint8)
    labels[1, 1, 1] = 2
    labels[2, 2, 2] = 1
    labels[3, 3, 3] = 3
    if value_shift:
        labels = np.roll(labels, value_shift, axis=0)
    nib.save(nib.Nifti1Image(labels, np.eye(4)), path)  # type: ignore[no-untyped-call]


def _make_crossval_tree(tmp_path: Path) -> dict[str, Path]:
    ground_truth = tmp_path / "labelsTr"
    model = tmp_path / "nnUNetTrainer_100epochs__nnUNetPlans__3d_fullres"
    accumulated = model / "crossval_results_folds_0_1_2_3_4"
    output = tmp_path / "report"
    for directory in (ground_truth, model, accumulated, output):
        directory.mkdir(parents=True, exist_ok=True)
    dataset = _dataset_json(tmp_path / "dataset.json")
    (model / "dataset.json").write_bytes(dataset.read_bytes())
    (model / "plans.json").write_text("{}\n", encoding="utf-8")
    (accumulated / "dataset.json").write_bytes(dataset.read_bytes())
    (accumulated / "plans.json").write_text("{}\n", encoding="utf-8")
    (accumulated / "summary.json").write_text("{}\n", encoding="utf-8")

    case_ids = [f"BraTS-GLI-TEST-{index:03d}" for index in range(5)]
    splits: list[dict[str, list[str]]] = []
    for fold, case_id in enumerate(case_ids):
        train = [candidate for candidate in case_ids if candidate != case_id]
        splits.append({"train": train, "val": [case_id]})
        _write_nifti(ground_truth / f"{case_id}.nii.gz")
        _write_nifti(accumulated / f"{case_id}.nii.gz")
        validation = model / f"fold_{fold}" / "validation"
        validation.mkdir(parents=True)
        (model / f"fold_{fold}" / "checkpoint_final.pth").write_bytes(b"checkpoint")
        (validation / "summary.json").write_text("{}\n", encoding="utf-8")
        _write_nifti(validation / f"{case_id}.nii.gz")
        np.savez_compressed(
            validation / f"{case_id}.npz",
            probabilities=np.full((3, 5, 5, 5), 0.5, dtype=np.float32),
        )
        # Inventory only: the evaluator must never deserialize nnU-Net pickle files.
        (validation / f"{case_id}.pkl").write_bytes(b"opaque nnU-Net properties")
    split_path = tmp_path / "splits_final.json"
    split_path.write_text(json.dumps(splits), encoding="utf-8")
    return {
        "ground_truth": ground_truth,
        "model": model,
        "accumulated": accumulated,
        "output": output,
        "dataset": dataset,
        "splits": split_path,
    }


def test_complete_crossval_writes_pooled_fold_and_probability_provenance(
    tmp_path: Path,
) -> None:
    paths = _make_crossval_tree(tmp_path)

    summary = evaluate_five_fold_cross_validation(
        ground_truth_dir=paths["ground_truth"],
        model_dir=paths["model"],
        accumulated_prediction_dir=paths["accumulated"],
        splits_json=paths["splits"],
        dataset_json=paths["dataset"],
        output_dir=paths["output"],
        expected_case_count=5,
        require_probabilities=True,
    )

    assert summary["valid"] is True
    assert summary["folds"] == [0, 1, 2, 3, 4]
    assert summary["validation_case_counts"] == [1, 1, 1, 1, 1]
    assert summary["total_cases"] == 5
    assert summary["each_case_validated_once"] is True
    assert summary["probabilities_retained"] is True
    assert summary["probability_source_channel_order"] == ["WT", "TC", "ET"]
    assert summary["probability_canonical_order"] == ["ET", "TC", "WT"]
    assert summary["pooled"]["Dice"] == {"ET": 1.0, "TC": 1.0, "WT": 1.0}
    assert len(summary["per_fold"]) == 5
    assert summary["macro_std"]["Dice"] == {"ET": 0.0, "TC": 0.0, "WT": 0.0}

    output = paths["output"]
    for name in (
        "metrics_per_case.csv",
        "metrics_summary.csv",
        "metrics_summary.json",
        "evaluation_protocol.json",
        "crossval_metrics_by_fold.csv",
        "crossval_summary.json",
        "crossval_integrity.json",
    ):
        assert (output / name).is_file()
    assert all((output / "fold_metrics" / f"fold_{fold}").is_dir() for fold in range(5))
    with (output / "crossval_metrics_by_fold.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 10
    assert {int(row["fold"]) for row in rows} == set(range(5))
    protocol = json.loads((output / "evaluation_protocol.json").read_text(encoding="utf-8"))
    assert protocol["evaluation_scope"] == "five_fold_out_of_fold"
    assert protocol["case_count"] == 5


def test_crossval_rejects_a_case_validated_twice_before_writing_metrics(tmp_path: Path) -> None:
    paths = _make_crossval_tree(tmp_path)
    splits = json.loads(paths["splits"].read_text(encoding="utf-8"))
    splits[1]["val"] = list(splits[0]["val"])
    splits[1]["train"] = [
        case_id for case_id in splits[1]["train"] + ["BraTS-GLI-TEST-001"]
        if case_id not in splits[1]["val"]
    ]
    paths["splits"].write_text(json.dumps(splits), encoding="utf-8")

    with pytest.raises(ValueError, match="validated in folds"):
        evaluate_five_fold_cross_validation(
            ground_truth_dir=paths["ground_truth"],
            model_dir=paths["model"],
            accumulated_prediction_dir=paths["accumulated"],
            splits_json=paths["splits"],
            dataset_json=paths["dataset"],
            output_dir=paths["output"],
            expected_case_count=5,
            require_probabilities=True,
        )
    assert not (paths["output"] / "metrics_summary.csv").exists()


def test_probability_contract_rejects_reordered_dataset_labels(tmp_path: Path) -> None:
    path = tmp_path / "dataset.json"
    path.write_text(
        json.dumps(
            {
                "labels": {
                    "background": 0,
                    "enhancing_tumor": 3,
                    "tumor_core": [1, 3],
                    "whole_tumor": [1, 2, 3],
                },
                "regions_class_order": [3, 1, 2],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Unsafe BraTS region declaration/order"):
        validate_brats_region_probability_contract(path)
