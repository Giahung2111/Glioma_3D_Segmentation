from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pytest

from glioma_seg.ensembles.canonical_probabilities import (
    CANONICAL_REGION_ORDER,
    SCHEMA_NAME,
    SEGRESNET_REGION_CONVERSION,
    SEGRESNET_REGION_ORDER,
    segresnet_regions_to_canonical,
    write_canonical_probability_npz,
)
from glioma_seg.evaluation.model_crossval import (
    MANIFEST_SCHEMA,
    evaluate_model_cross_validation,
    load_model_crossval_manifest,
)


def _labels() -> np.ndarray:
    labels = np.zeros((5, 5, 5), dtype=np.uint8)
    labels[1, 1, 1] = 2
    labels[2, 2, 2] = 1
    labels[3, 3, 3] = 3
    return labels


def _write_nifti(path: Path, *, affine: np.ndarray | None = None, shift: int = 0) -> None:
    values = _labels()
    if shift:
        values = np.roll(values, shift, axis=0)
    nib.save(
        nib.Nifti1Image(values, np.eye(4) if affine is None else affine),  # type: ignore[no-untyped-call]
        path,
    )


def _write_probability(path: Path, *, case_id: str, reference: Path) -> None:
    native = np.empty((3, 5, 5, 5), dtype=np.float32)
    native[0].fill(0.8)  # TC
    native[1].fill(0.9)  # WT
    native[2].fill(0.7)  # ET
    write_canonical_probability_npz(
        path,
        case_id=case_id,
        probabilities=segresnet_regions_to_canonical(native),
        native_channel_order=SEGRESNET_REGION_ORDER,
        conversion=SEGRESNET_REGION_CONVERSION,
        reference_nifti=reference,
    )


def _relative(path: Path, base: Path) -> str:
    return str(path.relative_to(base))


def _make_artifacts(tmp_path: Path) -> dict[str, Any]:
    ground_truth = tmp_path / "labelsTr"
    pooled_predictions = tmp_path / "pooled" / "predictions"
    pooled_probabilities = tmp_path / "pooled" / "probabilities"
    output = tmp_path / "evaluation"
    for directory in (ground_truth, pooled_predictions, pooled_probabilities):
        directory.mkdir(parents=True)

    case_ids = [f"BraTS-GLI-MODEL-{index:03d}" for index in range(5)]
    fold_entries: list[dict[str, object]] = []
    splits: list[dict[str, list[str]]] = []
    fold_prediction_dirs: list[Path] = []
    fold_probability_dirs: list[Path] = []
    for fold, case_id in enumerate(case_ids):
        gt = ground_truth / f"{case_id}.nii.gz"
        _write_nifti(gt)
        train = [candidate for candidate in case_ids if candidate != case_id]
        splits.append({"train": train, "val": [case_id]})

        prediction_dir = tmp_path / "folds" / f"fold_{fold}" / "predictions"
        probability_dir = tmp_path / "folds" / f"fold_{fold}" / "probabilities"
        prediction_dir.mkdir(parents=True)
        probability_dir.mkdir(parents=True)
        fold_prediction_dirs.append(prediction_dir)
        fold_probability_dirs.append(probability_dir)
        _write_nifti(prediction_dir / f"{case_id}.nii.gz")
        _write_nifti(pooled_predictions / f"{case_id}.nii.gz")
        _write_probability(probability_dir / f"{case_id}.npz", case_id=case_id, reference=gt)
        _write_probability(pooled_probabilities / f"{case_id}.npz", case_id=case_id, reference=gt)
        fold_entries.append(
            {
                "fold": fold,
                "prediction_dir": _relative(prediction_dir, tmp_path),
                "probability_dir": _relative(probability_dir, tmp_path),
                "provenance": {
                    "fold_manifest_sha256": f"{fold + 1:064x}",
                    "checkpoint": f"fold_{fold}/checkpoint.pt",
                },
            }
        )

    splits_path = tmp_path / "splits_final.json"
    splits_path.write_text(json.dumps(splits), encoding="utf-8")
    manifest_path = tmp_path / "model_crossval_artifacts.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema": MANIFEST_SCHEMA,
                "backend": "segresnet",
                "model_id": "monai_model_zoo_brats_seg_resnet_100ep",
                "model_provenance": {
                    "framework": "MONAI",
                    "framework_version": "1.4.0",
                    "source_commit": "46a5272196a6c2590ca2589029eed8e4d56ff008",
                    "recipe_commit": "b9e4d04bb2a073110bde9e5c05c9690241e938b6",
                },
                "prediction_tta_state": "OFF",
                "folds": fold_entries,
                "pooled_prediction_dir": _relative(pooled_predictions, tmp_path),
                "pooled_probability_dir": _relative(pooled_probabilities, tmp_path),
                "probability_contract": {
                    "required": True,
                    "schema": SCHEMA_NAME,
                    "native_channel_order": list(SEGRESNET_REGION_ORDER),
                    "canonical_channel_order": list(CANONICAL_REGION_ORDER),
                    "conversion": SEGRESNET_REGION_CONVERSION,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "ground_truth": ground_truth,
        "pooled_predictions": pooled_predictions,
        "pooled_probabilities": pooled_probabilities,
        "output": output,
        "case_ids": case_ids,
        "splits": splits_path,
        "manifest": manifest_path,
        "fold_prediction_dirs": fold_prediction_dirs,
        "fold_probability_dirs": fold_probability_dirs,
    }


def test_backend_neutral_crossval_writes_complete_atomic_evidence(tmp_path: Path) -> None:
    paths = _make_artifacts(tmp_path)
    manifest = load_model_crossval_manifest(paths["manifest"])

    assert manifest.backend == "segresnet"
    assert manifest.model_id == "monai_model_zoo_brats_seg_resnet_100ep"
    assert manifest.folds[0].prediction_dir.is_absolute()
    assert manifest.probability_contract.native_channel_order == ("TC", "WT", "ET")

    summary = evaluate_model_cross_validation(
        ground_truth_dir=paths["ground_truth"],
        splits_json=paths["splits"],
        artifact_manifest=paths["manifest"],
        output_dir=paths["output"],
        expected_case_count=5,
    )

    assert summary["valid"] is True
    assert summary["backend"] == "segresnet"
    assert summary["folds"] == [0, 1, 2, 3, 4]
    assert summary["validation_case_counts"] == [1, 1, 1, 1, 1]
    assert summary["each_case_validated_once"] is True
    assert summary["probabilities_retained"] is True
    assert summary["probability_source_channel_order"] == ["TC", "WT", "ET"]
    assert summary["probability_canonical_order"] == ["ET", "TC", "WT"]
    assert summary["pooled"]["Dice"] == {"ET": 1.0, "TC": 1.0, "WT": 1.0}
    assert summary["macro_std"]["Dice"] == {"ET": 0.0, "TC": 0.0, "WT": 0.0}

    output = paths["output"]
    assert isinstance(output, Path)
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

    with (output / "metrics_per_case.csv").open("r", encoding="utf-8", newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 5
    with (output / "crossval_metrics_by_fold.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        assert len(list(csv.DictReader(handle))) == 10
    integrity = json.loads((output / "crossval_integrity.json").read_text(encoding="utf-8"))
    assert integrity["pooled_matches_fold_predictions"] is True
    assert integrity["artifact_manifest_sha256"]
    assert integrity["pooled_probability_inventory"]["count"] == 5
    assert [row["prediction_count"] for row in integrity["fold_inventories"]] == [1] * 5
    assert [row["probability_inventory"]["count"] for row in integrity["fold_inventories"]] == [
        1
    ] * 5


def test_model_crossval_rejects_duplicate_validation_before_publication(tmp_path: Path) -> None:
    paths = _make_artifacts(tmp_path)
    splits_path = paths["splits"]
    assert isinstance(splits_path, Path)
    splits = json.loads(splits_path.read_text(encoding="utf-8"))
    splits[1]["val"] = list(splits[0]["val"])
    splits[1]["train"] = [
        case_id
        for case_id in splits[1]["train"] + ["BraTS-GLI-MODEL-001"]
        if case_id not in splits[1]["val"]
    ]
    splits_path.write_text(json.dumps(splits), encoding="utf-8")

    with pytest.raises(ValueError, match="validated in folds"):
        evaluate_model_cross_validation(
            ground_truth_dir=paths["ground_truth"],
            splits_json=splits_path,
            artifact_manifest=paths["manifest"],
            output_dir=paths["output"],
            expected_case_count=5,
        )
    output = paths["output"]
    assert isinstance(output, Path)
    assert not (output / "metrics_summary.csv").exists()
    assert not (output / "fold_metrics").exists()


def test_model_crossval_rejects_incomplete_canonical_probability_inventory(
    tmp_path: Path,
) -> None:
    paths = _make_artifacts(tmp_path)
    probability_dirs = paths["fold_probability_dirs"]
    case_ids = paths["case_ids"]
    assert isinstance(probability_dirs, list)
    assert isinstance(case_ids, list)
    (probability_dirs[2] / f"{case_ids[2]}.npz").unlink()

    with pytest.raises(ValueError, match="canonical probability inventory mismatch"):
        evaluate_model_cross_validation(
            ground_truth_dir=paths["ground_truth"],
            splits_json=paths["splits"],
            artifact_manifest=paths["manifest"],
            output_dir=paths["output"],
            expected_case_count=5,
        )
    output = paths["output"]
    assert isinstance(output, Path)
    assert not (output / "crossval_integrity.json").exists()


def test_model_crossval_uses_common_evaluator_for_geometry_and_rejects_mismatch(
    tmp_path: Path,
) -> None:
    paths = _make_artifacts(tmp_path)
    prediction_dirs = paths["fold_prediction_dirs"]
    case_ids = paths["case_ids"]
    assert isinstance(prediction_dirs, list)
    assert isinstance(case_ids, list)
    wrong_affine = np.eye(4)
    wrong_affine[0, 3] = 8.0
    _write_nifti(prediction_dirs[0] / f"{case_ids[0]}.nii.gz", affine=wrong_affine)

    with pytest.raises(ValueError, match="Affine mismatch"):
        evaluate_model_cross_validation(
            ground_truth_dir=paths["ground_truth"],
            splits_json=paths["splits"],
            artifact_manifest=paths["manifest"],
            output_dir=paths["output"],
            expected_case_count=5,
        )
    output = paths["output"]
    assert isinstance(output, Path)
    assert not (output / "metrics_per_case.csv").exists()


def test_model_crossval_rejects_pooled_mask_that_is_not_the_fold_oof_mask(
    tmp_path: Path,
) -> None:
    paths = _make_artifacts(tmp_path)
    pooled = paths["pooled_predictions"]
    case_ids = paths["case_ids"]
    assert isinstance(pooled, Path)
    assert isinstance(case_ids, list)
    _write_nifti(pooled / f"{case_ids[4]}.nii.gz", shift=1)

    with pytest.raises(ValueError, match="differs from its declared fold prediction"):
        evaluate_model_cross_validation(
            ground_truth_dir=paths["ground_truth"],
            splits_json=paths["splits"],
            artifact_manifest=paths["manifest"],
            output_dir=paths["output"],
            expected_case_count=5,
        )
    output = paths["output"]
    assert isinstance(output, Path)
    assert not (output / "crossval_summary.json").exists()
