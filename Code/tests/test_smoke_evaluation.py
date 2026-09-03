from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np

from glioma_seg.evaluation.smoke import prepare_and_evaluate_smoke


def _write_mask(path: Path) -> None:
    labels = np.zeros((8, 8, 8), dtype=np.uint8)
    labels[2:5, 2:5, 2:5] = 2
    labels[3:5, 3:5, 3:5] = 1
    labels[4, 4, 4] = 3
    nib.save(nib.Nifti1Image(labels, np.eye(4)), path)  # type: ignore[no-untyped-call]


def test_smoke_evaluation_records_backend_model_and_truthful_scope(tmp_path: Path) -> None:
    case_id = "BraTS-GLI-test"
    ground_truth = tmp_path / "ground_truth"
    predictions = tmp_path / "predictions"
    output = tmp_path / "report"
    ground_truth.mkdir()
    predictions.mkdir()
    _write_mask(ground_truth / f"{case_id}.nii.gz")
    _write_mask(predictions / f"{case_id}.nii.gz")
    manifest = tmp_path / "fold_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "glioma_model_fold_manifest_v1",
                "smoke": True,
                "backend": "segresnet",
                "model_id": "monai_model_zoo_brats_seg_resnet",
                "fold": 0,
                "validation_case_ids": [case_id],
            }
        ),
        encoding="utf-8",
    )

    result = prepare_and_evaluate_smoke(
        fold_manifest_json=manifest,
        full_ground_truth_dir=ground_truth,
        prediction_dir=predictions,
        output_dir=output,
    )

    assert result["valid"] is True
    protocol = json.loads((output / "evaluation_protocol.json").read_text(encoding="utf-8"))
    assert protocol["evaluation_scope"] == "real_data_smoke_test_not_full_cross_validation"
    assert protocol["backend"] == "segresnet"
    assert protocol["model_id"] == "monai_model_zoo_brats_seg_resnet"
    assert protocol["folds"] == [0]
    assert protocol["case_ids"] == [case_id]
