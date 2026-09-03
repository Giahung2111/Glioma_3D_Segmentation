from __future__ import annotations

import json
from pathlib import Path

import pytest

from glioma_seg.reporting.report import ReportInputs, generate_reports


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _fullcv_inputs(
    tmp_path: Path,
    *,
    backend: str,
    model: str,
    framework: str,
    source_order: list[str],
    preprocessing: str,
) -> ReportInputs:
    output = tmp_path / backend
    output.mkdir()
    split_path = tmp_path / f"{backend}_splits_final.json"
    split_path.write_text("[]", encoding="utf-8")

    experiment_path = output / "experiment.json"
    _write_json(
        experiment_path,
        {
            "experiment_id": f"{backend}-full-cv",
            "experiment_kind": "fullcv",
            "backend": backend,
            "model_display_name": model,
            "framework": framework,
            "framework_version": "test-pinned-version",
            "classification": f"100-epoch compute-limited {model} comparison",
            "dataset": "Dataset501_BraTS2023GLI",
            "dataset_id": 501,
            "dataset_case_count": 1251,
            "trainer": f"{model} project trainer",
            "epochs": 100,
            "total_training_epochs": 500,
            "configuration": f"{model} pinned recipe",
            "architecture": f"recorded {model} architecture",
            "crop_size": [128, 128, 128],
            "batch_size": 2,
            "target_spacing": [1.0, 1.0, 1.0],
            "num_workers": 4,
            "TTA_state": "OFF",
            "preprocessing_description": preprocessing,
            "limitations": [f"{model} test limitation from experiment.json."],
        },
    )
    metrics_path = output / "metrics_summary.csv"
    metrics_path.write_text(
        "metric,ET,TC,WT,total_cases\n"
        "Dice,0.8,0.9,0.95,1251\n"
        "HD95,4,5,6,1251\n",
        encoding="utf-8",
    )
    per_fold = [
        {
            "fold": fold,
            "case_count": 251 if fold == 0 else 250,
            "metrics": {
                "Dice": {"ET": 0.8, "TC": 0.9, "WT": 0.95},
                "HD95": {"ET": 4.0, "TC": 5.0, "WT": 6.0},
            },
        }
        for fold in range(5)
    ]
    crossval_path = output / "crossval_summary.json"
    probability_conversion = (
        "background,NCR,ED,ET -> ET,TC,WT by recorded region sums"
        if backend == "mednext"
        else "TC,WT,ET -> ET,TC,WT by channel permutation [2,0,1]"
    )
    _write_json(
        crossval_path,
        {
            "valid": True,
            "backend": backend,
            "folds": [0, 1, 2, 3, 4],
            "total_cases": 1251,
            "each_case_validated_once": True,
            "split_source": str(split_path),
            "validation_case_counts": [251, 250, 250, 250, 250],
            "probabilities_retained": True,
            "probability_source_channel_order": source_order,
            "probability_canonical_order": ["ET", "TC", "WT"],
            "probability_conversion": probability_conversion,
            "per_fold": per_fold,
        },
    )
    preprocessing_path = output / "preprocessing_artifacts.json"
    _write_json(
        preprocessing_path,
        {
            "valid": True,
            "details": {"splits_file": str(split_path), "splits_created": True},
        },
    )
    return ReportInputs(
        output_dir=output,
        experiment_json=experiment_path,
        metrics_summary_csv=metrics_path,
        preprocessing_artifacts_json=preprocessing_path,
        crossval_summary_json=crossval_path,
    )


@pytest.mark.parametrize(
    ("backend", "model", "framework", "source_order", "conversion_text"),
    [
        (
            "mednext",
            "MedNeXt-S (3x3x3)",
            "MedNeXt v1",
            ["background", "NCR", "ED", "ET"],
            (
                "uses recorded conversion: background,NCR,ED,ET -> ET,TC,WT "
                "by recorded region sums"
            ),
        ),
        (
            "segresnet",
            "SegResNet",
            "MONAI",
            ["TC", "WT", "ET"],
            "requires explicit reordering [2, 0, 1]",
        ),
    ],
)
def test_fullcv_reports_use_manifest_model_identity_and_probability_provenance(
    tmp_path: Path,
    backend: str,
    model: str,
    framework: str,
    source_order: list[str],
    conversion_text: str,
) -> None:
    preprocessing = f"Recorded {model} preprocessing from experiment.json."
    inputs = _fullcv_inputs(
        tmp_path,
        backend=backend,
        model=model,
        framework=framework,
        source_order=source_order,
        preprocessing=preprocessing,
    )

    summary_path, weekly_path = generate_reports(inputs)

    summary = summary_path.read_text(encoding="utf-8")
    weekly = weekly_path.read_text(encoding="utf-8")
    combined = summary + weekly
    assert f"5-Fold {model} Cross-Validation" in summary
    assert f"5-Fold {model} BraTS 2023 GLI Cross-Validation" in weekly
    assert f"| Model | {model} |" in summary
    assert f"| Framework | {framework} |" in summary
    assert f"Measure a reproducible 100-epoch-per-fold {model} experiment" in summary
    assert preprocessing in summary
    assert preprocessing in weekly
    assert f"{model} source order {','.join(source_order)}" in summary
    assert conversion_text in summary
    assert f"{model} test limitation from experiment.json." in summary

    false_nnunet_claims = (
        "nnU-Net format conversion",
        "Architecture from actual plans",
        "nnU-Net version",
        "nnU-Net source order",
        "standard 1,000-epoch nnU-Net",
    )
    for claim in false_nnunet_claims:
        assert claim not in combined
