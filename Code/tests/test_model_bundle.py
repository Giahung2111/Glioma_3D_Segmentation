from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from glioma_seg.reporting.model_bundle import (
    ModelReportBundleError,
    audit_model_report_bundle,
    finalize_model_report_bundle,
)
from glioma_seg.utils.hashing import sha256_file


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _fold_counts(case_count: int, folds: list[int]) -> list[int]:
    quotient, remainder = divmod(case_count, len(folds))
    return [quotient + (1 if index < remainder else 0) for index in range(len(folds))]


def _make_bundle(
    tmp_path: Path,
    *,
    backend: str,
    kind: str,
    case_count: int,
    folds: list[int],
) -> Path:
    model = "MedNeXt-S" if backend == "mednext" else "SegResNet"
    model_id = f"{backend}_test_model"
    experiment_id = f"{backend}-{kind}-test"
    root = tmp_path / experiment_id
    root.mkdir()
    native_order = (
        ["background", "NCR", "ED", "ET"]
        if backend == "mednext"
        else ["TC", "WT", "ET"]
    )
    conversion = f"recorded {backend} native-to-canonical conversion"
    counts = _fold_counts(case_count, folds)
    case_ids = [f"case-{index:04d}" for index in range(case_count)]

    _write_json(
        root / "experiment.json",
        {
            "schema": "glioma_model_experiment_v1",
            "experiment_id": experiment_id,
            "experiment_kind": kind,
            "backend": backend,
            "model": model,
            "model_id": model_id,
        },
    )
    _write_json(root / "environment.json", {"backend": backend, "python": "3.11"})
    _write_json(
        root / "data_validation.json",
        {
            "valid": True,
            "dataset_kind": "training",
            "actual_case_count": max(case_count, 1251 if kind == "smoke" else case_count),
            "errors": [],
        },
    )
    _write_csv(
        root / "data_validation.csv",
        ["record_type", "dataset_kind", "actual_case_count", "status"],
        [
            {
                "record_type": "dataset_summary",
                "dataset_kind": "training",
                "actual_case_count": max(
                    case_count, 1251 if kind == "smoke" else case_count
                ),
                "status": "PASS",
            }
        ],
    )

    if kind == "fullcv":
        artifact_manifest = {
            "schema": "glioma_model_crossval_artifacts_v1",
            "backend": backend,
            "model_id": model_id,
            "model_provenance": {"source_commit": "a" * 40},
            "folds": [
                {
                    "fold": fold,
                    "prediction_dir": f"results/fold_{fold}/predictions",
                    "probability_dir": f"results/fold_{fold}/probabilities",
                }
                for fold in folds
            ],
            "probability_contract": {
                "required": True,
                "schema": "glioma_canonical_probabilities_v1",
                "native_channel_order": native_order,
                "canonical_channel_order": ["ET", "TC", "WT"],
                "conversion": conversion,
            },
        }
        artifact_path = root / "crossval_artifact_manifest.json"
        _write_json(artifact_path, artifact_manifest)
        per_fold = [
            {
                "fold": fold,
                "case_count": count,
                "metrics": {
                    "Dice": {"ET": 0.8, "TC": 0.9, "WT": 0.95},
                    "HD95": {"ET": 4.0, "TC": 5.0, "WT": 6.0},
                },
            }
            for fold, count in zip(folds, counts, strict=True)
        ]
        common_crossval = {
            "valid": True,
            "backend": backend,
            "model_id": model_id,
            "folds": folds,
            "total_cases": case_count,
            "validation_case_counts": counts,
            "each_case_validated_once": True,
            "probabilities_retained": True,
            "probability_source_channel_order": native_order,
            "probability_canonical_order": ["ET", "TC", "WT"],
        }
        _write_json(
            root / "crossval_summary.json",
            {
                **common_crossval,
                "probability_conversion": conversion,
                "per_fold": per_fold,
            },
        )
        integrity = {
            **common_crossval,
            "schema": "glioma_model_crossval_integrity_v1",
            "evaluation_scope": "five_fold_out_of_fold",
            "artifact_manifest_sha256": sha256_file(artifact_path),
            "pooled_prediction_count": case_count,
            "pooled_matches_fold_predictions": True,
            "probability_contract": artifact_manifest["probability_contract"],
            "pooled_probability_inventory": {"count": case_count},
            "fold_inventories": [
                {
                    "fold": fold,
                    "validation_case_count": count,
                    "prediction_count": count,
                    "probability_inventory": {"count": count},
                }
                for fold, count in zip(folds, counts, strict=True)
            ],
        }
    else:
        assert len(folds) == 1
        integrity = {
            "valid": True,
            "scope": "real_data_smoke_test_not_full_cross_validation",
            "backend": backend,
            "model_id": model_id,
            "fold": folds[0],
            "total_cases": case_count,
            "case_ids": case_ids,
            "each_case_validated_once": True,
        }
    integrity_path = root / "crossval_integrity.json"
    _write_json(integrity_path, integrity)

    metric_summary_rows = [
        {
            "metric": metric,
            "ET": value,
            "TC": value,
            "WT": value,
            "total_cases": case_count,
        }
        for metric, value in (("Dice", 0.9), ("HD95", 5.0))
    ]
    _write_csv(
        root / "metrics_summary.csv",
        ["metric", "ET", "TC", "WT", "total_cases"],
        metric_summary_rows,
    )
    _write_json(root / "metrics_summary.json", metric_summary_rows)
    _write_csv(
        root / "metrics_per_case.csv",
        ["case_id", "dice_et", "dice_tc", "dice_wt"],
        [
            {"case_id": case_id, "dice_et": 0.9, "dice_tc": 0.9, "dice_wt": 0.9}
            for case_id in case_ids
        ],
    )
    if kind == "fullcv":
        _write_csv(
            root / "crossval_metrics_by_fold.csv",
            ["fold", "case_count", "metric", "ET", "TC", "WT"],
            [
                {
                    "fold": fold,
                    "case_count": count,
                    "metric": metric,
                    "ET": value,
                    "TC": value,
                    "WT": value,
                }
                for fold, count in zip(folds, counts, strict=True)
                for metric, value in (("Dice", 0.9), ("HD95", 5.0))
            ],
        )
    _write_json(
        root / "evaluation_protocol.json",
        {
            "backend": backend,
            "model_id": model_id,
            "case_count": case_count,
            "case_ids": case_ids,
            "region_order": ["ET", "TC", "WT"],
        },
    )

    if kind == "smoke":
        offset = 0
        for fold, count in zip(folds, counts, strict=True):
            fold_case_ids = case_ids[offset : offset + count]
            offset += count
            fold_root = root / "folds" / f"fold_{fold}"
            _write_json(
                fold_root / "fold_manifest.json",
                {
                    "schema": "glioma_model_fold_manifest_v1",
                    "backend": backend,
                    "model_id": model_id,
                    "experiment_id": experiment_id,
                    "fold": fold,
                    "target_epochs": 3,
                    "smoke": True,
                    "validation_case_count": count,
                    "validation_case_ids": fold_case_ids,
                },
            )
            _write_json(
                fold_root / "artifact_audit.json",
                {
                    "valid": True,
                    "complete": True,
                    "fold": fold,
                    "experiment_id": experiment_id,
                    "validation_case_count": count,
                },
            )
            _write_json(
                fold_root / "runtime.json",
                {
                    "fold": fold,
                    "target_epochs": 3,
                    "number_of_epochs": 3,
                    "stopped_for_resume_test": False,
                },
            )
            _write_json(fold_root / "gpu_summary.json", {"samples": 3})
            _write_json(
                fold_root / "train_history.json",
                [{"epoch": epoch} for epoch in (1, 2, 3)],
            )
            _write_json(
                fold_root / "validation_summary.json",
                {
                    "valid": True,
                    "fold": fold,
                    "case_count": count,
                    "case_ids": fold_case_ids,
                    "canonical_channel_order": ["ET", "TC", "WT"],
                },
            )
        smoke_ground_truth = root / "smoke_ground_truth"
        smoke_ground_truth.mkdir()
        for case_id in case_ids:
            (smoke_ground_truth / f"{case_id}.nii.gz").write_bytes(b"smoke-gt")

    _write_json(
        root / "official_brats_metrics_status.json",
        {
            "available": True,
            "case_count": case_count,
            "region_order": ["ET", "TC", "WT"],
            "version_or_commit": "43c905242b2eecf421d4ab2da7af8ece9777d322",
        },
    )
    (root / "official_brats_evaluator.log").write_text(
        "official evaluation completed\n", encoding="utf-8"
    )
    _write_csv(
        root / "official_lesionwise_metrics_per_case.csv",
        ["case_id", "dice_et", "dice_tc", "dice_wt"],
        [
            {"case_id": case_id, "dice_et": 0.8, "dice_tc": 0.8, "dice_wt": 0.8}
            for case_id in case_ids
        ],
    )
    _write_csv(
        root / "official_lesionwise_metrics_summary.csv",
        ["metric", "ET", "TC", "WT", "total_cases"],
        metric_summary_rows,
    )
    _write_json(
        root / "official_lesionwise_metrics_summary.json",
        {
            "case_count": case_count,
            "region_order": ["ET", "TC", "WT"],
            "summary": metric_summary_rows,
        },
    )

    _write_json(
        root / "failure_statistics.json",
        {
            "analysis_scope": (
                "completed five-fold out-of-fold predictions; no training or inference"
                if kind == "fullcv"
                else (
                    "real-data smoke subset; NOT full cross-validation; "
                    "failure-statistics computation only"
                )
            ),
            "case_count": case_count,
            "case_region_count": case_count * 3,
            "regions": ["ET", "TC", "WT"],
            "sources": {
                "crossval_integrity_sha256": sha256_file(integrity_path),
                "metrics_per_case_sha256": sha256_file(root / "metrics_per_case.csv"),
            },
            "validation": {"all_evidence_consistent": True},
        },
    )
    _write_csv(
        root / "failure_statistics_per_case_region.csv",
        ["case_id", "region", "any_major_error"],
        [
            {"case_id": case_id, "region": region, "any_major_error": False}
            for case_id in case_ids
            for region in ("ET", "TC", "WT")
        ],
    )
    _write_csv(root / "failure_cases.csv", ["case_id", "failure_type"], [])

    _write_json(root / "runtime.json", {"total_seconds": 10.0})
    _write_json(
        root / "inference_runtime.json",
        {"number_of_cases": case_count, "mean_seconds_per_case": 1.0},
    )
    _write_json(root / "gpu_summary.json", {"peak_memory_used_mb": 1000.0})
    prefix = "5-Fold" if kind == "fullcv" else "Preliminary"
    (root / "summary.md").write_text(f"# {prefix} {model} report\n", encoding="utf-8")
    (root / "weekly_discussion.md").write_text(
        f"# {prefix} {model} weekly discussion\n", encoding="utf-8"
    )
    (root / "config_snapshot").mkdir()
    (root / "config_snapshot" / "model.yaml").write_text("model: pinned\n", encoding="utf-8")
    (root / "logs").mkdir()
    (root / "logs" / "pipeline.log").write_text("completed\n", encoding="utf-8")
    return root


@pytest.mark.parametrize("backend", ["mednext", "segresnet"])
def test_fullcv_model_bundle_is_a_hashed_atomic_receipt(
    tmp_path: Path, backend: str
) -> None:
    root = _make_bundle(
        tmp_path,
        backend=backend,
        kind="fullcv",
        case_count=1251,
        folds=[0, 1, 2, 3, 4],
    )

    destination = finalize_model_report_bundle(root)

    assert destination == root / "report_manifest.json"
    manifest = json.loads(destination.read_text(encoding="utf-8"))
    assert manifest["valid"] is True
    assert manifest["backend"] == backend
    assert manifest["expected_case_count"] == 1251
    assert manifest["expected_folds"] == [0, 1, 2, 3, 4]
    assert manifest["is_final_baseline"] is True
    assert manifest["baseline_status"] == "final_full_cross_validation_baseline"
    assert manifest["probability_contract"]["canonical_channel_order"] == [
        "ET",
        "TC",
        "WT",
    ]
    artifacts = {row["path"]: row for row in manifest["artifacts"]}
    assert "report_manifest.json" not in artifacts
    assert set(artifacts) >= {
        "experiment.json",
        "failure_statistics.json",
        "config_snapshot/model.yaml",
        "logs/pipeline.log",
    }
    for relative, record in artifacts.items():
        path = root / relative
        assert record["size_bytes"] == path.stat().st_size
        assert record["sha256"] == sha256_file(path)
    assert not (root / "report_manifest.json.tmp").exists()


def test_smoke_bundle_requires_explicit_scope_and_is_never_final_baseline(
    tmp_path: Path,
) -> None:
    root = _make_bundle(
        tmp_path,
        backend="mednext",
        kind="smoke",
        case_count=3,
        folds=[0],
    )
    assert not (root / "crossval_summary.json").exists()
    assert not (root / "crossval_artifact_manifest.json").exists()

    with pytest.raises(ModelReportBundleError, match="require explicit"):
        audit_model_report_bundle(root)

    audit = audit_model_report_bundle(root, expected_case_count=3, expected_folds=[0])
    assert audit["is_final_baseline"] is False
    assert not (root / "report_manifest.json").exists()

    destination = finalize_model_report_bundle(
        root, expected_case_count=3, expected_folds=[0]
    )
    manifest = json.loads(destination.read_text(encoding="utf-8"))
    assert manifest["expected_case_count"] == 3
    assert manifest["expected_folds"] == [0]
    assert manifest["is_final_baseline"] is False
    assert manifest["baseline_status"] == "smoke_test_not_final_baseline"
    assert manifest["probability_contract"]["fully_pooled_contract_verified"] is False


def test_bundle_rejects_identity_mismatch_without_publishing(tmp_path: Path) -> None:
    root = _make_bundle(
        tmp_path,
        backend="segresnet",
        kind="fullcv",
        case_count=1251,
        folds=[0, 1, 2, 3, 4],
    )
    summary_path = root / "crossval_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["backend"] = "mednext"
    _write_json(summary_path, summary)

    with pytest.raises(ModelReportBundleError, match="backend disagrees"):
        finalize_model_report_bundle(root)

    assert not (root / "report_manifest.json").exists()
    assert not (root / "report_manifest.json.tmp").exists()


def test_bundle_rejects_noncanonical_probability_contract_atomically(
    tmp_path: Path,
) -> None:
    root = _make_bundle(
        tmp_path,
        backend="mednext",
        kind="fullcv",
        case_count=1251,
        folds=[0, 1, 2, 3, 4],
    )
    destination = root / "report_manifest.json"
    destination.write_text("sentinel\n", encoding="utf-8")
    artifact_path = root / "crossval_artifact_manifest.json"
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    artifact["probability_contract"]["canonical_channel_order"] = ["WT", "TC", "ET"]
    _write_json(artifact_path, artifact)

    with pytest.raises(ModelReportBundleError, match="must be ET,TC,WT"):
        finalize_model_report_bundle(root)

    assert destination.read_text(encoding="utf-8") == "sentinel\n"
    assert not (root / "report_manifest.json.tmp").exists()
