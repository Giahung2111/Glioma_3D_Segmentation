"""Audit and finalize backend-neutral MedNeXt/SegResNet report bundles.

This module is intentionally separate from the legacy nnU-Net bundle logic. It
only reads completed experiment evidence and atomically publishes a hashed
``report_manifest.json`` receipt; it never trains, predicts, preprocesses, or
modifies upstream repositories.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from glioma_seg.monitoring.timing import write_json_atomic
from glioma_seg.utils.hashing import sha256_file

REPORT_MANIFEST_SCHEMA = "glioma_model_report_manifest_v1"
REPORT_MANIFEST_FILENAME = "report_manifest.json"
CANONICAL_REGION_ORDER = ("ET", "TC", "WT")
FULL_CV_FOLDS = (0, 1, 2, 3, 4)
FULL_CV_CASE_COUNT = 1251
OFFICIAL_EVALUATOR_COMMIT = "43c905242b2eecf421d4ab2da7af8ece9777d322"

COMMON_REQUIRED_FILES = (
    "experiment.json",
    "environment.json",
    "data_validation.json",
    "data_validation.csv",
    "crossval_integrity.json",
    "metrics_per_case.csv",
    "metrics_summary.csv",
    "metrics_summary.json",
    "evaluation_protocol.json",
    "official_brats_metrics_status.json",
    "official_brats_evaluator.log",
    "official_lesionwise_metrics_per_case.csv",
    "official_lesionwise_metrics_summary.csv",
    "official_lesionwise_metrics_summary.json",
    "failure_statistics.json",
    "failure_statistics_per_case_region.csv",
    "failure_cases.csv",
    "runtime.json",
    "inference_runtime.json",
    "gpu_summary.json",
    "summary.md",
    "weekly_discussion.md",
)
FULL_CV_REQUIRED_FILES = (
    "crossval_summary.json",
    "crossval_metrics_by_fold.csv",
    "crossval_artifact_manifest.json",
)

_BACKEND_ALIASES = {
    "mednext": "mednext",
    "mednextv1": "mednext",
    "segresnet": "segresnet",
    "monaisegresnet": "segresnet",
}
_BACKEND_DISPLAY = {"mednext": "MedNeXt", "segresnet": "SegResNet"}


class ModelReportBundleError(RuntimeError):
    """Raised when model report evidence is missing or inconsistent."""


def _normalized_identifier(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def _required_file(root: Path, relative: str) -> Path:
    path = root / relative
    if path.is_symlink():
        raise ModelReportBundleError(f"Required artifact must not be a symlink: {path}")
    if not path.is_file() or path.stat().st_size == 0:
        raise ModelReportBundleError(f"Required artifact is missing or empty: {path}")
    return path


def _load_json(path: Path, *, description: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ModelReportBundleError(
            f"{description} is not valid UTF-8 JSON: {path}: {exc}"
        ) from exc


def _load_json_object(path: Path, *, description: str) -> dict[str, Any]:
    value = _load_json(path, description=description)
    if not isinstance(value, dict):
        raise ModelReportBundleError(f"{description} must contain a JSON object: {path}")
    return value


def _read_text(path: Path, *, description: str) -> str:
    try:
        value = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ModelReportBundleError(f"{description} is not valid UTF-8: {path}: {exc}") from exc
    if not value.strip():
        raise ModelReportBundleError(f"{description} contains no text: {path}")
    return value


def _csv_rows(path: Path, *, description: str) -> tuple[list[str], list[dict[str, str]]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = list(reader.fieldnames or [])
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ModelReportBundleError(f"{description} is not readable CSV: {path}: {exc}") from exc
    if not fields:
        raise ModelReportBundleError(f"{description} has no CSV header: {path}")
    return fields, rows


def _nonempty_text(mapping: Mapping[str, Any], *keys: str) -> str:
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise ModelReportBundleError(f"Missing non-empty identity field: {'/'.join(keys)}")


def _backend(value: Any, *, source: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ModelReportBundleError(f"{source} must record a non-empty backend")
    normalized = _BACKEND_ALIASES.get(_normalized_identifier(value))
    if normalized is None:
        raise ModelReportBundleError(
            f"{source} backend must be MedNeXt or SegResNet, got {value!r}"
        )
    return normalized


def _positive_integer(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ModelReportBundleError(f"{field} must be a positive integer")
    return int(value)


def _fold_list(value: Any, *, field: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ModelReportBundleError(f"{field} must be a non-empty JSON list")
    if any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in value):
        raise ModelReportBundleError(f"{field} must contain non-negative integer fold IDs")
    folds = tuple(value)
    if len(set(folds)) != len(folds):
        raise ModelReportBundleError(f"{field} contains duplicate fold IDs")
    return folds


def _resolve_scope(
    experiment: Mapping[str, Any],
    expected_case_count: int | None,
    expected_folds: Sequence[int] | None,
) -> tuple[str, int, tuple[int, ...], bool]:
    kind = experiment.get("experiment_kind")
    if kind == "fullcv":
        if expected_case_count not in (None, FULL_CV_CASE_COUNT):
            raise ModelReportBundleError("Full-CV bundles must contain exactly 1,251 cases")
        if expected_folds is not None and tuple(expected_folds) != FULL_CV_FOLDS:
            raise ModelReportBundleError("Full-CV bundles must contain folds 0,1,2,3,4")
        return "fullcv", FULL_CV_CASE_COUNT, FULL_CV_FOLDS, True
    if kind != "smoke":
        raise ModelReportBundleError("experiment_kind must be 'fullcv' or 'smoke'")
    if expected_case_count is None or expected_folds is None:
        raise ModelReportBundleError(
            "Smoke bundles require explicit expected_case_count and expected_folds"
        )
    count = _positive_integer(expected_case_count, field="expected_case_count")
    folds = _fold_list(list(expected_folds), field="expected_folds")
    return "smoke", count, folds, False


def _validate_identity(
    experiment: Mapping[str, Any],
    environment: Mapping[str, Any],
) -> tuple[str, str, str, str]:
    backend = _backend(experiment.get("backend"), source="experiment.json")
    model_id = _nonempty_text(experiment, "model_id")
    model_display = _nonempty_text(
        experiment,
        "model_display_name",
        "model_display",
        "model_name",
        "model",
    )
    experiment_id = _nonempty_text(experiment, "experiment_id")
    identity = _normalized_identifier(f"{model_display} {model_id}")
    if backend not in identity:
        raise ModelReportBundleError(
            f"Experiment model identity {model_display!r}/{model_id!r} does not match {backend}"
        )
    if "backend" in environment:
        environment_backend = _backend(environment.get("backend"), source="environment.json")
        if environment_backend != backend:
            raise ModelReportBundleError("environment.json backend disagrees with experiment.json")
    if "model_id" in environment and environment.get("model_id") != model_id:
        raise ModelReportBundleError("environment.json model_id disagrees with experiment.json")
    return backend, model_id, model_display, experiment_id


def _validate_data(
    validation: Mapping[str, Any], *, expected_case_count: int, final_baseline: bool
) -> None:
    if validation.get("valid") is not True:
        raise ModelReportBundleError("data_validation.json is not marked valid")
    if validation.get("dataset_kind") not in (None, "training"):
        raise ModelReportBundleError("data_validation.json must describe the training dataset")
    actual = validation.get("actual_case_count", validation.get("valid_case_count"))
    actual_count = _positive_integer(actual, field="data_validation actual_case_count")
    if final_baseline and actual_count != expected_case_count:
        raise ModelReportBundleError(
            f"Full-CV data validation expected {expected_case_count} cases, got {actual_count}"
        )
    if not final_baseline and actual_count < expected_case_count:
        raise ModelReportBundleError(
            "Smoke evaluation cannot contain more cases than the validated dataset"
        )
    if validation.get("errors") not in (None, []):
        raise ModelReportBundleError("data_validation.json contains validation errors")


def _validate_record_scope(
    record: Mapping[str, Any],
    *,
    description: str,
    backend: str,
    model_id: str,
    expected_case_count: int,
    expected_folds: tuple[int, ...],
) -> list[int]:
    if record.get("valid") is not True:
        raise ModelReportBundleError(f"{description} is not marked valid")
    if _backend(record.get("backend"), source=description) != backend:
        raise ModelReportBundleError(f"{description} backend disagrees with experiment.json")
    if record.get("model_id") != model_id:
        raise ModelReportBundleError(f"{description} model_id disagrees with experiment.json")
    if _fold_list(record.get("folds"), field=f"{description}.folds") != expected_folds:
        raise ModelReportBundleError(f"{description} folds do not match the expected scope")
    if record.get("total_cases") != expected_case_count:
        raise ModelReportBundleError(f"{description} total_cases does not match the expected scope")
    if record.get("each_case_validated_once") is not True:
        raise ModelReportBundleError(f"{description} does not verify each case exactly once")
    counts = record.get("validation_case_counts")
    if (
        not isinstance(counts, list)
        or len(counts) != len(expected_folds)
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in counts
        )
        or sum(counts) != expected_case_count
    ):
        raise ModelReportBundleError(
            f"{description} validation_case_counts do not partition {expected_case_count} cases"
        )
    return counts


def _validate_probability_contract(
    artifact_manifest: Mapping[str, Any],
    *,
    backend: str,
    model_id: str,
    expected_folds: tuple[int, ...],
) -> dict[str, Any]:
    if artifact_manifest.get("schema") != "glioma_model_crossval_artifacts_v1":
        raise ModelReportBundleError("crossval artifact manifest schema is not supported")
    if _backend(artifact_manifest.get("backend"), source="artifact manifest") != backend:
        raise ModelReportBundleError("crossval artifact manifest backend identity disagrees")
    if artifact_manifest.get("model_id") != model_id:
        raise ModelReportBundleError("crossval artifact manifest model_id disagrees")
    provenance = artifact_manifest.get("model_provenance")
    if not isinstance(provenance, dict) or not provenance:
        raise ModelReportBundleError("crossval artifact manifest lacks model provenance")
    fold_records = artifact_manifest.get("folds")
    if not isinstance(fold_records, list):
        raise ModelReportBundleError("crossval artifact manifest folds must be a JSON list")
    manifest_folds = []
    for record in fold_records:
        if not isinstance(record, dict):
            raise ModelReportBundleError("crossval artifact manifest fold entries must be objects")
        manifest_folds.append(record.get("fold"))
    if tuple(manifest_folds) != expected_folds:
        raise ModelReportBundleError("crossval artifact manifest fold IDs disagree")
    contract = artifact_manifest.get("probability_contract")
    if not isinstance(contract, dict):
        raise ModelReportBundleError("crossval artifact manifest lacks probability_contract")
    native = contract.get("native_channel_order")
    canonical = contract.get("canonical_channel_order")
    if contract.get("required") is not True:
        raise ModelReportBundleError("Canonical probability retention must be required")
    if contract.get("schema") != "glioma_canonical_probabilities_v1":
        raise ModelReportBundleError("Canonical probability schema is not supported")
    if (
        not isinstance(native, list)
        or not native
        or any(not isinstance(value, str) or not value.strip() for value in native)
        or len(set(native)) != len(native)
    ):
        raise ModelReportBundleError("Native probability channel order is invalid")
    if canonical != list(CANONICAL_REGION_ORDER):
        raise ModelReportBundleError("Canonical probability order must be ET,TC,WT")
    conversion = contract.get("conversion")
    if not isinstance(conversion, str) or not conversion.strip():
        raise ModelReportBundleError("Probability conversion provenance is missing")
    return {
        "required": True,
        "schema": contract["schema"],
        "native_channel_order": native,
        "canonical_channel_order": canonical,
        "conversion": conversion.strip(),
    }


def _validate_crossval_probability_evidence(
    record: Mapping[str, Any],
    *,
    description: str,
    contract: Mapping[str, Any],
) -> None:
    if record.get("probabilities_retained") is not True:
        raise ModelReportBundleError(f"{description} does not retain all probabilities")
    if record.get("probability_source_channel_order") != contract["native_channel_order"]:
        raise ModelReportBundleError(f"{description} native probability order disagrees")
    if record.get("probability_canonical_order") != list(CANONICAL_REGION_ORDER):
        raise ModelReportBundleError(f"{description} canonical probability order disagrees")


def _validate_per_fold_summary(
    summary: Mapping[str, Any],
    *,
    expected_folds: tuple[int, ...],
    validation_counts: Sequence[int],
) -> None:
    records = summary.get("per_fold")
    if not isinstance(records, list) or len(records) != len(expected_folds):
        raise ModelReportBundleError("crossval_summary.json lacks the expected per-fold records")
    for record, fold, count in zip(records, expected_folds, validation_counts, strict=True):
        if not isinstance(record, dict) or record.get("fold") != fold:
            raise ModelReportBundleError("crossval_summary.json per-fold identity disagrees")
        if record.get("case_count") != count:
            raise ModelReportBundleError("crossval_summary.json per-fold case count disagrees")
        metrics = record.get("metrics")
        if not isinstance(metrics, dict) or not {"Dice", "HD95"}.issubset(metrics):
            raise ModelReportBundleError("crossval_summary.json per-fold metrics are incomplete")


def _validate_integrity_details(
    integrity: Mapping[str, Any],
    *,
    artifact_manifest_path: Path,
    contract: Mapping[str, Any],
    expected_case_count: int,
    expected_folds: tuple[int, ...],
    validation_counts: Sequence[int],
) -> None:
    if integrity.get("schema") != "glioma_model_crossval_integrity_v1":
        raise ModelReportBundleError("crossval_integrity.json schema is not supported")
    if integrity.get("artifact_manifest_sha256") != sha256_file(artifact_manifest_path):
        raise ModelReportBundleError("crossval artifact manifest hash disagrees with integrity")
    if integrity.get("pooled_prediction_count") != expected_case_count:
        raise ModelReportBundleError("Pooled prediction count is incomplete")
    if integrity.get("pooled_matches_fold_predictions") is not True:
        raise ModelReportBundleError("Pooled predictions do not match declared fold outputs")
    recorded_contract = integrity.get("probability_contract")
    if not isinstance(recorded_contract, dict) or any(
        recorded_contract.get(key) != value for key, value in contract.items()
    ):
        raise ModelReportBundleError("crossval_integrity probability contract disagrees")
    pooled_probability = integrity.get("pooled_probability_inventory")
    if (
        not isinstance(pooled_probability, dict)
        or pooled_probability.get("count") != expected_case_count
    ):
        raise ModelReportBundleError("Pooled canonical probability inventory is incomplete")
    inventories = integrity.get("fold_inventories")
    if not isinstance(inventories, list) or len(inventories) != len(expected_folds):
        raise ModelReportBundleError("crossval_integrity fold inventories are incomplete")
    for record, fold, count in zip(inventories, expected_folds, validation_counts, strict=True):
        probability = record.get("probability_inventory") if isinstance(record, dict) else None
        if (
            not isinstance(record, dict)
            or record.get("fold") != fold
            or record.get("validation_case_count") != count
            or record.get("prediction_count") != count
            or not isinstance(probability, dict)
            or probability.get("count") != count
        ):
            raise ModelReportBundleError(f"Fold {fold} prediction/probability inventory disagrees")


def _validate_smoke_integrity(
    integrity: Mapping[str, Any],
    *,
    backend: str,
    model_id: str,
    expected_case_count: int,
    expected_folds: tuple[int, ...],
) -> set[str]:
    if integrity.get("valid") is not True:
        raise ModelReportBundleError("Smoke crossval_integrity.json is not marked valid")
    if integrity.get("scope") != "real_data_smoke_test_not_full_cross_validation":
        raise ModelReportBundleError("Smoke integrity must be explicitly labelled NOT full CV")
    if _backend(integrity.get("backend"), source="smoke integrity") != backend:
        raise ModelReportBundleError("Smoke integrity backend identity disagrees")
    if integrity.get("model_id") != model_id:
        raise ModelReportBundleError("Smoke integrity model_id disagrees")
    if integrity.get("total_cases") != expected_case_count:
        raise ModelReportBundleError("Smoke integrity total_cases disagrees")
    if integrity.get("each_case_validated_once") is not True:
        raise ModelReportBundleError("Smoke integrity does not verify each case exactly once")
    if "folds" in integrity:
        recorded_folds = _fold_list(integrity.get("folds"), field="smoke integrity folds")
    else:
        if len(expected_folds) != 1 or integrity.get("fold") != expected_folds[0]:
            raise ModelReportBundleError("Smoke integrity fold identity disagrees")
        recorded_folds = (int(integrity["fold"]),)
    if recorded_folds != expected_folds:
        raise ModelReportBundleError("Smoke integrity folds disagree with the explicit scope")
    case_ids = integrity.get("case_ids")
    if (
        not isinstance(case_ids, list)
        or len(case_ids) != expected_case_count
        or any(not isinstance(case_id, str) or not case_id for case_id in case_ids)
        or len(set(case_ids)) != expected_case_count
    ):
        raise ModelReportBundleError("Smoke integrity case IDs are incomplete or duplicated")
    return set(case_ids)


def _validate_smoke_fold_evidence(
    root: Path,
    *,
    backend: str,
    model_id: str,
    experiment_id: str,
    expected_folds: tuple[int, ...],
    expected_case_ids: set[str],
) -> None:
    manifest_case_ids: set[str] = set()
    for fold in expected_folds:
        fold_root = root / "folds" / f"fold_{fold}"
        required = (
            "fold_manifest.json",
            "artifact_audit.json",
            "runtime.json",
            "gpu_summary.json",
            "train_history.json",
            "validation_summary.json",
        )
        for name in required:
            _required_file(root, (fold_root / name).relative_to(root).as_posix())
        manifest = _load_json_object(
            fold_root / "fold_manifest.json", description=f"smoke fold {fold} manifest"
        )
        if manifest.get("schema") != "glioma_model_fold_manifest_v1":
            raise ModelReportBundleError(f"Smoke fold {fold} manifest schema is invalid")
        if _backend(manifest.get("backend"), source=f"smoke fold {fold}") != backend:
            raise ModelReportBundleError(f"Smoke fold {fold} backend identity disagrees")
        if (
            manifest.get("model_id") != model_id
            or manifest.get("experiment_id") != experiment_id
            or manifest.get("fold") != fold
            or manifest.get("smoke") is not True
        ):
            raise ModelReportBundleError(f"Smoke fold {fold} manifest ownership disagrees")
        case_ids = manifest.get("validation_case_ids")
        if (
            not isinstance(case_ids, list)
            or not case_ids
            or len(case_ids) != len(set(case_ids))
            or manifest.get("validation_case_count") != len(case_ids)
        ):
            raise ModelReportBundleError(f"Smoke fold {fold} validation IDs are invalid")
        overlap = manifest_case_ids & set(case_ids)
        if overlap:
            raise ModelReportBundleError(f"Smoke cases appear in multiple folds: {sorted(overlap)}")
        manifest_case_ids.update(case_ids)
        target_epochs = _positive_integer(
            manifest.get("target_epochs"), field=f"smoke fold {fold} target_epochs"
        )

        audit = _load_json_object(
            fold_root / "artifact_audit.json", description=f"smoke fold {fold} audit"
        )
        if (
            audit.get("valid") is not True
            or audit.get("complete") is not True
            or audit.get("fold") != fold
            or audit.get("experiment_id") != experiment_id
            or audit.get("validation_case_count") != len(case_ids)
        ):
            raise ModelReportBundleError(f"Smoke fold {fold} artifact audit is incomplete")
        runtime = _load_json_object(
            fold_root / "runtime.json", description=f"smoke fold {fold} runtime"
        )
        if (
            runtime.get("fold") != fold
            or runtime.get("target_epochs") != target_epochs
            or runtime.get("number_of_epochs") != target_epochs
            or runtime.get("stopped_for_resume_test") is not False
        ):
            raise ModelReportBundleError(f"Smoke fold {fold} runtime is incomplete")
        history = _load_json(
            fold_root / "train_history.json", description=f"smoke fold {fold} history"
        )
        if not isinstance(history, list) or len(history) != target_epochs:
            raise ModelReportBundleError(f"Smoke fold {fold} training history is incomplete")
        validation = _load_json_object(
            fold_root / "validation_summary.json",
            description=f"smoke fold {fold} validation summary",
        )
        if (
            validation.get("valid") is not True
            or validation.get("fold") != fold
            or validation.get("case_count") != len(case_ids)
            or validation.get("case_ids") != case_ids
            or validation.get("canonical_channel_order") != list(CANONICAL_REGION_ORDER)
        ):
            raise ModelReportBundleError(f"Smoke fold {fold} validation summary disagrees")
        gpu = _load_json_object(
            fold_root / "gpu_summary.json", description=f"smoke fold {fold} GPU summary"
        )
        if not gpu:
            raise ModelReportBundleError(f"Smoke fold {fold} GPU summary is empty")
    if manifest_case_ids != expected_case_ids:
        raise ModelReportBundleError("Smoke fold manifests do not match evaluated case IDs")

    ground_truth = root / "smoke_ground_truth"
    if not ground_truth.is_dir() or ground_truth.is_symlink():
        raise ModelReportBundleError("Smoke ground-truth subset is missing")
    names = {
        path.name.removesuffix(".nii.gz")
        for path in ground_truth.glob("*.nii.gz")
        if path.is_file() and not path.is_symlink() and path.stat().st_size > 0
    }
    if names != expected_case_ids:
        raise ModelReportBundleError("Smoke ground-truth inventory disagrees with integrity")


def _validate_case_csv(
    path: Path,
    *,
    description: str,
    expected_case_count: int,
) -> set[str]:
    fields, rows = _csv_rows(path, description=description)
    if "case_id" not in fields:
        raise ModelReportBundleError(f"{description} lacks case_id")
    case_ids = [row.get("case_id", "").strip() for row in rows]
    if len(case_ids) != expected_case_count or any(not case_id for case_id in case_ids):
        raise ModelReportBundleError(f"{description} does not contain {expected_case_count} cases")
    if len(set(case_ids)) != expected_case_count:
        raise ModelReportBundleError(f"{description} contains duplicate case IDs")
    return set(case_ids)


def _validate_metric_summary_csv(
    path: Path, *, description: str, expected_case_count: int
) -> None:
    fields, rows = _csv_rows(path, description=description)
    required = {"metric", "ET", "TC", "WT", "total_cases"}
    if not required.issubset(fields):
        raise ModelReportBundleError(f"{description} lacks semantic metric columns")
    by_metric = {row.get("metric"): row for row in rows}
    if not {"Dice", "HD95"}.issubset(by_metric):
        raise ModelReportBundleError(f"{description} must contain Dice and HD95")
    for metric in ("Dice", "HD95"):
        try:
            total = int(by_metric[metric]["total_cases"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ModelReportBundleError(f"{description} has invalid total_cases") from exc
        if total != expected_case_count:
            raise ModelReportBundleError(f"{description} total_cases disagrees")


def _validate_metric_summary_json(
    path: Path, *, description: str, expected_case_count: int
) -> None:
    payload = _load_json(path, description=description)
    if not isinstance(payload, list):
        raise ModelReportBundleError(f"{description} must contain a JSON list")
    by_metric = {
        row.get("metric"): row for row in payload if isinstance(row, dict)
    }
    if not {"Dice", "HD95"}.issubset(by_metric):
        raise ModelReportBundleError(f"{description} must contain Dice and HD95")
    for metric in ("Dice", "HD95"):
        row = by_metric[metric]
        if row.get("total_cases") != expected_case_count:
            raise ModelReportBundleError(f"{description} total_cases disagrees")
        if any(region not in row for region in CANONICAL_REGION_ORDER):
            raise ModelReportBundleError(f"{description} lacks ET/TC/WT values")


def _validate_crossval_metrics_csv(
    path: Path,
    *,
    expected_folds: tuple[int, ...],
    validation_counts: Sequence[int],
) -> None:
    fields, rows = _csv_rows(path, description="cross-validation metric summary")
    required = {"fold", "case_count", "metric", "ET", "TC", "WT"}
    if not required.issubset(fields):
        raise ModelReportBundleError("crossval_metrics_by_fold.csv lacks required columns")
    expected = {
        (str(fold), str(count), metric)
        for fold, count in zip(expected_folds, validation_counts, strict=True)
        for metric in ("Dice", "HD95")
    }
    actual = {(row["fold"], row["case_count"], row["metric"]) for row in rows}
    if actual != expected:
        raise ModelReportBundleError("crossval_metrics_by_fold.csv scope disagrees")


def _validate_semantic_outputs(
    root: Path,
    *,
    backend: str,
    model_id: str,
    expected_case_count: int,
    expected_folds: tuple[int, ...],
    validation_counts: Sequence[int] | None,
) -> set[str]:
    case_ids = _validate_case_csv(
        root / "metrics_per_case.csv",
        description="semantic per-case metrics",
        expected_case_count=expected_case_count,
    )
    _validate_metric_summary_csv(
        root / "metrics_summary.csv",
        description="semantic metric summary CSV",
        expected_case_count=expected_case_count,
    )
    _validate_metric_summary_json(
        root / "metrics_summary.json",
        description="semantic metric summary JSON",
        expected_case_count=expected_case_count,
    )
    if validation_counts is not None:
        _validate_crossval_metrics_csv(
            root / "crossval_metrics_by_fold.csv",
            expected_folds=expected_folds,
            validation_counts=validation_counts,
        )
    protocol = _load_json_object(
        root / "evaluation_protocol.json", description="semantic evaluation protocol"
    )
    if protocol.get("case_count") != expected_case_count:
        raise ModelReportBundleError("evaluation_protocol.json case_count disagrees")
    if protocol.get("region_order") != list(CANONICAL_REGION_ORDER):
        raise ModelReportBundleError("evaluation_protocol.json region order must be ET,TC,WT")
    if _backend(protocol.get("backend"), source="evaluation_protocol.json") != backend:
        raise ModelReportBundleError("evaluation_protocol.json backend identity disagrees")
    if protocol.get("model_id") != model_id:
        raise ModelReportBundleError("evaluation_protocol.json model_id disagrees")
    protocol_ids = protocol.get("case_ids")
    if protocol_ids is not None and (
        not isinstance(protocol_ids, list) or set(protocol_ids) != case_ids
    ):
        raise ModelReportBundleError("evaluation_protocol.json case IDs disagree")
    return case_ids


def _validate_official_outputs(root: Path, *, expected_case_count: int) -> set[str]:
    status = _load_json_object(
        root / "official_brats_metrics_status.json", description="official metric status"
    )
    if status.get("available") is not True or status.get("case_count") != expected_case_count:
        raise ModelReportBundleError("Official lesion-wise metrics are incomplete")
    if status.get("region_order") != list(CANONICAL_REGION_ORDER):
        raise ModelReportBundleError("Official metric region order must be ET,TC,WT")
    if status.get("version_or_commit") != OFFICIAL_EVALUATOR_COMMIT:
        raise ModelReportBundleError("Official evaluator commit is not the pinned project commit")
    case_ids = _validate_case_csv(
        root / "official_lesionwise_metrics_per_case.csv",
        description="official lesion-wise per-case metrics",
        expected_case_count=expected_case_count,
    )
    _validate_metric_summary_csv(
        root / "official_lesionwise_metrics_summary.csv",
        description="official lesion-wise summary CSV",
        expected_case_count=expected_case_count,
    )
    summary = _load_json_object(
        root / "official_lesionwise_metrics_summary.json",
        description="official lesion-wise summary JSON",
    )
    if summary.get("case_count") != expected_case_count:
        raise ModelReportBundleError("Official lesion-wise summary case_count disagrees")
    if summary.get("region_order") != list(CANONICAL_REGION_ORDER):
        raise ModelReportBundleError("Official lesion-wise summary region order disagrees")
    rows = summary.get("summary")
    recorded_metrics = {
        row.get("metric") for row in rows if isinstance(row, dict)
    } if isinstance(rows, list) else set()
    if not {"Dice", "HD95"}.issubset(recorded_metrics):
        raise ModelReportBundleError("Official lesion-wise summary is incomplete")
    return case_ids


def _validate_failure_outputs(
    root: Path,
    *,
    expected_case_ids: set[str],
    integrity_path: Path,
    experiment_kind: str,
    integrity_scope: str | None,
) -> None:
    expected_case_count = len(expected_case_ids)
    statistics = _load_json_object(
        root / "failure_statistics.json", description="failure statistics"
    )
    if statistics.get("case_count") != expected_case_count:
        raise ModelReportBundleError("failure_statistics.json case_count disagrees")
    if statistics.get("case_region_count") != expected_case_count * len(
        CANONICAL_REGION_ORDER
    ):
        raise ModelReportBundleError("failure_statistics.json case-region count disagrees")
    if statistics.get("regions") != list(CANONICAL_REGION_ORDER):
        raise ModelReportBundleError("failure_statistics.json region order disagrees")
    analysis_scope = statistics.get("analysis_scope")
    if experiment_kind == "smoke":
        normalized_scope = (
            _normalized_identifier(analysis_scope)
            if isinstance(analysis_scope, str)
            else ""
        )
        if (
            integrity_scope != "real_data_smoke_test_not_full_cross_validation"
            or "realdatasmokesubset" not in normalized_scope
            or "notfullcrossvalidation" not in normalized_scope
        ):
            raise ModelReportBundleError(
                "Smoke failure statistics must be explicitly labelled NOT full CV"
            )
    elif (
        not isinstance(analysis_scope, str)
        or "fivefoldoutoffold" not in _normalized_identifier(analysis_scope)
    ):
        raise ModelReportBundleError("Full-CV failure statistics scope is not five-fold OOF")
    sources = statistics.get("sources")
    metrics_path = root / "metrics_per_case.csv"
    if (
        not isinstance(sources, dict)
        or sources.get("crossval_integrity_sha256") != sha256_file(integrity_path)
        or sources.get("metrics_per_case_sha256") != sha256_file(metrics_path)
    ):
        raise ModelReportBundleError("Failure statistics integrity provenance disagrees")
    validation = statistics.get("validation")
    if not isinstance(validation, dict) or not validation or not all(
        value is True for value in validation.values()
    ):
        raise ModelReportBundleError("Failure statistics validation checks did not all pass")

    fields, rows = _csv_rows(
        root / "failure_statistics_per_case_region.csv",
        description="failure statistics per-case-region CSV",
    )
    if not {"case_id", "region"}.issubset(fields):
        raise ModelReportBundleError("Failure statistics CSV lacks case_id/region")
    pairs = [(row.get("case_id", "").strip(), row.get("region", "")) for row in rows]
    if len(pairs) != expected_case_count * len(CANONICAL_REGION_ORDER):
        raise ModelReportBundleError("Failure statistics CSV has the wrong case-region count")
    if any(not case_id or region not in CANONICAL_REGION_ORDER for case_id, region in pairs):
        raise ModelReportBundleError("Failure statistics CSV has invalid case-region identity")
    if len(set(pairs)) != len(pairs):
        raise ModelReportBundleError("Failure statistics CSV contains duplicate case-region rows")
    case_regions: dict[str, set[str]] = {}
    for case_id, region in pairs:
        case_regions.setdefault(case_id, set()).add(region)
    if len(case_regions) != expected_case_count or any(
        regions != set(CANONICAL_REGION_ORDER) for regions in case_regions.values()
    ):
        raise ModelReportBundleError("Each case must have exactly ET,TC,WT failure rows")
    if set(case_regions) != expected_case_ids:
        raise ModelReportBundleError("Failure statistics case IDs disagree with semantic metrics")
    failure_fields, _ = _csv_rows(
        root / "failure_cases.csv", description="ranked failure cases"
    )
    if "case_id" not in failure_fields:
        raise ModelReportBundleError("failure_cases.csv lacks case_id")


def _validate_narratives(
    root: Path, *, backend: str, experiment_kind: str
) -> None:
    summary = _read_text(root / "summary.md", description="summary Markdown")
    weekly = _read_text(root / "weekly_discussion.md", description="weekly Markdown")
    marker = backend
    if marker not in _normalized_identifier(summary.splitlines()[0]):
        raise ModelReportBundleError("summary.md title does not match the experiment model")
    if marker not in _normalized_identifier(weekly.splitlines()[0]):
        raise ModelReportBundleError("weekly_discussion.md title does not match the model")
    if experiment_kind == "fullcv" and (
        "5fold" not in _normalized_identifier(summary.splitlines()[0])
        or "5fold" not in _normalized_identifier(weekly.splitlines()[0])
    ):
        raise ModelReportBundleError("Full-CV narratives are not labelled as five-fold results")
    if experiment_kind == "smoke" and not any(
        marker in _normalized_identifier(summary.splitlines()[0])
        for marker in ("smoke", "preliminary")
    ):
        raise ModelReportBundleError("Smoke summary must be labelled smoke or preliminary")


def _validate_supporting_artifacts(root: Path) -> None:
    for name, description in (
        ("runtime.json", "runtime summary"),
        ("inference_runtime.json", "inference runtime summary"),
        ("gpu_summary.json", "GPU summary"),
    ):
        payload = _load_json_object(root / name, description=description)
        if not payload:
            raise ModelReportBundleError(f"{description} must not be empty")
    for directory_name, suffix in (("config_snapshot", None), ("logs", ".log")):
        directory = root / directory_name
        if not directory.is_dir() or directory.is_symlink():
            raise ModelReportBundleError(f"Required directory is missing: {directory}")
        files = [
            path
            for path in directory.rglob("*")
            if path.is_file() and (suffix is None or path.suffix.casefold() == suffix)
        ]
        if not files or any(path.is_symlink() or path.stat().st_size == 0 for path in files):
            raise ModelReportBundleError(
                f"{directory_name} must contain non-empty, non-symlinked evidence files"
            )
    _read_text(
        root / "official_brats_evaluator.log", description="official evaluator log"
    )


def _artifact_inventory(root: Path) -> list[dict[str, Any]]:
    inventory: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix().casefold()):
        if path.is_symlink():
            raise ModelReportBundleError(f"Report bundle must not contain symlinks: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if relative == REPORT_MANIFEST_FILENAME:
            continue
        if path.name.endswith(".tmp"):
            raise ModelReportBundleError(f"Report bundle contains a staging artifact: {path}")
        inventory.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return inventory


def audit_model_report_bundle(
    experiment_dir: str | Path,
    *,
    expected_case_count: int | None = None,
    expected_folds: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Validate a completed model report directory without modifying it."""

    root = Path(experiment_dir).resolve()
    if not root.is_dir():
        raise ModelReportBundleError(f"Experiment report directory does not exist: {root}")
    for relative in COMMON_REQUIRED_FILES:
        _required_file(root, relative)

    experiment = _load_json_object(root / "experiment.json", description="experiment manifest")
    if experiment.get("schema") != "glioma_model_experiment_v1":
        raise ModelReportBundleError("experiment.json schema is not supported")
    environment = _load_json_object(root / "environment.json", description="environment report")
    if not environment:
        raise ModelReportBundleError("environment.json must not be empty")
    kind, case_count, folds, final_baseline = _resolve_scope(
        experiment, expected_case_count, expected_folds
    )
    backend, model_id, model_display, experiment_id = _validate_identity(
        experiment, environment
    )
    if root.name != experiment_id:
        raise ModelReportBundleError("Experiment directory name disagrees with experiment_id")
    validation = _load_json_object(
        root / "data_validation.json", description="data validation report"
    )
    _validate_data(
        validation,
        expected_case_count=case_count,
        final_baseline=final_baseline,
    )

    integrity_path = root / "crossval_integrity.json"
    integrity = _load_json_object(
        integrity_path, description="cross-validation integrity"
    )
    contract: dict[str, Any] | None = None
    integrity_scope: str | None = None
    if kind == "fullcv":
        for relative in FULL_CV_REQUIRED_FILES:
            _required_file(root, relative)
        artifact_path = root / "crossval_artifact_manifest.json"
        artifact_manifest = _load_json_object(
            artifact_path, description="cross-validation artifact manifest"
        )
        contract = _validate_probability_contract(
            artifact_manifest,
            backend=backend,
            model_id=model_id,
            expected_folds=folds,
        )
        summary = _load_json_object(
            root / "crossval_summary.json", description="cross-validation summary"
        )
        summary_counts = _validate_record_scope(
            summary,
            description="crossval_summary.json",
            backend=backend,
            model_id=model_id,
            expected_case_count=case_count,
            expected_folds=folds,
        )
        integrity_counts = _validate_record_scope(
            integrity,
            description="crossval_integrity.json",
            backend=backend,
            model_id=model_id,
            expected_case_count=case_count,
            expected_folds=folds,
        )
        if summary_counts != integrity_counts:
            raise ModelReportBundleError(
                "Cross-validation summary/integrity fold counts disagree"
            )
        if summary_counts != [251, 250, 250, 250, 250]:
            raise ModelReportBundleError("Full-CV fold counts must be 251,250,250,250,250")
        if integrity.get("evaluation_scope") != "five_fold_out_of_fold":
            raise ModelReportBundleError("Full-CV integrity evaluation scope is invalid")
        _validate_crossval_probability_evidence(
            summary, description="crossval_summary.json", contract=contract
        )
        _validate_crossval_probability_evidence(
            integrity, description="crossval_integrity.json", contract=contract
        )
        if summary.get("probability_conversion") != contract["conversion"]:
            raise ModelReportBundleError("crossval_summary probability conversion disagrees")
        _validate_per_fold_summary(
            summary,
            expected_folds=folds,
            validation_counts=summary_counts,
        )
        _validate_integrity_details(
            integrity,
            artifact_manifest_path=artifact_path,
            contract=contract,
            expected_case_count=case_count,
            expected_folds=folds,
            validation_counts=summary_counts,
        )
        semantic_case_ids = _validate_semantic_outputs(
            root,
            backend=backend,
            model_id=model_id,
            expected_case_count=case_count,
            expected_folds=folds,
            validation_counts=summary_counts,
        )
    else:
        smoke_case_ids = _validate_smoke_integrity(
            integrity,
            backend=backend,
            model_id=model_id,
            expected_case_count=case_count,
            expected_folds=folds,
        )
        _validate_smoke_fold_evidence(
            root,
            backend=backend,
            model_id=model_id,
            experiment_id=experiment_id,
            expected_folds=folds,
            expected_case_ids=smoke_case_ids,
        )
        semantic_case_ids = _validate_semantic_outputs(
            root,
            backend=backend,
            model_id=model_id,
            expected_case_count=case_count,
            expected_folds=folds,
            validation_counts=None,
        )
        if semantic_case_ids != smoke_case_ids:
            raise ModelReportBundleError("Smoke semantic metrics disagree with integrity case IDs")
        integrity_scope = str(integrity["scope"])

    official_case_ids = _validate_official_outputs(root, expected_case_count=case_count)
    if official_case_ids != semantic_case_ids:
        raise ModelReportBundleError("Official and semantic metric case IDs disagree")
    _validate_failure_outputs(
        root,
        expected_case_ids=semantic_case_ids,
        integrity_path=integrity_path,
        experiment_kind=kind,
        integrity_scope=integrity_scope,
    )
    _validate_narratives(root, backend=backend, experiment_kind=kind)
    _validate_supporting_artifacts(root)

    inventory = _artifact_inventory(root)
    required = set(COMMON_REQUIRED_FILES)
    if kind == "fullcv":
        required.update(FULL_CV_REQUIRED_FILES)
    recorded = {item["path"] for item in inventory}
    if not required.issubset(recorded):
        raise ModelReportBundleError("Required artifacts disappeared during inventory")
    return {
        "schema": REPORT_MANIFEST_SCHEMA,
        "valid": True,
        "report_directory": str(root),
        "path_kind": "relative_to_report_directory",
        "experiment_id": experiment_id,
        "experiment_kind": kind,
        "backend": backend,
        "model": model_display,
        "model_id": model_id,
        "expected_case_count": case_count,
        "expected_folds": list(folds),
        "baseline_status": (
            "final_full_cross_validation_baseline"
            if final_baseline
            else "smoke_test_not_final_baseline"
        ),
        "is_final_baseline": final_baseline,
        "probability_contract": contract
        or {
            "scope": "smoke_fold_evidence_only_not_full_cross_validation",
            "canonical_channel_order": list(CANONICAL_REGION_ORDER),
            "fully_pooled_contract_verified": False,
        },
        "artifact_count": len(inventory),
        "artifacts": inventory,
    }


def finalize_model_report_bundle(
    experiment_dir: str | Path,
    *,
    expected_case_count: int | None = None,
    expected_folds: Sequence[int] | None = None,
) -> Path:
    """Audit and atomically publish ``report_manifest.json``."""

    payload = audit_model_report_bundle(
        experiment_dir,
        expected_case_count=expected_case_count,
        expected_folds=expected_folds,
    )
    destination = Path(experiment_dir).resolve() / REPORT_MANIFEST_FILENAME
    write_json_atomic(destination, payload)
    published = _load_json_object(destination, description="published report manifest")
    if published != payload:
        raise ModelReportBundleError("Published report_manifest.json failed verification")
    return destination


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--expected-case-count", type=int)
    parser.add_argument("--expected-folds", type=int, nargs="+")
    parser.add_argument("--audit-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.audit_only:
        payload = audit_model_report_bundle(
            args.experiment_dir,
            expected_case_count=args.expected_case_count,
            expected_folds=args.expected_folds,
        )
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        destination = finalize_model_report_bundle(
            args.experiment_dir,
            expected_case_count=args.expected_case_count,
            expected_folds=args.expected_folds,
        )
        print(destination)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper
    raise SystemExit(main())


__all__ = [
    "ModelReportBundleError",
    "audit_model_report_bundle",
    "build_arg_parser",
    "finalize_model_report_bundle",
    "main",
]
