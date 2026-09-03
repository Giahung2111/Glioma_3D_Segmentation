"""Backend-neutral five-fold out-of-fold evaluation.

The evaluator consumes a single, immutable JSON manifest that declares the
hard-mask and canonical-probability directories for every fold and for the
pooled out-of-fold cohort.  It deliberately knows nothing about trainer
directory layouts.  Model backends only need to produce the declared
artifacts; geometry and semantic metrics remain owned by the common project
evaluator.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from glioma_seg.ensembles.canonical_probabilities import (
    CANONICAL_REGION_ORDER,
    validate_canonical_probability_npz,
)
from glioma_seg.ensembles.canonical_probabilities import (
    SCHEMA_NAME as CANONICAL_PROBABILITY_SCHEMA,
)
from glioma_seg.monitoring.timing import write_json_atomic
from glioma_seg.utils.hashing import sha256_file

from .evaluate import (
    PREDICTION_TTA_STATES,
    EvaluationArtifacts,
    case_id_from_nifti,
    evaluate_directories,
)
from .regions import REGION_ORDER

FOLDS: tuple[int, ...] = (0, 1, 2, 3, 4)
MANIFEST_SCHEMA = "glioma_model_crossval_artifacts_v1"
OWNED_FILES: tuple[str, ...] = (
    "metrics_per_case.csv",
    "metrics_summary.csv",
    "metrics_summary.json",
    "evaluation_protocol.json",
    "crossval_metrics_by_fold.csv",
    "crossval_summary.json",
    "crossval_integrity.json",
)


@dataclass(frozen=True, slots=True)
class ProbabilityContract:
    required: bool
    schema: str
    native_channel_order: tuple[str, ...]
    canonical_channel_order: tuple[str, ...]
    conversion: str


@dataclass(frozen=True, slots=True)
class FoldArtifacts:
    fold: int
    prediction_dir: Path
    probability_dir: Path | None
    provenance: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class ModelCrossvalManifest:
    path: Path
    backend: str
    model_id: str
    model_provenance: Mapping[str, Any]
    prediction_tta_state: str
    folds: tuple[FoldArtifacts, ...]
    pooled_prediction_dir: Path
    pooled_probability_dir: Path | None
    probability_contract: ProbabilityContract


def _read_json(path: Path, *, description: str) -> Any:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"{description} is missing or empty: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{description} is not valid UTF-8 JSON: {path}: {exc}") from exc


def _nonempty_string(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _string_order(value: Any, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a JSON list")
    order = tuple(_nonempty_string(item, field=f"{field}[]") for item in value)
    if not order or len(set(order)) != len(order):
        raise ValueError(f"{field} must contain unique, non-empty names")
    return order


def _mapping(value: Any, *, field: str, require_nonempty: bool = False) -> Mapping[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{field} must be a JSON object with string keys")
    if require_nonempty and not value:
        raise ValueError(f"{field} must not be empty")
    return value


def _resolve_path(value: Any, *, field: str, base_dir: Path) -> Path:
    raw = _nonempty_string(value, field=field)
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = base_dir / candidate
    return candidate.resolve()


def _optional_path(value: Any, *, field: str, base_dir: Path) -> Path | None:
    if value is None:
        return None
    return _resolve_path(value, field=field, base_dir=base_dir)


def load_model_crossval_manifest(path: str | Path) -> ModelCrossvalManifest:
    """Load and strictly normalize a backend artifact manifest."""

    source = Path(path).resolve()
    payload = _read_json(source, description="model cross-validation artifact manifest")
    root = _mapping(payload, field="manifest")
    schema = _nonempty_string(root.get("schema"), field="schema")
    if schema != MANIFEST_SCHEMA:
        raise ValueError(f"Manifest schema must be {MANIFEST_SCHEMA!r}, got {schema!r}")

    backend = _nonempty_string(root.get("backend"), field="backend")
    model_id = _nonempty_string(root.get("model_id"), field="model_id")
    model_provenance = _mapping(
        root.get("model_provenance"), field="model_provenance", require_nonempty=True
    )
    tta_state = _nonempty_string(root.get("prediction_tta_state"), field="prediction_tta_state")
    if tta_state not in PREDICTION_TTA_STATES:
        raise ValueError(
            f"prediction_tta_state must be one of {PREDICTION_TTA_STATES}, got {tta_state!r}"
        )

    contract_raw = _mapping(root.get("probability_contract"), field="probability_contract")
    required = contract_raw.get("required")
    if not isinstance(required, bool):
        raise ValueError("probability_contract.required must be a boolean")
    probability_schema = _nonempty_string(
        contract_raw.get("schema"), field="probability_contract.schema"
    )
    if probability_schema != CANONICAL_PROBABILITY_SCHEMA:
        raise ValueError(
            "probability_contract.schema must be "
            f"{CANONICAL_PROBABILITY_SCHEMA!r}, got {probability_schema!r}"
        )
    native_order = _string_order(
        contract_raw.get("native_channel_order"),
        field="probability_contract.native_channel_order",
    )
    canonical_order = _string_order(
        contract_raw.get("canonical_channel_order"),
        field="probability_contract.canonical_channel_order",
    )
    if canonical_order != CANONICAL_REGION_ORDER:
        raise ValueError(
            "probability_contract.canonical_channel_order must be "
            f"{CANONICAL_REGION_ORDER}, got {canonical_order}"
        )
    conversion = _nonempty_string(
        contract_raw.get("conversion"), field="probability_contract.conversion"
    )
    contract = ProbabilityContract(
        required=required,
        schema=probability_schema,
        native_channel_order=native_order,
        canonical_channel_order=canonical_order,
        conversion=conversion,
    )

    fold_values = root.get("folds")
    if not isinstance(fold_values, list) or len(fold_values) != len(FOLDS):
        raise ValueError("Manifest must declare exactly five fold artifact entries")
    base_dir = source.parent
    fold_records: dict[int, FoldArtifacts] = {}
    for index, value in enumerate(fold_values):
        record = _mapping(value, field=f"folds[{index}]")
        raw_fold = record.get("fold")
        if isinstance(raw_fold, bool) or not isinstance(raw_fold, int) or raw_fold not in FOLDS:
            raise ValueError(f"folds[{index}].fold must be one of {FOLDS}")
        if raw_fold in fold_records:
            raise ValueError(f"Manifest declares fold {raw_fold} more than once")
        provenance_value = record.get("provenance", {})
        provenance = _mapping(provenance_value, field=f"folds[{index}].provenance")
        fold_records[raw_fold] = FoldArtifacts(
            fold=raw_fold,
            prediction_dir=_resolve_path(
                record.get("prediction_dir"),
                field=f"folds[{index}].prediction_dir",
                base_dir=base_dir,
            ),
            probability_dir=_optional_path(
                record.get("probability_dir"),
                field=f"folds[{index}].probability_dir",
                base_dir=base_dir,
            ),
            provenance=provenance,
        )
    if set(fold_records) != set(FOLDS):
        raise ValueError(f"Manifest fold IDs must be exactly {FOLDS}")

    pooled_prediction_dir = _resolve_path(
        root.get("pooled_prediction_dir"), field="pooled_prediction_dir", base_dir=base_dir
    )
    pooled_probability_dir = _optional_path(
        root.get("pooled_probability_dir"),
        field="pooled_probability_dir",
        base_dir=base_dir,
    )
    probability_paths = [fold_records[fold].probability_dir for fold in FOLDS]
    any_probabilities = pooled_probability_dir is not None or any(
        value is not None for value in probability_paths
    )
    all_probabilities = pooled_probability_dir is not None and all(
        value is not None for value in probability_paths
    )
    if any_probabilities and not all_probabilities:
        raise ValueError(
            "Canonical probability retention is partial: all five fold directories and the "
            "pooled directory must be declared together"
        )
    if contract.required and not all_probabilities:
        raise ValueError("The probability contract requires complete fold and pooled directories")

    return ModelCrossvalManifest(
        path=source,
        backend=backend,
        model_id=model_id,
        model_provenance=model_provenance,
        prediction_tta_state=tta_state,
        folds=tuple(fold_records[fold] for fold in FOLDS),
        pooled_prediction_dir=pooled_prediction_dir,
        pooled_probability_dir=pooled_probability_dir,
        probability_contract=contract,
    )


def _nifti_inventory(directory: Path) -> dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"NIfTI directory does not exist: {directory}")
    result: dict[str, Path] = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file() or not (path.name.endswith(".nii.gz") or path.suffix == ".nii"):
            continue
        case_id = case_id_from_nifti(path)
        if not case_id or case_id in result:
            raise ValueError(f"Duplicate/empty NIfTI case ID in {directory}: {path.name}")
        if path.stat().st_size == 0:
            raise ValueError(f"Empty NIfTI artifact: {path}")
        result[case_id] = path.resolve()
    return result


def _probability_inventory(directory: Path) -> dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Canonical probability directory does not exist: {directory}")
    result: dict[str, Path] = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() != ".npz":
            continue
        case_id = path.stem
        if not case_id or case_id in result:
            raise ValueError(f"Duplicate/empty canonical probability case ID: {path}")
        if path.stat().st_size == 0:
            raise ValueError(f"Empty canonical probability artifact: {path}")
        result[case_id] = path.resolve()
    return result


def _assert_exact_ids(actual: Mapping[str, Path], expected: set[str], *, description: str) -> None:
    missing = sorted(expected - set(actual))
    extra = sorted(set(actual) - expected)
    if missing or extra:
        raise ValueError(
            f"{description} inventory mismatch: expected={len(expected)}, actual={len(actual)}, "
            f"missing={missing[:10]}, extra={extra[:10]}"
        )


def _validate_splits(
    splits_json: Path, dataset_case_ids: set[str]
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    payload = _read_json(splits_json, description="five-fold split JSON")
    if not isinstance(payload, list) or len(payload) != len(FOLDS):
        raise ValueError(f"Expected exactly five folds in {splits_json}")
    normalized: list[dict[str, Any]] = []
    fold_by_case: dict[str, int] = {}
    for fold in FOLDS:
        split = payload[fold]
        if (
            not isinstance(split, dict)
            or not isinstance(split.get("train"), list)
            or not isinstance(split.get("val"), list)
        ):
            raise ValueError(f"Fold {fold} must contain train and val lists")
        train_list = [str(value) for value in split["train"]]
        val_list = [str(value) for value in split["val"]]
        train = set(train_list)
        val = set(val_list)
        if len(train) != len(train_list) or len(val) != len(val_list):
            raise ValueError(f"Fold {fold} contains duplicate case IDs")
        if not train or not val:
            raise ValueError(f"Fold {fold} has an empty train or validation partition")
        overlap = sorted(train & val)
        if overlap:
            raise ValueError(f"Fold {fold} train/validation leakage: {overlap[:10]}")
        if train | val != dataset_case_ids:
            missing = sorted(dataset_case_ids - (train | val))
            unknown = sorted((train | val) - dataset_case_ids)
            raise ValueError(
                f"Fold {fold} does not partition the complete dataset: "
                f"missing={missing[:10]}, unknown={unknown[:10]}"
            )
        for case_id in val_list:
            if case_id in fold_by_case:
                raise ValueError(
                    f"Case {case_id} is validated in folds {fold_by_case[case_id]} and {fold}"
                )
            fold_by_case[case_id] = fold
        normalized.append({"fold": fold, "train": train_list, "val": val_list})
    if set(fold_by_case) != dataset_case_ids:
        missing = sorted(dataset_case_ids - set(fold_by_case))
        raise ValueError(f"Five-fold validation union is incomplete: {missing[:10]}")
    return normalized, fold_by_case


def _summary_map(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for row in rows:
        metric = str(row.get("metric", "")).strip()
        if metric in {"Dice", "HD95"}:
            output[metric] = {region: float(row[region]) for region in REGION_ORDER}
    if set(output) != {"Dice", "HD95"}:
        raise ValueError("Evaluation summary must contain Dice and HD95")
    return output


def _macro_statistics(
    per_fold: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    means: dict[str, dict[str, float]] = {}
    standard_deviations: dict[str, dict[str, float]] = {}
    for metric in ("Dice", "HD95"):
        means[metric] = {}
        standard_deviations[metric] = {}
        for region in REGION_ORDER:
            values = np.asarray(
                [float(record["metrics"][metric][region]) for record in per_fold],
                dtype=float,
            )
            if values.size != len(FOLDS) or not np.all(np.isfinite(values)):
                raise ValueError(f"Non-finite/missing per-fold {metric} {region} values")
            means[metric][region] = float(np.mean(values))
            standard_deviations[metric][region] = float(np.std(values, ddof=1))
    return means, standard_deviations


def _write_fold_summary_csv(path: Path, fold_artifacts: Sequence[EvaluationArtifacts]) -> None:
    fields = (
        "fold",
        "case_count",
        "metric",
        "unit",
        "direction",
        "ET",
        "TC",
        "WT",
        "ET_n_valid",
        "ET_n_excluded",
        "TC_n_valid",
        "TC_n_excluded",
        "WT_n_valid",
        "WT_n_excluded",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for fold, artifacts in zip(FOLDS, fold_artifacts, strict=True):
            for row in artifacts.summary:
                writer.writerow({"fold": fold, "case_count": len(artifacts.cases), **row})


def _owned_targets(destination: Path) -> list[Path]:
    return [destination / name for name in (*OWNED_FILES, "fold_metrics")]


def _refuse_existing_outputs(destination: Path, *, overwrite: bool) -> None:
    if overwrite:
        return
    existing = [str(path) for path in _owned_targets(destination) if path.exists()]
    if existing:
        raise FileExistsError(
            "Cross-validation artifacts already exist; pass --overwrite only after verifying "
            f"the experiment identity: {existing}"
        )


def _publish_stage(stage: Path, destination: Path, *, overwrite: bool) -> None:
    names = (*OWNED_FILES, "fold_metrics")
    for name in names:
        if not (stage / name).exists():
            raise RuntimeError(f"Staged cross-validation artifact is missing: {stage / name}")
    destination.mkdir(parents=True, exist_ok=True)
    existing = [destination / name for name in names if (destination / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Cross-validation artifacts appeared during evaluation; refusing publication: "
            f"{[str(path) for path in existing]}"
        )

    with tempfile.TemporaryDirectory(prefix=".model_cv_backup_", dir=destination) as temporary:
        backup = Path(temporary)
        moved_new: list[Path] = []
        moved_old: list[tuple[Path, Path]] = []
        try:
            for target in existing:
                saved = backup / target.name
                os.replace(target, saved)
                moved_old.append((saved, target))
            for name in names:
                target = destination / name
                os.replace(stage / name, target)
                moved_new.append(target)
        except Exception:
            for target in reversed(moved_new):
                if target.is_dir():
                    shutil.rmtree(target, ignore_errors=True)
                else:
                    target.unlink(missing_ok=True)
            for saved, target in reversed(moved_old):
                if saved.exists():
                    os.replace(saved, target)
            raise


def _assert_same_nifti_labels(first: Path, second: Path, *, case_id: str) -> None:
    """Verify that the pooled mask is the same OOF mask, allowing gzip byte differences."""

    if sha256_file(first) == sha256_file(second):
        return
    try:
        import nibabel as nib
    except ImportError as exc:  # pragma: no cover - dependency preflight is environment-specific
        raise RuntimeError("nibabel is required to verify pooled OOF masks") from exc
    try:
        first_image: Any = nib.load(str(first))
        second_image: Any = nib.load(str(second))
        first_data = np.asanyarray(first_image.dataobj)
        second_data = np.asanyarray(second_image.dataobj)
    except Exception as exc:
        raise ValueError(f"Unable to compare pooled OOF mask for {case_id}: {exc}") from exc
    if first_data.shape != second_data.shape or not np.array_equal(first_data, second_data):
        raise ValueError(
            f"Pooled OOF prediction for {case_id} differs from its declared fold prediction"
        )


def _validate_probability_directory(
    *,
    directory: Path,
    expected: set[str],
    ground_truth: Mapping[str, Path],
    contract: ProbabilityContract,
    description: str,
) -> dict[str, Any]:
    inventory = _probability_inventory(directory)
    _assert_exact_ids(inventory, expected, description=description)
    for case_id in sorted(expected):
        validate_canonical_probability_npz(
            inventory[case_id],
            reference_nifti=ground_truth[case_id],
            expected_case_id=case_id,
            expected_native_channel_order=contract.native_channel_order,
            expected_conversion=contract.conversion,
        )
    return {
        "directory": str(directory),
        "count": len(inventory),
        "bytes": sum(path.stat().st_size for path in inventory.values()),
    }


def evaluate_model_cross_validation(
    *,
    ground_truth_dir: str | Path,
    splits_json: str | Path,
    artifact_manifest: str | Path,
    output_dir: str | Path,
    expected_case_count: int = 1251,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Verify and evaluate a complete backend-neutral five-fold OOF experiment."""

    if (
        isinstance(expected_case_count, bool)
        or not isinstance(expected_case_count, int)
        or expected_case_count < 1
    ):
        raise ValueError("expected_case_count must be a positive integer")
    gt_directory = Path(ground_truth_dir).resolve()
    split_path = Path(splits_json).resolve()
    destination = Path(output_dir).resolve()
    manifest = load_model_crossval_manifest(artifact_manifest)
    _refuse_existing_outputs(destination, overwrite=overwrite)

    gt_files = _nifti_inventory(gt_directory)
    if len(gt_files) != expected_case_count:
        raise ValueError(
            f"Expected {expected_case_count} ground-truth cases, found {len(gt_files)}"
        )
    dataset_case_ids = set(gt_files)
    splits, fold_by_case = _validate_splits(split_path, dataset_case_ids)

    fold_inventories: list[dict[str, Any]] = []
    fold_prediction_by_case: dict[str, Path] = {}
    probabilities_declared = manifest.pooled_probability_dir is not None
    for split, fold_artifacts in zip(splits, manifest.folds, strict=True):
        fold = int(split["fold"])
        if fold_artifacts.fold != fold:
            raise ValueError(
                "Manifest fold order mismatch: "
                f"split fold {fold}, artifact fold {fold_artifacts.fold}"
            )
        expected = set(split["val"])
        masks = _nifti_inventory(fold_artifacts.prediction_dir)
        _assert_exact_ids(masks, expected, description=f"fold {fold} validation mask")
        for case_id, path in masks.items():
            if case_id in fold_prediction_by_case:
                raise ValueError(f"Prediction for {case_id} is declared by more than one fold")
            fold_prediction_by_case[case_id] = path
        probability_inventory: dict[str, Any] | None = None
        if probabilities_declared:
            assert fold_artifacts.probability_dir is not None
            probability_inventory = _validate_probability_directory(
                directory=fold_artifacts.probability_dir,
                expected=expected,
                ground_truth=gt_files,
                contract=manifest.probability_contract,
                description=f"fold {fold} canonical probability",
            )
        fold_inventories.append(
            {
                "fold": fold,
                "train_case_count": len(split["train"]),
                "validation_case_count": len(split["val"]),
                "prediction_dir": str(fold_artifacts.prediction_dir),
                "prediction_count": len(masks),
                "prediction_bytes": sum(path.stat().st_size for path in masks.values()),
                "probability_inventory": probability_inventory,
                "provenance": dict(fold_artifacts.provenance),
            }
        )
    _assert_exact_ids(
        fold_prediction_by_case,
        dataset_case_ids,
        description="union of fold validation masks",
    )

    pooled_files = _nifti_inventory(manifest.pooled_prediction_dir)
    _assert_exact_ids(pooled_files, dataset_case_ids, description="pooled OOF prediction")
    pooled_probability_inventory: dict[str, Any] | None = None
    if probabilities_declared:
        assert manifest.pooled_probability_dir is not None
        pooled_probability_inventory = _validate_probability_directory(
            directory=manifest.pooled_probability_dir,
            expected=dataset_case_ids,
            ground_truth=gt_files,
            contract=manifest.probability_contract,
            description="pooled canonical probability",
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{destination.name}.model_cv_stage_", dir=destination.parent
    ) as temporary:
        stage = Path(temporary)
        fold_metric_artifacts: list[EvaluationArtifacts] = []
        per_fold: list[dict[str, Any]] = []
        for split, fold_artifacts in zip(splits, manifest.folds, strict=True):
            fold = int(split["fold"])
            print(
                f"[MODEL-CV-EVAL] Fold {fold + 1}/5: evaluating {len(split['val'])} "
                "out-of-fold cases...",
                flush=True,
            )
            artifacts = evaluate_directories(
                gt_directory,
                fold_artifacts.prediction_dir,
                stage / "fold_metrics" / f"fold_{fold}",
                expected_case_ids=split["val"],
                strict_predictions=True,
                split_source=split_path,
                fold=fold,
                prediction_provenance=(
                    f"{manifest.backend}:{manifest.model_id} fold {fold} artifact manifest "
                    f"{manifest.path}"
                ),
                prediction_tta_state=manifest.prediction_tta_state,
            )
            fold_metric_artifacts.append(artifacts)
            per_fold.append(
                {
                    "fold": fold,
                    "case_count": len(artifacts.cases),
                    "metrics": _summary_map(artifacts.summary),
                }
            )

        pooled_artifacts = evaluate_directories(
            gt_directory,
            manifest.pooled_prediction_dir,
            stage,
            expected_case_ids=sorted(fold_by_case),
            strict_predictions=True,
            split_source=split_path,
            fold=None,
            prediction_provenance=(
                f"{manifest.backend}:{manifest.model_id} pooled OOF artifact manifest "
                f"{manifest.path}"
            ),
            prediction_tta_state=manifest.prediction_tta_state,
        )
        for case_id in sorted(dataset_case_ids):
            _assert_same_nifti_labels(
                fold_prediction_by_case[case_id], pooled_files[case_id], case_id=case_id
            )

        _write_fold_summary_csv(stage / "crossval_metrics_by_fold.csv", fold_metric_artifacts)
        macro_mean, macro_std = _macro_statistics(per_fold)
        validation_counts = [len(split["val"]) for split in splits]
        probability_contract_payload = {
            "required": manifest.probability_contract.required,
            "schema": manifest.probability_contract.schema,
            "native_channel_order": list(manifest.probability_contract.native_channel_order),
            "canonical_channel_order": list(manifest.probability_contract.canonical_channel_order),
            "conversion": manifest.probability_contract.conversion,
        }
        integrity = {
            "valid": True,
            "schema": "glioma_model_crossval_integrity_v1",
            "evaluation_scope": "five_fold_out_of_fold",
            "backend": manifest.backend,
            "model_id": manifest.model_id,
            "model_provenance": dict(manifest.model_provenance),
            "prediction_tta_state": manifest.prediction_tta_state,
            "artifact_manifest": str(manifest.path),
            "artifact_manifest_sha256": sha256_file(manifest.path),
            "folds": list(FOLDS),
            "total_cases": len(dataset_case_ids),
            "validation_case_counts": validation_counts,
            "split_source": str(split_path),
            "split_sha256": sha256_file(split_path),
            "ground_truth_dir": str(gt_directory),
            "pooled_prediction_dir": str(manifest.pooled_prediction_dir),
            "each_case_validated_once": True,
            "pooled_prediction_count": len(pooled_files),
            "pooled_matches_fold_predictions": True,
            "probabilities_retained": probabilities_declared,
            "probability_source_channel_order": list(
                manifest.probability_contract.native_channel_order
            ),
            "probability_canonical_order": list(
                manifest.probability_contract.canonical_channel_order
            ),
            "probability_contract": probability_contract_payload,
            "probability_validation_reference": (
                "exact ground-truth NIfTI identity and original-space geometry"
            ),
            "pooled_probability_inventory": pooled_probability_inventory,
            "fold_inventories": fold_inventories,
        }
        summary = {
            "valid": True,
            "evaluation_scope": "five_fold_out_of_fold",
            "backend": manifest.backend,
            "model_id": manifest.model_id,
            "model_provenance": dict(manifest.model_provenance),
            "prediction_tta_state": manifest.prediction_tta_state,
            "folds": list(FOLDS),
            "total_cases": len(dataset_case_ids),
            "validation_case_counts": validation_counts,
            "split_source": str(split_path),
            "each_case_validated_once": True,
            "probabilities_retained": probabilities_declared,
            "probability_source_channel_order": list(
                manifest.probability_contract.native_channel_order
            ),
            "probability_canonical_order": list(
                manifest.probability_contract.canonical_channel_order
            ),
            "probability_conversion": manifest.probability_contract.conversion,
            "per_fold": per_fold,
            "macro_mean": macro_mean,
            "macro_std": macro_std,
            "macro_std_definition": (
                "sample standard deviation across five fold-level means (ddof=1)"
            ),
            "pooled": _summary_map(pooled_artifacts.summary),
            "pooled_definition": (
                f"finite-case mean over all {len(dataset_case_ids)} out-of-fold predictions"
            ),
        }
        write_json_atomic(stage / "crossval_integrity.json", integrity)
        write_json_atomic(stage / "crossval_summary.json", summary)
        protocol_path = stage / "evaluation_protocol.json"
        protocol = _read_json(protocol_path, description="pooled evaluation protocol")
        protocol.update(
            {
                "evaluation_scope": "five_fold_out_of_fold",
                "backend": manifest.backend,
                "model_id": manifest.model_id,
                "folds": list(FOLDS),
                "each_case_validated_once": True,
                "fold_assignment_source": str(split_path),
                "artifact_manifest": str(manifest.path),
                "probability_contract": probability_contract_payload,
                "crossval_integrity": str((destination / "crossval_integrity.json").resolve()),
            }
        )
        write_json_atomic(protocol_path, protocol)
        _publish_stage(stage, destination, overwrite=overwrite)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify and evaluate backend-neutral five-fold out-of-fold predictions from an "
            "artifact manifest."
        )
    )
    parser.add_argument("--ground-truth-dir", type=Path, required=True)
    parser.add_argument("--splits-json", type=Path, required=True)
    parser.add_argument("--artifact-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-case-count", type=int, default=1251)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    summary = evaluate_model_cross_validation(
        ground_truth_dir=args.ground_truth_dir,
        splits_json=args.splits_json,
        artifact_manifest=args.artifact_manifest,
        output_dir=args.output_dir,
        expected_case_count=args.expected_case_count,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


__all__ = [
    "FOLDS",
    "MANIFEST_SCHEMA",
    "ModelCrossvalManifest",
    "ProbabilityContract",
    "evaluate_model_cross_validation",
    "load_model_crossval_manifest",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
