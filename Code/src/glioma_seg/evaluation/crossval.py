"""Strict five-fold out-of-fold evaluation for the BraTS 2023 baseline.

This module does not train, predict, or modify nnU-Net. It verifies the
official five-fold partition and the artifacts produced by nnU-Net, evaluates
each fold and the accumulated out-of-fold masks through the existing semantic
metric implementation, and publishes aggregate provenance for reporting.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from glioma_seg.ensembles.nnunet_probabilities import (
    NNUNET_REGION_CHANNEL_ORDER,
    validate_brats_region_probability_contract,
)
from glioma_seg.monitoring.timing import write_json_atomic
from glioma_seg.utils.hashing import sha256_file

from .evaluate import EvaluationArtifacts, case_id_from_nifti, evaluate_directories
from .regions import REGION_ORDER

FOLDS: tuple[int, ...] = (0, 1, 2, 3, 4)
OWNED_FILES: tuple[str, ...] = (
    "metrics_per_case.csv",
    "metrics_summary.csv",
    "metrics_summary.json",
    "evaluation_protocol.json",
    "crossval_metrics_by_fold.csv",
    "crossval_summary.json",
    "crossval_integrity.json",
)


def _read_json(path: Path, *, description: str) -> Any:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"{description} is missing or empty: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{description} is not valid UTF-8 JSON: {path}: {exc}") from exc


def _nifti_inventory(directory: Path) -> dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"NIfTI directory does not exist: {directory}")
    result: dict[str, Path] = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file() or not (path.name.endswith(".nii.gz") or path.suffix == ".nii"):
            continue
        case_id = case_id_from_nifti(path)
        if case_id in result:
            raise ValueError(f"Duplicate NIfTI case ID {case_id!r} in {directory}")
        if path.stat().st_size == 0:
            raise ValueError(f"Empty NIfTI artifact: {path}")
        result[case_id] = path.resolve()
    return result


def _suffix_inventory(directory: Path, suffix: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in sorted(directory.glob(f"*{suffix}")):
        if not path.is_file():
            continue
        case_id = path.name[: -len(suffix)]
        if not case_id or case_id in result:
            raise ValueError(f"Duplicate/empty {suffix} case ID in {directory}: {path.name}")
        if path.stat().st_size == 0:
            raise ValueError(f"Empty {suffix} artifact: {path}")
        result[case_id] = path.resolve()
    return result


def _assert_exact_ids(
    actual: Mapping[str, Path], expected: set[str], *, description: str
) -> None:
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
    payload = _read_json(splits_json, description="nnU-Net splits_final.json")
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
        if metric not in {"Dice", "HD95"}:
            continue
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
                [float(record["metrics"][metric][region]) for record in per_fold], dtype=float
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


def _publish_stage(stage: Path, destination: Path, *, overwrite: bool) -> None:
    targets = [destination / name for name in OWNED_FILES]
    targets.append(destination / "fold_metrics")
    existing = [path for path in targets if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Cross-validation artifacts already exist; pass --overwrite only after verifying the "
            f"experiment identity: {[str(path) for path in existing]}"
        )
    destination.mkdir(parents=True, exist_ok=True)
    for target in existing:
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()
    for name in (*OWNED_FILES, "fold_metrics"):
        source = stage / name
        if not source.exists():
            raise RuntimeError(f"Staged cross-validation artifact is missing: {source}")
        os.replace(source, destination / name)


def _refuse_existing_outputs(destination: Path, *, overwrite: bool) -> None:
    if overwrite:
        return
    targets = [destination / name for name in (*OWNED_FILES, "fold_metrics")]
    existing = [str(path) for path in targets if path.exists()]
    if existing:
        raise FileExistsError(
            "Cross-validation artifacts already exist; pass --overwrite only after verifying the "
            f"experiment identity: {existing}"
        )


def evaluate_five_fold_cross_validation(
    *,
    ground_truth_dir: str | Path,
    model_dir: str | Path,
    accumulated_prediction_dir: str | Path,
    splits_json: str | Path,
    dataset_json: str | Path,
    output_dir: str | Path,
    expected_case_count: int = 1251,
    require_probabilities: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Verify and evaluate one complete five-fold nnU-Net OOF experiment."""

    if expected_case_count < 1:
        raise ValueError("expected_case_count must be positive")
    gt_directory = Path(ground_truth_dir).resolve()
    trainer_directory = Path(model_dir).resolve()
    accumulated_directory = Path(accumulated_prediction_dir).resolve()
    split_path = Path(splits_json).resolve()
    dataset_path = Path(dataset_json).resolve()
    destination = Path(output_dir).resolve()
    _refuse_existing_outputs(destination, overwrite=overwrite)
    gt_files = _nifti_inventory(gt_directory)
    if len(gt_files) != expected_case_count:
        raise ValueError(
            f"Expected {expected_case_count} ground-truth cases, found {len(gt_files)}"
        )
    dataset_case_ids = set(gt_files)
    splits, fold_by_case = _validate_splits(split_path, dataset_case_ids)
    probability_provenance = validate_brats_region_probability_contract(dataset_path)

    if not trainer_directory.is_dir():
        raise FileNotFoundError(f"nnU-Net trained model directory is missing: {trainer_directory}")
    trainer_dataset_path = trainer_directory / "dataset.json"
    trainer_plans_path = trainer_directory / "plans.json"
    for required in (trainer_dataset_path, trainer_plans_path):
        if not required.is_file() or required.stat().st_size == 0:
            raise FileNotFoundError(f"Trained-model provenance artifact is missing: {required}")
    validate_brats_region_probability_contract(trainer_dataset_path)
    source_dataset_payload = _read_json(dataset_path, description="source dataset.json")
    trainer_dataset_payload = _read_json(
        trainer_dataset_path, description="trained-model dataset.json"
    )
    if trainer_dataset_payload != source_dataset_payload:
        raise ValueError("Source and trained-model dataset.json semantics differ")
    accumulated_files = _nifti_inventory(accumulated_directory)
    _assert_exact_ids(
        accumulated_files, dataset_case_ids, description="accumulated OOF prediction"
    )
    for required_name in ("dataset.json", "plans.json", "summary.json"):
        required = accumulated_directory / required_name
        if not required.is_file() or required.stat().st_size == 0:
            raise FileNotFoundError(f"Official accumulated-CV artifact is missing: {required}")
    accumulated_dataset_path = accumulated_directory / "dataset.json"
    validate_brats_region_probability_contract(accumulated_dataset_path)
    if _read_json(
        accumulated_dataset_path, description="accumulated dataset.json"
    ) != source_dataset_payload:
        raise ValueError("Source and accumulated dataset.json semantics differ")
    if _read_json(
        accumulated_directory / "plans.json", description="accumulated plans.json"
    ) != _read_json(trainer_plans_path, description="trained-model plans.json"):
        raise ValueError("Trained-model and accumulated plans.json semantics differ")

    fold_inventories: list[dict[str, Any]] = []
    any_probability_artifact = False
    all_probabilities_complete = True
    for split in splits:
        fold = int(split["fold"])
        expected = set(split["val"])
        fold_directory = trainer_directory / f"fold_{fold}"
        validation_directory = fold_directory / "validation"
        checkpoint = fold_directory / "checkpoint_final.pth"
        validation_summary = validation_directory / "summary.json"
        for required in (checkpoint, validation_summary):
            if not required.is_file() or required.stat().st_size == 0:
                raise FileNotFoundError(f"Completed fold artifact is missing: {required}")
        masks = _nifti_inventory(validation_directory)
        _assert_exact_ids(masks, expected, description=f"fold {fold} validation mask")
        probabilities = _suffix_inventory(validation_directory, ".npz")
        properties = _suffix_inventory(validation_directory, ".pkl")
        any_probability_artifact = any_probability_artifact or bool(probabilities or properties)
        probabilities_complete = set(probabilities) == expected and set(properties) == expected
        all_probabilities_complete = all_probabilities_complete and probabilities_complete
        if (probabilities or properties) and not probabilities_complete:
            _assert_exact_ids(probabilities, expected, description=f"fold {fold} probability")
            _assert_exact_ids(
                properties,
                expected,
                description=f"fold {fold} probability properties",
            )
        fold_inventories.append(
            {
                "fold": fold,
                "train_case_count": len(split["train"]),
                "validation_case_count": len(split["val"]),
                "mask_count": len(masks),
                "npz_count": len(probabilities),
                "pkl_count": len(properties),
                "npz_bytes": sum(path.stat().st_size for path in probabilities.values()),
                "pkl_bytes": sum(path.stat().st_size for path in properties.values()),
                "checkpoint_final": str(checkpoint),
                "validation_directory": str(validation_directory),
            }
        )
    if require_probabilities and not all_probabilities_complete:
        raise ValueError(
            "Complete paired .npz/.pkl probability artifacts are required for all folds"
        )
    if any_probability_artifact and not all_probabilities_complete:
        raise ValueError("Partial probability retention detected across the five folds")
    probabilities_retained = bool(any_probability_artifact and all_probabilities_complete)

    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".crossval_stage_", dir=destination) as temporary:
        stage = Path(temporary)
        fold_artifacts: list[EvaluationArtifacts] = []
        per_fold: list[dict[str, Any]] = []
        for split in splits:
            fold = int(split["fold"])
            print(
                f"[CV-EVAL] Fold {fold + 1}/5: evaluating {len(split['val'])} "
                "out-of-fold cases...",
                flush=True,
            )
            artifacts = evaluate_directories(
                gt_directory,
                trainer_directory / f"fold_{fold}" / "validation",
                stage / "fold_metrics" / f"fold_{fold}",
                expected_case_ids=split["val"],
                strict_predictions=True,
                split_source=split_path,
                fold=fold,
                prediction_provenance=(
                    f"nnU-Net perform_actual_validation output (fold_{fold}/validation)"
                ),
                prediction_tta_state="DEFAULT_MIRRORING",
            )
            fold_artifacts.append(artifacts)
            per_fold.append(
                {
                    "fold": fold,
                    "case_count": len(artifacts.cases),
                    "metrics": _summary_map(artifacts.summary),
                }
            )
            print(f"[CV-EVAL] Fold {fold + 1}/5 complete.", flush=True)
        print(
            f"[CV-EVAL] Pooled evaluation: {len(fold_by_case)} out-of-fold cases...",
            flush=True,
        )
        pooled_artifacts = evaluate_directories(
            gt_directory,
            accumulated_directory,
            stage,
            expected_case_ids=sorted(fold_by_case),
            strict_predictions=True,
            split_source=split_path,
            fold=None,
            prediction_provenance=(
                "nnUNetv2_accumulate_crossval_results output from folds 0,1,2,3,4"
            ),
            prediction_tta_state="DEFAULT_MIRRORING",
        )
        print("[CV-EVAL] Pooled evaluation complete; publishing verified artifacts.", flush=True)
        _write_fold_summary_csv(stage / "crossval_metrics_by_fold.csv", fold_artifacts)
        macro_mean, macro_std = _macro_statistics(per_fold)
        validation_counts = [len(split["val"]) for split in splits]
        integrity = {
            "valid": True,
            "folds": list(FOLDS),
            "total_cases": len(dataset_case_ids),
            "validation_case_counts": validation_counts,
            "split_source": str(split_path),
            "split_sha256": sha256_file(split_path),
            "ground_truth_dir": str(gt_directory),
            "model_dir": str(trainer_directory),
            "accumulated_prediction_dir": str(accumulated_directory),
            "each_case_validated_once": True,
            "accumulated_prediction_count": len(accumulated_files),
            "probabilities_retained": probabilities_retained,
            "probability_source_channel_order": list(NNUNET_REGION_CHANNEL_ORDER),
            "probability_canonical_order": list(REGION_ORDER),
            "probability_reorder_indices": probability_provenance[
                "canonical_reorder_indices"
            ],
            "probability_npz_key": "probabilities",
            "source_dataset_json_sha256": sha256_file(dataset_path),
            "trained_model_dataset_json_sha256": sha256_file(trainer_dataset_path),
            "accumulated_dataset_json_sha256": sha256_file(accumulated_dataset_path),
            "pickle_opened_during_audit": False,
            "fold_inventories": fold_inventories,
        }
        summary = {
            "valid": True,
            "evaluation_scope": "five_fold_out_of_fold",
            "folds": list(FOLDS),
            "total_cases": len(dataset_case_ids),
            "validation_case_counts": validation_counts,
            "split_source": str(split_path),
            "each_case_validated_once": True,
            "probabilities_retained": probabilities_retained,
            "probability_source_channel_order": list(NNUNET_REGION_CHANNEL_ORDER),
            "probability_canonical_order": list(REGION_ORDER),
            "per_fold": per_fold,
            "macro_mean": macro_mean,
            "macro_std": macro_std,
            "macro_std_definition": (
                "sample standard deviation across five fold-level means (ddof=1)"
            ),
            "pooled": _summary_map(pooled_artifacts.summary),
            "pooled_definition": "finite-case mean over all 1251 out-of-fold predictions",
        }
        write_json_atomic(stage / "crossval_integrity.json", integrity)
        write_json_atomic(stage / "crossval_summary.json", summary)
        protocol_path = stage / "evaluation_protocol.json"
        protocol = _read_json(protocol_path, description="pooled evaluation protocol")
        protocol.update(
            {
                "evaluation_scope": "five_fold_out_of_fold",
                "folds": list(FOLDS),
                "each_case_validated_once": True,
                "fold_assignment_source": str(split_path),
                "crossval_integrity": str((destination / "crossval_integrity.json").resolve()),
            }
        )
        write_json_atomic(protocol_path, protocol)
        _publish_stage(stage, destination, overwrite=overwrite)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify and evaluate complete five-fold nnU-Net out-of-fold predictions."
    )
    parser.add_argument("--ground-truth-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--accumulated-prediction-dir", type=Path, required=True)
    parser.add_argument("--splits-json", type=Path, required=True)
    parser.add_argument("--dataset-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-case-count", type=int, default=1251)
    parser.add_argument("--require-probabilities", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    summary = evaluate_five_fold_cross_validation(
        ground_truth_dir=args.ground_truth_dir,
        model_dir=args.model_dir,
        accumulated_prediction_dir=args.accumulated_prediction_dir,
        splits_json=args.splits_json,
        dataset_json=args.dataset_json,
        output_dir=args.output_dir,
        expected_case_count=args.expected_case_count,
        require_probabilities=args.require_probabilities,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
