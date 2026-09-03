"""Synchronize MedNeXt fold evidence and aggregate runtime telemetry.

The Windows runner historically performed this bookkeeping in PowerShell.
Keeping the operation in Python gives Windows and Linux runners one artifact
contract and prevents shell-specific JSON handling from drifting.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from glioma_seg.monitoring.timing import write_json_atomic
from glioma_seg.reporting.crossval import aggregate_crossval_telemetry

EVIDENCE_FILES = (
    "fold_manifest.json",
    "runtime.json",
    "gpu_summary.json",
    "gpu_samples.csv",
    "train_history.json",
    "validation_summary.json",
)


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _copy_fold_evidence(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for name in EVIDENCE_FILES:
        source_path = source / name
        if source_path.is_file():
            shutil.copy2(source_path, destination / name)
    for source_path in source.glob("gpu_samples_segment_*"):
        if source_path.is_file():
            shutil.copy2(source_path, destination / source_path.name)


def sync_and_aggregate_mednext(
    project_root: Path,
    experiment_id: str,
    *,
    folds: Sequence[int],
    smoke: bool,
) -> dict[str, Any]:
    """Publish per-fold evidence and common runtime/inference summaries."""

    root = project_root.resolve()
    normalized_folds = tuple(int(fold) for fold in folds)
    expected_folds = (0,) if smoke else (0, 1, 2, 3, 4)
    if normalized_folds != expected_folds:
        raise ValueError(f"Expected folds {expected_folds}, got {normalized_folds}")
    workspace = root / "Workspace"
    report_dir = workspace / "reports" / experiment_id
    result_root = workspace / "model_results" / "mednext" / experiment_id
    experiment_path = report_dir / "experiment.json"
    experiment = _load_object(experiment_path)
    if experiment.get("experiment_id") != experiment_id or experiment.get("backend") != "mednext":
        raise ValueError("Experiment identity does not match MedNeXt telemetry aggregation")

    total_inference_seconds = 0.0
    case_ids: list[str] = []
    source_summaries: list[str] = []
    for fold in normalized_folds:
        source = result_root / f"fold_{fold}"
        destination = report_dir / "folds" / f"fold_{fold}"
        _copy_fold_evidence(source, destination)
        summary_path = destination / "validation_summary.json"
        summary = _load_object(summary_path)
        if summary.get("valid") is not True or summary.get("fold") != fold:
            raise ValueError(f"Fold {fold} validation summary is invalid: {summary_path}")
        fold_ids = summary.get("case_ids")
        if (
            not isinstance(fold_ids, list)
            or not fold_ids
            or any(not isinstance(case_id, str) or not case_id for case_id in fold_ids)
        ):
            raise ValueError(f"Fold {fold} validation case IDs are invalid")
        if summary.get("case_count") != len(fold_ids):
            raise ValueError(f"Fold {fold} validation case count disagrees")
        inference_seconds = summary.get("inference_total_seconds")
        if not isinstance(inference_seconds, (int, float)) or inference_seconds <= 0:
            raise ValueError(f"Fold {fold} inference duration is invalid")
        total_inference_seconds += float(inference_seconds)
        case_ids.extend(fold_ids)
        source_summaries.append(str(summary_path.resolve()))

    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Inference timing case IDs are duplicated across folds")
    expected_cases = 2 if smoke else 1251
    if len(case_ids) != expected_cases:
        raise ValueError(f"Expected {expected_cases} inference cases, got {len(case_ids)}")

    if smoke:
        fold_report = report_dir / "folds" / "fold_0"
        shutil.copy2(fold_report / "runtime.json", report_dir / "runtime.json")
        shutil.copy2(fold_report / "gpu_summary.json", report_dir / "gpu_summary.json")
    else:
        aggregate_crossval_telemetry(report_dir, expected_epochs_per_fold=100)

    inference = {
        "stage": (
            "mednext_smoke_validation_inference" if smoke else "mednext_five_fold_oof_inference"
        ),
        "backend": "mednext",
        "model_id": str(experiment["model_id"]),
        "total_seconds": total_inference_seconds,
        "number_of_cases": len(case_ids),
        "mean_seconds_per_case": total_inference_seconds / len(case_ids),
        "case_ids": case_ids,
        "timing_scope": "fresh_complete_run",
        "timing_details": (
            "official MedNeXt sliding-window validation, NIfTI/NPZ export, "
            "and native validation evaluation"
        ),
        "timing_comparable": True,
        "tta_state": "OFF",
        "source_validation_summaries": source_summaries,
    }
    write_json_atomic(report_dir / "inference_runtime.json", inference)
    return {
        "valid": True,
        "experiment_id": experiment_id,
        "folds": list(normalized_folds),
        "case_count": len(case_ids),
        "report_directory": str(report_dir.resolve()),
        "inference_runtime": str((report_dir / "inference_runtime.json").resolve()),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--fold", action="append", type=int, required=True)
    parser.add_argument("--smoke", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = sync_and_aggregate_mednext(
        args.project_root,
        args.experiment_id,
        folds=args.fold,
        smoke=args.smoke,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["EVIDENCE_FILES", "sync_and_aggregate_mednext"]
