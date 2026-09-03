"""Evaluate a real-data smoke fold without pretending it is full cross-validation."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from glioma_seg.monitoring.timing import write_json_atomic

from .evaluate import evaluate_directories


def prepare_and_evaluate_smoke(
    *,
    fold_manifest_json: str | Path,
    full_ground_truth_dir: str | Path,
    prediction_dir: str | Path,
    output_dir: str | Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    manifest_path = Path(fold_manifest_json).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema") != "glioma_model_fold_manifest_v1"
        or manifest.get("smoke") is not True
    ):
        raise ValueError("Smoke evaluation requires a smoke fold manifest")
    case_ids = [str(value) for value in manifest.get("validation_case_ids", [])]
    if not case_ids or len(case_ids) != len(set(case_ids)):
        raise ValueError("Smoke validation IDs must be non-empty and unique")
    source_gt = Path(full_ground_truth_dir).resolve()
    predictions = Path(prediction_dir).resolve()
    destination = Path(output_dir).resolve()
    subset_gt = destination / "smoke_ground_truth"
    if subset_gt.exists():
        existing = {path.name.removesuffix(".nii.gz") for path in subset_gt.glob("*.nii.gz")}
        if existing != set(case_ids):
            raise FileExistsError(f"Conflicting smoke ground-truth subset exists: {subset_gt}")
    else:
        destination.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".smoke_gt_", dir=destination))
        try:
            for case_id in case_ids:
                source = source_gt / f"{case_id}.nii.gz"
                target = staging / source.name
                if not source.is_file():
                    raise FileNotFoundError(f"Smoke ground truth is missing: {source}")
                try:
                    os.link(source, target)
                except OSError:
                    shutil.copy2(source, target)
            os.replace(staging, subset_gt)
        finally:
            if staging.exists():
                shutil.rmtree(staging)

    owned = (
        destination / "metrics_per_case.csv",
        destination / "metrics_summary.csv",
        destination / "metrics_summary.json",
        destination / "evaluation_protocol.json",
    )
    if any(path.exists() for path in owned) and not overwrite:
        raise FileExistsError("Smoke metric artifacts already exist; use verified resume/overwrite")
    artifacts = evaluate_directories(
        subset_gt,
        predictions,
        destination,
        expected_case_ids=case_ids,
        strict_predictions=True,
        split_source=manifest_path,
        fold=int(manifest["fold"]),
        prediction_provenance=(
            f"{manifest['backend']} real-data smoke validation output; "
            f"model={manifest['model_id']}"
        ),
        prediction_tta_state="OFF",
    )
    protocol_path = Path(artifacts.protocol_json)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol.update(
        {
            "evaluation_scope": "real_data_smoke_test_not_full_cross_validation",
            "backend": manifest["backend"],
            "model_id": manifest["model_id"],
            "folds": [int(manifest["fold"])],
            "case_ids": case_ids,
            "each_case_validated_once": True,
        }
    )
    write_json_atomic(protocol_path, protocol)
    integrity = {
        "valid": True,
        "scope": "real_data_smoke_test_not_full_cross_validation",
        "backend": manifest["backend"],
        "model_id": manifest["model_id"],
        "fold": int(manifest["fold"]),
        "total_cases": len(case_ids),
        "case_ids": case_ids,
        "each_case_validated_once": True,
        "ground_truth_dir": str(subset_gt),
        "prediction_dir": str(predictions),
        "fold_manifest": str(manifest_path),
    }
    write_json_atomic(destination / "crossval_integrity.json", integrity)
    return {
        "valid": True,
        "case_count": len(artifacts.cases),
        "ground_truth_dir": str(subset_gt),
        "metrics_per_case": str(artifacts.per_case_csv),
        "metrics_summary": str(artifacts.summary_csv),
        "evaluation_protocol": str(artifacts.protocol_json),
        "integrity": str((destination / "crossval_integrity.json").resolve()),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold-manifest", type=Path, required=True)
    parser.add_argument("--ground-truth-dir", type=Path, required=True)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = prepare_and_evaluate_smoke(
        fold_manifest_json=args.fold_manifest,
        full_ground_truth_dir=args.ground_truth_dir,
        prediction_dir=args.prediction_dir,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
