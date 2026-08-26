"""NIfTI directory evaluation and reproducible metric artifact writing."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .regions import REGION_LABELS, REGION_ORDER
from .semantic_metrics import CaseMetrics, evaluate_case, per_case_fieldnames, summarize_cases

PREDICTION_TTA_STATES = ("OFF", "DEFAULT_MIRRORING", "UNKNOWN")


@dataclass(frozen=True)
class EvaluationArtifacts:
    """Paths and in-memory results produced by directory evaluation."""

    cases: tuple[CaseMetrics, ...]
    summary: tuple[dict[str, Any], ...]
    per_case_csv: Path
    summary_csv: Path
    summary_json: Path
    protocol_json: Path


def case_id_from_nifti(path: str | Path) -> str:
    """Return the case identifier from a .nii or .nii.gz path."""

    name = Path(path).name
    if name.endswith(".nii.gz"):
        return name[: -len(".nii.gz")]
    if name.endswith(".nii"):
        return name[: -len(".nii")]
    raise ValueError(f"Expected a NIfTI filename, got {name!r}")


def _nifti_files(directory: Path) -> dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"NIfTI directory does not exist: {directory}")
    files: dict[str, Path] = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file() or not (path.name.endswith(".nii.gz") or path.suffix == ".nii"):
            continue
        case_id = case_id_from_nifti(path)
        if case_id in files:
            raise ValueError(f"Duplicate NIfTI case ID {case_id!r} in {directory}")
        files[case_id] = path
    return files


def _load_nifti(path: Path) -> tuple[np.ndarray, np.ndarray, tuple[float, ...]]:
    try:
        import nibabel as nib
    except ImportError as exc:  # pragma: no cover - dependency check is environment-specific
        raise RuntimeError("nibabel is required to evaluate NIfTI files") from exc

    image: Any = nib.load(str(path))
    data = np.asanyarray(image.dataobj)
    if data.ndim != 3:
        raise ValueError(f"Expected a 3D label map at {path}, got shape {data.shape}")
    spacing = tuple(float(value) for value in image.header.get_zooms()[: data.ndim])
    return data, np.asarray(image.affine, dtype=float), spacing


def evaluate_nifti_pair(
    ground_truth_path: str | Path,
    prediction_path: str | Path,
    *,
    case_id: str | None = None,
    affine_atol: float = 1e-4,
) -> CaseMetrics:
    """Evaluate one aligned prediction/ground-truth NIfTI pair."""

    gt_path = Path(ground_truth_path)
    pred_path = Path(prediction_path)
    resolved_case_id = case_id or case_id_from_nifti(gt_path)
    gt, gt_affine, spacing = _load_nifti(gt_path)
    pred, pred_affine, pred_spacing = _load_nifti(pred_path)
    if gt.shape != pred.shape:
        raise ValueError(
            f"Geometry mismatch for {resolved_case_id}: GT shape {gt.shape}, "
            f"prediction shape {pred.shape}"
        )
    if not np.allclose(gt_affine, pred_affine, rtol=0.0, atol=affine_atol):
        raise ValueError(
            f"Affine mismatch for {resolved_case_id}; refusing to compare unaligned arrays"
        )
    if not np.allclose(spacing, pred_spacing, rtol=0.0, atol=1e-6):
        raise ValueError(
            f"Spacing mismatch for {resolved_case_id}: GT {spacing}, prediction {pred_spacing}"
        )
    return evaluate_case(gt, pred, spacing, case_id=resolved_case_id)


def _write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, float | np.floating) and not np.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def evaluate_directories(
    ground_truth_dir: str | Path,
    prediction_dir: str | Path,
    output_dir: str | Path,
    *,
    expected_case_ids: Sequence[str] | None = None,
    strict_predictions: bool = True,
    split_source: str | Path | None = None,
    fold: int | None = None,
    prediction_provenance: str | None = None,
    prediction_tta_state: str = "UNKNOWN",
) -> EvaluationArtifacts:
    """Evaluate matching local-fold NIfTIs and write per-case/summary artifacts.

    This function requires real ground-truth files.  It cannot evaluate the
    public BraTS validation set, for which ground truth is not available.
    Missing expected predictions are always an error and are never skipped.
    """

    if prediction_tta_state not in PREDICTION_TTA_STATES:
        raise ValueError(
            f"prediction_tta_state must be one of {PREDICTION_TTA_STATES}, "
            f"got {prediction_tta_state!r}"
        )
    gt_directory = Path(ground_truth_dir).resolve()
    prediction_directory = Path(prediction_dir).resolve()
    gt_files = _nifti_files(gt_directory)
    pred_files = _nifti_files(prediction_directory)
    if expected_case_ids is None:
        selected_ids = sorted(gt_files)
    else:
        selected_ids = sorted(dict.fromkeys(str(value) for value in expected_case_ids))

    missing_gt = [case_id for case_id in selected_ids if case_id not in gt_files]
    missing_predictions = [case_id for case_id in selected_ids if case_id not in pred_files]
    if missing_gt:
        raise FileNotFoundError(
            f"Missing ground truth for {len(missing_gt)} cases: {missing_gt[:10]}"
        )
    if missing_predictions:
        raise FileNotFoundError(
            f"Missing predictions for {len(missing_predictions)} cases: {missing_predictions[:10]}"
        )
    if not selected_ids:
        raise ValueError("No cases selected for evaluation")

    unexpected_predictions = sorted(set(pred_files) - set(selected_ids))
    if strict_predictions and unexpected_predictions:
        raise ValueError(
            "Prediction directory contains cases outside the evaluation split: "
            f"{unexpected_predictions[:10]}"
        )

    cases = tuple(
        evaluate_nifti_pair(gt_files[case_id], pred_files[case_id], case_id=case_id)
        for case_id in selected_ids
    )
    summary = tuple(summarize_cases(cases))
    destination = Path(output_dir)
    per_case_csv = destination / "metrics_per_case.csv"
    summary_csv = destination / "metrics_summary.csv"
    summary_json = destination / "metrics_summary.json"
    protocol_json = destination / "evaluation_protocol.json"
    _write_csv(per_case_csv, (case.as_flat_dict() for case in cases), per_case_fieldnames())
    summary_fields = [
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
        "total_cases",
    ]
    _write_csv(summary_csv, summary, summary_fields)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(
        json.dumps(_json_safe(summary), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    protocol = {
        "metric_type": "standard semantic region-wise",
        "region_order": list(REGION_ORDER),
        "regions": {region: list(REGION_LABELS[region]) for region in REGION_ORDER},
        "ground_truth_required": True,
        "ground_truth_dir": str(gt_directory),
        "prediction_dir": str(prediction_directory),
        "prediction_provenance": prediction_provenance or "NOT RECORDED",
        "prediction_tta_state": prediction_tta_state,
        "split_source": str(Path(split_source).resolve()) if split_source is not None else None,
        "fold": fold,
        "case_count": len(selected_ids),
        "case_ids": selected_ids,
        "strict_prediction_inventory": strict_predictions,
        "dice_empty_mask_policy": (
            "both-empty is undefined (NaN); one-sided empty is 0; finite values are summarized"
        ),
        "hd95_empty_mask_policy": (
            "undefined (NaN) whenever either mask is empty; finite values are summarized"
        ),
        "hd95_distance_space": "physical millimetres from NIfTI voxel spacing",
    }
    protocol_json.write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return EvaluationArtifacts(
        cases,
        summary,
        per_case_csv,
        summary_csv,
        summary_json,
        protocol_json,
    )


def validation_ids_from_splits(path: str | Path, fold: int) -> list[str]:
    """Read nnU-Net splits_final.json and return one fold's validation IDs."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"nnU-Net split file does not exist: {source}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected a list of folds in {source}")
    if fold < 0 or fold >= len(payload):
        raise IndexError(f"Fold {fold} is outside split range 0..{len(payload) - 1}")
    split = payload[fold]
    if not isinstance(split, dict) or not isinstance(split.get("val"), list):
        raise ValueError(f"Fold {fold} in {source} does not contain a validation list")
    validation = [str(case_id) for case_id in split["val"]]
    training = {str(case_id) for case_id in split.get("train", [])}
    overlap = sorted(training.intersection(validation))
    if overlap:
        raise ValueError(f"Fold {fold} has train/validation leakage: {overlap[:10]}")
    if len(validation) != len(set(validation)):
        raise ValueError(f"Fold {fold} validation list contains duplicate case IDs")
    if not validation:
        raise ValueError(f"Fold {fold} validation list is empty")
    return validation


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute standard semantic BraTS 2023 ET/TC/WT Dice and physical-spacing HD95."
    )
    parser.add_argument("--ground-truth-dir", type=Path, required=True)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--splits-json", type=Path)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument(
        "--prediction-provenance",
        default=None,
        help="Artifact-backed description of how the evaluated predictions were generated.",
    )
    parser.add_argument(
        "--prediction-tta-state",
        choices=PREDICTION_TTA_STATES,
        default="UNKNOWN",
        help="TTA state of the evaluated prediction masks, distinct from timing inference.",
    )
    parser.add_argument(
        "--allow-extra-predictions",
        action="store_true",
        help=(
            "Allow predictions not in the selected fold "
            "(selected cases are still evaluated explicitly)."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    case_ids = validation_ids_from_splits(args.splits_json, args.fold) if args.splits_json else None
    artifacts = evaluate_directories(
        args.ground_truth_dir,
        args.prediction_dir,
        args.output_dir,
        expected_case_ids=case_ids,
        strict_predictions=not args.allow_extra_predictions,
        split_source=args.splits_json,
        fold=args.fold if args.splits_json else None,
        prediction_provenance=args.prediction_provenance,
        prediction_tta_state=args.prediction_tta_state,
    )
    print(
        json.dumps(
            {
                "cases_evaluated": len(artifacts.cases),
                "metrics_per_case": str(artifacts.per_case_csv.resolve()),
                "metrics_summary": str(artifacts.summary_csv.resolve()),
                "metrics_summary_json": str(artifacts.summary_json.resolve()),
                "evaluation_protocol": str(artifacts.protocol_json.resolve()),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by integration scripts
    raise SystemExit(main())
