"""Readable PNG overlays for representative T1c/FLAIR failure cases."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from glioma_seg.evaluation.regions import REGION_ORDER, validate_brats_labels
from glioma_seg.evaluation.semantic_metrics import CaseMetrics

from .slices import select_informative_slices

LABEL_COLORS: Mapping[int, tuple[float, float, float]] = {
    1: (0.95, 0.20, 0.20),  # NCR
    2: (0.20, 0.55, 1.00),  # ED
    3: (1.00, 0.80, 0.10),  # ET
}
LABEL_NAMES: Mapping[int, str] = {1: "NCR (1)", 2: "ED (2)", 3: "ET (3)"}


def normalize_mri_slice(image: ArrayLike) -> NDArray[np.float32]:
    """Robustly scale one display slice without changing stored source data."""

    array = np.asarray(image, dtype=np.float32)
    finite = np.isfinite(array)
    if not np.any(finite):
        return np.zeros(array.shape, dtype=np.float32)
    foreground = finite & (array != 0)
    values = array[foreground] if np.any(foreground) else array[finite]
    low, high = np.percentile(values, (1.0, 99.0))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return np.zeros(array.shape, dtype=np.float32)
    normalized = np.clip((array - low) / (high - low), 0.0, 1.0)
    normalized[~finite] = 0.0
    return np.asarray(normalized, dtype=np.float32)


def overlay_labels(
    image: ArrayLike, labels: ArrayLike, *, alpha: float = 0.45
) -> NDArray[np.float32]:
    """Blend NCR/ED/ET colors onto a 2D grayscale MRI slice."""

    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be in [0, 1]")
    base = normalize_mri_slice(image)
    label_array = validate_brats_labels(labels)
    if base.ndim != 2 or label_array.ndim != 2:
        raise ValueError("overlay_labels expects 2D image and label arrays")
    if base.shape != label_array.shape:
        raise ValueError(f"Image/label shapes differ: {base.shape} vs {label_array.shape}")
    rgb = np.repeat(base[..., None], 3, axis=-1)
    for label, color in LABEL_COLORS.items():
        mask = label_array == label
        if np.any(mask):
            rgb[mask] = (1.0 - alpha) * rgb[mask] + alpha * np.asarray(color)
    return np.clip(rgb, 0.0, 1.0)


def _take(volume: NDArray[Any], index: int, axis: int) -> NDArray[Any]:
    # Transpose for a stable radiological-review-friendly display orientation.
    return np.rot90(np.take(volume, index, axis=axis))


def _metric_value(metrics: CaseMetrics | Mapping[str, Any], metric: str, region: str) -> float:
    if isinstance(metrics, CaseMetrics):
        return float(getattr(metrics.regions[region], metric))
    key = f"dice_{region.lower()}" if metric == "dice" else f"hd95_{region.lower()}_mm"
    try:
        return float(metrics[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def _format_metrics(metrics: CaseMetrics | Mapping[str, Any] | None) -> str:
    if metrics is None:
        return "Metrics not recorded"
    pieces: list[str] = []
    for metric, label in (("dice", "Dice"), ("hd95_mm", "HD95 mm")):
        values = []
        for region in REGION_ORDER:
            value = _metric_value(metrics, metric, region)
            values.append(f"{region}={value:.3f}" if np.isfinite(value) else f"{region}=undefined")
        pieces.append(f"{label}: " + ", ".join(values))
    return " | ".join(pieces)


def create_failure_figure(
    *,
    case_id: str,
    t1c: ArrayLike,
    flair: ArrayLike,
    ground_truth: ArrayLike,
    prediction: ArrayLike,
    output_path: str | Path,
    metrics: CaseMetrics | Mapping[str, Any] | None = None,
    axis: int = 2,
    n_slices: int = 3,
    alpha: float = 0.45,
    dpi: int = 150,
) -> Path:
    """Create a PNG with side-by-side GT/prediction overlays for T1c and FLAIR."""

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    gt = validate_brats_labels(ground_truth, name="ground truth")
    pred = validate_brats_labels(prediction, name="prediction")
    t1c_array = np.asarray(t1c)
    flair_array = np.asarray(flair)
    if gt.ndim != 3:
        raise ValueError(f"Expected 3D volumes, got GT shape {gt.shape}")
    if not (gt.shape == pred.shape == t1c_array.shape == flair_array.shape):
        raise ValueError(
            f"Volume shapes differ: T1c={t1c_array.shape}, FLAIR={flair_array.shape}, "
            f"GT={gt.shape}, prediction={pred.shape}"
        )
    if not case_id.strip():
        raise ValueError("case_id must be non-empty")

    normalized_axis = axis % gt.ndim
    slices = select_informative_slices(gt, pred, axis=normalized_axis, n_slices=n_slices)
    modalities = (("T1c", t1c_array), ("FLAIR", flair_array))
    rows = len(slices) * len(modalities)
    figure, axes = plt.subplots(rows, 2, figsize=(9.0, max(3.2, rows * 3.0)), squeeze=False)
    for slice_offset, slice_index in enumerate(slices):
        gt_slice = _take(gt, slice_index, normalized_axis)
        pred_slice = _take(pred, slice_index, normalized_axis)
        for modality_offset, (modality_name, volume) in enumerate(modalities):
            row = slice_offset * len(modalities) + modality_offset
            image_slice = _take(volume, slice_index, normalized_axis)
            for column, (label_name, labels) in enumerate(
                (("Ground truth", gt_slice), ("Prediction", pred_slice))
            ):
                axis_object = axes[row, column]
                axis_object.imshow(
                    overlay_labels(image_slice, labels, alpha=alpha), interpolation="nearest"
                )
                axis_object.set_title(
                    f"{modality_name} · {label_name} · axis {normalized_axis}, slice {slice_index}",
                    fontsize=9,
                )
                axis_object.axis("off")

    legend = [
        Patch(facecolor=LABEL_COLORS[label], edgecolor="none", label=LABEL_NAMES[label])
        for label in (1, 2, 3)
    ]
    figure.legend(handles=legend, loc="lower center", ncol=3, frameon=False)
    figure.suptitle(f"{case_id}\n{_format_metrics(metrics)}", fontsize=11)
    figure.tight_layout(rect=(0.0, 0.035, 1.0, 0.96))
    destination = Path(output_path)
    if destination.suffix.lower() != ".png":
        raise ValueError(f"Failure figure output must be a PNG, got {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return destination


def create_failure_figure_from_nifti(
    *,
    case_id: str,
    t1c_path: str | Path,
    flair_path: str | Path,
    ground_truth_path: str | Path,
    prediction_path: str | Path,
    output_path: str | Path,
    metrics: CaseMetrics | Mapping[str, Any] | None = None,
    axis: int = 2,
    n_slices: int = 3,
) -> Path:
    """Load aligned NIfTIs and create a T1c/FLAIR failure PNG."""

    try:
        import nibabel as nib
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("nibabel is required for NIfTI visualization") from exc

    paths = [Path(t1c_path), Path(flair_path), Path(ground_truth_path), Path(prediction_path)]
    images: list[Any] = [nib.load(str(path)) for path in paths]
    arrays = [np.asanyarray(image.dataobj) for image in images]
    reference_shape = arrays[0].shape
    reference_affine = np.asarray(images[0].affine)
    for path, image, array in zip(paths[1:], images[1:], arrays[1:], strict=False):
        if array.shape != reference_shape:
            raise ValueError(f"NIfTI shape mismatch at {path}: {array.shape} vs {reference_shape}")
        if not np.allclose(image.affine, reference_affine, rtol=0.0, atol=1e-4):
            raise ValueError(f"NIfTI affine mismatch at {path}")
    return create_failure_figure(
        case_id=case_id,
        t1c=arrays[0],
        flair=arrays[1],
        ground_truth=arrays[2],
        prediction=arrays[3],
        output_path=output_path,
        metrics=metrics,
        axis=axis,
        n_slices=n_slices,
    )


def _case_nifti(directory: Path, case_id: str) -> Path:
    candidates = (directory / f"{case_id}.nii.gz", directory / f"{case_id}.nii")
    existing = [path for path in candidates if path.is_file()]
    if len(existing) != 1:
        raise FileNotFoundError(
            f"Expected exactly one NIfTI for {case_id} in {directory}; found {existing}"
        )
    return existing[0]


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"CSV artifact does not exist: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def generate_failure_figures(
    *,
    raw_training_dir: str | Path,
    ground_truth_dir: str | Path,
    prediction_dir: str | Path,
    failure_cases_csv: str | Path,
    metrics_per_case_csv: str | Path,
    output_dir: str | Path,
    max_cases: int = 15,
    axis: int = 2,
    n_slices: int = 3,
) -> Path:
    """Generate one deduplicated T1c/FLAIR PNG per selected failure case."""

    if max_cases < 1:
        raise ValueError("max_cases must be positive")
    failure_rows = _read_csv(Path(failure_cases_csv))
    metric_rows = _read_csv(Path(metrics_per_case_csv))
    metrics_by_case = {str(row.get("case_id", "")): row for row in metric_rows}
    if len(metrics_by_case) != len(metric_rows) or "" in metrics_by_case:
        raise ValueError("metrics_per_case.csv has duplicate or empty case IDs")
    case_ids: list[str] = []
    for row in failure_rows:
        case_id = str(row.get("case_id", "")).strip()
        if not case_id:
            raise ValueError("failure_cases.csv contains an empty case_id")
        if case_id not in case_ids:
            case_ids.append(case_id)
        if len(case_ids) >= max_cases:
            break
    if not case_ids:
        raise ValueError("failure_cases.csv contains no cases to visualize")

    raw_root = Path(raw_training_dir)
    gt_root = Path(ground_truth_dir)
    pred_root = Path(prediction_dir)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, str]] = []
    for case_id in case_ids:
        if case_id not in metrics_by_case:
            raise KeyError(f"No per-case metrics found for selected case {case_id}")
        case_dir = raw_root / case_id
        t1c_path = case_dir / f"{case_id}-t1c.nii.gz"
        flair_path = case_dir / f"{case_id}-t2f.nii.gz"
        missing = [str(path) for path in (t1c_path, flair_path) if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                f"Missing raw visualization modalities for {case_id}: {missing}"
            )
        figure_path = destination / f"{case_id}_t1c_flair_gt_pred.png"
        create_failure_figure_from_nifti(
            case_id=case_id,
            t1c_path=t1c_path,
            flair_path=flair_path,
            ground_truth_path=_case_nifti(gt_root, case_id),
            prediction_path=_case_nifti(pred_root, case_id),
            output_path=figure_path,
            metrics=metrics_by_case[case_id],
            axis=axis,
            n_slices=n_slices,
        )
        manifest_rows.append(
            {
                "case_id": case_id,
                "figure_path": str(figure_path.resolve()),
                "modalities": "T1c;T2-FLAIR",
                "overlay": "ground_truth;prediction",
            }
        )
    manifest = destination / "figures_manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["case_id", "figure_path", "modalities", "overlay"]
        )
        writer.writeheader()
        writer.writerows(manifest_rows)
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate T1c/FLAIR failure overlay PNGs.")
    parser.add_argument("--raw-training-dir", type=Path, required=True)
    parser.add_argument("--ground-truth-dir", type=Path, required=True)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--failure-cases-csv", type=Path, required=True)
    parser.add_argument("--metrics-per-case-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-cases", type=int, default=15)
    parser.add_argument("--axis", type=int, default=2)
    parser.add_argument("--n-slices", type=int, default=3)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    manifest = generate_failure_figures(
        raw_training_dir=args.raw_training_dir,
        ground_truth_dir=args.ground_truth_dir,
        prediction_dir=args.prediction_dir,
        failure_cases_csv=args.failure_cases_csv,
        metrics_per_case_csv=args.metrics_per_case_csv,
        output_dir=args.output_dir,
        max_cases=args.max_cases,
        axis=args.axis,
        n_slices=args.n_slices,
    )
    print(json.dumps({"figures_manifest": str(manifest.resolve())}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
