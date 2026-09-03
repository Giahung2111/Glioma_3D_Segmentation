"""Aggregate fold-scoped runtime and GPU evidence for a five-fold CV report."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from glioma_seg.monitoring.timing import write_json_atomic

FULL_CV_FOLDS = (0, 1, 2, 3, 4)


def _load_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required fold artifact is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _finite_number(
    value: object,
    *,
    field: str,
    path: Path,
    minimum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric in {path}: {value!r}")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise ValueError(f"{field} is invalid in {path}: {value!r}")
    return result


def _positive_integer(value: object, *, field: str, path: Path) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field} must be a positive integer in {path}: {value!r}")
    return value


def _weighted_mean(values: Sequence[tuple[float, int]]) -> float:
    denominator = sum(weight for _, weight in values)
    if denominator < 1:
        raise ValueError("Cannot compute a weighted mean without positive weights")
    return sum(value * weight for value, weight in values) / denominator


def aggregate_crossval_telemetry(
    experiment_dir: str | Path,
    *,
    folds: Sequence[int] = FULL_CV_FOLDS,
    expected_epochs_per_fold: int = 100,
) -> tuple[Path, Path, Path]:
    """Validate and combine all fold telemetry without hiding per-fold evidence."""

    normalized_folds = tuple(int(fold) for fold in folds)
    if normalized_folds != FULL_CV_FOLDS:
        raise ValueError(f"Full CV requires folds {FULL_CV_FOLDS}, got {normalized_folds}")
    if expected_epochs_per_fold < 1:
        raise ValueError("expected_epochs_per_fold must be positive")

    destination = Path(experiment_dir).resolve()
    if not destination.is_dir():
        raise FileNotFoundError(f"Experiment directory does not exist: {destination}")

    runtime_records: list[dict[str, Any]] = []
    gpu_records: list[dict[str, Any]] = []
    table_rows: list[dict[str, Any]] = []
    epoch_means: list[tuple[float, int]] = []
    gpu_utilizations: list[tuple[float, int]] = []
    gpu_powers: list[tuple[float, int]] = []

    for fold in normalized_folds:
        fold_dir = destination / "folds" / f"fold_{fold}"
        runtime_path = fold_dir / "runtime.json"
        gpu_path = fold_dir / "gpu_summary.json"
        runtime = _load_object(runtime_path)
        gpu = _load_object(gpu_path)

        epochs = _positive_integer(
            runtime.get("number_of_epochs"), field="number_of_epochs", path=runtime_path
        )
        if epochs != expected_epochs_per_fold:
            raise ValueError(
                f"Fold {fold} records {epochs} epochs, expected {expected_epochs_per_fold}"
            )
        total_seconds = _finite_number(
            runtime.get("total_seconds"), field="total_seconds", path=runtime_path, minimum=0.0
        )
        epoch_mean = _finite_number(
            runtime.get("average_seconds_per_epoch"),
            field="average_seconds_per_epoch",
            path=runtime_path,
            minimum=0.0,
        )
        samples = _positive_integer(gpu.get("samples"), field="samples", path=gpu_path)
        peak_memory = _finite_number(
            gpu.get("peak_memory_used_mb"),
            field="peak_memory_used_mb",
            path=gpu_path,
            minimum=0.0,
        )
        mean_utilization = _finite_number(
            gpu.get("mean_gpu_utilization_percent"),
            field="mean_gpu_utilization_percent",
            path=gpu_path,
            minimum=0.0,
        )
        if mean_utilization > 100:
            raise ValueError(f"GPU utilization exceeds 100% in {gpu_path}")
        peak_temperature = _finite_number(
            gpu.get("peak_temperature_c"),
            field="peak_temperature_c",
            path=gpu_path,
            minimum=0.0,
        )
        mean_power_raw = gpu.get("mean_power_w")
        mean_power = (
            None
            if mean_power_raw is None
            else _finite_number(
                mean_power_raw, field="mean_power_w", path=gpu_path, minimum=0.0
            )
        )

        runtime_records.append({"fold": fold, **runtime, "source": str(runtime_path)})
        gpu_records.append({"fold": fold, **gpu, "source": str(gpu_path)})
        epoch_means.append((epoch_mean, epochs))
        gpu_utilizations.append((mean_utilization, samples))
        if mean_power is not None:
            gpu_powers.append((mean_power, samples))
        table_rows.append(
            {
                "fold": fold,
                "epochs": epochs,
                "training_seconds": total_seconds,
                "training_hours": total_seconds / 3600.0,
                "average_seconds_per_epoch": epoch_mean,
                "peak_memory_used_mb": peak_memory,
                "mean_gpu_utilization_percent": mean_utilization,
                "peak_temperature_c": peak_temperature,
                "mean_power_w": mean_power,
                "runtime_json": str(runtime_path),
                "gpu_summary_json": str(gpu_path),
            }
        )

    total_seconds = sum(float(record["total_seconds"]) for record in runtime_records)
    fold_epoch_medians = [
        float(record["epoch_seconds_median"])
        for record in runtime_records
        if isinstance(record.get("epoch_seconds_median"), (int, float))
        and not isinstance(record.get("epoch_seconds_median"), bool)
        and math.isfinite(float(record["epoch_seconds_median"]))
    ]
    epoch_mins = [
        float(record["epoch_seconds_min"])
        for record in runtime_records
        if isinstance(record.get("epoch_seconds_min"), (int, float))
        and math.isfinite(float(record["epoch_seconds_min"]))
    ]
    epoch_maxes = [
        float(record["epoch_seconds_max"])
        for record in runtime_records
        if isinstance(record.get("epoch_seconds_max"), (int, float))
        and math.isfinite(float(record["epoch_seconds_max"]))
    ]
    runtime_summary: dict[str, Any] = {
        "stage": "five_fold_cross_validation_training",
        "folds": list(normalized_folds),
        "epochs_per_fold": expected_epochs_per_fold,
        "number_of_epochs": expected_epochs_per_fold * len(normalized_folds),
        "total_seconds": total_seconds,
        "total_hours": total_seconds / 3600.0,
        "average_seconds_per_epoch": _weighted_mean(epoch_means),
        "epoch_seconds_min": min(epoch_mins) if epoch_mins else None,
        "epoch_seconds_median": (
            statistics.median(fold_epoch_medians) if fold_epoch_medians else None
        ),
        "epoch_seconds_median_scope": "median_of_fold_medians",
        "epoch_seconds_max": max(epoch_maxes) if epoch_maxes else None,
        "fold_records": runtime_records,
    }
    gpu_summary: dict[str, Any] = {
        "scope": "five_fold_cross_validation_training",
        "folds": list(normalized_folds),
        "samples": sum(int(record["samples"]) for record in gpu_records),
        "peak_memory_used_mb": max(float(record["peak_memory_used_mb"]) for record in gpu_records),
        "dedicated_memory_total_mb": max(
            float(record["dedicated_memory_total_mb"])
            for record in gpu_records
            if isinstance(record.get("dedicated_memory_total_mb"), (int, float))
        ),
        "mean_gpu_utilization_percent": _weighted_mean(gpu_utilizations),
        "peak_temperature_c": max(float(record["peak_temperature_c"]) for record in gpu_records),
        "mean_power_w": _weighted_mean(gpu_powers) if gpu_powers else None,
        "backend": "aggregate_of_fold_summaries",
        "errors": [
            f"fold_{record['fold']}: {error}"
            for record in gpu_records
            for error in record.get("errors", [])
        ],
        "fold_records": gpu_records,
    }

    runtime_output = destination / "runtime.json"
    gpu_output = destination / "gpu_summary.json"
    csv_output = destination / "fold_training_summary.csv"
    write_json_atomic(runtime_output, runtime_summary)
    write_json_atomic(gpu_output, gpu_summary)
    with csv_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(table_rows[0]))
        writer.writeheader()
        writer.writerows(table_rows)

    manifest_path = destination / "experiment.json"
    manifest = _load_object(manifest_path)
    manifest.update(
        {
            "folds": list(normalized_folds),
            "epochs": expected_epochs_per_fold,
            "epochs_per_fold": expected_epochs_per_fold,
            "total_training_epochs": expected_epochs_per_fold * len(normalized_folds),
            "training_seconds": total_seconds,
            "average_epoch_seconds": runtime_summary["average_seconds_per_epoch"],
            "peak_vram_mb": gpu_summary["peak_memory_used_mb"],
            "mean_gpu_utilization": gpu_summary["mean_gpu_utilization_percent"],
            "aggregate_runtime": str(runtime_output),
            "aggregate_gpu_summary": str(gpu_output),
            "fold_training_summary": str(csv_output),
        }
    )
    write_json_atomic(manifest_path, manifest)
    return runtime_output, gpu_output, csv_output


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate verified per-fold runtime and GPU summaries for full CV."
    )
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--expected-epochs-per-fold", type=int, default=100)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    runtime, gpu, table = aggregate_crossval_telemetry(
        args.experiment_dir,
        expected_epochs_per_fold=args.expected_epochs_per_fold,
    )
    print(
        json.dumps(
            {
                "runtime": str(runtime),
                "gpu_summary": str(gpu),
                "fold_training_summary": str(table),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
