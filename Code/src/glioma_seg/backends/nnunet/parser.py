"""Parsers for nnU-Net logs, plans, and benchmark metadata."""

from __future__ import annotations

import json
import math
import re
import statistics
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TypeGuard

_EPOCH_RE = re.compile(r"(?:^|\s)epoch\s*[: ]\s*(\d+)", re.IGNORECASE)
_TRAIN_LOSS_RE = re.compile(r"train(?:ing)?[_ ]loss\s*[: ]\s*(-?\d+(?:\.\d+)?(?:e[+-]?\d+)?)", re.I)
_VAL_LOSS_RE = re.compile(r"val(?:idation)?[_ ]loss\s*[: ]\s*(-?\d+(?:\.\d+)?(?:e[+-]?\d+)?)", re.I)
_PSEUDO_DICE_RE = re.compile(r"(?:pseudo[ _-]?dice|global[ _-]?dice)\s*[: ]\s*(.+)$", re.I)
_EPOCH_TIME_RE = re.compile(r"epoch\s*time\s*[: ]\s*(\d+(?:\.\d+)?)\s*s?", re.I)


@dataclass
class TrainingProgress:
    current_epoch: int | None = None
    latest_train_loss: float | None = None
    latest_validation_loss: float | None = None
    latest_pseudo_dice: str | None = None
    epoch_durations_seconds: list[float] | None = None

    def __post_init__(self) -> None:
        if self.epoch_durations_seconds is None:
            self.epoch_durations_seconds = []

    def update(self, line: str) -> None:
        if match := _EPOCH_RE.search(line):
            self.current_epoch = int(match.group(1))
        if match := _TRAIN_LOSS_RE.search(line):
            self.latest_train_loss = float(match.group(1))
        if match := _VAL_LOSS_RE.search(line):
            self.latest_validation_loss = float(match.group(1))
        if match := _PSEUDO_DICE_RE.search(line):
            self.latest_pseudo_dice = match.group(1).strip()
        if match := _EPOCH_TIME_RE.search(line):
            assert self.epoch_durations_seconds is not None
            self.epoch_durations_seconds.append(float(match.group(1)))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def summarize_plans(plans_path: Path, configuration: str = "3d_fullres") -> dict[str, Any]:
    """Extract actual planned values without assuming a network architecture."""

    plans = load_json(plans_path)
    configurations = plans.get("configurations", {})
    if configuration not in configurations:
        available = ", ".join(sorted(configurations)) or "none"
        raise KeyError(f"Configuration {configuration!r} missing; available: {available}")
    config = configurations[configuration]
    architecture = config.get("architecture") or {}
    arch_kwargs = architecture.get("arch_kwargs") or config.get("network_arch_init_kwargs") or {}
    features_per_stage = arch_kwargs.get("features_per_stage")
    number_of_stages = arch_kwargs.get("n_stages")
    if number_of_stages is None and isinstance(features_per_stage, list):
        number_of_stages = len(features_per_stage)
    if number_of_stages is None:
        number_of_stages = len(arch_kwargs.get("n_conv_per_stage", [])) or None
    return {
        "plans_file": str(plans_path),
        "plans_name": plans.get("plans_name"),
        "configuration": configuration,
        "target_spacing": config.get("spacing"),
        "median_image_size_in_voxels": config.get("median_image_size_in_voxels"),
        "patch_size": config.get("patch_size"),
        "batch_size": config.get("batch_size"),
        "architecture": architecture.get("network_class_name")
        or config.get("network_arch_class_name"),
        "architecture_class": architecture.get("network_class_name")
        or config.get("network_arch_class_name"),
        "number_of_stages": number_of_stages,
        "features_per_stage": features_per_stage,
        "normalization_schemes": config.get("normalization_schemes"),
        "use_mask_for_norm": config.get("use_mask_for_norm"),
        "resampling_fn_data": config.get("resampling_fn_data"),
        "resampling_fn_seg": config.get("resampling_fn_seg"),
        "resampling_fn_probabilities": config.get("resampling_fn_probabilities"),
        "preprocessor_name": config.get("preprocessor_name"),
        "data_identifier": config.get("data_identifier"),
        "raw_configuration": config,
    }


def summarize_fingerprint(fingerprint_path: Path) -> dict[str, Any]:
    fingerprint = load_json(fingerprint_path)
    spacings = fingerprint.get("spacings")
    shapes = fingerprint.get("shapes_after_crop")

    def component_median(rows: Any) -> list[float] | None:
        if not isinstance(rows, list) or not rows or not all(isinstance(row, list) for row in rows):
            return None
        width = len(rows[0])
        if width == 0 or any(len(row) != width for row in rows):
            return None
        try:
            return [
                float(statistics.median(float(row[index]) for row in rows))
                for index in range(width)
            ]
        except (TypeError, ValueError):
            return None

    return {
        "fingerprint_file": str(fingerprint_path),
        "spacings": spacings,
        "original_median_spacing": component_median(spacings),
        "shapes_after_crop": shapes,
        "original_median_shape_after_crop": component_median(shapes),
        "median_relative_size_after_cropping": fingerprint.get(
            "median_relative_size_after_cropping"
        ),
        "foreground_intensity_properties_per_channel": fingerprint.get(
            "foreground_intensity_properties_per_channel"
        ),
    }


def find_latest_benchmark_result(results_root: Path) -> Path | None:
    candidates = list(results_root.rglob("benchmark_result.json")) if results_root.exists() else []
    return max(candidates, key=lambda path: path.stat().st_mtime_ns) if candidates else None


def summarize_benchmark(
    benchmark_path: Path,
    *,
    measured_wall_seconds: float | None = None,
    observed_mean_epoch_seconds: float | None = None,
    expected_record: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = load_json(benchmark_path)
    records = [(key, record) for key, record in payload.items() if isinstance(record, Mapping)]
    if not records:
        raise ValueError(f"No benchmark record found in {benchmark_path}")
    if expected_record:
        matching_records = [
            (key, record)
            for key, record in records
            if all(str(record.get(field)) == str(value) for field, value in expected_record.items())
        ]
        if len(matching_records) != 1:
            raise ValueError(
                "Could not identify exactly one benchmark record for the current "
                f"environment in {benchmark_path}; matches={len(matching_records)}"
            )
        record_key, record = matching_records[0]
    elif len(records) == 1:
        record_key, record = records[0]
    else:
        raise ValueError(
            f"Multiple benchmark records exist in {benchmark_path}; expected environment metadata"
        )
    fastest = record.get("fastest_epoch")
    fastest_epoch = float(fastest) if finite_number(fastest) and float(fastest) > 0 else None
    wall_mean = measured_wall_seconds / 5.0 if measured_wall_seconds else None
    parsed_mean = (
        float(observed_mean_epoch_seconds)
        if finite_number(observed_mean_epoch_seconds) and float(observed_mean_epoch_seconds) > 0
        else None
    )
    # Prefer actual parsed epoch durations. The official fastest epoch is the
    # fallback; wall/5 includes initialization and is retained only for audit.
    estimate_basis = parsed_mean or fastest_epoch
    estimates = {
        str(epochs): estimate_basis * epochs if estimate_basis is not None else None
        for epochs in (20, 50, 100, 1000)
    }
    recommendation = None
    if estimates["50"] is not None:
        recommendation = (
            "nnUNetTrainer_50epochs" if estimates["50"] <= 3 * 60 * 60 else "nnUNetTrainer_20epochs"
        )
    return {
        "benchmark_result_file": str(benchmark_path),
        "official_record_key": record_key,
        "official_record": dict(record),
        "fastest_epoch_seconds": fastest_epoch,
        "observed_mean_epoch_seconds": parsed_mean,
        "observed_wall_seconds": measured_wall_seconds,
        "wall_seconds_divided_by_five": wall_mean,
        "runtime_estimate_basis_seconds_per_epoch": estimate_basis,
        "linear_runtime_estimates_seconds": estimates,
        "estimate_note": "Linear rough estimate; not a guaranteed training duration.",
        "recommended_preliminary_trainer": recommendation,
    }


def finite_number(value: object) -> TypeGuard[int | float]:
    return isinstance(value, (int, float)) and math.isfinite(float(value))
