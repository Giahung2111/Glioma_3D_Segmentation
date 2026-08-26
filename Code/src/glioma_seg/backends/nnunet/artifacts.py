"""Validation/finalization of official nnU-Net v2.8.1 preprocessing artifacts.

The official trainer creates ``splits_final.json`` lazily on first startup.
This project needs to inspect and archive Fold 0 before a long training run, so
we call nnU-Net's own deterministic split helper after preprocessing completes.
No nnU-Net algorithm or model code is copied here.
"""

from __future__ import annotations

import argparse
import json
import pickle
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from glioma_seg.backends.nnunet.parser import load_json, summarize_plans
from glioma_seg.monitoring.timing import write_json_atomic
from glioma_seg.utils.paths import ensure_within


class PreprocessingArtifactError(RuntimeError):
    """Raised when official preprocessing output is incomplete or incompatible."""


@dataclass(frozen=True, slots=True)
class ArtifactCheck:
    name: str
    ok: bool
    detail: str


@dataclass(slots=True)
class PreprocessingArtifactReport:
    dataset_name: str
    configuration: str
    plans_name: str
    checks: list[ArtifactCheck] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def valid(self) -> bool:
        return all(check.ok for check in self.checks)

    def add(self, name: str, ok: bool, detail: str) -> None:
        self.checks.append(ArtifactCheck(name=name, ok=ok, detail=detail))

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "dataset_name": self.dataset_name,
            "configuration": self.configuration,
            "plans_name": self.plans_name,
            "checks": [asdict(check) for check in self.checks],
            "details": self.details,
        }

    def require_valid(self) -> None:
        failures = [check for check in self.checks if not check.ok]
        if failures:
            summary = "; ".join(f"{check.name}: {check.detail}" for check in failures[:8])
            raise PreprocessingArtifactError(f"Preprocessing artifact validation failed: {summary}")


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        return load_json(path)
    except (OSError, ValueError, TypeError) as exc:
        raise PreprocessingArtifactError(f"Cannot read JSON object {path}: {exc}") from exc


def _raw_case_ids(raw_dataset_dir: Path, dataset_json: Mapping[str, Any]) -> tuple[str, ...]:
    channels = dataset_json.get("channel_names")
    if not isinstance(channels, Mapping) or not channels:
        raise PreprocessingArtifactError("dataset.json channel_names must be a non-empty mapping")
    channel_keys = sorted(str(key) for key in channels)
    if channel_keys != ["0", "1", "2", "3"]:
        raise PreprocessingArtifactError(f"Expected channels 0,1,2,3; got {channel_keys}")
    normalized_channels = {str(key): str(value).casefold() for key, value in channels.items()}
    expected_channels = {"0": "t1n", "1": "t1c", "2": "t2w", "3": "t2f"}
    if normalized_channels != expected_channels:
        raise PreprocessingArtifactError(
            f"BraTS channel mapping must be T1n/T1c/T2w/T2F; got {dict(channels)}"
        )
    images_tr = raw_dataset_dir / "imagesTr"
    labels_tr = raw_dataset_dir / "labelsTr"
    if not images_tr.is_dir() or not labels_tr.is_dir():
        raise PreprocessingArtifactError("imagesTr and labelsTr must exist")

    case_ids = tuple(
        sorted(path.name.removesuffix("_0000.nii.gz") for path in images_tr.glob("*_0000.nii.gz"))
    )
    expected_images = {
        f"{case_id}_{channel:04d}.nii.gz"
        for case_id in case_ids
        for channel in range(len(channel_keys))
    }
    expected_labels = {f"{case_id}.nii.gz" for case_id in case_ids}
    actual_images = {path.name for path in images_tr.iterdir() if path.is_file()}
    actual_labels = {path.name for path in labels_tr.iterdir() if path.is_file()}
    if actual_images != expected_images:
        missing = sorted(expected_images - actual_images)
        extra = sorted(actual_images - expected_images)
        raise PreprocessingArtifactError(
            f"imagesTr inventory mismatch: missing={missing[:5]}, extra={extra[:5]}"
        )
    if actual_labels != expected_labels:
        missing = sorted(expected_labels - actual_labels)
        extra = sorted(actual_labels - expected_labels)
        raise PreprocessingArtifactError(
            f"labelsTr inventory mismatch: missing={missing[:5]}, extra={extra[:5]}"
        )
    if any(path.stat().st_size <= 0 for path in (*images_tr.iterdir(), *labels_tr.iterdir())):
        raise PreprocessingArtifactError("Raw nnU-Net inventory contains an empty file")
    return case_ids


def _validate_region_contract(dataset_json: Mapping[str, Any]) -> tuple[bool, str]:
    expected_labels = {
        "background": 0,
        "whole_tumor": [1, 2, 3],
        "tumor_core": [1, 3],
        "enhancing_tumor": 3,
    }
    labels = dataset_json.get("labels")
    class_order = dataset_json.get("regions_class_order")
    if labels != expected_labels or list(labels or {}) != list(expected_labels):
        return False, f"labels/order={labels}"
    if class_order != [2, 1, 3]:
        return False, f"regions_class_order={class_order}"
    try:
        # Use the pinned upstream implementation as the compatibility oracle.
        from nnunetv2.utilities.label_handling.label_handling import (  # type: ignore
            LabelManager,
        )

        manager = LabelManager(dict(labels), regions_class_order=list(class_order))
        regions = manager.all_regions
        expected_regions = [(1, 2, 3), (1, 3), 3]
        compatible = (
            manager.has_regions
            and manager.num_segmentation_heads == 3
            and manager.all_labels == [0, 1, 2, 3]
            and regions == expected_regions
        )
        return (
            compatible,
            f"heads={manager.num_segmentation_heads}, regions={regions}, "
            f"labels={manager.all_labels}",
        )
    except Exception as exc:
        return False, f"upstream LabelManager rejected contract: {type(exc).__name__}: {exc}"


def _validate_case_artifacts(
    configuration_dir: Path, expected_case_ids: tuple[str, ...]
) -> tuple[bool, str]:
    """Validate v2.8.1 Blosc2 metadata without decompressing image volumes."""

    if not configuration_dir.is_dir():
        return False, f"Missing configuration directory: {configuration_dir}"
    expected = {
        name
        for case_id in expected_case_ids
        for name in (f"{case_id}.b2nd", f"{case_id}_seg.b2nd", f"{case_id}.pkl")
    }
    entries = tuple(configuration_dir.iterdir())
    actual = {path.name for path in entries}
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    non_files = sorted(path.name for path in entries if not path.is_file())
    empty = sorted(path.name for path in entries if path.is_file() and path.stat().st_size <= 0)
    corrupt: list[str] = []
    if not missing and not extra and not non_files and not empty:
        try:
            import blosc2  # type: ignore[import-untyped]
        except ImportError as exc:
            return False, f"Official nnU-Net Blosc2 dependency is unavailable: {exc}"
        required_properties = {
            "spacing",
            "shape_before_cropping",
            "bbox_used_for_cropping",
            "shape_after_cropping_and_before_resampling",
            "class_locations",
        }
        for case_id in expected_case_ids:
            try:
                data = blosc2.open(urlpath=str(configuration_dir / f"{case_id}.b2nd"))
                segmentation = blosc2.open(urlpath=str(configuration_dir / f"{case_id}_seg.b2nd"))
                with (configuration_dir / f"{case_id}.pkl").open("rb") as stream:
                    properties = pickle.load(stream)  # noqa: S301 - trusted nnU-Net output
                data_shape = tuple(int(value) for value in data.shape)
                segmentation_shape = tuple(int(value) for value in segmentation.shape)
                if (
                    len(data_shape) != 4
                    or data_shape[0] != 4
                    or len(segmentation_shape) != 4
                    or segmentation_shape[0] != 1
                    or data_shape[1:] != segmentation_shape[1:]
                    or any(value <= 0 for value in data_shape)
                ):
                    raise ValueError(
                        f"incompatible shapes data={data_shape}, seg={segmentation_shape}"
                    )
                if str(data.dtype) != "float32" or str(segmentation.dtype) != "int16":
                    raise ValueError(
                        f"incompatible dtypes data={data.dtype}, seg={segmentation.dtype}"
                    )
                if not isinstance(properties, Mapping):
                    raise ValueError("properties pickle is not a mapping")
                missing_properties = sorted(required_properties - set(properties))
                if missing_properties:
                    raise ValueError(f"properties missing keys {missing_properties}")
            except Exception as exc:
                corrupt.append(f"{case_id}: {type(exc).__name__}: {exc}")
    ok = not missing and not extra and not non_files and not empty and not corrupt
    detail = (
        f"format=Blosc2, cases={len(expected_case_ids)}, files={len(actual)}, "
        f"missing={missing[:5]}, extra={extra[:5]}, non_files={non_files[:5]}, "
        f"empty={empty[:5]}, corrupt={corrupt[:5]}"
    )
    return ok, detail


def _validate_ground_truth(
    gt_directory: Path, expected_case_ids: tuple[str, ...]
) -> tuple[bool, str]:
    if not gt_directory.is_dir():
        return False, f"Missing ground-truth directory: {gt_directory}"
    expected = {f"{case_id}.nii.gz" for case_id in expected_case_ids}
    entries = tuple(gt_directory.iterdir())
    actual = {path.name for path in entries}
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    empty = sorted(path.name for path in entries if path.is_file() and path.stat().st_size <= 0)
    ok = not missing and not extra and not empty and all(path.is_file() for path in entries)
    return ok, (f"cases={len(actual)}, missing={missing[:5]}, extra={extra[:5]}, empty={empty[:5]}")


def _official_splits(case_ids: tuple[str, ...]) -> list[dict[str, list[str]]]:
    try:
        from nnunetv2.utilities.crossval_split import (  # type: ignore[import-not-found]
            generate_crossval_split,
        )
    except ImportError as exc:
        raise PreprocessingArtifactError(
            "Official nnU-Net cross-validation helper is unavailable"
        ) from exc
    if len(case_ids) < 5:
        raise PreprocessingArtifactError("At least five cases are required for official 5-fold CV")
    generated = generate_crossval_split(list(sorted(case_ids)), seed=12345, n_splits=5)
    return [
        {
            "train": [str(case_id) for case_id in fold["train"]],
            "val": [str(case_id) for case_id in fold["val"]],
        }
        for fold in generated
    ]


def _validate_or_create_splits(
    splits_path: Path,
    case_ids: tuple[str, ...],
    *,
    ensure_splits: bool,
) -> tuple[bool, str, bool]:
    expected = _official_splits(case_ids)
    created = False
    if not splits_path.exists():
        if not ensure_splits:
            return False, f"Missing: {splits_path}", created
        # This is exactly nnUNetTrainer.do_split's helper, seed, and case ordering.
        write_json_atomic(splits_path, expected)
        created = True
    try:
        with splits_path.open("r", encoding="utf-8") as handle:
            actual = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"Cannot read {splits_path}: {exc}", created
    if actual != expected:
        return (
            False,
            "Existing split differs from official nnU-Net 5-fold split "
            "(sorted IDs, seed=12345); it was not overwritten",
            created,
        )
    fold_sizes = [(len(fold["train"]), len(fold["val"])) for fold in expected]
    return True, f"official seed=12345, fold_sizes={fold_sizes}", created


def validate_preprocessing_artifacts(
    *,
    raw_dataset_dir: Path,
    preprocessed_dataset_dir: Path,
    dataset_name: str,
    configuration: str = "3d_fullres",
    plans_name: str = "nnUNetPlans",
    expected_case_count: int = 1251,
    ensure_splits: bool = False,
) -> PreprocessingArtifactReport:
    """Validate exact baseline artifacts and optionally finalize official splits."""

    raw_dataset_dir = raw_dataset_dir.resolve()
    preprocessed_dataset_dir = preprocessed_dataset_dir.resolve()
    report = PreprocessingArtifactReport(dataset_name, configuration, plans_name)
    report.details.update(
        {
            "raw_dataset_dir": str(raw_dataset_dir),
            "preprocessed_dataset_dir": str(preprocessed_dataset_dir),
            "expected_case_count": expected_case_count,
        }
    )

    dataset_json_path = raw_dataset_dir / "dataset.json"
    try:
        dataset_json = _read_json_object(dataset_json_path)
        case_ids = _raw_case_ids(raw_dataset_dir, dataset_json)
        count = dataset_json.get("numTraining")
        count_ok = count == expected_case_count and len(case_ids) == expected_case_count
        report.add(
            "raw dataset inventory",
            count_ok,
            f"dataset.json={count}, discovered={len(case_ids)}, expected={expected_case_count}",
        )
        report.details["case_ids"] = list(case_ids)
        report.details["case_count"] = len(case_ids)
        region_ok, region_detail = _validate_region_contract(dataset_json)
        report.add("upstream region contract", region_ok, region_detail)
    except (OSError, PreprocessingArtifactError) as exc:
        report.add("raw dataset inventory", False, str(exc))
        return report

    preprocessed_json_path = preprocessed_dataset_dir / "dataset.json"
    try:
        preprocessed_json = _read_json_object(preprocessed_json_path)
        report.add(
            "preprocessed dataset.json copy",
            preprocessed_json == dataset_json,
            str(preprocessed_json_path),
        )
    except (OSError, PreprocessingArtifactError) as exc:
        report.add("preprocessed dataset.json copy", False, str(exc))

    plans_path = preprocessed_dataset_dir / f"{plans_name}.json"
    plans_summary: dict[str, Any] | None = None
    try:
        plans = _read_json_object(plans_path)
        plans_summary = summarize_plans(plans_path, configuration)
        config = plans_summary["raw_configuration"]
        normalization = config.get("normalization_schemes")
        spacing = config.get("spacing")
        patch = config.get("patch_size")
        plan_ok = (
            plans.get("dataset_name") == dataset_name
            and plans.get("plans_name") == plans_name
            and config.get("preprocessor_name") == "DefaultPreprocessor"
            and isinstance(normalization, list)
            and len(normalization) == 4
            and isinstance(spacing, list)
            and len(spacing) == 3
            and all(isinstance(value, (int, float)) and value > 0 for value in spacing)
            and isinstance(patch, list)
            and len(patch) == 3
            and all(isinstance(value, int) and value > 0 for value in patch)
            and isinstance(config.get("batch_size"), int)
            and config["batch_size"] > 0
            and bool(plans_summary.get("architecture_class"))
            and bool(config.get("resampling_fn_data"))
            and bool(config.get("resampling_fn_seg"))
            and bool(config.get("data_identifier"))
        )
        report.add(
            "official plans/configuration",
            plan_ok,
            (
                f"dataset={plans.get('dataset_name')}, plans={plans.get('plans_name')}, "
                f"configuration={configuration}, preprocessor={config.get('preprocessor_name')}, "
                f"data_identifier={config.get('data_identifier')}"
            ),
        )
        report.details["plans"] = plans_summary
    except (
        AttributeError,
        KeyError,
        OSError,
        PreprocessingArtifactError,
        TypeError,
        ValueError,
    ) as exc:
        report.add("official plans/configuration", False, str(exc))

    fingerprint_path = preprocessed_dataset_dir / "dataset_fingerprint.json"
    try:
        fingerprint = _read_json_object(fingerprint_path)
        spacings = fingerprint.get("spacings")
        shapes = fingerprint.get("shapes_after_crop")
        fingerprint_ok = (
            isinstance(spacings, list)
            and len(spacings) == expected_case_count
            and isinstance(shapes, list)
            and len(shapes) == expected_case_count
            and isinstance(fingerprint.get("foreground_intensity_properties_per_channel"), Mapping)
        )
        report.add(
            "dataset fingerprint",
            fingerprint_ok,
            f"spacings={len(spacings) if isinstance(spacings, list) else 'invalid'}, "
            f"shapes={len(shapes) if isinstance(shapes, list) else 'invalid'}",
        )
    except (OSError, PreprocessingArtifactError) as exc:
        report.add("dataset fingerprint", False, str(exc))

    if plans_summary is not None and isinstance(plans_summary.get("data_identifier"), str):
        try:
            configuration_dir = ensure_within(
                preprocessed_dataset_dir / plans_summary["data_identifier"],
                preprocessed_dataset_dir,
                label="preprocessed configuration directory",
            )
            artifacts_ok, artifacts_detail = _validate_case_artifacts(configuration_dir, case_ids)
            report.add("preprocessed Blosc2 case inventory", artifacts_ok, artifacts_detail)
            report.details["configuration_directory"] = str(configuration_dir)
            report.details["preprocessed_format"] = "Blosc2"
        except (OSError, ValueError) as exc:
            report.add("preprocessed Blosc2 case inventory", False, str(exc))
    else:
        report.add(
            "preprocessed Blosc2 case inventory",
            False,
            "Cannot resolve configuration data_identifier from plans",
        )

    try:
        gt_ok, gt_detail = _validate_ground_truth(
            preprocessed_dataset_dir / "gt_segmentations", case_ids
        )
        report.add("preprocessed ground-truth inventory", gt_ok, gt_detail)
    except OSError as exc:
        report.add("preprocessed ground-truth inventory", False, str(exc))

    # Never create a split while another preprocessing artifact is incomplete.
    base_valid = report.valid
    if base_valid:
        try:
            split_ok, split_detail, created = _validate_or_create_splits(
                preprocessed_dataset_dir / "splits_final.json",
                case_ids,
                ensure_splits=ensure_splits,
            )
            report.add("official five-fold split", split_ok, split_detail)
            report.details["splits_created"] = created
            report.details["splits_file"] = str(preprocessed_dataset_dir / "splits_final.json")
        except (OSError, PreprocessingArtifactError) as exc:
            report.add("official five-fold split", False, str(exc))
    else:
        report.add(
            "official five-fold split",
            False,
            "Not created or accepted because preprocessing artifacts are incomplete",
        )
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dataset-dir", type=Path, required=True)
    parser.add_argument("--preprocessed-dataset-dir", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--configuration", default="3d_fullres")
    parser.add_argument("--plans-name", default="nnUNetPlans")
    parser.add_argument("--expected-case-count", type=int, default=1251)
    parser.add_argument("--ensure-splits", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        report = validate_preprocessing_artifacts(
            raw_dataset_dir=args.raw_dataset_dir,
            preprocessed_dataset_dir=args.preprocessed_dataset_dir,
            dataset_name=args.dataset_name,
            configuration=args.configuration,
            plans_name=args.plans_name,
            expected_case_count=args.expected_case_count,
            ensure_splits=args.ensure_splits,
        )
        write_json_atomic(args.output.resolve(), report.to_dict())
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"Preprocessing artifact audit failed: {type(exc).__name__}: {exc}")
        return 2

    print(
        f"Preprocessing artifact audit: {'PASS' if report.valid else 'FAIL'}; "
        f"cases={report.details.get('case_count', 'unknown')}; output={args.output.resolve()}"
    )
    for check in report.checks:
        if not check.ok:
            print(f"  - {check.name}: {check.detail}")
    return 0 if report.valid else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
