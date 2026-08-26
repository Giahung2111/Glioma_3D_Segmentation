"""Strict, read-only NIfTI validation for BraTS 2023 GLI datasets."""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import NDArray

from glioma_seg.utils.hashing import FileFingerprint, fingerprint_file
from glioma_seg.utils.logging import configure_logging

from .brats2023 import ALLOWED_LABELS, MODALITIES, BraTSCase
from .discover import DiscoveredDataset, DiscoveryError, discover_brats_datasets

DatasetValidationKind = Literal["training", "validation"]
ProgressCallback = Callable[[int, int, float], None]
DEFAULT_VALIDATION_WORKERS = 2


class DatasetValidationError(RuntimeError):
    """Raised when invalid data is supplied to a downstream stage."""


@dataclass(frozen=True, slots=True)
class NiftiFileValidation:
    role: str
    path: str
    fingerprint: FileFingerprint | None = None
    shape: tuple[int, ...] | None = None
    spacing: tuple[float, ...] | None = None
    orientation: tuple[str | None, ...] | None = None
    affine: tuple[tuple[float, ...], ...] | None = None
    dtype: str | None = None
    all_finite: bool | None = None
    unique_labels: tuple[int, ...] | None = None
    errors: tuple[str, ...] = ()

    @property
    def valid(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["valid"] = self.valid
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> NiftiFileValidation:
        fingerprint_value = value.get("fingerprint")
        fingerprint = (
            FileFingerprint.from_dict(fingerprint_value)
            if isinstance(fingerprint_value, Mapping)
            else None
        )
        shape_value = value.get("shape")
        spacing_value = value.get("spacing")
        orientation_value = value.get("orientation")
        affine_value = value.get("affine")
        labels_value = value.get("unique_labels")
        errors_value = value.get("errors", [])
        if not isinstance(errors_value, Sequence) or isinstance(errors_value, (str, bytes)):
            raise ValueError("NIfTI validation errors must be a sequence")
        return cls(
            role=str(value["role"]),
            path=str(value["path"]),
            fingerprint=fingerprint,
            shape=tuple(int(item) for item in shape_value) if shape_value is not None else None,
            spacing=(
                tuple(float(item) for item in spacing_value) if spacing_value is not None else None
            ),
            orientation=(
                tuple(item if item is None else str(item) for item in orientation_value)
                if orientation_value is not None
                else None
            ),
            affine=(
                tuple(tuple(float(item) for item in row) for row in affine_value)
                if affine_value is not None
                else None
            ),
            dtype=str(value["dtype"]) if value.get("dtype") is not None else None,
            all_finite=(bool(value["all_finite"]) if value.get("all_finite") is not None else None),
            unique_labels=(
                tuple(int(item) for item in labels_value) if labels_value is not None else None
            ),
            errors=tuple(str(item) for item in errors_value),
        )


@dataclass(frozen=True, slots=True)
class CaseValidation:
    case_id: str
    directory: str
    files: tuple[NiftiFileValidation, ...]
    missing_roles: tuple[str, ...] = ()
    unexpected_files: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def valid(self) -> bool:
        return not self.errors and not self.missing_roles and all(item.valid for item in self.files)

    @property
    def file_map(self) -> dict[str, NiftiFileValidation]:
        return {item.role: item for item in self.files}

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "directory": self.directory,
            "valid": self.valid,
            "missing_roles": list(self.missing_roles),
            "unexpected_files": list(self.unexpected_files),
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "files": [item.to_dict() for item in self.files],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CaseValidation:
        raw_files = value.get("files")
        if not isinstance(raw_files, list):
            raise ValueError("Case validation files must be a list")
        return cls(
            case_id=str(value["case_id"]),
            directory=str(value["directory"]),
            files=tuple(NiftiFileValidation.from_dict(item) for item in raw_files),
            missing_roles=tuple(str(item) for item in value.get("missing_roles", [])),
            unexpected_files=tuple(str(item) for item in value.get("unexpected_files", [])),
            errors=tuple(str(item) for item in value.get("errors", [])),
            warnings=tuple(str(item) for item in value.get("warnings", [])),
        )


@dataclass(frozen=True, slots=True)
class ValidationReport:
    schema_version: int
    created_at_utc: str
    dataset_root: str
    dataset_kind: DatasetValidationKind
    expected_case_count: int | None
    cases: tuple[CaseValidation, ...]
    validation_workers: int = 1
    elapsed_seconds: float | None = None
    duplicate_case_ids: tuple[str, ...] = ()
    non_case_entries: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def actual_case_count(self) -> int:
        return len(self.cases)

    @property
    def valid_case_count(self) -> int:
        return sum(case.valid for case in self.cases)

    @property
    def valid(self) -> bool:
        return (
            not self.errors
            and not self.duplicate_case_ids
            and self.valid_case_count == self.actual_case_count
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "created_at_utc": self.created_at_utc,
            "dataset_root": self.dataset_root,
            "dataset_kind": self.dataset_kind,
            "expected_case_count": self.expected_case_count,
            "actual_case_count": self.actual_case_count,
            "valid_case_count": self.valid_case_count,
            "valid": self.valid,
            "validation_workers": self.validation_workers,
            "elapsed_seconds": self.elapsed_seconds,
            "duplicate_case_ids": list(self.duplicate_case_ids),
            "non_case_entries": list(self.non_case_entries),
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "cases": [case.to_dict() for case in self.cases],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ValidationReport:
        raw_cases = value.get("cases")
        if not isinstance(raw_cases, list):
            raise ValueError("Validation report cases must be a list")
        kind = value.get("dataset_kind")
        if kind not in ("training", "validation"):
            raise ValueError(f"Invalid dataset_kind: {kind}")
        expected = value.get("expected_case_count")
        if expected is not None and (isinstance(expected, bool) or not isinstance(expected, int)):
            raise ValueError("expected_case_count must be an integer or null")
        report = cls(
            schema_version=int(value["schema_version"]),
            created_at_utc=str(value["created_at_utc"]),
            dataset_root=str(value["dataset_root"]),
            dataset_kind=kind,
            expected_case_count=expected,
            cases=tuple(CaseValidation.from_dict(item) for item in raw_cases),
            validation_workers=int(value.get("validation_workers", 1)),
            elapsed_seconds=(
                float(value["elapsed_seconds"])
                if value.get("elapsed_seconds") is not None
                else None
            ),
            duplicate_case_ids=tuple(str(item) for item in value.get("duplicate_case_ids", [])),
            non_case_entries=tuple(str(item) for item in value.get("non_case_entries", [])),
            errors=tuple(str(item) for item in value.get("errors", [])),
            warnings=tuple(str(item) for item in value.get("warnings", [])),
        )
        if report.schema_version != 1:
            raise ValueError(f"Unsupported validation report schema: {report.schema_version}")
        # Stored derived fields are checked so hand-edited/stale reports cannot claim success.
        if "actual_case_count" in value and value["actual_case_count"] != report.actual_case_count:
            raise ValueError("Validation report actual_case_count is inconsistent")
        if "valid_case_count" in value and value["valid_case_count"] != report.valid_case_count:
            raise ValueError("Validation report valid_case_count is inconsistent")
        if "valid" in value and value["valid"] is not report.valid:
            raise ValueError("Validation report valid flag is inconsistent")
        return report

    @classmethod
    def read_json(cls, path: str | Path) -> ValidationReport:
        with Path(path).open("r", encoding="utf-8") as stream:
            value = json.load(stream)
        if not isinstance(value, dict):
            raise ValueError("Validation report must be a JSON object")
        return cls.from_dict(value)

    def require_valid(self, *, kind: DatasetValidationKind | None = None) -> None:
        if kind is not None and self.dataset_kind != kind:
            raise DatasetValidationError(
                f"Expected a {kind} validation report, got {self.dataset_kind}"
            )
        if not self.valid:
            raise DatasetValidationError(
                f"Dataset validation failed: {self.valid_case_count}/{self.actual_case_count} "
                f"valid cases; global errors={list(self.errors)}"
            )


@dataclass(slots=True)
class _LoadedNifti:
    record: NiftiFileValidation
    affine_array: NDArray[np.float64] | None = field(repr=False, default=None)


def _load_and_validate_nifti(
    role: str,
    path: Path,
    *,
    require_3d: bool,
    scan_voxel_values: bool,
) -> _LoadedNifti:
    try:
        import nibabel as nib
    except ImportError as exc:  # pragma: no cover - environment failure, not a data case
        raise RuntimeError("nibabel is required for NIfTI validation") from exc

    errors: list[str] = []
    fingerprint: FileFingerprint | None = None
    shape: tuple[int, ...] | None = None
    spacing: tuple[float, ...] | None = None
    orientation: tuple[str | None, ...] | None = None
    affine: tuple[tuple[float, ...], ...] | None = None
    affine_array: NDArray[np.float64] | None = None
    dtype: str | None = None
    all_finite: bool | None = None
    unique_labels: tuple[int, ...] | None = None

    try:
        fingerprint = fingerprint_file(path)
    except (OSError, ValueError) as exc:
        errors.append(f"cannot fingerprint file: {exc}")

    try:
        image = cast("nib.Nifti1Image", nib.load(str(path)))
        shape = tuple(int(item) for item in image.shape)
        if require_3d and len(shape) != 3:
            errors.append(f"expected a 3D volume, got shape {shape}")
        affine_array = np.asarray(image.affine, dtype=np.float64)
        if affine_array.shape != (4, 4) or not np.isfinite(affine_array).all():
            errors.append("affine must be a finite 4x4 matrix")
        else:
            affine = tuple(tuple(float(item) for item in row) for row in affine_array)
        zooms = image.header.get_zooms()  # type: ignore[no-untyped-call]
        spacing = tuple(float(item) for item in zooms[: min(3, len(zooms))])
        if len(spacing) != 3 or not np.isfinite(spacing).all() or min(spacing) <= 0:
            errors.append(f"invalid voxel spacing: {spacing}")
        orientation = tuple(
            nib.aff2axcodes(image.affine)  # type: ignore[no-untyped-call]
        )
        dtype = str(image.get_data_dtype())  # type: ignore[no-untyped-call]

        if scan_voxel_values:
            voxels = np.asanyarray(image.dataobj)
            all_finite = bool(np.isfinite(voxels).all())
            if not all_finite:
                errors.append("volume contains NaN or infinite values")
            if role == "seg" and all_finite:
                rounded = np.rint(voxels)
                if not np.array_equal(voxels, rounded):
                    errors.append("segmentation contains non-integer values")
                else:
                    unique_labels = tuple(int(item) for item in np.unique(rounded))
                    unexpected = sorted(set(unique_labels) - ALLOWED_LABELS)
                    if unexpected:
                        errors.append(
                            f"segmentation labels {unexpected} are outside allowed labels "
                            "[0, 1, 2, 3]"
                        )
    except Exception as exc:  # nibabel emits several exception subclasses for corrupt files
        errors.append(f"cannot read NIfTI: {type(exc).__name__}: {exc}")

    return _LoadedNifti(
        record=NiftiFileValidation(
            role=role,
            path=str(path.resolve()),
            fingerprint=fingerprint,
            shape=shape,
            spacing=spacing,
            orientation=orientation,
            affine=affine,
            dtype=dtype,
            all_finite=all_finite,
            unique_labels=unique_labels,
            errors=tuple(errors),
        ),
        affine_array=affine_array,
    )


def _validate_geometry(
    loaded: Mapping[str, _LoadedNifti],
    *,
    affine_atol: float,
    affine_rtol: float,
    spacing_atol: float,
) -> list[str]:
    errors: list[str] = []
    reference = loaded.get("t1n")
    if reference is None:
        return errors
    for role in (*MODALITIES[1:], "seg"):
        candidate = loaded.get(role)
        if candidate is None:
            continue
        if (
            candidate.record.shape is not None
            and reference.record.shape is not None
            and candidate.record.shape != reference.record.shape
        ):
            errors.append(
                f"geometry mismatch: {role} shape {candidate.record.shape} != "
                f"t1n {reference.record.shape}"
            )
        if candidate.record.spacing is not None and reference.record.spacing is not None:
            if not np.allclose(
                candidate.record.spacing,
                reference.record.spacing,
                rtol=0,
                atol=spacing_atol,
            ):
                errors.append(
                    f"geometry mismatch: {role} spacing {candidate.record.spacing} != "
                    f"t1n {reference.record.spacing}"
                )
        if candidate.record.orientation != reference.record.orientation:
            errors.append(
                f"orientation mismatch: {role} {candidate.record.orientation} != "
                f"t1n {reference.record.orientation}"
            )
        if candidate.affine_array is not None and reference.affine_array is not None:
            if not np.allclose(
                candidate.affine_array,
                reference.affine_array,
                rtol=affine_rtol,
                atol=affine_atol,
            ):
                errors.append(f"affine mismatch: {role} is not compatible with t1n")
    return errors


def validate_case(
    case: BraTSCase,
    *,
    dataset_kind: DatasetValidationKind,
    affine_atol: float = 1e-5,
    affine_rtol: float = 1e-5,
    spacing_atol: float = 1e-6,
    require_3d: bool = True,
    scan_voxel_values: bool = True,
) -> CaseValidation:
    """Validate all available files and cross-modality geometry for one case."""

    require_segmentation = dataset_kind == "training"
    missing = case.missing_roles(require_segmentation=require_segmentation)
    errors: list[str] = []
    warnings: list[str] = []
    if not case.directory_name_matches:
        errors.append(
            f"case directory name '{case.directory.name}' does not match case ID '{case.case_id}'"
        )
    if missing:
        errors.append(f"missing required files: {', '.join(missing)}")
    if dataset_kind == "validation" and case.has_segmentation:
        errors.append("official validation case unexpectedly contains a segmentation")
    if case.unexpected_files:
        warnings.append(f"unexpected files: {len(case.unexpected_files)}")

    loaded: dict[str, _LoadedNifti] = {}
    for role, path in case.files:
        loaded[role] = _load_and_validate_nifti(
            role,
            path,
            require_3d=require_3d,
            scan_voxel_values=scan_voxel_values,
        )
    errors.extend(
        _validate_geometry(
            loaded,
            affine_atol=affine_atol,
            affine_rtol=affine_rtol,
            spacing_atol=spacing_atol,
        )
    )
    return CaseValidation(
        case_id=case.case_id,
        directory=str(case.directory.resolve()),
        files=tuple(loaded[role].record for role in sorted(loaded)),
        missing_roles=missing,
        unexpected_files=tuple(str(path) for path in case.unexpected_files),
        errors=tuple(errors),
        warnings=tuple(warnings),
    )


def validate_dataset(
    dataset: DiscoveredDataset,
    *,
    dataset_kind: DatasetValidationKind,
    expected_case_count: int | None,
    affine_atol: float = 1e-5,
    affine_rtol: float = 1e-5,
    spacing_atol: float = 1e-6,
    require_3d: bool = True,
    scan_voxel_values: bool = True,
    workers: int = 1,
    progress_every_cases: int = 10,
    progress_callback: ProgressCallback | None = None,
) -> ValidationReport:
    """Fully validate every discovered case; broken cases are retained in the report."""

    if not 1 <= workers <= 8:
        raise ValueError("workers must be between 1 and 8 for memory-safe NIfTI validation")
    if progress_every_cases <= 0:
        raise ValueError("progress_every_cases must be positive")

    errors: list[str] = []
    warnings: list[str] = []
    if expected_case_count is not None and dataset.case_count != expected_case_count:
        errors.append(
            f"case count mismatch: expected {expected_case_count}, found {dataset.case_count}"
        )
    if dataset.kind != dataset_kind and dataset.kind != "mixed":
        errors.append(f"dataset content looks like {dataset.kind}, expected {dataset_kind}")
    if dataset.non_case_entries:
        warnings.append(f"non-case entries: {len(dataset.non_case_entries)}")

    ordered_cases = tuple(
        sorted(dataset.cases, key=lambda case: (case.case_id, str(case.directory)))
    )
    started = time.perf_counter()
    completed = 0
    results: list[CaseValidation | None] = [None] * len(ordered_cases)

    def run(case: BraTSCase) -> CaseValidation:
        return validate_case(
            case,
            dataset_kind=dataset_kind,
            affine_atol=affine_atol,
            affine_rtol=affine_rtol,
            spacing_atol=spacing_atol,
            require_3d=require_3d,
            scan_voxel_values=scan_voxel_values,
        )

    def record_progress() -> None:
        if progress_callback is not None and (
            completed % progress_every_cases == 0 or completed == len(ordered_cases)
        ):
            progress_callback(completed, len(ordered_cases), time.perf_counter() - started)

    if workers == 1:
        for index, case in enumerate(ordered_cases):
            try:
                results[index] = run(case)
            except Exception as exc:
                # Retain the case as an explicit failure; never skip it silently.
                results[index] = CaseValidation(
                    case_id=case.case_id,
                    directory=str(case.directory.resolve()),
                    files=(),
                    errors=(f"validation worker failed: {type(exc).__name__}: {exc}",),
                )
            completed += 1
            record_progress()
    else:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="nifti-validation") as pool:
            future_indices: dict[Future[CaseValidation], int] = {
                pool.submit(run, case): index for index, case in enumerate(ordered_cases)
            }
            for future in as_completed(future_indices):
                index = future_indices[future]
                case = ordered_cases[index]
                try:
                    results[index] = future.result()
                except Exception as exc:
                    results[index] = CaseValidation(
                        case_id=case.case_id,
                        directory=str(case.directory.resolve()),
                        files=(),
                        errors=(f"validation worker failed: {type(exc).__name__}: {exc}",),
                    )
                completed += 1
                record_progress()

    if any(result is None for result in results):  # pragma: no cover - defensive invariant
        raise RuntimeError("Internal validation error: one or more cases produced no result")
    cases = tuple(result for result in results if result is not None)
    elapsed_seconds = time.perf_counter() - started
    return ValidationReport(
        schema_version=1,
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        dataset_root=str(dataset.root.resolve()),
        dataset_kind=dataset_kind,
        expected_case_count=expected_case_count,
        cases=cases,
        validation_workers=workers,
        elapsed_seconds=elapsed_seconds,
        duplicate_case_ids=dataset.duplicate_case_ids,
        non_case_entries=tuple(str(path) for path in dataset.non_case_entries),
        errors=tuple(errors),
        warnings=tuple(warnings),
    )


def _atomic_write(path: Path, content: str) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def write_validation_json(report: ValidationReport, path: str | Path) -> None:
    _atomic_write(
        Path(path),
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False, sort_keys=False) + "\n",
    )


def write_validation_csv(report: ValidationReport, path: str | Path) -> None:
    """Write one explicit row per case, including all broken cases."""

    import io

    output = io.StringIO(newline="")
    fieldnames = [
        "record_type",
        "dataset_kind",
        "expected_case_count",
        "actual_case_count",
        "case_id",
        "case_directory",
        "status",
        "duplicate_case_ids",
        "non_case_entries",
        "global_errors",
        "global_warnings",
        "missing_roles",
        "unexpected_files",
        "errors",
        "warnings",
        "t1n_path",
        "t1c_path",
        "t2w_path",
        "t2f_path",
        "seg_path",
        "shape",
        "spacing",
        "orientation",
        "unique_labels",
    ]
    writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerow(
        {
            "record_type": "dataset_summary",
            "dataset_kind": report.dataset_kind,
            "expected_case_count": report.expected_case_count,
            "actual_case_count": report.actual_case_count,
            "status": "PASS" if report.valid else "FAIL",
            "duplicate_case_ids": ";".join(report.duplicate_case_ids),
            "non_case_entries": ";".join(report.non_case_entries),
            "global_errors": "; ".join(report.errors),
            "global_warnings": "; ".join(report.warnings),
        }
    )
    for case in report.cases:
        files = case.file_map
        reference = files.get("t1n")
        segmentation = files.get("seg")
        file_errors = [f"{item.role}: {error}" for item in case.files for error in item.errors]
        writer.writerow(
            {
                "record_type": "case",
                "dataset_kind": report.dataset_kind,
                "expected_case_count": report.expected_case_count,
                "actual_case_count": report.actual_case_count,
                "case_id": case.case_id,
                "case_directory": case.directory,
                "status": "PASS" if case.valid else "FAIL",
                "missing_roles": ";".join(case.missing_roles),
                "unexpected_files": ";".join(case.unexpected_files),
                "errors": "; ".join((*case.errors, *file_errors)),
                "warnings": "; ".join(case.warnings),
                **{
                    f"{role}_path": files[role].path if role in files else ""
                    for role in (*MODALITIES, "seg")
                },
                "shape": (
                    "x".join(map(str, reference.shape)) if reference and reference.shape else ""
                ),
                "spacing": (
                    "x".join(map(str, reference.spacing)) if reference and reference.spacing else ""
                ),
                "orientation": (
                    "".join(item or "?" for item in reference.orientation)
                    if reference and reference.orientation
                    else ""
                ),
                "unique_labels": (
                    ";".join(map(str, segmentation.unique_labels))
                    if segmentation and segmentation.unique_labels is not None
                    else ""
                ),
            }
        )
    _atomic_write(Path(path), output.getvalue())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--kind", choices=("training", "validation"), default="training")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--expected-training-cases", type=int, default=1251)
    parser.add_argument("--expected-validation-cases", type=int, default=219)
    parser.add_argument("--affine-atol", type=float, default=1e-5)
    parser.add_argument("--affine-rtol", type=float, default=1e-5)
    parser.add_argument("--spacing-atol", type=float, default=1e-6)
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_VALIDATION_WORKERS,
        help="Concurrent case validators (1-8; default: 2 for bounded RAM use)",
    )
    parser.add_argument("--progress-every", type=int, default=10)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logger = configure_logging()
    logger.info("Discovering BraTS datasets by content under %s", args.data_root.resolve())
    discovery = discover_brats_datasets(args.data_root)
    try:
        dataset = discovery.require_unique(args.kind)
    except DiscoveryError as exc:
        logger.error("Dataset discovery failed: %s", exc)
        return 2
    expected = (
        args.expected_training_cases if args.kind == "training" else args.expected_validation_cases
    )
    logger.info("Validating %d %s cases from %s", dataset.case_count, args.kind, dataset.root)

    def log_progress(completed: int, total: int, elapsed: float) -> None:
        percent = 100.0 if total == 0 else completed * 100.0 / total
        logger.info(
            "Dataset validation progress: %d/%d (%.1f%%), elapsed %.1f s",
            completed,
            total,
            percent,
            elapsed,
        )

    report = validate_dataset(
        dataset,
        dataset_kind=args.kind,
        expected_case_count=expected,
        affine_atol=args.affine_atol,
        affine_rtol=args.affine_rtol,
        spacing_atol=args.spacing_atol,
        workers=args.workers,
        progress_every_cases=args.progress_every,
        progress_callback=log_progress,
    )
    write_validation_json(report, args.output_json)
    write_validation_csv(report, args.output_csv)
    logger.info(
        "Validation %s: %d/%d valid; JSON=%s; CSV=%s",
        "PASSED" if report.valid else "FAILED",
        report.valid_case_count,
        report.actual_case_count,
        args.output_json.resolve(),
        args.output_csv.resolve(),
    )
    return 0 if report.valid else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
