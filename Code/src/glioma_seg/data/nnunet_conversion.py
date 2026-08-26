"""Idempotent, source-preserving BraTS-to-nnU-Net v2 conversion."""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import tempfile
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from glioma_seg.config.loader import load_dataset_config
from glioma_seg.config.schema import DatasetConfig
from glioma_seg.utils.hashing import FileFingerprint, sha256_file
from glioma_seg.utils.logging import configure_logging
from glioma_seg.utils.paths import ensure_within

from .brats2023 import MODALITIES, nnunet_image_name, nnunet_label_name, source_filename
from .validate import CaseValidation, DatasetValidationError, ValidationReport

MaterializationMethod = Literal["hardlink", "copy", "reused_hardlink", "reused_file"]

# Keep progress visible when the module is executed with ``python -m``. In that
# mode ``__name__`` is ``__main__`` and would not inherit the project logger.
LOGGER = logging.getLogger("glioma_seg.data.nnunet_conversion")


class ConversionError(RuntimeError):
    """Raised when conversion cannot proceed without risking source or output data."""


@dataclass(frozen=True, slots=True)
class ConvertedFile:
    case_id: str
    role: str
    source: str
    destination: str
    method: MaterializationMethod
    size_bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class ConversionReport:
    schema_version: int
    created_at_utc: str
    dataset_id: int
    dataset_name: str
    dataset_directory: str
    training_cases: int
    test_cases: int
    files: tuple[ConvertedFile, ...]
    dataset_json: Mapping[str, Any]

    @property
    def method_counts(self) -> dict[str, int]:
        return dict(Counter(item.method for item in self.files))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "created_at_utc": self.created_at_utc,
            "dataset_id": self.dataset_id,
            "dataset_name": self.dataset_name,
            "dataset_directory": self.dataset_directory,
            "training_cases": self.training_cases,
            "test_cases": self.test_cases,
            "method_counts": self.method_counts,
            "dataset_json": dict(self.dataset_json),
            "files": [asdict(item) for item in self.files],
        }


def build_dataset_json(spec: DatasetConfig, *, num_training: int) -> dict[str, Any]:
    """Build dataset.json while preserving the medically meaningful region order."""

    if num_training <= 0:
        raise ConversionError("numTraining must come from at least one validated case")
    if num_training != spec.expected_training_cases:
        raise ConversionError(
            f"Validated case count {num_training} does not match expected "
            f"{spec.expected_training_cases}"
        )
    # Do not sort this mapping: nnU-Net maps region heads back in insertion order.
    return {
        "channel_names": dict(spec.channel_names),
        "labels": spec.nnunet_labels,
        "regions_class_order": list(spec.regions_class_order),
        "numTraining": num_training,
        "file_ending": spec.file_ending,
    }


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, sort_keys=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _write_dataset_json_idempotently(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as stream:
                current = json.load(stream)
        except (OSError, json.JSONDecodeError) as exc:
            raise ConversionError(f"Existing dataset.json is unreadable: {path}: {exc}") from exc
        if current != value:
            raise ConversionError(
                f"Refusing to overwrite conflicting dataset.json at {path}; "
                "remove or archive the generated dataset explicitly"
            )
        return
    _atomic_write_json(path, value)


def _destination_matches(destination: Path, fingerprint: FileFingerprint) -> bool:
    try:
        stat = destination.stat()
    except OSError:
        return False
    return stat.st_size == fingerprint.size_bytes and sha256_file(destination) == fingerprint.sha256


def _reuse_existing(
    source: Path, destination: Path, fingerprint: FileFingerprint
) -> MaterializationMethod:
    if not destination.is_file():
        raise ConversionError(f"Destination exists but is not a regular file: {destination}")
    if not _destination_matches(destination, fingerprint):
        raise ConversionError(f"Refusing to overwrite conflicting generated file: {destination}")
    try:
        return "reused_hardlink" if os.path.samefile(source, destination) else "reused_file"
    except OSError:
        return "reused_file"


def _copy_atomically(
    source: Path, destination: Path, fingerprint: FileFingerprint
) -> MaterializationMethod:
    partial = destination.with_name(f".{destination.name}.glioma-partial")
    ensure_within(partial, destination.parent, label="partial conversion file")
    if partial.exists():
        # This exact internal filename is recoverable output from an interrupted copy.
        if _destination_matches(partial, fingerprint):
            if destination.exists():
                partial.unlink()
                return _reuse_existing(source, destination, fingerprint)
            os.replace(partial, destination)
            return "copy"
        partial.unlink()

    shutil.copy2(source, partial)
    if not _destination_matches(partial, fingerprint):
        partial.unlink(missing_ok=True)
        raise ConversionError(f"Copied file failed checksum verification: {destination}")
    if destination.exists():
        partial.unlink()
        return _reuse_existing(source, destination, fingerprint)
    os.replace(partial, destination)
    return "copy"


def materialize_file(
    source: str | Path,
    destination: str | Path,
    fingerprint: FileFingerprint,
    *,
    prefer_hardlink: bool = True,
    source_hash_already_verified: bool = False,
) -> MaterializationMethod:
    """Create an output without overwriting, with hardlink then copy fallback."""

    source_path = Path(source).resolve()
    destination_path = Path(destination).resolve()
    if source_path != Path(fingerprint.path).resolve():
        raise ConversionError(f"Fingerprint path does not match source: {source_path}")
    if not fingerprint.matches(verify_hash=not source_hash_already_verified):
        raise ConversionError(f"Source changed after validation: {source_path}")
    destination_path.parent.mkdir(parents=True, exist_ok=True)

    if destination_path.exists():
        return _reuse_existing(source_path, destination_path, fingerprint)
    if prefer_hardlink:
        try:
            os.link(source_path, destination_path)
            # samefile proves both paths refer to the already-validated inode.
            if not os.path.samefile(source_path, destination_path):  # pragma: no cover
                raise ConversionError(f"Hardlink identity check failed: {destination_path}")
            return "hardlink"
        except FileExistsError:
            return _reuse_existing(source_path, destination_path, fingerprint)
        except OSError:
            # Cross-device, permissions, and unsupported filesystems all fall back safely.
            pass
    return _copy_atomically(source_path, destination_path, fingerprint)


def _validated_source(
    case: CaseValidation, role: str, dataset_root: Path
) -> tuple[Path, FileFingerprint]:
    try:
        record = case.file_map[role]
    except KeyError as exc:
        raise ConversionError(f"Validated case {case.case_id} has no {role} record") from exc
    if record.fingerprint is None:
        raise ConversionError(f"Validated case {case.case_id} {role} has no fingerprint")
    source = ensure_within(record.path, dataset_root, label="validated raw source")
    expected_name = source_filename(case.case_id, role)
    if source.name != expected_name:
        raise ConversionError(
            f"Validated source name mismatch for {case.case_id} {role}: {source.name}"
        )
    return source, record.fingerprint


def _validate_report_sources(
    report: ValidationReport,
    *,
    verify_hash: bool = True,
    progress_every: int = 25,
) -> None:
    report.require_valid()
    dataset_root = Path(report.dataset_root).resolve()
    case_ids: set[str] = set()
    required_roles = (*MODALITIES, "seg") if report.dataset_kind == "training" else MODALITIES
    started = time.monotonic()
    total = len(report.cases)
    for index, case in enumerate(report.cases, start=1):
        if case.case_id in case_ids:
            raise ConversionError(f"Duplicate case ID in validation report: {case.case_id}")
        case_ids.add(case.case_id)
        for role in required_roles:
            source, fingerprint = _validated_source(case, role, dataset_root)
            if not fingerprint.matches(verify_hash=verify_hash):
                raise ConversionError(f"Raw source changed after validation: {source}")
        if report.dataset_kind == "validation" and "seg" in case.file_map:
            raise ConversionError(
                f"Official validation case {case.case_id} has a segmentation; "
                "refusing imagesTs conversion"
            )
        if progress_every > 0 and (index % progress_every == 0 or index == total):
            LOGGER.info(
                "Source re-verification progress (%s): %d/%d (%.1f%%), elapsed %.1f s",
                report.dataset_kind,
                index,
                total,
                100.0 * index / total,
                time.monotonic() - started,
            )


def _assert_generated_directory(
    directory: Path, expected_names: set[str], *, require_complete: bool
) -> None:
    """Reject stale or foreign outputs instead of silently mixing conversions."""

    if not directory.exists():
        if require_complete and expected_names:
            raise ConversionError(f"Expected generated directory is missing: {directory}")
        return
    if not directory.is_dir():
        raise ConversionError(f"Generated output path is not a directory: {directory}")
    entries = list(directory.iterdir())
    unexpected = sorted(
        entry.name
        for entry in entries
        if entry.name not in expected_names
        and not (
            entry.name.startswith(".")
            and entry.name.endswith(".glioma-partial")
            and entry.name[1 : -len(".glioma-partial")] in expected_names
        )
    )
    if unexpected:
        raise ConversionError(f"Unexpected generated entries in {directory}: {unexpected[:10]}")
    if require_complete:
        actual = {entry.name for entry in entries if not entry.name.startswith(".")}
        missing = sorted(expected_names - actual)
        if missing:
            raise ConversionError(
                f"Generated outputs are incomplete in {directory}: {missing[:10]}"
            )


def convert_brats_to_nnunet(
    training_report: ValidationReport,
    output_root: str | Path,
    spec: DatasetConfig,
    *,
    official_validation_report: ValidationReport | None = None,
    prefer_hardlink: bool = True,
    report_path: str | Path | None = None,
) -> ConversionReport:
    """Materialize a validated dataset without modifying or silently dropping sources."""

    training_report.require_valid(kind="training")
    if training_report.actual_case_count != spec.expected_training_cases:
        raise DatasetValidationError(
            f"Expected {spec.expected_training_cases} validated training cases, "
            f"got {training_report.actual_case_count}"
        )
    _validate_report_sources(training_report)

    if official_validation_report is not None:
        official_validation_report.require_valid(kind="validation")
        if official_validation_report.actual_case_count != spec.expected_validation_cases:
            raise DatasetValidationError(
                f"Expected {spec.expected_validation_cases} official validation cases, "
                f"got {official_validation_report.actual_case_count}"
            )
        overlap = {case.case_id for case in training_report.cases} & {
            case.case_id for case in official_validation_report.cases
        }
        if overlap:
            raise ConversionError(
                f"Training and official validation IDs overlap: {sorted(overlap)[:10]}"
            )
        _validate_report_sources(official_validation_report)

    root = Path(output_root).resolve()
    dataset_directory = ensure_within(root / spec.dataset_name, root, label="nnU-Net dataset")
    images_tr = dataset_directory / "imagesTr"
    labels_tr = dataset_directory / "labelsTr"
    images_ts = dataset_directory / "imagesTs"

    expected_images_tr = {
        nnunet_image_name(case.case_id, modality)
        for case in training_report.cases
        for modality in MODALITIES
    }
    expected_labels_tr = {nnunet_label_name(case.case_id) for case in training_report.cases}
    expected_images_ts = (
        {
            nnunet_image_name(case.case_id, modality)
            for case in official_validation_report.cases
            for modality in MODALITIES
        }
        if official_validation_report is not None
        else set()
    )
    _assert_generated_directory(images_tr, expected_images_tr, require_complete=False)
    _assert_generated_directory(labels_tr, expected_labels_tr, require_complete=False)
    _assert_generated_directory(images_ts, expected_images_ts, require_complete=False)
    images_tr.mkdir(parents=True, exist_ok=True)
    labels_tr.mkdir(parents=True, exist_ok=True)
    if official_validation_report is not None:
        images_ts.mkdir(parents=True, exist_ok=True)

    converted: list[ConvertedFile] = []

    def add_file(case: CaseValidation, role: str, destination: Path, source_root: Path) -> None:
        destination = ensure_within(destination, dataset_directory, label="nnU-Net output")
        source, fingerprint = _validated_source(case, role, source_root)
        method = materialize_file(
            source,
            destination,
            fingerprint,
            prefer_hardlink=prefer_hardlink,
            source_hash_already_verified=True,
        )
        converted.append(
            ConvertedFile(
                case_id=case.case_id,
                role=role,
                source=str(source),
                destination=str(destination),
                method=method,
                size_bytes=fingerprint.size_bytes,
                sha256=fingerprint.sha256,
            )
        )

    training_root = Path(training_report.dataset_root).resolve()
    materialize_started = time.monotonic()
    sorted_training_cases = sorted(training_report.cases, key=lambda item: item.case_id)
    for index, case in enumerate(sorted_training_cases, start=1):
        for modality in MODALITIES:
            add_file(
                case,
                modality,
                images_tr / nnunet_image_name(case.case_id, modality),
                training_root,
            )
        add_file(case, "seg", labels_tr / nnunet_label_name(case.case_id), training_root)
        if index % 25 == 0 or index == len(sorted_training_cases):
            LOGGER.info(
                "Materialization progress (training): %d/%d (%.1f%%), elapsed %.1f s",
                index,
                len(sorted_training_cases),
                100.0 * index / len(sorted_training_cases),
                time.monotonic() - materialize_started,
            )

    if official_validation_report is not None:
        validation_root = Path(official_validation_report.dataset_root).resolve()
        sorted_validation_cases = sorted(
            official_validation_report.cases, key=lambda item: item.case_id
        )
        validation_started = time.monotonic()
        for index, case in enumerate(sorted_validation_cases, start=1):
            for modality in MODALITIES:
                add_file(
                    case,
                    modality,
                    images_ts / nnunet_image_name(case.case_id, modality),
                    validation_root,
                )
            if index % 25 == 0 or index == len(sorted_validation_cases):
                LOGGER.info(
                    "Materialization progress (official validation): "
                    "%d/%d (%.1f%%), elapsed %.1f s",
                    index,
                    len(sorted_validation_cases),
                    100.0 * index / len(sorted_validation_cases),
                    time.monotonic() - validation_started,
                )

    _assert_generated_directory(images_tr, expected_images_tr, require_complete=True)
    _assert_generated_directory(labels_tr, expected_labels_tr, require_complete=True)
    _assert_generated_directory(images_ts, expected_images_ts, require_complete=True)

    dataset_json = build_dataset_json(spec, num_training=training_report.actual_case_count)
    _write_dataset_json_idempotently(dataset_directory / "dataset.json", dataset_json)

    # A final metadata check catches any accidental write through a hardlink. The
    # full SHA-256 was verified immediately before conversion, avoiding three full
    # extra reads of a dataset that can be hundreds of gigabytes.
    _validate_report_sources(training_report, verify_hash=False)
    if official_validation_report is not None:
        _validate_report_sources(official_validation_report, verify_hash=False)

    result = ConversionReport(
        schema_version=1,
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        dataset_id=spec.dataset_id,
        dataset_name=spec.dataset_name,
        dataset_directory=str(dataset_directory),
        training_cases=training_report.actual_case_count,
        test_cases=(
            official_validation_report.actual_case_count
            if official_validation_report is not None
            else 0
        ),
        files=tuple(converted),
        dataset_json=dataset_json,
    )
    manifest = (
        Path(report_path).resolve()
        if report_path is not None
        else root / ".glioma_manifests" / f"{spec.dataset_name}_conversion.json"
    )
    _atomic_write_json(manifest, result.to_dict())
    return result


def _default_dataset_config() -> Path:
    return Path(__file__).resolve().parents[3] / "configs" / "datasets" / "brats2023_gli.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, help="Retained for CLI compatibility/audit logs")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--validation-json", type=Path, required=True)
    parser.add_argument("--official-validation-json", type=Path)
    parser.add_argument("--include-validation", action="store_true")
    parser.add_argument("--dataset-config", type=Path, default=_default_dataset_config())
    parser.add_argument("--report-json", type=Path)
    parser.add_argument(
        "--force-copy",
        action="store_true",
        help="Use verified copies instead of attempting hardlinks",
    )
    parser.add_argument("--expected-training-cases", type=int, default=1251)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logger = configure_logging()
    spec = load_dataset_config(args.dataset_config)
    if args.expected_training_cases != spec.expected_training_cases:
        logger.error(
            "CLI expected count %d conflicts with dataset config count %d",
            args.expected_training_cases,
            spec.expected_training_cases,
        )
        return 2
    if args.include_validation and args.official_validation_json is None:
        logger.error("--include-validation requires --official-validation-json")
        return 2
    if not args.include_validation and args.official_validation_json is not None:
        logger.error("Pass --include-validation when providing --official-validation-json")
        return 2

    try:
        training_report = ValidationReport.read_json(args.validation_json)
        validation_report = (
            ValidationReport.read_json(args.official_validation_json)
            if args.official_validation_json is not None
            else None
        )
        if args.data_root is not None:
            raw_root = args.data_root.resolve()
            report_root = Path(training_report.dataset_root).resolve()
            try:
                report_root.relative_to(raw_root)
            except ValueError as exc:
                raise ConversionError(
                    f"Validation report source {report_root} is outside --data-root {raw_root}"
                ) from exc
        result = convert_brats_to_nnunet(
            training_report,
            args.output_root,
            spec,
            official_validation_report=validation_report,
            prefer_hardlink=not args.force_copy,
            report_path=args.report_json,
        )
    except (ConversionError, DatasetValidationError, OSError, ValueError) as exc:
        logger.error("nnU-Net conversion failed: %s", exc)
        return 2

    logger.info(
        "Conversion complete: %d training, %d official validation cases; methods=%s; %s",
        result.training_cases,
        result.test_cases,
        result.method_counts,
        result.dataset_directory,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
