"""Complete and inventory a self-contained experiment report bundle.

This module owns report packaging only.  It never modifies nnU-Net outputs or
upstream source trees.  Required training-data validation artifacts are
materialized from the workspace-wide reports with byte-for-byte verification;
official-evaluator compatibility aliases are best-effort and never overwrite
divergent files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from glioma_seg.monitoring.timing import write_json_atomic
from glioma_seg.utils.hashing import sha256_file

TRAINING_VALIDATION_FILES: tuple[str, str] = (
    "data_validation.json",
    "data_validation.csv",
)
OFFICIAL_VALIDATION_FILES: tuple[str, str] = (
    "official_validation_data_validation.json",
    "official_validation_data_validation.csv",
)
OFFICIAL_ALIASES: Mapping[str, str] = {
    "official_lesionwise_metrics_summary.csv": "official_brats_metrics_summary.csv",
    "official_lesionwise_metrics_summary.json": "official_brats_metrics_summary.json",
    "official_lesionwise_metrics_per_case.csv": "official_brats_metrics_per_case.csv",
}
_KNOWN_LOG_ORDER: Mapping[str, int] = {
    "plan_and_preprocess.log": 10,
    "benchmark.log": 20,
    "train.log": 30,
    "predict.log": 40,
}
_ARTIFACT_KEY_RE = re.compile(r"[^A-Za-z0-9_]+")
_FRIENDLY_ARTIFACT_KEYS: Mapping[str, str] = {
    "summary.md": "summary",
    "weekly_discussion.md": "weekly_discussion",
    "pipeline.log": "pipeline_log",
    "data_validation.json": "data_validation_json",
    "data_validation.csv": "data_validation_csv",
    "official_validation_data_validation.json": "official_validation_data_validation_json",
    "official_validation_data_validation.csv": "official_validation_data_validation_csv",
    "official_brats_metrics_summary.csv": "official_brats_metrics_summary",
    "official_brats_metrics_summary.json": "official_brats_metrics_summary_json",
    "official_brats_metrics_per_case.csv": "official_brats_metrics_per_case",
}


class ReportBundleError(RuntimeError):
    """Raised when required report provenance is missing or inconsistent."""


@dataclass(frozen=True)
class MaterializedArtifact:
    path: str
    source: str
    sha256: str
    size_bytes: int
    method: Literal["existing", "hardlink", "copy"]


@dataclass(frozen=True)
class ReportBundleResult:
    phase: Literal["prepare", "finalize"]
    pipeline_log: str
    pipeline_sources: tuple[Mapping[str, Any], ...]
    validation_artifacts: tuple[MaterializedArtifact, ...]
    official_aliases: tuple[MaterializedArtifact, ...]
    official_alias_warning: str | None
    recorded_artifact_count: int | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_json_object(path: Path, *, description: str) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size == 0:
        raise ReportBundleError(f"{description} is missing or empty: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReportBundleError(f"{description} is not valid UTF-8 JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ReportBundleError(f"{description} must contain a JSON object: {path}")
    return value


def _validate_validation_pair(json_path: Path, csv_path: Path, *, dataset_kind: str) -> None:
    payload = _load_json_object(json_path, description=f"{dataset_kind} validation manifest")
    if payload.get("valid") is not True:
        raise ReportBundleError(
            f"{dataset_kind} validation manifest is not marked valid: {json_path}"
        )
    if payload.get("dataset_kind") != dataset_kind:
        raise ReportBundleError(
            f"{dataset_kind} validation manifest has dataset_kind={payload.get('dataset_kind')!r}: "
            f"{json_path}"
        )
    counts = (
        payload.get("expected_case_count"),
        payload.get("actual_case_count"),
        payload.get("valid_case_count"),
    )
    if any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in counts):
        raise ReportBundleError(
            f"{dataset_kind} validation manifest has invalid case counts {counts}: {json_path}"
        )
    if len(set(counts)) != 1:
        raise ReportBundleError(
            f"{dataset_kind} validation counts are incomplete/inconsistent {counts}: {json_path}"
        )
    if payload.get("errors") not in ([], None):
        raise ReportBundleError(f"{dataset_kind} validation manifest contains errors: {json_path}")
    if not csv_path.is_file() or csv_path.stat().st_size == 0:
        raise ReportBundleError(f"{dataset_kind} validation CSV is missing or empty: {csv_path}")
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            required_fields = {"record_type", "dataset_kind", "status"}
            if reader.fieldnames is None or not required_fields.issubset(reader.fieldnames):
                raise ReportBundleError(
                    f"{dataset_kind} validation CSV lacks required columns: {csv_path}"
                )
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ReportBundleError(
            f"Unable to parse {dataset_kind} validation CSV: {csv_path}"
        ) from exc
    summaries = [row for row in rows if row.get("record_type") == "dataset_summary"]
    cases = [row for row in rows if row.get("record_type") == "case"]
    if len(summaries) != 1 or summaries[0].get("dataset_kind") != dataset_kind:
        raise ReportBundleError(
            f"{dataset_kind} validation CSV must contain one matching dataset summary: {csv_path}"
        )
    if summaries[0].get("status") != "PASS" or any(row.get("status") != "PASS" for row in cases):
        raise ReportBundleError(f"{dataset_kind} validation CSV contains a failed row: {csv_path}")
    if len(cases) != counts[2]:
        raise ReportBundleError(
            f"{dataset_kind} validation CSV has {len(cases)} cases, "
            f"expected {counts[2]}: {csv_path}"
        )


def _materialize_verified(source: Path, destination: Path) -> MaterializedArtifact:
    if not source.is_file() or source.stat().st_size == 0:
        raise ReportBundleError(f"Artifact source is missing or empty: {source}")
    source = source.resolve()
    destination = destination.resolve()
    source_hash = sha256_file(source)
    if destination.exists():
        if not destination.is_file():
            raise ReportBundleError(f"Artifact destination is not a file: {destination}")
        destination_hash = sha256_file(destination)
        if destination_hash != source_hash:
            raise ReportBundleError(
                "Refusing divergent artifact destination: "
                f"source={source} ({source_hash}), destination={destination} ({destination_hash})"
            )
        return MaterializedArtifact(
            path=str(destination),
            source=str(source),
            sha256=source_hash,
            size_bytes=destination.stat().st_size,
            method="existing",
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    method: Literal["hardlink", "copy"]
    try:
        temporary.unlink()
        try:
            os.link(source, temporary)
            method = "hardlink"
        except OSError:
            shutil.copy2(source, temporary)
            method = "copy"
        if (
            sha256_file(temporary) != source_hash
            or temporary.stat().st_size != source.stat().st_size
        ):
            raise ReportBundleError(f"Materialized artifact failed verification: {temporary}")
        # The destination was checked above.  Publishing a temporary in the
        # same directory makes the final rename atomic on supported filesystems.
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    if sha256_file(destination) != source_hash:
        raise ReportBundleError(f"Published artifact failed verification: {destination}")
    return MaterializedArtifact(
        path=str(destination),
        source=str(source),
        sha256=source_hash,
        size_bytes=destination.stat().st_size,
        method=method,
    )


def materialize_validation_artifacts(
    workspace_reports: Path, experiment_dir: Path
) -> tuple[MaterializedArtifact, ...]:
    """Materialize required training and optional official-validation reports."""

    workspace_reports = workspace_reports.resolve()
    experiment_dir = experiment_dir.resolve()
    training_sources = tuple(workspace_reports / name for name in TRAINING_VALIDATION_FILES)
    missing_training = [str(path) for path in training_sources if not path.is_file()]
    if missing_training:
        raise ReportBundleError(
            f"Required global training-validation artifacts are missing: {missing_training}"
        )
    _validate_validation_pair(*training_sources, dataset_kind="training")
    materialized = [
        _materialize_verified(source, experiment_dir / source.name) for source in training_sources
    ]
    _validate_validation_pair(
        experiment_dir / TRAINING_VALIDATION_FILES[0],
        experiment_dir / TRAINING_VALIDATION_FILES[1],
        dataset_kind="training",
    )

    official_sources = tuple(workspace_reports / name for name in OFFICIAL_VALIDATION_FILES)
    official_presence = [path.is_file() for path in official_sources]
    if any(official_presence) and not all(official_presence):
        raise ReportBundleError(
            "Global official-validation provenance is partial; expected both JSON and CSV: "
            f"{[str(path) for path in official_sources]}"
        )
    local_official = tuple(experiment_dir / name for name in OFFICIAL_VALIDATION_FILES)
    if not any(official_presence) and any(path.exists() for path in local_official):
        raise ReportBundleError(
            "Experiment-local official-validation artifacts cannot be verified because the "
            "workspace-wide source pair is absent"
        )
    if all(official_presence):
        _validate_validation_pair(*official_sources, dataset_kind="validation")
        materialized.extend(
            _materialize_verified(source, experiment_dir / source.name)
            for source in official_sources
        )
        _validate_validation_pair(
            experiment_dir / OFFICIAL_VALIDATION_FILES[0],
            experiment_dir / OFFICIAL_VALIDATION_FILES[1],
            dataset_kind="validation",
        )
    return tuple(materialized)


def materialize_official_aliases(experiment_dir: Path) -> tuple[MaterializedArtifact, ...]:
    """Create compatibility aliases only for a successful canonical official run."""

    experiment_dir = experiment_dir.resolve()
    status_path = experiment_dir / "official_brats_metrics_status.json"
    alias_paths = tuple(experiment_dir / name for name in OFFICIAL_ALIASES.values())
    if not status_path.is_file():
        if any(path.exists() for path in alias_paths):
            raise ReportBundleError(
                "Official compatibility aliases exist without an evaluator status artifact"
            )
        return ()
    status = _load_json_object(status_path, description="official evaluator status")
    if status.get("available") is not True:
        if any(path.exists() for path in alias_paths):
            raise ReportBundleError(
                "Official compatibility aliases exist although the evaluator is unavailable"
            )
        return ()
    canonical_paths = tuple(experiment_dir / name for name in OFFICIAL_ALIASES)
    missing = [
        str(path) for path in canonical_paths if not path.is_file() or path.stat().st_size == 0
    ]
    if missing:
        raise ReportBundleError(
            "Official status reports success, but canonical lesion-wise artifacts are missing: "
            f"{missing}"
        )
    aliases = []
    for canonical_name, alias_name in OFFICIAL_ALIASES.items():
        aliases.append(
            _materialize_verified(experiment_dir / canonical_name, experiment_dir / alias_name)
        )
    return tuple(aliases)


def _log_sort_key(path: Path, experiment_dir: Path) -> tuple[int, str]:
    relative = path.relative_to(experiment_dir).as_posix()
    return (_KNOWN_LOG_ORDER.get(path.name, 100), relative.casefold())


def create_pipeline_log(experiment_dir: Path) -> tuple[Path, tuple[Mapping[str, Any], ...]]:
    """Atomically concatenate project stage logs with hash-labelled boundaries."""

    experiment_dir = experiment_dir.resolve()
    log_directory = experiment_dir / "logs"
    sources = list(log_directory.glob("*.log")) if log_directory.is_dir() else []
    sources.sort(key=lambda path: _log_sort_key(path, experiment_dir))
    official_log = experiment_dir / "official_brats_evaluator.log"
    if official_log.is_file():
        sources.append(official_log)
    if not sources:
        raise ReportBundleError(f"No stage logs were found below {experiment_dir}")
    empty = [str(path) for path in sources if path.stat().st_size == 0]
    if empty:
        raise ReportBundleError(f"Stage logs must be non-empty: {empty}")

    destination = experiment_dir / "pipeline.log"
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_records: list[Mapping[str, Any]] = []
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".pipeline.log.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            for source in sources:
                relative = source.relative_to(experiment_dir).as_posix()
                content = source.read_bytes()
                digest = hashlib.sha256(content).hexdigest()
                size = len(content)
                source_records.append({"path": relative, "sha256": digest, "size_bytes": size})
                output.write(
                    (
                        "=" * 80
                        + "\nBEGIN SOURCE LOG\n"
                        + f"path: {relative}\nsha256: {digest}\nsize_bytes: {size}\n"
                        + "-" * 80
                        + "\n"
                    ).encode("utf-8")
                )
                output.write(content)
                if not content.endswith((b"\n", b"\r")):
                    output.write(b"\n")
                output.write(
                    (
                        "-" * 80
                        + "\nEND SOURCE LOG\n"
                        + f"path: {relative}\nsha256: {digest}\n"
                        + "=" * 80
                        + "\n"
                    ).encode("utf-8")
                )
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination, tuple(source_records)


def _artifact_key(relative_path: str) -> str:
    key = _ARTIFACT_KEY_RE.sub("_", relative_path).strip("_").lower()
    return f"report_{key}"


def _record_final_artifacts(
    experiment_dir: Path,
    *,
    pipeline_sources: Sequence[Mapping[str, Any]],
    official_alias_warning: str | None,
    verified_aliases: Sequence[MaterializedArtifact],
) -> int:
    manifest_path = experiment_dir / "experiment.json"
    manifest = _load_json_object(manifest_path, description="experiment manifest")
    inventory: dict[str, dict[str, Any]] = {}
    raw_friendly = manifest.get("artifacts", {})
    if not isinstance(raw_friendly, dict):
        raise ReportBundleError("experiment.json artifacts must be a JSON object")
    friendly = dict(raw_friendly)
    verified_alias_paths = {Path(item.path).resolve() for item in verified_aliases}
    for alias_name in OFFICIAL_ALIASES.values():
        friendly.pop(_artifact_key(alias_name), None)
        friendly_key = _FRIENDLY_ARTIFACT_KEYS.get(alias_name)
        if friendly_key is not None:
            friendly.pop(friendly_key, None)
    for path in sorted(experiment_dir.rglob("*"), key=lambda item: item.as_posix().casefold()):
        if not path.is_file() or path == manifest_path or path.name.endswith(".tmp"):
            continue
        relative = path.relative_to(experiment_dir).as_posix()
        resolved = path.resolve()
        if relative in OFFICIAL_ALIASES.values() and resolved not in verified_alias_paths:
            continue
        inventory[relative] = {
            "path": str(resolved),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        friendly[_artifact_key(relative)] = str(resolved)
        friendly_key = _FRIENDLY_ARTIFACT_KEYS.get(relative)
        if friendly_key is not None:
            friendly[friendly_key] = str(resolved)
    required_final = (
        "summary.md",
        "weekly_discussion.md",
        "metrics_summary.csv",
        "pipeline.log",
        *TRAINING_VALIDATION_FILES,
    )
    missing = [name for name in required_final if name not in inventory]
    if missing:
        raise ReportBundleError(f"Required final report artifacts are missing: {missing}")
    manifest["artifacts"] = friendly
    manifest["final_artifacts"] = inventory
    manifest["report_bundle"] = {
        "pipeline_sources": list(pipeline_sources),
        "official_alias_warning": official_alias_warning,
        "artifact_count": len(inventory),
    }
    manifest["metrics_files"] = sorted(
        {
            str(value)
            for key, value in friendly.items()
            if isinstance(value, str) and ("metric" in key or key == "official_status")
        }
    )
    write_json_atomic(manifest_path, manifest)
    return len(inventory)


def complete_report_bundle(
    *,
    workspace_reports: str | Path,
    experiment_dir: str | Path,
    phase: Literal["prepare", "finalize"],
) -> ReportBundleResult:
    """Prepare report inputs and optionally finalize their manifest inventory."""

    reports = Path(workspace_reports).resolve()
    destination = Path(experiment_dir).resolve()
    if not destination.is_dir():
        raise ReportBundleError(f"Experiment report directory does not exist: {destination}")
    _load_json_object(destination / "experiment.json", description="experiment manifest")
    validations = materialize_validation_artifacts(reports, destination)
    aliases: tuple[MaterializedArtifact, ...] = ()
    alias_warning: str | None = None
    try:
        aliases = materialize_official_aliases(destination)
    except (OSError, ReportBundleError) as exc:
        # Compatibility aliases are intentionally optional.  Canonical official
        # artifacts remain authoritative and are used directly by reporting.
        alias_warning = f"Official compatibility aliases were not completed: {exc}"
    pipeline_log, pipeline_sources = create_pipeline_log(destination)
    recorded_count = None
    if phase == "finalize":
        recorded_count = _record_final_artifacts(
            destination,
            pipeline_sources=pipeline_sources,
            official_alias_warning=alias_warning,
            verified_aliases=aliases,
        )
    return ReportBundleResult(
        phase=phase,
        pipeline_log=str(pipeline_log),
        pipeline_sources=pipeline_sources,
        validation_artifacts=validations,
        official_aliases=aliases,
        official_alias_warning=alias_warning,
        recorded_artifact_count=recorded_count,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Complete an experiment-local report bundle.")
    parser.add_argument("--workspace-reports", type=Path, required=True)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--phase", choices=("prepare", "finalize"), required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = complete_report_bundle(
        workspace_reports=args.workspace_reports,
        experiment_dir=args.experiment_dir,
        phase=args.phase,
    )
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
