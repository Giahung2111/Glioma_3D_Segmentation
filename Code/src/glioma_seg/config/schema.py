"""Typed schemas and invariants for project configuration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """Raised when a configuration violates a project invariant."""


def _mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{field} must be a mapping")
    return value


def _integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{field} must be an integer")
    return value


@dataclass(frozen=True, slots=True)
class ProjectPathConfig:
    """Resolved paths used by project code, never by upstream nnU-Net source."""

    project_root: Path
    datasets_root: Path
    workspace_root: Path
    external_nnunet_root: Path

    @classmethod
    def from_mapping(cls, value: object, *, relative_to: Path) -> ProjectPathConfig:
        data = _mapping(value, "paths")

        def resolve(name: str) -> Path:
            raw = data.get(name)
            if not isinstance(raw, str) or not raw.strip():
                raise ConfigError(f"paths.{name} must be a non-empty string")
            path = Path(raw).expanduser()
            if not path.is_absolute():
                path = relative_to / path
            return path.resolve()

        return cls(
            project_root=resolve("project_root"),
            datasets_root=resolve("datasets_root"),
            workspace_root=resolve("workspace_root"),
            external_nnunet_root=resolve("external_nnunet_root"),
        )


@dataclass(frozen=True, slots=True)
class ValidationConfig:
    workers: int = 2
    progress_every_cases: int = 10
    affine_atol: float = 1e-5
    affine_rtol: float = 1e-5
    spacing_atol: float = 1e-6
    require_3d: bool = True
    scan_voxel_values: bool = True

    @classmethod
    def from_mapping(cls, value: object) -> ValidationConfig:
        data = _mapping(value, "validation")
        try:
            result = cls(
                workers=int(data.get("workers", 2)),
                progress_every_cases=int(data.get("progress_every_cases", 10)),
                affine_atol=float(data.get("affine_atol", 1e-5)),
                affine_rtol=float(data.get("affine_rtol", 1e-5)),
                spacing_atol=float(data.get("spacing_atol", 1e-6)),
                require_3d=bool(data.get("require_3d", True)),
                scan_voxel_values=bool(data.get("scan_voxel_values", True)),
            )
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"Invalid validation tolerance: {exc}") from exc
        if min(result.affine_atol, result.affine_rtol, result.spacing_atol) < 0:
            raise ConfigError("Validation tolerances must be non-negative")
        if not 1 <= result.workers <= 8:
            raise ConfigError("validation.workers must be between 1 and 8")
        if result.progress_every_cases <= 0:
            raise ConfigError("validation.progress_every_cases must be positive")
        return result


@dataclass(frozen=True, slots=True)
class BaseConfig:
    schema_version: int
    paths: ProjectPathConfig
    validation: ValidationConfig

    @classmethod
    def from_mapping(cls, value: object, *, config_directory: Path) -> BaseConfig:
        data = _mapping(value, "base config")
        schema_version = _integer(data.get("schema_version"), "schema_version")
        if schema_version != 1:
            raise ConfigError(f"Unsupported base schema_version: {schema_version}")
        return cls(
            schema_version=schema_version,
            paths=ProjectPathConfig.from_mapping(data.get("paths"), relative_to=config_directory),
            validation=ValidationConfig.from_mapping(data.get("validation", {})),
        )


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    """BraTS-to-nnU-Net mapping with ordering preserved."""

    schema_version: int
    dataset_id: int
    dataset_name: str
    description: str
    expected_training_cases: int
    expected_validation_cases: int
    case_id_pattern: str
    file_ending: str
    modalities: tuple[tuple[str, str], ...]
    channel_names: tuple[tuple[str, str], ...]
    raw_labels: tuple[tuple[str, int], ...]
    regions: tuple[tuple[str, int | tuple[int, ...]], ...]
    regions_class_order: tuple[int, ...]

    @classmethod
    def from_mapping(cls, value: object) -> DatasetConfig:
        data = _mapping(value, "dataset config")

        def nonempty_string(name: str) -> str:
            raw = data.get(name)
            if not isinstance(raw, str) or not raw.strip():
                raise ConfigError(f"{name} must be a non-empty string")
            return raw

        def string_pairs(name: str) -> tuple[tuple[str, str], ...]:
            raw = _mapping(data.get(name), name)
            if not all(isinstance(key, str) and isinstance(item, str) for key, item in raw.items()):
                raise ConfigError(f"{name} keys and values must be strings")
            return tuple(raw.items())

        labels_raw = _mapping(data.get("raw_labels"), "raw_labels")
        raw_labels: list[tuple[str, int]] = []
        for name, label in labels_raw.items():
            if not isinstance(name, str):
                raise ConfigError("raw_labels keys must be strings")
            raw_labels.append((name, _integer(label, f"raw_labels.{name}")))

        regions_raw = _mapping(data.get("regions"), "regions")
        regions: list[tuple[str, int | tuple[int, ...]]] = []
        for name, labels in regions_raw.items():
            if not isinstance(name, str):
                raise ConfigError("regions keys must be strings")
            if isinstance(labels, list):
                parsed: int | tuple[int, ...] = tuple(
                    _integer(label, f"regions.{name}") for label in labels
                )
            else:
                parsed = _integer(labels, f"regions.{name}")
            regions.append((name, parsed))

        class_order_raw = data.get("regions_class_order")
        if not isinstance(class_order_raw, list):
            raise ConfigError("regions_class_order must be a list")
        class_order = tuple(_integer(item, "regions_class_order") for item in class_order_raw)

        result = cls(
            schema_version=_integer(data.get("schema_version"), "schema_version"),
            dataset_id=_integer(data.get("dataset_id"), "dataset_id"),
            dataset_name=nonempty_string("dataset_name"),
            description=nonempty_string("description"),
            expected_training_cases=_integer(
                data.get("expected_training_cases"), "expected_training_cases"
            ),
            expected_validation_cases=_integer(
                data.get("expected_validation_cases"), "expected_validation_cases"
            ),
            case_id_pattern=nonempty_string("case_id_pattern"),
            file_ending=nonempty_string("file_ending"),
            modalities=string_pairs("modalities"),
            channel_names=string_pairs("channel_names"),
            raw_labels=tuple(raw_labels),
            regions=tuple(regions),
            regions_class_order=class_order,
        )
        result.validate_brats2023_invariants()
        return result

    def validate_brats2023_invariants(self) -> None:
        if self.schema_version != 1:
            raise ConfigError(f"Unsupported dataset schema_version: {self.schema_version}")
        if self.dataset_id != 501 or self.dataset_name != "Dataset501_BraTS2023GLI":
            raise ConfigError("The first baseline must use Dataset501_BraTS2023GLI")
        if self.expected_training_cases <= 0 or self.expected_validation_cases < 0:
            raise ConfigError("Expected case counts must be non-negative")
        if self.file_ending != ".nii.gz":
            raise ConfigError("BraTS 2023 conversion requires .nii.gz files")
        if self.modalities != (
            ("0000", "t1n"),
            ("0001", "t1c"),
            ("0002", "t2w"),
            ("0003", "t2f"),
        ):
            raise ConfigError("Modality order must be t1n, t1c, t2w, t2f")
        if dict(self.raw_labels) != {"background": 0, "NCR": 1, "ED": 2, "ET": 3}:
            raise ConfigError("BraTS 2023 raw labels must be 0/1/2/3")
        if self.regions != (
            ("background", 0),
            ("whole_tumor", (1, 2, 3)),
            ("tumor_core", (1, 3)),
            ("enhancing_tumor", 3),
        ):
            raise ConfigError("Region order/content must be background, WT, TC, ET")
        if self.regions_class_order != (2, 1, 3):
            raise ConfigError("regions_class_order must be [2, 1, 3]")

    @property
    def nnunet_labels(self) -> dict[str, int | list[int]]:
        """Return a fresh insertion-ordered mapping suitable for dataset.json."""

        return {
            name: list(labels) if isinstance(labels, tuple) else labels
            for name, labels in self.regions
        }
