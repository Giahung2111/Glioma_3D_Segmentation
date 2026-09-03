"""Backend-neutral, original-space probability artifacts for BraTS ensembles.

The artifact written here is deliberately stricter than a generic ``.npz``.
It contains canonical ET/TC/WT probabilities together with the native channel
declaration, the conversion that produced them, and the identity and geometry
of the exact reference NIfTI.  This prevents arrays from different models from
being combined merely because their NumPy shapes happen to match.
"""

from __future__ import annotations

import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
from numpy.typing import ArrayLike, NDArray

from glioma_seg.utils.hashing import sha256_file

CANONICAL_REGION_ORDER: tuple[str, ...] = ("ET", "TC", "WT")
MEDNEXT_MULTICLASS_ORDER: tuple[str, ...] = ("background", "NCR", "ED", "ET")
SEGRESNET_REGION_ORDER: tuple[str, ...] = ("TC", "WT", "ET")

MEDNEXT_MULTICLASS_CONVERSION = (
    "ET=p(ET);TC=p(NCR)+p(ET);WT=p(NCR)+p(ED)+p(ET)"
)
SEGRESNET_REGION_CONVERSION = "reorder:[TC,WT,ET]->[ET,TC,WT];indices=[2,0,1]"

SCHEMA_NAME = "glioma_canonical_probabilities_v1"
_SCHEMA_VERSION = 1
_EXPECTED_KEYS = frozenset(
    {
        "schema",
        "schema_version",
        "case_id",
        "probabilities",
        "native_channel_order",
        "canonical_channel_order",
        "conversion",
        "spatial_shape",
        "affine",
        "spacing_mm",
        "reference_nifti_name",
        "reference_sha256",
    }
)
_SAFE_CASE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class CanonicalProbabilityArtifact:
    """A verified canonical probability volume and its spatial provenance."""

    path: Path
    case_id: str
    probabilities: NDArray[np.float32]
    native_channel_order: tuple[str, ...]
    canonical_channel_order: tuple[str, ...]
    conversion: str
    spatial_shape: tuple[int, int, int]
    affine: NDArray[np.float64]
    spacing_mm: tuple[float, float, float]
    reference_nifti: Path
    reference_sha256: str


@dataclass(frozen=True, slots=True)
class _ReferenceGeometry:
    path: Path
    case_id: str
    shape: tuple[int, int, int]
    affine: NDArray[np.float64]
    spacing_mm: tuple[float, float, float]
    sha256: str


def _case_id_from_nifti(path: Path) -> str:
    name = path.name
    if name.endswith(".nii.gz"):
        return name.removesuffix(".nii.gz")
    if name.endswith(".nii"):
        return name.removesuffix(".nii")
    raise ValueError(f"Reference image must be a .nii or .nii.gz file: {path}")


def _validate_case_id(case_id: str) -> str:
    normalized = str(case_id).strip()
    if not normalized or _SAFE_CASE_ID.fullmatch(normalized) is None or normalized in {".", ".."}:
        raise ValueError(
            "case_id must contain only letters, digits, dot, underscore, and hyphen, "
            "and must not contain a path"
        )
    return normalized


def _reference_geometry(reference_nifti: str | Path) -> _ReferenceGeometry:
    path = Path(reference_nifti).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Reference NIfTI does not exist: {path}")
    case_id = _validate_case_id(_case_id_from_nifti(path))
    try:
        image: Any = nib.load(str(path))
    except Exception as exc:
        raise ValueError(f"Unable to read reference NIfTI {path}: {exc}") from exc
    shape = tuple(int(value) for value in image.shape)
    if len(shape) != 3 or any(value <= 0 for value in shape):
        raise ValueError(f"Reference NIfTI must be a non-empty 3D image, got {shape}: {path}")
    affine = np.asarray(image.affine, dtype=np.float64)
    if affine.shape != (4, 4) or not np.all(np.isfinite(affine)):
        raise ValueError(f"Reference NIfTI affine must be a finite 4x4 matrix: {path}")
    zooms = tuple(float(value) for value in image.header.get_zooms()[:3])
    if len(zooms) != 3 or not np.all(np.isfinite(zooms)) or any(value <= 0 for value in zooms):
        raise ValueError(f"Reference NIfTI spacing must contain three positive values: {path}")
    return _ReferenceGeometry(
        path=path,
        case_id=case_id,
        shape=(shape[0], shape[1], shape[2]),
        affine=affine,
        spacing_mm=(zooms[0], zooms[1], zooms[2]),
        sha256=sha256_file(path),
    )


def _validate_channel_order(value: tuple[str, ...] | list[str], *, field: str) -> tuple[str, ...]:
    order = tuple(str(item).strip() for item in value)
    if not order or any(not item for item in order) or len(set(order)) != len(order):
        raise ValueError(f"{field} must contain unique, non-empty channel names")
    return order


def _validate_probability_array(
    probabilities: ArrayLike,
    *,
    expected_channels: int,
    field: str,
) -> NDArray[np.float32]:
    raw = np.asarray(probabilities)
    if not np.issubdtype(raw.dtype, np.floating):
        raise TypeError(f"{field} must have a floating dtype, got {raw.dtype}")
    if raw.ndim != 4 or raw.shape[0] != expected_channels:
        raise ValueError(
            f"{field} must have shape ({expected_channels}, X, Y, Z), got {raw.shape}"
        )
    if any(int(value) <= 0 for value in raw.shape[1:]):
        raise ValueError(f"{field} spatial dimensions must be positive, got {raw.shape}")
    if not np.all(np.isfinite(raw)):
        raise ValueError(f"{field} contains NaN or infinite values")
    if np.any(raw < 0) or np.any(raw > 1):
        raise ValueError(f"{field} values must lie in [0, 1]")
    return np.asarray(raw, dtype=np.float32)


def mednext_multiclass_to_canonical(
    probabilities: ArrayLike,
    *,
    native_channel_order: tuple[str, ...] = MEDNEXT_MULTICLASS_ORDER,
    simplex_atol: float = 1e-5,
) -> NDArray[np.float32]:
    """Convert softmax ``background,NCR,ED,ET`` to overlapping ET,TC,WT.

    A multiclass MedNeXt output is accepted only when every voxel is a valid
    probability simplex.  Logits and independent sigmoid channels therefore
    fail closed instead of being silently treated as multiclass probabilities.
    """

    order = _validate_channel_order(native_channel_order, field="native_channel_order")
    if order != MEDNEXT_MULTICLASS_ORDER:
        raise ValueError(
            f"MedNeXt native channel order must be {MEDNEXT_MULTICLASS_ORDER}, got {order}"
        )
    if not np.isfinite(simplex_atol) or simplex_atol <= 0:
        raise ValueError("simplex_atol must be finite and positive")
    native = _validate_probability_array(
        probabilities,
        expected_channels=len(MEDNEXT_MULTICLASS_ORDER),
        field="MedNeXt probabilities",
    )
    sums = np.sum(native, axis=0, dtype=np.float64)
    if not np.allclose(sums, 1.0, rtol=0.0, atol=simplex_atol):
        maximum_error = float(np.max(np.abs(sums - 1.0)))
        raise ValueError(
            "MedNeXt multiclass probabilities must sum to 1 at every voxel; "
            f"maximum absolute error is {maximum_error:.6g}"
        )
    ncr = np.asarray(native[1], dtype=np.float64)
    edema = np.asarray(native[2], dtype=np.float64)
    enhancing = np.asarray(native[3], dtype=np.float64)
    canonical = np.stack(
        (enhancing, ncr + enhancing, ncr + edema + enhancing), axis=0
    )
    if np.any(canonical < -simplex_atol) or np.any(canonical > 1 + simplex_atol):
        raise ValueError("Converted MedNeXt region probabilities fall outside [0, 1]")
    return np.asarray(np.clip(canonical, 0.0, 1.0), dtype=np.float32)


def segresnet_regions_to_canonical(
    probabilities: ArrayLike,
    *,
    native_channel_order: tuple[str, ...] = SEGRESNET_REGION_ORDER,
) -> NDArray[np.float32]:
    """Reorder SegResNet sigmoid regions ``TC,WT,ET`` to ``ET,TC,WT``."""

    order = _validate_channel_order(native_channel_order, field="native_channel_order")
    if order != SEGRESNET_REGION_ORDER:
        raise ValueError(
            f"SegResNet native channel order must be {SEGRESNET_REGION_ORDER}, got {order}"
        )
    native = _validate_probability_array(
        probabilities,
        expected_channels=len(SEGRESNET_REGION_ORDER),
        field="SegResNet probabilities",
    )
    return np.asarray(native[[2, 0, 1]], dtype=np.float32)


def _string_scalar(value: NDArray[Any], *, field: str) -> str:
    if value.ndim != 0 or value.dtype.kind not in {"U", "S"}:
        raise ValueError(f"{field} must be a scalar non-object string")
    result = str(value.item()).strip()
    if not result:
        raise ValueError(f"{field} must not be empty")
    return result


def _string_vector(value: NDArray[Any], *, field: str) -> tuple[str, ...]:
    if value.ndim != 1 or value.dtype.kind not in {"U", "S"}:
        raise ValueError(f"{field} must be a one-dimensional non-object string array")
    return _validate_channel_order([str(item) for item in value.tolist()], field=field)


def _artifact_arrays(
    *,
    case_id: str,
    probabilities: ArrayLike,
    native_channel_order: tuple[str, ...],
    conversion: str,
    reference: _ReferenceGeometry,
) -> dict[str, Any]:
    validated_case_id = _validate_case_id(case_id)
    if validated_case_id != reference.case_id:
        raise ValueError(
            f"case_id {validated_case_id!r} does not match reference NIfTI {reference.case_id!r}"
        )
    native_order = _validate_channel_order(native_channel_order, field="native_channel_order")
    normalized_conversion = str(conversion).strip()
    if not normalized_conversion:
        raise ValueError("conversion must explicitly describe the native-to-canonical mapping")
    canonical = _validate_probability_array(
        probabilities,
        expected_channels=len(CANONICAL_REGION_ORDER),
        field="canonical probabilities",
    )
    if tuple(int(value) for value in canonical.shape[1:]) != reference.shape:
        raise ValueError(
            "Canonical probability shape does not match reference NIfTI: "
            f"{canonical.shape[1:]} vs {reference.shape}"
        )
    return {
        "schema": np.asarray(SCHEMA_NAME),
        "schema_version": np.asarray(_SCHEMA_VERSION, dtype=np.int64),
        "case_id": np.asarray(validated_case_id),
        "probabilities": canonical,
        "native_channel_order": np.asarray(native_order),
        "canonical_channel_order": np.asarray(CANONICAL_REGION_ORDER),
        "conversion": np.asarray(normalized_conversion),
        "spatial_shape": np.asarray(reference.shape, dtype=np.int64),
        "affine": np.asarray(reference.affine, dtype=np.float64),
        "spacing_mm": np.asarray(reference.spacing_mm, dtype=np.float64),
        "reference_nifti_name": np.asarray(reference.path.name),
        "reference_sha256": np.asarray(reference.sha256),
    }


def write_canonical_probability_npz(
    path: str | Path,
    *,
    case_id: str,
    probabilities: ArrayLike,
    native_channel_order: tuple[str, ...],
    conversion: str,
    reference_nifti: str | Path,
    overwrite: bool = False,
) -> Path:
    """Atomically write a canonical original-space probability artifact."""

    destination = Path(path).resolve()
    if destination.suffix.lower() != ".npz":
        raise ValueError(f"Canonical probability artifact must use .npz: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite canonical probabilities: {destination}")
    reference = _reference_geometry(reference_nifti)
    arrays = _artifact_arrays(
        case_id=case_id,
        probabilities=probabilities,
        native_channel_order=native_channel_order,
        conversion=conversion,
        reference=reference,
    )

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        if destination.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite canonical probabilities: {destination}")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def load_canonical_probability_npz(
    path: str | Path,
    *,
    reference_nifti: str | Path,
    expected_case_id: str | None = None,
    expected_native_channel_order: tuple[str, ...] | None = None,
    expected_conversion: str | None = None,
) -> CanonicalProbabilityArtifact:
    """Safely load and fully validate one canonical probability artifact."""

    source = Path(path).resolve()
    if source.suffix.lower() != ".npz" or not source.is_file():
        raise FileNotFoundError(f"Canonical probability .npz does not exist: {source}")
    reference = _reference_geometry(reference_nifti)
    try:
        with np.load(source, allow_pickle=False) as archive:
            keys = frozenset(archive.files)
            if keys != _EXPECTED_KEYS:
                missing = sorted(_EXPECTED_KEYS - keys)
                extra = sorted(keys - _EXPECTED_KEYS)
                raise ValueError(
                    f"Canonical probability keys differ from {SCHEMA_NAME}: "
                    f"missing={missing}, extra={extra}"
                )
            schema = _string_scalar(np.asarray(archive["schema"]), field="schema")
            version = np.asarray(archive["schema_version"])
            case_id = _validate_case_id(
                _string_scalar(np.asarray(archive["case_id"]), field="case_id")
            )
            native_order = _string_vector(
                np.asarray(archive["native_channel_order"]), field="native_channel_order"
            )
            canonical_order = _string_vector(
                np.asarray(archive["canonical_channel_order"]),
                field="canonical_channel_order",
            )
            conversion = _string_scalar(
                np.asarray(archive["conversion"]), field="conversion"
            )
            probabilities_raw = np.asarray(archive["probabilities"])
            spatial_shape_raw = np.asarray(archive["spatial_shape"])
            affine = np.asarray(archive["affine"])
            spacing_raw = np.asarray(archive["spacing_mm"])
            reference_name = _string_scalar(
                np.asarray(archive["reference_nifti_name"]),
                field="reference_nifti_name",
            )
            reference_hash = _string_scalar(
                np.asarray(archive["reference_sha256"]), field="reference_sha256"
            ).lower()
    except (OSError, ValueError, TypeError) as exc:
        raise ValueError(
            f"Unable to load safe canonical probabilities from {source}: {exc}"
        ) from exc

    if schema != SCHEMA_NAME:
        raise ValueError(f"Unsupported canonical probability schema: {schema!r}")
    if version.ndim != 0 or version.dtype != np.dtype(np.int64) or int(version.item()) != 1:
        raise ValueError(f"Unsupported canonical probability schema_version: {version!r}")
    if canonical_order != CANONICAL_REGION_ORDER:
        raise ValueError(
            f"canonical_channel_order must be {CANONICAL_REGION_ORDER}, got {canonical_order}"
        )
    if expected_case_id is not None and case_id != _validate_case_id(expected_case_id):
        raise ValueError(f"Expected case_id {expected_case_id!r}, found {case_id!r}")
    if case_id != reference.case_id:
        raise ValueError(
            f"Artifact case_id {case_id!r} does not match reference NIfTI {reference.case_id!r}"
        )
    if expected_native_channel_order is not None:
        expected_order = _validate_channel_order(
            expected_native_channel_order, field="expected_native_channel_order"
        )
        if native_order != expected_order:
            raise ValueError(
                f"Expected native channel order {expected_order}, found {native_order}"
            )
    if expected_conversion is not None and conversion != str(expected_conversion).strip():
        raise ValueError(
            f"Expected conversion {expected_conversion!r}, found {conversion!r}"
        )
    if probabilities_raw.dtype != np.dtype(np.float32):
        raise TypeError(
            f"Canonical probabilities must be stored as float32, got {probabilities_raw.dtype}"
        )
    probabilities = _validate_probability_array(
        probabilities_raw,
        expected_channels=len(CANONICAL_REGION_ORDER),
        field="canonical probabilities",
    )
    if (
        spatial_shape_raw.ndim != 1
        or spatial_shape_raw.shape != (3,)
        or spatial_shape_raw.dtype != np.dtype(np.int64)
    ):
        raise ValueError("spatial_shape must be an int64 vector with exactly three entries")
    spatial_shape = tuple(int(value) for value in spatial_shape_raw.tolist())
    if spatial_shape != tuple(int(value) for value in probabilities.shape[1:]):
        raise ValueError("Recorded spatial_shape differs from the probability array")
    if spatial_shape != reference.shape:
        raise ValueError(
            f"Recorded spatial_shape differs from reference: {spatial_shape} vs {reference.shape}"
        )
    if affine.shape != (4, 4) or affine.dtype != np.dtype(np.float64):
        raise ValueError("affine must be a float64 4x4 matrix")
    if not np.all(np.isfinite(affine)) or not np.allclose(
        affine, reference.affine, rtol=0.0, atol=1e-6
    ):
        raise ValueError("Recorded affine differs from the reference NIfTI")
    if spacing_raw.shape != (3,) or spacing_raw.dtype != np.dtype(np.float64):
        raise ValueError("spacing_mm must be a float64 vector with exactly three entries")
    if not np.all(np.isfinite(spacing_raw)) or np.any(spacing_raw <= 0):
        raise ValueError("spacing_mm must contain finite positive values")
    if not np.allclose(
        spacing_raw, np.asarray(reference.spacing_mm), rtol=0.0, atol=1e-6
    ):
        raise ValueError("Recorded spacing_mm differs from the reference NIfTI")
    if reference_name != reference.path.name:
        raise ValueError(
            f"Recorded reference filename {reference_name!r} differs from {reference.path.name!r}"
        )
    if _SHA256.fullmatch(reference_hash) is None:
        raise ValueError("reference_sha256 must be a lowercase 64-character SHA-256 digest")
    if reference_hash != reference.sha256:
        raise ValueError("Reference NIfTI SHA-256 differs from the artifact declaration")

    return CanonicalProbabilityArtifact(
        path=source,
        case_id=case_id,
        probabilities=probabilities,
        native_channel_order=native_order,
        canonical_channel_order=canonical_order,
        conversion=conversion,
        spatial_shape=(spatial_shape[0], spatial_shape[1], spatial_shape[2]),
        affine=np.asarray(affine, dtype=np.float64),
        spacing_mm=(float(spacing_raw[0]), float(spacing_raw[1]), float(spacing_raw[2])),
        reference_nifti=reference.path,
        reference_sha256=reference_hash,
    )


def validate_canonical_probability_npz(
    path: str | Path,
    *,
    reference_nifti: str | Path,
    expected_case_id: str | None = None,
    expected_native_channel_order: tuple[str, ...] | None = None,
    expected_conversion: str | None = None,
) -> CanonicalProbabilityArtifact:
    """Validate an artifact and return the same verified record as the safe loader."""

    return load_canonical_probability_npz(
        path,
        reference_nifti=reference_nifti,
        expected_case_id=expected_case_id,
        expected_native_channel_order=expected_native_channel_order,
        expected_conversion=expected_conversion,
    )


__all__ = [
    "CANONICAL_REGION_ORDER",
    "MEDNEXT_MULTICLASS_CONVERSION",
    "MEDNEXT_MULTICLASS_ORDER",
    "SCHEMA_NAME",
    "SEGRESNET_REGION_CONVERSION",
    "SEGRESNET_REGION_ORDER",
    "CanonicalProbabilityArtifact",
    "load_canonical_probability_npz",
    "mednext_multiclass_to_canonical",
    "segresnet_regions_to_canonical",
    "validate_canonical_probability_npz",
    "write_canonical_probability_npz",
]
