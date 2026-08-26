"""BraTS 2023 GLI naming and medical-label constants."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

BRATS_CASE_ID_PATTERN: Final[str] = r"BraTS-GLI-[0-9]{5}-[0-9]{3}"
CASE_ID_RE: Final[re.Pattern[str]] = re.compile(rf"^{BRATS_CASE_ID_PATTERN}$")
SOURCE_FILE_RE: Final[re.Pattern[str]] = re.compile(
    rf"^(?P<case_id>{BRATS_CASE_ID_PATTERN})-(?P<role>t1n|t1c|t2w|t2f|seg)\.nii\.gz$"
)

MODALITIES: Final[tuple[str, ...]] = ("t1n", "t1c", "t2w", "t2f")
REQUIRED_TRAINING_ROLES: Final[tuple[str, ...]] = (*MODALITIES, "seg")
ALLOWED_LABELS: Final[frozenset[int]] = frozenset({0, 1, 2, 3})
CHANNEL_BY_MODALITY: Final[Mapping[str, str]] = {
    "t1n": "0000",
    "t1c": "0001",
    "t2w": "0002",
    "t2f": "0003",
}


class NamingError(ValueError):
    """Raised for a filename that is not valid BraTS 2023 GLI input."""


def parse_source_filename(filename: str) -> tuple[str, str]:
    match = SOURCE_FILE_RE.fullmatch(filename)
    if match is None:
        raise NamingError(f"Not a BraTS 2023 GLI NIfTI filename: {filename}")
    return match.group("case_id"), match.group("role")


def source_filename(case_id: str, role: str) -> str:
    if CASE_ID_RE.fullmatch(case_id) is None:
        raise NamingError(f"Invalid BraTS 2023 GLI case ID: {case_id}")
    if role not in REQUIRED_TRAINING_ROLES:
        raise NamingError(f"Unknown BraTS role: {role}")
    return f"{case_id}-{role}.nii.gz"


def nnunet_image_name(case_id: str, modality: str) -> str:
    if CASE_ID_RE.fullmatch(case_id) is None:
        raise NamingError(f"Invalid BraTS 2023 GLI case ID: {case_id}")
    try:
        channel = CHANNEL_BY_MODALITY[modality]
    except KeyError as exc:
        raise NamingError(f"Unknown BraTS modality: {modality}") from exc
    return f"{case_id}_{channel}.nii.gz"


def nnunet_label_name(case_id: str) -> str:
    if CASE_ID_RE.fullmatch(case_id) is None:
        raise NamingError(f"Invalid BraTS 2023 GLI case ID: {case_id}")
    return f"{case_id}.nii.gz"


@dataclass(frozen=True, slots=True)
class BraTSCase:
    """Files attributed to one case without assuming its parent directory name."""

    case_id: str
    directory: Path
    files: tuple[tuple[str, Path], ...]
    unexpected_files: tuple[Path, ...] = ()

    def __post_init__(self) -> None:
        if CASE_ID_RE.fullmatch(self.case_id) is None:
            raise NamingError(f"Invalid BraTS 2023 GLI case ID: {self.case_id}")
        roles = [role for role, _ in self.files]
        if len(roles) != len(set(roles)):
            raise NamingError(f"Duplicate file roles for {self.case_id}: {roles}")
        if any(role not in REQUIRED_TRAINING_ROLES for role in roles):
            raise NamingError(f"Unknown file role for {self.case_id}: {roles}")

    @property
    def file_map(self) -> dict[str, Path]:
        return dict(self.files)

    @property
    def has_segmentation(self) -> bool:
        return "seg" in self.file_map

    @property
    def directory_name_matches(self) -> bool:
        return self.directory.name == self.case_id

    def missing_roles(self, *, require_segmentation: bool) -> tuple[str, ...]:
        required = REQUIRED_TRAINING_ROLES if require_segmentation else MODALITIES
        available = self.file_map
        return tuple(role for role in required if role not in available)
