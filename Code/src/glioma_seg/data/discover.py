"""Discover BraTS source directories by file content instead of folder names."""

from __future__ import annotations

import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .brats2023 import SOURCE_FILE_RE, BraTSCase

DatasetKind = Literal["training", "validation", "mixed"]


class DiscoveryError(RuntimeError):
    """Raised when dataset content cannot be selected unambiguously."""


@dataclass(frozen=True, slots=True)
class DiscoveredDataset:
    root: Path
    kind: DatasetKind
    cases: tuple[BraTSCase, ...]
    non_case_entries: tuple[Path, ...]
    duplicate_case_ids: tuple[str, ...]

    @property
    def case_count(self) -> int:
        return len(self.cases)


@dataclass(frozen=True, slots=True)
class DatasetDiscovery:
    scanned_root: Path
    datasets: tuple[DiscoveredDataset, ...]
    non_dataset_entries: tuple[Path, ...]

    def candidates(self, kind: DatasetKind) -> tuple[DiscoveredDataset, ...]:
        return tuple(dataset for dataset in self.datasets if dataset.kind == kind)

    def require_unique(self, kind: Literal["training", "validation"]) -> DiscoveredDataset:
        candidates = self.candidates(kind)
        if len(candidates) != 1:
            details = ", ".join(f"{item.root} ({item.case_count})" for item in candidates)
            raise DiscoveryError(
                f"Expected exactly one {kind} dataset under {self.scanned_root}; "
                f"found {len(candidates)}: {details or 'none'}"
            )
        return candidates[0]


def _case_directories(root: Path) -> list[BraTSCase]:
    discovered: list[BraTSCase] = []
    for current, directory_names, filenames in os.walk(root, followlinks=False):
        # Reparse points/symlinks must never lead discovery outside the raw root.
        directory_names[:] = [
            name for name in directory_names if not (Path(current) / name).is_symlink()
        ]
        matching: dict[str, dict[str, Path]] = defaultdict(dict)
        for filename in filenames:
            match = SOURCE_FILE_RE.fullmatch(filename)
            if match is None:
                continue
            case_id = match.group("case_id")
            role = match.group("role")
            matching[case_id][role] = Path(current, filename).resolve()
        if not matching:
            continue

        directory = Path(current).resolve()
        all_files = tuple(
            sorted(
                (entry.resolve() for entry in directory.iterdir() if entry.is_file()),
                key=lambda item: item.name,
            )
        )
        for case_id, role_paths in sorted(matching.items()):
            attributed = set(role_paths.values())
            unexpected = tuple(path for path in all_files if path not in attributed)
            discovered.append(
                BraTSCase(
                    case_id=case_id,
                    directory=directory,
                    files=tuple(sorted(role_paths.items())),
                    unexpected_files=unexpected,
                )
            )
    return discovered


def _non_case_entries(dataset_root: Path, cases: tuple[BraTSCase, ...]) -> tuple[Path, ...]:
    case_directories = {case.directory for case in cases}
    entries: list[Path] = []
    for entry in dataset_root.iterdir():
        resolved = entry.resolve()
        if resolved not in case_directories:
            entries.append(resolved)
    return tuple(sorted(entries, key=str))


def discover_brats_datasets(raw_root: str | Path) -> DatasetDiscovery:
    """Scan recursively and group case folders by their immediate parent.

    Folder names such as ``ASNR-MICCAI-...`` are deliberately ignored. A
    directory becomes a case candidate only when it contains at least one
    correctly named BraTS 2023 GLI NIfTI file.
    """

    root = Path(raw_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Raw data root does not exist or is not a directory: {root}")

    grouped: dict[Path, list[BraTSCase]] = defaultdict(list)
    for case in _case_directories(root):
        grouped[case.directory.parent].append(case)

    datasets: list[DiscoveredDataset] = []
    for dataset_root, case_list in sorted(grouped.items(), key=lambda item: str(item[0])):
        cases = tuple(sorted(case_list, key=lambda case: (case.case_id, str(case.directory))))
        segmentation_flags = {case.has_segmentation for case in cases}
        if True in segmentation_flags:
            # A partially broken training set must remain selectable so validation
            # can report each missing segmentation instead of losing the whole set.
            kind: DatasetKind = "training"
        elif segmentation_flags == {False}:
            kind = "validation"
        else:
            kind = "mixed"
        counts = Counter(case.case_id for case in cases)
        duplicates = tuple(sorted(case_id for case_id, count in counts.items() if count > 1))
        datasets.append(
            DiscoveredDataset(
                root=dataset_root,
                kind=kind,
                cases=cases,
                non_case_entries=_non_case_entries(dataset_root, cases),
                duplicate_case_ids=duplicates,
            )
        )

    top_level_expected: set[Path] = set()
    for dataset in datasets:
        for entry in root.iterdir():
            resolved = entry.resolve()
            if dataset.root == resolved or dataset.root.is_relative_to(resolved):
                top_level_expected.add(resolved)
    non_dataset_entries = tuple(
        sorted(
            (
                entry.resolve()
                for entry in root.iterdir()
                if entry.resolve() not in top_level_expected
            ),
            key=str,
        )
    )
    return DatasetDiscovery(
        scanned_root=root,
        datasets=tuple(datasets),
        non_dataset_entries=non_dataset_entries,
    )
