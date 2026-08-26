"""Centralized project paths and containment guards."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def find_project_root(start: str | Path | None = None) -> Path:
    """Find the root by structure, not by assuming a machine-specific name."""

    candidate = Path.cwd() if start is None else Path(start)
    candidate = candidate.resolve()
    if candidate.is_file():
        candidate = candidate.parent
    for directory in (candidate, *candidate.parents):
        if (directory / "Code").is_dir() and (directory / "Datasets").is_dir():
            return directory
    raise FileNotFoundError(
        f"Could not find a project root containing Code and Datasets above {candidate}"
    )


def ensure_within(path: str | Path, parent: str | Path, *, label: str = "path") -> Path:
    """Resolve a path and reject targets outside an explicitly allowed parent."""

    resolved = Path(path).resolve()
    root = Path(parent).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} must remain inside {root}: {resolved}") from exc
    return resolved


@dataclass(frozen=True, slots=True)
class ProjectPaths:
    root: Path
    datasets: Path
    code: Path
    external_nnunet: Path
    workspace: Path
    nnunet_raw: Path
    nnunet_preprocessed: Path
    nnunet_results: Path
    predictions: Path
    telemetry: Path
    reports: Path
    cache: Path

    @classmethod
    def from_root(cls, root: str | Path) -> ProjectPaths:
        resolved = Path(root).resolve()
        workspace = resolved / "Workspace"
        return cls(
            root=resolved,
            datasets=resolved / "Datasets",
            code=resolved / "Code",
            external_nnunet=resolved / "External" / "nnUNet",
            workspace=workspace,
            nnunet_raw=workspace / "nnUNet_raw",
            nnunet_preprocessed=workspace / "nnUNet_preprocessed",
            nnunet_results=workspace / "nnUNet_results",
            predictions=workspace / "predictions",
            telemetry=workspace / "telemetry",
            reports=workspace / "reports",
            cache=workspace / "cache",
        )

    @classmethod
    def discover(cls, start: str | Path | None = None) -> ProjectPaths:
        return cls.from_root(find_project_root(start))

    def create_workspace(self) -> None:
        """Create only generated-data directories; raw data remains untouched."""

        for directory in (
            self.nnunet_raw,
            self.nnunet_preprocessed,
            self.nnunet_results,
            self.predictions,
            self.telemetry,
            self.reports,
            self.cache,
        ):
            ensure_within(directory, self.workspace, label="workspace directory").mkdir(
                parents=True, exist_ok=True
            )
