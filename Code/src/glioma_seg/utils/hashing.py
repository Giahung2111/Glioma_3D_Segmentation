"""Streaming hashes and immutable file identity records."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Hash a file without loading a 3D volume into memory."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class FileFingerprint:
    path: str
    size_bytes: int
    mtime_ns: int
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> FileFingerprint:
        try:
            path = value["path"]
            size_bytes = value["size_bytes"]
            mtime_ns = value["mtime_ns"]
            digest = value["sha256"]
        except KeyError as exc:
            raise ValueError(f"Fingerprint missing field: {exc.args[0]}") from exc
        if not isinstance(path, str) or not isinstance(digest, str):
            raise ValueError("Fingerprint path and sha256 must be strings")
        if isinstance(size_bytes, bool) or not isinstance(size_bytes, int):
            raise ValueError("Fingerprint size_bytes must be an integer")
        if isinstance(mtime_ns, bool) or not isinstance(mtime_ns, int):
            raise ValueError("Fingerprint mtime_ns must be an integer")
        return cls(path=path, size_bytes=size_bytes, mtime_ns=mtime_ns, sha256=digest)

    def matches(self, *, verify_hash: bool = True) -> bool:
        candidate = Path(self.path)
        try:
            stat = candidate.stat()
        except OSError:
            return False
        if stat.st_size != self.size_bytes or stat.st_mtime_ns != self.mtime_ns:
            return False
        return not verify_hash or sha256_file(candidate) == self.sha256


def fingerprint_file(path: str | Path) -> FileFingerprint:
    candidate = Path(path).resolve()
    stat = candidate.stat()
    if not candidate.is_file():
        raise ValueError(f"Not a file: {candidate}")
    return FileFingerprint(
        path=str(candidate),
        size_bytes=stat.st_size,
        mtime_ns=stat.st_mtime_ns,
        sha256=sha256_file(candidate),
    )
