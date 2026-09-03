"""Create or verify the exact canonical BraTS five-fold split.

The completed nnU-Net baseline used nnU-Net's sorted-case, seed-12345 split.
``Workspace`` is intentionally ignored by Git, so a fresh research host does
not receive ``splits_final.json``.  This module reproduces that *data split*
with the same scikit-learn primitive used by nnU-Net, verifies the known file
hash, and writes it atomically.  It does not implement or modify model code.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from sklearn.model_selection import KFold  # type: ignore[import-untyped]

from glioma_seg.monitoring.timing import write_json_atomic
from glioma_seg.utils.hashing import sha256_file

EXPECTED_CASE_COUNT = 1251
EXPECTED_FOLDS = 5
EXPECTED_SEED = 12345
EXPECTED_VALIDATION_COUNTS = (251, 250, 250, 250, 250)
EXPECTED_SPLIT_SHA256 = "a9b8aaef82974d52aa3652624c4902d0515c73a573f8bb8f24ad7982b943ed7b"


def case_ids_from_labels(labels_dir: Path, *, expected_count: int) -> tuple[str, ...]:
    """Return the exact sorted converted-case inventory."""

    root = labels_dir.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Converted label directory is missing: {root}")
    case_ids = tuple(sorted(path.name.removesuffix(".nii.gz") for path in root.glob("*.nii.gz")))
    if len(case_ids) != expected_count:
        raise ValueError(f"Expected {expected_count} converted labels, found {len(case_ids)}")
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("Converted label inventory contains duplicate case IDs")
    return case_ids


def generate_canonical_splits(case_ids: Sequence[str]) -> list[dict[str, list[str]]]:
    """Use nnU-Net's exact sorted-case KFold parameters."""

    normalized = tuple(sorted(str(case_id) for case_id in case_ids))
    if len(normalized) != EXPECTED_CASE_COUNT or len(set(normalized)) != len(normalized):
        raise ValueError(
            f"Canonical split requires {EXPECTED_CASE_COUNT} unique case IDs, got {len(normalized)}"
        )
    splitter = KFold(n_splits=EXPECTED_FOLDS, shuffle=True, random_state=EXPECTED_SEED)
    result: list[dict[str, list[str]]] = []
    for train_indices, validation_indices in splitter.split(normalized):
        result.append(
            {
                "train": [normalized[int(index)] for index in train_indices],
                "val": [normalized[int(index)] for index in validation_indices],
            }
        )
    counts = tuple(len(fold["val"]) for fold in result)
    if counts != EXPECTED_VALIDATION_COUNTS:
        raise AssertionError(f"Unexpected canonical fold sizes: {counts}")
    return result


def ensure_canonical_split(labels_dir: Path, output: Path) -> dict[str, Any]:
    """Atomically create the canonical split or reject conflicting evidence."""

    case_ids = case_ids_from_labels(labels_dir, expected_count=EXPECTED_CASE_COUNT)
    expected = generate_canonical_splits(case_ids)
    destination = output.resolve()
    created = False
    if destination.exists():
        try:
            actual = json.loads(destination.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Existing split is unreadable: {destination}: {exc}") from exc
        if actual != expected:
            raise ValueError(
                f"Existing split differs from the canonical nnU-Net seed-12345 split: {destination}"
            )
    else:
        write_json_atomic(destination, expected)
        created = True
    digest = sha256_file(destination)
    if digest != EXPECTED_SPLIT_SHA256:
        raise ValueError(
            "Canonical split bytes/hash differ from the completed nnU-Net baseline: "
            f"expected={EXPECTED_SPLIT_SHA256}, actual={digest}"
        )
    return {
        "valid": True,
        "created": created,
        "path": str(destination),
        "sha256": digest,
        "case_count": len(case_ids),
        "fold_count": len(expected),
        "seed": EXPECTED_SEED,
        "validation_case_counts": list(EXPECTED_VALIDATION_COUNTS),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = ensure_canonical_split(args.labels_dir, args.output)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "EXPECTED_CASE_COUNT",
    "EXPECTED_FOLDS",
    "EXPECTED_SEED",
    "EXPECTED_SPLIT_SHA256",
    "EXPECTED_VALIDATION_COUNTS",
    "case_ids_from_labels",
    "ensure_canonical_split",
    "generate_canonical_splits",
]
