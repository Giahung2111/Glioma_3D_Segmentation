"""Project-owned Windows path adapter for the upstream Linux-oriented v1 code.

No upstream file is edited.  The compatibility shim only replaces basename
helpers whose original implementations split paths on the literal ``/``.
Model architecture, preprocessing mathematics, augmentation, loss, optimizer,
and inference remain the official implementation.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from collections.abc import Sequence
from typing import Any


def _basename(value: str) -> str:
    return str(value).replace("\\", "/").rsplit("/", 1)[-1]


def apply_windows_path_compatibility() -> dict[str, Any]:
    """Patch path-only helpers in memory and report exactly what changed."""

    nnunet_mednext: Any = importlib.import_module("nnunet_mednext")
    sys.modules.setdefault("nnunet", nnunet_mednext)

    planning_utils: Any = importlib.import_module(
        "nnunet_mednext.experiment_planning.utils"
    )
    cropping: Any = importlib.import_module("nnunet_mednext.preprocessing.cropping")
    preprocessing: Any = importlib.import_module(
        "nnunet_mednext.preprocessing.preprocessing"
    )
    dataset_analyzer: Any = importlib.import_module(
        "nnunet_mednext.experiment_planning.DatasetAnalyzer"
    )

    original_create_lists = planning_utils.create_lists_from_splitted_dataset

    def case_identifier(case: Sequence[str]) -> str:
        return _basename(str(case[0])).removesuffix(".nii.gz")[:-5]

    def case_identifier_from_npz(path: str) -> str:
        return _basename(str(path)).removesuffix(".npz")

    def create_lists(base_folder: str) -> tuple[list[list[str]], dict[int, str]]:
        cases, modalities = original_create_lists(base_folder)
        return (
            [[str(value).replace("\\", "/") for value in case] for case in cases],
            modalities,
        )

    cropping.get_case_identifier = case_identifier
    cropping.get_case_identifier_from_npz = case_identifier_from_npz
    cropping.get_patient_identifiers_from_cropped_files = lambda folder: [
        case_identifier_from_npz(value)
        for value in planning_utils.subfiles(folder, join=True, suffix=".npz")
    ]
    dataset_analyzer.get_patient_identifiers_from_cropped_files = (
        cropping.get_patient_identifiers_from_cropped_files
    )
    planning_utils.create_lists_from_splitted_dataset = create_lists
    preprocessing.get_case_identifier_from_npz = case_identifier_from_npz
    return {
        "active": os.name == "nt",
        "scope": "path parsing only",
        "patched_helpers": [
            "preprocessing.cropping.get_case_identifier",
            "preprocessing.cropping.get_case_identifier_from_npz",
            "experiment_planning.utils.create_lists_from_splitted_dataset",
            "experiment_planning.DatasetAnalyzer.get_patient_identifiers_from_cropped_files",
            "preprocessing.preprocessing.get_case_identifier_from_npz",
            "sys.modules['nnunet'] import alias to unchanged nnunet_mednext package",
        ],
        "upstream_source_modified": False,
    }


def run_official_preprocessing(
    *,
    task_id: int,
    planner_3d: str,
    threads: int,
    verify_dataset_integrity: bool,
) -> None:
    """Invoke the official planning entry point under the path-only shim."""

    apply_windows_path_compatibility()
    nnunet_plan_and_preprocess: Any = importlib.import_module(
        "nnunet_mednext.experiment_planning.nnUNet_plan_and_preprocess"
    )

    arguments = [
        "mednextv1_plan_and_preprocess",
        "-t",
        str(task_id),
        "-pl3d",
        planner_3d,
        "-pl2d",
        "None",
        "-tl",
        str(threads),
        "-tf",
        str(threads),
    ]
    if verify_dataset_integrity:
        arguments.append("--verify_dataset_integrity")
    previous = sys.argv
    try:
        sys.argv = arguments
        nnunet_plan_and_preprocess.main()
    finally:
        sys.argv = previous


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    preprocess = subparsers.add_parser("preprocess")
    preprocess.add_argument("--task-id", type=int, required=True)
    preprocess.add_argument("--planner-3d", required=True)
    preprocess.add_argument("--threads", type=int, default=8)
    preprocess.add_argument("--verify-dataset-integrity", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.command == "preprocess":
        run_official_preprocessing(
            task_id=args.task_id,
            planner_3d=args.planner_3d,
            threads=args.threads,
            verify_dataset_integrity=args.verify_dataset_integrity,
        )
        return 0
    raise AssertionError(args.command)  # pragma: no cover


__all__ = ["apply_windows_path_compatibility", "run_official_preprocessing"]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
