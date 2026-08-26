"""Audit a complete, TTA-off nnU-Net inference timing run without rerunning it."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from glioma_seg.monitoring.timing import write_json_atomic

EXPECTED_CHANNELS = ("0000", "0001", "0002", "0003")


def inference_input_case_ids(input_dir: str | Path) -> tuple[str, ...]:
    """Require exactly four nnU-Net modality files for every timing case."""

    directory = Path(input_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"Inference input directory does not exist: {directory}")
    channels_by_case: dict[str, set[str]] = {}
    malformed: list[str] = []
    for path in sorted(directory.iterdir()):
        if not path.is_file() or not path.name.endswith(".nii.gz"):
            continue
        stem = path.name.removesuffix(".nii.gz")
        case_id, separator, channel = stem.rpartition("_")
        if not separator or not case_id or channel not in EXPECTED_CHANNELS:
            malformed.append(path.name)
            continue
        channels_by_case.setdefault(case_id, set()).add(channel)
    if malformed:
        raise ValueError(f"Malformed nnU-Net inference filenames: {malformed[:10]}")
    if not channels_by_case:
        raise ValueError(f"No nnU-Net inference cases found in {directory}")
    incomplete = {
        case_id: sorted(set(EXPECTED_CHANNELS) - channels)
        for case_id, channels in channels_by_case.items()
        if channels != set(EXPECTED_CHANNELS)
    }
    if incomplete:
        raise ValueError(f"Inference cases do not have exactly four modalities: {incomplete}")
    return tuple(sorted(channels_by_case))


def prediction_case_ids(prediction_dir: str | Path) -> tuple[str, ...]:
    """Return nnU-Net segmentation IDs from one flat prediction directory."""

    directory = Path(prediction_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"Prediction directory does not exist: {directory}")
    case_ids = tuple(
        sorted(
            path.name.removesuffix(".nii.gz")
            for path in directory.iterdir()
            if path.is_file() and path.name.endswith(".nii.gz")
        )
    )
    if not case_ids:
        raise ValueError(f"No prediction NIfTIs found in {directory}")
    if len(case_ids) != len(set(case_ids)):
        raise ValueError(f"Duplicate prediction case IDs in {directory}")
    return case_ids


def audit_inference_timing(
    *,
    input_dir: str | Path,
    prediction_dir: str | Path,
    runtime_json: str | Path,
    finalize_fresh_run: bool = False,
) -> dict[str, Any]:
    """Validate exact case identity and truthful timing/TTA metadata.

    ``finalize_fresh_run`` is only for an invocation that started with an empty
    prediction directory. It marks the runtime as a comparable complete run so
    a later pipeline resume can reuse it without timing an already-complete set.
    """

    input_cases = inference_input_case_ids(input_dir)
    output_cases = prediction_case_ids(prediction_dir)
    if input_cases != output_cases:
        missing = sorted(set(input_cases) - set(output_cases))
        extra = sorted(set(output_cases) - set(input_cases))
        raise ValueError(
            f"Inference output inventory mismatch: missing={missing[:10]}, extra={extra[:10]}"
        )
    runtime_path = Path(runtime_json)
    if not runtime_path.is_file():
        raise FileNotFoundError(f"Inference runtime artifact is missing: {runtime_path}")
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    if not isinstance(runtime, dict):
        raise ValueError(f"Expected a JSON object in {runtime_path}")
    if runtime.get("tta_state") != "OFF":
        raise ValueError(f"Primary timing must use TTA OFF, got {runtime.get('tta_state')!r}")
    if runtime.get("number_of_cases") != len(input_cases):
        raise ValueError(
            "Inference timing denominator differs from the exact input/output inventory: "
            f"runtime={runtime.get('number_of_cases')}, inventory={len(input_cases)}"
        )
    total = runtime.get("total_seconds")
    mean = runtime.get("mean_seconds_per_case")
    if not isinstance(total, int | float) or not math.isfinite(total) or total <= 0:
        raise ValueError(f"Invalid inference total_seconds: {total!r}")
    if not isinstance(mean, int | float) or not math.isfinite(mean) or mean <= 0:
        raise ValueError(f"Invalid inference mean_seconds_per_case: {mean!r}")
    if not math.isclose(mean, total / len(input_cases), rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("Inference mean_seconds_per_case does not match total/case_count")
    recorded_output = runtime.get("output_dir")
    if recorded_output and Path(str(recorded_output)).resolve() != Path(prediction_dir).resolve():
        raise ValueError("Inference runtime output_dir differs from the audited predictions")

    if finalize_fresh_run:
        runtime.update(
            {
                "timing_scope": "fresh_complete_run",
                "timing_comparable": True,
                "case_ids": list(input_cases),
                "input_dir": str(Path(input_dir).resolve()),
                "output_dir": str(Path(prediction_dir).resolve()),
            }
        )
        write_json_atomic(runtime_path.resolve(), runtime)
    elif runtime.get("timing_scope") != "fresh_complete_run" or not bool(
        runtime.get("timing_comparable")
    ):
        raise ValueError("Existing timing artifact is not marked as a fresh complete run")
    return runtime


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--runtime-json", type=Path, required=True)
    parser.add_argument("--finalize-fresh-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    runtime = audit_inference_timing(
        input_dir=args.input_dir,
        prediction_dir=args.prediction_dir,
        runtime_json=args.runtime_json,
        finalize_fresh_run=args.finalize_fresh_run,
    )
    print(json.dumps(runtime, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
