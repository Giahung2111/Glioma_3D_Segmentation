"""Strict adapter for an externally supplied official BraTS 2023 evaluator.

No lesion-wise formula is approximated in this module.  An official result is
reported only after an explicitly configured, version-pinned implementation
runs successfully and creates the declared artifacts.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class OfficialMetricStatus:
    available: bool
    reason: str
    source: str | None
    version_or_commit: str | None
    command: tuple[str, ...] | None
    outputs: tuple[str, ...]
    timestamp_utc: str


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_status(output_dir: Path, status: OfficialMetricStatus) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "official_brats_metrics_status.json"
    path.write_text(
        json.dumps(asdict(status), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return path


def mark_official_metrics_unavailable(output_dir: str | Path, reason: str) -> OfficialMetricStatus:
    """Record an honest unavailable status without creating metric CSVs."""

    if not reason.strip():
        raise ValueError("An explicit reason is required when official metrics are unavailable")
    status = OfficialMetricStatus(
        available=False,
        reason=reason.strip(),
        source=None,
        version_or_commit=None,
        command=None,
        outputs=(),
        timestamp_utc=_now_utc(),
    )
    _write_status(Path(output_dir), status)
    return status


@dataclass(frozen=True)
class OfficialBraTS2023CommandAdapter:
    """Opt-in command adapter for a pinned official implementation.

    The command must contain ``{ground_truth}``, ``{predictions}``, and
    ``{output_dir}`` placeholders.  This project intentionally does not guess
    a CLI or silently fall back to its standard semantic metrics.
    """

    command_template: tuple[str, ...]
    source: str
    version_or_commit: str
    expected_outputs: tuple[str, ...] = (
        "official_lesionwise_metrics_per_case.csv",
        "official_lesionwise_metrics_summary.csv",
    )

    def __post_init__(self) -> None:
        if not self.source.strip() or not self.version_or_commit.strip():
            raise ValueError(
                "Official evaluator source and package version/git commit are required"
            )
        combined = "\n".join(self.command_template)
        required = ("{ground_truth}", "{predictions}", "{output_dir}")
        missing = [placeholder for placeholder in required if placeholder not in combined]
        if missing:
            raise ValueError(f"Official evaluator command is missing placeholders: {missing}")
        if not self.expected_outputs:
            raise ValueError("At least one expected official output must be declared")

    def run(
        self,
        ground_truth_dir: str | Path,
        prediction_dir: str | Path,
        output_dir: str | Path,
    ) -> OfficialMetricStatus:
        gt_dir = Path(ground_truth_dir).resolve()
        pred_dir = Path(prediction_dir).resolve()
        destination = Path(output_dir).resolve()
        if not gt_dir.is_dir():
            raise FileNotFoundError(f"Ground-truth directory does not exist: {gt_dir}")
        if not pred_dir.is_dir():
            raise FileNotFoundError(f"Prediction directory does not exist: {pred_dir}")
        destination.mkdir(parents=True, exist_ok=True)
        substitutions = {
            "ground_truth": str(gt_dir),
            "predictions": str(pred_dir),
            "output_dir": str(destination),
        }
        command = tuple(token.format(**substitutions) for token in self.command_template)
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        log_path = destination / "official_brats_evaluator.log"
        log_path.write_text(
            f"command: {command!r}\nexit_code: {completed.returncode}\n"
            f"\n[stdout]\n{completed.stdout}\n[stderr]\n{completed.stderr}\n",
            encoding="utf-8",
        )
        if completed.returncode != 0:
            status = OfficialMetricStatus(
                available=False,
                reason=(
                    f"Pinned official evaluator failed with exit code {completed.returncode}; "
                    f"see {log_path.name}"
                ),
                source=self.source,
                version_or_commit=self.version_or_commit,
                command=command,
                outputs=(),
                timestamp_utc=_now_utc(),
            )
            _write_status(destination, status)
            return status

        missing = [name for name in self.expected_outputs if not (destination / name).is_file()]
        if missing:
            status = OfficialMetricStatus(
                available=False,
                reason=(
                    "Evaluator exited successfully but declared official artifacts "
                    f"are missing: {missing}"
                ),
                source=self.source,
                version_or_commit=self.version_or_commit,
                command=command,
                outputs=(),
                timestamp_utc=_now_utc(),
            )
            _write_status(destination, status)
            return status

        outputs = tuple(str((destination / name).resolve()) for name in self.expected_outputs)
        status = OfficialMetricStatus(
            available=True,
            reason=(
                "Pinned official BraTS 2023 evaluator completed and declared "
                "artifacts were verified"
            ),
            source=self.source,
            version_or_commit=self.version_or_commit,
            command=command,
            outputs=outputs,
            timestamp_utc=_now_utc(),
        )
        _write_status(destination, status)
        return status


def evaluate_official_brats2023(
    ground_truth_dir: str | Path,
    prediction_dir: str | Path,
    output_dir: str | Path,
    *,
    adapter: OfficialBraTS2023CommandAdapter | None,
) -> OfficialMetricStatus:
    """Run only a configured official adapter, otherwise report unavailable."""

    if adapter is None:
        return mark_official_metrics_unavailable(
            output_dir,
            "official lesion-wise metric unavailable: no pinned official "
            "BraTS 2023 evaluator was configured",
        )
    return adapter.run(ground_truth_dir, prediction_dir, output_dir)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a pinned official BraTS 2023 evaluator, or record why it is unavailable."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ground-truth-dir", type=Path)
    parser.add_argument("--prediction-dir", type=Path)
    parser.add_argument("--source")
    parser.add_argument("--version-or-commit")
    parser.add_argument("--expected-output", action="append")
    parser.add_argument("--unavailable-reason")
    parser.add_argument(
        "--official-command",
        nargs=argparse.REMAINDER,
        help=(
            "Pinned evaluator command tokens containing {ground_truth}, {predictions}, "
            "and {output_dir}; this option must be last."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.unavailable_reason:
        if args.official_command:
            raise ValueError("Choose either --unavailable-reason or --official-command")
        status = mark_official_metrics_unavailable(args.output_dir, args.unavailable_reason)
    else:
        required = {
            "ground_truth_dir": args.ground_truth_dir,
            "prediction_dir": args.prediction_dir,
            "source": args.source,
            "version_or_commit": args.version_or_commit,
            "official_command": args.official_command,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise ValueError(
                "Official evaluation requires an explicit unavailable reason or all pinned adapter "
                f"arguments; missing={missing}"
            )
        adapter = OfficialBraTS2023CommandAdapter(
            command_template=tuple(args.official_command),
            source=args.source,
            version_or_commit=args.version_or_commit,
            expected_outputs=tuple(args.expected_output)
            if args.expected_output
            else OfficialBraTS2023CommandAdapter.expected_outputs,
        )
        status = adapter.run(args.ground_truth_dir, args.prediction_dir, args.output_dir)
    print(json.dumps(asdict(status), indent=2, ensure_ascii=False))
    return 0 if status.available or args.unavailable_reason else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
