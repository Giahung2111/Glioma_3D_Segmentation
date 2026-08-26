"""Pinned runner for the unmodified official BraTS 2023 lesion-wise metrics.

The runner deliberately executes ``External/BraTS-2023-Metrics/metrics.py`` in
an explicitly selected Python environment.  It does not copy or reimplement
the formula.  Final ``official_*`` metric artifacts are staged only after all
matched cases finish successfully.

This file also contains a private worker mode so it can be executed directly
by the Python 3.9 compatibility environment without installing ``glioma-seg``
there.  The worker uses only the pinned evaluator's own dependencies.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REGION_ORDER: tuple[str, ...] = ("ET", "TC", "WT")
OFFICIAL_REPOSITORY_COMMIT = "43c905242b2eecf421d4ab2da7af8ece9777d322"
OFFICIAL_PER_CASE_FILENAME = "official_lesionwise_metrics_per_case.csv"
OFFICIAL_SUMMARY_FILENAME = "official_lesionwise_metrics_summary.csv"
OFFICIAL_SUMMARY_JSON_FILENAME = "official_lesionwise_metrics_summary.json"
OFFICIAL_STATUS_FILENAME = "official_brats_metrics_status.json"
OFFICIAL_LOG_FILENAME = "official_brats_evaluator.log"

OFFICIAL_COLUMNS: tuple[str, ...] = (
    "Num_TP",
    "Num_FP",
    "Num_FN",
    "Sensitivity",
    "Specificity",
    "Legacy_Dice",
    "Legacy_HD95",
    "GT_Complete_Volume",
    "LesionWise_Score_Dice",
    "LesionWise_Score_HD95",
)

PER_CASE_FIELDS: tuple[str, ...] = (
    "case_id",
    "dice_et",
    "dice_tc",
    "dice_wt",
    "hd95_et_mm",
    "hd95_tc_mm",
    "hd95_wt_mm",
    "num_tp_et",
    "num_tp_tc",
    "num_tp_wt",
    "num_fp_et",
    "num_fp_tc",
    "num_fp_wt",
    "num_fn_et",
    "num_fn_tc",
    "num_fn_wt",
    "sensitivity_et",
    "sensitivity_tc",
    "sensitivity_wt",
    "specificity_et",
    "specificity_tc",
    "specificity_wt",
    "legacy_dice_et",
    "legacy_dice_tc",
    "legacy_dice_wt",
    "legacy_hd95_et_mm",
    "legacy_hd95_tc_mm",
    "legacy_hd95_wt_mm",
    "gt_complete_volume_et_mm3",
    "gt_complete_volume_tc_mm3",
    "gt_complete_volume_wt_mm3",
)


class OfficialEvaluationError(RuntimeError):
    """Raised when pinned official evaluation cannot be completed faithfully."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", suffix=".tmp", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", suffix=".tmp", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(text)
    temporary.replace(path)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _case_id(path: Path) -> str:
    if path.name.endswith(".nii.gz"):
        return path.name[: -len(".nii.gz")]
    if path.suffix == ".nii":
        return path.stem
    raise ValueError(f"Not a NIfTI filename: {path.name}")


def _nifti_files(directory: Path) -> dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"NIfTI directory does not exist: {directory}")
    files: dict[str, Path] = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file() or not (path.name.endswith(".nii.gz") or path.suffix == ".nii"):
            continue
        identifier = _case_id(path)
        if identifier in files:
            raise ValueError(f"Duplicate case ID {identifier!r} in {directory}")
        files[identifier] = path.resolve()
    return files


def _validation_ids(splits_json: Path, fold: int) -> list[str]:
    payload = json.loads(splits_json.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or fold < 0 or fold >= len(payload):
        raise ValueError(f"Invalid fold {fold} for split file {splits_json}")
    split = payload[fold]
    if not isinstance(split, dict) or not isinstance(split.get("val"), list):
        raise ValueError(f"Fold {fold} does not contain a validation list")
    validation = [str(value) for value in split["val"]]
    training = {str(value) for value in split.get("train", [])}
    overlap = sorted(training.intersection(validation))
    if overlap:
        raise ValueError(f"Train/validation leakage in fold {fold}: {overlap[:10]}")
    if not validation or len(validation) != len(set(validation)):
        raise ValueError(f"Fold {fold} validation IDs are empty or duplicated")
    return validation


def _git(official_root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ("git", "-C", str(official_root), *arguments),
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode != 0:
        raise OfficialEvaluationError(
            f"git {' '.join(arguments)} failed for {official_root}: {completed.stderr.strip()}"
        )
    return completed.stdout.strip()


def verify_official_repository(official_root: str | Path, expected_commit: str) -> dict[str, str]:
    """Verify commit and tracked-file cleanliness without touching upstream."""

    root = Path(official_root).resolve()
    metrics_path = root / "metrics.py"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"Official metrics.py does not exist: {metrics_path}")
    commit = _git(root, "rev-parse", "HEAD")
    if commit.lower() != expected_commit.lower():
        raise OfficialEvaluationError(
            f"Official repository commit mismatch: expected {expected_commit}, found {commit}"
        )
    tracked_status = _git(root, "status", "--porcelain", "--untracked-files=no")
    if tracked_status:
        raise OfficialEvaluationError(
            "Official repository has tracked modifications; refusing to call a modified formula: "
            f"{tracked_status}"
        )
    try:
        source = _git(root, "remote", "get-url", "origin")
    except OfficialEvaluationError:
        source = str(root)
    return {
        "root": str(root),
        "source": source,
        "commit": commit,
        "metrics_sha256": _sha256(metrics_path),
    }


def _python_identity(python_executable: Path) -> dict[str, str]:
    if not python_executable.is_file():
        raise FileNotFoundError(
            f"Official compatibility Python does not exist: {python_executable}"
        )
    code = (
        "import json,platform,sys; "
        "print(json.dumps({'executable':sys.executable,"
        "'version':platform.python_version()}))"
    )
    completed = subprocess.run(
        (str(python_executable), "-c", code),
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode != 0:
        raise OfficialEvaluationError(
            f"Cannot run official compatibility Python: {completed.stderr.strip()}"
        )
    try:
        identity = json.loads(completed.stdout.strip())
    except json.JSONDecodeError as exc:
        raise OfficialEvaluationError(
            f"Compatibility Python returned an invalid identity: {completed.stdout!r}"
        ) from exc
    return {"executable": str(identity["executable"]), "version": str(identity["version"])}


def _matched_jobs(
    ground_truth_dir: Path,
    prediction_dir: Path,
    case_ids: Sequence[str] | None,
    strict_predictions: bool,
) -> list[dict[str, str]]:
    ground_truth = _nifti_files(ground_truth_dir)
    predictions = _nifti_files(prediction_dir)
    selected = sorted(ground_truth) if case_ids is None else sorted(dict.fromkeys(case_ids))
    if not selected:
        raise ValueError("No cases selected for official evaluation")
    missing_gt = [value for value in selected if value not in ground_truth]
    missing_pred = [value for value in selected if value not in predictions]
    if missing_gt or missing_pred:
        raise FileNotFoundError(
            f"Unmatched official-evaluation cases: missing_gt={missing_gt[:10]}, "
            f"missing_prediction={missing_pred[:10]}"
        )
    extra = sorted(set(predictions) - set(selected))
    if strict_predictions and extra:
        raise ValueError(
            f"Prediction directory contains cases outside the selected fold: {extra[:10]}"
        )
    return [
        {
            "case_id": value,
            "ground_truth": str(ground_truth[value]),
            "prediction": str(predictions[value]),
        }
        for value in selected
    ]


def _import_official_metrics(official_root: Path) -> Any:
    metrics_path = official_root / "metrics.py"
    sys.path.insert(0, str(official_root))
    spec = importlib.util.spec_from_file_location("pinned_brats2023_metrics", metrics_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load official evaluator at {metrics_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _validate_worker_pair(prediction: Path, ground_truth: Path, case_id: str) -> None:
    import nibabel as nib
    import numpy as np

    pred_image: Any = nib.load(str(prediction))
    gt_image: Any = nib.load(str(ground_truth))
    pred = np.asanyarray(pred_image.dataobj)
    gt = np.asanyarray(gt_image.dataobj)
    if pred.ndim != 3 or gt.ndim != 3 or pred.shape != gt.shape:
        raise ValueError(
            f"{case_id}: label shapes are not matching 3D arrays: {pred.shape} vs {gt.shape}"
        )
    if not np.allclose(pred_image.affine, gt_image.affine, rtol=0.0, atol=1e-4):
        raise ValueError(f"{case_id}: prediction and ground-truth affines differ")
    pred_spacing = pred_image.header.get_zooms()[:3]
    gt_spacing = gt_image.header.get_zooms()[:3]
    if not np.allclose(pred_spacing, gt_spacing, rtol=0.0, atol=1e-6):
        raise ValueError(f"{case_id}: prediction and ground-truth spacing differs")
    for name, array in (("prediction", pred), ("ground truth", gt)):
        if not np.all(np.isfinite(array)) or not np.all(array == np.rint(array)):
            raise ValueError(f"{case_id}: {name} has non-finite or non-integer labels")
        unexpected = sorted(set(int(value) for value in np.unique(array)) - {0, 1, 2, 3})
        if unexpected:
            raise ValueError(
                f"{case_id}: {name} has labels outside BraTS 2023 0/1/2/3: {unexpected}"
            )


def _official_dataframe_to_regions(dataframe: Any, case_id: str) -> dict[str, dict[str, float]]:
    records = dataframe.to_dict(orient="records")
    regions: dict[str, dict[str, float]] = {}
    for record in records:
        label = str(record.get("Labels", "")).upper()
        if label not in REGION_ORDER or label in regions:
            raise ValueError(
                f"{case_id}: official evaluator returned invalid/duplicate label {label!r}"
            )
        missing = [column for column in OFFICIAL_COLUMNS if column not in record]
        if missing:
            raise ValueError(f"{case_id}: official evaluator omitted columns {missing}")
        values = {column: float(record[column]) for column in OFFICIAL_COLUMNS}
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError(
                f"{case_id}: official evaluator returned a non-finite value for {label}"
            )
        regions[label] = values
    missing_regions = [region for region in REGION_ORDER if region not in regions]
    if missing_regions:
        raise ValueError(f"{case_id}: official evaluator omitted regions {missing_regions}")
    return regions


def _worker_main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--job-json", type=Path, required=True)
    parser.add_argument("--result-json", type=Path, required=True)
    parser.add_argument("--challenge-name", default="BraTS-GLI")
    args = parser.parse_args(argv)
    jobs = json.loads(args.job_json.read_text(encoding="utf-8"))
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("Official worker job manifest must be a non-empty list")
    official = _import_official_metrics(args.official_root.resolve())
    results: list[dict[str, Any]] = []
    for index, job in enumerate(jobs, start=1):
        case_id = str(job["case_id"])
        prediction = Path(job["prediction"])
        ground_truth = Path(job["ground_truth"])
        print(f"[official] {index}/{len(jobs)} {case_id}", flush=True)
        _validate_worker_pair(prediction, ground_truth, case_id)
        dataframe = official.get_LesionWiseResults(
            str(prediction), str(ground_truth), args.challenge_name, output=None
        )
        results.append(
            {
                "case_id": case_id,
                "regions": _official_dataframe_to_regions(dataframe, case_id),
            }
        )
    _atomic_json(args.result_json, results)
    print(f"[official] completed {len(results)}/{len(jobs)} cases", flush=True)
    return 0


def flatten_official_case(result: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten one untouched official result into the stable ET/TC/WT CSV schema."""

    case_id = str(result.get("case_id", "")).strip()
    regions = result.get("regions")
    if not case_id or not isinstance(regions, Mapping):
        raise ValueError("Official worker result lacks case_id or regions")
    row: dict[str, Any] = {"case_id": case_id}
    column_map = {
        "dice": "LesionWise_Score_Dice",
        "hd95": "LesionWise_Score_HD95",
        "num_tp": "Num_TP",
        "num_fp": "Num_FP",
        "num_fn": "Num_FN",
        "sensitivity": "Sensitivity",
        "specificity": "Specificity",
        "legacy_dice": "Legacy_Dice",
        "legacy_hd95": "Legacy_HD95",
        "gt_complete_volume": "GT_Complete_Volume",
    }
    for region in REGION_ORDER:
        values = regions.get(region)
        if not isinstance(values, Mapping):
            raise ValueError(f"Official worker result for {case_id} lacks region {region}")
        suffix = region.lower()
        row[f"dice_{suffix}"] = float(values[column_map["dice"]])
        row[f"hd95_{suffix}_mm"] = float(values[column_map["hd95"]])
        for stem in ("num_tp", "num_fp", "num_fn", "sensitivity", "specificity", "legacy_dice"):
            row[f"{stem}_{suffix}"] = float(values[column_map[stem]])
        row[f"legacy_hd95_{suffix}_mm"] = float(values[column_map["legacy_hd95"]])
        row[f"gt_complete_volume_{suffix}_mm3"] = float(values[column_map["gt_complete_volume"]])
    if not all(math.isfinite(float(value)) for key, value in row.items() if key != "case_id"):
        raise ValueError(f"Official worker result for {case_id} contains non-finite values")
    return row


def summarize_official_cases(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Mean official per-case scores with explicit finite denominators."""

    if not rows:
        raise ValueError("Cannot summarize zero official cases")
    summary: list[dict[str, Any]] = []
    for metric, stem, unit, direction in (
        ("Dice", "dice", "", "higher_is_better"),
        ("HD95", "hd95", "mm", "lower_is_better"),
    ):
        output: dict[str, Any] = {
            "metric": metric,
            "unit": unit,
            "direction": direction,
            "total_cases": len(rows),
        }
        for region in REGION_ORDER:
            key = f"{stem}_{region.lower()}" if stem == "dice" else f"{stem}_{region.lower()}_mm"
            values = [float(row[key]) for row in rows]
            finite = [value for value in values if math.isfinite(value)]
            output[region] = sum(finite) / len(finite) if finite else float("nan")
            output[f"{region}_n_valid"] = len(finite)
            output[f"{region}_n_excluded"] = len(values) - len(finite)
        summary.append(output)
    return summary


def _status_payload(
    *,
    available: bool,
    reason: str,
    repository: Mapping[str, str] | None,
    python_identity: Mapping[str, str] | None,
    command: Sequence[str] | None,
    outputs: Sequence[Path],
    case_count: int,
    challenge_name: str,
) -> dict[str, Any]:
    return {
        "available": available,
        "reason": reason,
        "source": repository.get("source") if repository else None,
        "version_or_commit": repository.get("commit") if repository else None,
        "official_root": repository.get("root") if repository else None,
        "metrics_module_sha256": repository.get("metrics_sha256") if repository else None,
        "python_executable": python_identity.get("executable") if python_identity else None,
        "python_version": python_identity.get("version") if python_identity else None,
        "command": list(command) if command else None,
        "outputs": [str(path.resolve()) for path in outputs],
        "case_count": case_count,
        "challenge_name": challenge_name,
        "region_order": list(REGION_ORDER),
        "aggregation": "arithmetic mean of official per-case lesion-wise scores over finite cases",
        "timestamp_utc": _utc_now(),
    }


def run_official_evaluation(
    *,
    ground_truth_dir: str | Path,
    prediction_dir: str | Path,
    output_dir: str | Path,
    official_root: str | Path,
    python_executable: str | Path,
    expected_commit: str = OFFICIAL_REPOSITORY_COMMIT,
    splits_json: str | Path | None = None,
    fold: int = 0,
    strict_predictions: bool = True,
    challenge_name: str = "BraTS-GLI",
) -> dict[str, Any]:
    """Run the pinned upstream evaluator for every matched case.

    On any failure, no new official metric CSV/summary JSON is published.  A
    truthful unavailable status and execution log are written instead.
    """

    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    targets = (
        destination / OFFICIAL_PER_CASE_FILENAME,
        destination / OFFICIAL_SUMMARY_FILENAME,
        destination / OFFICIAL_SUMMARY_JSON_FILENAME,
    )
    existing = [str(path) for path in targets if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing official metric artifacts; "
            "use a new experiment directory: "
            f"{existing}"
        )

    repository: dict[str, str] | None = None
    python_identity: dict[str, str] | None = None
    worker_command: tuple[str, ...] | None = None
    jobs: list[dict[str, str]] = []
    log_lines: list[str] = []
    try:
        repository = verify_official_repository(official_root, expected_commit)
        python_path = Path(python_executable).resolve()
        python_identity = _python_identity(python_path)
        selected = _validation_ids(Path(splits_json), fold) if splits_json is not None else None
        jobs = _matched_jobs(
            Path(ground_truth_dir),
            Path(prediction_dir),
            selected,
            strict_predictions,
        )
        with tempfile.TemporaryDirectory(prefix="brats_official_") as temporary_name:
            temporary = Path(temporary_name)
            job_path = temporary / "jobs.json"
            result_path = temporary / "results.json"
            _atomic_json(job_path, jobs)
            worker_command = (
                str(python_path),
                str(Path(__file__).resolve()),
                "--worker",
                "--official-root",
                repository["root"],
                "--job-json",
                str(job_path),
                "--result-json",
                str(result_path),
                "--challenge-name",
                challenge_name,
            )
            environment = os.environ.copy()
            environment["PYTHONUTF8"] = "1"
            process = subprocess.Popen(
                worker_command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=environment,
            )
            assert process.stdout is not None
            for line in process.stdout:
                clean = line.rstrip("\r\n")
                log_lines.append(clean)
                print(clean, flush=True)
            exit_code = process.wait()
            if exit_code != 0:
                raise OfficialEvaluationError(
                    f"Pinned official evaluator worker failed with exit code {exit_code}"
                )
            if not result_path.is_file():
                raise OfficialEvaluationError("Official worker completed without a result manifest")
            raw_results = json.loads(result_path.read_text(encoding="utf-8"))
            if not isinstance(raw_results, list) or len(raw_results) != len(jobs):
                raise OfficialEvaluationError(
                    "Official worker returned "
                    f"{len(raw_results) if isinstance(raw_results, list) else 'invalid'} "
                    f"results for {len(jobs)} cases"
                )

        rows = [flatten_official_case(result) for result in raw_results]
        if [row["case_id"] for row in rows] != [job["case_id"] for job in jobs]:
            raise OfficialEvaluationError(
                "Official result case order/identity differs from the job manifest"
            )
        summary = summarize_official_cases(rows)
        summary_fields = (
            "metric",
            "unit",
            "direction",
            "ET",
            "TC",
            "WT",
            "ET_n_valid",
            "ET_n_excluded",
            "TC_n_valid",
            "TC_n_excluded",
            "WT_n_valid",
            "WT_n_excluded",
            "total_cases",
        )
        with tempfile.TemporaryDirectory(
            prefix="brats_official_stage_", dir=destination
        ) as stage_name:
            stage = Path(stage_name)
            staged_per_case = stage / OFFICIAL_PER_CASE_FILENAME
            staged_summary = stage / OFFICIAL_SUMMARY_FILENAME
            staged_json = stage / OFFICIAL_SUMMARY_JSON_FILENAME
            _write_csv(staged_per_case, rows, PER_CASE_FIELDS)
            _write_csv(staged_summary, summary, summary_fields)
            _atomic_json(
                staged_json,
                {
                    "metric_type": "official BraTS 2023 lesion-wise",
                    "region_order": list(REGION_ORDER),
                    "case_count": len(rows),
                    "source": repository["source"],
                    "commit": repository["commit"],
                    "metrics_module_sha256": repository["metrics_sha256"],
                    "summary": summary,
                },
            )
            for staged, target in zip(
                (staged_per_case, staged_summary, staged_json), targets, strict=False
            ):
                staged.replace(target)

        status = _status_payload(
            available=True,
            reason=(
                "Pinned unmodified official BraTS 2023 get_LesionWiseResults "
                "completed for all matched cases"
            ),
            repository=repository,
            python_identity=python_identity,
            command=worker_command,
            outputs=targets,
            case_count=len(rows),
            challenge_name=challenge_name,
        )
        _atomic_json(destination / OFFICIAL_STATUS_FILENAME, status)
        _atomic_text(
            destination / OFFICIAL_LOG_FILENAME,
            "\n".join(
                [
                    f"source: {repository['source']}",
                    f"commit: {repository['commit']}",
                    f"metrics_sha256: {repository['metrics_sha256']}",
                    f"python: {python_identity['executable']} ({python_identity['version']})",
                    f"command: {worker_command!r}",
                    *log_lines,
                    "",
                ]
            ),
        )
        return status
    except Exception as exc:
        # Targets did not exist at entry, so any that exist here are partial
        # publications created by this invocation and must not survive a failed
        # official run.
        for target in targets:
            if target.exists():
                target.unlink()
        reason = f"official lesion-wise metric unavailable: {type(exc).__name__}: {exc}"
        status = _status_payload(
            available=False,
            reason=reason,
            repository=repository,
            python_identity=python_identity,
            command=worker_command,
            outputs=(),
            case_count=0,
            challenge_name=challenge_name,
        )
        _atomic_json(destination / OFFICIAL_STATUS_FILENAME, status)
        _atomic_text(
            destination / OFFICIAL_LOG_FILENAME,
            "\n".join([reason, f"command: {worker_command!r}", *log_lines, ""]),
        )
        raise OfficialEvaluationError(reason) from exc


def _runner_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the pinned, unmodified official BraTS 2023 lesion-wise evaluator."
    )
    parser.add_argument("--ground-truth-dir", type=Path, required=True)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument(
        "--python", dest="python_executable", type=Path, default=Path(sys.executable)
    )
    parser.add_argument("--expected-commit", default=OFFICIAL_REPOSITORY_COMMIT)
    parser.add_argument("--splits-json", type=Path)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--allow-extra-predictions", action="store_true")
    parser.add_argument("--challenge-name", choices=("BraTS-GLI",), default="BraTS-GLI")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if "--worker" in arguments:
        return _worker_main(arguments)
    args = _runner_parser().parse_args(arguments)
    try:
        status = run_official_evaluation(
            ground_truth_dir=args.ground_truth_dir,
            prediction_dir=args.prediction_dir,
            output_dir=args.output_dir,
            official_root=args.official_root,
            python_executable=args.python_executable,
            expected_commit=args.expected_commit,
            splits_json=args.splits_json,
            fold=args.fold,
            strict_predictions=not args.allow_extra_predictions,
            challenge_name=args.challenge_name,
        )
    except OfficialEvaluationError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(status, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
