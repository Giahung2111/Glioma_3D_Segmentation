"""Generate summary.md and six-slide weekly notes from real artifacts only."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .tables import key_value_markdown_table, load_metric_summary, metric_markdown_table


@dataclass(frozen=True)
class ReportInputs:
    output_dir: Path
    experiment_json: Path
    metrics_summary_csv: Path
    environment_json: Path | None = None
    runtime_json: Path | None = None
    inference_runtime_json: Path | None = None
    gpu_summary_json: Path | None = None
    data_validation_json: Path | None = None
    official_validation_json: Path | None = None
    preprocessing_artifacts_json: Path | None = None
    official_status_json: Path | None = None
    official_summary_csv: Path | None = None
    evaluation_protocol_json: Path | None = None
    failure_cases_csv: Path | None = None
    figures_dir: Path | None = None
    figures_manifest_csv: Path | None = None


def _load_json(path: Path | None, *, required: bool = False) -> dict[str, Any]:
    if path is None:
        if required:
            raise ValueError("A required JSON artifact path was not supplied")
        return {}
    if not path.is_file():
        if required:
            raise FileNotFoundError(f"Required JSON artifact does not exist: {path}")
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _recorded(mapping: Mapping[str, Any], key: str, source_name: str) -> str:
    value = mapping.get(key)
    if value is None or value == "":
        return f"NOT AVAILABLE — {key} was not recorded in {source_name}"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, list | tuple):
        return ", ".join(str(item) for item in value)
    return str(value)


def _first_recorded(mappings: tuple[tuple[Mapping[str, Any], str], ...], *keys: str) -> str:
    for mapping, _source in mappings:
        for key in keys:
            value = mapping.get(key)
            if value is not None and value != "":
                if isinstance(value, bool):
                    return "yes" if value else "no"
                if isinstance(value, list | tuple):
                    return ", ".join(str(item) for item in value)
                return str(value)
    joined = "/".join(keys)
    sources = ", ".join(source for _, source in mappings)
    return f"NOT AVAILABLE — {joined} was not recorded in {sources}"


def _official_validation_scope(validation: Mapping[str, Any]) -> str:
    """Return a provenance-backed description of the image-only validation set."""

    if not validation:
        return "NOT AVAILABLE — no official-validation data-validation artifact was supplied"
    if validation.get("valid") is not True or validation.get("dataset_kind") != "validation":
        return "NOT AVAILABLE — the supplied official-validation artifact is not valid"
    expected = validation.get("expected_case_count")
    actual = validation.get("actual_case_count")
    valid = validation.get("valid_case_count")
    values = (expected, actual, valid)
    if any(not isinstance(value, int) or isinstance(value, bool) or value < 1 for value in values):
        return "NOT AVAILABLE — official-validation case counts are invalid"
    if len(set(values)) != 1 or validation.get("errors") not in ([], None):
        return "NOT AVAILABLE — official-validation case counts are inconsistent"
    return f"{valid} (modalities only; no public GT; never locally scored)"


def _verified_split_line(experiment: Mapping[str, Any], preprocessing: Mapping[str, Any]) -> str:
    """Render split/seed only when structured fields reconcile with audit evidence."""

    split = experiment.get("split")
    if not isinstance(split, Mapping):
        return "NOT AVAILABLE — structured split was not recorded in experiment.json"
    source = split.get("source")
    fold = split.get("fold")
    train_cases = split.get("train_cases")
    validation_cases = split.get("validation_cases")
    if (
        not isinstance(source, str)
        or not source.strip()
        or not isinstance(fold, int)
        or isinstance(fold, bool)
        or fold < 0
        or not isinstance(train_cases, int)
        or isinstance(train_cases, bool)
        or train_cases < 1
        or not isinstance(validation_cases, int)
        or isinstance(validation_cases, bool)
        or validation_cases < 1
    ):
        return "NOT AVAILABLE — structured split fields are invalid"
    if preprocessing.get("valid") is not True:
        return "NOT AVAILABLE — preprocessing artifact audit is not valid"
    details = preprocessing.get("details")
    if not isinstance(details, Mapping) or details.get("splits_created") is not True:
        return "NOT AVAILABLE — preprocessing artifact does not verify split creation"
    audited_source = details.get("splits_file")
    if not isinstance(audited_source, str) or not audited_source.strip():
        return "NOT AVAILABLE — preprocessing artifact does not record the split source"
    if os.path.normcase(os.path.abspath(source)) != os.path.normcase(
        os.path.abspath(audited_source)
    ):
        return "NOT AVAILABLE — experiment and preprocessing split sources disagree"
    checks = preprocessing.get("checks")
    if not isinstance(checks, list):
        return "NOT AVAILABLE — preprocessing artifact has no structured checks"
    matching_checks = [
        check
        for check in checks
        if isinstance(check, Mapping)
        and check.get("name") == "official five-fold split"
        and check.get("ok") is True
    ]
    if len(matching_checks) != 1:
        return "NOT AVAILABLE — official five-fold split audit did not pass exactly once"
    detail = matching_checks[0].get("detail")
    if not isinstance(detail, str):
        return "NOT AVAILABLE — official split audit detail was not recorded"
    seed_match = re.search(r"\bofficial seed=(\d+)\b", detail)
    sizes_match = re.search(r"\bfold_sizes=\[(.*)\]\s*$", detail)
    if seed_match is None or sizes_match is None:
        return "NOT AVAILABLE — official split audit lacks parseable seed/fold sizes"
    fold_sizes = [
        (int(train), int(validation))
        for train, validation in re.findall(r"\((\d+),\s*(\d+)\)", sizes_match.group(1))
    ]
    if len(fold_sizes) != 5:
        return "NOT AVAILABLE — official split audit does not contain five folds"
    if fold >= len(fold_sizes) or fold_sizes[fold] != (train_cases, validation_cases):
        return "NOT AVAILABLE — structured split counts disagree with the official split audit"
    seed = int(seed_match.group(1))
    return (
        f"source={source}; fold={fold}; train={train_cases}; validation={validation_cases}; "
        f"official seed={seed} (verified)"
    )


def _load_failures(path: Path | None) -> list[dict[str, str]]:
    if path is None or not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_figure_paths(path: Path | None) -> dict[str, Path]:
    if path is None or not path.is_file():
        return {}
    paths: dict[str, Path] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            case_id = str(row.get("case_id", "")).strip()
            figure_path = str(row.get("figure_path", "")).strip()
            if not case_id or not figure_path:
                raise ValueError(f"Invalid figure-manifest row in {path}: {row}")
            candidate = Path(figure_path)
            previous = paths.get(case_id)
            if previous is not None and previous != candidate:
                raise ValueError(f"Duplicate figure paths for {case_id} in {path}")
            paths[case_id] = candidate
    return paths


def _failure_table(
    rows: list[dict[str, str]],
    output_dir: Path,
    figures_dir: Path | None,
    figure_paths: Mapping[str, Path] | None = None,
    max_cases: int = 15,
) -> str:
    if not rows:
        return "Failure-case artifact NOT AVAILABLE — no completed failure_cases.csv was supplied."
    if max_cases < 1:
        raise ValueError("max_cases must be positive")
    lines = ["| Case | Region | Failure | Observation | Figure |", "|---|---|---|---|---|"]
    seen_cases: set[str] = set()
    for row in rows:
        case_id = row.get("case_id", "unrecorded")
        if case_id in seen_cases:
            continue
        seen_cases.add(case_id)
        region = row.get("region") or row.get("primary_region") or "not recorded"
        failure = row.get("failure_type") or row.get("primary_failure_type") or "not classified"
        observation = row.get("observation") or row.get("selection_reasons") or "not recorded"
        figure_cell = "not generated"
        figure_value: str | Path | None = row.get("figure") or row.get("figure_path")
        if not figure_value and figure_paths is not None:
            figure_value = figure_paths.get(case_id)
        if not figure_value and figures_dir is not None:
            conventional = figures_dir / f"{case_id}_t1c_flair_gt_pred.png"
            if conventional.is_file():
                figure_value = conventional
        if figure_value:
            figure_path = Path(figure_value)
            if not figure_path.is_absolute() and figures_dir is not None:
                figure_path = figures_dir / figure_path
            if figure_path.is_file():
                try:
                    relative = figure_path.resolve().relative_to(output_dir.resolve())
                    figure_cell = f"[{figure_path.name}]({relative.as_posix()})"
                except ValueError:
                    figure_cell = str(figure_path.resolve())
        escaped = [
            str(item).replace("|", "\\|").replace("\n", " ")
            for item in (case_id, region, failure, observation, figure_cell)
        ]
        lines.append("| " + " | ".join(escaped) + " |")
        if len(seen_cases) >= max_cases:
            break
    return "\n".join(lines)


def _official_section(inputs: ReportInputs) -> str:
    status = _load_json(inputs.official_status_json)
    if not status:
        return (
            "BraTS 2023 official lesion-wise metrics: **NOT AVAILABLE** — "
            "no official evaluator status artifact was supplied."
        )
    if not bool(status.get("available")):
        reason = status.get("reason") or "official evaluator did not provide a reason"
        return f"BraTS 2023 official lesion-wise metrics: **NOT AVAILABLE** — {reason}"
    if inputs.official_summary_csv is None or not inputs.official_summary_csv.is_file():
        return (
            "BraTS 2023 official lesion-wise metrics: **NOT AVAILABLE** — status says available, "
            "but the normalized official summary CSV was not supplied to reporting."
        )
    rows = load_metric_summary(inputs.official_summary_csv)
    identity = key_value_markdown_table(
        [
            ("Evaluator source", status.get("source", "not recorded")),
            ("Version/commit", status.get("version_or_commit", "not recorded")),
        ]
    )
    return identity + "\n\n" + metric_markdown_table(rows, title_prefix="Lesion-wise")


def _runtime_table(
    experiment: Mapping[str, Any],
    runtime: Mapping[str, Any],
    gpu: Mapping[str, Any],
    inference_runtime: Mapping[str, Any],
) -> str:
    training_sources = (
        (runtime, "runtime.json"),
        (gpu, "gpu_summary.json"),
        (experiment, "experiment.json"),
    )
    inference_sources = (
        (inference_runtime, "inference_runtime.json"),
        (experiment, "experiment.json"),
    )
    return key_value_markdown_table(
        [
            (
                "Training time (s)",
                _first_recorded(training_sources, "total_seconds", "training_seconds"),
            ),
            (
                "Average epoch time (s)",
                _first_recorded(
                    training_sources, "average_seconds_per_epoch", "average_epoch_seconds"
                ),
            ),
            (
                "Minimum epoch time (s)",
                _recorded(runtime, "epoch_seconds_min", "runtime.json"),
            ),
            (
                "Median epoch time (s)",
                _recorded(runtime, "epoch_seconds_median", "runtime.json"),
            ),
            (
                "Maximum epoch time (s)",
                _recorded(runtime, "epoch_seconds_max", "runtime.json"),
            ),
            (
                "Peak dedicated VRAM (MB)",
                _first_recorded(training_sources, "peak_memory_used_mb", "peak_vram_mb"),
            ),
            (
                "Mean GPU utilization (%)",
                _first_recorded(
                    training_sources,
                    "mean_gpu_utilization_percent",
                    "mean_gpu_utilization",
                ),
            ),
            (
                "Peak GPU temperature (°C)",
                _recorded(gpu, "peak_temperature_c", "gpu_summary.json"),
            ),
            (
                "Mean GPU power (W)",
                _recorded(gpu, "mean_power_w", "gpu_summary.json"),
            ),
            (
                "Inference time (s/case)",
                _first_recorded(
                    inference_sources,
                    "mean_seconds_per_case",
                    "inference_seconds_per_case",
                ),
            ),
            (
                "Inference cases timed",
                _first_recorded(inference_sources, "number_of_cases"),
            ),
            (
                "Inference timing scope",
                _first_recorded(inference_sources, "timing_scope"),
            ),
            (
                "Primary timing inference TTA state",
                _first_recorded(inference_sources, "tta_state", "inference_tta", "TTA_state"),
            ),
        ]
    )


def _load_all(
    inputs: ReportInputs,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    experiment = _load_json(inputs.experiment_json, required=True)
    environment = _load_json(inputs.environment_json)
    runtime = _load_json(inputs.runtime_json)
    inference_runtime = _load_json(inputs.inference_runtime_json)
    gpu = _load_json(inputs.gpu_summary_json)
    validation = _load_json(inputs.data_validation_json)
    return experiment, environment, runtime, inference_runtime, gpu, validation


def generate_summary_report(inputs: ReportInputs) -> Path:
    """Write summary.md, refusing to proceed without real semantic metrics."""

    experiment, environment, runtime, inference_runtime, gpu, validation = _load_all(inputs)
    official_validation = _load_json(inputs.official_validation_json)
    preprocessing_artifacts = _load_json(inputs.preprocessing_artifacts_json)
    semantic_rows = load_metric_summary(inputs.metrics_summary_csv)
    failures = _load_failures(inputs.failure_cases_csv)
    figure_paths = _load_figure_paths(inputs.figures_manifest_csv)
    evaluation = _load_json(inputs.evaluation_protocol_json)
    output_dir = Path(inputs.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    recorded_fold = _recorded(experiment, "fold", "experiment.json")
    recorded_epochs = _recorded(experiment, "epochs", "experiment.json")

    hardware_sources = ((environment, "environment.json"), (experiment, "experiment.json"))
    model_sources = ((experiment, "experiment.json"),)
    dataset_count = _first_recorded(
        ((validation, "data_validation.json"), (experiment, "experiment.json")),
        "validated_case_count",
        "valid_case_count",
        "actual_case_count",
        "dataset_case_count",
    )
    summary = f"""# Preliminary nnU-Net v2 Baseline — BraTS 2023 GLI

## 1. Experiment Objective

Establish a reproducible first baseline and validate the complete training/evaluation pipeline.

Experiment ID: {_recorded(experiment, "experiment_id", "experiment.json")}

## 2. Dataset

- Dataset: {_first_recorded(((experiment, "experiment.json"),), "dataset", "dataset_name")}
- Dataset ID: {_recorded(experiment, "dataset_id", "experiment.json")}
- Validated training cases: {dataset_count}
- Official validation cases: {_official_validation_scope(official_validation)}
- Modalities: T1n, T1c, T2w, T2-FLAIR
- Raw labels: NCR=1, ED=2, ET=3 (background=0)
- Evaluation regions: ET={{3}}, TC={{1,3}}, WT={{1,2,3}}

## 3. Preprocessing

BraTS-provided preprocessing comprises co-registration, atlas alignment,
approximately 1 mm isotropic sampling, and skull stripping. Local processing
is limited to nnU-Net format conversion, automatic fingerprinting/planning,
normalization, and model-specific preprocessing. No custom registration,
skull stripping, or intensity enhancement is claimed here.

## 4. Model

{
        key_value_markdown_table(
            [
                ("Framework", _first_recorded(model_sources, "framework", "model")),
                (
                    "nnU-Net version",
                    _first_recorded(
                        ((environment, "environment.json"), (experiment, "experiment.json")),
                        "nnUNet_version",
                        "nnunet_version",
                    ),
                ),
                ("Configuration", _recorded(experiment, "configuration", "experiment.json")),
                ("Fold", _recorded(experiment, "fold", "experiment.json")),
                ("Trainer", _recorded(experiment, "trainer", "experiment.json")),
                ("Epochs", _recorded(experiment, "epochs", "experiment.json")),
                ("Architecture", _recorded(experiment, "architecture", "experiment.json")),
            ]
        )
    }

## 5. Hardware

{
        key_value_markdown_table(
            [
                ("GPU", _first_recorded(hardware_sources, "GPU", "gpu", "gpu_name")),
                ("Dedicated VRAM", _first_recorded(hardware_sources, "gpu_vram_mb", "vram_mb")),
                ("NVIDIA driver", _first_recorded(hardware_sources, "nvidia_driver", "driver")),
                ("Python", _first_recorded(hardware_sources, "python", "python_version")),
                ("PyTorch", _first_recorded(hardware_sources, "torch", "torch_version")),
                (
                    "CUDA runtime reported by PyTorch",
                    _first_recorded(hardware_sources, "cuda", "cuda_runtime"),
                ),
                ("cuDNN", _first_recorded(hardware_sources, "cudnn", "cudnn_version")),
            ]
        )
    }

## 6. Training Configuration

{
        key_value_markdown_table(
            [
                ("Patch size", _recorded(experiment, "patch_size", "experiment.json")),
                ("Batch size", _recorded(experiment, "batch_size", "experiment.json")),
                ("Target spacing", _recorded(experiment, "target_spacing", "experiment.json")),
                (
                    "Data-augmentation workers",
                    _recorded(experiment, "nnUNet_n_proc_DA", "experiment.json"),
                ),
                (
                    "Primary timing inference TTA state",
                    _recorded(experiment, "TTA_state", "experiment.json"),
                ),
                (
                    "Split/seed",
                    _verified_split_line(experiment, preprocessing_artifacts),
                ),
            ]
        )
    }

## 7. Preliminary Results

### Standard semantic region-wise metrics

{
        key_value_markdown_table(
            [
                (
                    "Evaluated prediction source",
                    _recorded(evaluation, "prediction_provenance", "evaluation_protocol.json"),
                ),
                (
                    "Metric prediction TTA state",
                    _recorded(evaluation, "prediction_tta_state", "evaluation_protocol.json"),
                ),
                (
                    "Evaluated cases",
                    _recorded(evaluation, "case_count", "evaluation_protocol.json"),
                ),
            ]
        )
    }

{metric_markdown_table(semantic_rows)}

Each cell reports the finite-value denominator. Both-empty Dice and all
one-sided/both-empty HD95 values are undefined and excluded, with per-case
states retained in `metrics_per_case.csv`.

### Official BraTS 2023 lesion-wise metrics

{_official_section(inputs)}

## 8. Runtime

{_runtime_table(experiment, runtime, gpu, inference_runtime)}

## 9. Preliminary Failure Analysis

{_failure_table(failures, output_dir, inputs.figures_dir, figure_paths)}

Classifications describe observed mask/metric patterns and operational
thresholds; they are not asserted medical causes.

## 10. Limitations

- This is a preliminary single-fold result (recorded fold: {recorded_fold}),
  not a completed five-fold nnU-Net reproduction.
- The recorded epoch budget is {recorded_epochs}; shortened 20/50-epoch runs
  are pipeline-validation experiments, not final baselines.
- Full 5-fold CV, ensemble evaluation, and external-hospital validation are
  outside this preliminary baseline.
- Official BraTS validation ground truth is not public and therefore cannot supply local Dice/HD95.
- Metric-prediction TTA and the separate primary timing-inference TTA are reported independently.
- Any missing field above is explicitly marked NOT AVAILABLE rather than replaced by an estimate.

## 11. Next Steps

Complete the full five-fold nnU-Net baseline, add a same-split MedNeXt baseline,
compare ET/TC/WT behavior, select research directions from observed failures,
and only then test boundary-aware or opt-in ensemble experiments. External
hospital evaluation remains a later, separately governed stage.
"""
    destination = output_dir / "summary.md"
    destination.write_text(summary.rstrip() + "\n", encoding="utf-8")
    return destination


def generate_weekly_discussion(inputs: ReportInputs) -> Path:
    """Write a six-slide Markdown discussion outline from the same artifacts."""

    experiment, environment, runtime, inference_runtime, gpu, validation = _load_all(inputs)
    official_validation = _load_json(inputs.official_validation_json)
    preprocessing_artifacts = _load_json(inputs.preprocessing_artifacts_json)
    semantic_rows = load_metric_summary(inputs.metrics_summary_csv)
    failures = _load_failures(inputs.failure_cases_csv)
    figure_paths = _load_figure_paths(inputs.figures_manifest_csv)
    evaluation = _load_json(inputs.evaluation_protocol_json)
    output_dir = Path(inputs.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer = _recorded(experiment, "trainer", "experiment.json")
    epochs = _recorded(experiment, "epochs", "experiment.json")
    dataset_count = _first_recorded(
        ((validation, "data_validation.json"), (experiment, "experiment.json")),
        "validated_case_count",
        "valid_case_count",
        "actual_case_count",
        "dataset_case_count",
    )
    top_failures = _failure_table(
        failures,
        output_dir,
        inputs.figures_dir,
        figure_paths,
        max_cases=3,
    )
    metric_prediction_source = _recorded(
        evaluation, "prediction_provenance", "evaluation_protocol.json"
    )
    metric_prediction_tta = _recorded(
        evaluation, "prediction_tta_state", "evaluation_protocol.json"
    )
    discussion = f"""# Weekly Discussion — Preliminary BraTS 2023 GLI Baseline

## Slide 1 — Research Scope

3D multimodal segmentation of pre-treatment adult glioma. The immediate goal
is a reproducible standard nnU-Net reference before proposing a method.

## Slide 2 — Benchmark

- BraTS 2023 Adult Glioma Pre-Treatment
- Validated training cases: {dataset_count}
- Official validation cases: {_official_validation_scope(official_validation)}
- T1n, T1c, T2w, T2-FLAIR
- Raw labels: NCR=1, ED=2, ET=3
- Regions: ET={{3}}, TC={{1,3}}, WT={{1,2,3}}
- Metrics: standard semantic Dice and physical-spacing HD95; official
  lesion-wise metrics only when the pinned official evaluator is available

## Slide 3 — Baseline and Preprocessing

- BraTS-provided aligned, skull-stripped NIfTI data
- nnU-Net format conversion and automatic planning/preprocessing
- Configuration: {_recorded(experiment, "configuration", "experiment.json")}
- Fold: {_recorded(experiment, "fold", "experiment.json")}
- Split/seed: {_verified_split_line(experiment, preprocessing_artifacts)}
- Trainer / epochs: {trainer} / {epochs}
- Architecture from actual plans: {_recorded(experiment, "architecture", "experiment.json")}

## Slide 4 — Preliminary Results

{metric_markdown_table(semantic_rows)}

Metric prediction source: {metric_prediction_source}

Metric prediction TTA: {metric_prediction_tta}

{_runtime_table(experiment, runtime, gpu, inference_runtime)}

### Official lesion-wise metrics

{_official_section(inputs)}

## Slide 5 — Failure Analysis

{top_failures}

These are observed patterns, not confirmed medical causes.

## Slide 6 — Future Flow

BraTS 2023 → nnU-Net full 5-fold baseline → same-fold MedNeXt baseline →
ET/TC/WT comparison → failure analysis → evidence-driven boundary/small-lesion
investigation → opt-in architecture/checkpoint/region ensemble experiments with
ET ⊆ TC ⊆ WT enforced → later external-hospital evaluation.
"""
    destination = output_dir / "weekly_discussion.md"
    destination.write_text(discussion.rstrip() + "\n", encoding="utf-8")
    return destination


def generate_reports(inputs: ReportInputs) -> tuple[Path, Path]:
    """Generate both required Markdown reports from a single artifact set."""

    return generate_summary_report(inputs), generate_weekly_discussion(inputs)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate artifact-backed preliminary summary and weekly discussion Markdown."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--experiment-json", type=Path, required=True)
    parser.add_argument("--metrics-summary-csv", type=Path, required=True)
    parser.add_argument("--environment-json", type=Path)
    parser.add_argument("--runtime-json", type=Path)
    parser.add_argument("--inference-runtime-json", type=Path)
    parser.add_argument("--gpu-summary-json", type=Path)
    parser.add_argument("--data-validation-json", type=Path)
    parser.add_argument("--official-validation-json", type=Path)
    parser.add_argument("--preprocessing-artifacts-json", type=Path)
    parser.add_argument("--official-status-json", type=Path)
    parser.add_argument("--official-summary-csv", type=Path)
    parser.add_argument("--evaluation-protocol-json", type=Path)
    parser.add_argument("--failure-cases-csv", type=Path)
    parser.add_argument("--figures-dir", type=Path)
    parser.add_argument("--figures-manifest-csv", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    summary, weekly = generate_reports(
        ReportInputs(
            output_dir=args.output_dir,
            experiment_json=args.experiment_json,
            metrics_summary_csv=args.metrics_summary_csv,
            environment_json=args.environment_json,
            runtime_json=args.runtime_json,
            inference_runtime_json=args.inference_runtime_json,
            gpu_summary_json=args.gpu_summary_json,
            data_validation_json=args.data_validation_json,
            official_validation_json=args.official_validation_json,
            preprocessing_artifacts_json=args.preprocessing_artifacts_json,
            official_status_json=args.official_status_json,
            official_summary_csv=args.official_summary_csv,
            evaluation_protocol_json=args.evaluation_protocol_json,
            failure_cases_csv=args.failure_cases_csv,
            figures_dir=args.figures_dir,
            figures_manifest_csv=args.figures_manifest_csv,
        )
    )
    print(
        json.dumps(
            {"summary": str(summary.resolve()), "weekly_discussion": str(weekly.resolve())},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
