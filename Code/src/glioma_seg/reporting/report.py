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
    crossval_summary_json: Path | None = None


@dataclass(frozen=True)
class ModelReportContext:
    """Manifest-derived model identity used to avoid backend-specific claims."""

    backend: str
    display_name: str
    framework: str
    probability_source_name: str
    is_nnunet: bool


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


def _first_text(mappings: Sequence[Mapping[str, Any]], *keys: str) -> str | None:
    for mapping in mappings:
        for key in keys:
            value = mapping.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _normalized_identifier(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _backend_display_name(backend: str) -> str:
    normalized = _normalized_identifier(backend)
    names = {
        "mednext": "MedNeXt",
        "mednextv1": "MedNeXt",
        "segresnet": "SegResNet",
        "monaisegresnet": "SegResNet",
        "nnunet": "nnU-Net v2",
        "nnunetv2": "nnU-Net v2",
    }
    return names.get(normalized, backend)


def _model_report_context(
    experiment: Mapping[str, Any], crossval: Mapping[str, Any]
) -> ModelReportContext:
    """Resolve model identity while keeping legacy manifests on the nnU-Net path."""

    mappings = (experiment, crossval)
    backend = _first_text(mappings, "backend", "backend_id")
    explicit_identity = _first_text(
        mappings,
        "model_display_name",
        "model_display",
        "model_name",
        "model",
    )
    framework = _first_text(mappings, "framework", "framework_name")

    if backend is not None:
        is_nnunet = _normalized_identifier(backend) in {"nnunet", "nnunetv2"}
    else:
        identity_tokens = " ".join(value for value in (explicit_identity, framework) if value)
        normalized_tokens = _normalized_identifier(identity_tokens)
        explicitly_other = "mednext" in normalized_tokens or "segresnet" in normalized_tokens
        # Historical manifests predate the backend field and are all nnU-Net.
        is_nnunet = not explicitly_other

    if is_nnunet:
        return ModelReportContext(
            backend=backend or "nnunetv2",
            display_name="nnU-Net v2",
            framework=framework or "nnU-Net v2",
            probability_source_name="nnU-Net",
            is_nnunet=True,
        )

    display_name = (
        explicit_identity
        or (_backend_display_name(backend) if backend is not None else None)
        or framework
        or "Segmentation model"
    )
    probability_source_name = _first_text(
        (crossval, experiment),
        "probability_source_model",
        "probability_source_name",
    )
    return ModelReportContext(
        backend=backend or _normalized_identifier(display_name),
        display_name=display_name,
        framework=framework or _backend_display_name(backend or display_name),
        probability_source_name=probability_source_name or display_name,
        is_nnunet=False,
    )


def _preprocessing_description(
    experiment: Mapping[str, Any], model: ModelReportContext
) -> str:
    if model.is_nnunet:
        return (
            "BraTS-provided preprocessing comprises co-registration, atlas alignment,\n"
            "approximately 1 mm isotropic sampling, and skull stripping. Local processing\n"
            "is limited to nnU-Net format conversion, automatic fingerprinting/planning,\n"
            "normalization, and model-specific preprocessing. No custom registration,\n"
            "skull stripping, or intensity enhancement is claimed here."
        )

    description = _first_text(
        (experiment,),
        "preprocessing_description",
        "preprocessing_text",
        "model_preprocessing_description",
    )
    preprocessing = experiment.get("preprocessing")
    if description is None and isinstance(preprocessing, str) and preprocessing.strip():
        description = preprocessing.strip()
    if description is None and isinstance(preprocessing, Mapping):
        description = _first_text(
            (preprocessing,), "description", "summary", "provenance"
        )
    if description is not None:
        return description
    return (
        "BraTS-provided aligned, skull-stripped NIfTI data were used. The experiment "
        "manifest does not record a backend-specific preprocessing description for "
        f"{model.display_name}; no additional registration, skull stripping, or intensity "
        "enhancement is claimed."
    )


def _reference_epoch_budget(experiment: Mapping[str, Any]) -> str | None:
    for key in ("reference_epochs", "standard_epochs", "original_recipe_epochs"):
        value = experiment.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return str(value)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _manifest_limitations(experiment: Mapping[str, Any]) -> list[str]:
    value = experiment.get("limitations")
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    if isinstance(value, list):
        return [item.strip() for item in value if isinstance(item, str) and item.strip()]
    return []


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


def _is_verified_full_crossval(crossval: Mapping[str, Any]) -> bool:
    folds = crossval.get("folds")
    return (
        crossval.get("valid") is True
        and folds == [0, 1, 2, 3, 4]
        and crossval.get("total_cases") == 1251
        and crossval.get("each_case_validated_once") is True
    )


def _report_is_full_crossval(
    experiment: Mapping[str, Any], crossval: Mapping[str, Any]
) -> bool:
    """Resolve report scope without silently downgrading a declared full CV run."""

    verified = _is_verified_full_crossval(crossval)
    if experiment.get("experiment_kind") == "fullcv" and not verified:
        raise ValueError(
            "A fullcv experiment requires a verified five-fold crossval_summary.json; "
            "refusing to generate a misleading preliminary/single-fold report"
        )
    if verified and experiment.get("experiment_kind") != "fullcv":
        raise ValueError(
            "A verified five-fold crossval_summary.json was supplied for an experiment "
            "that is not declared experiment_kind=fullcv"
        )
    return verified


def _verified_crossval_split_line(
    crossval: Mapping[str, Any], preprocessing: Mapping[str, Any]
) -> str:
    if not _is_verified_full_crossval(crossval):
        return "NOT AVAILABLE - the five-fold out-of-fold inventory was not fully verified"
    source = crossval.get("split_source")
    details = preprocessing.get("details")
    audited_source = details.get("splits_file") if isinstance(details, Mapping) else None
    if (
        preprocessing.get("valid") is not True
        or not isinstance(source, str)
        or not source.strip()
        or not isinstance(audited_source, str)
        or not audited_source.strip()
        or os.path.normcase(os.path.abspath(source))
        != os.path.normcase(os.path.abspath(audited_source))
    ):
        return "NOT AVAILABLE - cross-validation and preprocessing split evidence disagree"
    counts = crossval.get("validation_case_counts")
    rendered_counts = (
        ", ".join(str(value) for value in counts)
        if isinstance(counts, list)
        else str(counts)
    )
    return (
        f"source={source}; folds=0,1,2,3,4; validation counts={rendered_counts}; "
        "all 1,251 cases appear in validation exactly once (verified)"
    )


def _crossval_metrics_table(crossval: Mapping[str, Any]) -> str:
    if not _is_verified_full_crossval(crossval):
        return "Per-fold metric table NOT AVAILABLE - full CV evidence was not verified."
    per_fold = crossval.get("per_fold")
    if not isinstance(per_fold, list) or len(per_fold) != 5:
        return "Per-fold metric table NOT AVAILABLE - five fold summaries were not recorded."

    def value(entry: Mapping[str, Any], metric: str, region: str) -> str:
        metrics = entry.get("metrics")
        metric_values = metrics.get(metric) if isinstance(metrics, Mapping) else None
        raw = metric_values.get(region) if isinstance(metric_values, Mapping) else None
        if raw is None:
            return "undefined"
        try:
            return f"{float(raw):.4f}"
        except (TypeError, ValueError):
            return "undefined"

    lines = [
        "| Fold | Cases | Dice ET | Dice TC | Dice WT | "
        "HD95 ET (mm) | HD95 TC (mm) | HD95 WT (mm) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for entry in per_fold:
        if not isinstance(entry, Mapping):
            raise ValueError("crossval_summary per_fold entries must be JSON objects")
        lines.append(
            "| {fold} | {cases} | {dice_et} | {dice_tc} | {dice_wt} | "
            "{hd_et} | {hd_tc} | {hd_wt} |".format(
                fold=entry.get("fold", "?"),
                cases=entry.get("case_count", "?"),
                dice_et=value(entry, "Dice", "ET"),
                dice_tc=value(entry, "Dice", "TC"),
                dice_wt=value(entry, "Dice", "WT"),
                hd_et=value(entry, "HD95", "ET"),
                hd_tc=value(entry, "HD95", "TC"),
                hd_wt=value(entry, "HD95", "WT"),
            )
        )
    return "\n".join(lines)


def _probability_provenance(
    crossval: Mapping[str, Any], model: ModelReportContext | None = None
) -> str:
    if not _is_verified_full_crossval(crossval):
        return "NOT AVAILABLE - full-CV probability inventory was not verified"
    retained = crossval.get("probabilities_retained")
    source = crossval.get("probability_source_channel_order")
    canonical = crossval.get("probability_canonical_order")
    if model is None or model.is_nnunet:
        if (
            retained is not True
            or source != ["WT", "TC", "ET"]
            or canonical != ["ET", "TC", "WT"]
        ):
            return "NOT AVAILABLE - probability channel provenance is incomplete or inconsistent"
        return (
            "retained for every out-of-fold case; nnU-Net source order WT,TC,ET; "
            "project ensemble order ET,TC,WT requires explicit reordering [2,1,0]"
        )
    if (
        retained is not True
        or not isinstance(source, list)
        or not isinstance(canonical, list)
        or not source
        or any(not isinstance(channel, str) or not channel for channel in (*source, *canonical))
        or len(set(source)) != len(source)
        or len(set(canonical)) != len(canonical)
        or canonical != ["ET", "TC", "WT"]
    ):
        return "NOT AVAILABLE - probability channel provenance is incomplete or inconsistent"
    source_order = ",".join(source)
    canonical_order = ",".join(canonical)
    if len(source) == len(canonical) and set(source) == set(canonical):
        reorder_indices = [source.index(channel) for channel in canonical]
        if reorder_indices == list(range(len(source))):
            conversion = f"already canonical; no reordering required {reorder_indices}"
        else:
            conversion = f"requires explicit reordering {reorder_indices}"
    else:
        recorded_conversion = crossval.get("probability_conversion")
        if not isinstance(recorded_conversion, str) or not recorded_conversion.strip():
            return (
                "NOT AVAILABLE - non-permutation probability conversion provenance "
                "was not recorded"
            )
        conversion = f"uses recorded conversion: {recorded_conversion.strip()}"
    return (
        f"retained for every out-of-fold case; {model.probability_source_name} source order "
        f"{source_order}; project ensemble order {canonical_order} {conversion}"
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
    output_dir = output_dir.resolve()
    if figures_dir is not None:
        figures_dir = figures_dir.resolve()
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
        candidate_values: list[str | Path] = []
        row_value = row.get("figure") or row.get("figure_path")
        if row_value:
            candidate_values.append(row_value)
        if figure_paths is not None and figure_paths.get(case_id) is not None:
            candidate_values.append(figure_paths[case_id])
        if figures_dir is not None:
            candidate_values.append(figures_dir / f"{case_id}_t1c_flair_gt_pred.png")
        for figure_value in candidate_values:
            figure_path = Path(figure_value)
            if not figure_path.is_absolute() and figures_dir is not None:
                figure_path = figures_dir / figure_path
            if not figure_path.is_file():
                continue
            try:
                relative = figure_path.resolve().relative_to(output_dir.resolve())
                figure_cell = f"[{figure_path.name}]({relative.as_posix()})"
            except ValueError:
                figure_cell = str(figure_path.resolve())
            break
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
                "Training time (h)",
                _first_recorded(training_sources, "total_hours"),
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


def _report_title(
    experiment: Mapping[str, Any], model: ModelReportContext, *, full_crossval: bool
) -> str:
    if model.is_nnunet:
        if full_crossval:
            return "5-Fold nnU-Net v2 Cross-Validation — BraTS 2023 GLI"
        return "Preliminary nnU-Net v2 Baseline — BraTS 2023 GLI"
    explicit = _first_text((experiment,), "report_title", "title")
    if explicit is not None:
        return explicit
    if full_crossval:
        return f"5-Fold {model.display_name} Cross-Validation — BraTS 2023 GLI"
    return f"Preliminary {model.display_name} Baseline — BraTS 2023 GLI"


def _report_objective(
    experiment: Mapping[str, Any],
    model: ModelReportContext,
    recorded_epochs: str,
    *,
    full_crossval: bool,
) -> str:
    if model.is_nnunet:
        if full_crossval:
            return (
                "Measure a reproducible 100-epoch compute-limited nnU-Net experiment with "
                "complete five-fold out-of-fold evaluation."
            )
        return (
            "Establish a reproducible first baseline and validate the complete "
            "training/evaluation pipeline."
        )
    explicit = _first_text((experiment,), "report_objective", "objective")
    if explicit is not None:
        return explicit
    if full_crossval:
        return (
            f"Measure a reproducible {recorded_epochs}-epoch-per-fold "
            f"{model.display_name} experiment with complete five-fold out-of-fold evaluation."
        )
    return (
        f"Establish a reproducible {model.display_name} baseline and validate its complete "
        "training/evaluation pipeline."
    )


def _generic_fullcv_limitations(
    experiment: Mapping[str, Any],
    crossval: Mapping[str, Any],
    model: ModelReportContext,
    recorded_epochs: str,
) -> str:
    reference_epochs = _reference_epoch_budget(experiment)
    classification = _first_text(
        (experiment,), "baseline_classification", "classification", "experiment_classification"
    )
    if reference_epochs is not None:
        budget_detail = (
            f"This recorded run is not the {reference_epochs}-epoch reference recipe for "
            f"{model.display_name}."
        )
    elif classification is not None:
        budget_detail = (
            f"The manifest classifies it as {classification.replace('_', ' ')}; it is not "
            "presented as an unqualified reference recipe."
        )
    else:
        budget_detail = "Interpretation is limited to the recorded training budget."
    probability = _probability_provenance(crossval, model)
    lines = [
        "- This experiment evaluates all 1,251 training cases out-of-fold exactly once.",
        f"- The epoch budget is {recorded_epochs} per fold. {budget_detail}",
        f"- Probability provenance: {probability}.",
        "- Official BraTS validation ground truth is not public and cannot supply local Dice/HD95.",
        "- External-hospital validation and cross-model comparisons remain future work.",
    ]
    lines.extend(f"- {item.lstrip('- ').strip()}" for item in _manifest_limitations(experiment))
    lines.append(
        "- Any missing field is marked NOT AVAILABLE instead of being replaced by an estimate."
    )
    return "\n".join(lines)


def _generic_preliminary_limitations(
    experiment: Mapping[str, Any],
    model: ModelReportContext,
    recorded_fold: str,
    recorded_epochs: str,
) -> str:
    lines = [
        f"- This is a preliminary single-fold {model.display_name} result "
        f"(recorded fold: {recorded_fold}), not a completed five-fold reproduction.",
        f"- The recorded epoch budget is {recorded_epochs}; interpretation is limited to "
        "that recorded budget.",
        "- Full 5-fold CV, ensemble evaluation, and external-hospital validation are outside "
        "this preliminary baseline.",
        "- Official BraTS validation ground truth is not public and therefore cannot supply "
        "local Dice/HD95.",
        "- Metric-prediction TTA and the separate primary timing-inference TTA are reported "
        "independently.",
    ]
    lines.extend(f"- {item.lstrip('- ').strip()}" for item in _manifest_limitations(experiment))
    lines.append(
        "- Any missing field above is explicitly marked NOT AVAILABLE rather than replaced "
        "by an estimate."
    )
    return "\n".join(lines)


def _model_table_rows(
    experiment: Mapping[str, Any],
    environment: Mapping[str, Any],
    model: ModelReportContext,
    fold_value: str,
    epoch_rows: Sequence[tuple[str, Any]],
) -> list[tuple[str, Any]]:
    if model.is_nnunet:
        return [
            (
                "Framework",
                _first_recorded(
                    ((experiment, "experiment.json"),), "framework", "model"
                ),
            ),
            (
                "nnU-Net version",
                _first_recorded(
                    ((environment, "environment.json"), (experiment, "experiment.json")),
                    "nnUNet_version",
                    "nnunet_version",
                ),
            ),
            ("Configuration", _recorded(experiment, "configuration", "experiment.json")),
            ("Fold(s)", fold_value),
            ("Trainer", _recorded(experiment, "trainer", "experiment.json")),
            *epoch_rows,
            ("Architecture", _recorded(experiment, "architecture", "experiment.json")),
        ]
    sources = ((experiment, "experiment.json"), (environment, "environment.json"))
    return [
        ("Model", model.display_name),
        ("Framework", model.framework),
        (
            "Framework version",
            _first_recorded(
                sources,
                "framework_version",
                "model_version",
                "monai_version",
                "version",
            ),
        ),
        ("Configuration", _recorded(experiment, "configuration", "experiment.json")),
        ("Fold(s)", fold_value),
        ("Trainer", _recorded(experiment, "trainer", "experiment.json")),
        *epoch_rows,
        ("Architecture", _recorded(experiment, "architecture", "experiment.json")),
    ]


def _training_table_rows(
    experiment: Mapping[str, Any],
    crossval: Mapping[str, Any],
    model: ModelReportContext,
    split_line: str,
) -> list[tuple[str, Any]]:
    if model.is_nnunet:
        return [
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
            ("Split/seed", split_line),
            ("Saved probability evidence", _probability_provenance(crossval, model)),
        ]
    sources = ((experiment, "experiment.json"),)
    return [
        (
            "Patch/crop size",
            _first_recorded(sources, "patch_size", "crop_size", "roi_size"),
        ),
        ("Batch size", _recorded(experiment, "batch_size", "experiment.json")),
        ("Target spacing", _recorded(experiment, "target_spacing", "experiment.json")),
        (
            "Data-loader workers",
            _first_recorded(sources, "num_workers", "data_loader_workers", "workers"),
        ),
        (
            "Primary timing inference TTA state",
            _first_recorded(sources, "TTA_state", "tta_state", "inference_tta"),
        ),
        ("Split/seed", split_line),
        ("Saved probability evidence", _probability_provenance(crossval, model)),
    ]


def generate_summary_report(inputs: ReportInputs) -> Path:
    """Write summary.md, refusing to proceed without real semantic metrics."""

    experiment, environment, runtime, inference_runtime, gpu, validation = _load_all(inputs)
    official_validation = _load_json(inputs.official_validation_json)
    preprocessing_artifacts = _load_json(inputs.preprocessing_artifacts_json)
    semantic_rows = load_metric_summary(inputs.metrics_summary_csv)
    failures = _load_failures(inputs.failure_cases_csv)
    figure_paths = _load_figure_paths(inputs.figures_manifest_csv)
    evaluation = _load_json(inputs.evaluation_protocol_json)
    crossval = _load_json(inputs.crossval_summary_json)
    output_dir = Path(inputs.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    recorded_fold = _recorded(experiment, "fold", "experiment.json")
    recorded_epochs = _recorded(experiment, "epochs", "experiment.json")

    full_crossval = _report_is_full_crossval(experiment, crossval)
    model = _model_report_context(experiment, crossval)
    report_title = _report_title(experiment, model, full_crossval=full_crossval)
    objective = _report_objective(
        experiment,
        model,
        recorded_epochs,
        full_crossval=full_crossval,
    )
    preprocessing_text = _preprocessing_description(experiment, model)
    if full_crossval:
        fold_value = "0, 1, 2, 3, 4"
        split_line = _verified_crossval_split_line(crossval, preprocessing_artifacts)
        result_heading = "Five-Fold Cross-Validation Results"
        failure_heading = "Cross-Validation Failure Analysis"
        per_fold_section = (
            "### Per-fold semantic metrics\n\n"
            + _crossval_metrics_table(crossval)
            + "\n\n### Pooled out-of-fold semantic metrics"
        )
        if model.is_nnunet:
            limitations = f"""- This experiment evaluates all 1,251 training cases out-of-fold
  exactly once.
- The epoch budget is {recorded_epochs} per fold. This is a compute-limited
  cross-validation experiment, not the standard 1,000-epoch nnU-Net reference baseline.
- Stored validation probabilities follow nnU-Net channel order WT,TC,ET and must be
  explicitly reordered before project region-wise ensemble experiments.
- Official BraTS validation ground truth is not public and cannot supply local Dice/HD95.
- External-hospital validation and comparisons with MedNeXt/SegResNet remain future work.
- Any missing field is marked NOT AVAILABLE instead of being replaced by an estimate."""
            next_steps = (
                "Use this same split and evaluation protocol for MedNeXt and SegResNet, compare "
                "ET/TC/WT and lesion-wise behavior, then test evidence-driven checkpoint/model "
                "selection, probability ensembles, post-processing, and later external validation."
            )
        else:
            limitations = _generic_fullcv_limitations(
                experiment, crossval, model, recorded_epochs
            )
            next_steps = _first_text(
                (experiment,), "report_next_steps", "next_steps"
            ) or (
                f"Compare {model.display_name} with the other registered same-split baselines, "
                "analyze ET/TC/WT and lesion-wise behavior, then test evidence-driven "
                "checkpoint/model selection, probability ensembles, post-processing, and "
                "later external validation."
            )
    else:
        fold_value = recorded_fold
        split_line = _verified_split_line(experiment, preprocessing_artifacts)
        result_heading = "Preliminary Results"
        failure_heading = "Preliminary Failure Analysis"
        per_fold_section = "### Standard semantic region-wise metrics"
        if model.is_nnunet:
            limitations = f"""- This is a preliminary single-fold result
  (recorded fold: {recorded_fold}),
  not a completed five-fold nnU-Net reproduction.
- The recorded epoch budget is {recorded_epochs}; shortened 20/50-epoch runs are
  pipeline-validation experiments, not final baselines.
- Full 5-fold CV, ensemble evaluation, and external-hospital validation are outside
  this preliminary baseline.
- Official BraTS validation ground truth is not public and therefore cannot supply local Dice/HD95.
- Metric-prediction TTA and the separate primary timing-inference TTA are reported independently.
- Any missing field above is explicitly marked NOT AVAILABLE rather than replaced by an estimate."""
            next_steps = (
                "Complete the full five-fold nnU-Net baseline, add a same-split MedNeXt baseline, "
                "compare ET/TC/WT behavior, select research directions from observed failures, "
                "and only then test boundary-aware or opt-in ensemble experiments. External "
                "hospital evaluation remains a later, separately governed stage."
            )
        else:
            limitations = _generic_preliminary_limitations(
                experiment, model, recorded_fold, recorded_epochs
            )
            next_steps = _first_text(
                (experiment,), "report_next_steps", "next_steps"
            ) or (
                f"Complete the full five-fold {model.display_name} baseline, compare it under "
                "the shared ET/TC/WT protocol, and select later ensemble or post-processing "
                "experiments from observed failures."
            )

    hardware_sources = ((environment, "environment.json"), (experiment, "experiment.json"))
    dataset_count = _first_recorded(
        ((validation, "data_validation.json"), (experiment, "experiment.json")),
        "validated_case_count",
        "valid_case_count",
        "actual_case_count",
        "dataset_case_count",
    )
    epoch_rows: list[tuple[str, Any]]
    if full_crossval:
        epoch_rows = [
            ("Epochs per fold", recorded_epochs),
            (
                "Total trained epochs",
                _first_recorded(
                    ((experiment, "experiment.json"), (runtime, "runtime.json")),
                    "total_training_epochs",
                    "number_of_epochs",
                ),
            ),
        ]
    else:
        epoch_rows = [("Epochs", recorded_epochs)]

    summary = f"""# {report_title}

## 1. Experiment Objective

{objective}

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

{preprocessing_text}

## 4. Model

{
        key_value_markdown_table(
            _model_table_rows(
                experiment,
                environment,
                model,
                fold_value,
                epoch_rows,
            )
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
            _training_table_rows(
                experiment,
                crossval,
                model,
                split_line,
            )
        )
    }

## 7. {result_heading}

{per_fold_section}

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

## 9. {failure_heading}

{_failure_table(failures, output_dir, inputs.figures_dir, figure_paths)}

Classifications describe observed mask/metric patterns and operational
thresholds; they are not asserted medical causes.

## 10. Limitations

{limitations}

## 11. Next Steps

{next_steps}
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
    crossval = _load_json(inputs.crossval_summary_json)
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
    full_crossval = _report_is_full_crossval(experiment, crossval)
    model = _model_report_context(experiment, crossval)
    preprocessing_text = " ".join(_preprocessing_description(experiment, model).split())
    if full_crossval:
        if model.is_nnunet:
            discussion_title = "Weekly Discussion — 5-Fold BraTS 2023 GLI Cross-Validation"
            scope_goal = (
                "The immediate result is a complete 100-epoch compute-limited five-fold "
                "nnU-Net comparison reference; it is not the 1,000-epoch standard baseline."
            )
        else:
            discussion_title = _first_text(
                (experiment,), "discussion_title", "weekly_discussion_title"
            ) or (
                f"Weekly Discussion — 5-Fold {model.display_name} BraTS 2023 GLI "
                "Cross-Validation"
            )
            reference_epochs = _reference_epoch_budget(experiment)
            reference_clause = (
                f" It is not the {reference_epochs}-epoch reference recipe."
                if reference_epochs is not None
                else ""
            )
            scope_goal = (
                f"The immediate result is a complete {epochs}-epoch-per-fold five-fold "
                f"{model.display_name} comparison under the shared protocol."
                f"{reference_clause}"
            )
        fold_value = "0, 1, 2, 3, 4"
        total_epochs = _first_recorded(
            ((experiment, "experiment.json"), (runtime, "runtime.json")),
            "total_training_epochs",
            "number_of_epochs",
        )
        trainer_epoch_value = f"{trainer} / {epochs} per fold ({total_epochs} total)"
        split_line = _verified_crossval_split_line(crossval, preprocessing_artifacts)
        result_label = "Five-Fold Results"
        per_fold_table = _crossval_metrics_table(crossval) + "\n\n### Pooled out-of-fold metrics"
        if model.is_nnunet:
            future_flow = (
                "Same-split MedNeXt/SegResNet → ET/TC/WT and lesion-wise comparison → "
                "failure-driven checkpoint/model selection → explicitly reordered probability "
                "ensemble experiments → post-processing → official validation submission → "
                "later hospital external validation."
            )
        else:
            future_flow = _first_text(
                (experiment,), "future_flow", "report_future_flow"
            ) or (
                f"Compare {model.display_name} with other same-split registered baselines → "
                "ET/TC/WT and lesion-wise analysis → failure-driven checkpoint/model "
                "selection → provenance-checked probability ensemble experiments → "
                "post-processing → official validation submission → later hospital external "
                "validation."
            )
    else:
        if model.is_nnunet:
            discussion_title = "Weekly Discussion — Preliminary BraTS 2023 GLI Baseline"
            scope_goal = (
                "The immediate goal is a reproducible standard nnU-Net reference before "
                "proposing a method."
            )
        else:
            discussion_title = _first_text(
                (experiment,), "discussion_title", "weekly_discussion_title"
            ) or f"Weekly Discussion — Preliminary {model.display_name} BraTS 2023 GLI Baseline"
            scope_goal = (
                f"The immediate goal is a reproducible {model.display_name} reference under "
                "the shared evaluation protocol."
            )
        fold_value = _recorded(experiment, "fold", "experiment.json")
        trainer_epoch_value = f"{trainer} / {epochs}"
        split_line = _verified_split_line(experiment, preprocessing_artifacts)
        result_label = "Preliminary Results"
        per_fold_table = ""
        if model.is_nnunet:
            future_flow = (
                "BraTS 2023 → nnU-Net full 5-fold baseline → same-fold MedNeXt baseline → "
                "ET/TC/WT comparison → failure analysis → evidence-driven boundary/small-lesion "
                "investigation → opt-in architecture/checkpoint/region ensemble experiments with "
                "ET ⊆ TC ⊆ WT enforced → later external-hospital evaluation."
            )
        else:
            future_flow = _first_text(
                (experiment,), "future_flow", "report_future_flow"
            ) or (
                f"BraTS 2023 → {model.display_name} full 5-fold baseline → same-split model "
                "comparison → ET/TC/WT analysis → failure analysis → provenance-checked "
                "ensemble experiments → later external-hospital evaluation."
            )

    if model.is_nnunet:
        preprocessing_bullet = "- nnU-Net format conversion and automatic planning/preprocessing"
        architecture_bullet = (
            "- Architecture from actual plans: "
            + _recorded(experiment, "architecture", "experiment.json")
        )
    else:
        preprocessing_bullet = f"- {preprocessing_text}"
        architecture_bullet = (
            "- Architecture: " + _recorded(experiment, "architecture", "experiment.json")
        )
    discussion = f"""# {discussion_title}

## Slide 1 — Research Scope

3D multimodal segmentation of pre-treatment adult glioma. {scope_goal}

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
{preprocessing_bullet}
- Configuration: {_recorded(experiment, "configuration", "experiment.json")}
- Fold(s): {fold_value}
- Split/seed: {split_line}
- Trainer / epochs: {trainer_epoch_value}
{architecture_bullet}

## Slide 4 — {result_label}

{per_fold_table}

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

{future_flow}
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
    parser.add_argument("--crossval-summary-json", type=Path)
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
            crossval_summary_json=args.crossval_summary_json,
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
