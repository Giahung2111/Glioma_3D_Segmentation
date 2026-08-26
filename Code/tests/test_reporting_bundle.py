from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from glioma_seg.reporting.bundle import ReportBundleError, complete_report_bundle
from glioma_seg.reporting.report import ReportInputs, generate_reports
from glioma_seg.utils.hashing import sha256_file


def _write_validation_pair(root: Path, prefix: str, kind: str, count: int) -> None:
    json_path = root / f"{prefix}data_validation.json"
    csv_path = root / f"{prefix}data_validation.csv"
    json_path.write_text(
        json.dumps(
            {
                "valid": True,
                "dataset_kind": kind,
                "expected_case_count": count,
                "actual_case_count": count,
                "valid_case_count": count,
                "errors": [],
            }
        ),
        encoding="utf-8",
    )
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("record_type", "dataset_kind", "status"))
        writer.writeheader()
        writer.writerow({"record_type": "dataset_summary", "dataset_kind": kind, "status": "PASS"})
        for _ in range(count):
            writer.writerow({"record_type": "case", "dataset_kind": kind, "status": "PASS"})


def _make_bundle_tree(tmp_path: Path) -> tuple[Path, Path]:
    reports = tmp_path / "reports"
    experiment = reports / "experiment_a"
    (experiment / "logs").mkdir(parents=True)
    (experiment / "experiment.json").write_text(
        json.dumps({"experiment_id": "experiment_a", "artifacts": {"kept": "value"}}),
        encoding="utf-8",
    )
    _write_validation_pair(reports, "", "training", 2)
    (experiment / "logs" / "predict.log").write_text("predict\n", encoding="utf-8")
    (experiment / "logs" / "train.log").write_text("train\n", encoding="utf-8")
    (experiment / "logs" / "benchmark.log").write_text("benchmark\n", encoding="utf-8")
    (experiment / "logs" / "plan_and_preprocess.log").write_text("plan\n", encoding="utf-8")
    return reports, experiment


def _write_successful_official_artifacts(experiment: Path) -> None:
    (experiment / "official_brats_metrics_status.json").write_text(
        json.dumps({"available": True, "case_count": 2}), encoding="utf-8"
    )
    (experiment / "official_lesionwise_metrics_summary.csv").write_text(
        "metric,ET,TC,WT\nDice,1,1,1\nHD95,0,0,0\n", encoding="utf-8"
    )
    (experiment / "official_lesionwise_metrics_summary.json").write_text(
        json.dumps({"case_count": 2}), encoding="utf-8"
    )
    (experiment / "official_lesionwise_metrics_per_case.csv").write_text(
        "case_id,dice_et\na,1\nb,1\n", encoding="utf-8"
    )


def test_prepare_materializes_validation_aliases_and_deterministic_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reports, experiment = _make_bundle_tree(tmp_path)
    _write_validation_pair(reports, "official_validation_", "validation", 1)
    _write_successful_official_artifacts(experiment)
    (experiment / "official_brats_evaluator.log").write_text("official\n", encoding="utf-8")

    def unavailable_hardlink(_source: Path, _destination: Path) -> None:
        raise OSError("cross-device test")

    monkeypatch.setattr("glioma_seg.reporting.bundle.os.link", unavailable_hardlink)
    result = complete_report_bundle(
        workspace_reports=reports, experiment_dir=experiment, phase="prepare"
    )

    assert result.official_alias_warning is None
    assert all(item.method == "copy" for item in result.official_aliases)
    for name in (
        "data_validation.json",
        "data_validation.csv",
        "official_validation_data_validation.json",
        "official_validation_data_validation.csv",
    ):
        assert (experiment / name).read_bytes() == (reports / name).read_bytes()
    for canonical, alias in (
        ("official_lesionwise_metrics_summary.csv", "official_brats_metrics_summary.csv"),
        ("official_lesionwise_metrics_summary.json", "official_brats_metrics_summary.json"),
        ("official_lesionwise_metrics_per_case.csv", "official_brats_metrics_per_case.csv"),
    ):
        assert (experiment / canonical).read_bytes() == (experiment / alias).read_bytes()

    pipeline = (experiment / "pipeline.log").read_text(encoding="utf-8")
    expected_order = (
        "logs/plan_and_preprocess.log",
        "logs/benchmark.log",
        "logs/train.log",
        "logs/predict.log",
        "official_brats_evaluator.log",
    )
    assert tuple(record["path"] for record in result.pipeline_sources) == expected_order
    positions = [pipeline.index(f"path: {name}") for name in expected_order]
    assert positions == sorted(positions)
    for record in result.pipeline_sources:
        assert f"sha256: {record['sha256']}" in pipeline
    first_bytes = (experiment / "pipeline.log").read_bytes()
    complete_report_bundle(workspace_reports=reports, experiment_dir=experiment, phase="prepare")
    assert (experiment / "pipeline.log").read_bytes() == first_bytes
    assert not list(experiment.glob("*.tmp"))


def test_required_training_validation_is_fail_closed(tmp_path: Path) -> None:
    reports, experiment = _make_bundle_tree(tmp_path)
    (reports / "data_validation.csv").unlink()

    with pytest.raises(ReportBundleError, match="Required global training-validation"):
        complete_report_bundle(
            workspace_reports=reports, experiment_dir=experiment, phase="prepare"
        )

    assert not (experiment / "pipeline.log").exists()


def test_divergent_official_alias_is_optional_and_never_overwritten(tmp_path: Path) -> None:
    reports, experiment = _make_bundle_tree(tmp_path)
    _write_successful_official_artifacts(experiment)
    divergent = experiment / "official_brats_metrics_summary.csv"
    divergent.write_text("divergent\n", encoding="utf-8")

    result = complete_report_bundle(
        workspace_reports=reports, experiment_dir=experiment, phase="prepare"
    )

    assert result.official_alias_warning is not None
    assert "Refusing divergent artifact" in result.official_alias_warning
    assert divergent.read_text(encoding="utf-8") == "divergent\n"
    assert (experiment / "pipeline.log").is_file()
    (experiment / "metrics_summary.csv").write_text(
        "metric,ET,TC,WT\nDice,1,1,1\nHD95,0,0,0\n", encoding="utf-8"
    )
    (experiment / "summary.md").write_text("summary\n", encoding="utf-8")
    (experiment / "weekly_discussion.md").write_text("weekly\n", encoding="utf-8")

    finalized = complete_report_bundle(
        workspace_reports=reports, experiment_dir=experiment, phase="finalize"
    )

    manifest = json.loads((experiment / "experiment.json").read_text(encoding="utf-8"))
    assert finalized.official_alias_warning is not None
    assert "official_brats_metrics_summary.csv" not in manifest["final_artifacts"]
    assert "official_brats_metrics_summary" not in manifest["artifacts"]


def test_finalize_records_every_experiment_local_artifact_with_hashes(tmp_path: Path) -> None:
    reports, experiment = _make_bundle_tree(tmp_path)
    for name, content in (
        ("metrics_summary.csv", "metric,ET,TC,WT\nDice,1,1,1\nHD95,0,0,0\n"),
        ("summary.md", "summary\n"),
        ("weekly_discussion.md", "weekly\n"),
    ):
        (experiment / name).write_text(content, encoding="utf-8")
    (experiment / "figures").mkdir()
    (experiment / "figures" / "case.png").write_bytes(b"png")

    result = complete_report_bundle(
        workspace_reports=reports, experiment_dir=experiment, phase="finalize"
    )

    manifest = json.loads((experiment / "experiment.json").read_text(encoding="utf-8"))
    inventory = manifest["final_artifacts"]
    expected = {
        path.relative_to(experiment).as_posix()
        for path in experiment.rglob("*")
        if path.is_file() and path.name != "experiment.json"
    }
    assert set(inventory) == expected
    assert result.recorded_artifact_count == len(expected)
    for relative, record in inventory.items():
        path = experiment / relative
        assert record["path"] == str(path.resolve())
        assert record["size_bytes"] == path.stat().st_size
        assert record["sha256"] == sha256_file(path)
    assert manifest["artifacts"]["kept"] == "value"
    assert manifest["artifacts"]["report_pipeline_log"] == str(
        (experiment / "pipeline.log").resolve()
    )
    assert manifest["artifacts"]["summary"] == str((experiment / "summary.md").resolve())
    assert manifest["artifacts"]["weekly_discussion"] == str(
        (experiment / "weekly_discussion.md").resolve()
    )


def test_report_uses_verified_split_validation_and_telemetry_evidence(tmp_path: Path) -> None:
    output = tmp_path / "report"
    output.mkdir()
    split_source = tmp_path / "splits_final.json"
    experiment = output / "experiment.json"
    experiment.write_text(
        json.dumps(
            {
                "experiment_id": "evidence-test",
                "dataset": "Dataset501_BraTS2023GLI",
                "dataset_id": 501,
                "fold": 0,
                "epochs": 50,
                "split": {
                    "source": str(split_source),
                    "fold": 0,
                    "train_cases": 1000,
                    "validation_cases": 251,
                },
            }
        ),
        encoding="utf-8",
    )
    preprocessing = output / "preprocessing_artifacts.json"
    preprocessing.write_text(
        json.dumps(
            {
                "valid": True,
                "checks": [
                    {
                        "name": "official five-fold split",
                        "ok": True,
                        "detail": (
                            "official seed=12345, fold_sizes=[(1000, 251), (1001, 250), "
                            "(1001, 250), (1001, 250), (1001, 250)]"
                        ),
                    }
                ],
                "details": {"splits_created": True, "splits_file": str(split_source)},
            }
        ),
        encoding="utf-8",
    )
    official_validation = output / "official_validation_data_validation.json"
    official_validation.write_text(
        json.dumps(
            {
                "valid": True,
                "dataset_kind": "validation",
                "expected_case_count": 219,
                "actual_case_count": 219,
                "valid_case_count": 219,
                "errors": [],
            }
        ),
        encoding="utf-8",
    )
    metrics = output / "metrics_summary.csv"
    metrics.write_text(
        "metric,ET,TC,WT,total_cases\nDice,0.8,0.8,0.9,251\nHD95,2,3,4,251\n",
        encoding="utf-8",
    )
    runtime = output / "runtime.json"
    runtime.write_text(
        json.dumps(
            {
                "epoch_seconds_min": 103.85,
                "epoch_seconds_median": 117.775,
                "epoch_seconds_max": 138.4,
            }
        ),
        encoding="utf-8",
    )
    gpu = output / "gpu_summary.json"
    gpu.write_text(
        json.dumps({"peak_temperature_c": 88.0, "mean_power_w": 187.6385}),
        encoding="utf-8",
    )

    summary, weekly = generate_reports(
        ReportInputs(
            output_dir=output,
            experiment_json=experiment,
            metrics_summary_csv=metrics,
            runtime_json=runtime,
            gpu_summary_json=gpu,
            official_validation_json=official_validation,
            preprocessing_artifacts_json=preprocessing,
        )
    )

    for report in (summary, weekly):
        text = report.read_text(encoding="utf-8")
        assert "219 (modalities only; no public GT; never locally scored)" in text
        assert "fold=0; train=1000; validation=251; official seed=12345 (verified)" in text
        assert "88.0" in text
        assert "187.6385" in text
        assert "103.85" in text
        assert "117.775" in text
        assert "138.4" in text


def test_report_does_not_invent_seed_when_split_audit_disagrees(tmp_path: Path) -> None:
    output = tmp_path / "report"
    output.mkdir()
    split_source = tmp_path / "splits_final.json"
    experiment = output / "experiment.json"
    experiment.write_text(
        json.dumps(
            {
                "experiment_id": "bad-split",
                "split": {
                    "source": str(split_source),
                    "fold": 0,
                    "train_cases": 999,
                    "validation_cases": 252,
                },
            }
        ),
        encoding="utf-8",
    )
    preprocessing = output / "preprocessing_artifacts.json"
    preprocessing.write_text(
        json.dumps(
            {
                "valid": True,
                "checks": [
                    {
                        "name": "official five-fold split",
                        "ok": True,
                        "detail": (
                            "official seed=12345, fold_sizes=[(1000, 251), (1001, 250), "
                            "(1001, 250), (1001, 250), (1001, 250)]"
                        ),
                    }
                ],
                "details": {"splits_created": True, "splits_file": str(split_source)},
            }
        ),
        encoding="utf-8",
    )
    metrics = output / "metrics_summary.csv"
    metrics.write_text("metric,ET,TC,WT\nDice,0.8,0.8,0.9\nHD95,2,3,4\n", encoding="utf-8")

    summary, _ = generate_reports(
        ReportInputs(
            output_dir=output,
            experiment_json=experiment,
            metrics_summary_csv=metrics,
            preprocessing_artifacts_json=preprocessing,
        )
    )

    text = summary.read_text(encoding="utf-8")
    assert "structured split counts disagree" in text
    assert "official seed=12345 (verified)" not in text
