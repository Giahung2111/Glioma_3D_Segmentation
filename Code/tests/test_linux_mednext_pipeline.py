from __future__ import annotations

import json
from pathlib import Path

import pytest

from glioma_seg.backends.mednext import backend
from glioma_seg.backends.mednext.config import EXPECTED_MEDNEXT_COMMIT
from glioma_seg.data.canonical_splits import (
    EXPECTED_CASE_COUNT,
    EXPECTED_VALIDATION_COUNTS,
    case_ids_from_labels,
    generate_canonical_splits,
)
from glioma_seg.reporting.model_runtime import sync_and_aggregate_mednext

ROOT = Path(__file__).resolve().parents[2]
SETUP_SCRIPT = ROOT / "Code" / "scripts" / "setup_research_models_env.sh"
PIPELINE_SCRIPT = ROOT / "Code" / "scripts" / "run_mednext_cv_pipeline.sh"


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_canonical_split_generator_is_deterministic_and_partitions_every_case() -> None:
    case_ids = [f"BraTS-GLI-{index:05d}-000" for index in range(EXPECTED_CASE_COUNT)]

    first = generate_canonical_splits(case_ids)
    second = generate_canonical_splits(reversed(case_ids))

    assert first == second
    assert tuple(len(item["val"]) for item in first) == EXPECTED_VALIDATION_COUNTS
    validation = [case_id for item in first for case_id in item["val"]]
    assert len(validation) == EXPECTED_CASE_COUNT
    assert set(validation) == set(case_ids)
    assert len(validation) == len(set(validation))
    for item in first:
        assert set(item["train"]).isdisjoint(item["val"])
        assert set(item["train"]) | set(item["val"]) == set(case_ids)


def test_case_inventory_requires_exact_converted_label_count(tmp_path: Path) -> None:
    labels = tmp_path / "labelsTr"
    labels.mkdir()
    for name in ("case_b.nii.gz", "case_a.nii.gz"):
        (labels / name).write_bytes(b"test")

    assert case_ids_from_labels(labels, expected_count=2) == ("case_a", "case_b")
    with pytest.raises(ValueError, match="Expected 3 converted labels"):
        case_ids_from_labels(labels, expected_count=3)


def test_upstream_audit_accepts_submodule_gitfile_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "MedNeXt"
    repository.mkdir()
    (repository / ".git").write_text("gitdir: ../.git/modules/MedNeXt\n", encoding="utf-8")

    def fake_git_output(_repository: Path, *arguments: str) -> str:
        if arguments == ("rev-parse", "--is-inside-work-tree"):
            return "true"
        if arguments == ("rev-parse", "HEAD"):
            return EXPECTED_MEDNEXT_COMMIT
        if arguments == ("status", "--porcelain", "--untracked-files=no"):
            return ""
        raise AssertionError(arguments)

    monkeypatch.setattr(backend, "_git_output", fake_git_output)

    result = backend._check_upstream(repository)

    assert result["commit"] == EXPECTED_MEDNEXT_COMMIT
    assert result["tracked_clean"] is True


def test_smoke_runtime_publisher_copies_fold_evidence(tmp_path: Path) -> None:
    experiment_id = "mednext_s_k3_smoke_test"
    report = tmp_path / "Workspace" / "reports" / experiment_id
    result_fold = tmp_path / "Workspace" / "model_results" / "mednext" / experiment_id / "fold_0"
    _write_json(
        report / "experiment.json",
        {
            "experiment_id": experiment_id,
            "backend": "mednext",
            "model_id": "mednext_v1_s_kernel3",
        },
    )
    _write_json(result_fold / "runtime.json", {"total_seconds": 10.0})
    _write_json(result_fold / "gpu_summary.json", {"samples": 5})
    _write_json(
        result_fold / "validation_summary.json",
        {
            "valid": True,
            "fold": 0,
            "case_count": 2,
            "case_ids": ["case_a", "case_b"],
            "inference_total_seconds": 4.0,
        },
    )

    result = sync_and_aggregate_mednext(
        tmp_path,
        experiment_id,
        folds=(0,),
        smoke=True,
    )

    assert result["case_count"] == 2
    inference = json.loads((report / "inference_runtime.json").read_text(encoding="utf-8"))
    assert inference["mean_seconds_per_case"] == pytest.approx(2.0)
    assert json.loads((report / "runtime.json").read_text())["total_seconds"] == 10.0
    assert json.loads((report / "gpu_summary.json").read_text())["samples"] == 5


def test_linux_scripts_are_fail_closed_and_do_not_use_sudo() -> None:
    setup = SETUP_SCRIPT.read_text(encoding="utf-8")
    pipeline = PIPELINE_SCRIPT.read_text(encoding="utf-8")

    assert "\nsudo " not in setup
    assert "\nsudo " not in pipeline
    assert 'EVALUATOR_PYTHON_VERSION="3.9.23"' in setup
    assert setup.count('"python=$EVALUATOR_PYTHON_VERSION"') == 2
    assert "3.9.25" not in setup
    assert "--smoke-test" in pipeline
    assert "glioma_seg.backends.mednext.smoke_gate" in pipeline
    assert "--stop-after-epoch 1" in pipeline
    assert "glioma_seg.evaluation.official_runner" in pipeline
    assert "glioma_seg.analysis.failure_statistics" in pipeline
    assert "glioma_seg.reporting.model_bundle" in pipeline
    assert "flock -n" in pipeline
    assert "Code/.venv-models" in pipeline
