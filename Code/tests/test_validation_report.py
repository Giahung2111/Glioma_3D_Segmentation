import json
import time
from pathlib import Path

import pytest

from glioma_seg.data import validate as validation
from glioma_seg.data.brats2023 import BraTSCase
from glioma_seg.data.discover import DiscoveredDataset
from glioma_seg.data.validate import CaseValidation, ValidationReport


def test_validation_report_recomputes_derived_validity(tmp_path: Path) -> None:
    report = ValidationReport(
        schema_version=1,
        created_at_utc="2026-01-01T00:00:00+00:00",
        dataset_root=str(tmp_path),
        dataset_kind="training",
        expected_case_count=1,
        cases=(
            CaseValidation(
                case_id="BraTS-GLI-00001-000",
                directory=str(tmp_path / "BraTS-GLI-00001-000"),
                files=(),
                missing_roles=("t1n",),
                errors=("missing required files: t1n",),
            ),
        ),
    )
    value = report.to_dict()
    assert value["valid"] is False
    assert value["valid_case_count"] == 0

    value["valid"] = True
    path = tmp_path / "edited.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="valid flag is inconsistent"):
        ValidationReport.read_json(path)


def test_parallel_validation_is_ordered_visible_and_never_skips_cases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case_a = BraTSCase("BraTS-GLI-00001-000", tmp_path / "a", ())
    case_b = BraTSCase("BraTS-GLI-00002-000", tmp_path / "b", ())
    dataset = DiscoveredDataset(
        root=tmp_path,
        kind="training",
        cases=(case_b, case_a),
        non_case_entries=(),
        duplicate_case_ids=(),
    )

    def fake_validate(case: BraTSCase, **_: object) -> CaseValidation:
        if case.case_id.endswith("00001-000"):
            time.sleep(0.02)
            return CaseValidation(case.case_id, str(case.directory), ())
        raise RuntimeError("synthetic worker failure")

    monkeypatch.setattr(validation, "validate_case", fake_validate)
    progress: list[tuple[int, int]] = []
    report = validation.validate_dataset(
        dataset,
        dataset_kind="training",
        expected_case_count=2,
        workers=2,
        progress_every_cases=1,
        progress_callback=lambda completed, total, _elapsed: progress.append((completed, total)),
    )

    assert [case.case_id for case in report.cases] == [case_a.case_id, case_b.case_id]
    assert report.actual_case_count == 2
    assert report.valid_case_count == 1
    assert "synthetic worker failure" in report.cases[1].errors[0]
    assert progress[-1] == (2, 2)
