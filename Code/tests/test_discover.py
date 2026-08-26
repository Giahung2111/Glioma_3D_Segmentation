from pathlib import Path

from glioma_seg.data.brats2023 import MODALITIES, source_filename
from glioma_seg.data.discover import discover_brats_datasets


def _touch_case(directory: Path, case_id: str, *, segmentation: bool) -> None:
    directory.mkdir(parents=True)
    for role in (*MODALITIES, *(("seg",) if segmentation else ())):
        (directory / source_filename(case_id, role)).write_bytes(b"not-loaded-by-discovery")


def test_discovers_training_and_validation_by_content(tmp_path: Path) -> None:
    training_root = tmp_path / "folder-with-arbitrary-name"
    validation_root = tmp_path / "another-arbitrary-name"
    _touch_case(training_root / "BraTS-GLI-00001-000", "BraTS-GLI-00001-000", segmentation=True)
    _touch_case(
        validation_root / "BraTS-GLI-01001-000",
        "BraTS-GLI-01001-000",
        segmentation=False,
    )
    (tmp_path / "BraTS2023_2017_GLI_Mapping.xlsx").write_bytes(b"mapping")

    result = discover_brats_datasets(tmp_path)

    assert result.require_unique("training").root == training_root.resolve()
    assert result.require_unique("validation").root == validation_root.resolve()
    assert result.non_dataset_entries == ((tmp_path / "BraTS2023_2017_GLI_Mapping.xlsx").resolve(),)


def test_partial_training_set_remains_training_candidate(tmp_path: Path) -> None:
    dataset_root = tmp_path / "source"
    _touch_case(dataset_root / "BraTS-GLI-00001-000", "BraTS-GLI-00001-000", segmentation=True)
    _touch_case(
        dataset_root / "BraTS-GLI-00002-000",
        "BraTS-GLI-00002-000",
        segmentation=False,
    )

    dataset = discover_brats_datasets(tmp_path).require_unique("training")

    assert dataset.case_count == 2
    incomplete = next(case for case in dataset.cases if case.case_id.endswith("00002-000"))
    assert incomplete.missing_roles(require_segmentation=True) == ("seg",)


def test_case_directory_name_is_not_assumed_during_discovery(tmp_path: Path) -> None:
    dataset_root = tmp_path / "source"
    _touch_case(dataset_root / "wrong-folder-name", "BraTS-GLI-00001-000", segmentation=True)

    case = discover_brats_datasets(tmp_path).require_unique("training").cases[0]

    assert case.case_id == "BraTS-GLI-00001-000"
    assert not case.directory_name_matches
