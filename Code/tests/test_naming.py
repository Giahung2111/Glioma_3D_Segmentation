import pytest

from glioma_seg.data.brats2023 import (
    NamingError,
    nnunet_image_name,
    nnunet_label_name,
    parse_source_filename,
    source_filename,
)

CASE_ID = "BraTS-GLI-00008-001"


@pytest.mark.parametrize(
    ("role", "channel"),
    [("t1n", "0000"), ("t1c", "0001"), ("t2w", "0002"), ("t2f", "0003")],
)
def test_nnunet_channel_naming(role: str, channel: str) -> None:
    assert parse_source_filename(source_filename(CASE_ID, role)) == (CASE_ID, role)
    assert nnunet_image_name(CASE_ID, role) == f"{CASE_ID}_{channel}.nii.gz"


def test_nnunet_label_naming() -> None:
    assert parse_source_filename(f"{CASE_ID}-seg.nii.gz") == (CASE_ID, "seg")
    assert nnunet_label_name(CASE_ID) == f"{CASE_ID}.nii.gz"


@pytest.mark.parametrize(
    "filename",
    [
        "BraTS-GLI-00008-001-t1.nii.gz",
        "BraTS-GLI-00008-001-t1n.nii",
        "BraTS-GLI-8-1-t1n.nii.gz",
        "BraTS-GLI-00008-001-seg.nii.gz.tmp",
    ],
)
def test_invalid_source_names_are_rejected(filename: str) -> None:
    with pytest.raises(NamingError):
        parse_source_filename(filename)


def test_unknown_modality_is_rejected() -> None:
    with pytest.raises(NamingError, match="Unknown BraTS modality"):
        nnunet_image_name(CASE_ID, "flair")
