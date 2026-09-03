from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from glioma_seg.ensembles.canonical_probabilities import (
    CANONICAL_REGION_ORDER,
    MEDNEXT_MULTICLASS_CONVERSION,
    MEDNEXT_MULTICLASS_ORDER,
    SEGRESNET_REGION_CONVERSION,
    SEGRESNET_REGION_ORDER,
    load_canonical_probability_npz,
    mednext_multiclass_to_canonical,
    segresnet_regions_to_canonical,
    write_canonical_probability_npz,
)


def _reference(tmp_path: Path, case_id: str = "BraTS-GLI-00001-000") -> Path:
    path = tmp_path / f"{case_id}.nii.gz"
    affine = np.diag([1.5, 2.0, 2.5, 1.0])
    nib.save(nib.Nifti1Image(np.zeros((2, 3, 4), dtype=np.uint8), affine), path)
    return path


def _segresnet_native() -> np.ndarray:
    native = np.empty((3, 2, 3, 4), dtype=np.float32)
    native[0].fill(0.25)  # TC
    native[1].fill(0.75)  # WT
    native[2].fill(0.10)  # ET
    return native


def test_mednext_multiclass_conversion_uses_exact_region_sums() -> None:
    native = np.zeros((4, 2, 1, 1), dtype=np.float32)
    native[:, 0, 0, 0] = (0.1, 0.2, 0.3, 0.4)
    native[:, 1, 0, 0] = (0.7, 0.1, 0.1, 0.1)

    converted = mednext_multiclass_to_canonical(native)

    assert converted.dtype == np.float32
    assert converted.shape == (3, 2, 1, 1)
    np.testing.assert_allclose(converted[:, 0, 0, 0], (0.4, 0.6, 0.9))
    np.testing.assert_allclose(converted[:, 1, 0, 0], (0.1, 0.2, 0.3))


def test_mednext_conversion_rejects_wrong_order_logits_and_nonfinite_values() -> None:
    valid = np.full((4, 1, 1, 1), 0.25, dtype=np.float32)
    with pytest.raises(ValueError, match="native channel order"):
        mednext_multiclass_to_canonical(
            valid, native_channel_order=("background", "ED", "NCR", "ET")
        )
    invalid_simplex = valid.copy()
    invalid_simplex[0] = 0.5
    with pytest.raises(ValueError, match="sum to 1"):
        mednext_multiclass_to_canonical(invalid_simplex)
    nonfinite = valid.copy()
    nonfinite[1, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN or infinite"):
        mednext_multiclass_to_canonical(nonfinite)
    with pytest.raises(TypeError, match="floating dtype"):
        mednext_multiclass_to_canonical(np.zeros((4, 1, 1, 1), dtype=np.uint8))


def test_segresnet_conversion_reorders_tc_wt_et_to_et_tc_wt() -> None:
    native = _segresnet_native()

    converted = segresnet_regions_to_canonical(native)

    np.testing.assert_array_equal(converted[0], native[2])
    np.testing.assert_array_equal(converted[1], native[0])
    np.testing.assert_array_equal(converted[2], native[1])
    with pytest.raises(ValueError, match="native channel order"):
        segresnet_regions_to_canonical(native, native_channel_order=("ET", "TC", "WT"))


def test_atomic_writer_and_safe_loader_round_trip_geometry_and_provenance(
    tmp_path: Path,
) -> None:
    reference = _reference(tmp_path)
    output = tmp_path / "canonical.npz"
    canonical = segresnet_regions_to_canonical(_segresnet_native())

    written = write_canonical_probability_npz(
        output,
        case_id="BraTS-GLI-00001-000",
        probabilities=canonical,
        native_channel_order=SEGRESNET_REGION_ORDER,
        conversion=SEGRESNET_REGION_CONVERSION,
        reference_nifti=reference,
    )
    artifact = load_canonical_probability_npz(
        written,
        reference_nifti=reference,
        expected_case_id="BraTS-GLI-00001-000",
        expected_native_channel_order=SEGRESNET_REGION_ORDER,
        expected_conversion=SEGRESNET_REGION_CONVERSION,
    )

    assert written == output.resolve()
    assert artifact.case_id == "BraTS-GLI-00001-000"
    assert artifact.native_channel_order == SEGRESNET_REGION_ORDER
    assert artifact.canonical_channel_order == CANONICAL_REGION_ORDER
    assert artifact.spatial_shape == (2, 3, 4)
    assert artifact.spacing_mm == pytest.approx((1.5, 2.0, 2.5))
    assert artifact.reference_nifti == reference.resolve()
    assert len(artifact.reference_sha256) == 64
    np.testing.assert_array_equal(artifact.probabilities, canonical)
    assert not list(tmp_path.glob(".canonical.npz.*.tmp"))
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        write_canonical_probability_npz(
            output,
            case_id="BraTS-GLI-00001-000",
            probabilities=canonical,
            native_channel_order=SEGRESNET_REGION_ORDER,
            conversion=SEGRESNET_REGION_CONVERSION,
            reference_nifti=reference,
        )


def test_writer_rejects_case_shape_range_and_conversion_errors(tmp_path: Path) -> None:
    reference = _reference(tmp_path)
    canonical = segresnet_regions_to_canonical(_segresnet_native())
    common = {
        "native_channel_order": SEGRESNET_REGION_ORDER,
        "conversion": SEGRESNET_REGION_CONVERSION,
        "reference_nifti": reference,
    }
    with pytest.raises(ValueError, match="does not match reference"):
        write_canonical_probability_npz(
            tmp_path / "wrong_case.npz",
            case_id="another-case",
            probabilities=canonical,
            **common,
        )
    with pytest.raises(ValueError, match="shape does not match"):
        write_canonical_probability_npz(
            tmp_path / "wrong_shape.npz",
            case_id="BraTS-GLI-00001-000",
            probabilities=canonical[:, :, :, :2],
            **common,
        )
    out_of_range = canonical.copy()
    out_of_range[0, 0, 0, 0] = 1.1
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        write_canonical_probability_npz(
            tmp_path / "out_of_range.npz",
            case_id="BraTS-GLI-00001-000",
            probabilities=out_of_range,
            **common,
        )
    with pytest.raises(ValueError, match="conversion must explicitly describe"):
        write_canonical_probability_npz(
            tmp_path / "no_conversion.npz",
            case_id="BraTS-GLI-00001-000",
            probabilities=canonical,
            native_channel_order=SEGRESNET_REGION_ORDER,
            conversion=" ",
            reference_nifti=reference,
        )


def test_loader_rejects_reference_hash_change_and_expected_declaration_mismatch(
    tmp_path: Path,
) -> None:
    reference = _reference(tmp_path)
    output = write_canonical_probability_npz(
        tmp_path / "canonical.npz",
        case_id="BraTS-GLI-00001-000",
        probabilities=segresnet_regions_to_canonical(_segresnet_native()),
        native_channel_order=SEGRESNET_REGION_ORDER,
        conversion=SEGRESNET_REGION_CONVERSION,
        reference_nifti=reference,
    )
    with pytest.raises(ValueError, match="Expected native channel order"):
        load_canonical_probability_npz(
            output,
            reference_nifti=reference,
            expected_native_channel_order=MEDNEXT_MULTICLASS_ORDER,
        )
    with pytest.raises(ValueError, match="Expected conversion"):
        load_canonical_probability_npz(
            output,
            reference_nifti=reference,
            expected_conversion=MEDNEXT_MULTICLASS_CONVERSION,
        )

    affine = np.diag([1.5, 2.0, 2.5, 1.0])
    changed = np.zeros((2, 3, 4), dtype=np.uint8)
    changed[0, 0, 0] = 1
    nib.save(nib.Nifti1Image(changed, affine), reference)
    with pytest.raises(ValueError, match="SHA-256 differs"):
        load_canonical_probability_npz(output, reference_nifti=reference)


def test_loader_rejects_object_metadata_and_tampered_affine(tmp_path: Path) -> None:
    reference = _reference(tmp_path)
    output = write_canonical_probability_npz(
        tmp_path / "canonical.npz",
        case_id="BraTS-GLI-00001-000",
        probabilities=segresnet_regions_to_canonical(_segresnet_native()),
        native_channel_order=SEGRESNET_REGION_ORDER,
        conversion=SEGRESNET_REGION_CONVERSION,
        reference_nifti=reference,
    )
    with np.load(output, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}

    object_metadata = dict(arrays)
    object_metadata["case_id"] = np.asarray([object()], dtype=object)
    unsafe = tmp_path / "unsafe_object.npz"
    np.savez_compressed(unsafe, **object_metadata)
    with pytest.raises(ValueError, match="Unable to load safe canonical probabilities"):
        load_canonical_probability_npz(unsafe, reference_nifti=reference)

    tampered = dict(arrays)
    tampered["affine"] = np.asarray(tampered["affine"], dtype=np.float64).copy()
    tampered["affine"][0, 3] = 10.0
    wrong_affine = tmp_path / "wrong_affine.npz"
    np.savez_compressed(wrong_affine, **tampered)
    with pytest.raises(ValueError, match="affine differs"):
        load_canonical_probability_npz(wrong_affine, reference_nifti=reference)
