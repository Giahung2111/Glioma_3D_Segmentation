from __future__ import annotations

from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

import glioma_seg.visualization.overlays as overlays


def test_lps_nifti_is_canonicalized_so_anterior_marker_is_at_image_top(
    tmp_path: Path, monkeypatch: Any
) -> None:
    shape = (3, 4, 2)
    # BraTS-style LPS: array index y=0 has the largest world-A coordinate.
    affine = np.asarray(
        [
            [-1.0, 0.0, 0.0, 2.0],
            [0.0, -1.0, 0.0, 3.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    t1c = np.zeros(shape, dtype=np.int16)
    t1c[1, 0, 0] = 100  # anterior marker
    flair = np.zeros(shape, dtype=np.int16)
    labels = np.zeros(shape, dtype=np.uint8)
    paths = [tmp_path / name for name in ("t1c.nii.gz", "flair.nii.gz", "gt.nii.gz", "pred.nii.gz")]
    for path, array in zip(paths, (t1c, flair, labels, labels), strict=True):
        nib.save(nib.Nifti1Image(array, affine), path)  # type: ignore[no-untyped-call]

    captured: dict[str, Any] = {}

    def fake_create_failure_figure(**kwargs: Any) -> Path:
        captured.update(kwargs)
        return Path(kwargs["output_path"])

    monkeypatch.setattr(overlays, "create_failure_figure", fake_create_failure_figure)
    output = tmp_path / "figure.png"
    overlays.create_failure_figure_from_nifti(
        case_id="orientation-case",
        t1c_path=paths[0],
        flair_path=paths[1],
        ground_truth_path=paths[2],
        prediction_path=paths[3],
        output_path=output,
        n_slices=1,
    )

    canonical = np.asarray(captured["t1c"])
    displayed = overlays._take(canonical, 0, 2)
    assert np.max(displayed[0]) == 100
    assert np.max(displayed[-1]) == 0
