# Glioma 3D Segmentation

This repository contains project-owned orchestration, dataset conversion,
validation, evaluation, monitoring, and reporting code for the BraTS 2023
Adult Glioma Pre-Treatment benchmark. The standard baseline uses the official
nnU-Net v2 repository installed independently from `External/nnUNet`; upstream
nnU-Net source is not vendored or modified here.

Raw files under `Datasets` are treated as immutable. Generated nnU-Net data,
plans, results, predictions, telemetry, and reports belong under `Workspace`.

## Dataset conventions

- modalities: `t1n`, `t1c`, `t2w`, `t2f`
- raw labels: background `0`, NCR `1`, ED `2`, ET `3`
- regions: WT `{1,2,3}`, TC `{1,3}`, ET `{3}`
- nnU-Net reconstruction order: WT, TC, ET with
  `regions_class_order = [2, 1, 3]`

Never reinterpret BraTS 2023 ET as label 4.

## Development setup

From `Code`, in the project virtual environment:

```powershell
python -m pip install -e ".[dev]"
pytest
```

Validate the raw training data without changing it:

```powershell
python -m glioma_seg.data.validate `
  --data-root ..\Datasets `
  --output-json ..\Workspace\reports\data_validation.json `
  --output-csv ..\Workspace\reports\data_validation.csv `
  --expected-training-cases 1251
```

Validation uses two bounded worker threads by default and prints elapsed time
and percentage every ten completed cases. Adjust with `--workers` (1-8) and
`--progress-every`; report rows remain deterministically ordered by case ID.

Conversion is idempotent and refuses to overwrite conflicting files. It tries
an atomic hardlink first and falls back to an atomic copy:

```powershell
python -m glioma_seg.data.nnunet_conversion `
  --data-root ..\Datasets `
  --output-root ..\Workspace\nnUNet_raw `
  --validation-json ..\Workspace\reports\data_validation.json `
  --expected-training-cases 1251
```

To populate `imagesTs`, validate the official validation set separately, then
add `--include-validation --official-validation-json <report.json>`. No label
files are created for official validation cases.

The validation report records file identity (size, modification time, and
SHA-256). Conversion verifies it before materialization and checks source
metadata again afterward, preventing a stale validation result from being used.
