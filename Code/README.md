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

`Code\.venv` is a Conda prefix stored inside the project. It is a Python
environment, not a dotenv file named `.env`. The project PowerShell runners call
`Code\.venv\python.exe` directly, so activation is optional for pipeline runs.

For an interactive PowerShell terminal, run these commands once per terminal from
the project root:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
(& conda 'shell.powershell' 'hook') | Out-String | Invoke-Expression
conda activate (Resolve-Path .\Code\.venv).Path
. .\Code\scripts\setup_nnunet_env.ps1
python -c "import sys; print(sys.executable)"
```

The final command must print
`C:\Projects\Glioma_3D_Segmentation\Code\.venv\python.exe` on this workstation.
Then, from `Code`:

```powershell
python -m pip install -e ".[dev]"
pytest
```

Research-model baselines use a separate environment at `Code\.venv-models` so
MONAI, SegResNet, and MedNeXt dependencies do not disturb the nnU-Net baseline:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\Code\scripts\setup_research_models_env.ps1
```

## One-command 100-epoch five-fold experiment

The complete compute-limited workflow uses the official upstream
`nnUNetTrainer_100epochs`, folds 0 through 4, and retains validation `.npz`
probabilities. It prepares data, runs safety/test gates, trains folds sequentially,
evaluates all 1,251 out-of-fold predictions, analyzes failures, and generates the
final report bundle. It also times TTA-off inference with all five fold models on an
explicit ten-case subset and audits the exact timing denominator. Finalization is
fail-closed: all 1,251 pinned official lesion-wise evaluations must also succeed.

No environment activation is required for this one command from the project root:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\Code\scripts\run_nnunet_cv_pipeline.ps1 -ConfirmRun
```

If the terminal, Windows, or training process is interrupted, keep all output and
resume with the exact experiment ID printed by the first run:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\Code\scripts\run_nnunet_cv_pipeline.ps1 -ExperimentId <EXPERIMENT_ID> -Resume -ConfirmRun
```

Add `-PreflightOnly` to run preparation, tests, artifact checks, and the five-epoch
GPU benchmark without starting any 100-epoch fold. This protocol is explicitly a
100-epoch compute-limited cross-validation experiment, not the standard 1000-epoch
nnU-Net reference baseline.

## Same-split research-model baselines

SegResNet and MedNeXt are orchestrated through the same report, evaluation, telemetry,
and failure-analysis contract as nnU-Net. The default commands are full five-fold,
100-epoch, single-GPU runs; add `-SmokeTest` for the real-data resume smoke workflow.

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\Code\scripts\run_segresnet_cv_pipeline.ps1 -ConfirmRun
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\Code\scripts\run_mednext_cv_pipeline.ps1 -ConfirmRun
```

MedNeXt keeps the official MedNeXt-S-k3 trainer and 1 mm / 128x128x128 plan. On
this RTX 2080 Ti 11 GB workstation, the real MedNeXt memory preflight required
about 17.7 GB reserved memory, so the fail-closed runner stops before training
instead of silently changing batch size, patch size, or model variant.

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
