# Reproducible BraTS 2023 GLI pipeline

This project treats the original `Datasets` tree as read-only. Framework-format data,
preprocessing, checkpoints, predictions, telemetry, and reports are written under
`Workspace`. All project logic is under `Code`; the official nnU-Net checkout is an
external editable dependency under `External/nnUNet`.

## One-time environment setup

Use Python 3.10 or 3.11 in the isolated project environment. Install a stable CUDA
PyTorch build compatible with the installed driver before installing nnU-Net. PyTorch
2.9.* is explicitly rejected for this baseline because upstream documents a serious
3D-convolution plus AMP performance regression.

From the project root:

```powershell
conda create --prefix Code\.venv python=3.11 -y
Code\.venv\python.exe -m pip install --upgrade pip
# Install the selected stable CUDA PyTorch build here; do not update the NVIDIA driver.
git clone https://github.com/MIC-DKFZ/nnUNet.git External\nnUNet
git -C External\nnUNet checkout 0e495086eb108ff79afe106291e8c15bd2f2bc3a
Push-Location External\nnUNet
..\..\Code\.venv\python.exe -m pip install -e .
Pop-Location
Code\.venv\python.exe -m pip install -e "Code[dev]"
```

The pinned checkout is nnU-Net 2.8.1. If `External/nnUNet` already exists, verify its
remote, commit, and clean working tree; do not clone over or delete it. The system and
training readiness checks reject a missing editable installation, an unexpected commit,
or local upstream source changes.

PowerShell scripts resolve the project from their own path, call the isolated
`Code\.venv\python.exe` explicitly (with standard `venv` `Scripts\python.exe` as a
portable fallback), and set these variables on every run:

```text
nnUNet_raw=<project>\Workspace\nnUNet_raw
nnUNet_preprocessed=<project>\Workspace\nnUNet_preprocessed
nnUNet_results=<project>\Workspace\nnUNet_results
```

Dot-source `scripts/setup_nnunet_env.ps1` only when an interactive terminal also needs
the variables. Pipeline execution never depends on manual environment setup.

## Stages

Run scripts from any directory; their defaults remain project-relative.

1. `00_system_check.ps1` records Python, packages, CPU, RAM, free disk, GPU, dedicated
   VRAM, driver, PyTorch/CUDA/cuDNN, nnU-Net install metadata, and both Git commits.
2. `01_validate_raw_data.ps1` discovers folders by content and validates every training
   case/header/label without modifying source files. `-IncludeOfficialValidation`
   validates the label-free challenge validation set separately.
3. `02_prepare_nnunet.ps1` creates `Dataset501_BraTS2023GLI` through hardlinks where
   safe, with verified-copy fallback. It consumes the successful validation manifest.
4. `03_plan_and_preprocess.ps1` runs the official
   `nnUNetv2_plan_and_preprocess -d 501 -c 3d_fullres --verify_dataset_integrity` CLI.
   Output is streamed live and logged. Existing complete plans are retained unless an
   explicit `-Force` rerun is requested.
5. `04_benchmark_gpu.ps1` runs official
   `nnUNetTrainerBenchmark_5epochs`, records telemetry, reads the actual
   `benchmark_result.json`, and creates labeled linear runtime estimates.
6. `05_train_preliminary.ps1` accepts only the upstream 20/50-epoch trainer selected by
   the benchmark. It prints and persists the readiness gate before starting Fold 0.
   Use `-Resume` only after an interruption; this maps to nnU-Net `--c`.
7. `06_evaluate_preliminary.ps1` times single-fold inference with TTA off on an explicit
   Fold-0 subset, then evaluates all saved Fold-0 validation predictions against the
   matching training labels. `07_analyze_failures.ps1` ranks ET/TC/WT failures and
   creates T1c/FLAIR overlays.
8. `08_generate_report.ps1` creates artifact-backed `summary.md` and
   `weekly_discussion.md`, materializes the verified training/official-validation
   audit reports, and atomically assembles `pipeline.log` from stage logs with
   source-path/SHA-256 boundaries. Successful canonical official lesion-wise
   outputs receive verified compatibility aliases. `experiment.json` inventories
   every experiment-local final artifact with its size and SHA-256. Missing official
   lesion-wise evaluation remains `NOT AVAILABLE` with a reason, never an invented value.

The complete preliminary workflow is:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
Code\scripts\run_preliminary_pipeline.ps1
```

The process-scoped policy change is needed only on Windows hosts whose default policy
blocks local `.ps1` files; it is non-persistent and does not require administrator access.

The printed unique ID identifies a report directory and must be retained for resume:

```powershell
Code\scripts\run_preliminary_pipeline.ps1 `
  -ExperimentId nnunetv2_3dfullres_fold0_prelim_<timestamp>_<nonce> `
  -Resume
```

## Live monitoring and failures

Official nnU-Net stdout and stderr remain visible and are simultaneously appended to
`Workspace/reports/<experiment_id>/logs`. A compact status line appears about every
20 seconds with elapsed time, parsed epoch/loss/pseudo-Dice when present, utilization,
dedicated VRAM, and temperature. Upstream output and `progress.png` are not suppressed.

GPU samples are written every two seconds to
`Workspace/telemetry/<experiment_id>_gpu.csv`. `gpu_summary.json` reports peak observed
dedicated memory, mean utilization, peak temperature, and mean power when available.
Windows shared GPU memory is never included.

A failed command reports exact argv, exit code, a concise likely cause, the last relevant
lines, and the full log path. Output folders and checkpoints are preserved. A normal run
refuses existing checkpoints; an explicit resume requires `checkpoint_latest.pth`.

## Full five-fold baseline

The preliminary pipeline never invokes full CV. Later, after compute/storage review:

```powershell
Code\scripts\run_full_nnunet_cv.ps1 -ConfirmFullBaseline
```

This runs folds 0 through 4 with the official default `nnUNetTrainer` and `--npz`, then
uses the official cross-validation accumulation CLI. It never substitutes a shortened
trainer for the final scientific reference. Completed folds are retained; `-Resume`
continues only folds with an actual latest checkpoint.

## Output contract

The experiment report directory contains manifests, config snapshots, logs, semantic
metrics, runtime, GPU summary, failure rankings, figures, and Markdown reports. Large
images, preprocessed arrays, probabilities, checkpoints, and predictions stay under
ignored `Workspace` paths. The original NIfTI files are never rewritten, decompressed,
resampled in place, or deleted.
