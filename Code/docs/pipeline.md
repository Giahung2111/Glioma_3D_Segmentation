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

`Code\.venv` is a Conda prefix, not a `.env` dotenv file. On this Windows host, an
interactive terminal can activate it from the project root as follows:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
(& conda 'shell.powershell' 'hook') | Out-String | Invoke-Expression
conda activate (Resolve-Path .\Code\.venv).Path
. .\Code\scripts\setup_nnunet_env.ps1
python -c "import sys; print(sys.executable)"
```

Activation is only for interactive `python` or direct `nnUNetv2_*` commands. The
pipeline scripts always select the project interpreter and nnU-Net paths themselves.

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

Five-fold training stores each fold's log, runtime, readiness report, GPU summary,
and two-second telemetry stream separately. Its compact heartbeat includes the CV
fold position, current/total epoch, approximate completion and ETA when available,
losses, raw pseudo-Dice, utilization, dedicated VRAM, temperature, and power. The
upstream nnU-Net output remains visible and is never replaced by a cosmetic progress
bar.

A failed command reports exact argv, exit code, a concise likely cause, the last relevant
lines, and the full log path. Output folders and checkpoints are preserved. A normal run
refuses existing checkpoints. The legacy preliminary runner expects
`checkpoint_latest.pth`; the five-fold runner additionally audits upstream's safe
final/latest/best continuation cases and exact experiment ownership.

## Full five-fold baseline

The preliminary pipeline never invokes full CV. Later, after compute/storage review:

```powershell
Code\scripts\run_full_nnunet_cv.ps1 -ConfirmFullBaseline
```

This runs folds 0 through 4 with the official default `nnUNetTrainer` and `--npz`, then
uses the official cross-validation accumulation CLI. It never substitutes a shortened
trainer for the final scientific reference. Completed folds are retained; `-Resume`
continues only folds with an actual latest checkpoint.

## One-command 100-epoch five-fold workflow

The current next-step experiment is a compute-limited five-fold run using the
official upstream `nnUNetTrainer_100epochs`. It always runs folds 0 through 4 in
sequence on the single GPU and always saves `.npz` probabilities for later
checkpoint/model and region-wise ensemble work. It must not be reported as the
standard 1000-epoch nnU-Net reference.

From the project root, no manual activation or environment-variable setup is needed:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\Code\scripts\run_nnunet_cv_pipeline.ps1 -ConfirmRun
```

Before the long run, connect stable AC power, close other GPU-heavy applications,
leave the terminal open, and avoid Windows restart/update windows. Do not rename,
move, or delete the trainer/result/report directories while the process is active.
If interruption is necessary, press Ctrl+C once, allow graceful shutdown, and use
the printed resume command; never start a second copy to the same experiment.

The runner performs, in order: exclusive-process and active-training checks; Windows
sleep prevention; system and dependency checks; the complete project unit-test gate;
raw-data validation; idempotent conversion and preprocessing audit; the official
five-epoch GPU benchmark; five fold-specific readiness/train/validation-probability
audits; official CV accumulation; audited five-model TTA-off inference timing on ten
explicit cases; semantic and pinned official lesion-wise metrics; failure analysis
and face-up canonical figures; and the final report bundle. A five-fold bundle is
not marked complete if the pinned official evaluator fails or evaluates fewer than
all 1,251 out-of-fold cases.

For a preparation-only pass, add `-PreflightOnly`. This includes the short official
five-epoch GPU benchmark but does not start a 100-epoch fold:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\Code\scripts\run_nnunet_cv_pipeline.ps1 -PreflightOnly -ConfirmRun
```

An interrupted run preserves checkpoints and prints the exact resume command. Always
reuse its printed ID:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\Code\scripts\run_nnunet_cv_pipeline.ps1 `
  -ExperimentId <EXPERIMENT_ID> -Resume -ConfirmRun
```

Resume is evidence-based: each fold is skipped only after its owner manifest, final
checkpoint, validation summary, exact validation case IDs, `.nii.gz`, `.npz`, and
`.pkl` files pass audit. A valid latest/best/final checkpoint can be continued;
foreign or unsafe partial output is rejected. Cross-validation accumulation is built
only after all five folds pass strict audit, then verified against all 1,251 labels;
an interrupted derived output is preserved before a clean resume rebuild.
An exclusive lock prevents a second pipeline from training into the same outputs.
Windows sleep prevention is released in `finally`, including after failure.
Interrupted fold attempts retain separate logs, runtime, GPU summaries, and telemetry;
the canonical fold files aggregate every attempt instead of hiding pre-resume time.
If interruption happens before nnU-Net writes any checkpoint, `-Resume` first moves
that exactly owner-matched fold to a timestamped archive beside the model folder,
then performs a clean restart; foreign or checkpoint-bearing folders are never moved.

## Output contract

The experiment report directory contains manifests, config snapshots, logs, semantic
metrics, runtime, GPU summary, failure rankings, figures, and Markdown reports. Large
images, preprocessed arrays, probabilities, checkpoints, and predictions stay under
ignored `Workspace` paths. The original NIfTI files are never rewritten, decompressed,
resampled in place, or deleted.
