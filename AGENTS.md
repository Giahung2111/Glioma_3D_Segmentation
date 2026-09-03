# Glioma 3D Segmentation — Codex working instructions

These instructions apply to the entire repository.

## Before MedNeXt or JupyterHub work

Read these files completely before changing code or running a GPU job:

1. Code/docs/JUPYTERHUB_CODEX_MEDNEXT_HANDOFF_VI.md
2. Code/docs/JUPYTERHUB_MEDNEXT_RUNBOOK_VI.md
3. Code/configs/models/mednext.yaml
4. Code/configs/experiments/mednext_100epoch_cv.yaml

Treat the checked-out code, manifests, pinned commits, generated audit evidence,
and actual filesystem state as the sources of truth. The handoff is a dated
context snapshot; report any difference instead of silently guessing.

## Non-negotiable safety rules

- Never modify tracked source under External/. Upstream repositories are pinned
  reference implementations. Project-specific changes belong in Code/.
- Never delete, move, overwrite, or rename raw data or prior Workspace
  artifacts unless the user explicitly requests the exact operation and the
  resolved target has been verified first.
- Never use sudo, install/change a GPU driver, or alter machine-wide packages.
- Never start full MedNeXt training until all 1,251 raw cases pass validation
  and the complete real-data 3-epoch smoke pipeline issues a valid smoke gate
  for the current code/config/split/GPU.
- Never invent model parameters or metrics. Trace model behavior to the pinned
  upstream source and record any project-owned duration/hardware adaptation.
- Do not run two GPU pipelines on the same GPU. Inspect nvidia-smi first.
- Resume only with the exact experiment ID printed by the original run. Audit
  ownership, configuration, split, and checkpoint state before resuming.
- Preserve predictions and both native and canonical probability exports; they
  are required for later region-wise comparison and ensemble experiments.

## Collaboration style for this project

- Lead with the observed state: running, waiting, incomplete, failed, or passed.
- Give the user one safe copy/paste command block at a time, explain its purpose,
  and state the expected output before advancing.
- Distinguish warnings from fatal errors and conclusions from hypotheses.
- Do not promise that training is error-free. Use tests, preflight, smoke,
  manifests, and report-bundle audits as evidence.
- Before editing, run git status --short; preserve unrelated/user changes.
- After editing, run the narrow tests first and then the applicable full suite.

