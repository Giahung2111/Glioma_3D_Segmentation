# Experiment protocol

## Scientific target

The benchmark is 3D multimodal segmentation of pre-treatment adult glioma using BraTS
2023 Adult Glioma Pre-Treatment MRI. Inputs are T1n, T1c, T2w, and T2-FLAIR. Raw labels
remain in the BraTS 2023 convention:

- `0`: background
- `1`: NCR/non-enhancing or necrotic tumor core
- `2`: ED/peritumoral edematous-invaded tissue
- `3`: ET/enhancing tumor

The evaluated nested regions are `WT={1,2,3}`, `TC={1,3}`, and `ET={3}`, satisfying
`ET ⊆ TC ⊆ WT`. The legacy label 4 is never introduced. Region-based nnU-Net training
uses dictionary order WT, TC, ET and `regions_class_order=[2,1,3]`, so reconstruction
produces ED from WT-only voxels, NCR from TC, and ET last.

## Reference baseline

The first standard reference is unmodified official nnU-Net v2, configuration
`3d_fullres`, dataset 501. The preliminary experiment is Fold 0 with the official
20- or 50-epoch validation trainer chosen after the official five-epoch benchmark. It
exists to validate the pipeline, measure runtime, and observe errors. It must be called
“Preliminary single-fold nnU-Net experiment” or “Preliminary Fold-0 pipeline
validation,” never a full reproduction.

The final nnU-Net reference uses the default `nnUNetTrainer`, the default training
protocol, folds 0–4, and saved probabilities. Results from preliminary shortened
training must not be compared directly to challenge leaderboard results as if the
protocols were equal.

The intermediate five-fold run configured in `nnunet_100epoch_cv.yaml` uses the
unmodified official upstream `nnUNetTrainer_100epochs` and saved probabilities. It
provides complete out-of-fold evidence under the available compute budget, but its
manifest and reports must say `compute_limited_cross_validation`; it does not replace
or impersonate the default 1000-epoch reference.

Baseline one excludes pretrained weights, external data, synthetic data, GANs, custom
postprocessing, ensembles, registration augmentation, custom loss/architecture, and
unrequested normalization or image preprocessing. BraTS already provides registration,
common-space alignment, approximately 1 mm isotropic resampling, and skull stripping.
Local image preparation is limited to nnU-Net format conversion and official nnU-Net
fingerprinting/planning/normalization/resampling.

## Split and leakage controls

nnU-Net `splits_final.json` is the source of Fold-0 membership. Splits are case-level,
never slice-level. The readiness gate requires non-empty, disjoint train and validation
case sets and snapshots the split. Only predictions for the selected Fold-0 validation
case IDs are compared with their corresponding `labelsTr` files.

Official BraTS challenge validation images have no public segmentation ground truth.
They may be kept in `imagesTs` for prediction/submission only. They are never copied to
`imagesTr`, assigned fake labels, used for local Dice/HD95, or used for tuning hidden-GT
performance.

## Metrics

Primary local metrics are semantic per-case and aggregate Dice and physical-spacing
HD95 in this fixed presentation order: ET, TC, WT. Empty-set rules are explicit and
must not be silently dropped. Geometry mismatches and missing predictions are fatal.
Per-case rows are retained for failure analysis.

Official BraTS lesion-wise metrics are reported in a separate table only when produced
by a pinned official evaluator whose source/version and command are recorded. Otherwise
the status artifact says `NOT AVAILABLE` and explains why. Semantic and official
lesion-wise values are never merged or relabeled.

Inference timing for the primary preliminary result uses only Fold 0 and TTA off. It
records total wall time, case count, and mean seconds per case. A later default-mirroring
benchmark, if run, must be separate. Training timing is wall-clock time; epoch statistics
are derived from actual log durations when parseable.

## Readiness gate

Long training starts only if all critical checks pass:

- the exact editable nnU-Net 2.8.1 dependency resolves to `External/nnUNet`;
- upstream commit equals `0e495086eb108ff79afe106291e8c15bd2f2bc3a` and its tree is clean;
- official train, predict, and preprocess console entrypoints exist in `Code/.venv`;
- PyTorch is installed, is not 2.9.*, CUDA is available, and dedicated GPU telemetry
  reports sufficient RTX 2080 Ti memory;
- `dataset.json`, channel/region mapping, case counts, files, fingerprint, plans, and the
  requested `3d_fullres` configuration agree;
- the fold split is present, non-empty, and disjoint;
- actual architecture, patch, batch size, and target spacing were read from plans;
- a new run has no existing checkpoint, or an explicit resume has
  `checkpoint_latest.pth`.

The gate records the chosen host-aware `nnUNet_n_proc_DA`, leaves CPU/RAM for Windows,
and prints all critical failures. A failed gate exits before launching nnU-Net.

## Reproducibility and classification

Every run uses a timestamp-plus-random-nonce experiment ID to prevent collision. Its
manifest records commands, dataset/configuration/fold/trainer, environment, runtime,
telemetry summaries, output/checkpoint paths, and baseline classification. The
experiment snapshots `dataset.json`, `dataset_fingerprint.json`, `nnUNetPlans.json`,
`splits_final.json`, and the project experiment configuration where available.

A standard baseline has `baseline_classification=standard_reference_baseline` and
`upstream_source_modified=false`. Any changed trainer, architecture, loss, augmentation,
or upstream code is a custom experiment and must have a new ID/config/branch, exact diff
or patch provenance, scientific hypothesis, and explicit report label. A custom result
must never overwrite or be presented as the standard reference.
