# Official nnU-Net dependency policy

`External/nnUNet` is a separate clone of `MIC-DKFZ/nnUNet`, pinned initially to commit
`0e495086eb108ff79afe106291e8c15bd2f2bc3a` (version 2.8.1) and installed in editable
mode from inside that directory with `pip install -e .` using `Code/.venv`.

The first baseline never copies, reimplements, vendors, or edits nnU-Net trainers,
networks, losses, augmentation, planning, preprocessing, inference, or evaluation code.
`NNUNetV2Backend` invokes only the official console entrypoints. Project-specific data
discovery/conversion, orchestration, monitoring, evaluation, reporting, manifests, and
future ensemble behavior remain under `Code`.

Before every long baseline run, automation verifies the exact commit and an empty
upstream `git status --porcelain`. An editable install is for transparent provenance and
debugging; it is not authorization to change upstream. Generated caches or accidental
files that dirty the checkout must be understood and resolved before the standard
baseline continues.

If a future hypothesis truly requires an nnU-Net modification, it must be isolated from
the pinned reference checkout, version-controlled, and explicitly documented. At
minimum record:

- a new custom experiment ID and config;
- a branch/commit or reproducible patch containing the exact upstream diff;
- the unchanged upstream base commit;
- the hypothesis and affected trainer/network/loss/augmentation behavior;
- dependency and checkpoint incompatibilities;
- a report label such as `custom_nnunet_experiment`, never
  `standard_reference_baseline`;
- comparison against the untouched reference under the same folds and metrics.

Do not make a local edit “just to get the baseline running.” Fix project orchestration or
environment issues in `Code`; report genuine upstream defects separately. Never push or
publish the external clone unless explicitly authorized.
