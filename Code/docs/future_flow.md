# Evidence-driven future experiment flow

The project deliberately establishes trustworthy reference behavior before choosing a
proposed method:

```text
BraTS 2023 GLI
  -> official nnU-Net v2 preliminary Fold 0 (pipeline validation)
  -> official nnU-Net v2 default-protocol 5-fold reference
  -> MedNeXt on the same case-level folds and the same evaluator
  -> ET / TC / WT per-case comparison
  -> failure and boundary analysis
  -> one isolated hypothesis at a time
  -> external hospital evaluation only after protocol/data governance review
```

Method selection is conditional on measured evidence:

- Weak WT HD95 or systematic outer-boundary error motivates a boundary-aware experiment.
- Repeated missed small ET lesions motivates an ET/small-lesion sampling or objective
  experiment, with false-positive behavior tracked explicitly.
- Complementary nnU-Net and MedNeXt errors motivate architecture probability averaging.
- Models/checkpoints that are consistently best for different regions motivate a
  region-specific ensemble.
- Site/modality sensitivity motivates a separately controlled multimodal-fusion or
  domain-robustness study, not a silent change to the reference preprocessing.

Future comparisons must use identical folds, labels, metric definitions, empty-case
policy, TTA state, and checkpoint-selection rules. No external or challenge-validation
case may leak into fitting or local hidden-GT claims.

Potential ensembles remain disabled in baseline one. Later candidates include fold
probability averaging, architecture averaging, explicit weighted averaging,
region-specific weights, and checkpoint-specific combinations. Weight selection occurs
only on allowed training/CV data and must be isolated from the final evaluation set.
All region-specific outputs must be projected back to the nested constraint
`ET ⊆ TC ⊆ WT` before conversion to BraTS labels.

Each new experiment needs a version-controlled config and unique manifest. Change one
scientific variable where feasible, state the hypothesis before training, preserve the
reference artifacts, and record whether the experiment is a backbone comparison,
ablation, ensemble, postprocessing, boundary method, multimodal method, or external
validation. Negative and failed results remain reportable evidence.
