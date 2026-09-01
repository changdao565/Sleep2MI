# Reproducibility scope

## Included in this repository

The public package supports the following checks without participant data:

1. construct the Sleep2MI encoder from `configs/sleep2mi.json`;
2. verify the 68,322-parameter encoder backbone and 32-dimensional output;
3. evaluate the three representation-learning objective implementations on
   synthetic tensors;
4. compute the participant PCA2 geometry, permutation null distribution, and
   stratified bootstrap interval on synthetic features;
5. extract embeddings from a user-supplied preprocessed EEG-epoch array using a
   compatible user-supplied checkpoint;
6. compute the released geometry analysis from user-supplied participant-level
   features and prespecified binary group labels;
7. run the unit and smoke tests;
8. compute task-valid trial scores, equal-weight trial/seed summaries, and
   group-adjusted outer-fold predictions on caller-supplied arrays; and
9. inspect the aggregate CPU-efficiency table used in the manuscript.

The quickest verification sequence is:

```bash
python -m pip install -e ".[test]"
python examples/quickstart.py
python scripts/run_synthetic_smoke.py
python scripts/run_longitudinal_synthetic.py
python -m pytest -q
python scripts/verify_manifest.py
```

## Command-line reference workflows

`scripts/extract_embeddings.py` loads an EEG-epoch `.npy` array with shape
`(epochs, channels, samples)`, strictly loads a trusted checkpoint containing a
`model_state` mapping, and writes stage logits and 32-dimensional embeddings to
an `.npz` file. A checkpoint-embedded configuration is preferred; otherwise the
script uses `configs/sleep2mi.json`.

`scripts/compute_geometry.py` accepts a participant-by-feature `.npy` or `.npz`
array and prespecified binary labels. It reproduces the public core computation:
median imputation, standardization, PCA2, the between-to-within distance ratio,
the label-permutation test, and the group-stratified bootstrap interval.
The command does not define aptitude groups or aggregate epochs into
participant-level features; those analysis-specific steps are specified in the
Methods and Supplementary Information.

`scripts/run_longitudinal_synthetic.py` exercises the released feedback-score
and longitudinal-evaluation interface on generated four-class membership
outputs. For real window-level inputs, the data adapter first averages the
four-class membership vectors across complete windows within each held-out
trial. `task_valid_membership(...)` then computes the instructed-target
membership within the task-valid class set from that trial-level vector.
`aggregate_trial_and_seed_equal(...)` requires one score per trial, gives
trials equal weight within each participant/task/training run, and then gives
the predefined training-run summaries equal weight. `group_adjusted_oof(...)`
fits the score transform and regression model separately in each outer-training
fold. The held-out score map uses only counts from that fold's training scores.

## Configuration contract

`configs/sleep2mi.json` records the manuscript encoder, objective, and
self-supervised augmentation settings. `Sleep2MIConfig.from_json(...)` extracts
the encoder fields, while the objective module exposes the corresponding loss
weights, temperatures, and augmentation defaults. The test suite checks these
values against the public configuration.

The self-supervised contrastive learning (Self-supervised contrastive) branch
uses 64 Sleep-EDF records. Epoch
eligibility is determined from the signal arrays without reading sleep-stage
annotations. Accepted source arrays have epoch-by-time or
epoch-by-channel-by-time layout, at least two time samples, all finite values,
and per-channel standard deviation greater than `1e-8`. Each record contributes
at most 300 signal-valid epochs selected at deterministic, evenly spaced
chronological positions. Training uses batches of 64 for two epochs, maintains
participant-disjoint training and validation partitions, and selects the
checkpoint with minimum validation contrastive loss. These fields are recorded
under `self_supervised_training` in the public configuration.

The `longitudinal_feedback_evaluation` block records the downstream contract:
the valid class set for each task, equal trial/seed aggregation, the unranked
Session 1 information score, outer-training median imputation, empirical
midranks, population-standard-deviation scaling, and the reduced and full OLS
designs. The configuration does not include participant data or resampling
draws.

## Manuscript estimates

The repository is a reference implementation. Reproducing the numerical
estimates in the manuscript additionally requires:

- obtaining each public dataset from its original provider;
- applying the cohort definitions, preprocessing, physical-window inventories,
  and participant splits specified in the Methods and Supplementary
  Information;
- training or obtaining the relevant Sleep2MI and comparison-model checkpoints;
- supplying the locked participant folds, permutation draws, and bootstrap
  draws; and
- applying the complete downstream statistical procedures defined in the
  manuscript.

The repository does not contain raw EEG, participant-level embeddings or
predictions, resampling draws, trained checkpoints, third-party weights, or raw
timing traces.

## Release integrity

`MANIFEST.sha256` records SHA-256 hashes for the release files. The verification
script checks every listed file and, in a Git checkout, also checks that each
tracked release file is represented exactly once.
