# Sleep2MI

Reference implementation accompanying the manuscript:

> **Sleep Architecture Encodes Transferable Neural Phenotypes That Predict
> Brain-Computer Interface Aptitude Across Independent Awake Cohorts**

**Authors:** Chang Dao, Zijian Wang, and Juexiao Zhou

**Affiliation:** School of Data Science, The Chinese University of Hong Kong,
Shenzhen (CUHK-Shenzhen), Guangdong 518172, P.R. China

**Contact:** Juexiao Zhou ([juexiao.zhou@gmail.com](mailto:juexiao.zhou@gmail.com))

Chang Dao and Zijian Wang contributed equally. Juexiao Zhou is the corresponding
author.

## Overview

Sleep2MI learns EEG representations from sleep-stage and whole-recording sleep
structure supervision and applies the frozen encoder to independent awake
motor-imagery brain-computer interface cohorts. This repository provides:

- the multi-scale convolutional, channel-attention, bidirectional-GRU, and
  temporal-attention encoder with a 32-dimensional bottleneck;
- stage cross-entropy, self-supervised contrastive learning (Self-supervised
  contrastive) with time-frequency views, and sleep-structure supervised
  contrastive learning (Sleep-structure SupCon);
- record-level bag-attention pooling and paired-bag consistency;
- the participant-geometry statistic and its permutation and bootstrap tools;
- task-valid feedback-score aggregation and fold-local longitudinal evaluation
  utilities;
- synthetic examples and tests that do not require participant data; and
- the aggregate CPU-efficiency summary reported in the manuscript.

## Repository layout

| Path | Contents |
| --- | --- |
| `src/sleep2mi/` | Encoder, objectives, pooling, participant geometry, and longitudinal evaluation utilities |
| `configs/sleep2mi.json` | Manuscript architecture and objective settings |
| `examples/quickstart.py` | Minimal model-construction and inference example |
| `scripts/extract_embeddings.py` | Checkpoint-based EEG-epoch embedding extraction |
| `scripts/compute_geometry.py` | Participant PCA2 geometry, permutation, and bootstrap analysis |
| `scripts/run_synthetic_smoke.py` | End-to-end synthetic smoke test |
| `scripts/run_longitudinal_synthetic.py` | Participant-free check of feedback aggregation and outer-fold evaluation |
| `scripts/verify_manifest.py` | SHA-256 release-integrity check |
| `tests/` | Synthetic unit and configuration tests |
| `results/aggregate/` | Aggregate computational-efficiency results |
| `data/README.md` | Dataset roles and redistribution boundary |
| `docs/REPRODUCIBILITY.md` | Reproducibility scope and verification commands |
| `MODEL_CARD.md` | Intended use, release scope, and limitations |
| `requirements-tested.txt` | Exact package versions used for the public CPU test suite |

## Installation

Python 3.11 or later is required. From the repository root, install the package
and its test dependencies:

```bash
python -m pip install -e ".[test]"
```

## Quick start

Build the manuscript encoder from the public configuration and run it on a
synthetic 30-s sleep-EEG batch:

```bash
python examples/quickstart.py
```

Run the broader synthetic smoke test and test suite:

```bash
python scripts/run_synthetic_smoke.py
python scripts/run_longitudinal_synthetic.py
python -m pytest -q
```

### Tested environment and expected output

The public examples and tests require only a CPU; no non-standard hardware is
needed. They were verified on Windows 11 with Python 3.12.6 and on GitHub
Actions Linux runners with Python 3.11 and 3.12. The exact local package
versions are recorded in `requirements-tested.txt`.

`examples/quickstart.py` reports an input shape of `[2, 1, 3000]`, a stage-logit
shape of `[2, 5]`, an embedding shape of `[2, 32]`, and 68,322 encoder
parameters. The two smoke scripts print JSON summaries with finite geometry,
objective, and longitudinal-evaluation metrics. On the documented Windows CPU
test system, the quick start and each smoke script completed in approximately
4--5 s, and the 28-test suite completed in approximately 18 s. Runtime varies
with hardware and software environment.

Verify the checksums in the release manifest:

```bash
python scripts/verify_manifest.py
```

## Analysis entry points

Extract embeddings from a preprocessed EEG-epoch array with shape
`(epochs, channels, samples)` using a trusted Sleep2MI checkpoint:

```bash
python scripts/extract_embeddings.py \
  --input epochs.npy \
  --checkpoint sleep2mi_checkpoint.pt \
  --output embeddings.npz
```

The checkpoint must contain a `model_state` mapping and may contain its training
`config`. If no checkpoint configuration is present, the public manuscript
configuration is used. The output contains `embeddings` and `stage_logits`.

Compute the released participant-geometry statistic from a
participant-by-feature array and prespecified binary group labels:

```bash
python scripts/compute_geometry.py \
  --features participant_embeddings.npy \
  --labels fixed_group_labels.npy \
  --output-json geometry.json \
  --coordinates-output pca2_coordinates.npz
```

The geometry command median-imputes missing feature values, standardizes the
participant features, fits PCA2, computes the between-to-within group-distance
ratio, and evaluates it using 5,000 label permutations and 5,000
group-stratified bootstrap resamples by default. Infinite feature values are
rejected.

The public longitudinal helpers implement the task-matched feedback score and
the leakage-controlled downstream transform used in the manuscript. For each
held-out trial, first average the complete-window four-class membership vectors.
`task_valid_membership(...)` then divides the instructed-target component of
that trial-level mean vector by the summed membership of the valid classes for
that task: classes 1/2 for LR, 3/4 for UD, and all four classes for 2D. Trial
scores receive equal weight within each participant, task, and encoder training
run; the predefined training-run summaries then receive equal weight. The
released aggregator rejects duplicate rows for the same held-out trial so that
window-level ratios cannot be averaged accidentally.

`group_adjusted_oof(...)` repeats score imputation, empirical-midrank mapping,
standardization, nuisance transformation, and OLS fitting inside each outer
training fold. Held-out values are mapped only against the corresponding
outer-training distribution. The synthetic command exercises this interface
without using participant data:

```bash
python scripts/run_longitudinal_synthetic.py
```

## Configuration

`Sleep2MIConfig.from_json(...)` loads the encoder fields from
`configs/sleep2mi.json`. The same file records the manuscript loss weights,
contrastive temperatures, and self-supervised augmentation settings. The
Self-supervised contrastive training contract uses 64 Sleep-EDF records and
signal-valid epochs selected without access to sleep-stage annotations. It caps
each record at 300 epochs using deterministic, evenly spaced chronological
positions, uses batches of 64 for two training epochs, keeps training and
validation participants disjoint, and retains the checkpoint with minimum
validation contrastive loss. Signal validity requires an expected channel/time
shape, at least two time samples, all finite values, and per-channel standard
deviation greater than `1e-8`. The configuration and source defaults are
checked by the test suite.

The `longitudinal_feedback_evaluation` block records the task-valid class sets,
trial/seed aggregation rule, unranked Session 1 information check, and
outer-training-only empirical-midrank and standardization contract. It contains
method settings only; participant-level scores and outcomes are not included.

## Data and model availability

This repository does not redistribute public EEG recordings, participant-level
derived data, trained Sleep2MI checkpoints, or third-party model weights. Obtain
each dataset and external model from its original provider under the provider's
terms. Dataset roles and the release boundary are summarized in
[`data/README.md`](data/README.md) and
[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

## Reproducibility scope

The release is a reference implementation of the Sleep2MI architecture,
objective definitions, core participant-geometry computation, and longitudinal
feedback-score evaluator. Reproducing the manuscript estimates additionally
requires the public datasets and the preprocessing and resampling inventories
defined in the Methods and Supplementary Information. See
[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) for the exact boundary.

## Citation

Software citation metadata are provided in [`CITATION.cff`](CITATION.cff). A
paper DOI and journal citation will be added after publication. Until then, the
software can be cited as:

```bibtex
@software{sleep2mi_software,
  author  = {Dao, Chang and Wang, Zijian and Zhou, Juexiao},
  title   = {Sleep2MI},
  year    = {2026},
  version = {0.1.0},
  url     = {https://github.com/changdao565/Sleep2MI}
}
```
