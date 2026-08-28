# Sleep2MI

Reference implementation accompanying the manuscript:

> **Sleep-Structure-Supervised Representation Learning Enables Cross-Domain
> and Label-Free Transfer from Sleep to Motor-Imagery Brain-Computer Interfaces**

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
- synthetic examples and tests that do not require participant data; and
- the aggregate CPU-efficiency summary reported in the manuscript.

## Repository layout

| Path | Contents |
| --- | --- |
| `src/sleep2mi/` | Encoder, objectives, pooling, and participant geometry |
| `configs/sleep2mi.json` | Manuscript architecture and objective settings |
| `examples/quickstart.py` | Minimal model-construction and inference example |
| `scripts/extract_embeddings.py` | Checkpoint-based EEG-epoch embedding extraction |
| `scripts/compute_geometry.py` | Participant PCA2 geometry, permutation, and bootstrap analysis |
| `scripts/run_synthetic_smoke.py` | End-to-end synthetic smoke test |
| `scripts/verify_manifest.py` | SHA-256 release-integrity check |
| `tests/` | Synthetic unit and configuration tests |
| `results/aggregate/` | Aggregate computational-efficiency results |
| `data/README.md` | Dataset roles and redistribution boundary |
| `docs/REPRODUCIBILITY.md` | Reproducibility scope and verification commands |

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
python -m pytest -q
```

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

The geometry command standardizes the participant features, fits PCA2, computes
the between-to-within group-distance ratio, and evaluates it using 5,000 label
permutations and 5,000 group-stratified bootstrap resamples by default.

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

## Data and model availability

This repository does not redistribute public EEG recordings, participant-level
derived data, trained Sleep2MI checkpoints, or third-party model weights. Obtain
each dataset and external model from its original provider under the provider's
terms. Dataset roles and the release boundary are summarized in
[`data/README.md`](data/README.md) and
[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

## Reproducibility scope

The release is a reference implementation of the Sleep2MI architecture,
objective definitions, and core participant-geometry computation. Reproducing
the manuscript estimates additionally requires the public datasets and the
preprocessing and evaluation procedures defined in the Methods and
Supplementary Information. See
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
