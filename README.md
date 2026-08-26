# Sleep2MI

This repository contains the compact, reviewer-facing implementation of the
Sleep2MI encoder, its three representation-learning objectives, and the
label-free participant-geometry analysis used in the accompanying manuscript.

## Included

- the multi-scale CNN, channel attention, bidirectional GRU, temporal attention,
  and 32-dimensional embedding architecture;
- stage cross-entropy, time-frequency self-supervised contrastive learning, and
  sleep-structure supervised contrastive objectives;
- the study-specific participant geometry statistic and resampling utilities;
- a synthetic smoke test that exercises the model, objectives, and geometry
  without using participant data;
- aggregate computational-efficiency results reported in the manuscript.

Public EEG recordings, participant-level derived data, model checkpoints, and
third-party model weights are not redistributed. Dataset acquisition and cohort
definitions are described in the manuscript and the original dataset records.

## Environment

Python 3.11 or later is recommended. Install the package and test dependencies:

```bash
python -m pip install -e ".[test]"
```

Run the synthetic test suite:

```bash
python -m pytest -q
```

or run the standalone smoke test:

```bash
python scripts/run_synthetic_smoke.py
```

## Configuration

The manuscript configuration is stored in `configs/sleep2mi.json`. Paths in a
local data adapter should remain relative to the repository or be supplied at
runtime. No target-cohort labels are used to update the encoder.

## Reproducibility boundary

The code release reproduces the architecture, objective definitions, and core
geometry computation. Reproducing the manuscript estimates additionally
requires the public datasets and preprocessing inventories described in the
Methods and Supplementary Information. The aggregate timing table is provided
for verification without exposing participant-level records or third-party
weights.

