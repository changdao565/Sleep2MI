# Anonymous Supplementary Code

This repository contains the anonymized supplementary implementation
accompanying a manuscript on cross-domain transfer from sleep-supervised EEG
representations to motor-imagery BCI phenotypes.

## Included

- the proposed encoder architecture, comprising a multi-scale CNN, channel
  attention, a bidirectional GRU, temporal attention, and a 32-dimensional
  embedding;
- stage cross-entropy, time-frequency self-supervised contrastive learning, and
  sleep-structure supervised contrastive objectives;
- the study-specific participant geometry statistic and resampling utilities;
- a synthetic smoke test that exercises the model, objectives, and geometry
  without using participant data;
- the aggregate CPU-efficiency summary reported in the manuscript.

Public EEG recordings, participant-level derived data, model checkpoints, and
third-party model weights are not redistributed. Dataset acquisition and cohort
definitions are described in the manuscript and the original dataset records.

## Environment

Python 3.11 or later is recommended. Install the package and test dependencies:

```bash
python -m pip install -e ".[test]"
```

Run the test suite:

```bash
python -m pytest -q
```

or run the standalone smoke test:

```bash
python scripts/run_synthetic_smoke.py
```

## Configuration

The encoder configuration fields in `configs/sleep2mi.json` can be loaded
through the public `Sleep2MIConfig.from_json(...)` interface. No target-cohort
labels are used to update the encoder.

## Reproducibility scope

This release provides the proposed encoder architecture, objective definitions,
and core participant-geometry computation. Reproducing the manuscript estimates
additionally requires the public datasets and the preprocessing and evaluation
procedures described in the Methods and Supplementary Information. The aggregate
CPU-efficiency table is included as a machine-readable summary of the values
reported in the manuscript.
