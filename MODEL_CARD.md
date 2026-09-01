# Sleep2MI model card

## Model summary

Sleep2MI is a research implementation of an electroencephalographic (EEG)
encoder trained with sleep-stage and whole-recording sleep-structure
objectives. The encoder maps 30-second EEG epochs to a 32-dimensional
representation. The accompanying manuscript evaluates frozen representations
in independent awake motor-imagery brain-computer interface cohorts.

## Intended use

The release is intended for research on sleep representation learning,
cross-domain EEG analysis, participant-level geometry, and reproducibility of
the methods described in the accompanying manuscript. It may also be used as a
reference implementation for method comparison on appropriately licensed
research datasets.

## Out-of-scope use

The model is not a medical device and is not intended for diagnosis, treatment,
clinical decision-making, safety-critical monitoring, or autonomous decisions
about individual participants. Performance outside the datasets, cohorts,
montages, preprocessing steps, and endpoints evaluated in the manuscript has
not been established.

## Inputs and outputs

The encoder accepts preprocessed EEG epochs with shape `(batch, channels,
samples)` and returns five-class sleep-stage logits and a 32-dimensional
embedding. Channel selection, sampling rate, epoch length, and preprocessing
must be matched to the relevant manuscript protocol or documented by the user.

## Training and evaluation scope

The manuscript reports training on public sleep datasets and evaluation on
independent public sleep and motor-imagery EEG cohorts. Dataset roles, cohort
definitions, preprocessing, participant splits, and statistical procedures are
specified in the Methods and Supplementary Information. The public repository
contains the architecture, objective definitions, core geometry and
longitudinal-evaluation utilities, and synthetic tests. It does not contain raw
EEG, participant-level derived data, trained checkpoints, third-party model
weights, or the original providers' datasets.

## Limitations

- Results may not generalize to different acquisition systems, populations,
  tasks, channel montages, or preprocessing pipelines.
- The reported associations do not establish clinical utility or causality.
- Participant-level geometry depends on the prespecified cohort definitions and
  analysis protocol.
- Reproducing manuscript estimates requires the public datasets and processing
  inventories described in the paper as well as compatible checkpoints.

## Ethical and data considerations

Users are responsible for complying with the original dataset terms, research
ethics requirements, privacy protections, and local regulations. Do not use
this software to infer sensitive traits or make consequential decisions about
individuals.

## Version and contact

This card describes Sleep2MI version 0.1.0. Questions may be directed to the
corresponding author listed in `CITATION.cff`.
