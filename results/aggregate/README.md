# Aggregate results

`model_efficiency_summary.csv` contains the head-free CPU timing summary used in
the manuscript. Timing begins at model forward on deterministic post-processing
tensors and excludes file input/output, channel mapping, rereferencing,
filtering, resampling, standardization, window construction, and artifact
handling. Each model retains its paper-route input interface, so the table is an
operational comparison rather than a matched-input architecture benchmark.

Participant-level embeddings, predictions, bootstrap samples, permutation
samples, and raw timing traces are intentionally not included.

