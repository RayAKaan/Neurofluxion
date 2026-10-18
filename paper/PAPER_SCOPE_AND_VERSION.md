# Neurofluxion Paper — Scope and Version Record

## Working title
**Neurofluxion: An Interactive System for Exploring, Simulating, and Inspecting Neural-Network Computation**

## Paper type and intended audience
Implementation/system paper for a peer-reviewed computing, visualization, or software-artifact venue. IEEE-style two-column format remains the working target. This is not a new-learning-algorithm paper, model-accuracy benchmark, or user study.

## Research question
How can an interactive application connect practical framework-backed neural-network inference with a separately inspectable simulator so that users can construct networks, step through supported computation, and examine selected intermediate values and internal mechanisms?

## Scope boundaries
- Center the simulator and inspectable computation workflow: architecture construction, forward/backward execution, parameter/gradient inspection, and supported ANN/CNN/recurrent operations.
- Explain the TensorFlow/Keras inference/trace path separately from the NumPy simulator. They are distinct computational paths; do not imply shared weights or numerical equivalence without evidence.
- Cover model-registry exploration, training telemetry, and visualization as integrated supporting capabilities.
- Report functional and numerical validation, trace fidelity, and reproducibility—not model-accuracy rankings.
- Distinguish direct runtime values from derived, sampled, estimated, illustrative, or unsupported values.
- Do not claim universal interpretability, user-study outcomes, superiority, or novelty priority.

## Current source baseline
- Repository: https://github.com/RayAKaan/Neurofluxion
- Current main source revision used by the manuscript: `50de1fb7328d82c8f4799bbf3feffff8fd713216`
- Current manuscript branch: `research/final-implementation-paper`
- Manuscript implementation commit: `d1eb8189ac207554b06b737520b34217aa669bb3`

The current main revision is the evidence baseline for the manuscript implementation. A separate final publication freeze is still required after the manuscript branch is merged and the exact resulting main SHA is validated.

## Current evidence status
- [x] Scope and evidence boundaries recorded.
- [x] Source audit and simulator numerics audit recorded.
- [x] Research-only graph-level finite-difference oracles cover fixed Dense/Conv2D, RNN, LSTM, GRU, Embedding→Attention→RNN→Dense, and Conv2D→BatchNorm→Flatten→Dense configurations.
- [x] One-batch training-update contracts cover LSTM/GRU, Conv2D/BatchNorm, and Embedding/Attention/RNN/Dense configurations.
- [x] Adversarial browser tests reproduce stale inspection state after graph replacement and out-of-order response behavior; these remain expected failures and are not presented as fixes.
- [x] Live OpenAlex/Crossref metadata export completed on current main revision.
- [x] Live export recorded 12 successful queries, 2,400 raw records, 2,074 automated deduplicated candidates, and zero query/partial-query failures.
- [x] Evidence-grounded IEEE manuscript implemented in `paper/neurofluxion.tex`.
- [ ] Human literature screening, full-text extraction, duplicate adjudication beyond automated deduplication, and citation snowballing.
- [ ] Runtime/application fix for the known stale-response/graph-switch provenance defects; requires explicit authorization because runtime source is outside the research-only scope.
- [ ] Final figure capture from the running application, if required by the target venue.
- [ ] Final publication freeze: merge manuscript branch, record resulting main SHA, run the integrated validation suite against that exact SHA, and archive the final PDF/package.

## Live literature export
The authoritative live export is GitHub Actions workflow run `39172931277`, artifact `12153262023`, generated from main SHA `50de1fb7328d82c8f4799bbf3feffff8fd713216` on 2026-10-18. The live exporter ran with CPython 3.11.17 on Ubuntu 24.04.5 and reported:

- `raw_record_count`: 2400
- `deduplicated_record_count`: 2074
- `query_failures`: 0
- `successful_queries`: 12
- `partial_failed_queries`: 0
- `failed_queries`: 0

These counts describe a reproducible metadata export, not a completed systematic review. Screening and full-text review remain separate evidence gates.

## Publication claim discipline
The manuscript may state that the repository contains the documented architecture, that the selected fixed research-validation contracts passed, and that the live literature export completed without query failures. It must not convert these results into claims of arbitrary operator correctness, convergence, usability improvement, comprehensive literature coverage, or first/superior status.
