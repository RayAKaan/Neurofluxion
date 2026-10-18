# Neurofluxion Paper — Scope and Version Record

## Working title
**Neurofluxion: Design and Implementation of an Interactive Environment for Inspectable Neural-Network Computation**

## Paper type and intended audience
Implementation/system paper for a peer-reviewed computing, visualization, machine-learning systems, or software-artifact venue. IEEE-style two-column format remains the working target. The manuscript is intentionally structured as an implementation research paper rather than a new-learning-algorithm paper, model-accuracy benchmark, or user study.

## Research question
How can a full-stack application connect real framework-backed neural-network execution with an independently inspectable numerical simulator while preserving computational provenance, explicit component contracts, and reproducible validation?

## Scope boundaries
- Center the implementation: system requirements, component architecture, model-registry contracts, framework execution tracing, simulator construction/execution, frontend inspection workflow, training channels, and reproducibility infrastructure.
- Explain the TensorFlow/Keras execution path separately from the NumPy simulator. They are distinct computational paths; do not imply shared weights, shared state, or numerical equivalence without evidence.
- Treat the frontend as an inspection client over executable state, not as a generic explainability algorithm.
- Report functional and numerical validation, implementation tradeoffs, trace semantics, and reproducibility—not model-accuracy rankings.
- Distinguish direct runtime values from derived, sampled, bounded, estimated, illustrative, synthetic, or unsupported values.
- Do not claim universal interpretability, arbitrary graph correctness, convergence, usability improvement, human learning benefit, comprehensive literature coverage, or novelty/superiority priority.

## Current source baseline
- Repository: https://github.com/RayAKaan/Neurofluxion
- Main revision at the start of this manuscript rewrite: `942b7def72be02cad9df986da373e61a09908be8`
- Manuscript branch: `research/implementation-paper-rewrite`
- Manuscript rewrite commit: `5e1744c9eba0c03b45e5841c47e3b65b10180b61`

This branch is the implementation-paper rewrite proposed against the post-PR-61 main revision. A final publication freeze still requires merging the manuscript branch, recording the resulting main SHA, and rerunning the integrated validation suite against that exact SHA.

## Current evidence status
- [x] Scope and evidence boundaries recorded.
- [x] Source audit and simulator numerics audit recorded.
- [x] Fixed graph-level finite-difference oracles cover Dense/Conv2D, RNN, LSTM, GRU, Embedding→Attention→RNN→Dense, and Conv2D→BatchNorm→Flatten→Dense configurations.
- [x] One-batch training-update contracts cover LSTM/GRU, Conv2D/BatchNorm, and Embedding/Attention/RNN/Dense configurations.
- [x] Adversarial browser tests reproduce stale inspection state after graph replacement and out-of-order response behavior; these remain expected failures and are not presented as fixes.
- [x] Live OpenAlex/Crossref metadata export completed on the prior evidence baseline.
- [x] Implementation-focused IEEE-style manuscript rewrite completed in `paper/neurofluxion.tex`.
- [ ] Human literature screening, full-text extraction, duplicate adjudication beyond automated deduplication, and citation snowballing.
- [ ] Runtime/application fix for the known stale-response/graph-switch provenance defects; requires explicit authorization because runtime source is outside the research-only scope.
- [ ] Final figure capture from the running application, if required by the target venue.
- [ ] Final publication freeze: merge manuscript branch, record resulting main SHA, run the integrated validation suite against that exact SHA, compile the final LaTeX/PDF, and archive the final package.

## Live literature export
The authoritative live export is GitHub Actions workflow run `39172931277`, artifact `12153262023`, generated from main SHA `50de1fb7328d82c8f4799bbf3feffff8fd713216` on 2026-10-18. The exporter ran with CPython 3.11.17 on Ubuntu 24.04.5 and reported:

- `raw_record_count`: 2400
- `deduplicated_record_count`: 2074
- `query_failures`: 0
- `successful_queries`: 12
- `partial_failed_queries`: 0
- `failed_queries`: 0

These counts describe a reproducible metadata export, not a completed systematic review. Screening and full-text review remain separate evidence gates.

## Publication claim discipline
The manuscript may state that the repository contains the documented architecture and implementation, that selected fixed research-validation contracts passed, and that the literature export completed without query failures. It must not convert these results into claims of arbitrary operator correctness, convergence, usability improvement, comprehensive literature coverage, or first/superior status.
