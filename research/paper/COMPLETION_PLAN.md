# Neurofluxion Implementation Paper — Research Completion Plan

Working title: Neurofluxion: An Inspectable Simulator for Neural-Network Computation  
Baseline: main after PR #22, merge commit b3e0990da61d54b71b27963f3a664b30ccd47378  
Status: manuscript revision in progress; not submission-ready.

## 1. Research framing

Treat Neurofluxion as an implementation and empirical systems paper, not a new neural-network algorithm or a general interpretability method. The central question is:

What evidence must an interactive neural-network simulator preserve so displayed intermediate values can be tied to, reconstructed from, and distinguished from the computation that produced them?

Keep three kinds of claims separate:

1. Numerical fidelity: does the simulator compute values and gradients expected from independently specified mathematics?
2. Provenance integrity: does each visible value belong to the selected graph, parameter revision, and execution, or is it cached, stale, illustrative, or unavailable?
3. Analytical utility: do users answer concrete debugging or inspection questions more accurately or quickly? This requires a controlled user study.

The current manuscript has useful source-grounded observations and a narrow numerical audit. It does not yet demonstrate all-layer correctness, replay fidelity, scalability, user benefit, or a defensible novelty gap.

## 2. Existing evidence and its boundaries

- PR #19 corrected fixed dense-path MSE gradient scaling and returned-gradient ordering.
- PR #20 corrected the TrainingEngine MSE gradient contract.
- PR #21 added post-merge research audit documentation and corrected workflow exit-status capture.
- PR #22 corrected the mean-reduced BCE/sigmoid gradient contract and added finite-difference tests for both backward_full and TrainingEngine.
- PR #22 merge-preview CI run 38491607499 completed successfully: 11 tests passed across simulator smoke, simulator MSE, TrainingEngine MSE, and BCE gradient-contract tests.
- Research audit run 38491607397 completed successfully on merge-preview tree d2e64487b9c7ab495a2be9d7c22d22dd32150349, Python 3.11.17 and NumPy 2.4.6. The fixed dense case reported forward maximum absolute error 0.0, trace final-output error 0.0, passing selected MSE finite differences, and identical output/trace hashes on repeated runs.
- PR #22 was subsequently merged as b3e0990da61d54b71b27963f3a664b30ccd47378. CI evidence was collected on the PR merge-preview tree, not a fresh test run after the merge commit was created.

Not established: full repository test-suite status; numerical correctness for every registered operator, activation, loss, or optimizer; convergence; replay and cross-view consistency; complete asynchronous UI provenance; performance/scalability overhead; user benefit; or novelty against the whole literature.

## 3. Comparison with influential IEEE and adjacent systems

This is a targeted comparison set, not an exhaustive literature review. The full review should document databases, queries, dates, inclusion/exclusion rules, and screening decisions.

| Paper/system | Venue/year | Main task | Relevance and boundary |
|---|---|---|---|
| ActiVis | IEEE TVCG, 2018 | Coordinated exploration of model structure, activations, and data instances | Strong precedent for linked architecture and activation views; those views alone are not novel. DOI 10.1109/TVCG.2017.2744718 |
| DeepEyes | IEEE TVCG, 2018 | Progressive visual analytics for neural-network design | Iterative model design is established prior art. DOI 10.1109/TVCG.2017.2744358 |
| LSTMVis | IEEE TVCG, 2018 | Hidden-state dynamics in recurrent networks | Relevant to sequence inspection; dense-only tests cannot validate recurrent internals. DOI 10.1109/TVCG.2017.2744158 |
| TensorFlow Graph Visualizer | IEEE TVCG, 2018 | Hierarchical exploration of dataflow graphs | Graph visualization is not equivalent to numerical trace validation. DOI 10.1109/TVCG.2017.2744878 |
| CNNVis | IEEE TVCG, 2017 | CNN diagnosis through neuron/feature and connection views | General neural-network debugging is not a novel claim. DOI 10.1109/TVCG.2016.2598831 |
| GAN Lab | IEEE TVCG, 2019 | Browser-based stepwise GAN experimentation | Browser delivery and stepwise interaction are established. DOI 10.1109/TVCG.2018.2864500 |
| CNN Explainer | IEEE TVCG, 2021 | Multi-level interactive explanation of CNN computation; qualitative user study | Strong educational and operation-explanation precedent; no learning-benefit claim without a study. DOI 10.1109/TVCG.2020.3030418 |
| Net2Vis | IEEE TVCG, 2021 | Automatic generation of publication-ready CNN diagrams | Static architecture authoring differs from runtime numerical inspection. DOI 10.1109/TVCG.2021.3057483 |
| M2Lens | IEEE TVCG, 2022 | Multi-level analysis of multimodal model interactions | Adjacent explanation system; avoid claiming broad interpretability coverage. DOI 10.1109/TVCG.2021.3114794 |
| An In-Situ Visual Analytics Framework for DNNs | IEEE TVCG, 2024 | Visual analysis during DNN training | Important newer comparator; explain distinctions beyond integration with execution. DOI 10.1109/TVCG.2023.3339585 |
| Visual Analytics in Deep Learning survey | IEEE TVCG, 2019 issue | Taxonomy of visual analytics tasks and systems | Foundational task-level framing. DOI 10.1109/TVCG.2018.2843369 |
| Visualization for Trust in ML Revisited | IEEE Computer Graphics and Applications, 2024 | Survey of 542 visualization techniques | Recent landscape check; hand-picked papers are not a systematic review. DOI 10.1109/MCG.2024.3360881 |
| Visual Analytics for ML: A Data Perspective Survey | IEEE TVCG, 2024 | Survey of data-centric visual analytics across the ML lifecycle | Recent taxonomy and gap-check source. DOI 10.1109/TVCG.2024.3357065 |

Candidate distinction to investigate: a custom simulator path, selected operation-level arithmetic records, and an explicit claim/evidence treatment of numerical fidelity and UI-state provenance. This is a candidate framing, not a demonstrated novelty claim. It becomes stronger only if operator tests and adversarial provenance tests establish what the system guarantees and where it reports uncertainty.

## 4. Ordered evaluation plan

### Phase A — Freeze and reproducibility
- Pin the exact main commit and dependency/environment versions for every experiment.
- Record source SHA, command, environment, exit code, raw artifact, and analysis together.
- Generate all tables and figures from raw JSON or checked-in scripts.
- Do not label a PR merge-preview SHA as a post-merge main SHA.

Exit gate: immutable experiment manifest and exact evidence revision for each claim.

### Phase B — Dense operator and loss contracts
- Expand finite-difference tests across supported dense activations and loss/activation pairs.
- Test non-square matrices, multiple batch sizes, zero inputs, output-width variation, boundary targets, and numerical stability.
- Match the scalar loss reduction exactly; test multiple epsilon values and record normalized L2 and maximum absolute error.
- Add deterministic parameter-update tests against independently computed gradients and learning rates.

Exit gate: every claimed dense configuration has an independent oracle and declared tolerances.

### Phase C — Layer-family forward fidelity
- Add independent references for convolution, pooling, flatten, batch normalization, RNN, LSTM, GRU, embedding, attention, and residual operators only where implemented.
- For recurrent layers, compare full state trajectories, not only final outputs.
- Record accepted/rejected shapes and invalid transitions.
- Do not treat a registry entry as proof of end-to-end support.

Exit gate: coverage matrix labels each operator as source-inspected, forward-tested, gradient-tested, training-tested, traced, and UI-verified.

### Phase D — Provenance and replay
- Validate graph ID, graph revision, execution ID, operation ID, input/dataset identity, dtype, and provenance status in response envelopes.
- Test stale responses, overlapping runs, graph rebuilds, missing/mismatched weights, stop/restart, WebSocket reconnect, and out-of-order events.
- Ensure illustrative fallbacks are visibly labelled and cannot masquerade as live model state.
- Serialize and reconstruct deterministic executions; compare tensors and operation records.

Exit gate: adversarial tests show stale or illustrative state is not silently represented as current live execution.

### Phase E — Training semantics and convergence
- Test MSE and BCE contracts beyond current fixed cases.
- Test optimizer equations, loss reduction, parameter mutation, and reproducibility.
- Use small deterministic synthetic tasks with fixed seeds; report loss curves and failures.
- Limit claims to the optimizer/loss/layer combinations actually evaluated.

Exit gate: reproducible training tests support only measured configurations.

### Phase F — Trace and runtime cost
- Benchmark multiple graph sizes and operator mixes.
- Separate graph construction, plain forward execution, trace-enabled execution, step selection, serialization, API transport, and browser rendering.
- Use warm-up, repeated independent processes, raw samples, median/dispersion, trace bytes, peak memory, and environment metadata.
- Compare full-trace, selected-layer, selected-step, and summary-only modes if implemented.
- Do not call step selection incremental if it recomputes a full trace.

Exit gate: reproducible benchmark data and scripts generate each reported cost figure/table.

### Phase G — Related-work protocol
- Search IEEE Xplore, ACM Digital Library, IEEE VIS proceedings, and relevant publisher/author pages through the submission cutoff.
- Record queries, dates, inclusion/exclusion rules, and screening decisions.
- Read closest papers in full; compare analytical task, model family, execution coupling, trace granularity, provenance, validation, benchmark scale, and user evaluation.
- Claim only a gap that survives this comparison.

Exit gate: comparison matrix is traceable to primary papers; novelty language matches evidence.

### Phase H — Manuscript, venue, and artifact
- Separate implemented behavior from proposed recommendations.
- Separate measured results from unexecuted protocols.
- Cover construct, internal, external, and reproducibility threats.
- Generate figures/tables from source and inspect every compiled page.
- Select a venue before camera-ready formatting: IEEEtran conference style is not a universal IEEE journal template. IEEE Xplore and Scopus are not a single venue or page limit.
- Verify the selected venue’s scope, page limit, review expectations, and artifact policy.

Exit gate: clean build, frozen source, reproducible results, completed comparison protocol, and venue-specific compliance.

## 5. Publication readiness

The revised nine-page IEEEtran conference-format draft is a credible working implementation manuscript, not yet a submission-ready top-tier TVCG paper. It has TikZ figures, system description, limitations, an initial comparison table, and narrow CI-backed numerical evidence. It still lacks broad operator coverage, provenance/replay tests, performance measurements, a controlled user study, and a systematic literature review.

A top-tier visualization venue is likely to require a sharper task-centered contribution, evidence that displayed values are tied to the correct execution, adversarial stale-state tests, reproducible multi-size performance characterization, and a comparative user study if user effectiveness is claimed. If these experiments are not feasible, narrow the claims and choose a venue that accepts an implementation/engineering contribution. Do not present unmeasured performance or user benefit as results.

## 6. Immediate next actions

1. Update the evidence ledger to record PR #22 as merged and describe merge-preview evidence accurately.
2. Freeze current main and rerun targeted tests against that exact checkout.
3. Expand dense activation/loss tests, including BCE stability and output-width semantics.
4. Add deterministic parameter-update and training-convergence tests.
5. Audit API/frontend/WebSocket provenance and fallback paths.
6. Build the operator-family coverage matrix.
7. Expand the literature matrix and document the systematic search.
8. Finalize empirical results and choose the submission venue only after the evidence gates are met.
