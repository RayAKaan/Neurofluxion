# Neurofluxion Research Status

**Status date:** 2026-10-19
**Evidence revision (frozen):** `1ab65d16e20ccbb65b70598112bf5dd605c6179b`
**Application source unchanged by this research phase.** Every result below was
produced by research tooling committed at or before that revision.

This document supersedes the scattered status notes in `research/README.md`,
`research/VALIDATION_MATRIX.md`, `research/CROSS_PLATFORM_BENCHMARK.md` and
`research/paper/COMPLETION_PLAN.md`. Those documents remain as the historical
record of how the work arrived at this point; where they disagree with this file,
this file is correct.

---

## 1. What was executed

Eleven experiments, all writing immutable evidence bundles to
`research/evidence/<experiment_id>/<source_sha>/` with a checksummed
`manifest.json` recording the command, environment, interpreter and dependency
versions, git state, seed and configuration.

| ID | Script | Purpose | Result |
|---|---|---|---|
| EXP-01 | `run_graph_capability_audit.py` | structural capability probes | 47 probes, 6 expectation violations |
| EXP-02 | `run_gradient_validation.py` | finite-difference gradient validation | 364 comparable tensors, **0 failures** |
| EXP-03 | `run_trace_fidelity_audit.py` | forward and trace fidelity vs an independent reference | 20 cases, max error **1.11e-07** |
| EXP-04 | `run_training_update_validation.py` | optimizer update contracts and trajectories | 100 tensor checks, **0 failures** |
| EXP-05 | `run_performance_benchmark.py` | decomposed cost characterisation | 14 workloads x 60 samples |
| EXP-06 | `run_scalability_study.py` | one-factor scaling sweeps | 7 sweeps, 36 points |
| EXP-07 | `run_determinism_study.py` | in-process and cross-process determinism | all identical |
| EXP-08 | `run_defect_register.py` | reproduced runtime defects | **7 confirmed** |
| EXP-09 | `run_execution_path_audit.py` | execution-path and synthetic-value audit | 106 routes, 5 audits |
| EXP-10 | `run_literature_screening.py` | literature metadata audit and comparison matrix | 2,076 records, **0 export failures** |
| EXP-11 | `run_claim_audit.py` | claim/evidence ledger and overclaim scan | 17 claims adjudicated |

Shared research library: `research/lib/` — provenance capture, immutable
evidence writing, descriptive statistics with sample-size guards, numerical
comparison helpers, the canonical case registry, and **independent float64
reference implementations** of every supported operator written from the
standard mathematical definition rather than copied from the runtime.

## 2. Operator coverage

20 registered graph cases across seven families
(`research/lib/cases.py`), plus one declared-unsupported case:

- **dense** (5): relu stack, tanh/leaky stack, sigmoid+BCE, deep 4-hidden stack, batched (declared unsupported)
- **convolution** (5): valid, `same`+stride 2 with BatchNorm, smooth-activation control, max-pool, avg-pool
- **normalisation** (2): vector (degenerate), spatial after convolution
- **recurrent** (4): RNN, LSTM, GRU, LSTM with `return_sequences`
- **embedding/attention** (2): single-head composite, two-head request
- **residual** (1)
- **composite** (2): conv-pool-dense-residual, embedding-attention-LSTM

## 3. Numerical validation

**Method.** Central differences of the implementation's own scalar loss, swept
over five step sizes (1e-2 to 1e-4), compared per parameter tensor against both
shipped gradient implementations (`simulator.backward_engine` and
`simulator.training_engine`), under both packed-array and layer-attribute
parameter access.

**Result (EXP-02).**

| Quantity | Value |
|---|---|
| Comparable parameter-tensor checks | 364 (of 424 total) |
| Failures | **0** |
| Relative L2 error, median | **3.68e-06** |
| Relative L2 error, mean | 4.13e-04 |
| Relative L2 error, p95 | 1.17e-04 |
| Relative L2 error, max | 3.52e-02 |
| Max absolute error, median | 1.01e-06 |
| Max absolute error, max | 1.23e-04 |
| Inconclusive (not resolvable by a float32 objective) | 18 |
| Degenerate (numerically zero gradient, uninformative) | 24 |
| Parameter-sync unverified | 18 |

**Forward and trace fidelity (EXP-03).** Against the independent float64
reference: 20 supported cases, maximum final-output absolute error
**1.11e-07**, maximum per-layer absolute error **4.51e-07**, maximum trace
final-output error **1.11e-07**. Every declared trace property classified
`EXACTLY_VALIDATED` for all 20 cases.

**A methodological finding worth stating explicitly.** A ReLU configuration with
a large occupied kink band cannot be validated by central differences. In
`conv1_6x6_2_same_s2_batchnorm_flatten_dense`, 18 of 50 convolution
pre-activations lie within the largest perturbation step, and the numeric
conv-bias gradient converges to a value that disagrees with the analytic one by a
relative L2 of 1.88. The smooth-activation control
(`conv1_6x6_2_same_s2_batchnorm_flatten_tanh`) has zero elements in the kink
band and agrees to 1.2e-04. The apparent gradient defect is therefore an
artefact of the estimator, and the repository reports the kink occupancy rather
than the mismatch.

## 4. Training

**Optimizer update contracts (EXP-04).** SGD, momentum, RMSProp and Adam,
compared against independently implemented reference equations.

| Quantity | Value |
|---|---|
| Update measurements (cases x optimizers) | 19 |
| Parameter-tensor update checks | 100 |
| Failed update contracts | **0** |
| Relative update error, median | 6.37e-08 |
| Relative update error, max | 1.36e-03 |
| Tensors whose expected update is below the float32 measurement floor | 14 |
| Measurements that reduced the loss | 19 / 19 |
| Layer-propagation inconsistencies | **2** |

The two inconsistencies are D-01: after a full optimizer step the packed
attention array changes while all four attention projections used by the forward
pass are unchanged.

**Trajectories, reported separately.** 76 short single-batch optimisations of a
fixed deterministic task, 25 steps each. All 76 reduced the loss; 36 were
monotone non-increasing; relative loss reduction min 0.087, median 0.975, max
1.000. This is loss reduction for the tested configurations. It is **not**
evidence of general convergence, generalisation or optimisation behaviour.

## 5. Performance and scaling

**Decomposed benchmark (EXP-05), 14 workloads, 60 samples per component after 10
warm-ups, environment captured in the manifest.**

The previously recorded "trace overhead" ratio is dominated by JSON
serialisation of the record payload, not by the traced arithmetic:

| Workload | forward (ms) | trace total (ms) | trace JSON (ms) | paired ratio median [95% CI] |
|---|---:|---:|---:|---|
| dense 8-16-4 | 0.058 | 0.114 | 0.448 | 1.97 [1.79, 2.13] |
| dense 16-32-8 | 0.080 | 0.158 | 1.398 | 2.02 [1.96, 2.25] |
| dense 32-64-16 | 0.089 | 0.353 | 4.737 | 3.68 [3.67, 4.33] |
| dense 64-128-32 | 0.086 | 0.663 | 18.49 | 7.93 [7.46, 8.46] |
| dense 128-256-64 | 0.109 | 2.354 | 74.56 | 21.74 [20.35, 23.03] |
| conv 1x8-16 | 0.678 | 0.925 | 3.657 | 1.37 [1.32, 1.46] |
| conv 1x16-32 | 2.214 | 5.191 | 21.77 | 2.06 [1.96, 2.27] |
| conv 2x32-32 | 2.918 | 6.103 | 49.29 | 1.97 [1.92, 2.25] |
| lstm t8-d16-h16 | 0.414 | 0.430 | 0.158 | 1.09 [1.03, 1.20] |
| gru t8-d16-h16 | 0.352 | 0.373 | 0.191 | 1.08 [1.02, 1.15] |
| lstm t32-d32-h32 | 1.224 | 1.367 | 0.674 | 1.10 [1.00, 1.19] |
| gru t32-d32-h32 | 1.108 | 1.178 | 0.693 | 1.09 [1.03, 1.19] |
| lstm t64-d64-h64 | 2.981 | 2.698 | 2.789 | 1.04 [0.97, 1.06] |
| gru t64-d64-h64 | 1.954 | 2.259 | 2.436 | 1.16 [1.10, 1.21] |

Paired ratios range from 1.04x to 21.7x. Effect sizes (Cohen's d_z for paired
samples) range from -0.06 to 6.04; three recurrent workloads have intervals that
include 1.0 and near-zero effect sizes, because their layer-level traces carry
few values while their forward pass dominates. Dense graphs show up to 21.7x
because their operation-level traces serialise every intermediate tensor.

**Scaling sweeps (EXP-06), one factor at a time, fitted in log-log space and
reported with R-squared and tested range only.**

| Sweep | Tested range | forward exponent (R²) | trace exponent (R²) | trace payload exponent (R²) |
|---|---|---|---|---|
| depth | 1-10 | 0.09 (0.34) | 0.63 (0.89) | 0.91 (0.999) |
| width | 8-256 | 0.06 (0.35) | 0.76 (0.85) | 1.40 (0.989) |
| sequence length | 4-128 | 0.82 (0.991) | 0.89 (0.963) | 0.87 (0.996) |
| features | 8-128 | 0.02 (0.02) | 0.16 (0.84) | 0.84 (0.996) |
| spatial size | 8-64 | 1.43 (0.955) | 1.58 (0.989) | 1.99 (0.99999) |
| filters | 4-64 | -0.02 (0.02) | 0.36 (0.91) | 0.95 (0.999) |

The forward-pass exponent has a low R-squared for depth, width, features and
filters: over these ranges the forward pass is dominated by fixed per-layer
overhead rather than by the swept factor, so **no forward-pass scaling law is
claimed for those sweeps**. The trace and payload exponents are well fitted
(R² >= 0.84) and are the defensible result.

Quantities that do not vary with their sweep factor are reported as invariant
and are not fitted. Batch dimension and depth above the validator's layer-count
ceiling are recorded as boundary probes: batch 1 is supported, batches 2/4/8 and
depth 16 are not.

## 6. Determinism and reproducibility

**EXP-07.** 20 cases x 5 repeats for parameter construction, forward output,
serialised trace and a short training trajectory: **all bit-identical**. Three
cases repeated across five fresh interpreter processes produced identical
parameter and trace digests. `seed_insensitive_cases` = `batchnorm_vector_5_dense`,
independently corroborating the EXP-01 degeneracy finding: that graph's output
does not depend on initialisation at all.

Every evidence bundle records the exact source SHA, the full command line, git
branch and dirty state, OS, CPU, memory, interpreter and dependency versions, the
seed, the configuration, and SHA-256 checksums of every emitted file.

## 7. Reproduced runtime defects

**EXP-08.** Seven defects reproduced with minimal reproductions and measured
impact. **None is fixed**; fixing any of them requires explicit authorisation to
modify runtime source.

| ID | Severity | Defect |
|---|---|---|
| D-04 | **BLOCKING** | `POST /api/simulator/forward/full` declares `input: List[float]`; spatial and sequence graphs need nested arrays, so those graphs cannot execute through the route even though the engine accepts them directly. |
| D-01 | HIGH | Attention projection updates never reach the attention layer: the packed array changes, the four projections do not. Attention does not train. |
| D-03 | HIGH | BatchNorm degenerates on vector inputs: output equals beta for every input and the input gradient is exactly zero, so nothing below the layer can train through it. |
| D-02 | MEDIUM | `num_heads` is accepted and stored but never used; one-head and two-head graphs with identical projections produce identical outputs. Multi-head attention is not implemented. |
| D-05 | MEDIUM | `architecture_hash` is derived from neuron counts only and collides across distinct architectures. |
| D-07 | MEDIUM | `'same'` padding with stride > 1 does not produce the conventional `ceil(in/stride)` output size (5x5 instead of 3x3 at input 6x6, stride 2). |
| D-06 | INFORMATIONAL | The dense path has no batch dimension: a (3,3) input cannot be executed. Recorded as a declared boundary. |

## 8. Execution paths and synthetic values

**EXP-09.** 106 routes across six families: simulator graph session (57),
framework Lab trace (13), model-registry inference (9), legacy execution engine
(5), framework training (4), illustrative stream (1), other (17).

- Each path owns separate state. No verified import path links simulator graphs
  to registry weights.
- The `/stream` payload is a fixed 3-4-2 topology with a hard-coded weight table
  and closed-form loss/accuracy curves of a counter. It reads no configured
  graph, session or dataset.
- The Simulator demo requests a `softmax` output activation that the activation
  registry does not implement; the fallback is linear, so the displayed outputs
  are logits, not probabilities.
- The legacy `ExecutionEngine` dense backward pass was independently validated
  against central differences (3 checks, relative L2 2.1e-05 to 7.4e-05). Its
  `build_graph` seeds from an unseeded generator, so its initial weights are not
  reproducible.

## 9. Literature

**EXP-10.** The earlier recorded export covered only Crossref because all six
OpenAlex runs failed after page 1 (HTTP 401 on the cursor follow-up). The
exporter now uses page-based OpenAlex pagination, which does not require an API
key.

| Quantity | Value |
|---|---|
| Databases | OpenAlex, Crossref |
| Queries | 6 |
| Query/database runs | 12, **0 failures** |
| Raw records | 2,400 |
| Deduplicated records | 2,076 |
| Records found by both databases | 209 |
| Blank titles | 4 |
| Blank abstracts | 988 (**52.4%** abstract completeness) |
| Blank DOIs | 132 (93.6% DOI completeness) |
| Publication years | 1964-2026 |
| Distinct venues | 1,043 |
| False-merge candidates | **0** |
| Residual duplicate title+year groups needing adjudication | 48 |
| Triage: high / medium / low | 445 / 583 / 1,048 |
| Screening decisions made | **0** |

The triage is a versioned keyword rule that prioritises human review. It writes
no include/exclude decision and no PRISMA-style count may be derived from it.
This remains a **structured rapid review**, not a systematic review.

The comparison matrix covers 13 systems across 12 capability dimensions with an
explicit value vocabulary. Only the Neurofluxion row is backed by this
repository's own executed experiments.

## 10. Claim disposition

**EXP-11.** 17 ledger entries: **6 SUPPORTED, 4 PARTIALLY_SUPPORTED, 2
UNSUPPORTED, 5 INVALIDATED.**

| Status | Claims |
|---|---|
| SUPPORTED | C-01 separate execution paths, C-02 forward agreement with an independent reference, C-07 characterised inspection cost, C-08 determinism/reproducibility, C-15 paths do not share model state, C-16 some values are synthesised |
| PARTIALLY_SUPPORTED | C-03 gradient agreement, C-04 trace fidelity, C-05 optimizer update, C-06 loss reduction |
| UNSUPPORTED | C-13 user benefit, C-14 novelty/priority/superiority |
| INVALIDATED | C-09 universal operator correctness, C-10 provenance integrity, C-11 multi-head attention, C-12 attention parameters train, C-17 framework equivalence for convolution |

The overclaim scan classified 506 occurrences of priority/completeness/
generality/performance terms: 338 in prose, 168 as code identifiers or literals,
173 requiring human adjudication. It is a candidate generator, not a verdict.

## 11. Reproduction

```bash
# Environment: Python 3.12, NumPy 2.4.6, TensorFlow 2.21.0, pytest 9.1.1
export PYTHONPATH="$PWD/backend:$PWD/research"

python research/experiments/run_graph_capability_audit.py      --repo .
python research/experiments/run_gradient_validation.py        --repo .
python research/experiments/run_trace_fidelity_audit.py       --repo .
python research/experiments/run_training_update_validation.py --repo .
python research/experiments/run_defect_register.py            --repo .
python research/experiments/run_execution_path_audit.py       --repo .
python research/experiments/run_determinism_study.py          --repo .

# Timing-sensitive: run these with nothing else executing.
python research/experiments/run_performance_benchmark.py      --repo .
python research/experiments/run_scalability_study.py          --repo .

# Literature (live search) and derived artefacts
OPENALEX_MAILTO="<contact>" python research/experiments/literature_search_export.py --out ./export
python research/experiments/run_literature_screening.py --repo . --export ./export
python research/experiments/run_claim_audit.py                --repo .
python research/experiments/build_figures_and_tables.py       --repo .
```

`build_figures_and_tables.py` reports an unavailable figure rather than
approximating one when its evidence bundle is missing.

## 12. Standing limitations

1. Single machine, single process for all timing. Hosted-CI timings recorded in
   older documents are a different environment and must not be pooled with these.
2. `tracemalloc` peak is Python allocation peak, not resident set size.
3. No browser rendering, HTTP transport or concurrent-load cost is measured.
4. Browser provenance defects remain reproduced expected failures; fixing them
   needs runtime authorisation.
5. Literature discovery is metadata-only from two APIs, with 52.4% abstract
   completeness and no full-text eligibility stage.
6. Twenty fixed cases do not establish coverage of arbitrary shapes, losses,
   batch sizes or optimisations.