# Neurofluxion — Final Research Readiness Audit

**Audit date:** 2026-10-19
**Frozen evidence revision:** `750d5ab4a92c097391d669e81a70acb4ab8aaa76`
**Policy:** runtime and application source are read-only for this research phase.
Seven runtime defects are reproduced, quantified and documented; none is fixed.

This audit answers the questions required before the manuscript is written. It is
deliberately negative where the evidence is negative.

---

## 1. Scientific validity

**Experimentally demonstrated.**

- The NumPy simulator's forward computation agrees with an independent float64
  reference implementation across all 20 registered supported graph cases, to a
  maximum absolute error of 1.11e-07 (final output) and 4.51e-07 (per layer).
- Analytic gradients from **both** shipped gradient implementations
  (`backward_engine` and `training_engine`) agree with central differences over
  364 comparable parameter tensors, with 0 failures, a median relative L2 error
  of 3.68e-06 and a maximum absolute error of 1.23e-04.
- Operation-level trace records for dense-only graphs are reconstructable from
  their own recorded inputs, snapshots and activation names; layer-level trace
  records for every other family match the independent reference at the layer
  boundary. Trace ordering, step selection and payload completeness were exactly
  validated for all 20 cases.
- One optimizer step applies the documented reference equation for SGD,
  momentum, RMSProp and Adam across 19 cases, with 0 failed tensor checks and a
  maximum relative update error of 1.36e-03.
- 76 short single-batch optimisations of a fixed deterministic task all reduced
  the loss (relative reduction 0.087 to 1.000, median 0.975).
- Execution is bit-repeatable in-process over 5 repeats for parameter
  construction, forward output, serialised trace and a short training
  trajectory, and across 5 fresh interpreter processes for 3 cases.
- The legacy `ExecutionEngine` dense backward pass agrees with central
  differences independently of the graph-session engine.

**Partially demonstrated.**

- Gradient agreement is scoped per case, parameter tensor, dtype, step size and
  tolerance. 24 tensors have a numerically zero gradient and carry no
  information; 18 are marked `PARAM_SYNC_NOT_VERIFIED`; 18 are marked
  inconclusive.
- Trace fidelity is operation-level for dense-only graphs and layer-boundary for
  all others. The two are not interchangeable and must not be pooled.
- Cost characterisation is single-machine and single-process.

**Not demonstrated.**

- General convergence, generalisation, optimisation behaviour, or learning
  benefits of any kind.
- Cross-machine numerical or timing reproducibility.
- Correctness for shapes, losses, batch sizes, optimisers or activations outside
  the registered cases.
- Any browser-visible provenance invariant.
- Systematic literature coverage, dual screening, or full-text eligibility.

## 2. Numerical validation

| Item | Value |
|---|---|
| Registered graph cases | 20 supported + 1 declared unsupported |
| Families | dense, convolution, normalisation, recurrent, embedding/attention, residual, composite |
| Operators exercised | dense, Conv2D, MaxPool2D, AvgPool2D, Flatten, BatchNorm, RNN, LSTM, GRU, Embedding, Attention, Residual |
| Activations | relu, leaky_relu, tanh, sigmoid, linear |
| Losses | mean 0.5·(y−ŷ)², clipped mean BCE |
| dtypes | float32 forward; float64 for the independent reference and error metrics |
| Seeds | one fixed seed per case, recorded in the manifest |
| Tolerances | per case, declared before execution in `research/lib/cases.py` (typically atol 2e-3, rtol 2e-2) |
| Epsilon sweep | 1e-2, 3e-3, 1e-3, 3e-4, 1e-4 |

**Observed error distributions (364 comparable tensors).**

Relative L2 error: median 3.68e-06, mean 4.13e-04, p90 7.53e-05, p95 1.17e-04,
p99 1.33e-02, max 3.52e-02.
Max absolute error: median 1.01e-06, mean 8.36e-06, max 1.23e-04.

**A methodological result the manuscript should state.** Finite differences
cannot validate a ReLU network whose pre-activations occupy the perturbation
band. In `conv1_6x6_2_same_s2_batchnorm_flatten_dense`, 18 of 50 convolution
pre-activations lie within ±1e-2 of the ReLU kink, and the numeric conv-bias
gradient converges to a value disagreeing with the analytic one by relative L2
1.88. The smooth-activation control has 0 elements in the band and agrees to
1.2e-04. The repository reports kink occupancy rather than the mismatch, and
marks affected comparisons inconclusive.

## 3. Training

**Validated:** the optimizer applied the documented equation to the supplied
gradient, for SGD, momentum, RMSProp and Adam across 19 cases (100 tensor
checks, 0 failures); the loss decreased in 19 of 19 measured updates; the loss
decreased in 76 of 76 short trajectories.

**Not validated:** multi-batch accumulation, shuffling, dropout under training,
L2 regularisation effects, learning-rate schedulers, convergence on any real
dataset, generalisation, or stability beyond the tested ranges.

**Known defect inside the validated surface:** D-01. The attention projections
never change during training because the packed parameter array is not
propagated into the attention layer. The one-batch attention training contract
observes the packed array and therefore passes while the computation is
unaffected. The attention path is **not** demonstrated to train.

## 4. Performance

- **Trials:** 60 raw samples per component per workload after 10 warm-ups,
  across 14 workloads and 10 measured components; 25 samples per scaling point
  after 5 warm-ups across 36 points.
- **Hardware/environment:** captured in every manifest — a single Windows
  workstation, Python 3.12.10, NumPy 2.4.6, TensorFlow 2.21.0, CPU only. No
  discrete GPU was used or claimed.
- **Overhead:** paired trace/forward ratio 1.02x to 23.3x. Dense graphs dominate;
  recurrent workloads sit near 1.1x with confidence intervals that include 1.0
  and effect sizes near zero.
- **Attribution:** the residual is dominated by JSON serialisation of the record
  payload. For `dense_128_256_64`, trace construction is 2.81 ms while JSON
  serialisation alone is 96.2 ms. Recurrent workloads invert this: their
  forward pass dominates and serialisation is a small fraction.
- **Scaling:** trace-cost and payload exponents are well fitted (R² ≥ 0.85)
  across every tested sweep. Forward-pass exponents are well fitted only for
  depth, sequence length, spatial size and filters; for width (R² 0.69) and
  features (R² 0.56) the forward pass is dominated by fixed per-layer overhead,
  so **no forward-pass scaling law is claimed** for those two sweeps.
- **Limitations:** tracemalloc peak is Python allocation peak, not resident set
  size. No HTTP transport, browser rendering, concurrency or cold-start cost is
  measured. Absolute timings from the older hosted-CI runs in
  `CROSS_PLATFORM_BENCHMARK.md` are a different environment and must not be
  pooled with these.

## 5. User interface

- **Validated:** the backend renders a trace, and the browser E2E test
  `simulator-flow.spec.mjs` matches the entire rendered trace against a captured
  backend payload element by element. The Simulator page displays a global note
  stating that a random forward pass was run.
- **Reproduced defects, not fixed:** two adversarial browser tests remain
  expected failures. `stale-forward-provenance.spec.mjs` reproduces a stale
  forward response being displayed after the selected graph changed, and
  reproduces previous-graph inspection state surviving graph replacement.
  These are real provenance defects.
- **Severity for the manuscript:** HIGH for the system as a whole, because the
  central inspectability claim is "a displayed value belongs to the selected
  graph, execution and parameter revision". Two independent scenarios violate
  that invariant. Claim C-10 is marked INVALIDATED.
- **Consequence:** the paper cannot claim provenance integrity. It can claim
  that the defect is reproduced and characterised.
- **Runtime constraint:** fixing these requires authorising runtime changes.
  Under the current policy they remain documented limitations.

## 6. Literature

- **Databases:** OpenAlex and Crossref metadata APIs.
- **Queries:** 6 fixed queries; 12 query/database runs.
- **Export status:** complete. **0 failures** after the OpenAlex pagination fix.
- **Records:** 2,400 raw → **2,076 deduplicated**; 209 returned by both
  databases; 0 false-merge candidates; 48 residual duplicate title+year groups
  still requiring manual adjudication.
- **Metadata completeness:** 4 blank titles, 988 blank abstracts (**52.4%**
  abstract completeness), 132 blank DOIs (93.6% DOI completeness), years
  1964-2026, 1,043 distinct venues, 257 records without a venue.
- **Screening:** **0 decisions made.** A versioned keyword triage prioritises
  445 high / 583 medium / 1,048 low. Triage is not screening.
- **Included set:** 13 systems in a source-linked comparison matrix over 12
  capability dimensions.
- **What remains uncertain:** no IEEE Xplore, ACM DL, Scopus or Web of Science
  export; no dual independent screening; no full-text eligibility stage; no
  citation snowballing pass recorded as a completed protocol.
- **Status: structured rapid review. Not a systematic review.** No PRISMA
  counts exist and none may be reported.

## 7. Reproducibility

**Can the experiments be reproduced?** Yes. Every experiment is a single
command against a pinned checkout; §11 of `research/RESEARCH_STATUS.md` lists
them with their required environment.

- **SHA:** all eleven evidence bundles record
  `750d5ab4a92c097391d669e81a70acb4ab8aaa76`, the revision whose committed code
  produced them, from a clean working tree.
- **Environment:** OS, CPU, memory, interpreter and dependency versions recorded
  in every manifest.
- **Determinism:** independently re-measured by EXP-07, and re-measured again on
  Ubuntu 24.04 hosted runners by the `evidence-validation` workflow, where three
  cases again passed in-process and cross-process.
- **Integrity:** SHA-256 checksums for every emitted file in every manifest,
  verified on every CI run (11 bundles, 56 files, 0 problems).
- **Pinning:** every bundle's recorded source revision is verified on every CI
  run to contain the script that produced it (11 bundles, 0 problems).
- **Immutability:** evidence directories are keyed by experiment and source SHA;
  the writer refuses to overwrite an existing revision directory.

## 8. Claims

**Supported (6).** Separate execution paths; forward agreement with an
independent reference; characterised inspection cost; determinism and
reproducibility; no shared model state between paths; some user-visible values
are synthesised.

**Partially supported (4).** Gradient agreement; trace fidelity; optimizer
update contracts; loss reduction on fixed tasks. Each is scoped in the ledger by
case, parameter, tolerance and task.

**Unsupported (2).** User benefit (no study); novelty, priority or superiority
(no systematic review).

**Invalidated by reproduced evidence (5).** Universal operator correctness;
provenance integrity of displayed values; multi-head attention support;
attention parameters training; framework equivalence for convolution.

**Claimed nowhere and not tested.** Nothing in the manuscript should assert
usability, learning benefit, or priority.

## 9. Remaining blockers

Ordered by how much they cap the manuscript.

1. **D-04 (BLOCKING) — the forward trace route cannot execute spatial or
   sequence graphs.** `input: List[float]` cannot express a (1,5,5) input. The
   headline workflow "construct a graph → forward pass → inspect trace" is not
   reachable for convolutional or recurrent architectures through the
   application route, only through the engine API. Until this is fixed the paper
   cannot claim route-level coverage of those families.
2. **D-01 (HIGH) — attention does not train.** Any claim about the attention
   path learning is unsupported.
3. **D-03 (HIGH) — BatchNorm degenerates on vector inputs.** Any graph placing
   BatchNorm on a vector silently loses all gradient below it.
4. **Browser provenance defects (2 reproduced expected failures).** The
   inspectability claim is currently contradicted.
5. **D-07 (MEDIUM) — non-standard `same` padding with stride > 1.**
   Framework-equivalence claims for convolution are unsupported.
6. **No user study.** Usability and learning claims are permanently out of scope
   unless a human-subject protocol is run.
7. **Single-environment timing.** A second hardware class would strengthen the
   performance section but does not block a bounded claim.
8. **Literature completeness.** Metadata-only discovery with 52.4% abstract
   completeness. Sufficient to avoid priority claims; insufficient to claim
   systematic coverage.

## 10. Readiness verdict

**RESEARCH-READY WITH DOCUMENTED LIMITATIONS — PAPER GENERATION CAN BEGIN.**

The repository now contains executed, checksummed, source-pinned evidence for
forward fidelity, gradient correctness, optimizer update behaviour, trace
fidelity, performance decomposition, scaling, determinism, runtime defects,
execution-path separation and literature discovery, plus a machine-readable claim
ledger that explicitly invalidates five claims a reader might otherwise assume.

This is **not** "research-ready" without limitations. Three HIGH-or-worse defects
and two reproduced browser provenance defects directly bound what may be claimed,
and one of them (D-04) bounds the headline workflow for two of the three headline
layer families. The manuscript must be written inside those bounds.

## 11. Recommendation to the manuscript author

Write the paper as an **implementation and evidence study with a defect
register**, not as a system claim.

- Lead with the measurement infrastructure and the independent reference
  implementation; that is the transferable contribution.
- Present the operator-family coverage table and the defect register together.
  Reporting seven reproduced defects with measurements is a stronger scientific
  position than claiming correctness.
- State the numerical-validation methodology including the ReLU kink-band
  result. It shows the validation is calibrated rather than assumed.
- Present the decomposed cost result, not the old trace/forward ratio. The
  finding that serialisation, not tracing arithmetic, dominates the inspected
  cost is the defensible and more interesting one.
- Do not claim provenance integrity. Report the two reproduced browser failures
  and the recommended remediation.
- Avoid priority, superiority, usability and learning language entirely; the
  ledger marks those claims UNSUPPORTED.