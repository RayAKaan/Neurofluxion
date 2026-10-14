# Neurofluxion — Final Post-Merge Evidence Audit

**Audit date:** 2026-10-14  
**Repository:** `RayAKaan/Neurofluxion`  
**Frozen descriptive snapshot:** `d85aca48e7993c3b48c951c97480aaa6e0c2934a` (merge commit for PR #30)  
**Policy:** application/runtime code is read-only for this research phase. Research artifacts and manuscript may be updated; no runtime behavior was changed.

## Executive result

PRs #24–#30 are merged. The latest recorded `main` commit is `d85aca48e7993c3b48c951c97480aaa6e0c2934a`. The manuscript was reconciled to this source snapshot and rebuilt locally as an 11-page IEEEtran conference-format PDF. Three core diagrams were redrawn in TikZ, and a fourth measured-results graphic visualizes the trace-cost ratios. The LaTeX build completed successfully.

**Important boundary:** there is no evidence that one complete backend + frontend + browser + research suite was executed together on merge SHA `d85aca48e7993c3b48c951c97480aaa6e0c2934a`. Individual workflows passed on their recorded PR heads. The paper preserves that distinction and is a consolidated manuscript for the available evidence, not a claim of universal correctness or automatic publication readiness.

## PR and workflow reconciliation

| PR | Change scope | Latest inspected workflow evidence | Conclusion |
|---|---|---|---|
| #24 | Full-stack validation | [Run 38505557136](https://github.com/RayAKaan/Neurofluxion/actions/runs/38505557136): backend tests, frontend build, browser E2E all passed | Passed on PR #24 head `46ae01a…`; predates later changes |
| #25 | Legacy execution-engine numerical contracts | [Run 38493639390](https://github.com/RayAKaan/Neurofluxion/actions/runs/38493639390): frontend build, full backend suite, execution contracts passed; [research run 38493639458](https://github.com/RayAKaan/Neurofluxion/actions/runs/38493639458) passed | Branch-specific evidence; distinct from custom simulator guarantees |
| #26 | BatchNorm per-feature gradients | [Run 38493941513](https://github.com/RayAKaan/Neurofluxion/actions/runs/38493941513): BatchNorm contract and full backend suite passed; [research run 38493941567](https://github.com/RayAKaan/Neurofluxion/actions/runs/38493941567) passed | Focused operator and backend checks passed on PR head |
| #27 | Conv2D, pooling, flatten | [Run 38495772620](https://github.com/RayAKaan/Neurofluxion/actions/runs/38495772620) passed | Narrow operator contracts, not all graph-level combinations |
| #28 | Optimizer contracts and invalid-name rejection | [Run 38495430720](https://github.com/RayAKaan/Neurofluxion/actions/runs/38495430720): optimizer contracts and full backend suite passed; [research run 38495430548](https://github.com/RayAKaan/Neurofluxion/actions/runs/38495430548) passed | Run reported 62 backend tests passed and 12 opt-in heavy-inference tests skipped |
| #29 | Attention, recurrent, embedding, residual | [Run 38494730679](https://github.com/RayAKaan/Neurofluxion/actions/runs/38494730679) passed | Short-sequence and small-operator checks only |
| #30 | Research evidence consolidation | [Run 38506053406](https://github.com/RayAKaan/Neurofluxion/actions/runs/38506053406) passed on PR head `0864ec3…` | Dense research audit passed; not the full-stack suite |

These are separate workflow executions and must not be summed as one test suite. Workflow success is interpreted alongside the job scope and recorded test reports.

## Source-level findings confirmed for the paper

1. **Distinct execution paths:** the custom NumPy-oriented graph simulator, legacy execution API, and TensorFlow/Keras model/inference path must not be treated as interchangeable.
2. **Activation boundary:** the activation registry implements ReLU, sigmoid, tanh, leaky ReLU, and linear. Unknown activation names fall back to linear. The default simulator demo requests `softmax`, which is not registered; its outputs must not be presented as normalized class probabilities.
3. **Heterogeneous trace depth:** dense-only forward execution records matrix multiplication, bias addition, and activation as separate events. Other layer instances are generally recorded as one layer-level event.
4. **Step selection is not incremental execution:** the step API recomputes a full forward trace before selecting the requested event.
5. **In-memory graph lifecycle:** graph sessions are held in memory; graph IDs are not durable experiment identities or proof of cross-process persistence.
6. **Illustrative visual fallback:** absent or incompatible weight payloads can result in visually populated fallback diagrams. Such values must not be mistaken for live parameter state.
7. **Separate stream semantics:** a stream path uses a fixed topology and synthesized/epoch-varying values, so event delivery alone does not prove the selected graph's live numerical state.

These are source findings, not claims that every corresponding failure mode has an end-to-end regression test.

## Numerical and operator evidence currently available

- Fixed dense forward oracle: maximum absolute error reported as `0.0` for the tested 2–3–2 graph.
- Dense trace arithmetic: final output matched the fixed reference for the tested case; repeated output/trace hashes matched.
- MSE and sigmoid-output BCE: selected central finite-difference gradient contracts passed in recorded workflows.
- Training/optimizer: one small separable BCE task reduced loss under a fixed SGD setup; selected SGD/Adam and momentum/RMSProp reference contracts passed.
- BatchNorm: finite-difference checks for input, gamma, and beta passed in the focused workflow; full backend suite passed on the PR head.
- Conv2D/pooling/flatten: fixed small operator contracts passed.
- Attention/RNN/LSTM/GRU/embedding/residual: selected small operator and short-sequence contracts passed.
- Legacy execution path: PR #25's execution contract, backend suite, and frontend build passed on its PR head.
- Browser path: PR #24's backend tests, frontend build, and Simulator demo/trace-to-UI E2E passed on its PR head.

The evidence supports only the configurations explicitly tested. It does not establish every operator's graph integration, all optimizer/loss/activation combinations, general convergence, replay correctness, or the correctness of every UI surface.

## Trace-cost evidence

Five hosted-runner CI runs measured forward-only execution against full dense trace construction across four fixed graph widths. Each run used five warm-ups and 40 raw samples per path/configuration; output parity passed. The median trace/forward ratio ranges were:

| Widths | Median ratio range |
|---|---:|
| `[4, 8, 2]` | 2.36–2.46× |
| `[16, 32, 8]` | 4.43–4.86× |
| `[32, 64, 16]` | 9.65–11.13× |
| `[64, 128, 32]` | 26.43–32.34× |

The trace path constructs and serializes operation records while the baseline returns final outputs, so the comparison is a diagnostic, not a causal or universal tracing-overhead estimate. All five runs used the same hosted-runner class; another environment and decomposed end-to-end timings remain outstanding.

## Manuscript and diagram validation

- Source: `Neurofluxion_Final_IEEE_Paper.tex`
- PDF: `Neurofluxion_Final_IEEE_Paper.pdf`
- Format: `IEEEtran` conference class, US Letter
- Build: `latexmk -pdf -interaction=nonstopmode -halt-on-error Neurofluxion_Final_IEEE_Paper.tex`
- Build result: exit code `0`; 11 pages
- Visuals: system-path architecture, trace/provenance contract, evidence workflow, and a TikZ trace-cost ratio plot
- Scope: source snapshot is the PR #30 merge commit; per-PR workflow evidence is not represented as a single post-merge full-stack run.

## Remaining evidence gates

1. Run backend tests, frontend build, browser E2E, and dense research audit against one frozen main revision, preserving commit SHA, environment, command, exit code, logs, and raw artifacts.
2. Run graph-integrated forward/backward/training tests across supported layer families and accepted/rejected shape transitions.
3. Add adversarial provenance checks for graph rebuilds, stale/out-of-order responses, missing weight payloads, reconnects, and replay.
4. Re-run trace benchmarks in another environment and decompose numerical execution, trace creation, serialization, API transport, and browser rendering.
5. Conduct a documented systematic literature search and screen the closest primary papers; current related work is targeted, not systematic.
6. If claiming usability or learning benefit, run an appropriate controlled study; otherwise keep those claims out of scope.
7. Select a specific venue and verify its current scope, length, formatting, artifact, and anonymization requirements.

## Final interpretation

The paper is a consolidated, reproducible-format implementation manuscript with explicit evidence boundaries and improved diagrams. It is not scientifically honest to call all research gates complete while the one-revision full-stack rerun, comprehensive graph-integrated operator matrix, second-environment benchmark, systematic literature review, and any user study remain unperformed.
