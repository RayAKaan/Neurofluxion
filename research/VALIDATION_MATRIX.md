# Validation Matrix and Remaining Evidence

**Latest main at this reconciliation:** `35b30de7807cc0ce522ced93ee7f3b1ea8e8e8aa` (research-file cleanup after PR #38, 2026-10-14).  
**Frozen validation target:** `30ff803cb342c494310ff35832df43a092d3d74d` (PR #31 merge commit, 2026-10-14). The integrated workflow in PR #32 checked out this exact base SHA in all four jobs. PR #32's workflow run [38566683696](https://github.com/RayAKaan/Neurofluxion/actions/runs/38566683696) passed backend tests, frontend build, browser E2E, and the deterministic dense research audit. PR #32 itself merged as `7c35cc131ad732972a3bdcde400548a02863193b`; that merge commit adds the workflow, while the successful test run is evidence for the frozen target SHA `30ff803...`, not for the new workflow file or later merge commit.

This matrix deliberately distinguishes implemented support from independently demonstrated correctness. A layer appearing in a registry or graph builder is not evidence that its forward, backward, optimizer, trace, API, and UI paths all work together.

| Area | Current evidence | Remaining validation | Paper wording allowed now |
|---|---|---|---|
| Dense forward | Independent fixed-parameter oracle passed for one small graph | Multiple widths/depths, activations, and seeds | Agreement for the named fixed case only |
| Dense MSE gradients | Central finite differences passed for simulator and TrainingEngine test cases | More activation and output-width combinations | Tested MSE configurations only |
| Dense sigmoid + BCE gradients | Central finite differences passed in simulator and TrainingEngine test cases | Extreme logits, clipping behavior, alternate output activations | Tested sigmoid/BCE configurations only |
| Training behavior | Tiny deterministic separable BCE task passed in PR #23 research-branch CI: loss decreased under fixed SGD setup; general convergence untested | Confirm loss reduction and repeatability on exact merged revision; add regression tests for other optimizers | No convergence claim until result is recorded |
| SGD / Adam / momentum / RMSProp | Exact SGD arithmetic and a finite first Adam update passed in PR #23; PR #28 merged reference-equation tests for momentum, RMSProp, two-step Adam, and rejection of unknown optimizer names. Run [38495430720](https://github.com/RayAKaan/Neurofluxion/actions/runs/38495430720) passed optimizer contracts and the full backend suite (62 passed, 12 opt-in heavy-inference tests skipped). | Broader optimizer/scheduler combinations and graph-integrated training remain untested | Tested equations/configurations only; no universal optimizer or convergence claim |
| Trace arithmetic | Matmul, bias-add, activation reconstruction passed in one dense case | Layer families and shape regimes; validate event input/output snapshots | Dense event types and tested case only |
| `run_forward_step` | Source inspection shows it calls `run_forward_full` and selects a step afterward | Measure cost of repeated step requests; consider cached trace only if design and correctness warrant it | Do not claim incremental execution |
| Convolution / pooling / flatten | Classes and graph construction exist; PR #27 added Conv2D finite differences plus MaxPool2D/AvgPool2D forward/backward contracts | Broader channels, strides, padding, activations, graph integration, and trace semantics | One small Conv2D case, one 2×2 case per pooling type, and one Flatten shape contract passed 9 focused tests in run [38495772620](https://github.com/RayAKaan/Neurofluxion/actions/runs/38495772620); not end-to-end coverage |
| Batch normalization | PR #26 merged per-feature gradient reductions. Run [38493941513](https://github.com/RayAKaan/Neurofluxion/actions/runs/38493941513) passed the BatchNorm contract and full backend suite; research validation [38493941567](https://github.com/RayAKaan/Neurofluxion/actions/runs/38493941567) passed. | More dimensions, edge cases, and graph-integrated training remain open | Focused finite-difference correctness on tested cases only |
| RNN / LSTM / GRU | Simple tanh RNN, LSTM, and GRU input/gate gradients passed finite differences on short fixed sequences in PR #29 | More sequence lengths, return_sequences, gate gradients, hidden-state edge cases, graph integration/training | Narrow short-sequence cases only; no graph-level or end-to-end training claim |
| Embedding / attention / residual | Single-head attention parameter gradients, residual identity, and embedding repeated-index gradient accumulation passed in PR #29 | Multi-head/masking/long sequences, embedding index/gradient accumulation, graph integration/training | Narrow operator contracts only; no general family-level claim |
| API/UI | Separate routes exist for simulator and lab/execution paths | Route-by-route tests; verify whether values are computed, approximated, or synthetic; connect visible UI to source-of-truth route | No end-to-end user-facing correctness claim |
| Runtime cost | Five CI runs completed: 4 graph sizes × 40 samples/path/run; output parity passed; ratio ranges are recorded in CLAIM_EVIDENCE_LEDGER.md | Inspect raw samples, repeat across clean runs and another environment before generalizing | No universal overhead claim |
| Usability/learning outcomes | No controlled study | Requires study protocol, participants, consent/ethics as applicable, and analysis | No user-benefit claim |
| Related work / novelty | Initial targeted comparison in `RELATED_WORK.md` | Broader literature search and precise gap formulation | No “first,” “novel,” or superiority claim |

The source-level route map and provenance findings are recorded in [API_UI_PATH_AUDIT.md](API_UI_PATH_AUDIT.md). PR #25's legacy execution-engine fix is merged; run [38493639390](https://github.com/RayAKaan/Neurofluxion/actions/runs/38493639390) passed frontend build, full backend suite, and execution contracts on its PR head. PR #24's full-stack run [38505557136](https://github.com/RayAKaan/Neurofluxion/actions/runs/38505557136) passed backend tests, frontend build, and browser E2E on the PR #24 head.

PR #29's expanded suite passed 11 focused tests covering single-head attention gradients, RNN/LSTM/GRU gradients on short sequences, embedding repeated-index accumulation, residual identity, and simulator smoke tests ([run 38494730679](https://github.com/RayAKaan/Neurofluxion/actions/runs/38494730679)). This remains narrow operator-level evidence.

An optimizer fallback defect was found and corrected in merged PR #28: unknown optimizer names now raise a clear error rather than silently producing a no-op update. Reference-equation tests and the backend suite passed on the PR head; this is not a claim that every optimizer/scheduler combination is validated.

## Post-merge and next-round reconciliation (2026-10-14)

- PRs #24–#33 are merged. Latest main at this update is `203407474ca11232eb32e85e044e6e52e189e18c`.
- Frozen-revision integrated workflow [38566683696](https://github.com/RayAKaan/Neurofluxion/actions/runs/38566683696) passed backend tests, frontend build, browser E2E, and dense research audit on the frozen application SHA `30ff803cb342c494310ff35832df43a092d3d74d`. This is not evidence that every later merge SHA was tested.
- PR #34 added graph-integrated forward/trace consistency cases for dense, Conv2D/pooling/flatten, BatchNorm, RNN/LSTM/GRU, embedding/attention, and residual paths. Full backend run [38571326279](https://github.com/RayAKaan/Neurofluxion/actions/runs/38571326279) passed **98 tests, 12 skipped**. These are fixed-case graph/trace consistency checks, not independent gradient or training oracles for every family.
- PR #35's initial source audits [38571759236](https://github.com/RayAKaan/Neurofluxion/actions/runs/38571759236) and [38572250704](https://github.com/RayAKaan/Neurofluxion/actions/runs/38572250704) reported five GAP findings and one PASS. After reviewing the shared layout, the detector was corrected to account for the global Demo note; run [38573016303](https://github.com/RayAKaan/Neurofluxion/actions/runs/38573016303) reports four GAP, one PARTIAL, and one PASS. None reproduces browser races; remaining gaps need adversarial runtime tests or a separately authorized runtime fix.
- PR #36's cross-platform trace benchmark passed output parity for all four graph widths on both Ubuntu and Windows in [run 38571861118](https://github.com/RayAKaan/Neurofluxion/actions/runs/38571861118) and repeat [run 38572234533](https://github.com/RayAKaan/Neurofluxion/actions/runs/38572234533). Across two runs per OS, median trace/forward ratios ranged by increasing width: Linux 2.27–2.34×, 4.54–4.69×, 10.00–10.46×, 27.28–30.39×; Windows 2.13–2.14×, 3.76–4.21×, 7.56–8.36×, 21.29–24.08×. This is repeatability across hosted runner families, not independent hardware or universal overhead evidence.
- PR #37 expands the comparison set and logs the search protocol. Its review remains a structured rapid review, not a completed systematic review; no novelty-priority claim is supported.
- Open gates: graph-integrated backward/parameter-update/training contracts beyond the existing narrow cases; browser-level stale-response and provenance tests; a local or hardware-diverse benchmark environment; exportable database searches, deduplication, screening counts, full-text review, and citation snowballing. Runtime application code remains out of scope for these research-only changes.

## Next phase: graph-integrated backward and training (PR #39 workstream)

- PRs #34–#38 are now merged. Their results are recorded above and in the claim-evidence ledger.
- The next dependency is to test backward and update behavior through NetworkGraph, not just compare forward and trace outputs.
- New research PR adds deterministic dense BCE training contracts: finite nonzero gradients, actual parameter updates, loss reduction on a fixed one-sample task, and exact repeatability with fixed seed/no dropout.
- A graph-family diagnostic invokes backward_full on dense, Conv2D→MaxPool→Flatten, BatchNorm, RNN/LSTM/GRU, Embedding→Attention→RNN, and residual fixed cases. It reports PASS/GAP plus exceptions in a JSON artifact. PASS means finite gradients for that case only, not gradient correctness.
- Runtime code remains unchanged. Any defect exposed by the audit must be reproduced and fixed in a separately authorized runtime PR, or retained as an explicit limitation.
- This PR's dedicated workflow tests the new gating dense-training contracts and uploads the graph-family diagnostic. A passing report-generation job is not equivalent to every family passing.

## Recommended experiment order

1. Freeze exact source SHA and run smoke + MSE/BCE finite-difference contracts.
2. Verify the tiny-task training trajectory and SGD/Adam contracts.
3. Run and archive the trace-overhead benchmark; report raw sample distributions and environment.
4. Add operator-family contracts in dependency order: shape-only layers, convolution/pooling, batch normalization, recurrent layers, attention/residual.
5. Audit each API route and frontend call site to identify the actual computation source and any illustrative/synthetic output.
6. Update the paper only from checked artifacts; regenerate the PDF from versioned LaTeX and retain the source/command.
7. Select a venue only after its current scope, deadline, formatting, page limit, and anonymization rules are rechecked.

## Interpretation safeguards

- A green workflow may mean that the test harness completed, not that every numerical check passed. Inspect both exit status and the report's declared checks.
- Benchmarks compare paths with different work: the trace path constructs and serializes per-operation records, while the baseline returns a final output. The ratio is a diagnostic of those two concrete paths, not a causal estimate for all tracing implementations.
- A small synthetic task proves only that one configuration reduced its loss; it does not establish general convergence.
- A finite-difference pass is scoped to the tested graph, loss, activation, dtype, and tolerance.
