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
| RNN / LSTM / GRU | Layer-level short-sequence finite differences plus graph-integrated Conv2D→Flatten→Dense, RNN→Dense, LSTM→Dense, and GRU→Dense MSE gradient oracles (PRs #45–#46) | More sequence lengths, return_sequences, gate/state edge cases, non-dense updates, and graph-integrated training | Fixed tested graph/sequence cases only; no general training or convergence claim |
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
- PR #39's dedicated workflow [38574104561](https://github.com/RayAKaan/Neurofluxion/actions/runs/38574104561) passed deterministic dense BCE training contracts: finite nonzero gradients, parameter changes, fixed-sample loss reduction, and repeatability with fixed seed/no dropout.
- The same workflow's graph-family diagnostic returned finite gradient tensors for **8/8** fixed cases: dense, Conv2D→MaxPool→Flatten, BatchNorm, RNN, LSTM, GRU, Embedding→Attention→RNN, and residual. It reports PASS/GAP plus exceptions in a JSON artifact. PASS means finite gradients for that case only, not gradient correctness; independent gradient oracles and graph-family training checks remain open.
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


## Graph-integrated finite-difference evidence (PR #43 workstream)

- Dedicated workflow [38575599616](https://github.com/RayAKaan/Neurofluxion/actions/runs/38575599616) passed central finite-difference comparisons for two mixed-layer graphs: Conv2D→Flatten→Dense and short tanh RNN→Dense.
- Numerical protocol: epsilon 1e-3, rtol 2e-2, atol 2e-3, fixed seeds, MSE objective.
- Scope remains narrow to the graph shapes, parameters, inputs, targets, dtype, and tolerances tested. PR #46 adds fixed LSTM→Dense and GRU→Dense cases; PR #47 adds Embedding→Attention→RNN→Dense; PR #49 adds Conv2D→BatchNorm→Flatten→Dense. The dedicated workflows and integrated validation for these PRs passed. Alternative losses, broader dimensions, and general training behavior remain open.
- Runtime source unchanged; broader graph-family gradient oracles remain open.


## Post-PR #46 numerical validation update (2026-10-14)

- PR #46 merged as `0ef95cf472b5492140d836d475d4b4528e6648c6`.
- All four PR-head workflows passed: graph-integrated gradient oracles [38580463218](https://github.com/RayAKaan/Neurofluxion/actions/runs/38580463218), research validation [38580463113](https://github.com/RayAKaan/Neurofluxion/actions/runs/38580463113), frozen-main integrated validation [38580463272](https://github.com/RayAKaan/Neurofluxion/actions/runs/38580463272), and full-stack validation [38580463094](https://github.com/RayAKaan/Neurofluxion/actions/runs/38580463094).
- The independently checked graph cases now include Conv2D→Flatten→Dense, RNN→Dense, LSTM→Dense, and GRU→Dense under the fixed MSE finite-difference protocol. This is not broad coverage across shapes, losses, batch sizes, or training behavior.
- PR #47's composite Embedding→Attention→RNN→Dense and PR #49's Conv2D→BatchNorm→Flatten→Dense finite-difference oracles passed their dedicated and integrated workflows. PR #48's LSTM/GRU training-update contracts passed all checks; PR #53's Conv2D/BatchNorm training-update contract passed all checks. PR #54's Embedding→Attention→RNN→Dense training-update contract passed its dedicated and research checks; full-stack/frozen-main browser E2E was still running at this ledger revision. Keep runtime source unchanged unless separately authorized.


## Post-PR #47–#54 evidence reconciliation (2026-10-14)

- **PR #47 merged:** composite Embedding→single-head Attention→RNN→Dense graph finite-difference oracle passed dedicated workflow [38582499269](https://github.com/RayAKaan/Neurofluxion/actions/runs/38582499269), research [38582499212](https://github.com/RayAKaan/Neurofluxion/actions/runs/38582499212), frozen-main [38582499184](https://github.com/RayAKaan/Neurofluxion/actions/runs/38582499184), and full-stack [38582499053](https://github.com/RayAKaan/Neurofluxion/actions/runs/38582499053).
- **PR #49 merged:** Conv2D→BatchNorm→Flatten→Dense graph finite-difference oracle passed dedicated workflow [38583056241](https://github.com/RayAKaan/Neurofluxion/actions/runs/38583056241), research [38583055895](https://github.com/RayAKaan/Neurofluxion/actions/runs/38583055895), frozen-main [38583055940](https://github.com/RayAKaan/Neurofluxion/actions/runs/38583055940), and full-stack [38583055526](https://github.com/RayAKaan/Neurofluxion/actions/runs/38583055526).
- **PR #48 merged:** one-batch LSTM→Dense and GRU→Dense training-update contracts passed dedicated workflow [38582835706](https://github.com/RayAKaan/Neurofluxion/actions/runs/38582835706), research [38582834395](https://github.com/RayAKaan/Neurofluxion/actions/runs/38582834395), frozen-main [38582834465](https://github.com/RayAKaan/Neurofluxion/actions/runs/38582834465), and full-stack [38582834361](https://github.com/RayAKaan/Neurofluxion/actions/runs/38582834361).
- **PR #53 open at ledger revision:** Conv2D→BatchNorm→Flatten→Dense one-batch parameter-update contract passed dedicated workflow [38584049859](https://github.com/RayAKaan/Neurofluxion/actions/runs/38584049859), research [38584049658](https://github.com/RayAKaan/Neurofluxion/actions/runs/38584049658), frozen-main [38584049840](https://github.com/RayAKaan/Neurofluxion/actions/runs/38584049840), and full-stack [38584049641](https://github.com/RayAKaan/Neurofluxion/actions/runs/38584049641).
- **PR #54 open at ledger revision:** Embedding→Attention→RNN→Dense one-batch parameter-update contract passed dedicated workflow [38584232382](https://github.com/RayAKaan/Neurofluxion/actions/runs/38584232382) and research [38584232409](https://github.com/RayAKaan/Neurofluxion/actions/runs/38584232409). Full-stack and frozen-main E2E were still running; do not claim those gates passed until confirmed.
- **PR #51 open:** graph-switch inspection invalidation and stale-response race are separate expected-failure browser cases. Workflow [38583419259](https://github.com/RayAKaan/Neurofluxion/actions/runs/38583419259) is green because both known failures were expected, not because the UI invariants passed. Runtime changes remain unauthorized.
- **PR #52 open:** the OpenAlex/Crossref export script's offline tests passed in [38583891003](https://github.com/RayAKaan/Neurofluxion/actions/runs/38583891003). A live export has not yet been produced; the review remains incomplete pending live exports, manual duplicate adjudication, screening, full-text review, and citation snowballing.

All claims above are scoped to the stated fixed tests. They do not establish general operator correctness, model convergence, or UI provenance integrity. Runtime/application source remains read-only.


## Graph-family training checks completed (2026-10-14)

- PR #53's Conv2D→BatchNorm→Flatten→Dense one-batch training contract passed dedicated workflow [38584049859](https://github.com/RayAKaan/Neurofluxion/actions/runs/38584049859), research [38584049658](https://github.com/RayAKaan/Neurofluxion/actions/runs/38584049658), frozen-main [38584049840](https://github.com/RayAKaan/Neurofluxion/actions/runs/38584049840), and full-stack [38584049641](https://github.com/RayAKaan/Neurofluxion/actions/runs/38584049641).
- PR #54's Embedding→Attention→RNN→Dense one-batch training contract passed dedicated workflow [38584232382](https://github.com/RayAKaan/Neurofluxion/actions/runs/38584232382), research [38584232409](https://github.com/RayAKaan/Neurofluxion/actions/runs/38584232409), frozen-main [38584232291](https://github.com/RayAKaan/Neurofluxion/actions/runs/38584232291), and full-stack [38584232288](https://github.com/RayAKaan/Neurofluxion/actions/runs/38584232288).
- Both are one-batch parameter-update contracts. Neither establishes gradient correctness or convergence; the finite-difference tests are separate evidence.
