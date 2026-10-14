# Validation Matrix and Remaining Evidence

**Current source baseline:** Neurofluxion `main` at `d85aca48e7993c3b48c951c97480aaa6e0c2934a` (PR #30 merge commit, 2026-10-14). Historical workflow results below remain scoped to their recorded PR heads; this update does not claim a single full-stack rerun on this merge SHA.

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

## Post-merge reconciliation (2026-10-14)\n\n- PRs #24–#30 are merged; latest recorded main is `d85aca48e7993c3b48c951c97480aaa6e0c2934a`.\n- The PR #30 research-validation run [38506053406](https://github.com/RayAKaan/Neurofluxion/actions/runs/38506053406) passed on the PR head, but it is the dense audit workflow, not a full-stack suite.\n- Focused workflows for BatchNorm, Conv2D/pooling/flatten, optimizer equations, and attention/recurrent/embedding/residual operators passed on their respective PR heads. Do not aggregate these as one suite.\n- The full-stack run [38505557136](https://github.com/RayAKaan/Neurofluxion/actions/runs/38505557136) passed on PR #24's head. It predates later merged operator changes.\n- Remaining critical gate: run backend tests, frontend build, browser E2E, and research audit together on one frozen revision and retain all raw artifacts.\n\n## Recommended experiment order

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
