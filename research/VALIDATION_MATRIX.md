# Validation Matrix and Remaining Evidence

**Source baseline:** Neurofluxion `main` after PR #22 merge, commit `b3e0990da61d54b71b27963f3a664b30ccd47378` at the time this matrix was started. Re-pin the SHA whenever the baseline changes.

This matrix deliberately distinguishes implemented support from independently demonstrated correctness. A layer appearing in a registry or graph builder is not evidence that its forward, backward, optimizer, trace, API, and UI paths all work together.

| Area | Current evidence | Remaining validation | Paper wording allowed now |
|---|---|---|---|
| Dense forward | Independent fixed-parameter oracle passed for one small graph | Multiple widths/depths, activations, and seeds | Agreement for the named fixed case only |
| Dense MSE gradients | Central finite differences passed for simulator and TrainingEngine test cases | More activation and output-width combinations | Tested MSE configurations only |
| Dense sigmoid + BCE gradients | Central finite differences passed in simulator and TrainingEngine test cases | Extreme logits, clipping behavior, alternate output activations | Tested sigmoid/BCE configurations only |
| Training behavior | Tiny deterministic separable BCE task passed in PR #23 research-branch CI: loss decreased under fixed SGD setup; general convergence untested | Confirm loss reduction and repeatability on exact merged revision; add regression tests for other optimizers | No convergence claim until result is recorded |
| SGD / Adam | Exact SGD arithmetic and finite first Adam update tests passed in PR #23 CI; PR #28 adds momentum/RMSProp/two-step Adam references and rejects unknown optimizer names; 8 focused tests passed in run [38494373808](https://github.com/RayAKaan/Neurofluxion/actions/runs/38494373808), but the fix is not yet integrated | Momentum, RMSProp, Adam multi-step reference, unknown optimizer behavior | No all-optimizer correctness claim |
| Trace arithmetic | Matmul, bias-add, activation reconstruction passed in one dense case | Layer families and shape regimes; validate event input/output snapshots | Dense event types and tested case only |
| `run_forward_step` | Source inspection shows it calls `run_forward_full` and selects a step afterward | Measure cost of repeated step requests; consider cached trace only if design and correctness warrant it | Do not claim incremental execution |
| Convolution / pooling / flatten | Classes and graph construction exist; PR #27 added Conv2D finite differences plus MaxPool2D/AvgPool2D forward/backward contracts | Broader channels, strides, padding, activations, graph integration, and trace semantics | One small Conv2D case and one 2×2 case per pooling type passed 8 focused tests; not end-to-end coverage |
| Batch normalization | Layer implementation and graph integration exist; source audit found 2D backward reductions collapsed per-feature gradients to scalars | PR #26 proposes per-feature gradient correction with central finite-difference tests; full backend suite pending | Main baseline has a source-inspected gradient defect; do not claim BatchNorm backward correctness until PR #26's full suite passes and fix is integrated |
| RNN / LSTM / GRU | Simple tanh RNN input/parameter gradients passed central finite differences for one 3-step sequence in PR #29; LSTM/GRU remain untested | More sequence lengths, return_sequences, gate gradients, hidden-state edge cases, graph integration/training | One RNN configuration only; no LSTM/GRU or end-to-end training claim |
| Embedding / attention / residual | Single-head attention parameter gradients passed central finite differences; residual identity passed in PR #29; embedding remains untested | Multi-head/masking/long sequences, embedding index/gradient accumulation, graph integration/training | Narrow attention/residual contracts only; no general family-level claim |
| API/UI | Separate routes exist for simulator and lab/execution paths | Route-by-route tests; verify whether values are computed, approximated, or synthetic; connect visible UI to source-of-truth route | No end-to-end user-facing correctness claim |
| Runtime cost | Single-run diagnostic completed in PR #23 CI: 4 graph sizes × 40 samples/path; output parity passed; see CLAIM_EVIDENCE_LEDGER.md | Inspect raw samples, repeat across clean runs and another environment before generalizing | No universal overhead claim |
| Usability/learning outcomes | No controlled study | Requires study protocol, participants, consent/ethics as applicable, and analysis | No user-benefit claim |
| Related work / novelty | Initial targeted comparison in `RELATED_WORK.md` | Broader literature search and precise gap formulation | No “first,” “novel,” or superiority claim |

The source-level route map and provenance findings are recorded in [API_UI_PATH_AUDIT.md](API_UI_PATH_AUDIT.md). PR #25's proposed legacy execution-engine fix also passed full-stack CI on its branch: 67 backend tests passed, 12 opt-in heavy-inference tests skipped, and frontend build passed; rerun after merge.

PR #29 also passed 8 focused tests covering a tiny single-head attention gradient oracle, RNN gradients over a three-step sequence, residual identity, and simulator smoke tests ([run 38494538826](https://github.com/RayAKaan/Neurofluxion/actions/runs/38494538826)). This remains narrow operator-level evidence.

An optimizer fallback defect was also found: unknown optimizer names silently returned unchanged parameters. PR #28 proposes raising a clear error and adds reference-equation tests; its focused CI is pending.

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
