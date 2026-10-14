# Claim-to-Evidence Ledger

This ledger separates source inspection, executed tests, independent verification, and claims that remain unestablished. The evidence summarized here is a small deterministic dense-path diagnostic plus focused operator workflows, not an exhaustive operator audit.

## Current main reconciliation (2026-10-14)

- Latest recorded main commit at this reconciliation: `203407474ca11232eb32e85e044e6e52e189e18c`, merge commit for PR #33.
- PRs #24–#33 are merged. PR #32 added a research-only frozen-revision validation workflow; it did not modify runtime application code.
- Integrated workflow run [38566683696](https://github.com/RayAKaan/Neurofluxion/actions/runs/38566683696) passed backend tests, frontend build, browser E2E, and deterministic dense research audit on frozen application revision `30ff803cb342c494310ff35832df43a092d3d74d`.
- PRs #34–#37 are active research workstreams and their results are scoped to the exact tested PR revisions. They do not imply that their changes have merged to main or that the latest merge commit has been tested by the frozen-revision workflow.

## Next-round evidence (2026-10-14; PR heads, not merged-main claims)

### Graph-integrated forward/trace consistency

- PR [#34](https://github.com/RayAKaan/Neurofluxion/pull/34) added nine fixed graph cases: dense; Conv2D→MaxPool2D→Flatten; Conv2D→AvgPool2D→Flatten; BatchNorm; RNN; LSTM; GRU; Embedding→Attention→RNN; and residual.
- Full-stack run [38571326279](https://github.com/RayAKaan/Neurofluxion/actions/runs/38571326279) passed the backend suite: **98 passed, 12 skipped**. The added tests compare NetworkGraph.forward to run_forward_full for finite output, shape, and trace consistency.
- Evidence boundary: these are integration consistency checks, not independent numerical references for all operators, and they do not establish graph-integrated backward, parameter-update, or training correctness for every family.

### UI provenance source audit

- PR [#35](https://github.com/RayAKaan/Neurofluxion/pull/35) adds a source-pattern audit and retained JSON artifact.
- Initial runs [38571759236](https://github.com/RayAKaan/Neurofluxion/actions/runs/38571759236) and [38572250704](https://github.com/RayAKaan/Neurofluxion/actions/runs/38572250704) reported 5 GAP findings and 1 PASS. After correcting the detector to inspect the shared layout's global Demo note, [run 38573016303](https://github.com/RayAKaan/Neurofluxion/actions/runs/38573016303) reports **4 GAP, 1 PARTIAL, 1 PASS**. The partial item means a global note exists but per-value/per-panel provenance is not established.
- These are source-level findings, not reproduced browser failures. Browser race scenarios and visible-label assertions remain open. A green audit workflow means the report was generated, not that provenance safety passed.

### Cross-platform trace benchmark

- PR [#36](https://github.com/RayAKaan/Neurofluxion/pull/36) runs a fixed dense benchmark on Ubuntu and Windows, Python 3.11, NumPy 2.4.6, four widths, five warmups, and 40 raw samples per path/case.
- Both [run 38571861118](https://github.com/RayAKaan/Neurofluxion/actions/runs/38571861118) and repeat [run 38572234533](https://github.com/RayAKaan/Neurofluxion/actions/runs/38572234533) passed output parity for all cases on both OS runners with maximum absolute error 0.0.
- Across the two runs per OS, trace/forward median-ratio ranges by increasing width were Linux: 2.27–2.34×, 4.54–4.69×, 10.00–10.46×, 27.28–30.39×; Windows: 2.13–2.14×, 3.76–4.21×, 7.56–8.36×, 21.29–24.08×.
- Interpretation: the qualitative scaling pattern reproduced on these hosted runners, but timing is runner-specific and trace construction does additional work. This is not hardware-independent performance evidence.

### Related-work review

- PR [#37](https://github.com/RayAKaan/Neurofluxion/pull/37) expands the comparison set and records search queries, inclusion/exclusion rules, extraction fields, and review limitations in research/LITERATURE_REVIEW_PROTOCOL.md.
- This is a structured rapid-review phase, not a completed systematic review. The search did not produce exportable/deduplicated database results or a complete screening-count flow; do not use it to support “first,” novelty-priority, or superiority claims.

## Executed evidence

### Training and trace-cost validation (2026-10-14)

- Pull request: [#23](https://github.com/RayAKaan/Neurofluxion/pull/23), exact tested head `b473a2cd76af1dba57bf38eb34305eaac77bb683`.
- Training-contract CI: [run 38492813949](https://github.com/RayAKaan/Neurofluxion/actions/runs/38492813949), **10 passed**. This includes the existing MSE and BCE finite-difference contracts plus four new tests: tiny separable BCE loss reduction, repeatable fixed-order training trajectory, exact SGD arithmetic, and finite Adam update with step counter.
- Trace benchmark CI: [run 38492813970](https://github.com/RayAKaan/Neurofluxion/actions/runs/38492813970), **4 graph configurations × 40 raw samples per path**, warmups=5, output parity passed in all cases. Artifact `neurofluxion-trace-overhead` contains the JSON report and raw timing samples.
- Runner-specific median results (milliseconds; forward-only vs full trace construction):
  - widths `[4, 8, 2]`: 0.007349 ms vs 0.017313 ms; ratio 2.36×.
  - widths `[16, 32, 8]`: 0.007569 ms vs 0.033833 ms; ratio 4.47×.
  - widths `[32, 64, 16]`: 0.007805 ms vs 0.075351 ms; ratio 9.65×.
  - widths `[64, 128, 32]`: 0.008596 ms vs 0.227191 ms; ratio 26.43×.
- These are preliminary measurements from one GitHub Actions runner and one run. The compared paths do different work: the trace path emits operation records and serializes arrays to lists. Ratios are diagnostic for this implementation and environment, not a universal tracing-overhead estimate or a publishable performance conclusion.
- Research validation on the same PR head: [run 38492813935](https://github.com/RayAKaan/Neurofluxion/actions/runs/38492813935), workflow completed successfully and preserved the numerical audit artifact.


### Post-merge validation (2026-10-14)

- Pull request: [#21](https://github.com/RayAKaan/Neurofluxion/pull/21), tested by GitHub Actions run [38491189247](https://github.com/RayAKaan/Neurofluxion/actions/runs/38491189247).
- Main base at validation: `d49d73967bfffab5aea3b962c46a865328f38f2b`; PR merge revision checked out by Actions: `c05f8828b4c88076f4100c5eb86829412ffdac73`.
- Simulator smoke tests: 5 passed. Merged simulator and TrainingEngine gradient-contract regression tests: 4 passed. These are targeted tests, not the complete repository test suite.
- The audit harness reported forward oracle, trace oracle, finite-difference gradient check, and repeatability all passed; `failed_checks` was empty. The check is limited to the fixed dense case and selected MSE/activation configuration implemented by the harness.
- The first run exposed a shell exit-code capture typo in the workflow even though the numerical checks passed. The workflow was corrected and rerun; the corrected run completed without the shell error.
- Raw artifact: available from the run page above (`neurofluxion-research-audit`).


Research audit run: https://github.com/RayAKaan/Neurofluxion/actions/runs/38489714384
- Test environment: GitHub Actions, Ubuntu 24.04, Python 3.11.17, NumPy 2.4.6.
- Tested application baseline: main commit f6f43cf987697a0b09079937a9100c892a48e7a2, evaluated in the research-branch merge commit recorded in the raw JSON.
- Existing simulator smoke tests: 5 passed.
- Fixed dense forward oracle: passed; maximum absolute error 0.0.
- Dense trace arithmetic: passed; final-output error 0.0 and each checked event residual below 1.2e-8.
- Repeatability: identical output and trace hashes across repeated fixed-parameter runs.
- MSE gradient finite difference: failed before the proposed fix; normalized L2 discrepancy about 0.675 at epsilon 0.01, 0.003, and 0.001. Inspection of returned vectors identified layer-order mismatch plus output-width scaling mismatch.
- Timing: the harness recorded diagnostic samples for one tiny graph only. These are not publication-grade performance results and must not support general latency or overhead claims.

Follow-up fix PR: https://github.com/RayAKaan/Neurofluxion/pull/19
- Adds a correction for mean-reduced MSE scaling and gradient array ordering.
- Adds regression tests comparing returned gradient tensors to central finite differences.
- The fix was merged into `main` in PR #19 (merge commit `90957f5e7b80d52eef4692799399a9c36bfba8b5`). The post-merge fixed-case audit passed in run [38491189247](https://github.com/RayAKaan/Neurofluxion/actions/runs/38491189247). Scope remains one deterministic dense case; no all-layer claim is supported.

## Claim table

| ID | Candidate manuscript claim | Required evidence | Current status | Allowed wording |
|---|---|---|---|---|
| C-01 | Neurofluxion contains distinct NumPy simulator and framework-backed model paths. | Source audit at pinned commit and path boundaries. | SOURCE-INSPECTED | Describe paths separately; do not imply parameter or numerical equivalence. |
| C-02 | Dense forward computation agrees with an independent reference. | E1 fixed-parameter oracle. | EXECUTED, passed for one two-layer dense case | State the tested case and exact tolerance; do not generalize to all layers. |
| C-03 | Backward gradients agree with numerical derivatives. | E2 finite differences for each claimed loss/activation pair. | PRE-FIX BASELINE FAILED; post-merge fixed-case audit PASSED; broader coverage pending | Do not claim the baseline is correct. |
| C-04 | Dense trace events represent the recorded arithmetic. | E3 per-event reconstruction. | EXECUTED, passed for six dense events in one case | Restrict to the tested dense event types and fixed case. |
| C-05 | Repeated fixed-parameter execution is repeatable. | E4 repeated runs with trace normalization policy. | EXECUTED, passed for one graph/input | State fixed graph and input scope; no cross-machine determinism claim. |
| C-06 | Inspection has a characterized runtime cost. | E5 multi-size, repeated-process benchmark with raw samples and environment metadata. | PRELIMINARY DIAGNOSTIC EXECUTED across five Actions runs: 4 sizes × 40 samples/path/run, output parity passed. Per-size trace/forward median ratios ranged 2.36–2.46×, 4.43–4.86×, 9.65–11.13×, and 26.43–32.34× for increasing widths. | Repeatability is shown only within the same hosted-runner class; do not generalize or claim universal overhead. A second environment and a decomposed end-to-end measurement remain required. |
| C-07 | A layer family is supported end-to-end. | Construction, forward, backward, training, trace and UI evidence. | NOT ESTABLISHED beyond tested dense cases; other families are mapped in VALIDATION_MATRIX.md | Name only the specific tested path. |
| C-08 | The paper makes a differentiated research contribution. | Verified closest-work comparison and evidence-backed gap. | UNRESOLVED | Avoid “first,” “novel,” and “state of the art.” |
| C-09 | Neurofluxion improves learning or usability. | Controlled user study and analysis. | NOT TESTED / OUT OF SCOPE | Make no improvement claim. |
| C-10 | Existing simulator smoke tests pass. | Test command and logs. | EXECUTED, 5 smoke tests and 4 targeted gradient-contract tests passed on the PR #21 validation revision | State test files and run; do not imply full repository suite passed. |
| C-11 | The /api/execute backward and training endpoints implement real numerical learning. | Route-to-implementation audit and numerical tests. | SOURCE-INSPECTED finding F-02 shows placeholder behavior in ExecutionEngine; targeted frontend inspection shows SimulatorPage/useSimulatorApi call /api/simulator routes, while a full frontend-wide usage audit remains open. | Do not claim those execute endpoints are correct; keep main simulator path separate. |
| C-12 | TrainingEngine computes correct MSE gradients. | Independent finite differences against the exact mean-reduced loss, plus shape/order checks. | SOURCE-INSPECTED finding F-06 indicated scaling and duplicate-activation-derivative risks; correction merged via PR #20 (merge commit `d49d73967bfffab5aea3b962c46a865328f38f2b`); targeted finite-difference regression passed on the PR #21 validation revision; convergence and other losses/activations remain untested. | Do not claim training correctness until tests pass on the intended source revision. |
| C-13 | Mean-reduced BCE gradients are correct for sigmoid outputs. | Central finite differences for both simulator and TrainingEngine paths. | PR #22 is merged; focused CI reported 11 tests passed on its recorded head, including sigmoid-output BCE finite-difference checks. Broader output activations, extreme values, and full post-merge suite remain unverified. | Claim only the tested sigmoid/BCE configurations; do not claim general training correctness. |

## Rules

1. Update a row only when the evidence artifact exists.
2. Link each claim to source revision, experiment ID, command, raw output, and analysis.
3. Keep negative and mixed results.
4. A successful harness run does not establish all-layer correctness.
5. A compiled PDF establishes typesetting, not scientific validity.
6. Literature review status must distinguish initial search from a systematic review.
7. A fix-branch pass does not establish that main contains the fix.

### Training and optimizer contract status

- Tiny-task loss reduction and repeated fixed-order trajectory passed for one linearly separable binary task using sigmoid+BCE and SGD. This does not prove general convergence.
- Exact one-step SGD arithmetic and a finite first Adam update passed. Momentum, RMSProp, multi-step Adam reference behavior, scheduler combinations, and unknown optimizer handling remain untested.
- Remaining family/API/UI validation is tracked in [VALIDATION_MATRIX.md](VALIDATION_MATRIX.md).

### Additional operator finding: BatchNorm (2026-10-14)

- Source inspection of main revision `b3e0990da61d54b71b27963f3a664b30ccd47378` found that the 2D BatchNorm backward path reduces gamma/beta and variance/mean gradients across all elements instead of per feature. This is inconsistent with the forward path's per-feature statistics.
- PR [#26](https://github.com/RayAKaan/Neurofluxion/pull/26) corrected the axis reductions and added central finite-difference tests for input, gamma, and beta. The final PR-head workflow [38493941513](https://github.com/RayAKaan/Neurofluxion/actions/runs/38493941513) passed both the BatchNorm contract and full backend suite; research validation [38493941567](https://github.com/RayAKaan/Neurofluxion/actions/runs/38493941567) also passed. This remains narrow layer-level evidence, not proof of all graph-integrated BatchNorm configurations.

### Additional operator contracts: convolution and pooling (2026-10-14)

- PR [#27](https://github.com/RayAKaan/Neurofluxion/pull/27) adds a fixed single-channel Conv2D kernel/bias finite-difference check plus MaxPool2D and AvgPool2D forward/backward contracts. Nine focused tests passed in run [38495772620](https://github.com/RayAKaan/Neurofluxion/actions/runs/38495772620), adding a Flatten element-order and backward shape contract. This is narrow operator-level evidence, not graph-level or end-to-end training validation.

### Additional optimizer finding (2026-10-14)

- Source inspection found that an unrecognized optimizer name silently returned unchanged parameters, allowing an invalid optimizer configuration to produce a no-op update. PR [#28](https://github.com/RayAKaan/Neurofluxion/pull/28) changed this to a clear `ValueError` and added reference-equation tests for momentum, RMSProp, and two-step Adam. Eight focused tests passed in run [38494373808](https://github.com/RayAKaan/Neurofluxion/actions/runs/38494373808), and the full backend suite passed 62 tests with 12 opt-in heavy-inference skips in run [38495430720](https://github.com/RayAKaan/Neurofluxion/actions/runs/38495430720). PR #28 is merged; broader optimizer/scheduler and graph-integrated training coverage remains open.

### Additional operator contracts: attention, RNN, residual (2026-10-14)

- PR [#29](https://github.com/RayAKaan/Neurofluxion/pull/29) adds central finite-difference checks for a tiny single-head attention block's four projection matrices and a three-step tanh RNN's input/recurrent weights, bias, and input gradient, plus residual identity checks. Eleven focused tests passed in run [38494730679](https://github.com/RayAKaan/Neurofluxion/actions/runs/38494730679), adding finite-difference contracts for LSTM/GRU short-sequence gradients and repeated-index embedding accumulation. This does not validate multi-head attention, masking, long-sequence stability, or end-to-end graph training.

### Repeated trace benchmark runs

Five independent CI runs were inspected: [38492813970](https://github.com/RayAKaan/Neurofluxion/actions/runs/38492813970), [38492888157](https://github.com/RayAKaan/Neurofluxion/actions/runs/38492888157), [38493816167](https://github.com/RayAKaan/Neurofluxion/actions/runs/38493816167), [38493984213](https://github.com/RayAKaan/Neurofluxion/actions/runs/38493984213), and [38494414914](https://github.com/RayAKaan/Neurofluxion/actions/runs/38494414914). Each passed output parity for all four widths and retained 40 timing samples per path per width. Trace/forward median-ratio ranges across runs were 2.36–2.46×, 4.43–4.86×, 9.65–11.13×, and 26.43–32.34×. This supports repeatability within this CI runner class only; it is not a second-hardware validation or a general performance claim.

## Graph-integrated backward and training diagnostic (PR #39, 2026-10-14)

- Dedicated workflow [38574104561](https://github.com/RayAKaan/Neurofluxion/actions/runs/38574104561) passed deterministic dense BCE training contracts: finite nonzero gradients, parameter changes, fixed-sample loss reduction, and repeatability with fixed seed and dropout disabled.
- The graph-family audit artifact reported **8/8** fixed cases with finite gradient tensors: dense, Conv2D→MaxPool→Flatten, BatchNorm, RNN, LSTM, GRU, Embedding→Attention→RNN, and residual.
- Full-stack workflow [38574104816](https://github.com/RayAKaan/Neurofluxion/actions/runs/38574104816) passed backend tests, frontend build, and the existing browser E2E on the PR head.
- Scope: finite-gradient smoke results are not proof of gradient correctness. Independent finite differences/analytic oracles, non-dense parameter-update tests, and training trajectories for other families remain open.


## Graph-integrated finite-difference oracles (PR #43, 2026-10-14)

- Workflow [38575599616](https://github.com/RayAKaan/Neurofluxion/actions/runs/38575599616) passed central finite-difference checks for Conv2D→Flatten→Dense and short tanh RNN→Dense graph backward paths.
- Protocol: epsilon 1e-3, rtol 2e-2, atol 2e-3, fixed seeds, MSE.
- This is independent numerical evidence for two specific mixed-layer graphs. It does not generalize to all operator families, losses, shapes, or integrated training configurations.


## Post-PR #46 evidence reconciliation (2026-10-14)

PR #46 merged as `0ef95cf472b5492140d836d475d4b4528e6648c6`. Its four PR-head workflows all passed: graph-integrated gradient oracles [38580463218](https://github.com/RayAKaan/Neurofluxion/actions/runs/38580463218), research validation [38580463113](https://github.com/RayAKaan/Neurofluxion/actions/runs/38580463113), frozen-main integrated validation [38580463272](https://github.com/RayAKaan/Neurofluxion/actions/runs/38580463272), and full-stack validation [38580463094](https://github.com/RayAKaan/Neurofluxion/actions/runs/38580463094). The two new graph-level finite-difference cases—LSTM→Dense and GRU→Dense—passed on the PR head. This does not establish universal recurrent correctness or graph-integrated training.

The next proposed oracle combines Embedding→single-head Attention→RNN→Dense with fixed repeated token indices. Its outcome must be reported only after the dedicated workflow completes. Remaining independent gates include graph-integrated BatchNorm gradients, non-dense parameter-update/training contracts, browser stale-response/provenance scenarios, a complete reproducible literature review, and a frozen-source manuscript audit. Runtime application source remains read-only under the current scope.


## Post-PR #47–#54 ledger update (2026-10-14)

PR #47's composite Embedding→Attention→RNN→Dense finite-difference oracle and PR #49's Conv2D→BatchNorm→Flatten→Dense finite-difference oracle both passed dedicated and integrated validation before merge. PR #48's LSTM/GRU training-update contracts passed all four workflows before merge. PR #53's Conv2D/BatchNorm training-update contract passed its dedicated, research, frozen-main, and full-stack workflows. PR #54's Embedding/Attention/RNN/Dense training-update contract passed its dedicated and research workflows; full-stack and frozen-main browser E2E were still running at this revision.

The evidence is fixed-case only and must not be generalized to all shapes, losses, or convergence. The two UI provenance scenarios in PR #51 are expected failures, which keeps CI green while the runtime defects remain open. The reproducible literature export in PR #52 has passing offline tests but still needs a live export and manual screening. Runtime/application source remains read-only under the current scope.
