# Claim-to-Evidence Ledger

This ledger separates source inspection, executed tests, independent verification, and claims that remain unestablished. The evidence summarized here is a small deterministic dense-path diagnostic, not an exhaustive operator audit.

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
| C-06 | Inspection has a characterized runtime cost. | E5 multi-size, repeated-process benchmark with raw samples and environment metadata. | PRELIMINARY DIAGNOSTIC EXECUTED: 4 sizes × 40 samples/path, one Actions runner; median trace/forward ratios 2.36×, 4.47×, 9.65×, 26.43× for the tested widths. | Report only as a single-run, environment-specific diagnostic; do not generalize or claim universal overhead. Repeat in clean runs and a second environment before paper use. |
| C-07 | A layer family is supported end-to-end. | Construction, forward, backward, training, trace and UI evidence. | NOT ESTABLISHED beyond tested dense cases; other families are mapped in VALIDATION_MATRIX.md | Name only the specific tested path. |
| C-08 | The paper makes a differentiated research contribution. | Verified closest-work comparison and evidence-backed gap. | UNRESOLVED | Avoid “first,” “novel,” and “state of the art.” |
| C-09 | Neurofluxion improves learning or usability. | Controlled user study and analysis. | NOT TESTED / OUT OF SCOPE | Make no improvement claim. |
| C-10 | Existing simulator smoke tests pass. | Test command and logs. | EXECUTED, 5 smoke tests and 4 targeted gradient-contract tests passed on the PR #21 validation revision | State test files and run; do not imply full repository suite passed. |
| C-11 | The /api/execute backward and training endpoints implement real numerical learning. | Route-to-implementation audit and numerical tests. | SOURCE-INSPECTED finding F-02 shows placeholder behavior in ExecutionEngine; targeted frontend inspection shows SimulatorPage/useSimulatorApi call /api/simulator routes, while a full frontend-wide usage audit remains open. | Do not claim those execute endpoints are correct; keep main simulator path separate. |
| C-12 | TrainingEngine computes correct MSE gradients. | Independent finite differences against the exact mean-reduced loss, plus shape/order checks. | SOURCE-INSPECTED finding F-06 indicated scaling and duplicate-activation-derivative risks; correction merged via PR #20 (merge commit `d49d73967bfffab5aea3b962c46a865328f38f2b`); targeted finite-difference regression passed on the PR #21 validation revision; convergence and other losses/activations remain untested. | Do not claim training correctness until tests pass on the intended source revision. |
| C-13 | Mean-reduced BCE gradients are correct for sigmoid outputs. | Central finite differences for both simulator and TrainingEngine paths. | A separate defect was found after the MSE fixes: PR #22 adds scaling/activation-chain corrections and 2 BCE finite-difference tests; its focused CI reports 11 tests passed, but PR #22 is not merged, so main remains unverified for this contract. | Do not claim BCE or general training correctness until the fix is merged and revalidated on main. |

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
- PR [#26](https://github.com/RayAKaan/Neurofluxion/pull/26) corrects the axis reductions and adds central finite-difference tests for input, gamma, and beta. The focused contract passed 7 tests with simulator smoke tests in run [38493885673](https://github.com/RayAKaan/Neurofluxion/actions/runs/38493885673). A full backend rerun is pending on the updated PR head; do not claim the fix is integrated until merged and revalidated.

### Additional operator contracts: convolution and pooling (2026-10-14)

- PR [#27](https://github.com/RayAKaan/Neurofluxion/pull/27) adds a fixed single-channel Conv2D kernel/bias finite-difference check plus MaxPool2D and AvgPool2D forward/backward contracts. Eight focused tests passed in run [38494152856](https://github.com/RayAKaan/Neurofluxion/actions/runs/38494152856). This is narrow operator-level evidence, not graph-level or end-to-end training validation.

### Additional optimizer finding (2026-10-14)

- Source inspection found that an unrecognized optimizer name silently returned unchanged parameters, allowing an invalid optimizer configuration to produce a no-op update. PR [#28](https://github.com/RayAKaan/Neurofluxion/pull/28) changes this to a clear `ValueError` and adds reference-equation tests for momentum, RMSProp, and two-step Adam. CI is pending at the time of this update.
