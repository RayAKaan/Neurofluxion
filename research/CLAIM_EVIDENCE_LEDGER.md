# Claim-to-Evidence Ledger

This ledger separates source inspection, executed tests, independent verification, and claims that remain unestablished. The evidence summarized here is a small deterministic dense-path diagnostic, not an exhaustive operator audit.

## Executed evidence

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
- Its branch CI has passed the new regression test and existing simulator smoke tests. The fix is not yet part of main; rerun the research audit against the merged main revision before changing the paper's final correctness claims.

## Claim table

| ID | Candidate manuscript claim | Required evidence | Current status | Allowed wording |
|---|---|---|---|---|
| C-01 | Neurofluxion contains distinct NumPy simulator and framework-backed model paths. | Source audit at pinned commit and path boundaries. | SOURCE-INSPECTED | Describe paths separately; do not imply parameter or numerical equivalence. |
| C-02 | Dense forward computation agrees with an independent reference. | E1 fixed-parameter oracle. | EXECUTED, passed for one two-layer dense case | State the tested case and exact tolerance; do not generalize to all layers. |
| C-03 | Backward gradients agree with numerical derivatives. | E2 finite differences for each claimed loss/activation pair. | FAILED on pre-fix baseline; proposed fix branch tests pass; main must be re-audited after merge | Do not claim the baseline is correct. |
| C-04 | Dense trace events represent the recorded arithmetic. | E3 per-event reconstruction. | EXECUTED, passed for six dense events in one case | Restrict to the tested dense event types and fixed case. |
| C-05 | Repeated fixed-parameter execution is repeatable. | E4 repeated runs with trace normalization policy. | EXECUTED, passed for one graph/input | State fixed graph and input scope; no cross-machine determinism claim. |
| C-06 | Inspection has a characterized runtime cost. | E5 multi-size, repeated-process benchmark with raw samples. | NOT ESTABLISHED | Do not report the tiny-graph microtiming as a performance claim. |
| C-07 | A layer family is supported end-to-end. | Construction, forward, backward, training, trace and UI evidence. | NOT ESTABLISHED beyond tested dense case | Name only the specific tested path. |
| C-08 | The paper makes a differentiated research contribution. | Verified closest-work comparison and evidence-backed gap. | UNRESOLVED | Avoid “first,” “novel,” and “state of the art.” |
| C-09 | Neurofluxion improves learning or usability. | Controlled user study and analysis. | NOT TESTED / OUT OF SCOPE | Make no improvement claim. |
| C-10 | Existing simulator smoke tests pass. | Test command and logs. | EXECUTED, 5 passed on the research audit run | State test file and run; do not imply full repository suite passed. |
| C-11 | The /api/execute backward and training endpoints implement real numerical learning. | Route-to-implementation audit and numerical tests. | SOURCE-INSPECTED finding F-02 shows placeholder behavior in ExecutionEngine; targeted frontend inspection shows SimulatorPage/useSimulatorApi call /api/simulator routes, while a full frontend-wide usage audit remains open. | Do not claim those execute endpoints are correct; keep main simulator path separate. |

## Rules

1. Update a row only when the evidence artifact exists.
2. Link each claim to source revision, experiment ID, command, raw output, and analysis.
3. Keep negative and mixed results.
4. A successful harness run does not establish all-layer correctness.
5. A compiled PDF establishes typesetting, not scientific validity.
6. Literature review status must distinguish initial search from a systematic review.
7. A fix-branch pass does not establish that main contains the fix.
