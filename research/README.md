# Neurofluxion Research Evidence Package

This directory contains the research protocol and evidence ledger for the Neurofluxion implementation paper. It is intentionally separate from runtime application code: the work must characterize the pinned implementation, not change it to manufacture a stronger paper.

## Status

- Research status: PR #22 merged as `b3e0990da61d54b71b27963f3a664b30ccd47378`. Its merge-preview CI run [38491607499](https://github.com/RayAKaan/Neurofluxion/actions/runs/38491607499) passed 11 targeted simulator smoke and MSE/BCE gradient-contract tests. The dense audit run [38491607397](https://github.com/RayAKaan/Neurofluxion/actions/runs/38491607397) passed forward, trace, selected MSE finite-difference, and repeatability checks on merge-preview tree `d2e64487b9c7ab495a2be9d7c22d22dd32150349`. This is not a full repository test run or a fresh post-merge run; broader operator, provenance/replay, convergence, and performance validation remain open. See CLAIM_EVIDENCE_LEDGER.md and paper/COMPLETION_PLAN.md.
- Source baseline: record the exact commit SHA before every experimental run.
- If a local checkout is unavailable, the GitHub Actions workflow can run against the checked-out PR revision; record that revision separately from the intended frozen application baseline.
- This branch adds research documentation and audit tooling only. It does not change simulator behavior.

## Workflow

1. Freeze a source revision and record the environment.
2. Read EXPERIMENT_PROTOCOL.md before running tests.
3. Run the audit harness from the repository root with PYTHONPATH=backend.
4. Preserve raw JSON, stdout/stderr, exit code, and exact commit.
5. Update CLAIM_EVIDENCE_LEDGER.md only when evidence is actually available.
6. Re-run the full experiment from a clean environment before treating results as reproducible.
7. Revise the manuscript from evidence; never enter anticipated or illustrative values as measured results.

## Initial command

    PYTHONPATH=backend python research/experiments/run_simulator_audit.py --repo . --out research/artifacts/simulator-audit.json

The harness targets a small deterministic dense network. The pre-fix run found a gradient failure and preserved JSON/log artifacts. PRs #19 and #20 have since merged corrections. The post-merge audit passes for the fixed dense case only; this does not validate every layer family or the full repository test suite. This remains a diagnostic gate, not a complete validation of every layer family or a publication-grade performance benchmark.

## Evidence labels

- SOURCE-INSPECTED: verified from a named source revision, without execution.
- SCRIPT-CREATED: an experiment implementation exists, but has not been run.
- EXECUTED: the exact command completed and its raw output is retained.
- INDEPENDENTLY-VERIFIED: the result has been checked against an independent oracle or repeat run.
- BLOCKED: execution was attempted but prevented by a documented environment or access constraint.
- UNRESOLVED: evidence is incomplete or conflicting.

Never promote a claim's evidence label merely because a test or script exists.
