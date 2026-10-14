# Neurofluxion Research Evidence Package

This directory contains the research protocol and evidence ledger for the Neurofluxion implementation paper. It is intentionally separate from runtime application code: the work must characterize the pinned implementation, not change it to manufacture a stronger paper.

## Status

- Research status: protocol established; a first dense-path CI audit has executed. Forward and dense trace checks passed, repeatability passed, and the baseline MSE gradient check failed. See CLAIM_EVIDENCE_LEDGER.md.
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

The harness targets a small deterministic dense network. The first run found a gradient failure and therefore exited non-zero after preserving JSON/log artifacts. A separate fix PR addresses gradient scaling and result ordering; the audit must be rerun against main after that fix is merged. This remains a diagnostic gate, not a complete validation of every layer family or a publication-grade performance benchmark.

## Evidence labels

- SOURCE-INSPECTED: verified from a named source revision, without execution.
- SCRIPT-CREATED: an experiment implementation exists, but has not been run.
- EXECUTED: the exact command completed and its raw output is retained.
- INDEPENDENTLY-VERIFIED: the result has been checked against an independent oracle or repeat run.
- BLOCKED: execution was attempted but prevented by a documented environment or access constraint.
- UNRESOLVED: evidence is incomplete or conflicting.

Never promote a claim's evidence label merely because a test or script exists.
