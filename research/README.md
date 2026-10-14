# Neurofluxion Research Evidence Package

This directory contains the research protocol and evidence ledger for the Neurofluxion implementation paper. It is intentionally separate from runtime application code: the work must characterize the pinned implementation, not change it to manufacture a stronger paper.

## Status

- Research status: protocol established; experiments are not yet certified as executed.
- Source baseline: record the exact commit SHA before every experimental run.
- If a local checkout is unavailable, source inspection through repository tools is not a substitute for executed experiments.
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

The first harness targets a small deterministic dense network. It is a diagnostic gate, not a complete validation of every layer family or a publication-grade performance benchmark. A successful run supports only the checks explicitly reported in its output.

## Evidence labels

- SOURCE-INSPECTED: verified from a named source revision, without execution.
- SCRIPT-CREATED: an experiment implementation exists, but has not been run.
- EXECUTED: the exact command completed and its raw output is retained.
- INDEPENDENTLY-VERIFIED: the result has been checked against an independent oracle or repeat run.
- BLOCKED: execution was attempted but prevented by a documented environment or access constraint.
- UNRESOLVED: evidence is incomplete or conflicting.

Never promote a claim's evidence label merely because a test or script exists.
