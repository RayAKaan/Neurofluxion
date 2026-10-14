# Neurofluxion Research Evidence Package

This directory contains the research protocol and evidence ledger for the Neurofluxion implementation paper. It is intentionally separate from runtime application code: the work must characterize the pinned implementation, not change it to manufacture a stronger paper.

## Status

- Research status: fixed dense forward/trace/repeatability checks pass; MSE and sigmoid+BCE gradient finite-difference contracts pass; 10 training/gradient contracts passed in [run 38492813949](https://github.com/RayAKaan/Neurofluxion/actions/runs/38492813949). A four-size trace-cost diagnostic ran with 40 raw samples per path in [run 38492813970](https://github.com/RayAKaan/Neurofluxion/actions/runs/38492813970). Broader layer-family, API/UI, full-suite, and cross-environment validation remain open. See CLAIM_EVIDENCE_LEDGER.md and VALIDATION_MATRIX.md.
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


## Latest diagnostic: trace generation cost

The PR #23 benchmark compares final-output forward execution with full dense trace construction on four fixed graph widths. Five independent GitHub Actions runs showed per-size trace/forward median-ratio ranges of 2.36–2.46×, 4.43–4.86×, 9.65–11.13×, and 26.43–32.34× as graph widths increased. Each run used 40 raw samples per path per graph configuration and passed output parity. These figures are repeatable within the same hosted-runner class, but the paths do different work and they remain **preliminary diagnostic evidence only**. Raw JSON artifacts are attached to runs [38492813970](https://github.com/RayAKaan/Neurofluxion/actions/runs/38492813970), [38492888157](https://github.com/RayAKaan/Neurofluxion/actions/runs/38492888157), [38493816167](https://github.com/RayAKaan/Neurofluxion/actions/runs/38493816167), [38493984213](https://github.com/RayAKaan/Neurofluxion/actions/runs/38493984213), and [38494414914](https://github.com/RayAKaan/Neurofluxion/actions/runs/38494414914). A second environment and a decomposed end-to-end measurement are still needed before a broader performance conclusion.

See [VALIDATION_MATRIX.md](VALIDATION_MATRIX.md) for the remaining layer, API/UI, training, and paper evidence requirements.
