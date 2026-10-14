# Claim-to-Evidence Ledger

This ledger is deliberately conservative. The existence of code or a test is not evidence that an experiment passed.

| ID | Candidate manuscript claim | Required evidence | Current status | Allowed wording before execution |
|---|---|---|---|---|
| C-01 | Neurofluxion contains a separate NumPy simulator and TensorFlow/Keras-backed model path. | Source audit at a pinned commit; verify path boundaries. | SOURCE-INSPECTED, pending final snapshot reconciliation | “The repository contains distinct simulator and framework-backed paths.” |
| C-02 | Dense forward computation agrees with an independent reference. | E1 fixed-parameter oracle, raw output, pinned commit. | SCRIPT-CREATED; not executed here | “We specify a test to evaluate dense forward agreement.” |
| C-03 | Backward gradients agree with numerical derivatives. | E2 finite differences for declared loss/activation combinations and epsilon sweep. | SCRIPT-CREATED; not executed here | No positive correctness claim. |
| C-04 | Recorded operation events faithfully represent arithmetic. | E3 per-event reconstruction for every claimed operation. | SCRIPT-CREATED; not executed here | “The protocol evaluates trace arithmetic.” |
| C-05 | Repeated execution is deterministic. | E4 controlled repeated runs; define scope and metadata normalization. | SCRIPT-CREATED; not executed here | “Repeatability is evaluated under fixed parameters and inputs.” |
| C-06 | Step inspection has a known runtime cost. | E5 multi-scale benchmark, warm-up, repeated trials, raw samples. | NOT ESTABLISHED | No latency or overhead claim. |
| C-07 | A given layer family is supported end-to-end. | Construction, forward oracle, backward oracle where applicable, training, trace and UI evidence. | NOT ESTABLISHED for general layer coverage | Name only specific source-inspected or tested paths. |
| C-08 | The paper contributes something not already established by prior systems. | Verified literature review, closest-work comparison, evidence-backed difference. | UNRESOLVED | Do not claim “first,” “novel,” or “state of the art.” |
| C-09 | The tool improves learning or usability. | Appropriate controlled human-subject evaluation and analysis. | OUT OF SCOPE / NOT TESTED | Make no improvement claim. |
| C-10 | Repository tests pass. | Test command run against exact pinned checkout; logs and exit code. | NOT EXECUTED in current sandbox | Do not report pass status. |

## Rules

1. Update a row only when the evidence artifact exists.
2. Link each claim to source revision, experiment ID, command, raw output, and analysis.
3. Keep negative and mixed results.
4. A successful harness run does not establish all-layer correctness.
5. A compiled PDF establishes typesetting, not scientific validity.
6. Literature review status must distinguish initial search from a systematic review.
7. Use BLOCKED only when an attempted execution and its failure are recorded.
