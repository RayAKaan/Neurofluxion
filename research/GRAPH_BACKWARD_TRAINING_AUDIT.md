# Graph-Integrated Backward and Training Audit

**Status:** Active validation workstream. Runtime implementation is read-only in this research phase.

## Goals

1. Test that the dense NetworkGraph training path changes parameters, keeps gradients finite, reduces BCE on a fixed one-sample task, and is repeatable with a fixed seed and dropout disabled.
2. Run the existing backward_full path against fixed graph configurations spanning dense, Conv2D→MaxPool→Flatten, BatchNorm, RNN/LSTM/GRU, Embedding→Attention→RNN, and residual.
3. Preserve failures, exceptions, non-finite values, and empty-gradient outcomes as machine-readable GAP records.
4. Avoid confusing a finite-gradient smoke check with a correctness oracle.

## CI and outputs

Workflow: .github/workflows/graph-backward-training-audit.yml.

- deterministic-dense-training is a gating test job. It fails if the tested dense training behavior regresses.
- graph-backward-diagnostic emits research/artifacts/graph-backward-audit.json. A successful process exit means the report was generated; it does not mean all operator families passed.
- Each report records source SHA, platform, Python/NumPy versions, source hashes, per-case status, and failure reasons.

## Interpretation

A finite gradient is necessary but not sufficient for correct backpropagation. Independent analytic or finite-difference references are still required for each operator and relevant shape regime. The existing per-operator contracts should be retained as independent evidence, not replaced by this audit.

A dense training pass does not establish training correctness for convolutional, recurrent, attention, embedding, residual, or normalization graphs. Any GAP must remain an explicit limitation until it is independently investigated and, where appropriate, fixed in a separately authorized runtime PR.

## Next gates

- Inspect the first diagnostic artifact and prioritize actual graph-family failures.
- Add independent gradient oracles for each graph family that passes the smoke diagnostic.
- Add deterministic training trajectories for additional architectures only after their backward contracts are independently validated.
- Reconcile these findings into the validation matrix and manuscript without overstating coverage.
