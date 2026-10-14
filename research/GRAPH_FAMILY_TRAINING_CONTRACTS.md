# Graph-Family Training Contracts

## Scope

This test gate checks whether one TrainingEngine.train_batch call can propagate a finite, nonzero gradient and update parameters for fixed LSTM→sigmoid-Dense and GRU→sigmoid-Dense graphs. It supplements the graph-level finite-difference tests; it is not a numerical gradient oracle.

## Protocol

- Fixed three-step, two-feature input sequence and binary target.
- Hidden size two, one sigmoid output, binary cross-entropy, SGD, learning rate 0.02, dropout disabled.
- Fixed random seeds for each recurrent family.
- Assert finite loss/gradient norms, finite gradient tensors, initialized optimizer state, and actual recurrent-parameter mutation.

## Evidence boundary

A passing test establishes only that these fixed one-batch configurations produced finite gradients and updated recurrent and graph parameters. It does not establish gradient correctness, loss reduction, convergence, performance, arbitrary sequence lengths, or all recurrent gate/state cases. The numerical finite-difference oracle is a separate gate.

Runtime/application implementation is not changed by this workstream. If a test exposes a runtime defect, preserve the failing evidence and request explicit authorization before modifying runtime code.
