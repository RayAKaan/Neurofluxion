# Embedding–Attention–RNN Graph Training Contract

## Scope

This contract tests one optimizer step through Embedding → single-head Attention → RNN → sigmoid-Dense. It supplements the graph-level finite-difference oracle with a training-engine parameter-mutation check.

## Protocol

- Fixed token sequence [1, 2, 1], vocabulary size five, embedding/model width two, hidden size two.
- Binary cross-entropy, SGD, learning rate 0.01, dropout disabled, fixed seed.
- Assert finite gradients/loss, nonzero gradient norm, optimizer state, and updates to embedding, attention, recurrent, and output parameter tensors.
- Check that the two referenced embedding rows update, including repeated-token accumulation.

## Evidence boundary

A pass establishes one fixed training batch only. It does not prove gradient correctness, training convergence, masked or multi-head attention, arbitrary token sequences, or general optimizer stability. The finite-difference oracle is a separate test. Runtime source remains unchanged.
