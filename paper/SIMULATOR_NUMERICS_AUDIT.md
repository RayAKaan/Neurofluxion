# Simulator Numerics Audit — Backward Engine

**Target source:** `backend/simulator/backward_engine.py` at baseline `cae464530645c86a191531f0072b426f39fafcf0`.

This is a static source audit. No numerical tests were run in this environment.

## Observed behavior

### Dense-only branch
- `backward_full()` first calls `graph.forward(x)`, obtains the flattened final output, and computes the configured loss through `compute_loss()`.
- A dense-only graph is identified by configured layer types restricted to `input`, `dense`, and `output`.
- Reverse traversal computes layer deltas. For the final layer, MSE multiplies `(y_hat - y)` by the activation derivative; for other loss names, the code sets `delta = y_hat - y` directly.
- Earlier-layer deltas use the next layer's weight transpose multiplied by the current layer activation derivative.
- Weight gradients are computed as `outer(delta, a_prev)`; optional L2 adds `l2_lambda * W`. Bias gradients copy the delta.
- The function emits step records for delta, weight-gradient, and bias-gradient computations, then returns gradient arrays, deltas, input gradient, loss, and summary statistics.

### Mixed/non-dense branch
- The engine initializes the output gradient as `y_hat - y`, with an activation-derivative adjustment only for MSE when the last layer exposes an activation name and `Z`.
- It iterates instantiated layers in reverse, calls each layer's `backward()`, and collects parameter gradients from each parameter-bearing layer.
- The returned `deltas` list is initialized but not populated in the observed generic branch.
- The per-layer summary's `delta_norm` is populated from `np.linalg.norm(dw)` in this branch, so it represents the weight-gradient norm rather than a separately retained delta norm.
- `backward_step()` invokes the full backward computation before selecting a step; it is not incremental reverse-mode execution.

## Validation risks to resolve before claims
1. **Loss/activation compatibility:** for dense-only non-MSE losses, the final delta is hard-coded as `y_hat - y`. Verify which output activations and loss functions make this derivative valid; do not generalize to arbitrary activation/loss combinations.
2. **Parameter/layer indexing:** dense-only branch indexes graph-level parameter arrays by layer position. Validate behavior for every allowed dense-only configuration (including parameterless or unusual layer configs).
3. **Generic branch gradient semantics:** inspect each layer's `backward()` contract and independently verify representative Conv2D, pooling, normalization, RNN/LSTM/GRU, attention, and residual cases.
4. **Summary correctness:** generic `delta_norm` currently appears to reuse the weight-gradient norm. This is a reporting-semantic issue worth testing/fixing before paper screenshots or aggregate tables.
5. **Step granularity:** both forward-step and backward-step selection recompute the full pass; use “step-indexed replay/inspection” rather than implying computationally incremental stepping.
6. **L2 convention:** determine whether the loss adds the matching regularization term and whether the reported gradient corresponds to the exact stated objective.
7. **Finite-difference tests:** add deterministic tiny-network checks for dense MSE and supported classification-loss/activation combinations, then selected mixed layers where gradients are expected to be supported.

## Paper boundary
The manuscript may describe the engine's implemented control flow, but must not claim general backpropagation correctness or broad layer-wise gradient fidelity until the above checks pass. Any discrepancies should be reported as implementation limitations and/or fixed on a separate, traceable commit before the final validation freeze.
