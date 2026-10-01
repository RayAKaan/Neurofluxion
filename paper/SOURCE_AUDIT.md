# Neurofluxion Source Audit — Initial Pass

**Audit target:** `cae464530645c86a191531f0072b426f39fafcf0` (paper branch started from this source commit). This is a source inspection, not an executed test run.

## 1. Simulator graph representation (`backend/simulator/graph_engine.py`)

Observed directly in source:
- `NetworkGraph` stores configured `layers`, parallel `weights` and `biases` lists, instantiated layer objects, parameter-layer index mappings, pre-activation/activation lists, parameter/FLOP totals, an architecture hash, and an input shape.
- `build_graph()` first calls `validate_layers()` and raises `ValueError` when validation fails.
- The builder creates an `InputLayer`, then dispatches each subsequent configured layer to a specialized implementation. Dispatch includes dense/output, batch normalization, Conv2D, max/average pooling, flatten, vanilla RNN, LSTM, GRU, embedding, attention, and residual.
- The builder tracks a `current_shape` through this sequence. Recurrent layers preserve sequence length only when `return_sequences` is enabled. Attention retains `(t_len, d_model)` shape in this builder.
- Parameter-bearing layer instances contribute `params()` arrays to the graph-level `weights` and `biases` lists, with maps from instantiated-layer index to parameter-list index.
- `NetworkGraph.forward()` uses instantiated layers when present; it collects each layer output and appends `layer.Z` when available. A fallback dense-style path also exists.
- The architecture hash is derived from a comma-joined list of each layer's `neurons` value. It is not a full serialization hash of all layer configuration.

Paper implications / caveats:
- Describe the simulator as a validated, shape-propagating sequential layer composition in this builder—not as a general arbitrary-edge DAG unless another source path establishes that.
- Do not imply that graph-level `weights`/`biases` are a universal parameter schema for all layer families; inspect layer-specific `params()` behavior.
- Avoid calling the architecture hash a complete architecture fingerprint; its observed input is only the neuron-count sequence.
- Need inspect `validate_layers`, specialized layers, and parameter handling before enumerating supported semantics or asserting shape correctness.

## 2. Simulator forward trace (`backend/simulator/forward_engine.py`)

Observed directly in source:
- `run_forward_full()` converts input to `float32` and returns `(steps, final_output, layer_outputs)`.
- For dense-only graphs, it emits three records per configured layer: matrix multiplication (`W·a`), bias addition, and activation. Records include inputs/outputs; the matmul record includes a weight snapshot and the bias-add record includes a bias snapshot.
- For mixed/non-dense graphs, it calls each instantiated layer's `forward()` and emits one generic `forward` record per layer, with flattened input/output values. It stores graph activations and any exposed `Z` values.
- `run_forward_step()` calls `run_forward_full()` to regenerate the full sequence, then returns one indexed step plus completed-layer and partial-activation information.

Paper implications / caveats:
- The simulator's displayed step granularity differs by architecture: dense-only networks expose matmul/bias/activation substeps; mixed networks expose layer-level forward steps. State this distinction.
- The step API recomputes the full forward sequence before selecting a step; do not describe it as incremental execution without further evidence.
- Dense-only equation and trace can be described from the source. Numerical correctness still requires tests/reference checks.

## 3. Keras execution trace (`backend/services/execution_trace.py`)

Observed directly in source:
- The service builds a multi-output Keras model from the non-input layers and calls it to obtain intermediate outputs.
- Stage metadata includes layer identity/type/operation, activation, input/output shapes, parameter count, timing field, and output statistics.
- Conv/pooling feature maps are transposed into channel-major arrays. Flatten and other outputs can be bounded before JSON serialization; a `sampled` flag is attached to relevant stage records.
- Per-layer timing is measured by repeatedly constructing/running cumulative-prefix submodels and subtracting the prior prefix's elapsed time. It is therefore a cumulative-subgraph difference, not an isolated layer-kernel measurement.
- The trace's total timing is measured separately around a full activation-model prediction.
- Trace metadata includes trained/untrained status and execution timestamp.

Paper implications / caveats:
- Label Keras trace values separately from NumPy simulator outputs.
- Describe per-layer timings precisely as cumulative-subgraph-difference estimates/attributions; the method can include runtime overhead and measurement noise.
- The payload may be bounded/sampled; never describe every serialized activation as a complete tensor without checking its `sampled` flag and path.
- Inspect detail functions and tests before claiming neuron, convolution, or LSTM value fidelity.

## 4. Not yet audited
- `backend/simulator/backward_engine.py`, `optimizer_engine.py`, layer implementations and validation contracts.
- Session/state persistence and API behavior.
- Trace detail functions: dense neuron, convolution cell, LSTM gates/gradients.
- Registry adapter internals, model provenance/license details, inference pipeline.
- Training manager and both WebSocket channels.
- Frontend-to-backend wiring and real UI screenshots.
- Backend tests, frontend lint/build, numerical reference checks, heavy inference.

## Audit discipline
A file's presence, a test's existence, or a README statement is not equivalent to a passing test. This audit records source observations only; validation remains pending until commands are run and outputs retained.
