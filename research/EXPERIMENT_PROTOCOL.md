# Neurofluxion Experimental Protocol

## 1. Purpose and scope

This protocol evaluates the existing Neurofluxion implementation as it exists at a frozen source revision. The target is an implementation and empirical systems paper, not a new learning algorithm or model-accuracy comparison.

The initial protocol prioritizes a small deterministic dense network because its arithmetic can be reconstructed independently. Results do not generalize to convolutional, recurrent, attention, normalization, pooling, embedding, or residual paths unless those paths receive their own tests.

## 2. Source and environment freeze

For every run, record:

- Full Git commit SHA and dirty/clean working-tree state.
- Operating system and architecture.
- Python and NumPy versions.
- Relevant dependency versions.
- CPU/GPU information when available.
- Exact command, environment variables, start/end time, exit code, stdout and stderr.
- Raw output file checksum.

A run is valid for the paper only when the recorded source revision matches the revision named in the experiment manifest. If the checkout is not a Git repository, mark the commit as unavailable and do not call the result pinned-source evidence.

## 3. E1 — Forward-pass oracle

Use a fixed architecture, fixed parameters, and fixed input. Independently compute each dense stage as:

    z_l = W_l @ a_(l-1) + b_l
    a_l = phi_l(z_l)

Compare the simulator output to the independent result using maximum absolute error and relative L2 error. The initial dense test uses tanh in a hidden layer and linear output. Use a declared absolute tolerance of 1e-5 for the initial float32 case; report the actual error, dtype, and tolerance. Do not infer support for other activations or layers from this one case.

## 4. E2 — Gradient finite-difference check

Compare analytical gradients from the simulator with central finite differences of the exact scalar loss returned by the implementation:

    g_i(epsilon) = [L(theta_i + epsilon) - L(theta_i - epsilon)] / (2 epsilon)

Test more than one epsilon. Report normalized L2 discrepancy, maximum absolute discrepancy, analytical gradient norm, numerical gradient norm, and all tested epsilon values.

The oracle must match the implementation's loss reduction convention exactly. If a discrepancy occurs, preserve it as a finding; do not loosen tolerance after observing results without recording the rationale. The initial harness reports gradient diagnostics separately. A gradient failure is a research finding and must not be hidden by a successful forward pass.

## 5. E3 — Trace arithmetic

For each operation-level record, reconstruct its output from the values and parameter snapshot recorded for that event:

- Matrix multiplication: W @ input.
- Bias addition: input + bias.
- Activation: apply the named supported activation to the recorded pre-activation.

Validate each event individually and retain operation identity, layer identity, and step index. Do not group records only by operation name because the same operation may occur in multiple layers. A final-output match alone does not establish trace fidelity.

For non-dense layer paths, use layer-specific references and do not count a layer-level forward event as equivalent to operation-level tracing.

## 6. E4 — Repeatability

Under fixed parameters and input, repeat forward execution and trace generation. Compare output values and normalized trace content. Exclude nondeterministic metadata only if its exclusion is explicitly documented. Distinguish repeatability for a fixed graph from deterministic initialization after graph rebuild.

## 7. E5 — Inspection-step cost

The current step-navigation API may compute a full forward trace before selecting one step. Measure this as selection latency, not incremental per-step execution. A single tiny network is only a diagnostic. A publishable scaling study requires multiple network sizes, warm-up, repeated independent trials, raw samples, and environment metadata.

## 8. Reporting and acceptance rules

- Keep raw outputs; derive paper tables and figures from them.
- Report every tested case and every failure.
- Report unavailable/unsupported cases as such; do not silently omit them.
- Separate unit correctness, trace fidelity, repeatability, and timing claims.
- Do not make performance claims from a single-run or tiny-network timing.
- Do not claim all-layer support from dense-only experiments.
- Do not claim TensorFlow/Keras equivalence unless weights, inputs, dtype, operator semantics, and loss reductions are aligned.
- Do not claim improved learning outcomes without an appropriate user study.
- Do not put results in the abstract until the evidence ledger is updated and the run has been checked.

## 9. Expansion plan

After the dense gate is stable, add independent tests for each actually implemented operation family in this order:

1. Dense activations and loss/output combinations.
2. Parameter initialization, parameter counting, and shape propagation.
3. Convolution, pooling, and flattening.
4. Batch normalization.
5. RNN/LSTM/GRU.
6. Embedding, attention, and residual paths.
7. Training/optimizer updates and state persistence.
8. Framework-backed trace behavior and UI provenance, if executable in the available environment.

Each expansion requires its own expected semantics, fixtures, tolerances, raw output, and explicit coverage status.
