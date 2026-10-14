# Initial Source-Audit Findings

**Snapshot inspected:** main at commit f6f43cf987697a0b09079937a9100c892a48e7a2.

**Important scope rule:** these are static source observations. They are not execution results, and they must be rechecked against the exact source revision used for the final paper.

## F-01 — Distinct execution paths must not be conflated

The repository contains at least two relevant backend paths:

1. backend/api/simulator_routes.py imports simulator.graph_engine, simulator.forward_engine, and simulator.backward_engine. The main simulator endpoints invoke run_forward_full and backward_full.
2. backend/api/execute_routes.py imports simulator.execution_engine.get_engine and exposes /api/execute/build, /forward, /backward, and /train.

These paths are not interchangeable. Paper figures, experiments, and conclusions must identify which path was exercised. A smoke test of graph construction and forward tracing does not validate the other execution engine or the /api/execute/* routes.

## F-02 — ExecutionEngine backward and training methods contain placeholder behavior

Static inspection of backend/simulator/execution_engine.py shows that its backward method creates deltas from random values, assigns constant placeholder gradients, and returns a random loss value. Its train method generates random synthetic loss/accuracy histories rather than updating parameters from the supplied dataset.

The corresponding API routes expose these methods through /api/execute/backward and /api/execute/train. This is a source-level finding, not a claim that the main Simulator page uses those endpoints. Targeted inspection of frontend/src/pages/SimulatorPage.tsx and frontend/src/hooks/useSimulatorApi.ts shows that the main Simulator UI calls /api/simulator/* routes. frontend/src/api/execution.ts calls /api/lab/* routes, not /api/execute/*. A full frontend-wide call-site audit remains open, so the exposed /api/execute endpoints must not be represented as the active Simulator UI's training implementation.

**Paper consequence:** do not claim that the /api/execute/* path implements mathematically correct backpropagation or real dataset training. Keep any claims about the separate graph_engine/backward_engine path independent, and validate that path with the finite-difference harness. If the paper describes the whole product, this behavior must be included as a limitation or explicitly scoped out.

**Potential engineering follow-up:** if this API is intended to be a production training path, implement a separate correctness fix with tests and a distinct review. This research PR intentionally does not modify runtime code.

## F-03 — Dense activation coverage requires explicit qualification

The current simulator smoke test configures an output activation named softmax but comments that the activation registry does not define softmax and that the current result falls back to linear/logit output. The test checks shape and finiteness, not softmax normalization or classification semantics.

**Paper consequence:** do not describe softmax as supported by this simulator path until an implementation and oracle test establish its semantics. Do not treat the smoke test as proof of activation correctness.

## F-04 — Trace granularity is heterogeneous

Static inspection of backend/simulator/forward_engine.py shows that the dense-only path emits matrix multiplication, bias addition, and activation records, while other layer types are represented by a layer-level forward record.

**Paper consequence:** operation-level trace completeness must be evaluated separately from layer-level trace delivery. A trace event count alone is not a meaningful completeness measure across these paths.

## F-05 — Step selection performs a full forward trace first

run_forward_step calls run_forward_full and selects the requested event after the full trace is generated.

**Paper consequence:** characterize this as selection over a freshly generated full trace, not incremental computation or constant-time stepping. Any latency statement needs multi-size repeated measurements.

## Required follow-up checks

- [ ] Verify which frontend components call /api/execute/* and which call /api/simulator/*.
- [ ] Trace all route registrations and request/response schemas.
- [ ] Run the main simulator forward/backward path against independent references.
- [ ] Add a separate test demonstrating the /api/execute/* placeholder behavior if the finding is to be included as an empirical result; otherwise keep it clearly source-inspected.
- [ ] Reconcile these findings with the frozen commit and final manuscript scope.

## F-06 — TrainingEngine is a separate backward implementation

Static inspection of backend/simulator/training_engine.py shows a separate gradient path from backward_engine.py. In the layer-instance path, the MSE output derivative is pre-multiplied by the activation derivative before the output layer's backward method is called; DenseLayer.backward also applies that derivative. The MSE reduction factor is also absent. The dense-array fallback path additionally reverses gradient arrays after writing them by layer index.

**Paper consequence:** do not infer training-engine correctness from a passing /api/simulator/backward/full test. The training path needs its own finite-difference tests and should be evaluated separately. A focused fix and regression-test PR is open at https://github.com/RayAKaan/Neurofluxion/pull/20. Its branch CI passed 7 tests, including finite-difference checks for the declared dense MSE case; main must still be re-audited after merge.
