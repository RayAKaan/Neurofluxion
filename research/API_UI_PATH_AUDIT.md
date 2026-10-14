# API/UI Execution-Path Audit

**Audit date:** 2026-10-14  
**Inspected main baseline:** b3e0990da61d54b71b27963f3a664b30ccd47378  
**Scope:** source inspection of route ownership and data provenance. This is not a browser-driven end-to-end test.

## Path inventory

| User-facing surface | Frontend source | Backend route family | Computation source | Evidence / boundary |
|---|---|---|---|---|
| Simulator page | frontend/src/pages/SimulatorPage.tsx | /api/simulator/* | NetworkGraph, forward_engine.py, backward_engine.py, training_engine.py | The page builds a demo graph on mount and creates random demo input plus a one-hot target. This is useful for demonstration, but it is not a trained-dataset result or an accuracy claim. |
| Lab execution trace | frontend/src/api/execution.ts, frontend/src/pages/LabPage.tsx | /api/lab/* | TensorFlow/Keras model bundle and services/execution_trace.py | The path uses the inference model bundle unless useUntrainedWeights is explicitly set. This is a separate execution path from the NumPy simulator; don't imply shared weights. |
| Legacy execution API | No frontend references found in the inspected frontend source/search | /api/execute/* | simulator/execution_engine.py | On inspected main, backward returned random placeholder gradients/loss and training returned synthetic random metrics without parameter updates. PR #25 proposes replacing this with real dense-only numerics and explicit unsupported-layer rejection; it is not part of the inspected main baseline until merged. |
| Training WebSocket | Simulator training socket route | /api/simulator/ws/train (route family; see backend/api/simulator_train_ws.py) | TrainingEngine and graph/session state | Events show lifecycle/progress. They are not independent evidence of numerical correctness; use gradient, update, and convergence tests. |
| Snapshot/event stream | backend/api/stream_ws.py | stream WebSocket route | Fixed illustrative topology and generated values in the inspected implementation | The stream should not be presented as the selected graph's live state unless a verified graph/run correlation is added. The current manuscript already flags this boundary. |

## Consequences for the manuscript

1. Name the route family and execution engine for each experiment. “Neurofluxion output” is too ambiguous when the app contains multiple engines.
2. Separate the NumPy simulator from the TensorFlow/Keras Lab path. Do not infer parameter synchronization from their shared UI.
3. Treat SimulatorPage's generated random input and one-hot target as demo setup, not a dataset benchmark.
4. Do not use legacy /api/execute/* backward/training responses as numerical evidence until PR #25 is merged and its contracts pass on the integrated revision.
5. Treat snapshot-stream events as illustrative until source graph identity, run identity, and parameter/activation provenance are shown.
6. A frontend build is a compile check, not a browser integration test. Route wiring needs HTTP tests and, for visual consistency claims, browser-driven tests or a carefully scoped manual protocol.

## Validation coverage status

The full-stack validation workflow in PR #24 passed on revision 1a2ca171fbafafe86833f3dbe9b7f375a005540d: 58 backend tests passed, 12 heavy-inference tests were skipped by the documented opt-in, and the TypeScript/Vite frontend build succeeded. That run predates PR #25's runtime changes and must be repeated after integration. It does not substitute for browser E2E coverage.

## Remaining route/UI work

- Add HTTP-level tests for /api/execute/build, /forward, /backward, and /train after the execution-engine contract is integrated.
- Add HTTP-level simulator graph build → forward → backward checks using one graph ID and fixed parameters.
- Verify response shapes consumed by each frontend store, including error envelopes and stale-graph behavior.
- Add browser E2E coverage for simulator demo initialization, forward-step navigation, and Lab trace selection; assert that visible values match response payloads.
- Add explicit source labels in the UI for demo-generated inputs, untrained bundles, and trained model inference where users could otherwise confuse them.
