# Neurofluxion Simulator: Source-Level System Audit

**Historical audit target:** `RayAKaan/Neurofluxion`, `main` at commit `b3e0990da61d54b71b27963f3a664b30ccd47378` (PR #22 merge commit at the time of inspection).  
**Current-main reference when this document was reconciled:** `0866972ea5a9e9f2d1864681402e3e3659bdaf8a` (after PR #29).  
**Status:** architecture inventory and initial source findings; **not yet a completed line-by-line audit or a full execution validation**. Findings below are pinned to the historical audit target unless explicitly rechecked. Do not assume source-level observations still describe current `main` without validating them against the current commit.

## 1. Why the simulator must be the paper's technical center

Neurofluxion is not one execution path. The repository combines (a) a NumPy-based custom graph simulator, (b) a separate legacy-style `ExecutionEngine` API, (c) a TensorFlow/Keras-backed model/lab/inference path, and (d) frontend state and visualization layers. A paper that says “the simulator” without identifying which of these paths produced a result risks attributing behavior to the wrong implementation.

The core research question should be defined narrowly and tested: **what numerical computation does the custom simulator execute, what computation evidence does it emit, how does that evidence travel through the API to the UI, and under which architecture/layer/loss conditions is the behavior correct and repeatable?**

## 2. System map

### 2.1 Browser / frontend

- `frontend/src/pages/SimulatorPage.tsx`: simulator entry point; initializes demo state and builds a graph through `/api/simulator/architecture/build`.
- `frontend/src/hooks/useSimulatorApi.ts`: typed Axios wrapper for simulator REST endpoints.
- `frontend/src/api/client.ts`: shared Axios client; in Vite development, API calls use the `/backend` prefix, with proxy behavior configured in `frontend/vite.config.ts`. The simulator training WebSocket URL is built separately as `/backend/ws/simulator/train`.
- `frontend/src/store/architectureStore.ts`: editable layer configurations, presets, validation/build actions.
- `frontend/src/store/simulatorStore.ts`: graph ID, current input/target, active view and forward-step UI state.
- `frontend/src/store/datasetStore.ts`: dataset selection and client-side copies of returned train/test data.
- `frontend/src/store/computationStore.ts` and `frontend/src/store/backpropStore.ts`: forward trace and backward result state (must be followed through their action implementations during the next pass).
- `frontend/src/store/trainingSimStore.ts` and `frontend/src/hooks/useTrainingSimSocket.ts`: streamed training status, batch telemetry, epoch metrics and warnings.
- `frontend/src/components/simulator/*`: architecture editing, forward/backward panels, dataset controls, metrics, inspector, profiler and other views. These are presentation/interaction layers; they are not independent numerical oracles.

### 2.2 FastAPI application and custom simulator path

`backend/app.py` registers `backend/api/simulator_routes.py`, the specialized simulator routers, `backend/api/simulator_train_ws.py`, and other application routers. The main custom simulator routes use the prefix `/api/simulator`.

Core path:

1. The client submits layer configurations to `POST /api/simulator/architecture/validate` or `POST /api/simulator/architecture/build`.
2. `simulator_routes.py` converts API `LayerIn` records to `LayerConfig`, calls `validate_layers`, then calls `session_manager.create_graph`.
3. `session_manager.py` builds a `NetworkGraph` through `graph_engine.build_graph` and stores it in an in-memory dictionary under a UUID graph ID.
4. `graph_engine.py` validates shapes and constructs layer instances, plus parameter arrays and parameter-to-layer index mappings.
5. `POST /api/simulator/forward/full` fetches that graph and calls `forward_engine.run_forward_full`; it returns operation records, final output, step count and layer outputs.
6. `POST /api/simulator/forward/step` calls `run_forward_step`; source inspection shows this currently computes a full trace first and then selects the requested event. It is not incremental execution.
7. `POST /api/simulator/backward/full` calls `backward_engine.backward_full`, then caches returned gradients in the graph's training session for inspection endpoints.
8. `/api/simulator/dataset/*` creates or loads data and registers it in the process-local `dataset_manager`.
9. `/ws/simulator/train` resolves the graph and dataset IDs, constructs a `TrainingConfig`, runs `TrainingEngine.train_epoch`, and streams status, batch, epoch, warning and completion messages.

### 2.3 Numerical engine and parameter representation

The custom simulator has two execution styles inside the graph/training code:

- **Dense-only fast path:** `forward_engine.py` and parts of `backward_engine.py`/ `training_engine.py` operate directly on `graph.weights`, `graph.biases`, pre-activations and activations.
- **Layer-instance path:** `graph_engine.py` constructs `DenseLayer`, convolution, pooling, normalization, recurrent, embedding, attention and residual objects under `graph.layer_instances`. Each layer may own its own caches/parameters/backward method. `NetworkGraph` also carries `param_layer_indices` and `param_index_by_layer` to map parameterized layer instances to compact weight/bias arrays.

These two paths must not be assumed equivalent merely because they share a graph ID or response schema. The paper must report which path each experiment used and validate each supported operator family separately.

### 2.4 Distinct paths that must remain separate in all claims

- **Custom simulator UI:** source inspection of `SimulatorPage.tsx` and `useSimulatorApi.ts` points to `/api/simulator/*`; training uses `/ws/simulator/train`.
- **Separate execution API:** `backend/api/execute_routes.py` exposes `/api/execute/build`, `/forward`, `/backward`, and `/train`, backed by `backend/simulator/execution_engine.py`. This engine is separate from `graph_engine.py` and the graph-session routes.
- **Lab/trace API:** `backend/api/lab.py` uses TensorFlow/Keras model bundles and the `services.execution_trace` path for model activations, backward-related views, saliency and trace detail.
- **Prediction/model registry:** `backend/api/predict.py`, `backend/services/prediction_service.py`, `backend/services/inference.py` and `backend/model_registry/*` form the pretrained-model inference path.

Results from one path do not validate another.

## 3. API surface inventory (initial)

### Main simulator router (`/api/simulator`)

- Architecture: `/architecture/validate`, `/architecture/build`
- Forward: `/forward/full`, `/forward/step`
- Math/inspection: `/equations/layer`, `/inspect/weights`, `/inspect/activations`, `/inspect/gradients`, `/inspect/gradient_flow`, `/inspect/weight_history/{graph_id}`
- Backward: `/backward/full`, `/backward/step`
- Data: `/dataset/generate`, `/dataset/load_standard`, `/dataset/custom`, `/dataset/generate_sequence`, `/dataset/upload_image`
- Debug/replay/performance: `/debug/diagnose`, `/debug/apply_fix`, `/replay/snapshots/{graph_id}`, `/replay/load`, `/performance/estimate`
- Layer visualizations: feature maps, saliency, filter response, neuron atlas and sequence step/full
- Analysis: comparison setup/results, full profile, templates, import upload/build and code/image export

Specialized routers add landscape, embeddings, integrated gradients/SHAP/LIME/LRP comparison, adversarial attacks/robustness, compression/pruning/quantization, generative sampling, augmentation previews, experiment CRUD and assistant query. Every route family needs an implementation-level check before its existence is described as validated capability.

## 4. Data and state lifecycle

### 4.1 Graph state

`SessionManager` holds graph objects in process memory. A graph ID is a lookup key, not a persistent artifact identity: source currently shows no database-backed graph persistence in this manager, no graph revision ID, and no automatic cross-process sharing. The architecture hash observed in `NetworkGraph` is based only on a comma-joined neuron-count signature; it is not a cryptographic identity for all layer types, parameters, activations or configuration fields.

### 4.2 Dataset state

`DatasetManager` stores complete dataset dictionaries in a process-local map keyed by generated dataset ID. Synthetic data generation, standard dataset loading, sequence generation and custom points enter through different route helpers. The manager does not itself persist datasets to durable storage. Dataset identity, random seed, preprocessing, split membership and shape must be included in experiment manifests if results are to be reproducible.

### 4.3 Training and snapshots

The WebSocket training route resolves graph/dataset IDs, translates each point into NumPy arrays, and runs epochs via a `TrainingSession`. Epoch/batch metrics are streamed to the browser; snapshots are maintained separately by the snapshot manager/training session. Snapshot restoration changes graph parameter arrays and, for layer-instance graphs, calls `set_params` on parameterized layer instances. The audit must test that both parameter representations remain synchronized before/after updates and replay.

### 4.4 Visualization data is not automatically proof of computation

A value displayed in a panel may be a real operation output, a sampled/statistical summary, a computed explanation, an estimate, or an illustrative/stub response. Each visible feature must be traced to its source endpoint and numerical producer. UI labels and response types alone are insufficient evidence that an operation is mathematically correct.

## 5. Initial source-level findings to validate

These are source-inspection findings, not new test results:

1. **Separate placeholder engine:** `ExecutionEngine.backward` creates random deltas and constant gradients; `ExecutionEngine.train` emits synthetic random loss/accuracy history instead of optimizing against the supplied dataset. The exposed `/api/execute/*` routes must not be described as real backpropagation/training.
2. **Softmax ambiguity:** the initial source audit found no `softmax` entry in the custom activation registry; the dense smoke test's output configuration named softmax but fell back to linear behavior. This needs an explicit test and either a correction or a precise limitation.
3. **Unequal trace granularity:** dense-only forward execution emits separate matrix multiplication, bias-add and activation events; non-dense layer paths emit a single layer-level forward event. Report operation-level coverage separately from layer-level trace delivery.
4. **Step-navigation cost:** `run_forward_step` regenerates the full forward trace before selecting one event. Benchmark as trace generation plus selection, not incremental execution.
5. **Training backward-path divergence:** `TrainingEngine` has separate dense-array and layer-instance branches, while layer classes such as `DenseLayer` also apply their own activation derivative in `backward`. The layer-instance loss/activation contract requires independent finite-difference testing, especially to detect derivative scaling or double-application.
6. **State lifetime:** graph and dataset managers are in-memory process singletons. Persistence, multi-worker behavior, concurrent mutation, stale graph IDs and cleanup behavior need explicit investigation before any claim of durable experiment replay or multi-user isolation.
7. **Demo architecture:** `SimulatorPage.tsx` initializes a demo architecture with an output activation named `softmax`; until softmax semantics are verified, this demo should not be used as evidence of valid normalized class probabilities.

## 6. Audit and validation workstream

### A. Complete call graph and contracts
- Enumerate every frontend call to `/api/simulator/*`, `/api/execute/*`, `/api/lab/*`, prediction endpoints and both WebSockets.
- For every route, record request schema, response schema, side effects, errors, state store and visible component.
- Trace each user action end-to-end: architecture edit/build, input selection, forward run/step, backward run/step, dataset generation/loading, train/pause/resume/stop, snapshot replay, inspect/export.

### B. Numerical correctness
- Dense forward: fixed parameters, multiple activations, batch/vector shape cases, dtype and shape failures.
- Loss/gradient contracts: MSE and BCE reduction conventions, sigmoid/logit compatibility, output activation, L2 regularization, finite differences at multiple epsilon values.
- Optimizer semantics: SGD, momentum, RMSProp and Adam update equations/state reset and batch averaging.
- Layer families: each implemented layer's forward reference, shape propagation, gradient reference where training is claimed, edge cases and unsupported combinations.
- Synchronization: graph arrays versus layer-instance parameters across training, snapshots and imports.
- Determinism: fixed-seed initialization/data generation, repeated forward/trace, training order, replay and serialization.

### C. Trace and UI provenance
- Validate every trace record's operation, layer index, input, parameters, output and ordering against independent arithmetic.
- Define trace schema/version and identify graph, graph revision, execution, dataset/sample and parameter revision.
- Check step navigation, stale/out-of-order API responses, concurrent runs, replay and graph rebuilds.
- Mark outputs as exact computation, sampled summary, estimate, explanation or illustrative/stub; ensure the UI does not present estimates as measured runtime or unverified probabilities as calibrated confidence.

### D. System and performance
- Measure cold/warm build, forward, trace generation, step selection, backward and training throughput on declared architectures and sample sizes.
- Report repeated raw samples, medians/dispersion, environment and memory. Do not use the existing single tiny-network diagnostic as a performance claim.
- Test API validation, payload limits, error handling, cancellation, WebSocket disconnect, concurrency, graph/dataset cleanup and multi-worker deployment assumptions.

### E. Paper evidence
- Tie each statement, diagram, table cell and abstract claim to source lines, a reproducible test or a raw experiment artifact.
- Keep source-inspected, unit-tested, integration-tested and empirically measured claims separate.
- Do not generalize dense-only validation to CNN/RNN/attention or to the separate ExecutionEngine and TensorFlow/Keras lab/inference paths.

## 7. Initial architectural conclusion

The most defensible paper contribution is not “Neurofluxion visualizes neural networks” by itself. That space is established. The candidate technical contribution to test is the coupling of a custom numerical execution path with inspectable computation records and interactive inspection/replay. Its value depends on proving (i) the computation is correct for a stated operator/loss scope, (ii) the trace faithfully represents the computation, (iii) the UI consumes the correct run's evidence, and (iv) the additional observability cost is measured. Until those gates are passed, describe the system as an implementation and validation study in progress, not as a validated general-purpose simulator.

## 8. Source entry points inspected for this initial inventory

- `backend/app.py`
- `backend/api/simulator_routes.py`
- `backend/api/simulator_train_ws.py`
- `backend/api/execute_routes.py`
- `backend/api/lab.py`
- `backend/api/predict.py`
- `backend/simulator/graph_engine.py`
- `backend/simulator/forward_engine.py`
- `backend/simulator/backward_engine.py`
- `backend/simulator/training_engine.py`
- `backend/simulator/execution_engine.py`
- `backend/simulator/layers/__init__.py` and `layers/dense.py`
- `backend/simulator/activations.py`, `loss_functions.py`, `optimizer_engine.py`
- `backend/simulator/session_manager.py`, `dataset_manager.py`, `snapshot_manager.py`
- `frontend/src/pages/SimulatorPage.tsx`
- `frontend/src/hooks/useSimulatorApi.ts`, `useTrainingSimSocket.ts`
- `frontend/src/api/client.ts`
- `frontend/src/store/architectureStore.ts`, `simulatorStore.ts`, `datasetStore.ts`, `trainingSimStore.ts`
- `frontend/src/types/simulator.ts`
- `research/SOURCE_AUDIT_FINDINGS.md`, `research/EXPERIMENT_PROTOCOL.md`, `research/experiments/run_simulator_audit.py`

**Next action:** complete the call-site inventory and per-layer/loss test matrix against the exact post-merge commit before rewriting paper claims or presenting broad simulator correctness.


## 9. Frontend-to-backend contract findings from the next call-site pass

These findings are still source-level and must be confirmed by focused integration tests. They are recorded because the implementation paper must distinguish active UI paths from dormant/legacy code.

### 9.1 The main Run/Inspect training flow uses the WebSocket route

`RunInspectTab.tsx` constructs training commands for `useTrainingSimSocket`, and that hook connects to `SIM_WS_URL`, which resolves to the `/backend/ws/simulator/train` proxy path in development. The backend route is registered as `/ws/simulator/train`. It accepts `start`, `pause`, `resume`, `stop`, and `update_config` actions and emits status/batch/epoch/warning/complete/error messages. This is the primary training flow to trace for the main Simulator page.

However, the currently inspected `RunInspectTab` training-start handler sends a hard-coded config (5 epochs, batch size 16, learning rate 0.01, Adam, BCE, no L2) instead of the configuration held by `trainingSimStore`. Therefore the visible Hyperparameter panel and the actual WebSocket training configuration may diverge. Confirm this in an integration test and fix the UI-to-training contract before reporting that user-selected hyperparameters control the run.

### 9.2 A second training helper has a stale REST endpoint and synthetic progress

`frontend/src/hooks/useExecution.ts` contains a `train` method that posts to `/api/simulator/train/start`. The main `/api/simulator` router inventory contains no `/train/start` route; the registered simulator training entry point is the WebSocket. The helper then contains a comment and loop that simulate epoch progress locally rather than deriving progress from server messages.

This is a source-level integration defect **if that helper is reachable**. Its actual call sites must be enumerated before attributing this behavior to the active UI. The primary `RunInspectTab` handler inspected above uses the WebSocket route, so do not assume the stale helper is the path used by the main screen.

### 9.3 Device selection is split across API surfaces

`useExecution.ts` selects devices through `POST /api/device/select`, while `RunInspectTab.tsx` contains a device-change call to `POST /api/execute/device`. The latter endpoint changes the separate `ExecutionEngine` device preference; it does not establish that the graph-session NumPy simulator is executing on that device. UI wording and paper claims must avoid implying GPU acceleration of the custom graph engine from this setting alone.

### 9.4 Dataset UI state and server dataset state are distinct

The browser store keeps separate `trainData`, `testData`, `standardData`, and `sequenceData` fields, while the server `DatasetManager` stores the canonical dataset dictionary by ID. Synthetic generation populates the browser train/test arrays; the standard-dataset action, as currently inspected, sets `standardData` and `datasetId` without setting `trainData`/ `testData`. Components such as `ForwardPassPanel` choose a sample from `trainData`. This creates a likely UX mismatch for standard datasets that needs a test: training may resolve the server-side dataset ID correctly while the “New Sample” UI has no local samples to choose from.

### 9.5 WebSocket state needs protocol-level tests

The client calls `connect()` and, when the socket is not yet open, schedules a send after a fixed 300 ms delay. The server may not be ready within that delay; the code should queue messages and flush them on `onopen`, as the separate general-training socket does. The simulator socket handler also parses incoming JSON without a local parse-error guard and does not currently show an explicit `onerror` handler in the inspected hook. Test connection races, immediate start, disconnect/reconnect, server errors, pause/resume/stop, and multiple starts before using its metrics as a reliable experimental interface.

### 9.6 Main Simulator demo is not a valid softmax demonstration yet

`SimulatorPage.tsx` auto-builds a 16→8→2 demo and configures its output activation as `softmax`. Until the custom activation registry actually implements softmax and a test verifies `sum(p)=1), the demo output must not be described as a probability distribution. The current forward fallback behavior needs either implementation plus tests or an explicit UI fallback/unsupported-state treatment.

## 10. Concrete next validation batch

1. Add a route contract test asserting all frontend-called paths exist and that deprecated/stale endpoints are either removed or implemented.
2. Add a UI integration test proving that the exact hyperparameter values selected in the Build tab are transmitted in the WebSocket `start` payload and are reflected in the server's `TrainingConfig`.
3. Add a standard-dataset test proving that sample selection, forward input shape and server-side training all use the same dataset/split.
4. Add a WebSocket test for immediate send while CONNECTING, then stop/pause/resume and disconnect behavior.
5. Add a softmax activation contract test, or change the demo to a currently supported activation and clearly label logits versus probabilities.
6. Trace all uses of `useExecution` and all components that call `/api/execute/*` before deciding whether legacy placeholder behavior is reachable from a visible UI.
7. Add source-linked assertions to the paper's evidence ledger; do not present these static findings as runtime outcomes until the corresponding tests have run.
