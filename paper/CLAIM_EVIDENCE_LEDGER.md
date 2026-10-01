# Claim–Evidence Ledger

Status vocabulary: **Source-confirmed** = directly visible in audited source/docs; **Test-reported** = test exists or repo reports it, not necessarily rerun; **Verified** = independently executed in recorded environment; **Pending** = not yet audited; **Bounded** = claim must be qualified.

| ID | Candidate manuscript claim | Evidence anchor | Current status | Required action / wording boundary |
|---|---|---|---|---|
| C-01 | Neurofluxion combines a React/Vite client with a FastAPI backend. | `README.md`; `frontend/`; `backend/app.py` | Source-confirmed by README/tree | Verify exact app wiring during full audit. |
| C-02 | A NumPy simulator supports interactive construction and forward/backward exploration. | `backend/simulator/`; `frontend/src/components/simulator/`; README | Source-confirmed by README/tree | Audit graph representation, supported layer implementations, and exposed controls. |
| C-03 | The simulator has separate graph, forward, and backward components. | `backend/simulator/graph_engine.py`, `forward_engine.py`, `backward_engine.py` | Source-confirmed by source-tree presence | Inspect implementations and data contracts before mathematical claims. |
| C-04 | Lab trace reads actual Keras model layers and captures intermediate outputs. | `backend/services/execution_trace.py` | Source-confirmed; source inspected | Trace is a Keras path, not the NumPy simulator. Verify per-layer value semantics and architecture edge cases. |
| C-05 | Trace timing is measured from cumulative subgraph predictions, with per-layer values computed as differences. | `measure_per_layer_ms()` in `backend/services/execution_trace.py` | Source-confirmed | Describe as an attribution estimate derived from cumulative measurements, not isolated kernel timing. Investigate possible noise/negative clipping. |
| C-06 | Dense neuron detail exposes real pre-activation and per-input contributions. | `dense_neuron_detail()` in `backend/services/execution_trace.py` | Pending full source read | Verify formula, indexing, and test coverage. |
| C-07 | Convolution-cell and LSTM-gate details are available. | `conv_cell_detail()`, `lstm_gates_detail()`, `lstm_gate_gradients()` in trace service | Pending full source read | Establish exactly which values are computed vs retrieved, and supported conditions. |
| C-08 | Pretrained model registry supports family-appropriate real inputs and lazy loading. | README; `backend/model_registry/`; `backend/services/prediction_service.py` | Source-confirmed by README/tree | Audit adapter metadata, model availability, license/provenance, and heavy inference validation. |
| C-09 | Training telemetry is streamed over WebSockets. | README; `backend/training/manager.py`; `backend/api/train_ws.py` | Source-confirmed by README/tree | Verify emitted metrics and state transitions; distinguish simulator vs framework training. |
| C-10 | `/stream` represents actual neural-network execution. | README; `backend/api/stream_ws.py` | Explicitly not assumed | Audit; README describes topology + metrics stream. Label synthetic/animated if confirmed. |
| C-11 | Simulator explanations implement canonical IG/LIME/DeepSHAP/LRP. | `backend/simulator/interpretability/` | Not established | Do not claim canonical fidelity without equation-level audit and reference tests. |
| C-12 | System is numerically correct across supported operations. | `backend/tests/` | Pending | Run focused numerical tests; report operation-specific results and tolerances. |
| C-13 | Frontend build and backend test suite pass at paper baseline. | README test/build instructions | Test-reported only | Run and preserve exact commands, environment, SHA, output. |
| C-14 | System improves learning outcomes or usability. | No user study identified | Unsupported currently | Do not claim; requires a separately designed human-subject evaluation. |
| C-15 | Neurofluxion is novel/first/superior to existing tools. | No comparative study | Unsupported | Avoid priority and superiority claims; frame as an implementation/integration contribution. |

## Evidence capture template
For every verified claim, record:
- Claim ID and final wording
- Source file + line/function or test name
- Commit SHA
- Verification command and environment
- Outcome, logs/artifact path, limitations
- Manuscript section/table/figure where used
