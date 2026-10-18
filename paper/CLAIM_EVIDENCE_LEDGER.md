# Claim–Evidence Ledger

Status vocabulary: **Source-confirmed** = directly visible in audited source/docs; **Test-reported** = test exists or repo reports it, not necessarily rerun; **Verified** = independently executed in a recorded environment; **Pending** = not yet audited; **Bounded** = claim must be qualified.

| ID | Candidate manuscript claim | Evidence anchor | Current status | Required action / wording boundary |
|---|---|---|---|---|
| C-01 | Neurofluxion combines a React/Vite client with a FastAPI backend. | `README.md`; `frontend/`; `backend/app.py` | Source-confirmed | Describe architecture without inferring behavior from filenames alone. |
| C-02 | A NumPy simulator supports interactive construction and forward/backward exploration. | `backend/simulator/`; simulator frontend; README | Source-confirmed | Describe supported sequential construction and inspected execution; do not imply arbitrary DAG support. |
| C-03 | The simulator has separate graph, forward, and backward components. | `backend/simulator/graph_engine.py`, `forward_engine.py`, `backward_engine.py` | Source-confirmed | Treat the components as separate implementation units with distinct contracts. |
| C-04 | Lab trace reads actual Keras model layers and captures intermediate outputs. | `backend/services/execution_trace.py` | Source-confirmed | Trace is a Keras path, not the NumPy simulator. Payloads can be bounded/sampled. |
| C-05 | Trace timing is measured from cumulative subgraph predictions, with per-layer values computed as differences. | `measure_per_layer_ms()` | Source-confirmed | Describe as cumulative-subgraph timing attribution, not isolated kernel timing. |
| C-06 | Dense neuron detail exposes implementation-derived values for a selected unit. | `dense_neuron_detail()` | Bounded | Treat detail as an implementation-specific inspection endpoint; avoid unsupported general explanation claims. |
| C-07 | Convolution-cell and LSTM-gate detail endpoints exist. | trace detail functions | Bounded | Describe only fields and conditions established by the implementation/tests. |
| C-08 | Pretrained model registry supports family-appropriate real inputs and lazy loading. | README; `backend/model_registry/`; prediction service | Source-confirmed | Model provenance and availability remain model-specific. |
| C-09 | Framework training telemetry is streamed over WebSockets. | `backend/training/manager.py`; `backend/api/train_ws.py` | Source-confirmed | Distinguish framework training from simulator training. |
| C-10 | `/stream` represents actual neural-network execution. | `backend/api/stream_ws.py` | Explicitly rejected | It generates a synthetic topology/metrics stream from hard-coded values and formula-derived curves; do not call it live execution. |
| C-11 | Simulator explanations implement canonical IG/LIME/DeepSHAP/LRP. | `backend/simulator/interpretability/` | Not established | Do not claim canonical method fidelity. |
| C-12 | System is numerically correct across supported operations. | Research graph-gradient and training-contract workflows | Bounded, partial evidence | Fixed finite-difference graph oracles cover Dense/Conv2D, RNN, LSTM, GRU, Embedding→Attention→RNN→Dense, and Conv2D→BatchNorm→Flatten→Dense. Training-update contracts cover fixed LSTM/GRU, Conv2D/BatchNorm, and Embedding/Attention/RNN/Dense batches. Do not generalize to all shapes, losses, optimizers, or convergence. |
| C-13 | Backend tests/frontend build pass at a frozen publication revision. | CI workflows | Pending final freeze | Run the integrated suite against the exact final main SHA after this manuscript branch is merged. |
| C-14 | System improves learning outcomes or usability. | No controlled user study | Unsupported | Do not claim; requires human-subject evaluation. |
| C-15 | Neurofluxion is novel/first/superior to existing tools. | Related-work search | Unsupported | Avoid priority and superiority claims; frame as an implementation/integration contribution. |
| C-16 | Live literature export completed without query failures. | Actions run `39172931277`, artifact `12153262023` | Verified | 12 successful queries; 2,400 raw records; 2,074 automated deduplicated candidates; zero query, partial-query, or failed queries. Counts are not screening counts. |
| C-17 | Manuscript is evidence-grounded and IEEE-style. | `paper/neurofluxion.tex` on `research/final-implementation-paper` | Implemented | Final publication freeze still requires merge, exact-SHA integrated validation, and PDF/package generation. |

## Research validation evidence

- Graph-level central finite-difference checks cover fixed Dense/Conv2D, RNN, LSTM, GRU, Embedding→single-head Attention→RNN→Dense, and Conv2D→BatchNorm→Flatten→Dense configurations. These are independent numerical checks only for their stated graph, input, objective, dtype, seed and tolerances.
- One-batch optimizer-path checks cover LSTM/GRU, Conv2D/BatchNorm, and Embedding/Attention/RNN/Dense parameter updates. These do not prove gradient correctness or convergence.
- Adversarial browser tests expose stale inspection state after graph replacement and out-of-order responses. They remain expected failures until the runtime invariants are fixed and the tests pass as ordinary assertions.
- The literature export on main `50de1fb7328d82c8f4799bbf3feffff8fd713216` was executed on Ubuntu 24.04.5 with CPython 3.11.17. The exporter reported 12 successful queries, 2,400 raw records, 2,074 deduplicated candidates, and zero query failures. Artifact ID: `12153262023`.

## Publication gate remaining after manuscript implementation

1. Merge the final-manuscript branch.
2. Record the resulting main SHA as the publication candidate.
3. Run the integrated backend, frontend, browser, and research-validation workflows against that exact SHA.
4. Preserve the resulting logs/artifacts and update C-13.
5. Generate the final LaTeX/PDF/package from that exact revision.
6. If the venue requires live UI figures, capture them from the frozen application and label any schematic figures explicitly.
7. Do not claim systematic literature-review completion until screening/full-text/citation-snowballing evidence exists.
