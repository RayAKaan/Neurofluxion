# Neurofluxion Paper — Scope and Version Record

## Working title
**Neurofluxion: An Interactive System for Exploring, Simulating, and Inspecting Neural-Network Computation**

## Paper type and intended audience
Implementation/system paper for a peer-reviewed computing or visualization venue (IEEE-style two-column format is the working target; venue and page limit remain unselected). This is not a new-learning-algorithm paper, model-accuracy benchmark, or user study.

## Research question
How can an interactive application connect practical framework-backed neural-network inference with a separately inspectable simulator so that users can construct networks, step through supported computation, and examine selected intermediate values and internal mechanisms?

## Scope boundaries
- Center the simulator and inspectable computation workflow: architecture construction, forward/backward execution, parameter/gradient inspection, and supported ANN/CNN/recurrent operations.
- Explain the TensorFlow/Keras inference/trace path separately from the NumPy simulator. They are distinct computational paths; do not imply shared weights or numerical equivalence without evidence.
- Cover model-registry exploration, training telemetry, and visualization as integrated supporting capabilities.
- Report functional and numerical validation, trace fidelity, and reproducibility—not model-accuracy rankings.
- Distinguish direct runtime values from derived, sampled, estimated, illustrative, or unsupported values. Do not call approximate explainers canonical methods without verification.
- No fabricated measurements, screenshots, benchmarks, user-study findings, or claims of universal interpretability.

## Source baseline
- Repository: https://github.com/RayAKaan/NN-Visualizer
- Default branch at paper kickoff: `main`
- Source commit used to start this paper branch: `cae464530645c86a191531f0072b426f39fafcf0`
- Commit date reported by GitHub: 2026-09-11
- Commit message: `Backend: add real-time execution trace engine with per-layer timing and detail endpoints; Frontend: build execution workspace with progressive stage views, live math, tensor inspector, and execution/classic mode toggle`
- Paper branch: `research/neurofluxion-paper`
- Important: this is a starting snapshot, not the final experimental freeze. Re-freeze after audit/fixes and before final validation.

## Initial repository evidence
README at the source baseline documents a React/Vite/TypeScript client, FastAPI backend, TensorFlow/Keras inference/training, a NumPy simulator, registry-backed pretrained model adapters, and API tests. The README explicitly warns that the architecture-comparison path is same-input legacy comparison rather than a cross-task pretrained benchmark. It also describes `/stream` as topology + metrics streaming; this endpoint must be audited before any claim that it represents actual model execution.

The source tree includes `backend/services/execution_trace.py`, `backend/tests/test_execution_trace.py`, `backend/simulator/` engines and layer implementations, `backend/model_registry/`, and the frontend Lab execution workspace. Presence in the tree establishes implementation locations, not correctness or end-to-end behavior.

## Paper acceptance criteria
1. Every substantive implementation claim maps to source locations and/or test evidence in `CLAIM_EVIDENCE_LEDGER.md`.
2. Every reported result includes exact command, environment, commit SHA, and artifact path.
3. Figures use actual UI or explicitly labeled schematics; no synthetic values presented as measured runtime output.
4. Manuscript distinguishes TensorFlow/Keras trace data from NumPy simulator data.
5. No accuracy comparison unless the research question is explicitly changed (currently out of scope).
6. All citations are complete and verified; no `[?]` placeholders at submission.

## Current status (2026-09-24)
- [x] Scope and source-start commit recorded.
- [ ] Full simulator, execution-trace, model-registry, training, and frontend audit.
- [ ] Claim/evidence ledger populated from code and tests.
- [ ] Reproducible validation run and report.
- [ ] Figure capture and final LaTeX manuscript.
- [ ] Venue selection, formatting, references, and submission review.
