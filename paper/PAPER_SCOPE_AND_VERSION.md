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

## Initial status snapshot (2026-09-24; superseded by the current research status below)
- [x] Scope and source-start commit recorded.
- [ ] Full simulator, execution-trace, model-registry, training, and frontend audit.
- [ ] Claim/evidence ledger populated from code and tests.
- [ ] Reproducible validation run and report.
- [ ] Figure capture and final LaTeX manuscript.
- [ ] Venue selection, formatting, references, and submission review.


## Current status after research PRs #47–#55 (2026-10-14)

- [x] Research-only graph-level finite-difference oracles extended to fixed RNN/LSTM/GRU, Embedding→Attention→RNN→Dense, and Conv2D→BatchNorm→Flatten→Dense graphs.
- [x] One-batch training-update contracts added for recurrent, Conv2D/BatchNorm, and Embedding/Attention/RNN/Dense paths.
- [x] Claim/evidence ledger reconciled for these new bounded results.
- [x] Reproducible OpenAlex/Crossref metadata-export script, manifest, deduplicated records, and reviewer screening CSV implemented with offline tests.
- [ ] Execute and archive a live literature export; inspect source/query failures and adjudicate duplicates.
- [ ] Complete screening, full-text extraction, and citation snowballing; update counts only from the actual review log.
- [ ] Fix stale-response/graph-switch UI provenance defects only after explicit authorization to modify runtime/application code; current browser tests remain expected failures.
- [ ] Complete remaining full source audit and verify paper claims, figures, and citations against the frozen revision.
- [ ] Freeze a final source SHA and run full backend tests, frontend build, browser E2E, and numerical research audits against that exact SHA.
- [ ] Generate final LaTeX/PDF and submission package only after the evidence freeze; choose a venue and complete formatting/submission review.

## Live literature export audit (2026-10-14)

The first post-merge run is archived at [Actions run 38592362340](https://github.com/RayAKaan/Neurofluxion/actions/runs/38592362340), artifact ID `11907269115`. Its job was green but the manifest exposes six OpenAlex page-2 HTTP 401 failures; the output contains 1,200 raw Crossref records and 1,127 deduplicated candidates, with no human screening decisions. This is **not** a completed two-source search. The exporter is being hardened to retain partial-page results and mark any query failure as a failed workflow while preserving artifacts. Do not start final screening or report inclusion counts from this incomplete run as though it were comprehensive.

The source baseline recorded above is historical, not the current paper baseline. Do not regenerate or label a final paper package until the final source revision is frozen and the remaining evidence gates are recorded.


## Post-PR #57 export result and figure/page audit (2026-10-14)

PR #57 merged as `c715d5d5e5a8838b7aa2c56e625dcd26c1dc4419`. The push-triggered export run [38593723201](https://github.com/RayAKaan/Neurofluxion/actions/runs/38593723201) correctly failed with exit code 2 while preserving artifact ID `11908119812`. The manifest summary reported 1,800 raw records, 1,568 deterministically deduplicated candidates, six successful queries, six partial failures, and zero fully failed queries. The six partial failures are OpenAlex requests that could not retrieve page 2; the workflow environment showed an empty `OPENALEX_API_KEY`. These counts are discovery output only, not human-screened or included papers. Do not claim a complete two-source search until OpenAlex access is restored and the new manifest confirms all planned queries succeeded.

The tracked `paper/neurofluxion.tex` now contains a source-grounded dual-path architecture diagram in place of its placeholder. The companion `FIGURE_AUDIT_AND_8_PAGE_PLAN.md` audits the earlier TikZ draft and defines a compact three-figure layout. This does not make the manuscript submission-ready: source audit, literature screening, reference verification, frozen-SHA integrated validation, and a compiled eight-page proof remain open.
