# Figure Audit and Eight-Page IEEE Implementation-Paper Plan

**Audit date:** 2026-10-14  
**Repository baseline:** `c715d5d5e5a8838b7aa2c56e625dcd26c1dc4419`  
**Scope:** research figures and manuscript layout only; no simulator/runtime code is changed.

## Readiness conclusion

An eight-page IEEE conference-style implementation paper is feasible if the manuscript is narrowed to the inspectable-simulator implementation, the two independent execution paths, and the evidence actually collected. It is not yet proven to fit: the current tracked manuscript has not been compiled and counted at the current baseline, and the earlier working PDF was ten pages. Page compliance must be confirmed from a compiled PDF after selecting a specific venue and its author instructions.

The paper should contain three figures at most: (1) source-grounded dual-path system boundary, (2) dense trace/state granularity with explicit limits, and (3) a compact evidence figure or table generated from validated results. Until raw quantitative outputs are available, use an evidence matrix/table rather than a fabricated cost/error plot. Keep one small capability/evidence table and one related-work table only if they fit legibly.

## Audit of the earlier TikZ draft

### 1. Dual-path architecture diagram — revise

**Problem:** the earlier diagram places TensorFlow/Keras and the NumPy simulator under one API and shows a dashed connection from the Keras branch toward serialized payloads. Even with a note, this can suggest shared execution state or a common trace pipeline.

**Change:** show Keras tracing and simulator execution as separate horizontal lanes. Label the Keras lane as activation-model outputs, with some payloads bounded/sampled. Label the simulator lane as graph execution and serialized steps, with fine-grained matrix-multiply/bias/activation steps documented for dense-only execution and coarser layer-level steps for many other families. Show the `/stream` endpoint outside both lanes as a synthetic demonstration stream. Explicitly state that weights and numerical state are not shared automatically.

**Status:** the tracked `paper/neurofluxion.tex` placeholder has been replaced with this source-grounded vector diagram. It still requires a LaTeX build and visual inspection for overlap, text size, and column fit.

### 2. Graph/data-model diagram — simplify or merge

The prior graph-construction diagram shows layer configuration, validation, graph, parameter store, execution cache, and execution engines. It is conceptually useful but overlaps with the dual-path diagram. For an eight-page paper, merge only the essential simulator lifecycle into Figure 2 or remove this standalone figure. Keep state labels such as (X,Z,A) only where their meanings match the audited implementation.

### 3. Dense forward/backward trace diagram — retain with qualification

The earlier dense diagram is the most technically informative because it explains the observed finer granularity: matrix multiplication, bias addition, and activation can appear as distinct dense-only steps. Retain it only if the caption says that this granularity is not uniform across all layer families. Any backward arrows and gradient labels must be consistent with the exact tested graph/loss contract; avoid implying that every registered layer has verified backward support.

### 4. Granularity ladder — do not present as universal coverage

The model → layer → operation → numerical-state ladder is a useful explanatory abstraction, not evidence that all four levels are available for every operator. If retained, label it as an analysis taxonomy and add a visible note that operation-level trace detail is currently heterogeneous. Prefer combining it with the dense trace diagram to save space.

### 5. LSTM gate diagram — defer unless the recurrent detail contract is verified

The earlier diagram implies that gate activations, cell state, and hidden state are all retained as inspectable objects and that the displayed equations match the implementation. The current graph-gradient/training tests support only specified recurrent configurations; they do not by themselves establish every LSTM detail endpoint or UI view. Include this diagram only after checking the recurrent layer implementation, detail endpoint, and corresponding tests together. Otherwise report recurrent validation in a table and omit the diagram.

### 6. Optimizer-state diagram — remove from the eight-page main paper

The earlier diagram depicts SGD, Momentum, and Adam as a unified verified pipeline. A conceptual diagram is not a test of optimizer equations, configuration propagation, state reset, or parameter updates. The recent training contracts cover particular one-batch graph families, not every optimizer/configuration. Replace this figure with a concise evidence table and list only optimizer/loss combinations actually tested.

### 7. Trace/replay pipeline — qualify or remove “replay”

The prior trace pipeline ends in “Replay / Inspector / Equation Panel.” This wording implies replay fidelity. Replay/reconstruction is still an open validation gate. If a trace pipeline is retained, rename the endpoint “client inspection views” and state that deterministic replay has not yet been established.

### 8. Observability-cost plot with axis ticks — remove, do not cosmetically improve

The earlier TikZ source contains a plot titled “Normalized cost” with hard-coded coordinates (for example, 0.18 to 0.90) and a second “trace size” series. The x-axis uses numeric positions 0–4 relabeled as categories (Model, Layer, Operation, State, Trace), and the y-axis uses a normalized 0–1 scale without a defined denominator, units, measurement protocol, raw samples, uncertainty, or confidence interval. The dashed lines and “illustrative” caption reduce but do not eliminate the risk that readers interpret it as measured evidence.

**Decision:** remove this plot from the research paper. Do not merely relabel its ticks or polish the style. Replace it only after a reproducible benchmark produces raw timing and payload-size samples. A valid future plot should have a defined x-axis (graph size or explicit categorical labels), a defined y-axis and units, separate panels or clearly distinguishable series, repeated-run dispersion/error bars, and a caption naming the hardware, baseline, trace mode, and measurement boundary. Do not use a normalized axis unless the denominator and normalization procedure are explicitly defined.

### 9. UI screenshots — one authentic capture is optional, not a substitute for evidence

A real application screenshot could demonstrate the inspection workflow, but it must be captured from the running application and record the selected graph/model, input, execution mode, and whether values are direct, derived, sampled, illustrative, or unavailable. Do not use a mock screenshot or synthetic values presented as live output. If an authentic capture cannot be produced reproducibly, use the vector architecture diagram and omit the screenshot.

## Recommended eight-page allocation

This is a planning budget, not a verified page count.

| Page budget | Content | Space discipline |
|---|---|---|
| 1 | Title, abstract, introduction, contributions | One concise motivation; three bounded contributions |
| 2 | Related work and system boundary | Compact comparison; no broad novelty claim |
| 3 | Simulator representation and execution semantics | One equation block and the dense trace figure |
| 4 | Keras trace, simulator trace, state/provenance limitations | Dual-path architecture figure; define sampled/timing fields |
| 5 | Numerical and training evidence | Evidence matrix/table; state exact tested configurations |
| 6 | Validation protocol and failure modes | Include only open checks that explain boundaries; avoid presenting future work as results |
| 7 | Discussion, limitations, reproducibility, threats to validity | Concise construct/internal/external/reproducibility threats |
| 8 | Conclusion and references | Reserve enough room for complete, verified references; remove optional content before shrinking fonts |

## Page-reduction rules

1. Remove the illustrative cost plot and any unexecuted result tables.
2. Limit the main paper to three figures and two compact tables.
3. Merge overlapping architecture/data-model/granularity diagrams.
4. Replace long prose lists of operator families with an evidence matrix.
5. Delete repeated caveats when a single evidence-boundary table can carry them, but retain material limitations near the affected claim.
6. Keep equations only when they explain a source-confirmed operation or a reported validation metric.
7. Do not shrink labels below legibility to force an eight-page fit; simplify the figure instead.
8. Preserve full bibliographic details and verify each reference against a primary source. Never save pages by leaving TODOs, placeholder citations, or abbreviated author names.
9. After compiling, inspect all pages at normal zoom and check figure text, float placement, references, clipping, and page count. Compile to convergence and record the exact source SHA and toolchain.

## Publication blockers that page reduction cannot solve

- The live literature export at run 38593723201 failed as intended: six OpenAlex page-2 requests returned HTTP 401 with an empty `OPENALEX_API_KEY`; the artifact was preserved. This is not a complete two-source search.
- Human duplicate adjudication, title/abstract screening, full-text review, and citation snowballing are not complete.
- UI stale-response and graph-switch invalidation tests remain expected failures; green expected-failure handling does not mean the defect is fixed. Runtime changes remain outside the current research-only authorization.
- A final source SHA has not been frozen for the paper, and the full integrated validation has not been rerun against one frozen revision.
- The tracked bibliography and author metadata still contain TODOs. The manuscript must not be called submission-ready until they are resolved.

## Immediate order of work

1. Build and inspect the revised source-grounded architecture figure.
2. Replace the paper's remaining future-tense evaluation-plan sections with a short, factual report of the merged bounded numerical/training evidence plus a transparent limitations paragraph.
3. Complete the remaining source audit and update the claim ledger.
4. Restore OpenAlex access and rerun the literature export; then screen records manually.
5. Verify references and venue-specific requirements.
6. Freeze the source revision, run integrated CI and numerical audits on that exact SHA, compile the paper, inspect every page, and only then certify an eight-page package.
