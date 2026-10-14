# Structured Related-Work Review Protocol

**Review date:** 2026-10-14  
**Status:** Phase 1 structured rapid review; not a completed systematic review and not evidence of novelty.

## Research question

How do existing systems and studies support neural-network architecture exploration, intermediate-value inspection, operation-level explanation, debugging during training, and provenance/reproducibility? Which of these capabilities does the pinned Neurofluxion implementation demonstrably provide, and which remain unverified?

## Search sources and query log

Searches were conducted on 2026-10-14 using the web search interface. Search results were followed to primary publisher records, PubMed records, author/project pages, or official repositories where available.

| ID | Query | Search intent |
|---|---|---|
| Q1 | neural network visualization interactive inspection model debugging IEEE TVCG visualization survey | Broad visual-analytics systems and surveys |
| Q2 | neural network visualization tool runtime tracing intermediate activations computation graph visualization papers ActiVis CNN Explainer TensorFlow Graph Visualizer | Operation/activation inspection |
| Q3 | visual analytics deep learning debugging reproducibility provenance interactive neural network visualization research papers 2020 2025 | Recent systems and reproducibility |
| Q4 | IEEE TVCG visual analytics neural networks training interpretation graph provenance interactive system | Training-time analytics |
| Q5 | systematic literature review neural network visualization interactive inspection model debugging IEEE TVCG visualization survey 2024 | Systematic/state-of-the-art surveys |
| Q6 | DLProv deep learning model provenance traces W3C PROV 2025 software impacts | Workflow provenance |
| Q7 | Neural Network Visualization Builder NNVisBuilder 2024 TVCG paper source code | General-purpose visual analytics toolkit |
| Q8 | deep neural network in situ visual analytics framework IEEE TVCG 2024 training visualization DOI 3339585 | In-situ training analytics |
| Q9 | DeepEyes progressive visual analytics neural network design paper DOI abstract | Architecture design during training |
| Q10 | LSTMVis visual analysis hidden state dynamics recurrent neural networks paper DOI abstract | Sequence-state inspection |
| Q11 | TensorFlow Graph Visualizer IEEE TVCG 2018 DOI abstract | Dataflow graph visualization |
| Q12 | GAN Lab interactive visualization generative adversarial networks IEEE TVCG 2019 DOI abstract | Interactive training/education |

## Eligibility criteria

Include a work when it meets at least one criterion:
- peer-reviewed research on neural-network visualization, visual analytics, operation/activation inspection, model debugging, or training-time analysis;
- a survey that systematically categorizes such systems;
- a maintained or historically influential tool with a primary project/paper source;
- a provenance/reproducibility system that is relevant to distinguishing computed values from model/workflow artifacts.

Prefer primary sources (publisher records, papers, author pages, official repositories). Use abstracts and public metadata only for the claims they support.

Exclude from the core comparison:
- generic post-hoc explainability methods with no material relationship to interactive internal-state inspection;
- marketing pages or secondary summaries where a primary source is available;
- claims about user benefit or numerical correctness that the source does not evaluate.

## Screening and extraction

This phase records 12 directly relevant candidates surfaced by the queries. It is a targeted, query-driven set, not a complete result count: the search interface does not provide a stable, exhaustive database export or deduplicated screening log. The included set is therefore provisional. A full systematic review still needs named bibliographic databases, exportable result sets, duplicate removal, title/abstract screening, full-text screening, and a PRISMA-style flow/count record.

For each included work, extract:
1. system purpose and intended users;
2. model families and execution stage;
3. what is displayed (architecture, weights, activations, operation math, training history, provenance);
4. whether displayed values are live, precomputed, simulated, or replayed;
5. evaluation method (case study, user study, quantitative benchmark, or none);
6. availability of source code/data and reproducibility evidence;
7. direct overlap and remaining distinction from Neurofluxion.

## Provisional included set

| Work | Venue/year | Focus | Evidence and boundary for Neurofluxion |
|---|---|---|---|
| ActiVis | IEEE TVCG, 2018 | Coordinated architecture, activation, and instance/subset exploration | Establishes architecture/activation visualization and deployed model exploration as prior art. DOI: 10.1109/TVCG.2017.2744718 |
| DeepEyes | IEEE TVCG, 2018 | Progressive visual analytics for network design during training | Establishes training-time model design and filter/layer analysis as prior art. DOI: 10.1109/TVCG.2017.2744358 |
| TensorFlow Graph Visualizer | IEEE TVCG, 2018 | Hierarchical exploration of TensorFlow dataflow graphs | Establishes interactive dataflow-graph exploration; does not by itself validate Neurofluxion's numerical trace records. DOI: 10.1109/TVCG.2017.2744878 |
| LSTMVis | IEEE TVCG, 2018 | Hidden-state dynamics and pattern search for recurrent networks | Establishes sequence-state exploration; Neurofluxion's recurrent tests are currently narrow operator contracts, not an evaluated recurrent visualization study. DOI: 10.1109/TVCG.2017.2744158 |
| CNN Explainer | IEEE TVCG, 2021 | Interactive, multi-level explanation of CNN operations and architecture | Very close precedent for linking low-level arithmetic to high-level structure; reports a qualitative user study. DOI: 10.1109/TVCG.2020.3030418 |
| NNVisBuilder | IEEE TVCG, 2024 | Toolkit for building customizable neural-network visual analytics systems with PyTorch integration | Relevant toolkit comparator; a claim of uniqueness needs a precise capability/evaluation distinction. DOI: 10.1109/TVCG.2023.3326575 |
| In-Situ Visual Analytics Framework for DNNs | IEEE TVCG, 2024 | Visual analysis integrated with DNN training | Recent training-time comparator; the distinction cannot simply be “visualization integrated with execution.” DOI: 10.1109/TVCG.2023.3339585 |
| Visual Analytics in Deep Learning: An Interrogative Survey | IEEE TVCG, 2019 issue | Taxonomy organized around users, tasks, data, methods, timing, and context | Foundational survey that should guide coverage and task framing. DOI: 10.1109/TVCG.2018.2843369 |
| State of the Art of Visual Analytics for eXplainable Deep Learning | Computer Graphics Forum, 2023 | Systematic review of visual analytics for explainable deep learning | Reports a five-month process, 196 initially collected papers, 67 retained papers, dual reading, and 38 dimensions. Use it to expand snowballing; it does not exhaust every system relevant to simulator trace fidelity. DOI: 10.1111/cgf.14733 |
| Visual Analytics for Machine Learning: A Data Perspective Survey | IEEE TVCG, 2024 | Systematic review organized by data types and data-centric tasks | Reports 143 surveyed papers; relevant broad landscape but not restricted to operation-level neural-network inspection. DOI: 10.1109/TVCG.2024.3357065 |
| DLProv: a suite of provenance services for deep learning workflow analyses | Software Impacts, 2025 | Workflow-level lineage/provenance, W3C PROV-compatible provenance graphs | Important adjacent work on workflow lineage and reproducibility. It is not the same as per-operation UI trace correctness, but it prevents broad claims that ML provenance is unaddressed. DOI: 10.1016/j.simpa.2024.100730 |
| TensorFlow Playground | Open-source educational system | Interactive small-network construction, training, and decision-boundary exploration | Useful software comparator, not a peer-reviewed direct benchmark. Repository: github.com/tensorflow/playground |

## Preliminary synthesis

1. **Visualization itself is not novel.** Architecture views, activations, low-level CNN operations, recurrent hidden states, and training-time visual analytics have substantial prior art.
2. **A broad “provenance is missing from ML” claim is not defensible.** DLProv and workflow provenance research address lifecycle traceability; any proposed gap must be scoped to the exact granularity and guarantee.
3. **Potentially defensible framing:** an implementation/evidence study of one inspectable simulator, testing whether selected operation records reconstruct arithmetic and explicitly separating those records from layer-level summaries, demo values, and unrelated execution paths.
4. **Current evidence is insufficient for novelty or superiority claims.** The graph-integration checks and source-level UI audit are being added; no controlled usability study exists; browser-level stale-state tests and cross-platform timing results remain pending.
5. **Do not equate UI affordances with fidelity.** A plausible diagram or changing value is not evidence that it came from the selected graph and current parameter revision.
6. **Review completion gate:** expand the database search, obtain exportable records, deduplicate and screen with documented counts, read full texts for the closest comparators, update the comparison matrix, and only then state a narrow research gap.

## References and source records

- ActiVis: https://pubmed.ncbi.nlm.nih.gov/28866557/
- DeepEyes: https://doi.org/10.1109/TVCG.2017.2744358
- TensorFlow Graph Visualizer: https://pubmed.ncbi.nlm.nih.gov/28866562/
- LSTMVis: https://pubmed.ncbi.nlm.nih.gov/28866526/
- CNN Explainer: https://arxiv.org/abs/2004.15004
- NNVisBuilder: https://pubmed.ncbi.nlm.nih.gov/37922180/
- In-Situ Visual Analytics Framework: https://pubmed.ncbi.nlm.nih.gov/38051629/
- Interrogative survey: https://pmc.ncbi.nlm.nih.gov/articles/PMC6703958/
- State-of-the-art review: https://onlinelibrary.wiley.com/doi/full/10.1111/cgf.14733
- Data-perspective survey: https://pubmed.ncbi.nlm.nih.gov/38261496/
- DLProv: https://pmc.ncbi.nlm.nih.gov/articles/PMC12453841/
- TensorFlow Playground: https://github.com/tensorflow/playground
- W3C PROV primer: https://www.w3.org/TR/prov-primer/


## Supplementary search round (2026-10-14)

This supplementary round adds queries aimed at newer interactive-explainability and provenance work. It is still a targeted search using web-indexed publisher and repository records, not a database-export systematic review.

| ID | Query | Added scope |
|---|---|---|
| Q13 | neural-network-centric survey interactive explainability data-driven modeling 2026 | Recent interactive-explainability frameworks and write-in mechanisms |
| Q14 | neural network activation visualization ChannelExplorer IEEE TVCG 2026 | Recent layer/channel activation visual analytics |
| Q15 | machine learning provenance PROV-JSON yProv4ML 2025 | Fine-grained provenance capture in ML execution |
| Q16 | end-to-end provenance MLflow Git MLflow2PROV Information Systems 2025 | Pipeline/run/code evolution provenance |
| Q17 | machine learning experiment management tools controlled experiment practitioner survey 2024 | Empirical evidence about experiment traceability and user performance |

### Additional eligible candidates

| Work | Venue/year | Focus | Relevance and claim boundary |
|---|---|---|---|
| Interactive explainability in data-driven modeling: A neural-network-centric survey | Neurocomputing, 2026; DOI: 10.1016/j.neucom.2026.133535 | Neural-network-centric survey covering interactive explainability across data/knowledge, problem, model, and post-hoc levels; discusses executable write-ins and traceable updates. | Recent broad review. It further weakens any claim that interaction and traceable adjustment are unaddressed. It is not a direct test of Neurofluxion's arithmetic trace fidelity. |
| ChannelExplorer: Exploring Class Separability Through Activation Channel Visualization | IEEE TVCG, 2026; DOI: 10.1109/TVCG.2026.3669148 | Dataset-to-example drilldown with coordinated activation-channel, similarity, and heatmap views; supports CNNs, GANs, ResNet, and Stable Diffusion; reports expert evaluation. | Recent, close activation-visualization prior art. It focuses on class separability and image-model activation analysis, not a simulator's operation-by-operation numerical reconstruction. |
| yProv4ML: Effortless provenance tracking for machine learning systems | SoftwareX, 2025; DOI: 10.1016/j.softx.2025.102298 | Captures ML execution provenance in PROV-JSON with minimal code changes. | Directly rules out a broad claim that machine-learning execution provenance is absent. Distinguish any Neurofluxion claim by its exact graph/operation/value granularity and guarantees, if those are actually demonstrated. |
| Capturing end-to-end provenance for machine learning pipelines (MLflow2PROV) | Information Systems, 2025; DOI: 10.1016/j.is.2024.102495 | PROV-compliant lineage from MLflow artifacts and Git activity, including pipeline, experiment, and artifact relationships. | Direct prior art for end-to-end ML pipeline provenance and evolution. It is complementary to per-operation runtime trace validation. |
| Machine learning experiment management tools: a mixed-methods empirical study | Empirical Software Engineering, 2024; DOI: 10.1007/s10664-024-10444-w | Survey of 81 practitioners and controlled experiment with 15 student developers; evaluates tool support for tracking/retrieving ML assets and task performance. | Shows that user-benefit questions around traceability tools have empirical precedents. Neurofluxion cannot claim usability or learning benefit without its own appropriate evaluation. |

### Revised synthesis after supplementary search

1. Architecture visualization, activation exploration, interactive explainability, recurrent-state inspection, training-time visual analytics, and ML provenance all have established and continuing prior art.
2. Provenance is not a new general problem: yProv4ML addresses execution provenance; MLflow2PROV addresses end-to-end pipeline/code lineage; DLProv addresses workflow provenance. Neurofluxion's research question must be narrowed to whether specific operation-level records faithfully correspond to the selected graph execution and whether the UI preserves that association under state changes.
3. The candidate gap is an **evaluation question**, not a novelty assertion: which displayed simulator values can be independently reconstructed from operation records, which are layer summaries or synthetic/demo values, and what happens under graph replacement or delayed responses?
4. A browser test now reproduces a stale-response UI state defect on the tested branch (PR #40). This is evidence of a concrete provenance risk in the current implementation, not evidence of a novel technique or of user benefit.
5. The review remains incomplete. The search interface did not provide exportable, deduplicated database results; there is no PRISMA-style screening flow, dual-reviewer process, or full-text extraction table. These additional candidates are provisional and must be verified against full texts before the manuscript is submitted.

### Primary source links for supplementary candidates

- Interactive explainability survey (2026): https://doi.org/10.1016/j.neucom.2026.133535
- ChannelExplorer (2026): https://doi.org/10.1109/TVCG.2026.3669148
- yProv4ML (2025): https://doi.org/10.1016/j.softx.2025.102298
- MLflow2PROV (2025): https://doi.org/10.1016/j.is.2024.102495
- ML experiment management study (2024): https://doi.org/10.1007/s10664-024-10444-w
