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
