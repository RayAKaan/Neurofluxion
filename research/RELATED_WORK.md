# Related Work and Research Positioning

**Status:** Phase 1 structured rapid review recorded in [LITERATURE_REVIEW_PROTOCOL.md](LITERATURE_REVIEW_PROTOCOL.md). This is not a completed systematic review and does not establish novelty. Last searched: 2026-10-14.

## Comparison set

| Work | Primary purpose and evidence | Relationship to Neurofluxion | Boundary for manuscript claims |
|---|---|---|---|
| ActiVis (Kahng et al., IEEE TVCG 2018) | Interactive, coordinated architecture and activation views for instance- and subset-level exploration; deployment and case studies reported. DOI: [10.1109/TVCG.2017.2744718](https://doi.org/10.1109/TVCG.2017.2744718); [PubMed](https://pubmed.ncbi.nlm.nih.gov/28866557/). | Strong precedent for coordinated graph and activation exploration. | Do not claim architecture/activation visualization or multi-level inspection is new. |
| DeepEyes (Pezzotti et al., IEEE TVCG 2018) | Progressive visual analytics for network design during training; use cases include finding superfluous filters/layers and adapting models. DOI: [10.1109/TVCG.2017.2744358](https://doi.org/10.1109/TVCG.2017.2744358). | Prior art for training-time visual analysis and architecture refinement. | Do not position training-time interaction as a novelty by itself. |
| TensorFlow Graph Visualizer (Wongsuphasawat et al., IEEE TVCG 2018) | Hierarchical interactive exploration of TensorFlow dataflow graphs. [PubMed record](https://pubmed.ncbi.nlm.nih.gov/28866562/); DOI: [10.1109/TVCG.2017.2744878](https://doi.org/10.1109/TVCG.2017.2744878). | Establishes dataflow graph visualization and nested exploration. | Graph visualization is distinct from independently verifying numerical operation records. |
| LSTMVis (Strobelt et al., IEEE TVCG 2018) | Exploration of recurrent hidden-state dynamics, matching selected state patterns to similar patterns and annotations. DOI: [10.1109/TVCG.2017.2744158](https://doi.org/10.1109/TVCG.2017.2744158); [PubMed](https://pubmed.ncbi.nlm.nih.gov/28866526/). | Prior art for sequence-state visualization. | Neurofluxion's narrow recurrent operator tests do not establish equivalent visualization or user value. |
| CNN Explainer (Wang et al., IEEE TVCG 2021) | Interactive multi-level CNN explanation linking architecture to low-level operations; qualitative user study reported. DOI: [10.1109/TVCG.2020.3030418](https://doi.org/10.1109/TVCG.2020.3030418); [arXiv](https://arxiv.org/abs/2004.15004). | Close precedent for connecting arithmetic to network structure and interactive learning. | Do not claim educational benefit without a study; do not claim operation-level explanation is itself new. |
| TensorFlow Playground | Browser-based small-network experimentation and training. [Repository](https://github.com/tensorflow/playground); [official site](https://playground.tensorflow.org/). | Relevant educational software comparator. | Not a peer-reviewed direct benchmark; avoid superiority claims. |
| NNVisBuilder (Lai et al., IEEE TVCG 2024) | Toolkit for prototyping customizable neural-network visual analytics systems with PyTorch integration. DOI: [10.1109/TVCG.2023.3326575](https://doi.org/10.1109/TVCG.2023.3326575); [PubMed](https://pubmed.ncbi.nlm.nih.gov/37922180/). | Relevant general-purpose visual analytics toolkit. | Any distinction must be based on measured, specific guarantees rather than a broad claim of flexibility. |
| In-Situ Visual Analytics Framework for DNNs (Li et al., IEEE TVCG 2024) | Visual analytics integrated into deep-network training. DOI: [10.1109/TVCG.2023.3339585](https://doi.org/10.1109/TVCG.2023.3339585); [PubMed](https://pubmed.ncbi.nlm.nih.gov/38051629/). | Recent training-time comparator. | “Visualization integrated with execution” is not a sufficient novelty claim. |
| Visual Analytics in Deep Learning: An Interrogative Survey (Hohman et al., IEEE TVCG 2019 issue) | Survey organized around users, tasks, data, methods, timing, and context. DOI: [10.1109/TVCG.2018.2843369](https://doi.org/10.1109/TVCG.2018.2843369); [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC6703958/). | Provides task-level framing for a wider comparison. | The paper must define a narrow task and guarantee, not merely list features. |
| State of the Art of Visual Analytics for eXplainable Deep Learning (La Rosa et al., Computer Graphics Forum 2023) | Structured review describing a five-month process, 196 initially collected papers, and 67 retained papers, assessed by two readers and 38 dimensions. DOI: [10.1111/cgf.14733](https://doi.org/10.1111/cgf.14733); [full text](https://onlinelibrary.wiley.com/doi/full/10.1111/cgf.14733). | Important systematic landscape and snowballing source. | Its scope is visual analytics for explainable deep learning; it does not automatically cover all simulator-fidelity work. |
| Visual Analytics for Machine Learning: A Data Perspective Survey (Wang et al., IEEE TVCG 2024) | Systematic review of 143 papers organized around data types and data-centric tasks. DOI: [10.1109/TVCG.2024.3357065](https://doi.org/10.1109/TVCG.2024.3357065); [PubMed](https://pubmed.ncbi.nlm.nih.gov/38261496/). | Broad, recent map of visual analytics for ML. | Useful coverage check but not specific to operation-level trace correctness. |
| DLProv (Pina et al., Software Impacts 2025) | Framework-independent workflow provenance services and W3C PROV-compatible provenance graphs for deep-learning workflows. DOI: [10.1016/j.simpa.2024.100730](https://doi.org/10.1016/j.simpa.2024.100730); [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12453841/). | Adjacent provenance/reproducibility work; requires narrowing any claim that ML provenance is missing. | Workflow lineage differs from per-operation UI trace fidelity, but provenance itself is established prior art. |
| W3C PROV Primer | Standard primer for representing entities, activities, and relations describing digital-object origins. [W3C](https://www.w3.org/TR/prov-primer/). | Provides a vocabulary for describing provenance requirements. | Do not imply provenance concepts or standards are novel. |

## Evidence-backed positioning

The defensible framing is an **implementation and evidence study** of one specific system: which selected NumPy simulator operations can be reconstructed from recorded values, which paths are only layer-level summaries, and how demo values or separate execution paths are distinguished from actual graph computation.

The repository now has a frozen-revision integrated validation result: workflow [38566683696](https://github.com/RayAKaan/Neurofluxion/actions/runs/38566683696) passed backend tests, frontend build, browser E2E, and a dense numerical audit on the frozen target SHA `30ff803cb342c494310ff35832df43a092d3d74d`. This does not prove all-layer correctness or validate every later merge commit. The graph-integration contracts, UI provenance source audit, and cross-platform benchmark are being added in separate research PRs; their results must be checked before they are treated as evidence.

These sources establish that neural-network visualization, operation-level explanation, recurrent-state inspection, training-time visual analytics, and workflow provenance each have substantial prior art. Any distinct contribution must be narrowed to explicit, independently verified guarantees for this implementation. No user study currently supports learning or usability claims, and the structured review is not yet systematic enough to support a novelty-priority claim.

## Open questions

1. Which operator families have complete construction → forward → backward → parameter update → trace → API/UI coverage?
2. Are visible values linked to the correct graph, execution, and parameter revision under out-of-order responses and graph replacement?
3. How do raw timing distributions differ across environments, and how much of the ratio is trace-record creation and serialization?
4. What exact gap remains after exportable database searches, deduplication, dual screening, full-text review, and citation snowballing?

## References

See the full query log, eligibility rules, screening limitations, and provisional synthesis in [LITERATURE_REVIEW_PROTOCOL.md](LITERATURE_REVIEW_PROTOCOL.md).


## Supplementary related work added 2026-10-14

The second targeted search round added five sources that materially sharpen the claim boundaries:

- **Interactive explainability in data-driven modeling: A neural-network-centric survey** (Neurocomputing, 2026; [DOI](https://doi.org/10.1016/j.neucom.2026.133535)) reviews interactive write-in mechanisms across data/knowledge, problem, model, and post-hoc levels. It is broad interactive-explainability prior art, not a direct test of operation-trace arithmetic.
- **ChannelExplorer** (IEEE TVCG, 2026; [DOI](https://doi.org/10.1109/TVCG.2026.3669148)) explores activations across layers and channels, from dataset overview to individual examples, and reports expert evaluation. This is close activation-visualization prior art; its goal is class separability, not operation-level simulator trace fidelity.
- **yProv4ML** (SoftwareX, 2025; [DOI](https://doi.org/10.1016/j.softx.2025.102298)) captures ML execution provenance in PROV-JSON with minimal code modification.
- **MLflow2PROV** (Information Systems, 2025; [DOI](https://doi.org/10.1016/j.is.2024.102495)) captures PROV-compliant provenance relationships across MLflow artifacts and Git history.
- **Machine learning experiment management tools: a mixed-methods empirical study** (Empirical Software Engineering, 2024; [DOI](https://doi.org/10.1007/s10664-024-10444-w)) reports a survey of 81 practitioners and a controlled experiment with 15 student developers on asset tracking/retrieval performance.

This round further narrows the candidate research gap to an empirical **fidelity and state-association question**: can each visible simulator value be tied to the selected graph/execution and independently reconstructed from its operation record, especially after asynchronous state changes? It does not support a claim that interactive visualization, provenance, or traceability is novel in general.

The search remains a structured rapid review, not a completed systematic review. Search exports, deduplication, screening counts, full-text extraction, and independent screening remain open.
