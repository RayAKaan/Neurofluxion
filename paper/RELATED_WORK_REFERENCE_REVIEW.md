# Supplied Reference-Paper Review

**Status:** research-stage reference integration; not a completed systematic review.

**Source revision:** `7758e4ebf402c0ccd5a8bb5d3bf6de002cb4c580`

This record captures the eight papers supplied for the Neurofluxion paper work on 2026-10-19. The purpose is to preserve exactly how each source is being used, without turning a citation into evidence for claims that the source does not establish.

## 1. Neural network interpretability with layer-wise relevance propagation: novel techniques for neuron selection and visualization

- Authors: Deepshikha Bhati, Fnu Neha, Md Amiruzzaman, Angela Guercio, Deepak Kumar Shukla, Ben Ward.
- Source: arXiv:2412.05686 (submitted 2024-12-07).
- URL: https://arxiv.org/abs/2412.05686
- Source-supported relevance: LRP-based interpretability, selected-neuron analysis, neural-network graph visualization, critical-path visualization, heatmaps, and deconvolutional feature-map reconstruction are discussed as an interpretability workflow.
- Neurofluxion use: background for neuron/path visualization and for keeping any interpretability claim method-specific rather than calling a generic inspection panel a canonical XAI method.
- Boundary: the source is a VGG16 case study and does not establish Neurofluxion's own interpretability implementation.

## 2. VOICE: Variance of Induced Contrastive Explanations to quantify Uncertainty in Neural Network Interpretability

- Authors: Mohit Prabhushankar, Ghassan AlRegib.
- Source: arXiv:2406.00573 (submitted 2024-06-01).
- URL: https://arxiv.org/abs/2406.00573
- Source-supported relevance: gradient-based post-hoc visual explanations can exhibit predictive uncertainty; the paper proposes a plug-in approach to visualize and quantify remaining uncertainty and reports experiments across datasets, explanation methods, and architectures.
- Neurofluxion use: strengthens the manuscript's limitation that visualization/attribution output should not automatically be treated as a trustworthy explanation; future evaluation should distinguish numerical inspection from explanation reliability.
- Boundary: Neurofluxion does not currently claim to implement VOICE or an explanation-uncertainty metric.

## 3. Efficient Visualization of Neural Networks with Generative Models and Adversarial Perturbations

- Author: Athanasios Karagounis.
- Source: arXiv:2409.13559 (submitted 2024-09-20).
- URL: https://arxiv.org/abs/2409.13559
- Source-supported relevance: generative visualization is connected to adversarial perturbations, with fooling rate proposed as a quantitative visualization-quality signal in that work.
- Neurofluxion use: positions visualization evaluation as a distinct research question from simply rendering internal values, and supports avoiding unsupported claims that a visualization is inherently faithful.
- Boundary: Neurofluxion does not implement this generative/adversarial visualization method and the paper does not use its fooling-rate metric.

## 4. Visual Analytics in Explaining Neural Networks with Neuron Clustering

- Authors: Gulsum Alicioglu, Bo Sun.
- Venue: *AI*, 2024, 5(2), 465-481. DOI: 10.3390/ai5020023.
- URL: https://www.mdpi.com/2673-2688/5/2/23
- Source-supported relevance: combines visual analytics with neuron clustering, providing overall and detailed views of neuron clusters and addressing visual clutter in neural-network explanations.
- Neurofluxion use: direct related-work context for interactive architecture/neuron inspection and the tradeoff between detailed graph views and visual clutter.
- Boundary: its clustering approach and reported case-study results are not Neurofluxion functionality or evaluation results.

## 5. Interpretability of deep neural networks: A review of methods, classification and hardware

- Authors: Thanasis Antamis, Anastasis Drosou, Thanasis Vafeiadis, Alexandros Nizamis, Dimosthenis Ioannidis, Dimitrios Tzovaras.
- Venue: *Neurocomputing*, 601, 128204, 2024. DOI: 10.1016/j.neucom.2024.128204.
- URL: https://www.sciencedirect.com/science/article/abs/pii/S0925231224009755
- Source-supported relevance: broad review of DNN interpretability methods, classification, and hardware considerations.
- Neurofluxion use: establishes broader interpretability context and supports a taxonomy-oriented related-work framing rather than presenting one explanation technique as representative of the entire field.
- Boundary: review coverage does not validate Neurofluxion's specific simulator or frontend behavior.

## 6. Explainable Artificial Intelligence: A Systematic Review of Progress and Challenges

- Authors: Azza Abd El Monem Mohamed, Khaled J. Abdelqader, Khaled Shaalan.
- Venue: *Intelligent Systems with Applications*, 28, article 200595, 2025.
- URL: https://www.sciencedirect.com/science/article/pii/S2667305325001218
- Source-supported relevance: systematic review framing for XAI progress and challenges.
- Neurofluxion use: supports the manuscript's treatment of explainability as a broader field with unresolved evaluation and trust challenges.
- Boundary: this source is not evidence that Neurofluxion satisfies a systematic-review criterion or that its own explanations are trustworthy.

## 7. Explaining Deep Neural Networks and Beyond: A Review of Methods and Applications

- Authors: Wojciech Samek, Grégoire Montavon, Sebastian Lapuschkin, Christopher J. Anders, Klaus-Robert Müller.
- Venue: *Proceedings of the IEEE*, 109(3), 247-278, 2021. DOI: 10.1109/JPROC.2021.3060483.
- URL: https://proceedingsoftheieee.ieee.org/explaining-deep-neural-networks-and-beyond-a-review-of-methods-and-applications
- Source-supported relevance: foundational review of post-hoc DNN explanation methods, their theoretical foundations, comparative evaluation, practical use, and application scenarios.
- Neurofluxion use: primary background citation for the distinction between internal computation inspection and post-hoc explanation/attribution.
- Boundary: the review does not establish any Neurofluxion-specific XAI implementation.

## 8. Explainable Artificial Intelligence (XAI): What we know and what is left to attain Trustworthy Artificial Intelligence

- Authors: Sajid Ali, Tamer Abuhmed, Shaker El-Sappagh, Khan Muhammad, Jose M. Alonso-Moral, Roberto Confalonieri, Riccardo Guidotti, Javier Del Ser, Natalia Díaz-Rodríguez, Francisco Herrera.
- Venue: *Information Fusion*, 99, 101805, 2023. DOI: 10.1016/j.inffus.2023.101805.
- URL: https://www.sciencedirect.com/science/article/pii/S1566253523001148
- Source-supported relevance: XAI concepts, trustworthy-AI concerns, post-hoc explainability, and assessment challenges.
- Neurofluxion use: supports conservative wording around transparency, trust, and evaluation of explanations.
- Boundary: it does not validate Neurofluxion's implementation or establish a user-study result.

## Integration decision

These references are being used as related-work and framing sources, not as evidence that Neurofluxion implements the cited methods. The manuscript should explicitly distinguish:

1. **Execution inspection:** directly exposing values produced by a selected execution path.
2. **Visualization/visual analytics:** presenting those values or structures interactively.
3. **Post-hoc explanation/XAI:** attribution or explanation methods such as LRP, integrated gradients, LIME, SHAP, or related techniques.
4. **Explanation evaluation:** assessing faithfulness, uncertainty, robustness, or other properties of explanations.

This distinction is important because the current paper is an artifact-centered system paper. The supplied literature strengthens the related-work framing but does not remove the existing evidence boundary around Neurofluxion's own interpretability modules.

## Citation-audit note

The eight sources should be retained in the final bibliography only after bibliographic metadata is rechecked against the publisher/arXiv records at the final source freeze. The ScienceDirect and MDPI pages were not directly fetchable in the initial publisher requests, so metadata for those entries was cross-checked using independent indexed bibliographic records/search results. No search count, screening count, or systematic-review conclusion is inferred from this eight-paper list.
