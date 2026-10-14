# Related Work and Positioning Notes

**Status:** targeted initial literature scan, not a systematic review. Last checked: 2026-10-14. This document records verified source-level summaries and limits; it does not establish novelty.

## Comparison set

| Work | Primary purpose and evidence in source | Relationship to Neurofluxion | Boundary for manuscript claims |
|---|---|---|---|
| ActiVis (Kahng et al., IEEE TVCG 2018) | Interactive visual exploration of industry-scale deep neural networks and model results. Its abstract describes coordinated views including a computation-graph overview and neuron-activation view, with instance- and subset-level exploration; the authors report deployment and case studies. DOI: [10.1109/TVCG.2017.2744718](https://doi.org/10.1109/TVCG.2017.2744718); [PubMed record](https://pubmed.ncbi.nlm.nih.gov/28866557/). | Strong precedent for coordinated, multi-level visual inspection of model structure and activations. | Do not claim neural-network visualization or multi-level inspection is new. Distinguish the specific runtime traces and validation approach only if directly demonstrated. |
| CNN Explainer (Wang et al., IEEE TVCG 2021) | Interactive visualization for non-experts learning CNN internals. The paper describes model overview plus dynamic explanatory views connecting low-level mathematical operations and high-level structure; it reports a qualitative user study. DOI: [10.1109/TVCG.2020.3030418](https://doi.org/10.1109/TVCG.2020.3030418); [PubMed record](https://pubmed.ncbi.nlm.nih.gov/33048723/). | Closest precedent for connecting model-level structure to low-level operations for education. | Do not claim usability or learning gains without a study. Position any difference around inspectable execution semantics and tested implementation scope, not visualization alone. |
| Neural Network / TensorFlow Playground (Smilkov and Shan; open-source browser application) | Interactive browser-based experimentation with small neural networks, hyperparameters, training behavior, and decision regions. The official project README describes a TypeScript/D3 visualization; the official site describes it as a tiny educational network library. [Repository](https://github.com/tensorflow/playground); [official site](https://playground.tensorflow.org/). | Relevant baseline for interactive, trainable network visualization and educational exploration. | This is a software-system comparator, not a directly equivalent paper or a validated benchmark. Avoid unsupported superiority claims. |

## Evidence-backed positioning

The defensible contribution is currently an **implementation and validation study** of a specific system that combines a NumPy-backed, inspectable simulator path with separate framework-backed model execution paths. The paper should specify which path each experiment exercises and report the exact tested commit.

The preliminary dense-path audit on the pre-fix baseline found that forward-oracle comparison, trace arithmetic, and repeatability passed for one fixed dense configuration, while finite-difference gradients failed. After PRs #19 and #20 merged, a post-merge audit on the PR #21 merge revision reported all four harness checks passing: forward oracle, trace oracle, gradient finite difference, and repeatability. The workflow also exposed a shell exit-status capture bug; a correction is now included in this branch and must be rerun before treating the corrected workflow as clean. The raw evidence run is [here](https://github.com/RayAKaan/Neurofluxion/actions/runs/38491082862).

These checks are narrow: one fixed dense network and the audit harness's selected loss/activation configuration. They do **not** establish correctness for every supported layer, loss, activation, optimizer, API route, or frontend path. Nor do they establish runtime overhead, scalability, learning gains, or user-study outcomes.

## Research questions for the next validation round

1. Which layer families have complete construction → forward → backward → parameter update → trace → API/UI coverage?
2. Does each claimed loss/activation combination pass independent numerical-gradient checks?
3. Does the training path reduce loss on small deterministic synthetic tasks with reproducible seeds?
4. What is the measured inspection overhead across multiple graph sizes, with repeated trials and raw samples?
5. Which trace events are exact arithmetic records, and which are layer-level summaries or approximations?
6. After a full citation search, what narrowly defined gap remains beyond ActiVis, CNN Explainer, and Playground?

## References

1. M. Kahng, P. Y. Andrews, A. Kalro, and D. H. P. Chau, “ActiVis: Visual Exploration of Industry-Scale Deep Neural Network Models,” *IEEE Transactions on Visualization and Computer Graphics*, vol. 24, no. 1, pp. 88–97, 2018. DOI: [10.1109/TVCG.2017.2744718](https://doi.org/10.1109/TVCG.2017.2744718).
2. Z. J. Wang et al., “CNN Explainer: Learning Convolutional Neural Networks with Interactive Visualization,” *IEEE Transactions on Visualization and Computer Graphics*, vol. 27, no. 2, pp. 1396–1406, 2021. DOI: [10.1109/TVCG.2020.3030418](https://doi.org/10.1109/TVCG.2020.3030418).
3. D. Smilkov and S. Shan, *Neural Network Playground*, open-source project. [Repository](https://github.com/tensorflow/playground), [interactive site](https://playground.tensorflow.org/).
