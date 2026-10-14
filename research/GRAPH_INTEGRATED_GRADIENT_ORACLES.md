# Graph-Integrated Gradient Oracles

## Scope

This phase extends independent numerical validation beyond layer-isolated gradient checks. The tests call the actual graph-level backward API and compare returned parameter gradients with central finite differences of the scalar MSE objective.

Two fixed graphs are included:
1. Conv2D → Flatten → Dense output, exercising convolutional and dense parameters in one graph.
2. Short tanh RNN → Dense output, exercising recurrent and dense parameters in one graph.

The test synchronizes packed graph parameters into their owning layer instances before each objective evaluation. It uses fixed seeds and records explicit finite-difference tolerances in the test source.

## Interpretation

A pass establishes agreement only for these fixed graph shapes, parameter initializations, input/target values, loss, dtype, and tolerances. It does not validate other activations, shapes, batch sizes, optimizers, or all layer families. Finite differences are themselves sensitive to epsilon and floating-point error, so tolerances must be inspected rather than copied as a universal threshold.

## CI

Workflow: .github/workflows/graph-gradient-oracles.yml. The workflow gates on the two graph-integrated finite-difference tests. Any failure must be triaged as a test-oracle issue or an implementation defect; runtime source changes remain outside this research PR.

## First numerical result (2026-10-14)

The dedicated oracle workflow [38575599616](https://github.com/RayAKaan/Neurofluxion/actions/runs/38575599616) passed both graph-integrated finite-difference tests:
- Conv2D → Flatten → Dense output;
- short tanh RNN → Dense output.

The test uses central differences with epsilon 1e-3 and comparison tolerances rtol 2e-2 / atol 2e-3. This supports the tested mixed-layer MSE-gradient cases only; it does not establish correctness for other operators, losses, shapes, or training configurations.

The full-stack workflow [38575599333](https://github.com/RayAKaan/Neurofluxion/actions/runs/38575599333) also passed backend tests, frontend build, and the existing browser E2E on the PR head. This is integration-suite evidence for that revision; it does not extend the two finite-difference graph cases.


## Recurrent-family extension (PR #46 workstream)

Two additional fixed graph cases extend the same central finite-difference protocol to LSTM → Dense and GRU → Dense. Both use short three-step sequences, two-dimensional inputs, hidden size two, one linear output, fixed seeds, and the same MSE/tolerance settings.

PR #46 was merged as `0ef95cf472b5492140d836d475d4b4528e6648c6`. Its graph-oracle workflow [38580463218](https://github.com/RayAKaan/Neurofluxion/actions/runs/38580463218), research validation [38580463113](https://github.com/RayAKaan/Neurofluxion/actions/runs/38580463113), frozen-main integrated validation [38580463272](https://github.com/RayAKaan/Neurofluxion/actions/runs/38580463272), and full-stack validation [38580463094](https://github.com/RayAKaan/Neurofluxion/actions/runs/38580463094) all completed successfully. LSTM → Dense and GRU → Dense therefore passed the same fixed-case central finite-difference protocol on PR #46's tested head. The evidence remains limited to those graphs and the stated numerical setup; runtime simulator source was unchanged.


## Composite embedding-attention-recurrent graph (PR #47)

PR #47 merged after the dedicated graph-oracle workflow [38582499269](https://github.com/RayAKaan/Neurofluxion/actions/runs/38582499269), research validation [38582499212](https://github.com/RayAKaan/Neurofluxion/actions/runs/38582499212), frozen-main integrated validation [38582499184](https://github.com/RayAKaan/Neurofluxion/actions/runs/38582499184), and full-stack validation [38582499053](https://github.com/RayAKaan/Neurofluxion/actions/runs/38582499053) passed. The Embedding → single-head Attention → RNN → Dense oracle uses repeated discrete input indices, held fixed during parameter finite differences. It passed for that fixed graph, sequence, MSE objective, dtype, seed, and tolerance. It does not validate masked/multi-head attention, arbitrary sequence lengths, or training convergence.

## BatchNorm graph-integrated oracle (PR #49)

PR #49 merged after the dedicated BatchNorm graph-gradient workflow [38583056241](https://github.com/RayAKaan/Neurofluxion/actions/runs/38583056241), research validation [38583055895](https://github.com/RayAKaan/Neurofluxion/actions/runs/38583055895), frozen-main integrated validation [38583055940](https://github.com/RayAKaan/Neurofluxion/actions/runs/38583055940), and full-stack validation [38583055526](https://github.com/RayAKaan/Neurofluxion/actions/runs/38583055526) passed. The tested graph is Conv2D → BatchNorm → Flatten → Dense on one fixed spatial input. It does not establish arbitrary BatchNorm dimensions or running-stat behavior.
