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
