# BatchNorm Graph-Integrated Gradient Oracle

## Scope

This test compares analytic graph-level parameter gradients against central finite differences for one Conv2D → BatchNorm → Flatten → Dense graph. BatchNorm operates on a spatial feature map, avoiding the degenerate single-vector case where per-feature variance is zero.

## Protocol

- Fixed 1×3×3 input, one 2×2 convolution filter, one-channel BatchNorm, flatten, and one linear output.
- Mean-reduced MSE objective, fixed seed, central difference epsilon 1e-3, rtol 2e-2, atol 2e-3.
- Packed graph parameters are synchronized into their owning layer instances before each objective evaluation.

## Evidence boundary

A passing check supports this fixed graph, input, initialization, loss, dtype, and tolerance only. It does not establish arbitrary BatchNorm dimensions, training/evaluation running-stat semantics, broader convolution settings, or training convergence. Runtime source is unchanged; any numerical mismatch must be triaged without silently loosening tolerances.
