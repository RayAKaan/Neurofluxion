# Conv2D and BatchNorm Graph Training Contract

## Scope

This test exercises one TrainingEngine update through Conv2D → BatchNorm → Flatten → sigmoid-Dense. It complements the graph-integrated BatchNorm finite-difference oracle by checking optimizer-path integration and parameter mutation.

## Protocol

- Fixed 1×3×3 input and binary target.
- One 2×2 convolution filter, one-channel BatchNorm on the spatial feature map, flatten, and one sigmoid output.
- Binary cross-entropy, SGD, learning rate 0.01, dropout disabled, fixed seed.
- Assert finite loss/gradients, nonzero gradient norm, initialized optimizer state, and actual updates to convolution, BatchNorm, and output-Dense parameters.

## Evidence boundary

A pass establishes only this one-batch configuration. It does not establish gradient correctness (covered separately by finite differences), loss reduction, convergence, running-stat behavior at inference, arbitrary spatial sizes/channels, or general Conv2D/BatchNorm training. Runtime source is unchanged; any mismatch must be triaged separately.
