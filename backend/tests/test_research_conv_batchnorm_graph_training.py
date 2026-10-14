"""Graph-integrated training update contract for Conv2D→BatchNorm→Flatten→Dense."""
from __future__ import annotations

import numpy as np

from simulator.graph_engine import build_graph
from simulator.layers import LayerConfig
from simulator.training_engine import TrainingConfig, TrainingEngine


def test_conv_batchnorm_graph_training_updates_all_parameterized_families():
    np.random.seed(439)
    graph = build_graph([
        LayerConfig("input", neurons=9, input_shape=[1, 3, 3]),
        LayerConfig("conv2d", neurons=1, filters=1, kernel_size=2, padding="valid", activation="linear"),
        LayerConfig("batchnorm", neurons=1),
        LayerConfig("flatten", neurons=4),
        LayerConfig("output", neurons=1, activation="sigmoid"),
    ])
    engine = TrainingEngine(
        graph,
        TrainingConfig(
            epochs=1, batch_size=1, learning_rate=0.01, optimizer="sgd",
            loss_function="bce", shuffle=False, dropout_rate=0.0,
        ),
    )
    x = np.asarray([[[0.1, -0.2, 0.4], [0.3, 0.7, -0.5], [-0.6, 0.2, 0.8]]], dtype=np.float32)
    y = np.asarray([1.0], dtype=np.float32)
    before_weights = [value.copy() for value in graph.weights]
    before_biases = [value.copy() for value in graph.biases]

    loss, accuracy, gradient_norm = engine.train_batch([(x.copy(), y.copy())])

    assert np.isfinite(loss)
    assert np.isfinite(accuracy)
    assert np.isfinite(gradient_norm) and gradient_norm > 0.0
    assert all(np.isfinite(value).all() for value in engine.last_gradients["dW"])
    assert all(np.isfinite(value).all() for value in engine.last_gradients["db"])
    assert engine.optimizer_state is not None

    # Parameter tensors are ordered by parameterized layer: Conv2D, BatchNorm, Dense.
    assert len(graph.weights) == 3
    assert len(graph.biases) == 3
    assert not np.array_equal(before_weights[0], graph.weights[0]), "Conv2D kernel did not update"
    assert not np.array_equal(before_weights[1], graph.weights[1]) or not np.array_equal(
        before_biases[1], graph.biases[1]
    ), "BatchNorm gamma/beta did not update"
    assert not np.array_equal(before_weights[2], graph.weights[2]), "Dense output weights did not update"
