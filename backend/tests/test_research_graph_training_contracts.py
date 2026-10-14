"""Deterministic integrated training contracts for the dense NetworkGraph path."""
from __future__ import annotations

import numpy as np

from simulator.graph_engine import build_graph
from simulator.layers import LayerConfig
from simulator.loss_functions import compute_loss
from simulator.training_engine import TrainingConfig, TrainingEngine


def _make_graph():
    np.random.seed(20261014)
    return build_graph([
        LayerConfig("input", neurons=2, input_shape=[2]),
        LayerConfig("dense", neurons=3, activation="tanh"),
        LayerConfig("output", neurons=1, activation="sigmoid"),
    ])


def test_dense_graph_training_updates_parameters_and_reduces_fixed_sample_loss():
    graph = _make_graph()
    config = TrainingConfig(
        epochs=1,
        batch_size=1,
        learning_rate=0.05,
        optimizer="sgd",
        loss_function="bce",
        shuffle=False,
        dropout_rate=0.0,
    )
    engine = TrainingEngine(graph, config)
    x = np.asarray([0.2, -0.4], dtype=np.float32)
    y = np.asarray([1.0], dtype=np.float32)

    initial_output = np.asarray(graph.forward(x.copy()), dtype=np.float32).reshape(-1)
    initial_loss, _ = compute_loss(y, initial_output, "bce")
    initial_weights = [w.copy() for w in graph.weights]

    for _ in range(40):
        loss, _, grad_norm = engine.train_batch([(x.copy(), y.copy())])
        assert np.isfinite(loss)
        assert np.isfinite(grad_norm)
        assert grad_norm > 0.0

    final_output = np.asarray(graph.forward(x.copy()), dtype=np.float32).reshape(-1)
    final_loss, _ = compute_loss(y, final_output, "bce")

    assert np.isfinite(final_loss)
    assert final_loss < initial_loss, (initial_loss, final_loss)
    assert any(not np.array_equal(before, after) for before, after in zip(initial_weights, graph.weights))
    assert all(np.all(np.isfinite(g)) for g in engine.last_gradients["dW"])
    assert all(np.all(np.isfinite(g)) for g in engine.last_gradients["db"])
    assert engine.optimizer_state is not None


def test_dense_graph_training_is_repeatable_with_fixed_seed_and_no_dropout():
    def run_once():
        graph = _make_graph()
        engine = TrainingEngine(
            graph,
            TrainingConfig(
                epochs=1, batch_size=1, learning_rate=0.05, optimizer="sgd",
                loss_function="bce", shuffle=False, dropout_rate=0.0,
            ),
        )
        x = np.asarray([0.2, -0.4], dtype=np.float32)
        y = np.asarray([1.0], dtype=np.float32)
        for _ in range(10):
            engine.train_batch([(x.copy(), y.copy())])
        return [w.copy() for w in graph.weights], [b.copy() for b in graph.biases]

    weights_a, biases_a = run_once()
    weights_b, biases_b = run_once()
    for left, right in zip(weights_a + biases_a, weights_b + biases_b):
        np.testing.assert_array_equal(left, right)
