"""Deterministic training and optimizer contracts for the inspectable dense path."""
import numpy as np

from simulator.graph_engine import build_graph
from simulator.layers import LayerConfig
from simulator.loss_functions import compute_loss
from simulator.optimizer_engine import apply_update
from simulator.training_engine import TrainingConfig, TrainingEngine


def _binary_engine():
    graph = build_graph([
        LayerConfig(layer_type="input", neurons=1, input_shape=[1]),
        LayerConfig(layer_type="output", neurons=1, activation="sigmoid"),
    ])
    dense = next(layer for layer in graph.layer_instances if getattr(layer, "has_params", False))
    dense.set_params(
        np.array([[0.2]], dtype=np.float32),
        np.array([0.0], dtype=np.float32),
    )
    graph.weights = [dense.params()[0]]
    graph.biases = [dense.params()[1]]
    config = TrainingConfig(
        epochs=1,
        batch_size=2,
        learning_rate=0.2,
        optimizer="sgd",
        loss_function="bce",
        shuffle=False,
        dropout_rate=0.0,
        l2_lambda=0.0,
        snapshot_interval=0,
    )
    return TrainingEngine(graph, config)


def _dataset():
    return [
        (np.array([-1.0], dtype=np.float32), np.array([0.0], dtype=np.float32)),
        (np.array([1.0], dtype=np.float32), np.array([1.0], dtype=np.float32)),
    ]


def _loss(engine):
    values = []
    for x, y in _dataset():
        prediction = engine.graph.forward(x).reshape(-1)
        values.append(compute_loss(y, prediction, "bce")[0])
    return float(np.mean(values))


def test_sgd_training_reduces_loss_on_tiny_separable_dataset():
    engine = _binary_engine()
    before = _loss(engine)
    data = _dataset()
    for _ in range(80):
        engine.train_epoch(data, [], graph_id="training-contract")
    after = _loss(engine)
    assert np.isfinite(before) and np.isfinite(after)
    assert after < before * 0.75, (before, after)


def test_fixed_seed_and_order_produce_repeatable_training_trajectory():
    first = _binary_engine()
    second = _binary_engine()
    data_a = _dataset()
    data_b = _dataset()
    losses_a = []
    losses_b = []
    for _ in range(12):
        first.train_epoch(data_a, [], graph_id="repeatability-a")
        second.train_epoch(data_b, [], graph_id="repeatability-b")
        losses_a.append(_loss(first))
        losses_b.append(_loss(second))
    np.testing.assert_allclose(losses_a, losses_b, rtol=0.0, atol=1e-7)
    for a, b in zip(first.graph.weights, second.graph.weights):
        np.testing.assert_allclose(a, b, rtol=0.0, atol=1e-7)
    for a, b in zip(first.graph.biases, second.graph.biases):
        np.testing.assert_allclose(a, b, rtol=0.0, atol=1e-7)


def test_sgd_optimizer_applies_expected_parameter_update():
    weights = [np.array([[1.0, -2.0]], dtype=np.float32)]
    biases = [np.array([0.5], dtype=np.float32)]
    grad_w = [np.array([[0.25, -0.5]], dtype=np.float32)]
    grad_b = [np.array([0.75], dtype=np.float32)]
    new_w, new_b, state = apply_update(
        weights, biases, grad_w, grad_b, lr=0.1, optimizer="sgd", state=None
    )
    np.testing.assert_allclose(new_w[0], [[0.975, -1.95]], rtol=0, atol=1e-7)
    np.testing.assert_allclose(new_b[0], [0.425], rtol=0, atol=1e-7)
    assert state.t == 0


def test_adam_optimizer_updates_finitely_and_advances_step():
    weights = [np.array([[1.0, -2.0]], dtype=np.float32)]
    biases = [np.array([0.5], dtype=np.float32)]
    grad_w = [np.array([[0.25, -0.5]], dtype=np.float32)]
    grad_b = [np.array([0.75], dtype=np.float32)]
    new_w, new_b, state = apply_update(
        weights, biases, grad_w, grad_b, lr=0.01, optimizer="adam", state=None
    )
    assert state.t == 1
    assert all(np.isfinite(x).all() for x in [*new_w, *new_b])
    assert not np.array_equal(new_w[0], weights[0])
    assert not np.array_equal(new_b[0], biases[0])
