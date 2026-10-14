"""Independent finite-difference oracles for graph-integrated backward paths."""
from __future__ import annotations

import numpy as np
import pytest

from simulator.backward_engine import backward_full
from simulator.graph_engine import build_graph
from simulator.layers import LayerConfig
from simulator.loss_functions import compute_loss


def _sync_params(graph):
    # Some recurrent layers expose packed optimizer parameters rather than the
    # underlying matrices directly. Synchronize every graph parameter tensor
    # into its owning layer before each forward evaluation.
    for layer_index, parameter_index in graph.param_index_by_layer.items():
        graph.layer_instances[layer_index].set_params(
            graph.weights[parameter_index], graph.biases[parameter_index]
        )


def _objective(graph, x, target):
    _sync_params(graph)
    prediction = np.asarray(graph.forward(x.copy()), dtype=np.float64).reshape(-1)
    loss, _ = compute_loss(target.astype(np.float32), prediction.astype(np.float32), "mse")
    return float(loss)


def _assert_graph_gradients_match_finite_difference(graph, x, target, epsilon=1e-3):
    _sync_params(graph)
    result = backward_full(graph, x.tolist(), target.tolist(), "mse")
    analytic_w = [np.asarray(value, dtype=np.float64) for value in result["gradients_W"]]
    analytic_b = [np.asarray(value, dtype=np.float64) for value in result["gradients_b"]]
    assert len(analytic_w) == len(graph.weights)
    assert len(analytic_b) == len(graph.biases)

    numeric_w = [np.zeros_like(value, dtype=np.float64) for value in graph.weights]
    numeric_b = [np.zeros_like(value, dtype=np.float64) for value in graph.biases]
    for params, numeric in ((graph.weights, numeric_w), (graph.biases, numeric_b)):
        for tensor_index, tensor in enumerate(params):
            for index in np.ndindex(tensor.shape):
                original = float(tensor[index])
                tensor[index] = original + epsilon
                plus = _objective(graph, x, target)
                tensor[index] = original - epsilon
                minus = _objective(graph, x, target)
                tensor[index] = original
                numeric[tensor_index][index] = (plus - minus) / (2.0 * epsilon)

    for i, (analytic, numeric) in enumerate(zip(analytic_w, numeric_w)):
        np.testing.assert_allclose(analytic, numeric, rtol=2e-2, atol=2e-3, err_msg=f"dW parameter tensor {i}")
    for i, (analytic, numeric) in enumerate(zip(analytic_b, numeric_b)):
        np.testing.assert_allclose(analytic, numeric, rtol=2e-2, atol=2e-3, err_msg=f"db parameter tensor {i}")


def test_conv_flatten_dense_graph_backward_matches_finite_differences():
    np.random.seed(431)
    graph = build_graph([
        LayerConfig("input", neurons=9, input_shape=[1, 3, 3]),
        LayerConfig("conv2d", neurons=1, filters=1, kernel_size=2, padding="valid", activation="linear"),
        LayerConfig("flatten", neurons=4),
        LayerConfig("output", neurons=1, activation="linear"),
    ])
    x = np.asarray([[[0.1, -0.2, 0.4], [0.3, 0.7, -0.5], [-0.6, 0.2, 0.8]]], dtype=np.float32)
    target = np.asarray([0.25], dtype=np.float32)
    _assert_graph_gradients_match_finite_difference(graph, x, target)


def test_rnn_dense_graph_backward_matches_finite_differences():
    np.random.seed(432)
    graph = build_graph([
        LayerConfig("input", neurons=6, input_shape=[3, 2]),
        LayerConfig("rnn", neurons=2, hidden_size=2, return_sequences=False),
        LayerConfig("output", neurons=1, activation="linear"),
    ])
    x = np.asarray([[0.1, 0.2], [-0.3, 0.4], [0.5, -0.2]], dtype=np.float32)
    target = np.asarray([-0.15], dtype=np.float32)
    _assert_graph_gradients_match_finite_difference(graph, x, target)


@pytest.mark.parametrize("loss", ["mse"])
def test_oracle_loss_contract_is_finite(loss):
    prediction = np.asarray([0.3, -0.2], dtype=np.float32)
    target = np.asarray([0.1, 0.4], dtype=np.float32)
    value, _ = compute_loss(target, prediction, loss)
    assert np.isfinite(value)
