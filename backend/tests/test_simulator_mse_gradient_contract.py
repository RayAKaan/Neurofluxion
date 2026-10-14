"""Regression tests for the dense simulator's MSE gradient contract."""
import numpy as np

from simulator.backward_engine import backward_full
from simulator.graph_engine import build_graph
from simulator.layers import LayerConfig
from simulator.loss_functions import compute_loss


def _fixed_graph():
    layers = [
        LayerConfig(layer_type="input", neurons=2, input_shape=[2]),
        LayerConfig(layer_type="dense", neurons=3, activation="tanh"),
        LayerConfig(layer_type="output", neurons=2, activation="linear"),
    ]
    graph = build_graph(layers)
    weights = [
        np.array([[0.12, -0.31], [0.27, 0.18], [-0.22, 0.41]], dtype=np.float32),
        np.array([[0.33, -0.17, 0.26], [-0.11, 0.29, 0.08]], dtype=np.float32),
    ]
    biases = [
        np.array([0.03, -0.04, 0.02], dtype=np.float32),
        np.array([0.01, -0.02], dtype=np.float32),
    ]
    param_layers = [layer for layer in graph.layer_instances if getattr(layer, "has_params", False)]
    assert len(param_layers) == len(weights)
    for layer, weight, bias in zip(param_layers, weights, biases):
        layer.set_params(weight.copy(), bias.copy())
    graph.weights = [layer.params()[0] for layer in param_layers]
    graph.biases = [layer.params()[1] for layer in param_layers]
    return graph


def test_backward_gradients_follow_forward_parameter_order():
    graph = _fixed_graph()
    result = backward_full(graph, [0.4, -0.7], [0.2, -0.1], "mse")

    assert len(result["gradients_W"]) == len(graph.weights)
    assert len(result["gradients_b"]) == len(graph.biases)
    for index, weight in enumerate(graph.weights):
        assert np.asarray(result["gradients_W"][index]).shape == weight.shape
    for index, bias in enumerate(graph.biases):
        assert np.asarray(result["gradients_b"][index]).shape == bias.shape


def test_mean_reduced_mse_gradients_match_central_finite_differences():
    graph = _fixed_graph()
    x = np.array([0.4, -0.7], dtype=np.float32)
    target = np.array([0.2, -0.1], dtype=np.float32)
    result = backward_full(graph, x.tolist(), target.tolist(), "mse")
    analytic_w = [np.asarray(value, dtype=np.float64) for value in result["gradients_W"]]
    analytic_b = [np.asarray(value, dtype=np.float64) for value in result["gradients_b"]]

    def scalar_loss():
        prediction = np.asarray(graph.forward(x), dtype=np.float32).reshape(-1)
        return float(compute_loss(target, prediction, "mse")[0])

    epsilon = 1e-3
    numeric_w = [np.zeros_like(weight, dtype=np.float64) for weight in graph.weights]
    numeric_b = [np.zeros_like(bias, dtype=np.float64) for bias in graph.biases]
    for parameters, numeric in ((graph.weights, numeric_w), (graph.biases, numeric_b)):
        for layer_index, parameter in enumerate(parameters):
            for coordinate in np.ndindex(parameter.shape):
                original = float(parameter[coordinate])
                try:
                    parameter[coordinate] = original + epsilon
                    plus = scalar_loss()
                    parameter[coordinate] = original - epsilon
                    minus = scalar_loss()
                finally:
                    parameter[coordinate] = original
                numeric[layer_index][coordinate] = (plus - minus) / (2 * epsilon)

    for analytic, numeric in zip(analytic_w, numeric_w):
        np.testing.assert_allclose(analytic, numeric, rtol=2e-3, atol=2e-4)
    for analytic, numeric in zip(analytic_b, numeric_b):
        np.testing.assert_allclose(analytic, numeric, rtol=2e-3, atol=2e-4)
