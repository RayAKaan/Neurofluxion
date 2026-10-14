"""Regression tests for TrainingEngine's mean-reduced MSE gradient contract."""
import numpy as np

from simulator.graph_engine import build_graph
from simulator.layers import LayerConfig
from simulator.loss_functions import compute_loss
from simulator.training_engine import TrainingConfig, TrainingEngine


def _fixed_engine():
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
    for layer, weight, bias in zip(param_layers, weights, biases):
        layer.set_params(weight.copy(), bias.copy())
    graph.weights = [layer.params()[0] for layer in param_layers]
    graph.biases = [layer.params()[1] for layer in param_layers]
    config = TrainingConfig(loss_function="mse", dropout_rate=0.0, l2_lambda=0.0)
    return TrainingEngine(graph, config)


def test_training_backward_gradients_follow_forward_parameter_order():
    engine = _fixed_engine()
    x = np.array([0.4, -0.7], dtype=np.float32)
    target = np.array([0.2, -0.1], dtype=np.float32)
    activations, pre_acts, masks = engine._forward(x, training=False)
    gradients_w, gradients_b = engine._backward(activations, pre_acts, masks, target)

    assert len(gradients_w) == len(engine.graph.weights)
    assert len(gradients_b) == len(engine.graph.biases)
    for gradient, weight in zip(gradients_w, engine.graph.weights):
        assert gradient.shape == weight.shape
    for gradient, bias in zip(gradients_b, engine.graph.biases):
        assert gradient.shape == bias.shape


def test_training_mse_gradients_match_central_finite_differences():
    engine = _fixed_engine()
    graph = engine.graph
    x = np.array([0.4, -0.7], dtype=np.float32)
    target = np.array([0.2, -0.1], dtype=np.float32)
    activations, pre_acts, masks = engine._forward(x, training=False)
    analytic_w, analytic_b = engine._backward(activations, pre_acts, masks, target)

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
