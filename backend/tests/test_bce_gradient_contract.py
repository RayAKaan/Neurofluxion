"""Finite-difference contracts for mean-reduced BCE with sigmoid outputs."""
import numpy as np
from simulator.backward_engine import backward_full
from simulator.graph_engine import build_graph
from simulator.layers import LayerConfig
from simulator.loss_functions import compute_loss
from simulator.training_engine import TrainingConfig, TrainingEngine


def _fixed_graph():
    graph = build_graph(
        [
            LayerConfig(layer_type="input", neurons=2, input_shape=[2]),
            LayerConfig(layer_type="dense", neurons=3, activation="tanh"),
            LayerConfig(layer_type="output", neurons=2, activation="sigmoid"),
        ]
    )
    weights = [
        np.array([[0.2, -0.1], [-0.3, 0.25], [0.15, 0.4]], dtype=np.float32),
        np.array([[0.3, -0.2, 0.1], [-0.1, 0.25, 0.35]], dtype=np.float32),
    ]
    biases = [
        np.array([0.02, -0.04, 0.03], dtype=np.float32),
        np.array([0.05, -0.03], dtype=np.float32),
    ]
    param_layers = [layer for layer in graph.layer_instances if getattr(layer, "has_params", False)]
    assert len(param_layers) == 2
    for layer, weight, bias in zip(param_layers, weights, biases):
        layer.set_params(weight.copy(), bias.copy())
    graph.weights = [layer.params()[0] for layer in param_layers]
    graph.biases = [layer.params()[1] for layer in param_layers]
    return graph


def _numeric_gradients(graph, x, target):
    epsilon = 1e-3
    numeric_w = [np.zeros_like(w, dtype=np.float64) for w in graph.weights]
    numeric_b = [np.zeros_like(b, dtype=np.float64) for b in graph.biases]

    def scalar_loss():
        prediction = np.asarray(graph.forward(x), dtype=np.float32).reshape(-1)
        return float(compute_loss(target, prediction, "bce")[0])

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
    return numeric_w, numeric_b


def _assert_gradients(analytic_w, analytic_b, numeric_w, numeric_b):
    for analytic, numeric in zip(analytic_w, numeric_w):
        np.testing.assert_allclose(analytic, numeric, rtol=3e-3, atol=3e-4)
    for analytic, numeric in zip(analytic_b, numeric_b):
        np.testing.assert_allclose(analytic, numeric, rtol=3e-3, atol=3e-4)


def test_simulator_mean_bce_sigmoid_gradients_match_finite_differences():
    graph = _fixed_graph()
    x = np.array([0.4, -0.7], dtype=np.float32)
    target = np.array([0.2, 0.8], dtype=np.float32)
    result = backward_full(graph, x.tolist(), target.tolist(), "bce")
    numeric_w, numeric_b = _numeric_gradients(graph, x, target)
    _assert_gradients(result["gradients_W"], result["gradients_b"], numeric_w, numeric_b)


def test_training_engine_mean_bce_sigmoid_gradients_match_finite_differences():
    graph = _fixed_graph()
    engine = TrainingEngine(
        graph,
        TrainingConfig(loss_function="bce", dropout_rate=0.0, l2_lambda=0.0),
    )
    x = np.array([0.4, -0.7], dtype=np.float32)
    target = np.array([0.2, 0.8], dtype=np.float32)
    activations, pre_acts, masks = engine._forward(x, training=False)
    analytic_w, analytic_b = engine._backward(activations, pre_acts, masks, target)
    numeric_w, numeric_b = _numeric_gradients(graph, x, target)
    _assert_gradients(analytic_w, analytic_b, numeric_w, numeric_b)
