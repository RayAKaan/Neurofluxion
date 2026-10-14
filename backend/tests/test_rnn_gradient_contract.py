"""Finite-difference contracts for the simple recurrent layer."""
import numpy as np

from simulator.layers.rnn import RNNLayer


def _objective(layer, x):
    output = layer.forward(x)
    return float(0.5 * np.mean(output.astype(np.float64) ** 2))


def test_rnn_parameter_and_input_gradients_match_finite_differences():
    layer = RNNLayer(input_dim=2, hidden_dim=2, activation="tanh", return_sequences=False)
    layer.W_xh = np.array([[0.2, -0.1], [0.05, 0.3]], dtype=np.float32)
    layer.W_hh = np.array([[0.1, 0.04], [-0.08, 0.12]], dtype=np.float32)
    layer.b = np.array([0.02, -0.03], dtype=np.float32)
    x = np.array([[0.4, -0.7], [0.2, 0.5], [-0.3, 0.6]], dtype=np.float32)

    output = layer.forward(x)
    dx = layer.backward(output / output.size)
    analytic_wxh = layer.dW_xh.copy()
    analytic_whh = layer.dW_hh.copy()
    analytic_b = layer.db.copy()
    epsilon = 1e-3

    numeric_parameters = []
    for parameter in (layer.W_xh, layer.W_hh, layer.b):
        numeric = np.zeros_like(parameter, dtype=np.float64)
        for coordinate in np.ndindex(parameter.shape):
            original = float(parameter[coordinate])
            try:
                parameter[coordinate] = original + epsilon
                plus = _objective(layer, x)
                parameter[coordinate] = original - epsilon
                minus = _objective(layer, x)
            finally:
                parameter[coordinate] = original
            numeric[coordinate] = (plus - minus) / (2 * epsilon)
        numeric_parameters.append(numeric)

    numeric_x = np.zeros_like(x, dtype=np.float64)
    for coordinate in np.ndindex(x.shape):
        original = float(x[coordinate])
        try:
            x[coordinate] = original + epsilon
            plus = _objective(layer, x)
            x[coordinate] = original - epsilon
            minus = _objective(layer, x)
        finally:
            x[coordinate] = original
        numeric_x[coordinate] = (plus - minus) / (2 * epsilon)

    np.testing.assert_allclose(analytic_wxh, numeric_parameters[0], rtol=8e-3, atol=8e-4)
    np.testing.assert_allclose(analytic_whh, numeric_parameters[1], rtol=8e-3, atol=8e-4)
    np.testing.assert_allclose(analytic_b, numeric_parameters[2], rtol=8e-3, atol=8e-4)
    np.testing.assert_allclose(dx, numeric_x, rtol=8e-3, atol=8e-4)
