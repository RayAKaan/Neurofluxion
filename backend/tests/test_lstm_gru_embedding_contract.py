"""Finite-difference and accumulation contracts for LSTM, GRU, and Embedding."""
import numpy as np

from simulator.layers.embedding import EmbeddingLayer
from simulator.layers.gru import GRULayer
from simulator.layers.lstm import LSTMLayer


def _objective(layer, x):
    output = layer.forward(x)
    return float(0.5 * np.mean(output.astype(np.float64) ** 2))


def _check_recurrent_gradients(layer, x, weight_names, bias_names, rtol=1.2e-2, atol=1.2e-3):
    output = layer.forward(x)
    dx = layer.backward(output / output.size)
    epsilon = 1e-3
    numeric_weights = []
    for name in weight_names:
        parameter = getattr(layer, name)
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
        numeric_weights.append(numeric)

    numeric_biases = []
    for name in bias_names:
        parameter = getattr(layer, name)
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
        numeric_biases.append(numeric)

    for name, expected in zip(weight_names, numeric_weights):
        actual = getattr(layer, "dW")
        # Packed gradient rows follow the same gate order as params().
        gate_index = weight_names.index(name)
        hidden = layer.hidden_dim
        actual_gate = actual[gate_index * hidden : (gate_index + 1) * hidden]
        np.testing.assert_allclose(actual_gate, expected, rtol=rtol, atol=atol)
    for gate_index, expected in enumerate(numeric_biases):
        hidden = layer.hidden_dim
        actual_gate = layer.db[gate_index * hidden : (gate_index + 1) * hidden]
        np.testing.assert_allclose(actual_gate, expected, rtol=rtol, atol=atol)

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
    np.testing.assert_allclose(dx, numeric_x, rtol=rtol, atol=atol)


def test_lstm_gate_parameter_and_input_gradients_match_finite_differences():
    layer = LSTMLayer(input_dim=2, hidden_dim=2, return_sequences=False)
    layer.W_f = np.array([[0.1, -0.2, 0.05, 0.03], [-0.04, 0.12, 0.08, -0.06]], dtype=np.float32)
    layer.W_i = np.array([[0.07, 0.02, -0.1, 0.05], [0.11, -0.03, 0.06, 0.09]], dtype=np.float32)
    layer.W_o = np.array([[-0.02, 0.08, 0.1, -0.04], [0.05, 0.07, -0.03, 0.02]], dtype=np.float32)
    layer.W_c = np.array([[0.04, -0.06, 0.12, 0.03], [-0.08, 0.1, 0.02, 0.07]], dtype=np.float32)
    layer.b_f = np.array([0.01, -0.02], dtype=np.float32)
    layer.b_i = np.array([0.02, 0.01], dtype=np.float32)
    layer.b_o = np.array([-0.01, 0.03], dtype=np.float32)
    layer.b_c = np.array([0.0, -0.02], dtype=np.float32)
    x = np.array([[0.2, -0.3], [0.5, 0.1]], dtype=np.float32)
    _check_recurrent_gradients(
        layer, x,
        ["W_f", "W_i", "W_o", "W_c"],
        ["b_f", "b_i", "b_o", "b_c"],
    )


def test_gru_gate_parameter_and_input_gradients_match_finite_differences():
    layer = GRULayer(input_dim=2, hidden_dim=2, return_sequences=False)
    layer.W_r = np.array([[0.1, -0.2, 0.05, 0.03], [-0.04, 0.12, 0.08, -0.06]], dtype=np.float32)
    layer.W_z = np.array([[0.07, 0.02, -0.1, 0.05], [0.11, -0.03, 0.06, 0.09]], dtype=np.float32)
    layer.W_h = np.array([[-0.02, 0.08, 0.1, -0.04], [0.05, 0.07, -0.03, 0.02]], dtype=np.float32)
    layer.b_r = np.array([0.01, -0.02], dtype=np.float32)
    layer.b_z = np.array([0.02, 0.01], dtype=np.float32)
    layer.b_h = np.array([-0.01, 0.03], dtype=np.float32)
    x = np.array([[0.2, -0.3], [0.5, 0.1]], dtype=np.float32)
    _check_recurrent_gradients(
        layer, x,
        ["W_r", "W_z", "W_h"],
        ["b_r", "b_z", "b_h"],
    )


def test_embedding_accumulates_gradients_for_repeated_indices():
    layer = EmbeddingLayer(vocab_size=4, embedding_dim=2)
    layer.E = np.arange(8, dtype=np.float32).reshape(4, 2) / 10
    indices = np.array([1, 2, 1], dtype=np.int64)
    output = layer.forward(indices)
    np.testing.assert_array_equal(output, layer.E[[1, 2, 1]])
    upstream = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    dx = layer.backward(upstream)
    expected = np.zeros_like(layer.E)
    expected[1] = [6.0, 8.0]
    expected[2] = [3.0, 4.0]
    np.testing.assert_array_equal(layer.dE, expected)
    np.testing.assert_array_equal(dx, np.zeros_like(indices, dtype=np.float32))
