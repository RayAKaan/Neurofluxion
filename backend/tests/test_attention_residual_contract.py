"""Finite-difference contract for single-head self-attention and residual identity."""
import numpy as np

from simulator.layers.attention import AttentionLayer
from simulator.layers.residual import ResidualLayer


def _attention_objective(layer, x, target):
    output = layer.forward(x)
    return float(0.5 * np.mean((output.astype(np.float64) - target) ** 2))


def test_single_head_attention_parameter_gradients_match_finite_differences():
    layer = AttentionLayer(d_model=2, num_heads=1)
    layer.W_q = np.array([[0.2, -0.1], [0.05, 0.3]], dtype=np.float32)
    layer.W_k = np.array([[-0.2, 0.15], [0.25, 0.1]], dtype=np.float32)
    layer.W_v = np.array([[0.3, 0.05], [-0.1, 0.2]], dtype=np.float32)
    layer.W_o = np.array([[0.15, -0.25], [0.2, 0.1]], dtype=np.float32)
    x = np.array([[0.4, -0.7], [0.2, 0.5]], dtype=np.float32)
    target = np.array([[0.1, 0.3], [-0.2, 0.4]], dtype=np.float32)

    output = layer.forward(x)
    dx = layer.backward((output - target) / output.size)
    analytic = [np.asarray(layer.dW[index]) for index in range(4)]
    parameters = [layer.W_q, layer.W_k, layer.W_v, layer.W_o]
    epsilon = 1e-3
    numeric = [np.zeros_like(parameter, dtype=np.float64) for parameter in parameters]

    for parameter, expected in zip(parameters, numeric):
        for coordinate in np.ndindex(parameter.shape):
            original = float(parameter[coordinate])
            try:
                parameter[coordinate] = original + epsilon
                plus = _attention_objective(layer, x, target)
                parameter[coordinate] = original - epsilon
                minus = _attention_objective(layer, x, target)
            finally:
                parameter[coordinate] = original
            expected[coordinate] = (plus - minus) / (2 * epsilon)

    for actual, expected in zip(analytic, numeric):
        np.testing.assert_allclose(actual, expected, rtol=8e-3, atol=8e-4)
    assert dx.shape == x.shape
    assert np.isfinite(dx).all()


def test_residual_layer_preserves_forward_and_backward_values():
    layer = ResidualLayer()
    x = np.array([[1.0, -2.0], [0.5, 0.25]], dtype=np.float32)
    upstream = np.array([[0.2, 0.3], [-0.4, 0.1]], dtype=np.float32)
    np.testing.assert_array_equal(layer.forward(x), x)
    np.testing.assert_array_equal(layer.backward(upstream), upstream)
