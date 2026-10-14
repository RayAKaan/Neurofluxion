"""Finite-difference contracts for BatchNormLayer's per-feature gradients."""
import numpy as np

from simulator.layers.batchnorm import BatchNormLayer


def _objective(layer, x):
    output = layer.forward(x)
    return float(0.5 * np.mean(output.astype(np.float64) ** 2))


def test_batchnorm_input_and_parameter_gradients_match_finite_differences():
    layer = BatchNormLayer(num_features=2, eps=1e-4)
    layer.gamma = np.array([1.3, 0.7], dtype=np.float32)
    layer.beta = np.array([0.1, -0.2], dtype=np.float32)
    x = np.array(
        [[0.2, -1.0], [1.1, 0.3], [-0.4, 1.4], [0.8, -0.2]],
        dtype=np.float32,
    )

    output = layer.forward(x)
    upstream = output / output.size
    analytic_x = layer.backward(upstream)
    analytic_gamma, analytic_beta = layer.grads()
    assert analytic_gamma.shape == layer.gamma.shape
    assert analytic_beta.shape == layer.beta.shape

    epsilon = 1e-3
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

    numeric_gamma = np.zeros_like(layer.gamma, dtype=np.float64)
    numeric_beta = np.zeros_like(layer.beta, dtype=np.float64)
    for parameter, numeric in ((layer.gamma, numeric_gamma), (layer.beta, numeric_beta)):
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

    np.testing.assert_allclose(analytic_x, numeric_x, rtol=5e-3, atol=5e-4)
    np.testing.assert_allclose(analytic_gamma, numeric_gamma, rtol=5e-3, atol=5e-4)
    np.testing.assert_allclose(analytic_beta, numeric_beta, rtol=5e-3, atol=5e-4)


def test_batchnorm_single_vector_has_well_defined_gradient_shapes():
    layer = BatchNormLayer(num_features=3)
    x = np.array([0.2, -0.5, 1.0], dtype=np.float32)
    output = layer.forward(x)
    dx = layer.backward(np.ones_like(output))
    dgamma, dbeta = layer.grads()
    assert dx.shape == x.shape
    assert dgamma.shape == layer.gamma.shape
    assert dbeta.shape == layer.beta.shape
    np.testing.assert_allclose(dx, np.zeros_like(x), atol=1e-7)
    np.testing.assert_allclose(dgamma, np.zeros_like(layer.gamma), atol=1e-7)
