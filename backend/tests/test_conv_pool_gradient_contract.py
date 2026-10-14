"""Independent forward/backward contracts for Conv2D and 2D pooling layers."""
import numpy as np

from simulator.layers.conv2d import Conv2DConfig, Conv2DLayer
from simulator.layers.pooling import AvgPool2DLayer, MaxPool2DLayer, PoolConfig


def test_conv2d_forward_and_parameter_gradients_match_finite_differences():
    layer = Conv2DLayer(Conv2DConfig(
        c_in=1, c_out=1, kernel_size=(2, 2), stride=1,
        padding="valid", activation="linear", init="xavier",
    ))
    layer.K = np.array([[[[0.2, -0.1], [0.3, 0.4]]]], dtype=np.float32)
    layer.b = np.array([0.05], dtype=np.float32)
    x = np.array([[[0.2, -0.7, 0.4], [1.0, 0.3, -0.2], [-0.5, 0.8, 0.6]]], dtype=np.float32)
    target = np.array([[[0.1, -0.2], [0.3, 0.4]]], dtype=np.float32)

    output = layer.forward(x)
    assert output.shape == (1, 2, 2)
    upstream = (output - target) / output.size
    layer.backward(upstream)
    analytic_kernel = layer.dK.copy()
    analytic_bias = layer.db.copy()

    def objective():
        prediction = layer.forward(x)
        return float(0.5 * np.mean((prediction.astype(np.float64) - target) ** 2))

    epsilon = 1e-3
    numeric_kernel = np.zeros_like(layer.K, dtype=np.float64)
    for coordinate in np.ndindex(layer.K.shape):
        original = float(layer.K[coordinate])
        try:
            layer.K[coordinate] = original + epsilon
            plus = objective()
            layer.K[coordinate] = original - epsilon
            minus = objective()
        finally:
            layer.K[coordinate] = original
        numeric_kernel[coordinate] = (plus - minus) / (2 * epsilon)

    numeric_bias = np.zeros_like(layer.b, dtype=np.float64)
    for coordinate in np.ndindex(layer.b.shape):
        original = float(layer.b[coordinate])
        try:
            layer.b[coordinate] = original + epsilon
            plus = objective()
            layer.b[coordinate] = original - epsilon
            minus = objective()
        finally:
            layer.b[coordinate] = original
        numeric_bias[coordinate] = (plus - minus) / (2 * epsilon)

    np.testing.assert_allclose(analytic_kernel, numeric_kernel, rtol=5e-3, atol=5e-4)
    np.testing.assert_allclose(analytic_bias, numeric_bias, rtol=5e-3, atol=5e-4)


def test_maxpool_forward_and_backward_route_gradient_to_unique_maximum():
    layer = MaxPool2DLayer(PoolConfig(pool_size=2, stride=2))
    x = np.array([[[1.0, 4.0], [2.0, 3.0]]], dtype=np.float32)
    output = layer.forward(x)
    np.testing.assert_allclose(output, [[[4.0]]])
    dx = layer.backward(np.ones_like(output))
    np.testing.assert_allclose(dx, [[[0.0, 1.0], [0.0, 0.0]]])


def test_avgpool_forward_and_backward_match_expected_average():
    layer = AvgPool2DLayer(PoolConfig(pool_size=2, stride=2))
    x = np.array([[[1.0, 3.0], [5.0, 7.0]]], dtype=np.float32)
    output = layer.forward(x)
    np.testing.assert_allclose(output, [[[4.0]]])
    dx = layer.backward(np.ones_like(output))
    np.testing.assert_allclose(dx, np.full((1, 2, 2), 0.25, dtype=np.float32))


def test_flatten_forward_and_backward_preserve_element_order_and_shape():
    from simulator.layers.flatten import FlattenLayer

    layer = FlattenLayer()
    x = np.arange(6, dtype=np.float32).reshape(1, 2, 3)
    flattened = layer.forward(x)
    np.testing.assert_array_equal(flattened, np.arange(6, dtype=np.float32))
    upstream = np.array([0.2, -0.1, 0.3, 0.4, -0.5, 0.6], dtype=np.float32)
    restored = layer.backward(upstream)
    assert restored.shape == x.shape
    np.testing.assert_array_equal(restored.reshape(-1), upstream)
