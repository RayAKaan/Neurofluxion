"""Numerical contracts for the standalone /api/execute engine."""
import numpy as np
import pytest

from simulator.execution_engine import ExecutionEngine


def _engine():
    engine = ExecutionEngine(device="cpu")
    engine.build_graph([
        {"type": "input", "neurons": 2},
        {"type": "dense", "neurons": 3, "activation": "tanh"},
        {"type": "output", "neurons": 2, "activation": "linear"},
    ])
    engine._graph["weights"] = [
        np.array([[0.12, -0.31, 0.27], [0.18, -0.22, 0.41]], dtype=np.float32),
        np.array([[0.33, -0.17], [0.26, -0.11], [0.29, 0.08]], dtype=np.float32),
    ]
    engine._graph["biases"] = [
        np.array([0.03, -0.04, 0.02], dtype=np.float32),
        np.array([0.01, -0.02], dtype=np.float32),
    ]
    return engine


def _scalar_mse(engine, x, target):
    prediction = np.asarray(engine.forward(x).final_output).reshape(-1)
    return float(0.5 * np.mean((prediction - target) ** 2))


def test_forward_uses_chained_layer_dimensions_and_returns_finite_output():
    engine = _engine()
    result = engine.forward([0.4, -0.7])
    assert result.final_output.shape == (2,)
    assert np.isfinite(result.final_output).all()
    assert engine._graph["weights"][0].shape == (2, 3)
    assert engine._graph["weights"][1].shape == (3, 2)


def test_backward_gradients_match_central_finite_differences():
    engine = _engine()
    x = np.array([0.4, -0.7], dtype=np.float32)
    target = np.array([0.2, -0.1], dtype=np.float32)
    engine.forward(x.tolist())
    result = engine.backward(target.tolist(), loss_function="mse")
    analytic_w = [np.asarray(layer.gradient_weights) for layer in result.layers]
    analytic_b = [np.asarray(layer.gradient_bias) for layer in result.layers]
    epsilon = 1e-3
    numeric_w = [np.zeros_like(w, dtype=np.float64) for w in engine._graph["weights"]]
    numeric_b = [np.zeros_like(b, dtype=np.float64) for b in engine._graph["biases"]]

    for parameters, numeric in ((engine._graph["weights"], numeric_w), (engine._graph["biases"], numeric_b)):
        for layer_idx, parameter in enumerate(parameters):
            for coordinate in np.ndindex(parameter.shape):
                original = float(parameter[coordinate])
                try:
                    parameter[coordinate] = original + epsilon
                    plus = _scalar_mse(engine, x, target)
                    parameter[coordinate] = original - epsilon
                    minus = _scalar_mse(engine, x, target)
                finally:
                    parameter[coordinate] = original
                numeric[layer_idx][coordinate] = (plus - minus) / (2 * epsilon)

    for actual, expected in zip(analytic_w, numeric_w):
        np.testing.assert_allclose(actual, expected, rtol=3e-3, atol=3e-4)
    for actual, expected in zip(analytic_b, numeric_b):
        np.testing.assert_allclose(actual, expected, rtol=3e-3, atol=3e-4)
    assert np.isfinite(result.loss_value)
    assert result.total_gradient_norm > 0


def test_backward_requires_a_fresh_forward_and_matching_target_shape():
    engine = _engine()
    with pytest.raises(ValueError, match="Run forward"):
        engine.backward([0.0, 1.0])
    engine.forward([0.1, 0.2])
    with pytest.raises(ValueError, match="Target shape"):
        engine.backward([1.0])


def test_training_uses_real_bce_loss_and_reduces_loss():
    engine = ExecutionEngine(device="cpu")
    engine.build_graph([
        {"type": "input", "neurons": 1},
        {"type": "output", "neurons": 1, "activation": "sigmoid"},
    ])
    engine._graph["weights"] = [np.array([[0.2]], dtype=np.float32)]
    engine._graph["biases"] = [np.array([0.0], dtype=np.float32)]
    data = {
        "x_train": [[-1.0], [1.0]],
        "y_train": [[0.0], [1.0]],
    }
    result = engine.train(data, epochs=80, learning_rate=0.2, loss_function="bce")
    assert len(result.loss_history) == 80
    assert np.isfinite(result.loss_history).all()
    assert result.loss_history[-1] < result.loss_history[0] * 0.75
    assert 0.0 <= result.final_metrics["train_accuracy"] <= 1.0


def test_unsupported_layer_types_are_rejected_instead_of_simulated():
    engine = ExecutionEngine(device="cpu")
    with pytest.raises(ValueError, match="supports dense feed-forward"):
        engine.build_graph([
            {"type": "input", "neurons": 4},
            {"type": "conv2d", "neurons": 8},
        ])
