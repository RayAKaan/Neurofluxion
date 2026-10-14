"""Reference-equation contracts for optimizer state and invalid configuration."""
import numpy as np
import pytest

from simulator.optimizer_engine import apply_update


def _parameters():
    return [np.array([1.0, -2.0], dtype=np.float64)], [np.array([0.5], dtype=np.float64)]


def test_momentum_update_matches_two_step_reference():
    weights, biases = _parameters()
    grad_w = [np.array([0.25, -0.5], dtype=np.float64)]
    grad_b = [np.array([0.75], dtype=np.float64)]
    lr, momentum = 0.1, 0.8
    expected_w = weights[0].copy()
    expected_b = biases[0].copy()
    velocity_w = np.zeros_like(expected_w)
    velocity_b = np.zeros_like(expected_b)
    state = None

    for _ in range(2):
        velocity_w = momentum * velocity_w + grad_w[0]
        velocity_b = momentum * velocity_b + grad_b[0]
        expected_w = expected_w - lr * velocity_w
        expected_b = expected_b - lr * velocity_b
        weights, biases, state = apply_update(
            weights, biases, grad_w, grad_b, lr=lr,
            optimizer="momentum", state=state, momentum=momentum,
        )
        np.testing.assert_allclose(weights[0], expected_w, rtol=0, atol=1e-12)
        np.testing.assert_allclose(biases[0], expected_b, rtol=0, atol=1e-12)


def test_rmsprop_update_matches_reference_equation():
    weights, biases = _parameters()
    grad_w = [np.array([0.25, -0.5], dtype=np.float64)]
    grad_b = [np.array([0.75], dtype=np.float64)]
    lr, beta2, eps = 0.01, 0.9, 1e-8
    expected_vw = (1 - beta2) * grad_w[0] ** 2
    expected_vb = (1 - beta2) * grad_b[0] ** 2
    expected_w = weights[0] - lr * grad_w[0] / (np.sqrt(expected_vw) + eps)
    expected_b = biases[0] - lr * grad_b[0] / (np.sqrt(expected_vb) + eps)

    new_w, new_b, state = apply_update(
        weights, biases, grad_w, grad_b, lr=lr,
        optimizer="rmsprop", state=None, beta2=beta2, eps=eps,
    )
    np.testing.assert_allclose(new_w[0], expected_w, rtol=0, atol=1e-12)
    np.testing.assert_allclose(new_b[0], expected_b, rtol=0, atol=1e-12)
    np.testing.assert_allclose(state.v_w[0], expected_vw, rtol=0, atol=1e-12)
    np.testing.assert_allclose(state.v_b[0], expected_vb, rtol=0, atol=1e-12)


def test_adam_two_step_state_matches_reference_equations():
    weights, biases = _parameters()
    grads_w = [
        np.array([0.25, -0.5], dtype=np.float64),
        np.array([-0.1, 0.3], dtype=np.float64),
    ]
    grads_b = [
        np.array([0.75], dtype=np.float64),
        np.array([-0.2], dtype=np.float64),
    ]
    lr, beta1, beta2, eps = 0.01, 0.9, 0.999, 1e-8
    mw, vw = np.zeros_like(weights[0]), np.zeros_like(weights[0])
    mb, vb = np.zeros_like(biases[0]), np.zeros_like(biases[0])
    state = None

    for t, (gw, gb) in enumerate(zip(grads_w, grads_b), start=1):
        mw = beta1 * mw + (1 - beta1) * gw
        vw = beta2 * vw + (1 - beta2) * gw**2
        mb = beta1 * mb + (1 - beta1) * gb
        vb = beta2 * vb + (1 - beta2) * gb**2
        expected_w = weights[0] - lr * (mw / (1 - beta1**t)) / (np.sqrt(vw / (1 - beta2**t)) + eps)
        expected_b = biases[0] - lr * (mb / (1 - beta1**t)) / (np.sqrt(vb / (1 - beta2**t)) + eps)
        weights, biases, state = apply_update(
            weights, biases, [gw], [gb], lr=lr,
            optimizer="adam", state=state, beta1=beta1, beta2=beta2, eps=eps,
        )
        np.testing.assert_allclose(weights[0], expected_w, rtol=0, atol=1e-12)
        np.testing.assert_allclose(biases[0], expected_b, rtol=0, atol=1e-12)
        assert state.t == t


def test_unknown_optimizer_is_rejected_instead_of_silently_skipping_updates():
    weights, biases = _parameters()
    with pytest.raises(ValueError, match="Unsupported optimizer"):
        apply_update(
            weights, biases,
            [np.ones_like(weights[0])], [np.ones_like(biases[0])],
            lr=0.01, optimizer="adma", state=None,
        )
