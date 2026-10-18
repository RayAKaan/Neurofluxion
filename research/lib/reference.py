"""Independent float64 reference implementations of the simulator's operators.

Why this module exists
----------------------
The repository's other numerical evidence differentiates the simulator's own
loss function. That is necessary but not sufficient: a forward pass that is wrong
and a matching loss definition can still produce an internally consistent
objective whose finite-difference gradient agrees with the implementation's
analytic gradient. Only an implementation written independently of the runtime
can break that degeneracy.

Scope and honesty constraints
-----------------------------
* Every operator here is written from its standard mathematical definition in
  float64. It is not a copy of ``backend/simulator/layers/*``.
* Where the simulator's definition deviates from the standard definition (for
  example the padding rule for ``same`` with stride greater than one), the
  deviation is detected and reported by :func:`compare_forward`; the reference is
  never adjusted to match the runtime.
* The reference is research instrumentation. It is never imported by runtime code.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

Array = np.ndarray


# ---------------------------------------------------------------------------
# activations
# ---------------------------------------------------------------------------
def sigmoid(x: Array) -> Array:
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x: Array) -> Array:
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


ACTIVATIONS = {
    "relu": lambda x: np.maximum(0.0, x),
    "sigmoid": sigmoid,
    "tanh": np.tanh,
    "leaky_relu": lambda x: np.where(x > 0, x, 0.01 * x),
    "linear": lambda x: x,
}


def apply_activation(name: Optional[str], x: Array) -> Array:
    key = (name or "linear").lower()
    if key not in ACTIVATIONS:
        raise KeyError(f"reference activation '{key}' is not defined")
    return ACTIVATIONS[key](x)


# ---------------------------------------------------------------------------
# losses
# ---------------------------------------------------------------------------
def mse_loss(y_true: Array, y_pred: Array) -> float:
    """Mean of 0.5*(y_true - y_pred)^2, the simulator's documented dense contract."""
    return float(np.mean(0.5 * (np.asarray(y_true, dtype=np.float64) - np.asarray(y_pred, dtype=np.float64)) ** 2))


def bce_loss(y_true: Array, y_pred: Array, eps: float = 1e-7) -> float:
    """Clipped mean binary cross-entropy, matching the simulator's declared objective."""
    p = np.clip(np.asarray(y_pred, dtype=np.float64).ravel(), eps, 1.0 - eps)
    t = np.clip(np.asarray(y_true, dtype=np.float64).ravel(), eps, eps)
    return float(np.mean(-(t * np.log(p) + (1.0 - t) * np.log(1.0 - p))))


def objective(y_true: Array, y_pred: Array, loss_fn: str) -> float:
    return mse_loss(y_true, y_pred) if (loss_fn or "bce").lower() == "mse" else bce_loss(y_true, y_pred)


# ---------------------------------------------------------------------------
# operators
# ---------------------------------------------------------------------------
def dense(x: Array, w: Array, b: Array, activation: Optional[str]) -> Array:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    return apply_activation(activation, np.asarray(w, dtype=np.float64) @ x + np.asarray(b, dtype=np.float64))


def conv2d(
    x: Array,
    kernel: Array,
    bias: Array,
    stride: int = 1,
    padding: str = "valid",
    activation: str = "relu",
) -> Array:
    """NCHW convolution.

    ``padding="same"`` uses the TensorFlow/Keras convention
    ``out = ceil(in / stride)`` with the padding distributed so the kernel stays
    centred. The simulator's own ``same`` rule differs when ``stride > 1``; that
    difference is detected by the caller, never silently absorbed. Pass
    ``padding="same_simulator"`` to reproduce the simulator's geometry when an
    arithmetic (rather than a geometry) comparison is required.
    """
    x = np.asarray(x, dtype=np.float64)
    kernel = np.asarray(kernel, dtype=np.float64)
    c_out, c_in, k_h, k_w = kernel.shape
    _, h, w = x.shape
    if padding in {"same", "same_simulator"}:
        if padding == "same_simulator":
            pad_h = max((h - 1) * stride + k_h - h, 0) // 2
            pad_w = max((w - 1) * stride + k_w - w, 0) // 2
            x = np.pad(x, ((0, 0), (pad_h, pad_h), (pad_w, pad_w)), mode="constant")
        else:
            out_h = math.ceil(h / stride)
            out_w = math.ceil(w / stride)
            pad_total_h = max((out_h - 1) * stride + k_h - h, 0)
            pad_total_w = max((out_w - 1) * stride + k_w - w, 0)
            pad_h_before = pad_total_h // 2
            pad_w_before = pad_total_w // 2
            x = np.pad(
                x,
                (
                    (0, 0),
                    (pad_h_before, pad_total_h - pad_h_before),
                    (pad_w_before, pad_total_w - pad_w_before),
                ),
                mode="constant",
            )
        h, w = x.shape[1], x.shape[2]
    out_h = (h - k_h) // stride + 1
    out_w = (w - k_w) // stride + 1
    out = np.zeros((c_out, out_h, out_w), dtype=np.float64)
    for oc in range(c_out):
        for i in range(out_h):
            for j in range(out_w):
                patch = x[:, i * stride : i * stride + k_h, j * stride : j * stride + k_w]
                out[oc, i, j] = float(np.sum(patch * kernel[oc])) + float(
                    np.asarray(bias, dtype=np.float64)[oc]
                )
    return apply_activation(activation, out)


def max_pool2d(x: Array, pool_size: int, stride: Optional[int] = None) -> Array:
    x = np.asarray(x, dtype=np.float64)
    s = stride or pool_size
    c, h, w = x.shape
    out_h = (h - pool_size) // s + 1
    out_w = (w - pool_size) // s + 1
    out = np.zeros((c, out_h, out_w), dtype=np.float64)
    for i in range(out_h):
        for j in range(out_w):
            patch = x[:, i * s : i * s + pool_size, j * s : j * s + pool_size]
            out[:, i, j] = patch.reshape(c, -1).max(axis=1)
    return out


def avg_pool2d(x: Array, pool_size: int, stride: Optional[int] = None) -> Array:
    x = np.asarray(x, dtype=np.float64)
    s = stride or pool_size
    c, h, w = x.shape
    out_h = (h - pool_size) // s + 1
    out_w = (w - pool_size) // s + 1
    out = np.zeros((c, out_h, out_w), dtype=np.float64)
    for i in range(out_h):
        for j in range(out_w):
            patch = x[:, i * s : i * s + pool_size, j * s : j * s + pool_size]
            out[:, i, j] = patch.reshape(c, -1).mean(axis=1)
    return out


def flatten(x: Array) -> Array:
    return np.asarray(x, dtype=np.float64).reshape(-1)


def batchnorm(x: Array, gamma: Array, beta: Array, eps: float = 1e-5) -> Array:
    """Standard batch normalisation over the batch/spatial dimensions."""
    x = np.asarray(x, dtype=np.float64)
    gamma = np.asarray(gamma, dtype=np.float64)
    beta = np.asarray(beta, dtype=np.float64)
    if x.ndim == 1:
        mean = x
        var = np.zeros_like(x)
    elif x.ndim == 3:
        mean = x.mean(axis=(1, 2))
        var = x.var(axis=(1, 2))
    else:
        mean = x.mean(axis=0)
        var = x.var(axis=0)
    shape = (-1, 1, 1) if x.ndim == 3 else -1
    normalised = (x - mean.reshape(shape)) / np.sqrt(var.reshape(shape) + eps)
    return gamma.reshape(shape) * normalised + beta.reshape(shape)


def embedding(indices: Array, table: Array) -> Array:
    return np.asarray(table, dtype=np.float64)[np.asarray(indices, dtype=np.int64).reshape(-1)]


def attention(x: Array, w_q: Array, w_k: Array, w_v: Array, w_o: Array) -> Array:
    """Single-head scaled dot-product attention, the standard definition."""
    x = np.asarray(x, dtype=np.float64)
    q = x @ np.asarray(w_q, dtype=np.float64)
    k = x @ np.asarray(w_k, dtype=np.float64)
    v = x @ np.asarray(w_v, dtype=np.float64)
    scale = math.sqrt(q.shape[-1])
    scores = (q @ k.T) / scale
    shifted = scores - np.max(scores, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    weights = exp / np.sum(exp, axis=-1, keepdims=True)
    return (weights @ v) @ np.asarray(w_o, dtype=np.float64)


def rnn(x: Array, w_xh: Array, w_hh: Array, b: Array, return_sequences: bool = False) -> Array:
    x = np.asarray(x, dtype=np.float64)
    steps = x.shape[0]
    hidden = w_xh.shape[0]
    h_prev = np.zeros(hidden, dtype=np.float64)
    states: List[Array] = []
    for t in range(steps):
        h = np.tanh(w_xh @ x[t] + w_hh @ h_prev + b)
        states.append(h)
        h_prev = h
    stacked = np.stack(states)
    return stacked if return_sequences else stacked[-1]


def _sigmoid(x: Array) -> Array:
    return 1.0 / (1.0 + np.exp(-x))


def lstm(
    x: Array,
    w_f: Array,
    w_i: Array,
    w_o: Array,
    w_c: Array,
    b_f: Array,
    b_i: Array,
    b_o: Array,
    b_c: Array,
    return_sequences: bool = False,
) -> Array:
    x = np.asarray(x, dtype=np.float64)
    steps = x.shape[0]
    hidden = w_f.shape[0]
    h_prev = np.zeros(hidden, dtype=np.float64)
    c_prev = np.zeros(hidden, dtype=np.float64)
    states: List[Array] = []
    for t in range(steps):
        concat = np.concatenate([h_prev, x[t]])
        f = _sigmoid(w_f @ concat + b_f)
        i = _sigmoid(w_i @ concat + b_i)
        o = _sigmoid(w_o @ concat + b_o)
        g = np.tanh(w_c @ concat + b_c)
        c_prev = f * c_prev + i * g
        h_prev = o * np.tanh(c_prev)
        states.append(h_prev)
    stacked = np.stack(states)
    return stacked if return_sequences else stacked[-1]


def gru(
    x: Array,
    w_r: Array,
    w_z: Array,
    w_h: Array,
    b_r: Array,
    b_z: Array,
    b_h: Array,
    return_sequences: bool = False,
) -> Array:
    x = np.asarray(x, dtype=np.float64)
    steps = x.shape[0]
    hidden = w_r.shape[0]
    h_prev = np.zeros(hidden, dtype=np.float64)
    states: List[Array] = []
    for t in range(steps):
        concat = np.concatenate([h_prev, x[t]])
        r = _sigmoid(w_r @ concat + b_r)
        z = _sigmoid(w_z @ concat + b_z)
        h_tilde = np.tanh(w_h @ np.concatenate([r * h_prev, x[t]]) + b_h)
        h_prev = (1.0 - z) * h_prev + z * h_tilde
        states.append(h_prev)
    stacked = np.stack(states)
    return stacked if return_sequences else stacked[-1]


def residual(x: Array) -> Array:
    return np.asarray(x, dtype=np.float64)


# ---------------------------------------------------------------------------
# graph-level reference forward
# ---------------------------------------------------------------------------
@dataclass
class ReferenceStep:
    layer_index: int
    layer_type: str
    input_shape: List[int]
    output_shape: List[int]
    values: Array


@dataclass
class ReferenceRun:
    output: Array
    steps: List[ReferenceStep] = field(default_factory=list)
    cache: Dict[int, Dict[str, Any]] = field(default_factory=dict)


def _unpack(layer, graph, kind: str = "weights"):
    """Read the named parameters a layer actually uses in its forward pass."""
    lt = getattr(layer, "layer_type", "")
    if lt == "conv2d":
        return layer.K, layer.b
    if lt == "dense":
        return layer.W, layer.b
    if lt == "batchnorm":
        return layer.gamma, layer.beta
    if lt == "embedding":
        return layer.E, np.zeros(1)
    if lt == "attention":
        return np.stack([layer.W_q, layer.W_k, layer.W_v, layer.W_o]), np.zeros(1)
    if lt == "rnn":
        return np.concatenate([layer.W_xh, layer.W_hh], axis=1), layer.b
    if lt == "lstm":
        return (
            np.concatenate([layer.W_f, layer.W_i, layer.W_o, layer.W_c], axis=0),
            np.concatenate([layer.b_f, layer.b_i, layer.b_o, layer.b_c], axis=0),
        )
    if lt == "gru":
        return (
            np.concatenate([layer.W_r, layer.W_z, layer.W_h], axis=0),
            np.concatenate([layer.b_r, layer.b_z, layer.b_h], axis=0),
        )
    return None, None


def reference_forward(graph, x: Array, strict_padding: bool = True) -> ReferenceRun:
    """Execute a simulator graph with the independent float64 reference.

    ``strict_padding`` selects the standard ``same``-padding convention. When the
    simulator's padding rule differs, the caller can re-run with
    ``strict_padding=False`` to reproduce the simulator's own geometry and
    isolate padding from arithmetic.
    """
    current = np.asarray(x, dtype=np.float64)
    run = ReferenceRun(output=current)
    for index, layer in enumerate(graph.layer_instances):
        lt = getattr(layer, "layer_type", "")
        step_input = current
        if lt == "input":
            current = np.asarray(current, dtype=np.float64)
        elif lt == "dense":
            w, b = _unpack(layer, graph)
            current = dense(current, w, b, getattr(layer, "activation_name", "linear"))
        elif lt == "conv2d":
            w, b = _unpack(layer, graph)
            padding = layer.cfg.padding
            if padding != "valid" and not strict_padding:
                padding = "same_simulator"
            current = conv2d(
                current,
                w,
                b,
                stride=layer.cfg.stride,
                padding=padding,
                activation=layer.cfg.activation,
            )
        elif lt == "maxpool2d":
            current = max_pool2d(current, layer.cfg.pool_size, layer.cfg.stride)
        elif lt == "avgpool2d":
            current = avg_pool2d(current, layer.cfg.pool_size, layer.cfg.stride)
        elif lt == "flatten":
            current = flatten(current)
        elif lt == "batchnorm":
            gamma, beta = _unpack(layer, graph)
            current = batchnorm(current, gamma, beta, layer.eps)
        elif lt == "embedding":
            table, _ = _unpack(layer, graph)
            current = embedding(current, table)
        elif lt == "attention":
            w, _ = _unpack(layer, graph)
            current = attention(current, w[0], w[1], w[2], w[3])
        elif lt == "rnn":
            w, b = _unpack(layer, graph)
            current = rnn(current, w[:, : layer.input_dim], w[:, layer.input_dim :], b, layer.return_sequences)
        elif lt == "lstm":
            w, b = _unpack(layer, graph)
            h = layer.hidden_dim
            current = lstm(
                current,
                w[0:h],
                w[h : 2 * h],
                w[2 * h : 3 * h],
                w[3 * h : 4 * h],
                b[0:h],
                b[h : 2 * h],
                b[2 * h : 3 * h],
                b[3 * h : 4 * h],
                layer.return_sequences,
            )
        elif lt == "gru":
            w, b = _unpack(layer, graph)
            h = layer.hidden_dim
            current = gru(
                current,
                w[0:h],
                w[h : 2 * h],
                w[2 * h : 3 * h],
                b[0:h],
                b[h : 2 * h],
                b[2 * h : 3 * h],
                layer.return_sequences,
            )
        elif lt == "residual":
            current = residual(current)
        else:
            raise KeyError(f"reference forward has no definition for layer type '{lt}'")
        run.steps.append(
            ReferenceStep(
                layer_index=index,
                layer_type=lt,
                input_shape=list(np.shape(step_input)),
                output_shape=list(np.shape(current)),
                values=current,
            )
        )
    run.output = current
    return run


def _conv_simulator_padding(x: Array, kernel: Array, bias: Array, stride: int, activation: str) -> Array:
    """Reproduce the simulator's own ``same``-padding geometry (retained helper)."""
    return conv2d(x, kernel, bias, stride=stride, padding="same_simulator", activation=activation)


# ---------------------------------------------------------------------------
# comparison helpers
# ---------------------------------------------------------------------------
def compare_forward(
    actual: Array,
    expected: Array,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> Dict[str, Any]:
    a = np.asarray(actual, dtype=np.float64)
    b = np.asarray(expected, dtype=np.float64)
    if a.shape != b.shape:
        return {
            "comparable": False,
            "reason": "shape mismatch",
            "actual_shape": list(a.shape),
            "reference_shape": list(b.shape),
            "max_abs_error": None,
            "passed": False,
        }
    diff = float(np.max(np.abs(a - b))) if a.size else 0.0
    return {
        "comparable": True,
        "actual_shape": list(a.shape),
        "reference_shape": list(b.shape),
        "max_abs_error": diff,
        "relative_l2_error": float(
            np.linalg.norm(a - b) / (np.linalg.norm(a) + np.linalg.norm(b) + 1e-15)
        ),
        "atol": atol,
        "rtol": rtol,
        "passed": bool(np.allclose(a, b, atol=atol, rtol=rtol)),
    }


def reference_objective_factory(graph, x: Array, target: Array, loss_fn: str, strict_padding: bool = True):
    """Scalar objective evaluated with the independent float64 reference."""

    def evaluate() -> float:
        run = reference_forward(graph, np.asarray(x), strict_padding=strict_padding)
        return objective(target, run.output.reshape(-1), loss_fn)

    return evaluate


def reference_objective_central_difference(
    objective_fn,
    array: Array,
    epsilon: float,
    setter,
) -> Array:
    """Central difference of an independent float64 objective."""
    work = np.array(array, dtype=np.float64, copy=True)
    grad = np.zeros_like(work)

    def commit() -> None:
        setter(work)

    for index in np.ndindex(work.shape):
        original = float(work[index])
        try:
            work[index] = original + epsilon
            commit()
            plus = objective_fn()
            work[index] = original - epsilon
            commit()
            minus = objective_fn()
        finally:
            work[index] = original
            commit()
        grad[index] = (plus - minus) / (2.0 * epsilon)
    return grad