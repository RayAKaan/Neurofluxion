from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class OptimizerState:
    t: int
    m_w: List[np.ndarray]
    v_w: List[np.ndarray]
    m_b: List[np.ndarray]
    v_b: List[np.ndarray]


def _init_state(weights: List[np.ndarray], biases: List[np.ndarray]) -> OptimizerState:
    return OptimizerState(
        t=0,
        m_w=[np.zeros_like(w) for w in weights],
        v_w=[np.zeros_like(w) for w in weights],
        m_b=[np.zeros_like(b) for b in biases],
        v_b=[np.zeros_like(b) for b in biases],
    )


def apply_update(
    weights: List[np.ndarray],
    biases: List[np.ndarray],
    grads_w: List[np.ndarray],
    grads_b: List[np.ndarray],
    lr: float,
    optimizer: str,
    state: OptimizerState | None,
    momentum: float = 0.9,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> Tuple[List[np.ndarray], List[np.ndarray], OptimizerState]:
    opt = (optimizer or "sgd").lower()
    if state is None:
        state = _init_state(weights, biases)

    if opt == "sgd":
        new_w = [w - lr * gw for w, gw in zip(weights, grads_w)]
        new_b = [b - lr * gb for b, gb in zip(biases, grads_b)]
        return new_w, new_b, state

    if opt in {"sgd_momentum", "momentum"}:
        state.m_w = [momentum * m + gw for m, gw in zip(state.m_w, grads_w)]
        state.m_b = [momentum * m + gb for m, gb in zip(state.m_b, grads_b)]
        new_w = [w - lr * m for w, m in zip(weights, state.m_w)]
        new_b = [b - lr * m for b, m in zip(biases, state.m_b)]
        return new_w, new_b, state

    if opt == "rmsprop":
        state.v_w = [beta2 * v + (1 - beta2) * (gw ** 2) for v, gw in zip(state.v_w, grads_w)]
        state.v_b = [beta2 * v + (1 - beta2) * (gb ** 2) for v, gb in zip(state.v_b, grads_b)]
        new_w = [w - lr * gw / (np.sqrt(v) + eps) for w, gw, v in zip(weights, grads_w, state.v_w)]
        new_b = [b - lr * gb / (np.sqrt(v) + eps) for b, gb, v in zip(biases, grads_b, state.v_b)]
        return new_w, new_b, state

    if opt == "adam":
        state.t += 1
        state.m_w = [beta1 * m + (1 - beta1) * gw for m, gw in zip(state.m_w, grads_w)]
        state.v_w = [beta2 * v + (1 - beta2) * (gw ** 2) for v, gw in zip(state.v_w, grads_w)]
        state.m_b = [beta1 * m + (1 - beta1) * gb for m, gb in zip(state.m_b, grads_b)]
        state.v_b = [beta2 * v + (1 - beta2) * (gb ** 2) for v, gb in zip(state.v_b, grads_b)]

        m_w_hat = [m / (1 - beta1 ** state.t) for m in state.m_w]
        v_w_hat = [v / (1 - beta2 ** state.t) for v in state.v_w]
        m_b_hat = [m / (1 - beta1 ** state.t) for m in state.m_b]
        v_b_hat = [v / (1 - beta2 ** state.t) for v in state.v_b]

        new_w = [w - lr * m / (np.sqrt(v) + eps) for w, m, v in zip(weights, m_w_hat, v_w_hat)]
        new_b = [b - lr * m / (np.sqrt(v) + eps) for b, m, v in zip(biases, m_b_hat, v_b_hat)]
        return new_w, new_b, state

    raise ValueError(
        f"Unsupported optimizer '{optimizer}'. "
        "Choose from sgd, sgd_momentum, momentum, rmsprop, or adam."
    )

