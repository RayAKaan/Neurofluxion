"""Research contracts that exercise operator families through NetworkGraph.

These are integration smoke/consistency checks, not independent numerical
oracles for every operator. Operator-specific finite-difference tests remain
separate and must be cited separately.
"""
from __future__ import annotations

import numpy as np
import pytest

from simulator.forward_engine import run_forward_full
from simulator.graph_engine import build_graph
from simulator.layers import LayerConfig, validate_layers


def _cases():
    # name, layers, input, expected output shape
    yield (
        "dense",
        [
            LayerConfig("input", neurons=3, input_shape=[3]),
            LayerConfig("dense", neurons=4, activation="tanh"),
            LayerConfig("output", neurons=2, activation="linear"),
        ],
        np.array([0.2, -0.4, 0.7], dtype=np.float32),
        (2,),
    )
    yield (
        "conv-pool-flatten",
        [
            LayerConfig("input", neurons=16, input_shape=[1, 4, 4]),
            LayerConfig("conv2d", neurons=2, filters=2, kernel_size=3, padding="same", activation="relu"),
            LayerConfig("maxpool2d", neurons=2, pool_size=2),
            LayerConfig("flatten", neurons=8),
            LayerConfig("output", neurons=2, activation="linear"),
        ],
        np.arange(16, dtype=np.float32).reshape(1, 4, 4) / 10.0,
        (2,),
    )
    yield (
        "batchnorm",
        [
            LayerConfig("input", neurons=3, input_shape=[3]),
            LayerConfig("batchnorm", neurons=3),
            LayerConfig("output", neurons=2, activation="linear"),
        ],
        np.array([0.2, -0.4, 0.7], dtype=np.float32),
        (2,),
    )
    for kind in ("rnn", "lstm", "gru"):
        yield (
            kind,
            [
                LayerConfig("input", neurons=6, input_shape=[3, 2]),
                LayerConfig(kind, neurons=3, hidden_size=3, return_sequences=False),
                LayerConfig("output", neurons=2, activation="linear"),
            ],
            np.array([[0.1, 0.2], [-0.3, 0.4], [0.5, -0.2]], dtype=np.float32),
            (2,),
        )
    yield (
        "embedding-attention-rnn",
        [
            LayerConfig("input", neurons=3, input_shape=[3]),
            LayerConfig("embedding", neurons=2, embedding_dim=2, vocab_size=5),
            LayerConfig("attention", neurons=2, num_heads=1),
            LayerConfig("rnn", neurons=3, hidden_size=3, return_sequences=False),
            LayerConfig("output", neurons=2, activation="linear"),
        ],
        np.array([1, 2, 1], dtype=np.float32),
        (2,),
    )
    yield (
        "residual",
        [
            LayerConfig("input", neurons=3, input_shape=[3]),
            LayerConfig("dense", neurons=3, activation="tanh"),
            LayerConfig("residual", neurons=3),
            LayerConfig("output", neurons=2, activation="linear"),
        ],
        np.array([0.2, -0.4, 0.7], dtype=np.float32),
        (2,),
    )
    yield (
        "avgpool-flatten",
        [
            LayerConfig("input", neurons=16, input_shape=[1, 4, 4]),
            LayerConfig("conv2d", neurons=1, filters=1, kernel_size=1, padding="valid", activation="linear"),
            LayerConfig("avgpool2d", neurons=1, pool_size=2),
            LayerConfig("flatten", neurons=4),
            LayerConfig("output", neurons=2, activation="linear"),
        ],
        np.arange(16, dtype=np.float32).reshape(1, 4, 4) / 10.0,
        (2,),
    )


CASES = list(_cases())


@pytest.mark.parametrize(
    "name,layers,input_value,expected_shape",
    CASES,
    ids=[case[0] for case in CASES],
)
def test_graph_forward_and_trace_agree_for_operator_families(name, layers, input_value, expected_shape):
    """The graph and trace paths must agree for these fixed integration cases."""
    np.random.seed(20261014)
    validation = validate_layers(layers)
    assert validation.valid, f"{name}: {validation.errors}"

    graph = build_graph(layers)
    expected = np.asarray(graph.forward(input_value.copy()), dtype=np.float64)
    steps, traced_output, layer_outputs = run_forward_full(graph, input_value.tolist())
    actual = np.asarray(traced_output, dtype=np.float64)

    assert expected.shape == expected_shape, f"{name}: graph output shape {expected.shape}"
    assert actual.shape == expected_shape, f"{name}: trace output shape {actual.shape}"
    assert np.all(np.isfinite(expected)), f"{name}: graph output contains non-finite values"
    assert np.all(np.isfinite(actual)), f"{name}: trace output contains non-finite values"
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5, err_msg=name)
    assert steps, f"{name}: trace must contain at least one event"
    assert layer_outputs, f"{name}: trace must expose at least one layer output"
