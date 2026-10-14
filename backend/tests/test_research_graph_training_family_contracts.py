"""Research contracts for recurrent graph parameter updates through TrainingEngine."""
from __future__ import annotations

import numpy as np
import pytest

from simulator.graph_engine import build_graph
from simulator.layers import LayerConfig
from simulator.training_engine import TrainingConfig, TrainingEngine


def _make_graph(kind: str, seed: int):
    np.random.seed(seed)
    return build_graph([
        LayerConfig("input", neurons=6, input_shape=[3, 2]),
        LayerConfig(kind, neurons=2, hidden_size=2, return_sequences=False),
        LayerConfig("output", neurons=1, activation="sigmoid"),
    ])


@pytest.mark.parametrize(("kind", "seed"), [("lstm", 436), ("gru", 437)])
def test_recurrent_graph_training_batch_updates_recurrent_and_output_parameters(kind, seed):
    graph = _make_graph(kind, seed)
    engine = TrainingEngine(
        graph,
        TrainingConfig(
            epochs=1,
            batch_size=1,
            learning_rate=0.02,
            optimizer="sgd",
            loss_function="bce",
            shuffle=False,
            dropout_rate=0.0,
        ),
    )
    x = np.asarray([[0.2, -0.3], [0.5, 0.1], [-0.4, 0.6]], dtype=np.float32)
    y = np.asarray([1.0], dtype=np.float32)
    before = [value.copy() for value in graph.weights + graph.biases]

    loss, accuracy, gradient_norm = engine.train_batch([(x.copy(), y.copy())])

    assert np.isfinite(loss)
    assert np.isfinite(accuracy)
    assert np.isfinite(gradient_norm) and gradient_norm > 0.0
    assert engine.optimizer_state is not None
    assert engine.last_gradients["dW"] and engine.last_gradients["db"]
    assert all(np.isfinite(value).all() for value in engine.last_gradients["dW"])
    assert all(np.isfinite(value).all() for value in engine.last_gradients["db"])

    after = graph.weights + graph.biases
    assert len(before) == len(after)
    changed_indices = [i for i, (old, new) in enumerate(zip(before, after)) if not np.array_equal(old, new)]
    assert changed_indices, f"{kind}: optimizer step did not change any graph parameters"

    recurrent_param_index = graph.param_index_by_layer[1]
    assert not np.array_equal(
        before[recurrent_param_index], graph.weights[recurrent_param_index]
    ), f"{kind}: recurrent parameter tensor did not update"
