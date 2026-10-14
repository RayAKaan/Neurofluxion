"""Graph-integrated training update contract for Embedding→Attention→RNN→Dense."""
from __future__ import annotations

import numpy as np

from simulator.graph_engine import build_graph
from simulator.layers import LayerConfig
from simulator.training_engine import TrainingConfig, TrainingEngine


def test_embedding_attention_rnn_graph_training_updates_each_parameter_family():
    np.random.seed(440)
    graph = build_graph([
        LayerConfig("input", neurons=3, input_shape=[3]),
        LayerConfig("embedding", neurons=2, embedding_dim=2, vocab_size=5),
        LayerConfig("attention", neurons=2, num_heads=1),
        LayerConfig("rnn", neurons=2, hidden_size=2, return_sequences=False),
        LayerConfig("output", neurons=1, activation="sigmoid"),
    ])
    engine = TrainingEngine(
        graph,
        TrainingConfig(
            epochs=1, batch_size=1, learning_rate=0.01, optimizer="sgd",
            loss_function="bce", shuffle=False, dropout_rate=0.0,
        ),
    )
    # Repeated token 1 exercises accumulation into a shared embedding row.
    x = np.asarray([1, 2, 1], dtype=np.float32)
    y = np.asarray([1.0], dtype=np.float32)
    before = [value.copy() for value in graph.weights]

    loss, accuracy, gradient_norm = engine.train_batch([(x.copy(), y.copy())])

    assert np.isfinite(loss)
    assert np.isfinite(accuracy)
    assert np.isfinite(gradient_norm) and gradient_norm > 0.0
    assert all(np.isfinite(value).all() for value in engine.last_gradients["dW"])
    assert all(np.isfinite(value).all() for value in engine.last_gradients["db"])
    assert engine.optimizer_state is not None

    # Param order: embedding table, packed attention projections, packed RNN weights, dense weights.
    assert len(graph.weights) == 4
    changed = [not np.array_equal(old, new) for old, new in zip(before, graph.weights)]
    assert changed == [True, True, True, True], f"unexpected parameter update pattern: {changed}"
    assert np.any(graph.weights[0][1] != before[0][1]), "repeated token row 1 did not update"
    assert np.any(graph.weights[0][2] != before[0][2]), "token row 2 did not update"
