"""Trace-step consistency contracts for dense forward inspection."""
import numpy as np
import pytest

from simulator.forward_engine import run_forward_full, run_forward_step
from simulator.graph_engine import build_graph
from simulator.layers import LayerConfig


def _graph():
    graph = build_graph([
        LayerConfig(layer_type="input", neurons=2, input_shape=[2]),
        LayerConfig(layer_type="dense", neurons=3, activation="tanh"),
        LayerConfig(layer_type="output", neurons=2, activation="linear"),
    ])
    weights = [
        np.array([[0.12, -0.31], [0.27, 0.18], [-0.22, 0.41]], dtype=np.float32),
        np.array([[0.33, -0.17, 0.26], [-0.11, 0.29, 0.08]], dtype=np.float32),
    ]
    biases = [
        np.array([0.03, -0.04, 0.02], dtype=np.float32),
        np.array([0.01, -0.02], dtype=np.float32),
    ]
    parameter_layers = [layer for layer in graph.layer_instances if getattr(layer, "has_params", False)]
    for layer, weight, bias in zip(parameter_layers, weights, biases):
        layer.set_params(weight.copy(), bias.copy())
    graph.weights = [layer.params()[0] for layer in parameter_layers]
    graph.biases = [layer.params()[1] for layer in parameter_layers]
    return graph


def test_each_step_matches_full_trace_and_completed_activation_prefix():
    graph = _graph()
    x = [0.4, -0.7]
    steps, final_output, layer_outputs = run_forward_full(graph, x)
    assert len(steps) == 6

    for index, expected_step in enumerate(steps):
        result = run_forward_step(graph, x, index)
        assert result["step"] == expected_step
        expected_completed = sorted({
            step["layer_index"]
            for step in steps[: index + 1]
            if step["operation"] in {"activation", "forward"}
        })
        assert result["completed_layers"] == expected_completed
        assert set(result["partial_activations"]) == {str(layer) for layer in expected_completed}
        for layer in expected_completed:
            np.testing.assert_allclose(
                result["partial_activations"][str(layer)], layer_outputs[layer], rtol=0, atol=0
            )
    np.testing.assert_allclose(final_output, graph.activations[-1], rtol=0, atol=0)


def test_step_selection_rejects_out_of_range_indices():
    graph = _graph()
    with pytest.raises(IndexError, match="step_index out of range"):
        run_forward_step(graph, [0.4, -0.7], -1)
    with pytest.raises(IndexError, match="step_index out of range"):
        run_forward_step(graph, [0.4, -0.7], 6)
