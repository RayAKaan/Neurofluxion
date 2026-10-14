#!/usr/bin/env python3
"""Deterministic dense-path diagnostic for a Neurofluxion checkout.

Run from the repository root:
  PYTHONPATH=backend python research/experiments/run_simulator_audit.py \
    --repo . --out research/artifacts/simulator-audit.json

The script is diagnostic rather than a full validation suite. It never changes
repository source. It records measurements before returning a failing exit code.
A successful dense-path run does not establish correctness for other layer types.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ABS_TOL = 1e-5
GRAD_TOL = 2e-3


def commit_for(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def digest(value: object) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def max_abs(a: object, b: object) -> float:
    left = np.asarray(a, dtype=np.float64)
    right = np.asarray(b, dtype=np.float64)
    if left.shape != right.shape:
        raise ValueError(f"shape mismatch: {left.shape} != {right.shape}")
    if left.size == 0:
        return 0.0
    return float(np.max(np.abs(left - right)))


def normalized_l2(a: object, b: object) -> float:
    left = np.asarray(a, dtype=np.float64).ravel()
    right = np.asarray(b, dtype=np.float64).ravel()
    denom = np.linalg.norm(left) + np.linalg.norm(right) + 1e-12
    return float(np.linalg.norm(left - right) / denom)


def run(root: Path) -> dict:
    from simulator.layers import LayerConfig, validate_layers
    from simulator.graph_engine import build_graph
    from simulator.forward_engine import run_forward_full, run_forward_step
    from simulator.backward_engine import backward_full
    from simulator.loss_functions import compute_loss

    layers = [
        LayerConfig(layer_type="input", neurons=2, input_shape=[2]),
        LayerConfig(layer_type="dense", neurons=3, activation="tanh"),
        LayerConfig(layer_type="output", neurons=2, activation="linear"),
    ]
    validation = validate_layers(layers)
    if not validation.valid:
        raise RuntimeError(f"test architecture rejected: {validation.errors}")
    graph = build_graph(layers)

    weights = [
        np.array([[0.12, -0.31], [0.27, 0.18], [-0.22, 0.41]], dtype=np.float32),
        np.array([[0.33, -0.17, 0.26], [-0.11, 0.29, 0.08]], dtype=np.float32),
    ]
    biases = [
        np.array([0.03, -0.04, 0.02], dtype=np.float32),
        np.array([0.01, -0.02], dtype=np.float32),
    ]
    x = np.array([0.4, -0.7], dtype=np.float32)
    target = np.array([0.2, -0.1], dtype=np.float32)

    param_layers = [layer for layer in graph.layer_instances if getattr(layer, "has_params", False)]
    if len(param_layers) != len(weights):
        raise RuntimeError(f"expected 2 parameterized layers, found {len(param_layers)}")

    def set_fixed_params() -> None:
        for layer, w, b in zip(param_layers, weights, biases):
            layer.set_params(w.copy(), b.copy())
        graph.weights = [layer.params()[0] for layer in param_layers]
        graph.biases = [layer.params()[1] for layer in param_layers]

    set_fixed_params()

    # Independent forward oracle: explicit arithmetic, not graph.forward().
    z1 = weights[0] @ x + biases[0]
    a1 = np.tanh(z1)
    z2 = weights[1] @ a1 + biases[1]
    expected = z2
    actual = np.asarray(graph.forward(x), dtype=np.float64).reshape(-1)
    forward_error = max_abs(actual, expected)

    steps, trace_output, _ = run_forward_full(graph, x.tolist())
    trace_events = []
    for event in steps:
        op = event.get("operation")
        inp = np.asarray(event.get("input_values", []), dtype=np.float64)
        out = np.asarray(event.get("output_values", []), dtype=np.float64)
        if op == "matmul":
            ref = np.asarray(event["weights_snapshot"], dtype=np.float64) @ inp
        elif op == "bias_add":
            ref = inp + np.asarray(event["bias_snapshot"], dtype=np.float64)
        elif op == "activation":
            name = event.get("activation_name")
            if name == "tanh":
                ref = np.tanh(inp)
            elif name == "relu":
                ref = np.maximum(inp, 0.0)
            elif name == "sigmoid":
                ref = 1.0 / (1.0 + np.exp(-inp))
            elif name == "linear":
                ref = inp
            else:
                ref = None
        else:
            ref = None
        trace_events.append({
            "step_index": event.get("step_index"),
            "layer_index": event.get("layer_index"),
            "operation": op,
            "max_abs_residual": None if ref is None else max_abs(out, ref),
            "checked": ref is not None,
        })
    trace_final_error = max_abs(trace_output, expected)
    trace_checked = [e for e in trace_events if e["checked"]]
    trace_pass = (
        trace_final_error <= ABS_TOL
        and bool(trace_checked)
        and len(trace_checked) == len(trace_events)
        and all(e["max_abs_residual"] <= ABS_TOL for e in trace_checked)
    )

    # Finite differences of the exact implemented scalar loss.
    back = backward_full(graph, x.tolist(), target.tolist(), "mse")
    analytic_w = [np.asarray(g, dtype=np.float64) for g in back["gradients_W"]]
    analytic_b = [np.asarray(g, dtype=np.float64) for g in back["gradients_b"]]
    analytic = np.concatenate([g.ravel() for g in analytic_w + analytic_b])

    def scalar_loss() -> float:
        pred = np.asarray(graph.forward(x), dtype=np.float32).reshape(-1)
        value = compute_loss(target, pred, "mse")[0]
        return float(value)

    epsilons = [1e-2, 3e-3, 1e-3]
    gradient_runs = []
    for eps in epsilons:
        fd_w = [np.zeros_like(w, dtype=np.float64) for w in weights]
        fd_b = [np.zeros_like(b, dtype=np.float64) for b in biases]
        for params, fd_params in ((graph.weights, fd_w), (graph.biases, fd_b)):
            for layer_index, arr in enumerate(params):
                for ix in np.ndindex(arr.shape):
                    original = float(arr[ix])
                    try:
                        arr[ix] = original + eps
                        plus = scalar_loss()
                        arr[ix] = original - eps
                        minus = scalar_loss()
                    finally:
                        arr[ix] = original
                    fd_params[layer_index][ix] = (plus - minus) / (2.0 * eps)
        numeric = np.concatenate([g.ravel() for g in fd_w + fd_b])
        gradient_runs.append({
            "epsilon": eps,
            "normalized_l2_error": normalized_l2(analytic, numeric),
            "max_abs_error": max_abs(analytic, numeric),
            "analytic_norm": float(np.linalg.norm(analytic)),
            "finite_difference_norm": float(np.linalg.norm(numeric)),
            "analytic_gradient": analytic.tolist(),
            "finite_difference_gradient": numeric.tolist(),
            "gradient_error": (analytic - numeric).tolist(),
        })

    # Re-run with the exact same fixed parameters and input.
    set_fixed_params()
    output_a = np.asarray(graph.forward(x)).tolist()
    trace_a = run_forward_full(graph, x.tolist())[0]
    output_b = np.asarray(graph.forward(x)).tolist()
    trace_b = run_forward_full(graph, x.tolist())[0]
    repeatability = {
        "outputs_equal": output_a == output_b,
        "trace_hash_equal": digest(trace_a) == digest(trace_b),
        "output_hash": digest(output_a),
        "trace_hash": digest(trace_a),
    }

    # Selection latency is explicitly not described as incremental execution.
    for _ in range(5):
        run_forward_full(graph, x.tolist())
    full_samples = []
    for _ in range(30):
        start = time.perf_counter()
        run_forward_full(graph, x.tolist())
        full_samples.append((time.perf_counter() - start) * 1000.0)
    selected_steps = []
    for index in sorted({0, max(0, len(steps) // 2), len(steps) - 1}):
        start = time.perf_counter()
        selected = run_forward_step(graph, x.tolist(), index)
        selected_steps.append({
            "step_index": index,
            "latency_ms": (time.perf_counter() - start) * 1000.0,
            "operation": selected["step"]["operation"],
        })

    gradient_pass = bool(gradient_runs) and all(
        item["normalized_l2_error"] <= GRAD_TOL for item in gradient_runs
    )
    repeat_pass = repeatability["outputs_equal"] and repeatability["trace_hash_equal"]
    return {
        "metadata": {
            "commit": commit_for(root),
            "root": str(root),
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "command_scope": "small fixed dense network only",
        },
        "tolerances": {"forward_and_trace_abs": ABS_TOL, "gradient_normalized_l2": GRAD_TOL},
        "architecture": [
            {"type": layer.layer_type, "neurons": layer.neurons, "activation": layer.activation}
            for layer in layers
        ],
        "forward_oracle": {
            "max_abs_error": forward_error,
            "expected": expected.tolist(),
            "actual": actual.tolist(),
            "pass": forward_error <= ABS_TOL,
        },
        "trace_oracle": {
            "final_output_max_abs_error": trace_final_error,
            "events": trace_events,
            "pass": trace_pass,
        },
        "gradient_finite_difference": {
            "loss": "mse",
            "note": "Numerical derivatives are computed from the implementation's own scalar loss; check the exact reduction convention.",
            "runs": gradient_runs,
            "pass": gradient_pass,
        },
        "repeatability": {**repeatability, "pass": repeat_pass},
        "timing": {
            "full_forward_ms_samples": full_samples,
            "full_forward_median_ms": float(np.median(full_samples)),
            "step_selection": selected_steps,
            "note": "Diagnostic only. Publication-grade scaling requires multiple graph sizes, independent processes, and a controlled environment.",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=".", help="Path to a Neurofluxion checkout")
    parser.add_argument("--out", default="research/artifacts/simulator-audit.json")
    args = parser.parse_args()
    root = Path(args.repo).resolve()
    try:
        result = run(root)
    except Exception as exc:
        print(f"Audit failed before producing results: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {
        "output": str(output),
        "commit": result["metadata"]["commit"],
        "forward_pass": result["forward_oracle"]["pass"],
        "trace_pass": result["trace_oracle"]["pass"],
        "gradient_pass": result["gradient_finite_difference"]["pass"],
        "repeatability_pass": result["repeatability"]["pass"],
    }
    print(json.dumps(summary, indent=2))
    required = ("forward_oracle", "trace_oracle", "gradient_finite_difference", "repeatability")
    return 0 if all(result[key]["pass"] for key in required) else 1


if __name__ == "__main__":
    raise SystemExit(main())
