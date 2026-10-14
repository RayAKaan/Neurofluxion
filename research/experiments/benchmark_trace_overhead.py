#!/usr/bin/env python3
"""Reproducible, non-gating diagnostic for forward execution vs trace generation.

This benchmark intentionally records raw samples and environment metadata. It is
not a claim of general runtime overhead: the baseline and traced paths differ in
both computation and event serialization, and results are hardware-dependent.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

from simulator.forward_engine import run_forward_full
from simulator.graph_engine import build_graph
from simulator.layers import LayerConfig


def percentile_summary(samples_ms: list[float]) -> dict:
    values = np.asarray(samples_ms, dtype=np.float64)
    return {
        "n": int(values.size),
        "median_ms": float(np.median(values)),
        "p25_ms": float(np.percentile(values, 25)),
        "p75_ms": float(np.percentile(values, 75)),
        "iqr_ms": float(np.percentile(values, 75) - np.percentile(values, 25)),
        "min_ms": float(np.min(values)),
        "max_ms": float(np.max(values)),
        "raw_samples_ms": [float(x) for x in values],
    }


def make_graph(widths: list[int], seed: int):
    configs = [LayerConfig(layer_type="input", neurons=widths[0], input_shape=[widths[0]])]
    configs.extend(
        LayerConfig(
            layer_type="dense" if index < len(widths) - 1 else "output",
            neurons=widths[index],
            activation="tanh" if index < len(widths) - 1 else "linear",
        )
        for index in range(1, len(widths))
    )
    graph = build_graph(configs)
    rng = np.random.default_rng(seed)
    parameter_layers = [layer for layer in graph.layer_instances if getattr(layer, "has_params", False)]
    for layer in parameter_layers:
        w, b = layer.params()
        layer.set_params(
            rng.normal(0, 0.1, size=w.shape).astype(np.float32),
            rng.normal(0, 0.1, size=b.shape).astype(np.float32),
        )
    graph.weights = [layer.params()[0] for layer in parameter_layers]
    graph.biases = [layer.params()[1] for layer in parameter_layers]
    return graph


def timed_ms(fn) -> float:
    start = time.perf_counter_ns()
    fn()
    return (time.perf_counter_ns() - start) / 1_000_000.0


def run_case(widths: list[int], warmups: int, repetitions: int, seed: int) -> dict:
    graph = make_graph(widths, seed)
    x = np.random.default_rng(seed + 1).normal(size=(widths[0],)).astype(np.float32)
    reference = np.asarray(graph.forward(x), dtype=np.float32).reshape(-1)
    trace_steps, traced_output, _ = run_forward_full(graph, x.tolist())
    np.testing.assert_allclose(np.asarray(traced_output), reference, rtol=1e-5, atol=1e-6)

    for _ in range(warmups):
        graph.forward(x)
        run_forward_full(graph, x.tolist())

    baseline_samples = [timed_ms(lambda: graph.forward(x)) for _ in range(repetitions)]
    trace_samples = [timed_ms(lambda: run_forward_full(graph, x.tolist())) for _ in range(repetitions)]
    baseline = percentile_summary(baseline_samples)
    traced = percentile_summary(trace_samples)
    baseline_median = baseline["median_ms"]
    trace_median = traced["median_ms"]
    return {
        "layer_widths": widths,
        "input_dimensions": widths[0],
        "output_dimensions": widths[-1],
        "dense_layers": len(widths) - 1,
        "trace_events": len(trace_steps),
        "output_parity": True,
        "forward_only": baseline,
        "forward_with_trace": traced,
        "median_ratio_trace_over_forward": (
            float(trace_median / baseline_median) if baseline_median > 0 else None
        ),
        "median_delta_ms": float(trace_median - baseline_median),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20261014)
    args = parser.parse_args()
    if args.warmups < 0 or args.repetitions < 10:
        parser.error("warmups must be >= 0 and repetitions must be >= 10")

    cases = [
        [4, 8, 2],
        [16, 32, 8],
        [32, 64, 16],
        [64, 128, 32],
    ]
    report = {
        "schema_version": 1,
        "benchmark": "dense_forward_vs_trace_generation",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_revision": __import__("os").environ.get("GITHUB_SHA", "local-unpinned"),
        "environment": {
            "python": sys.version,
            "python_implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "numpy": np.__version__,
            "pid": __import__("os").getpid(),
        },
        "method": {
            "warmups_per_path_per_case": args.warmups,
            "repetitions_per_path_per_case": args.repetitions,
            "seed": args.seed,
            "summary": "median, p25, p75, IQR, min, max; all raw samples retained",
            "caveat": "diagnostic only; compares layer-instance forward with dense trace construction and serialization; not a universal overhead estimate",
        },
        "cases": [
            run_case(widths, args.warmups, args.repetitions, args.seed + index * 17)
            for index, widths in enumerate(cases)
        ],
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "cases": [
            {
                "widths": case["layer_widths"],
                "forward_median_ms": case["forward_only"]["median_ms"],
                "trace_median_ms": case["forward_with_trace"]["median_ms"],
                "median_ratio": case["median_ratio_trace_over_forward"],
                "events": case["trace_events"],
                "output_parity": case["output_parity"],
            }
            for case in report["cases"]
        ],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
