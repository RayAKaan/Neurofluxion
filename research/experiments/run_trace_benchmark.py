#!/usr/bin/env python3
"""Cross-platform diagnostic comparing final-output forward and full trace paths."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

CASES = [(4, 8, 2), (16, 32, 8), (32, 64, 16), (64, 128, 32)]
WARMUPS = 5
SAMPLES = 40
ATOL = 1e-5
RTOL = 1e-5


def commit_for(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(values: list[float]) -> dict:
    return {
        "samples_ms": values,
        "median_ms": float(np.median(values)),
        "p90_ms": float(np.percentile(values, 90)),
        "min_ms": float(np.min(values)),
        "max_ms": float(np.max(values)),
    }


def run(root: Path) -> dict:
    from simulator.forward_engine import run_forward_full
    from simulator.graph_engine import build_graph
    from simulator.layers import LayerConfig

    results = []
    for case_index, (input_width, hidden_width, output_width) in enumerate(CASES):
        np.random.seed(20261014 + case_index)
        layers = [
            LayerConfig("input", neurons=input_width, input_shape=[input_width]),
            LayerConfig("dense", neurons=hidden_width, activation="tanh"),
            LayerConfig("output", neurons=output_width, activation="linear"),
        ]
        graph = build_graph(layers)
        x = np.linspace(-0.75, 0.75, input_width, dtype=np.float32)

        baseline = np.asarray(graph.forward(x.copy()), dtype=np.float64)
        trace_steps, trace_output, _ = run_forward_full(graph, x.tolist())
        traced = np.asarray(trace_output, dtype=np.float64)
        max_abs = (float(np.max(np.abs(baseline - traced))) if baseline.size else 0.0) if baseline.shape == traced.shape else float("inf")
        parity = baseline.shape == traced.shape and np.allclose(baseline, traced, atol=ATOL, rtol=RTOL)
        if not parity:
            results.append({
                "widths": [input_width, hidden_width, output_width],
                "output_parity": False,
                "max_abs_error": max_abs,
                "trace_event_count": len(trace_steps),
                "error": "forward and trace outputs did not agree within declared tolerance",
            })
            continue

        for _ in range(WARMUPS):
            graph.forward(x.copy())
            run_forward_full(graph, x.tolist())

        forward_ms = []
        for _ in range(SAMPLES):
            started = time.perf_counter_ns()
            graph.forward(x.copy())
            forward_ms.append((time.perf_counter_ns() - started) / 1e6)

        trace_ms = []
        for _ in range(SAMPLES):
            started = time.perf_counter_ns()
            run_forward_full(graph, x.tolist())
            trace_ms.append((time.perf_counter_ns() - started) / 1e6)

        forward_summary = summarize(forward_ms)
        trace_summary = summarize(trace_ms)
        results.append({
            "widths": [input_width, hidden_width, output_width],
            "output_parity": bool(parity),
            "max_abs_error": max_abs,
            "trace_event_count": len(trace_steps),
            "forward_only": forward_summary,
            "full_trace": trace_summary,
            "median_trace_to_forward_ratio": (
                trace_summary["median_ms"] / forward_summary["median_ms"]
                if forward_summary["median_ms"] > 0 else None
            ),
            "work_note": "The trace path constructs per-operation records and snapshots; this is a path comparison, not a causal estimate for tracing in general.",
        })

    passed = len(results) == len(CASES) and all(item.get("output_parity") for item in results)
    return {
        "schema_version": 1,
        "experiment": "cross-platform dense forward versus full trace diagnostic",
        "metadata": {
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "commit": commit_for(root),
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python": sys.version,
            "numpy": np.__version__,
            "warmups_per_path": WARMUPS,
            "samples_per_path_per_case": SAMPLES,
            "cases": [list(case) for case in CASES],
            "tolerance": {"atol": ATOL, "rtol": RTOL},
        },
        "results": results,
        "pass": passed,
        "interpretation": "Diagnostic evidence only. Compare distributions and environment metadata; do not pool unlike runners or claim universal overhead.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=".")
    parser.add_argument("--out", default="research/artifacts/trace-benchmark-cross-platform.json")
    args = parser.parse_args()
    root = Path(args.repo).resolve()
    report = run(root)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "commit": report["metadata"]["commit"],
        "system": report["metadata"]["system"],
        "pass": report["pass"],
        "case_count": len(report["results"]),
    }, indent=2))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
