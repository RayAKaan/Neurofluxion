#!/usr/bin/env python3
"""Diagnostic audit of graph-integrated backward paths; gaps are reported, not hidden."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np


def commit_for(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cases(LayerConfig):
    yield "dense", [
        LayerConfig("input", neurons=3, input_shape=[3]),
        LayerConfig("dense", neurons=4, activation="tanh"),
        LayerConfig("output", neurons=2, activation="linear"),
    ], np.array([0.2, -0.4, 0.7], dtype=np.float32)
    yield "conv-maxpool-flatten", [
        LayerConfig("input", neurons=16, input_shape=[1, 4, 4]),
        LayerConfig("conv2d", neurons=2, filters=2, kernel_size=3, padding="same", activation="relu"),
        LayerConfig("maxpool2d", neurons=2, pool_size=2),
        LayerConfig("flatten", neurons=8),
        LayerConfig("output", neurons=2, activation="linear"),
    ], np.arange(16, dtype=np.float32).reshape(1, 4, 4) / 10.0
    yield "batchnorm", [
        LayerConfig("input", neurons=3, input_shape=[3]),
        LayerConfig("batchnorm", neurons=3),
        LayerConfig("output", neurons=2, activation="linear"),
    ], np.array([0.2, -0.4, 0.7], dtype=np.float32)
    for kind in ("rnn", "lstm", "gru"):
        yield kind, [
            LayerConfig("input", neurons=6, input_shape=[3, 2]),
            LayerConfig(kind, neurons=3, hidden_size=3, return_sequences=False),
            LayerConfig("output", neurons=2, activation="linear"),
        ], np.array([[0.1, 0.2], [-0.3, 0.4], [0.5, -0.2]], dtype=np.float32)
    yield "embedding-attention-rnn", [
        LayerConfig("input", neurons=3, input_shape=[3]),
        LayerConfig("embedding", neurons=2, embedding_dim=2, vocab_size=5),
        LayerConfig("attention", neurons=2, num_heads=1),
        LayerConfig("rnn", neurons=3, hidden_size=3, return_sequences=False),
        LayerConfig("output", neurons=2, activation="linear"),
    ], np.array([1, 2, 1], dtype=np.float32)
    yield "residual", [
        LayerConfig("input", neurons=3, input_shape=[3]),
        LayerConfig("dense", neurons=3, activation="tanh"),
        LayerConfig("residual", neurons=3),
        LayerConfig("output", neurons=2, activation="linear"),
    ], np.array([0.2, -0.4, 0.7], dtype=np.float32)


def audit(root: Path) -> dict:
    from simulator.backward_engine import backward_full
    from simulator.graph_engine import build_graph
    from simulator.layers import LayerConfig, validate_layers

    results = []
    for index, (name, configs, x) in enumerate(cases(LayerConfig)):
        np.random.seed(20261014 + index)
        result = {
            "case": name,
            "status": "GAP",
            "output_shape": None,
            "gradient_tensor_count": 0,
            "all_gradients_finite": False,
        }
        try:
            validation = validate_layers(configs)
            if not validation.valid:
                result["reason"] = "invalid graph config: " + "; ".join(validation.errors)
            else:
                graph = build_graph(configs)
                yhat = np.asarray(graph.forward(x.copy()), dtype=np.float32).reshape(-1)
                target = np.linspace(-0.2, 0.2, yhat.size, dtype=np.float32)
                backward = backward_full(graph, x.reshape(-1).tolist(), target.tolist(), "mse")
                tensors = []
                for key in ("gradients_W", "gradients_b"):
                    tensors.extend(np.asarray(value, dtype=np.float64) for value in backward.get(key, []))
                input_gradient = np.asarray(backward.get("input_gradient", []), dtype=np.float64)
                finite = bool(
                    tensors
                    and all(np.all(np.isfinite(value)) for value in tensors)
                    and np.all(np.isfinite(input_gradient))
                    and np.isfinite(float(backward["loss_value"]))
                )
                result.update({
                    "status": "PASS" if finite else "GAP",
                    "output_shape": list(yhat.shape),
                    "gradient_tensor_count": len(tensors),
                    "all_gradients_finite": finite,
                    "loss_value": float(backward["loss_value"]),
                    "input_gradient_size": int(input_gradient.size),
                })
                if not finite:
                    result["reason"] = "empty gradient set or non-finite loss/gradient"
        except Exception as exc:
            result["reason"] = f"{type(exc).__name__}: {exc}"
        results.append(result)

    audit_paths = [
        root / "backend/simulator/backward_engine.py",
        root / "backend/simulator/training_engine.py",
        root / "backend/simulator/optimizer_engine.py",
        root / "backend/simulator/graph_engine.py",
    ]
    return {
        "schema_version": 1,
        "audit_kind": "fixed-case graph-integrated backward diagnostic; not finite-difference proof",
        "metadata": {
            "commit": commit_for(root),
            "platform": platform.platform(),
            "python": sys.version,
            "numpy": np.__version__,
            "source_sha256": {str(path.relative_to(root)): digest(path) for path in audit_paths},
        },
        "summary": {
            "cases": len(results),
            "pass": sum(item["status"] == "PASS" for item in results),
            "gaps": sum(item["status"] == "GAP" for item in results),
        },
        "results": results,
        "interpretation": (
            "PASS means the selected graph's backward path returned finite gradients for this fixed input/target. "
            "It does not prove gradient correctness; independent finite differences or analytic oracles are required. "
            "GAP entries are retained as findings. A successful process exit means only that the report was generated."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=".")
    parser.add_argument("--out", default="research/artifacts/graph-backward-audit.json")
    args = parser.parse_args()
    root = Path(args.repo).resolve()
    report = audit(root)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), **report["summary"], "commit": report["metadata"]["commit"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
