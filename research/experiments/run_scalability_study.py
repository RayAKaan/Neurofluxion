#!/usr/bin/env python3
"""EXP-06 Controlled scalability study.

Purpose
-------
Replace ad-hoc "larger model" observations with controlled one-factor-at-a-time
sweeps, each measured with repeated trials, and report the fitted scaling
exponent *only inside the tested regime*.

Sweeps
------
``depth``        number of hidden dense layers (1, 2, 4, 8, 16)
``width``        hidden dense width (8 ... 256)
``sequence``     recurrent sequence length (4 ... 128)
``features``     recurrent input feature count (8 ... 128)
``spatial``      convolutional input spatial size (8 ... 64)
``filters``      convolutional filter count (4 ... 64)
``complexity``   graph nodes, parameterised layers, residual branches,
                 recurrent timesteps, trace records and trace payload bytes
``batch``        batch dimension, recorded as a declared boundary probe

Scaling treatment
-----------------
For each sweep the median of each measured quantity is regressed against the
sweep variable in log-log space. The fitted slope is reported as a descriptive
exponent for the tested range together with the R-squared of the fit and the
tested range, so no claim is made outside it.

Usage
-----
    PYTHONPATH=backend:research python research/experiments/run_scalability_study.py --repo .
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
import tracemalloc
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib import stats as stat  # noqa: E402
from lib.evidence import EvidenceWriter  # noqa: E402
from lib.provenance import RunProvenance, environment_report, utc_now_iso  # noqa: E402

EXPERIMENT_ID = "EXP-06-scalability-study"
DEFAULT_SEED = 20261018
DEFAULT_SAMPLES = 25
DEFAULT_WARMUPS = 5

DEPTHS = (1, 2, 4, 8, 10)
DEPTH_BOUNDARY_PROBE = 16
WIDTHS = (8, 16, 32, 64, 128, 256)
SEQUENCES = (4, 8, 16, 32, 64, 128)
FEATURES = (8, 16, 32, 64, 128)
SPATIALS = (8, 16, 32, 64)
FILTERS = (4, 8, 16, 32, 64)
BATCHES = (1, 2, 4, 8)


def dense_depth_layers(in_dim: int, hidden: int, depth: int, out_dim: int):
    from simulator.layers import LayerConfig

    layers = [LayerConfig("input", neurons=in_dim, input_shape=[in_dim])]
    for _ in range(depth):
        layers.append(LayerConfig("dense", neurons=hidden, activation="tanh"))
    layers.append(LayerConfig("output", neurons=out_dim, activation="linear"))
    return layers


def recurrent_layers(steps: int, features: int, hidden: int, family: str = "lstm"):
    from simulator.layers import LayerConfig

    return [
        LayerConfig("input", input_shape=[steps, features]),
        LayerConfig(family, hidden_size=hidden, return_sequences=False),
        LayerConfig("output", neurons=2, activation="linear"),
    ]


def conv_layers(channels: int, filters: int, spatial: int):
    from simulator.layers import LayerConfig

    return [
        LayerConfig("input", input_shape=[channels, spatial, spatial]),
        LayerConfig("conv2d", filters=filters, kernel_size=3, stride=1, padding="same", activation="relu"),
        LayerConfig("maxpool2d", pool_size=2),
        LayerConfig("flatten"),
        LayerConfig("output", neurons=8, activation="linear"),
    ]


# ---------------------------------------------------------------------------
def _measure(fn: Callable[[], Any], samples: int, warmups: int) -> List[float]:
    for _ in range(warmups):
        fn()
    out: List[float] = []
    for _ in range(samples):
        gc.collect()
        started = time.perf_counter_ns()
        fn()
        out.append((time.perf_counter_ns() - started) / 1e6)
    return out


def _peak(fn: Callable[[], Any]) -> Optional[int]:
    gc.collect()
    tracemalloc.start()
    try:
        fn()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return int(peak)


def measure_point(layers, x: np.ndarray, seed: int, samples: int, warmups: int) -> Dict[str, Any]:
    from simulator.forward_engine import run_forward_full
    from simulator.graph_engine import build_graph

    np.random.seed(seed)
    graph = build_graph(layers)
    x_list = np.asarray(x).tolist()
    steps, output, _ = run_forward_full(graph, x_list)
    payload = json.dumps(steps)
    return {
        "forward_samples_ms": _measure(lambda: graph.forward(np.asarray(x, dtype=np.float32)), samples, warmups),
        "trace_samples_ms": _measure(lambda: run_forward_full(graph, x_list), samples, warmups),
        "forward_peak_bytes": _peak(lambda: graph.forward(np.asarray(x, dtype=np.float32))),
        "trace_peak_bytes": _peak(lambda: run_forward_full(graph, x_list)),
        "trace_records": len(steps),
        "trace_payload_bytes": len(payload.encode("utf-8")),
        "packed_parameter_scalars": int(
            sum(int(np.size(w)) for w in graph.weights) + sum(int(np.size(b)) for b in graph.biases)
        ),
        "graph_nodes": len(graph.layer_instances),
        "parameterised_layers": len(graph.param_layer_indices),
        "residual_branches": sum(1 for layer in graph.layer_instances if getattr(layer, "layer_type", "") == "residual"),
        "recurrent_timesteps": int(np.shape(x)[0]) if np.ndim(x) == 2 else 0,
        "output_size": int(np.size(output)),
        "input_elements": int(np.size(x)),
        "total_params_declared": graph.total_params,
    }


def fit_exponent(xs: Sequence[float], ys: Sequence[float]) -> Dict[str, Any]:
    """Least-squares slope in log-log space, reported only for the tested range."""
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    mask = (x > 0) & (y > 0)
    if int(mask.sum()) < 3:
        return {"fitted": False, "reason": "fewer than three positive points"}
    spread = float(np.max(y[mask]) - np.min(y[mask])) / float(np.max(y[mask]) + 1e-15)
    if spread < 1e-9:
        # A quantity that does not vary with the sweep factor has no exponent;
        # reporting a fitted slope for it would manufacture a scaling claim.
        return {
            "fitted": False,
            "reason": "quantity is invariant across the sweep factor",
            "constant_value": float(y[mask][0]),
            "points": int(mask.sum()),
            "tested_range": [float(np.min(x[mask])), float(np.max(x[mask]))],
        }
    lx, ly = np.log(x[mask]), np.log(y[mask])
    slope, intercept = np.polyfit(lx, ly, 1)
    predicted = slope * lx + intercept
    ss_res = float(np.sum((ly - predicted) ** 2))
    ss_tot = float(np.sum((ly - np.mean(ly)) ** 2))
    return {
        "fitted": True,
        "exponent": float(slope),
        "r_squared": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else None,
        "points": int(mask.sum()),
        "tested_range": [float(np.min(x[mask])), float(np.max(x[mask]))],
        "scope_note": "Descriptive exponent for the tested range only; no extrapolation is claimed.",
    }


# ---------------------------------------------------------------------------
def sweep(name: str, points: List[Dict[str, Any]], samples: int, warmups: int, seed: int) -> Dict[str, Any]:
    results: List[Dict[str, Any]] = []
    for point in points:
        record = {"sweep": name, **point}
        try:
            record.update(
                {
                    "measurements": measure_point(
                        point["layers"], point["x"], seed + point.get("value", 0), samples, warmups
                    ),
                    "status": "MEASURED",
                }
            )
        except Exception as exc:  # noqa: BLE001
            record["status"] = "ERROR"
            record["exception"] = f"{type(exc).__name__}: {exc}"
        record.pop("layers", None)
        record.pop("x", None)
        results.append(record)

    measured = [r for r in results if r["status"] == "MEASURED"]
    fits: Dict[str, Any] = {}
    if len(measured) >= 3:
        xs = [float(r["value"]) for r in measured]
        fits = {
            "forward_ms": fit_exponent(xs, [stat.describe(r["measurements"]["forward_samples_ms"])["median"] for r in measured]),
            "trace_ms": fit_exponent(xs, [stat.describe(r["measurements"]["trace_samples_ms"])["median"] for r in measured]),
            "trace_payload_bytes": fit_exponent(xs, [r["measurements"]["trace_payload_bytes"] for r in measured]),
            "trace_records": fit_exponent(xs, [r["measurements"]["trace_records"] for r in measured]),
            "packed_parameter_scalars": fit_exponent(xs, [r["measurements"]["packed_parameter_scalars"] for r in measured]),
        }
    return {"sweep": name, "points": results, "fits": fits, "points_measured": len(measured)}


def run(repo_root: Path, samples: int, warmups: int, seed: int) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    sweeps: List[Dict[str, Any]] = []

    sweeps.append(
        sweep(
            "depth",
            [
                {
                    "value": d,
                    "layers": dense_depth_layers(32, 32, d, 8),
                    "x": rng.uniform(-0.8, 0.8, size=32).astype(np.float32),
                }
                for d in DEPTHS
            ],
            samples,
            warmups,
            seed,
        )
    )
    sweeps.append(
        sweep(
            "width",
            [
                {
                    "value": w,
                    "layers": dense_depth_layers(32, w, 2, 8),
                    "x": rng.uniform(-0.8, 0.8, size=32).astype(np.float32),
                }
                for w in WIDTHS
            ],
            samples,
            warmups,
            seed,
        )
    )
    sweeps.append(
        sweep(
            "sequence",
            [
                {
                    "value": t,
                    "layers": recurrent_layers(t, 32, 32, "lstm"),
                    "x": rng.uniform(-0.8, 0.8, size=(t, 32)).astype(np.float32),
                }
                for t in SEQUENCES
            ],
            samples,
            warmups,
            seed,
        )
    )
    sweeps.append(
        sweep(
            "features",
            [
                {
                    "value": d,
                    "layers": recurrent_layers(16, d, 32, "gru"),
                    "x": rng.uniform(-0.8, 0.8, size=(16, d)).astype(np.float32),
                }
                for d in FEATURES
            ],
            samples,
            warmups,
            seed,
        )
    )
    sweeps.append(
        sweep(
            "spatial",
            [
                {
                    "value": s,
                    "layers": conv_layers(1, 8, s),
                    "x": rng.uniform(-0.8, 0.8, size=(1, s, s)).astype(np.float32),
                }
                for s in SPATIALS
            ],
            samples,
            warmups,
            seed,
        )
    )
    sweeps.append(
        sweep(
            "filters",
            [
                {
                    "value": f,
                    "layers": conv_layers(1, f, 16),
                    "x": rng.uniform(-0.8, 0.8, size=(1, 16, 16)).astype(np.float32),
                }
                for f in FILTERS
            ],
            samples,
            warmups,
            seed,
        )
    )

    # batch dimension: a declared boundary probe rather than a scaling sweep
    batch_points = []
    for b in BATCHES:
        layers = dense_depth_layers(16, 16, 1, 4)
        record = {"sweep": "batch", "value": b}
        try:
            np.random.seed(seed + b)
            from simulator.graph_engine import build_graph

            graph = build_graph(layers)
            x = rng.uniform(-0.8, 0.8, size=(b, 16)).astype(np.float32)
            out = graph.forward(x.copy())
            record.update({"status": "SUPPORTED", "output_shape": list(np.shape(out))})
        except Exception as exc:  # noqa: BLE001
            record.update({"status": "UNSUPPORTED", "exception": f"{type(exc).__name__}: {exc}"})
        batch_points.append(record)
    sweeps.append({"sweep": "batch", "points": batch_points, "fits": {}, "points_measured": len(batch_points)})

    # depth beyond the validator's layer-count ceiling: a declared boundary probe
    depth_boundary = []
    for depth in (DEPTH_BOUNDARY_PROBE,):
        record = {"sweep": "depth_boundary", "value": depth}
        try:
            np.random.seed(seed)
            from simulator.graph_engine import build_graph

            build_graph(dense_depth_layers(32, 32, depth, 8))
            record["status"] = "SUPPORTED"
        except Exception as exc:  # noqa: BLE001
            record.update({"status": "UNSUPPORTED", "exception": f"{type(exc).__name__}: {exc}"})
        depth_boundary.append(record)
    sweeps.append(
        {"sweep": "depth_boundary", "points": depth_boundary, "fits": {}, "points_measured": len(depth_boundary)}
    )

    total_points = sum(len(s["points"]) for s in sweeps)
    measured_points = sum(s["points_measured"] for s in sweeps)
    errors = [
        {"sweep": s["sweep"], "value": p["value"], "exception": p.get("exception")}
        for s in sweeps
        for p in s["points"]
        if p["status"] == "ERROR"
    ]
    summary = {
        "sweeps": [s["sweep"] for s in sweeps],
        "points_total": total_points,
        "points_measured": measured_points,
        "points_error": len(errors),
        "errors": errors,
        "fitted_exponents": {s["sweep"]: s["fits"] for s in sweeps if s["fits"]},
        "batch_boundary": {
            p["value"]: p["status"] for p in next(s for s in sweeps if s["sweep"] == "batch")["points"]
        },
        "depth_boundary": {
            p["value"]: p["status"]
            for p in next(s for s in sweeps if s["sweep"] == "depth_boundary")["points"]
        },
        "depth_boundary_detail": {
            p["value"]: p.get("exception")
            for p in next(s for s in sweeps if s["sweep"] == "depth_boundary")["points"]
        },
        "status": "PASS" if not errors else "FINDINGS_PRESENT",
    }
    return {"sweeps": sweeps, "summary": summary}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=".")
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--warmups", type=int, default=DEFAULT_WARMUPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--allow-overwrite", action="store_true")
    args = parser.parse_args()

    repo_root = Path(args.repo).resolve()
    provenance = RunProvenance(
        experiment_id=EXPERIMENT_ID,
        script=Path(__file__).as_posix(),
        command=[sys.executable, *sys.argv],
        seed=args.seed,
        root=repo_root,
        notes={"runtime_source_modified": False},
    )
    try:
        result = run(repo_root, args.samples, args.warmups, args.seed)
    except Exception as exc:  # noqa: BLE001
        print(f"scalability study failed: {exc}", file=sys.stderr)
        traceback.print_exc()
        provenance.exit_code = 2
        provenance.finished_at_utc = utc_now_iso()
        return 2

    provenance.exit_code = 0 if result["summary"]["status"] == "PASS" else 1
    provenance.finished_at_utc = utc_now_iso()

    writer = EvidenceWriter(
        experiment_id=EXPERIMENT_ID,
        repo_root=repo_root,
        provenance=provenance,
        seed=args.seed,
        allow_overwrite=args.allow_overwrite,
        configuration={
            "samples_per_measurement": args.samples,
            "warmups_per_measurement": args.warmups,
            "depth": list(DEPTHS),
            "width": list(WIDTHS),
            "sequence": list(SEQUENCES),
            "features": list(FEATURES),
            "spatial": list(SPATIALS),
            "filters": list(FILTERS),
            "batch": list(BATCHES),
        },
        summary_payload=result["summary"],
        extra_raw={"sweeps": result["sweeps"], "environment": environment_report()},
        tables={
            "scaling_points": [
                {
                    "sweep": point.get("sweep"),
                    "value": point.get("value"),
                    "status": point.get("status"),
                    "forward_median_ms": (stat.describe(point["measurements"]["forward_samples_ms"])["median"] if point.get("measurements") else None),
                    "forward_ci95_low": ((stat.describe(point["measurements"]["forward_samples_ms"])["confidence_interval_95_mean"] or {}).get("low") if point.get("measurements") else None),
                    "forward_ci95_high": ((stat.describe(point["measurements"]["forward_samples_ms"])["confidence_interval_95_mean"] or {}).get("high") if point.get("measurements") else None),
                    "trace_median_ms": (stat.describe(point["measurements"]["trace_samples_ms"])["median"] if point.get("measurements") else None),
                    "trace_peak_bytes": (point["measurements"]["trace_peak_bytes"] if point.get("measurements") else None),
                    "forward_peak_bytes": (point["measurements"]["forward_peak_bytes"] if point.get("measurements") else None),
                    "trace_records": (point["measurements"]["trace_records"] if point.get("measurements") else None),
                    "trace_payload_bytes": (point["measurements"]["trace_payload_bytes"] if point.get("measurements") else None),
                    "packed_parameter_scalars": (point["measurements"]["packed_parameter_scalars"] if point.get("measurements") else None),
                    "graph_nodes": (point["measurements"]["graph_nodes"] if point.get("measurements") else None),
                    "parameterised_layers": (point["measurements"]["parameterised_layers"] if point.get("measurements") else None),
                    "recurrent_timesteps": (point["measurements"]["recurrent_timesteps"] if point.get("measurements") else None),
                    "input_elements": (point["measurements"]["input_elements"] if point.get("measurements") else None),
                    "exception": point.get("exception"),
                }
                for sweep_record in result["sweeps"]
                for point in sweep_record["points"]
            ],
            "scaling_exponents": [
                {"sweep": sweep_name, "quantity": quantity, **fit}
                for sweep_name, fits in result["summary"]["fitted_exponents"].items()
                for quantity, fit in fits.items()
            ],
        },
        interpretation={
            "purpose": "One-factor-at-a-time scaling characterisation inside the tested regime.",
            "exponent_semantics": (
                "A fitted log-log slope describes how the median of a measured quantity "
                "changed across the tested values of one factor. It is descriptive, is "
                "reported with its R-squared and tested range, and is not extrapolated."
            ),
            "batch_dimension": (
                "The batch dimension is recorded as a declared boundary probe: the dense "
                "layer flattens its input, so batched dense execution is not supported "
                "and no batch scaling curve exists."
            ),
        },
        limitations=[
            "Each sweep varies one factor only; interactions between factors are not measured.",
            "Fitted exponents describe the tested range on this machine and environment only.",
            "Peak memory is tracemalloc's Python allocation peak, not resident set size.",
            "No frontend, HTTP transport or concurrent-load behaviour is measured.",
        ],
    )
    manifest = writer.write()
    summary = result["summary"]
    print(
        f"[{EXPERIMENT_ID}] sweeps={len(summary['sweeps'])} "
        f"points={summary['points_measured']}/{summary['points_total']} "
        f"errors={summary['points_error']} status={summary['status']}"
    )
    print(f"evidence: research/evidence/{EXPERIMENT_ID}/{manifest['source_sha']}")
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())