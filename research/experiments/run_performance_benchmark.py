#!/usr/bin/env python3
"""EXP-05 Decomposed performance benchmark.

Purpose
-------
The previously recorded benchmark compared "final-output forward" against "full
trace construction" and reported the ratio. That comparison is still valid as a
description, but it attributes the difference to "tracing" when the trace path
also constructs per-operation records, parameter snapshots and JSON-serialisable
lists. This experiment decomposes the cost so each component can be attributed.

Measured components
------------------
``build``            graph construction from layer configurations
``forward``          layer-instance forward pass returning the final tensor
``trace_total``      run_forward_full
``trace_record_projection``  projecting an existing trace into per-record value lists
``trace_json``       serialisation of the trace records alone
``step_first``       run_forward_step at index 0
``step_middle``      run_forward_step at the middle record
``step_last``        run_forward_step at the final record
``backward``         backward_full
``train_batch``      one TrainingEngine batch

Statistical policy
------------------
Every component is measured with warm-ups followed by repeated raw samples.
Medians, dispersion, percentiles and t-based 95% confidence intervals for the
mean are reported per component per case. Overhead is reported as a paired
per-trial ratio, never as a ratio of two independently summarised statistics.

Comparisons are only made between components that run on the same workload. A
component that does strictly more work is labelled as such.

Usage
-----
    PYTHONPATH=backend:research python research/experiments/run_performance_benchmark.py \
        --repo . --samples 60
"""
from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
import time
import tracemalloc
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib import cases as case_registry  # noqa: E402
from lib import stats as stat  # noqa: E402
from lib.evidence import EvidenceWriter  # noqa: E402
from lib.provenance import RunProvenance, environment_report, utc_now_iso  # noqa: E402

EXPERIMENT_ID = "EXP-05-performance-benchmark"
DEFAULT_SEED = 20261018
DEFAULT_SAMPLES = 60
DEFAULT_WARMUPS = 10

DENSE_WIDTHS = ((8, 16, 4), (16, 32, 8), (32, 64, 16), (64, 128, 32), (128, 256, 64))


def dense_layers(in_dim: int, hidden: int, out_dim: int):
    from simulator.layers import LayerConfig

    return [
        LayerConfig("input", neurons=in_dim, input_shape=[in_dim]),
        LayerConfig("dense", neurons=hidden, activation="tanh"),
        LayerConfig("dense", neurons=hidden, activation="tanh"),
        LayerConfig("output", neurons=out_dim, activation="linear"),
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


def recurrent_layers(steps: int, features: int, hidden: int, family: str = "lstm"):
    from simulator.layers import LayerConfig

    return [
        LayerConfig("input", input_shape=[steps, features]),
        LayerConfig(family, hidden_size=hidden, return_sequences=False),
        LayerConfig("output", neurons=2, activation="linear"),
    ]


# ---------------------------------------------------------------------------
def _time_samples(fn: Callable[[], Any], samples: int, warmups: int) -> List[float]:
    for _ in range(warmups):
        fn()
    collected: List[float] = []
    for _ in range(samples):
        gc.collect()
        started = time.perf_counter_ns()
        fn()
        collected.append((time.perf_counter_ns() - started) / 1e6)
    return collected


def _peak_bytes(fn: Callable[[], Any]) -> Optional[int]:
    gc.collect()
    tracemalloc.start()
    try:
        fn()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return int(peak)


def benchmark_graph(layers, x: np.ndarray, target: np.ndarray, seed: int, samples: int, warmups: int) -> Dict[str, Any]:
    from simulator.backward_engine import backward_full
    from simulator.forward_engine import run_forward_full, run_forward_step
    from simulator.graph_engine import build_graph
    from simulator.training_engine import TrainingConfig, TrainingEngine

    np.random.seed(seed)
    build_samples = _time_samples(lambda: build_graph(layers), samples=max(10, samples // 3), warmups=3)
    graph = build_graph(layers)
    x_list = np.asarray(x).tolist()
    target_list = np.asarray(target).reshape(-1).tolist()

    baseline = np.asarray(graph.forward(np.asarray(x, dtype=np.float32)), dtype=np.float64).reshape(-1)
    steps, trace_output, _ = run_forward_full(graph, x_list)
    traced = np.asarray(trace_output, dtype=np.float64).reshape(-1)
    parity = bool(
        baseline.shape == traced.shape and np.allclose(baseline, traced, atol=1e-5, rtol=1e-5)
    )

    components: Dict[str, Any] = {}

    def timed(name: str, fn: Callable[[], Any]) -> None:
        components[name] = {
            "samples_ms": _time_samples(fn, samples, warmups),
            "peak_traced_bytes": _peak_bytes(fn),
        }

    timed("forward", lambda: graph.forward(np.asarray(x, dtype=np.float32)))
    timed("trace_total", lambda: run_forward_full(graph, x_list))
    timed(
        "trace_record_projection",
        lambda: [
            {
                "step_index": i,
                "layer_index": s.get("layer_index"),
                "operation": s.get("operation"),
                "output_values": list(
                    np.asarray(s.get("output_values", []), dtype=np.float64).reshape(-1)
                ),
            }
            for i, s in enumerate(steps)
        ],
    )
    timed("trace_json", lambda: json.dumps(steps))
    if steps:
        mid = len(steps) // 2
        last = len(steps) - 1
        timed("step_first", lambda: run_forward_step(graph, x_list, 0))
        timed("step_middle", lambda: run_forward_step(graph, x_list, mid))
        timed("step_last", lambda: run_forward_step(graph, x_list, last))
    timed("backward", lambda: backward_full(graph, x_list, target_list, "mse"))

    engine = TrainingEngine(
        graph,
        TrainingConfig(
            epochs=1,
            batch_size=1,
            learning_rate=0.001,
            optimizer="sgd",
            loss_function="mse",
            shuffle=False,
            dropout_rate=0.0,
        ),
    )
    timed("train_batch", lambda: engine.train_batch([(np.asarray(x, dtype=np.float32), np.asarray(target, dtype=np.float32))]))

    summary = {
        name: stat.describe(values["samples_ms"]) for name, values in components.items()
    }
    return {
        "components": components,
        "summary": summary,
        "trace_event_count": len(steps),
        "output_parity": parity,
        "max_abs_parity_error": float(np.max(np.abs(baseline - traced))) if baseline.size else 0.0,
        "trace_payload_bytes": len(json.dumps(steps).encode("utf-8")),
    }


# ---------------------------------------------------------------------------
def build_workloads(samples: int) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(DEFAULT_SEED)
    workloads: List[Dict[str, Any]] = []
    for in_dim, hidden, out_dim in DENSE_WIDTHS:
        layers = dense_layers(in_dim, hidden, out_dim)
        x = rng.uniform(-0.8, 0.8, size=in_dim).astype(np.float32)
        target = rng.uniform(0.2, 0.8, size=out_dim).astype(np.float32)
        workloads.append(
            {
                "workload_id": f"dense_{in_dim}_{hidden}_{out_dim}",
                "family": "dense",
                "layers": layers,
                "x": x,
                "target": target,
                "shape_signature": [in_dim, hidden, hidden, out_dim],
            }
        )
    for channels, filters, spatial in ((1, 8, 16), (1, 16, 32), (2, 32, 32)):
        layers = conv_layers(channels, filters, spatial)
        x = rng.uniform(-0.8, 0.8, size=(channels, spatial, spatial)).astype(np.float32)
        target = rng.uniform(0.2, 0.8, size=8).astype(np.float32)
        workloads.append(
            {
                "workload_id": f"conv_{channels}x{filters}_{spatial}",
                "family": "conv",
                "layers": layers,
                "x": x,
                "target": target,
                "shape_signature": [channels, filters, spatial, spatial],
            }
        )
    for steps, features, hidden in ((8, 16, 16), (32, 32, 32), (64, 64, 64)):
        for family in ("lstm", "gru"):
            layers = recurrent_layers(steps, features, hidden, family)
            x = rng.uniform(-0.8, 0.8, size=(steps, features)).astype(np.float32)
            target = rng.uniform(0.2, 0.8, size=2).astype(np.float32)
            workloads.append(
                {
                    "workload_id": f"{family}_t{steps}_d{features}_h{hidden}",
                    "family": family,
                    "layers": layers,
                    "x": x,
                    "target": target,
                    "shape_signature": [steps, features, hidden],
                }
            )
    return workloads


def run(repo_root: Path, samples: int, warmups: int, seed: int) -> Dict[str, Any]:
    workloads = build_workloads(samples)
    records: List[Dict[str, Any]] = []
    for workload in workloads:
        record: Dict[str, Any] = {
            "workload_id": workload["workload_id"],
            "family": workload["family"],
            "shape_signature": workload["shape_signature"],
        }
        try:
            result = benchmark_graph(
                workload["layers"], workload["x"], workload["target"], seed, samples, warmups
            )
            record.update(result)
            record["status"] = "MEASURED"
        except Exception as exc:  # noqa: BLE001
            record["status"] = "ERROR"
            record["exception"] = f"{type(exc).__name__}: {exc}"
            record["traceback"] = traceback.format_exc(limit=6)
        records.append(record)

    overheads: List[Dict[str, Any]] = []
    for record in records:
        if record.get("status") != "MEASURED":
            continue
        components = record["components"]
        forward = components["forward"]["samples_ms"]
        trace_total = components["trace_total"]["samples_ms"]
        ratios = stat.ratio_stats(trace_total, forward)
        records_component = components.get("trace_record_projection", {}).get("samples_ms")
        json_samples = components.get("trace_json", {}).get("samples_ms")
        overheads.append(
            {
                "workload_id": record["workload_id"],
                "family": record["family"],
                "shape_signature": record["shape_signature"],
                "forward_median_ms": components["forward"]["samples_ms"]
                and stat.describe(forward)["median"],
                "trace_total_median_ms": stat.describe(trace_total)["median"],
                "trace_record_projection_median_ms": stat.describe(records_component)["median"]
                if records_component
                else None,
                "trace_json_median_ms": stat.describe(json_samples)["median"] if json_samples else None,
                "paired_ratio": ratios["paired_ratio"],
                "ratio_of_medians": ratios["ratio_of_medians"],
                "n_pairs": ratios["n_pairs"],
                "effect_size_paired": stat.effect_size_paired(trace_total, forward),
                "output_parity": record["output_parity"],
            }
        )

    measured = [r for r in records if r.get("status") == "MEASURED"]
    parity_failures = [r["workload_id"] for r in measured if not r["output_parity"]]
    summary = {
        "workloads_measured": len(measured),
        "workloads_failed": len(records) - len(measured),
        "output_parity_failures": parity_failures,
        "samples_per_component": samples,
        "warmups_per_component": warmups,
        "median_forward_ms_by_workload": {
            r["workload_id"]: stat.describe(r["components"]["forward"]["samples_ms"])["median"]
            for r in measured
        },
        "median_trace_total_ms_by_workload": {
            r["workload_id"]: stat.describe(r["components"]["trace_total"]["samples_ms"])["median"]
            for r in measured
        },
        "paired_trace_overhead_ratio": stat.describe(
            [o["paired_ratio"]["median"] for o in overheads]
        ),
        "effect_size_trace_vs_forward": stat.describe(
            [o["effect_size_paired"]["cohens_dz"] for o in overheads if o["effect_size_paired"]]
        ),
        "status": "PASS" if len(measured) == len(records) and not parity_failures else "FINDINGS_PRESENT",
    }
    return {"records": records, "overheads": overheads, "summary": summary}


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
        notes={"runtime_source_modified": False, "environment_role": "primary_local_environment"},
    )
    try:
        result = run(repo_root, args.samples, args.warmups, args.seed)
    except Exception as exc:  # noqa: BLE001
        print(f"performance benchmark failed: {exc}", file=sys.stderr)
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
            "samples_per_component": args.samples,
            "warmups_per_component": args.warmups,
            "dense_widths": [list(w) for w in DENSE_WIDTHS],
            "components": [
                "build",
                "forward",
                "trace_total",
                "trace_record_projection",
                "trace_json",
                "step_first",
                "step_middle",
                "step_last",
                "backward",
                "train_batch",
            ],
            "work_note": (
                "trace_total constructs per-operation records with parameter snapshots "
                "and returns JSON-serialisable lists. It therefore performs strictly more "
                "work than 'forward' and its ratio must not be described as the causal "
                "cost of tracing in general."
            ),
        },
        summary_payload=result["summary"],
        extra_raw={
            "workload_records": result["records"],
            "overheads": result["overheads"],
            "environment": environment_report(),
        },
        tables={
            "component_medians": [
                {
                    "workload_id": r["workload_id"],
                    "family": r["family"],
                    "component": component,
                    "median_ms": stat.describe(values["samples_ms"])["median"],
                    "mean_ms": stat.describe(values["samples_ms"])["mean"],
                    "std_ms": stat.describe(values["samples_ms"])["std"],
                    "min_ms": stat.describe(values["samples_ms"])["min"],
                    "max_ms": stat.describe(values["samples_ms"])["max"],
                    "ci95_low": (stat.describe(values["samples_ms"])["confidence_interval_95_mean"] or {}).get("low"),
                    "ci95_high": (stat.describe(values["samples_ms"])["confidence_interval_95_mean"] or {}).get("high"),
                    "peak_traced_bytes": values["peak_traced_bytes"],
                    "samples": len(values["samples_ms"]),
                }
                for r in result["records"]
                if r.get("status") == "MEASURED"
                for component, values in r["components"].items()
            ],
            "trace_overhead": [
                {
                    "workload_id": o["workload_id"],
                    "family": o["family"],
                    "shape_signature": o["shape_signature"],
                    "forward_median_ms": o["forward_median_ms"],
                    "trace_total_median_ms": o["trace_total_median_ms"],
                    "trace_record_projection_median_ms": o["trace_record_projection_median_ms"],
                    "trace_json_median_ms": o["trace_json_median_ms"],
                    "paired_ratio_median": (o["paired_ratio"] or {}).get("median"),
                    "paired_ratio_mean": (o["paired_ratio"] or {}).get("mean"),
                    "paired_ratio_ci95_low": ((o["paired_ratio"] or {}).get("confidence_interval_95_mean") or {}).get("low"),
                    "paired_ratio_ci95_high": ((o["paired_ratio"] or {}).get("confidence_interval_95_mean") or {}).get("high"),
                    "ratio_of_medians": o["ratio_of_medians"],
                    "cohens_dz": (o["effect_size_paired"] or {}).get("cohens_dz"),
                    "output_parity": o["output_parity"],
                }
                for o in result["overheads"]
            ],
            "component_samples": [
                {
                    "workload_id": r["workload_id"],
                    "component": component,
                    "sample_index": index,
                    "milliseconds": value,
                }
                for r in result["records"]
                if r.get("status") == "MEASURED"
                for component, values in r["components"].items()
                for index, value in enumerate(values["samples_ms"])
            ],
        },
        interpretation={
            "purpose": "Decomposed, repeated-measurement cost characterisation of the simulator paths.",
            "overhead_definition": (
                "Overhead is reported as the distribution of per-trial paired ratios "
                "with its median, dispersion and a t-based 95% interval. The "
                "ratio-of-medians is reported only as a descriptive point estimate."
            ),
            "attribution": (
                "trace_record_projection and trace_json separate value-list projection "
                "from JSON serialisation, so the residual can be attributed rather than "
                "assumed. trace_record_projection is a lower bound on record construction "
                "because run_forward_full builds the same structures inside its own timing."
            ),
            "measurement": (
                "Memory is reported as tracemalloc's peak traced Python allocation, "
                "which is not resident set size and not a device-memory measurement."
            ),
        },
        limitations=[
            "Single-machine, single-process measurements. Hosted CI runners and this "
            "workstation are different environments and their absolute timings must not be pooled.",
            "tracemalloc measures Python-level allocations only; NumPy buffers and the "
            "interpreter runtime are outside its scope.",
            "No frontend rendering, HTTP transport or WebSocket cost is included; those "
            "measure a different amount of work on a different path.",
            "Warm-ups and garbage collection are controlled, but CPU frequency scaling "
            "and background load are not pinned.",
        ],
    )
    manifest = writer.write()
    summary = result["summary"]
    print(
        f"[{EXPERIMENT_ID}] workloads={summary['workloads_measured']} "
        f"parity_failures={len(summary['output_parity_failures'])} "
        f"median_paired_ratio={summary['paired_trace_overhead_ratio']['median']} "
        f"status={summary['status']}"
    )
    print(f"evidence: research/evidence/{EXPERIMENT_ID}/{manifest['source_sha']}")
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())