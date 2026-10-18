#!/usr/bin/env python3
"""EXP-07 Determinism and reproducibility study.

Purpose
-------
Establish what is reproducible, at what granularity, and where reproducibility
stops. "Repeatable" is not a single property: a system can be bit-reproducible
within one process, reproducible across processes, and still vary across
machines. Each level is measured separately.

Levels measured
---------------
``in_process_build``      identical layer configuration + seed -> identical parameters
``in_process_forward``    identical parameters + input -> bitwise identical output
``in_process_trace``      identical parameters + input -> identical serialised trace
``in_process_training``   fixed seed, fixed task -> identical loss trajectory
``cross_process_build``   new interpreter -> identical parameters
``cross_process_trace``   new interpreter -> identical serialised trace
``seed_sensitivity``      how much a different seed changes the result

Usage
-----
    PYTHONPATH=backend:research python research/experiments/run_determinism_study.py --repo .
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib import cases as case_registry  # noqa: E402
from lib import stats as stat  # noqa: E402
from lib.evidence import EvidenceWriter  # noqa: E402
from lib.numerics import graph_forward  # noqa: E402
from lib.provenance import RunProvenance, environment_report, utc_now_iso  # noqa: E402

EXPERIMENT_ID = "EXP-07-determinism-study"
DEFAULT_SEED = 20261018
REPEATS = 5

CROSS_PROCESS_SNIPPET = """
import hashlib, json, sys
import numpy as np
sys.path.insert(0, {backend!r})
from simulator.graph_engine import build_graph
from simulator.forward_engine import run_forward_full
from simulator.layers import LayerConfig

case_id = {case_id!r}
seed = {seed!r}
if case_id == "dense_3_4_2_relu":
    layers = [
        LayerConfig("input", neurons=3, input_shape=[3]),
        LayerConfig("dense", neurons=4, activation="relu"),
        LayerConfig("output", neurons=2, activation="linear"),
    ]
    x = np.linspace(-0.7, 0.7, 3, dtype=np.float32)
elif case_id == "lstm_t4_d3_h3_dense":
    layers = [
        LayerConfig("input", input_shape=[4, 3]),
        LayerConfig("lstm", hidden_size=3, return_sequences=False),
        LayerConfig("output", neurons=2, activation="linear"),
    ]
    x = np.random.default_rng(0).uniform(-0.8, 0.8, size=(4, 3)).astype(np.float32)
else:
    layers = [
        LayerConfig("input", input_shape=[1, 6, 6]),
        LayerConfig("conv2d", filters=2, kernel_size=3, stride=2, padding="same", activation="tanh"),
        LayerConfig("batchnorm"),
        LayerConfig("flatten"),
        LayerConfig("output", neurons=1, activation="linear"),
    ]
    x = np.random.default_rng(0).uniform(-0.8, 0.8, size=(1, 6, 6)).astype(np.float32)

np.random.seed(seed)
graph = build_graph(layers)
weights = hashlib.sha256(
    b"".join(np.ascontiguousarray(w).tobytes() for w in graph.weights)
    + b"".join(np.ascontiguousarray(b).tobytes() for b in graph.biases)
).hexdigest()
steps, output, _ = run_forward_full(graph, np.asarray(x).tolist())
print(json.dumps({{"weights_sha256": weights, "trace_sha256": hashlib.sha256(json.dumps(steps, sort_keys=True).encode()).hexdigest()}}))
"""


def digest_arrays(arrays: List[np.ndarray]) -> str:
    hasher = hashlib.sha256()
    for array in arrays:
        hasher.update(np.ascontiguousarray(np.asarray(array, dtype=np.float32)).tobytes())
    return hasher.hexdigest()


def in_process_checks(case, seed: int, repeats: int) -> Dict[str, Any]:
    from simulator.forward_engine import run_forward_full
    from simulator.graph_engine import build_graph
    from simulator.training_engine import TrainingConfig, TrainingEngine
    from simulator.loss_functions import compute_loss

    x = case.make_input()
    results: Dict[str, Any] = {"case_id": case.case_id, "repeats": repeats}

    build_hashes = []
    for _ in range(repeats):
        # The same seed must be used every time: determinism is about the same
        # inputs producing the same state, not about different seeds colliding.
        np.random.seed(seed)
        graph = build_graph(list(case.layers))
        build_hashes.append(
            digest_arrays([np.asarray(w) for w in graph.weights] + [np.asarray(b) for b in graph.biases])
        )
    results["in_process_build"] = {
        "hashes": build_hashes,
        "identical": len(set(build_hashes)) == 1,
        "distinct_hashes": len(set(build_hashes)),
    }

    np.random.seed(seed)
    graph = build_graph(list(case.layers))
    x_list = np.asarray(x).tolist()
    outputs: List[str] = []
    traces: List[str] = []
    for _ in range(repeats):
        output = graph_forward(graph, x)
        outputs.append(digest_arrays([output]))
        steps, _, _ = run_forward_full(graph, x_list)
        traces.append(
            hashlib.sha256(json.dumps(steps, sort_keys=True).encode("utf-8")).hexdigest()
        )
    results["in_process_forward"] = {"hashes": outputs, "identical": len(set(outputs)) == 1}
    results["in_process_trace"] = {"hashes": traces, "identical": len(set(traces)) == 1}

    prediction = graph_forward(graph, x)
    target = case.make_target(int(prediction.size))
    trajectories: List[List[float]] = []
    for _ in range(repeats):
        # Rebuild from the same seed: a training trajectory is only repeatable if
        # its starting state is reproducible.
        np.random.seed(seed)
        repeat_graph = build_graph(list(case.layers))
        engine = TrainingEngine(
            repeat_graph,
            TrainingConfig(
                epochs=1,
                batch_size=1,
                learning_rate=0.01,
                optimizer="adam",
                loss_function=case.loss_fn,
                shuffle=False,
                dropout_rate=0.0,
            ),
        )
        losses = []
        for _ in range(5):
            losses.append(
                float(
                    compute_loss(
                        np.asarray(target, dtype=np.float32),
                        graph_forward(repeat_graph, x),
                        case.loss_fn,
                    )[0]
                )
            )
            engine.train_batch([(np.asarray(x, dtype=np.float32), np.asarray(target, dtype=np.float32))])
        trajectories.append(losses)
    results["in_process_training"] = {
        "trajectories": trajectories,
        "identical": all(np.allclose(t, trajectories[0], atol=0.0, rtol=0.0) for t in trajectories),
        "final_losses": [t[-1] for t in trajectories],
        "final_loss_dispersion": stat.describe([t[-1] for t in trajectories]),
    }

    # Seed sensitivity: a different initialisation must change the result, or the
    # seed is not actually controlling anything.
    seed_outputs = []
    for offset in (0, 1, 2):
        np.random.seed(seed + 1000 * offset)
        other = build_graph(list(case.layers))
        seed_outputs.append(graph_forward(other, x))
    results["seed_sensitivity"] = {
        "max_abs_difference_seed_offset_1": float(np.max(np.abs(seed_outputs[1] - seed_outputs[0]))),
        "max_abs_difference_seed_offset_2": float(np.max(np.abs(seed_outputs[2] - seed_outputs[0]))),
        "outputs_differ": bool(
            not np.allclose(seed_outputs[1], seed_outputs[0], atol=0.0, rtol=0.0)
        ),
    }
    return results


def cross_process_checks(repo_root: Path, case_ids: List[str], seed: int, repeats: int) -> List[Dict[str, Any]]:
    backend = str(Path(repo_root) / "backend")
    results: List[Dict[str, Any]] = []
    for case_id in case_ids:
        weights_hashes: List[str] = []
        trace_hashes: List[str] = []
        errors: List[str] = []
        for _ in range(repeats):
            code = CROSS_PROCESS_SNIPPET.format(backend=backend, case_id=case_id, seed=seed)
            try:
                completed = subprocess.run(
                    [sys.executable, "-c", code],
                    capture_output=True,
                    text=True,
                    timeout=300,
                    cwd=str(repo_root),
                )
                if completed.returncode != 0:
                    errors.append(completed.stderr[-800:])
                    continue
                payload = json.loads(completed.stdout.strip().splitlines()[-1])
                weights_hashes.append(payload["weights_sha256"])
                trace_hashes.append(payload["trace_sha256"])
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{type(exc).__name__}: {exc}")
        results.append(
            {
                "case_id": case_id,
                "processes": len(weights_hashes),
                "errors": errors,
                "cross_process_build_identical": len(set(weights_hashes)) == 1 and bool(weights_hashes),
                "cross_process_trace_identical": len(set(trace_hashes)) == 1 and bool(trace_hashes),
                "weights_hashes": weights_hashes,
                "trace_hashes": trace_hashes,
            }
        )
    return results


def run(repo_root: Path, selected, seed: int, repeats: int, cross_process_cases: List[str]) -> Dict[str, Any]:
    in_process = []
    for case in selected:
        try:
            in_process.append(in_process_checks(case, seed, repeats))
        except Exception as exc:  # noqa: BLE001
            in_process.append(
                {"case_id": case.case_id, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(limit=6)}
            )
    cross = cross_process_checks(repo_root, cross_process_cases, seed, repeats)

    unsupported = {c.case_id for c in selected if c.expectation == "unsupported"}
    build_failures = [
        r["case_id"]
        for r in in_process
        if r.get("case_id") not in unsupported and not r.get("in_process_build", {}).get("identical")
    ]
    forward_failures = [
        r["case_id"]
        for r in in_process
        if r.get("case_id") not in unsupported and not r.get("in_process_forward", {}).get("identical")
    ]
    trace_failures = [
        r["case_id"]
        for r in in_process
        if r.get("case_id") not in unsupported and not r.get("in_process_trace", {}).get("identical")
    ]
    training_failures = [
        r["case_id"]
        for r in in_process
        if r.get("case_id") not in unsupported and not r.get("in_process_training", {}).get("identical")
    ]
    cross_failures = [
        r["case_id"]
        for r in cross
        if not r["cross_process_build_identical"] or not r["cross_process_trace_identical"]
    ]
    expected_unsupported = [
        r["case_id"]
        for r in in_process
        if r.get("case_id") in unsupported and (r.get("in_process_build", {}).get("identical") is True)
    ]
    errors = [r["case_id"] for r in in_process if "error" in r and r["case_id"] not in unsupported]
    seed_insensitive = [
        r["case_id"]
        for r in in_process
        if r.get("seed_sensitivity", {}).get("outputs_differ") is False
    ]
    summary = {
        "repeats": repeats,
        "cases_checked_in_process": len(in_process) - len(unsupported),
        "cases_declared_unsupported": sorted(unsupported),
        "declared_unsupported_cases_that_unexpectedly_succeeded": expected_unsupported,
        "cases_errored": errors,
        "in_process_build_failures": build_failures,
        "in_process_forward_failures": forward_failures,
        "in_process_trace_failures": trace_failures,
        "in_process_training_failures": training_failures,
        "cross_process_cases": cross_process_cases,
        "cross_process_failures": cross_failures,
        "seed_insensitive_cases": seed_insensitive,
        "seed_sensitivity_all_distinct": not seed_insensitive,
        "status": "PASS"
        if not (build_failures or forward_failures or trace_failures or training_failures or cross_failures or errors)
        else "FINDINGS_PRESENT",
    }
    return {"in_process": in_process, "cross_process": cross, "summary": summary}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=".")
    parser.add_argument("--cases", nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument(
        "--cross-process-cases",
        nargs="*",
        default=["dense_3_4_2_relu", "lstm_t4_d3_h3_dense", "conv1_6x6_2_same_s2_batchnorm_flatten_tanh"],
    )
    parser.add_argument("--allow-overwrite", action="store_true")
    args = parser.parse_args()

    repo_root = Path(args.repo).resolve()
    selected = case_registry.select_cases(args.cases)
    provenance = RunProvenance(
        experiment_id=EXPERIMENT_ID,
        script=Path(__file__).as_posix(),
        command=[sys.executable, *sys.argv],
        seed=args.seed,
        root=repo_root,
        notes={"runtime_source_modified": False, "repeats": args.repeats},
    )
    try:
        result = run(repo_root, selected, args.seed, args.repeats, args.cross_process_cases)
    except Exception as exc:  # noqa: BLE001
        print(f"determinism study failed: {exc}", file=sys.stderr)
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
            "repeats": args.repeats,
            "cross_process_cases": args.cross_process_cases,
            "cases": [c.case_id for c in selected],
        },
        summary_payload=result["summary"],
        extra_raw={
            "in_process": result["in_process"],
            "cross_process": result["cross_process"],
            "environment": environment_report(),
        },
        tables={
            "determinism_summary": [
                {
                    "case_id": r.get("case_id"),
                    "build_identical": r.get("in_process_build", {}).get("identical"),
                    "forward_identical": r.get("in_process_forward", {}).get("identical"),
                    "trace_identical": r.get("in_process_trace", {}).get("identical"),
                    "training_identical": r.get("in_process_training", {}).get("identical"),
                    "training_final_loss": (r.get("in_process_training", {}).get("final_losses") or [None])[0],
                    "training_final_loss_spread": (r.get("in_process_training", {}).get("final_loss_dispersion") or {}).get("max"),
                    "seed_outputs_differ": r.get("seed_sensitivity", {}).get("outputs_differ"),
                    "seed_max_abs_difference": r.get("seed_sensitivity", {}).get("max_abs_difference_seed_offset_1"),
                    "error": r.get("error"),
                }
                for r in result["in_process"]
            ],
            "cross_process_summary": [
                {
                    "case_id": r["case_id"],
                    "processes": r["processes"],
                    "build_identical": r["cross_process_build_identical"],
                    "trace_identical": r["cross_process_trace_identical"],
                    "errors": r["errors"],
                }
                for r in result["cross_process"]
            ],
        },
        interpretation={
            "purpose": "Separates in-process repeatability from cross-process reproducibility.",
            "levels": (
                "A case can be in-process repeatable and still differ across interpreters. "
                "Both levels are measured independently."
            ),
            "seed_sensitivity": (
                "If different seeds produce identical outputs the seed is not controlling "
                "initialisation and a reproducibility claim would be vacuous."
            ),
        },
        limitations=[
            "Cross-process determinism is measured on one machine with one interpreter build; "
            "cross-machine and cross-library determinism are not claimed.",
            "NumPy may select different kernels for different array shapes or CPU features; "
            "no BLAS-thread pinning is applied here.",
            "Training repeatability is measured with shuffle and dropout disabled.",
        ],
    )
    manifest = writer.write()
    summary = result["summary"]
    print(
        f"[{EXPERIMENT_ID}] cases={summary['cases_checked_in_process']} "
        f"build_failures={len(summary['in_process_build_failures'])} "
        f"forward_failures={len(summary['in_process_forward_failures'])} "
        f"training_failures={len(summary['in_process_training_failures'])} "
        f"cross_process_failures={len(summary['cross_process_failures'])} "
        f"status={summary['status']}"
    )
    print(f"evidence: research/evidence/{EXPERIMENT_ID}/{manifest['source_sha']}")
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())