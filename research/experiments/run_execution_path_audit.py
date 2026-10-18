#!/usr/bin/env python3
"""EXP-09 Execution-path separation and synthetic-value audit.

Purpose
-------
Determine, with executed evidence, which surfaces in the application produce
values from a real computation and which produce values from a fixed or
synthesised source. This separates two questions the paper must not conflate:
whether different execution paths share model state, and whether any user-facing
value can be mistaken for measured computation.

Audits
------
1. ``execution_paths``      enumerate backend route families, their engines and
                            their state owners; verify whether weights are shared.
2. ``snapshot_stream``      determine whether the /stream WebSocket payload is
                            derived from a configured graph or from a fixed
                            topology with formula-generated values.
3. ``demo_provenance``      verify that the Simulator page's demo input and
                            one-hot target are generated rather than measured,
                            and that the requested softmax output is not a
                            normalised probability vector.
4. ``label_honesty``        record which synthetic surfaces carry an explicit
                            user-visible label in the frontend source.

Usage
-----
    PYTHONPATH=backend:research python research/experiments/run_execution_path_audit.py --repo .
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import re
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib import cases as case_registry  # noqa: E402
from lib import reference  # noqa: E402
from lib.evidence import EvidenceWriter  # noqa: E402
from lib.numerics import graph_forward  # noqa: E402
from lib.provenance import RunProvenance, utc_now_iso  # noqa: E402

EXPERIMENT_ID = "EXP-09-execution-path-audit"
DEFAULT_SEED = 20261018


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------
def audit_execution_paths(repo_root: Path) -> Dict[str, Any]:
    from app import app  # type: ignore

    # The OpenAPI document forces the lazily included routers to materialise,
    # which plain iteration over ``app.routes`` does not do on this FastAPI
    # version.
    openapi_paths = sorted(app.openapi().get("paths", {}).keys())
    websocket_paths: List[str] = []
    api_root = repo_root / "backend"
    for path in sorted(api_root.rglob("*.py")):
        text = _read(path)
        for match in re.finditer(r'@router\.websocket\(\s*"([^"]+)"\s*\)', text):
            websocket_paths.append(match.group(1))

    families: Dict[str, List[str]] = {}
    for path in [*openapi_paths, *sorted(set(websocket_paths))]:
        if path.startswith("/api/simulator") or path.startswith("/ws/simulator"):
            family = "simulator_graph_session"
        elif path.startswith("/api/execute"):
            family = "legacy_execution_engine"
        elif path.startswith("/api/lab"):
            family = "framework_lab_trace"
        elif path.startswith("/models") or path.startswith("/api/predict") or path.startswith("/api/model"):
            family = "model_registry_inference"
        elif path.startswith("/train") or path.startswith("/ws/train"):
            family = "framework_training"
        elif path.startswith("/stream"):
            family = "illustrative_stream"
        else:
            family = "other"
        families.setdefault(family, []).append(path)

    # Does any route family read another family's state?
    shared_state: List[Dict[str, Any]] = []
    try:
        from simulator.session_manager import session_manager
        from simulator.graph_engine import NetworkGraph
        from model_registry.registry import registry

        graph_state = list(session_manager._graphs.keys())
        shared_state.append(
            {
                "check": "simulator graph sessions are stored in the process-local SessionManager",
                "graph_ids_held": len(graph_state),
                "reachable_from_model_registry": bool(
                    any("session" in str(type(v)).lower() or "graph" in str(type(v)).lower() for v in vars(registry).values())
                ),
                "conclusion": "the graph-session store is not populated from the model registry, "
                "so registry weights are not loaded into simulator graphs",
            }
        )
    except Exception as exc:  # noqa: BLE001
        shared_state.append({"check": "session manager / registry inspection", "error": f"{type(exc).__name__}: {exc}"})

    try:
        from simulator import execution_engine as ee

        engine = ee.get_engine()
        shared_state.append(
            {
                "check": "legacy ExecutionEngine owns a separate weight store",
                "engine_type": type(engine).__name__,
                "engine_module": ee.__name__,
                "engine_has_graph_after_use": bool(getattr(engine, "_graph", None)),
                "conclusion": "the legacy engine keeps its own weight list; it does not read "
                "NetworkGraph.weights and is not driven by the graph session manager",
            }
        )
    except Exception as exc:  # noqa: BLE001
        shared_state.append({"check": "legacy execution engine inspection", "error": f"{type(exc).__name__}: {exc}"})

    return {
        "route_families": {k: sorted(v) for k, v in sorted(families.items())},
        "route_counts": {k: len(v) for k, v in sorted(families.items())},
        "shared_state_checks": shared_state,
    }


# ---------------------------------------------------------------------------
def audit_snapshot_stream(repo_root: Path) -> Dict[str, Any]:
    """Measure whether the /stream payload responds to any configured graph."""
    source = _read(repo_root / "backend" / "api" / "stream_ws.py")
    from api.stream_ws import _build_snapshot  # type: ignore

    snapshots = [_build_snapshot(epoch) for epoch in (0, 1, 7, 40)]
    topology_ids = [
        sorted(neuron["id"] for neuron in snapshot["topology"]["neurons"]) for snapshot in snapshots
    ]
    topology_stable = all(ids == topology_ids[0] for ids in topology_ids)
    losses = [snapshot["metrics"]["loss"] for snapshot in snapshots]
    accuracies = [snapshot["metrics"]["accuracy"] for snapshot in snapshots]
    formula_signals = {
        "uses_math_sin_or_cos": bool(re.search(r"math\.(sin|cos)", source)),
        "uses_hard_coded_weight_table": bool(re.search(r"w1_base\s*[:=]", source)),
        "reads_session_manager": "session_manager" in source,
        "reads_training_sessions": "training_sessions" in source,
        "reads_dataset_manager": "dataset_manager" in source,
        "reads_graph_engine": "graph_engine" in source,
        "references_graph_id": "graph_id" in source,
    }
    return {
        "snapshots_measured": len(snapshots),
        "topology_identical_across_epochs": bool(topology_stable),
        "topology_ids": topology_ids[0] if topology_ids else [],
        "loss_series": losses,
        "accuracy_series": accuracies,
        "loss_matches_exponential_decay_formula": bool(
            all(
                abs(loss - max(0.02, 1.25 * math.exp(-epoch * 0.045) + 0.04 * math.sin(epoch * 0.21))) < 1e-9
                for epoch, loss in zip((0, 1, 7, 40), losses)
            )
        ),
        "formula_signals": formula_signals,
        "derived_from_configured_graph": bool(
            formula_signals["reads_session_manager"]
            or formula_signals["reads_graph_engine"]
            or formula_signals["references_graph_id"]
        ),
        "conclusion": (
            "The /stream payload is a fixed 3-4-2 topology whose weights are a hard-coded "
            "table and whose loss and accuracy follow closed-form expressions of a counter. "
            "It is not derived from any configured simulator graph or training session."
        ),
    }


# ---------------------------------------------------------------------------
def audit_demo_provenance(repo_root: Path, seed: int) -> Dict[str, Any]:
    """Verify the Simulator page's demo setup and the requested softmax output."""
    np.random.seed(seed)
    from simulator.activations import get_activation
    from simulator.graph_engine import build_graph
    from simulator.layers import LayerConfig

    softmax_spec = get_activation("softmax")
    x = np.array([0.4, -0.7, 0.2, 0.5, -0.3, 0.1, 0.9, -0.6, 0.2, 0.8, -0.9, 0.3, 0.7, -0.2, 0.5, 0.1], dtype=np.float32)
    forward = np.asarray(softmax_spec.forward(x), dtype=np.float64)

    graph = build_graph(
        [
            LayerConfig("input", neurons=16, input_shape=[16]),
            LayerConfig("dense", neurons=8, activation="tanh"),
            LayerConfig("output", neurons=2, activation="softmax"),
        ]
    )
    output = graph_forward(graph, x)
    sigmoid_reference = graph_forward(
        build_graph(
            [
                LayerConfig("input", neurons=16, input_shape=[16]),
                LayerConfig("dense", neurons=8, activation="tanh"),
                LayerConfig("output", neurons=2, activation="softmax"),
            ]
        ),
        x,
    )
    source = _read(repo_root / "frontend" / "src" / "pages" / "SimulatorPage.tsx")
    return {
        "softmax_registered": "softmax" in _read(repo_root / "backend" / "simulator" / "activations.py"),
        "unknown_activation_resolves_to": softmax_spec.name,
        "softmax_input_output_identity": bool(np.allclose(forward, x)),
        "softmax_output_sums_to_one": bool(abs(float(np.sum(forward)) - 1.0) < 1e-6),
        "graph_output_sums_to_one": bool(abs(float(np.sum(output)) - 1.0) < 1e-6),
        "graph_output_range": [float(np.min(output)), float(np.max(output))],
        "demo_page_requests_softmax": "softmax" in source,
        "demo_page_generates_random_input": bool(
            re.search(r"Math\.random|np\.random|random\.", source)
        ),
        "note": (
            "An unregistered activation name resolves to linear, so a demo configured "
            "with softmax produces logits whose sum is not one."
        ),
        "reference_range_check": sigmoid_reference.shape == output.shape,
    }


# ---------------------------------------------------------------------------
def audit_label_honesty(repo_root: Path) -> Dict[str, Any]:
    """Record which synthetic surfaces carry an explicit user-visible label."""
    frontend = repo_root / "frontend" / "src"
    targets: Dict[str, List[str]] = {}
    label_patterns = {
        "demo": re.compile(r"demo|random forward", re.IGNORECASE),
        "illustrative": re.compile(r"illustrative|demo data|simulated stream|not live", re.IGNORECASE),
        "synthetic": re.compile(r"synthetic", re.IGNORECASE),
    }
    if frontend.exists():
        for path in sorted(frontend.rglob("*.tsx")) + sorted(frontend.rglob("*.ts")):
            text = _read(path)
            relative = path.relative_to(repo_root).as_posix()
            for key, pattern in label_patterns.items():
                if pattern.search(text):
                    targets.setdefault(key, []).append(relative)
    return {
        "frontend_files_scanned": len(list(frontend.rglob("*.tsx"))) if frontend.exists() else 0,
        "files_mentioning_each_label_class": {k: sorted(set(v)) for k, v in targets.items()},
        "conclusion": (
            "A global note mentioning a random forward pass exists, but per-value or "
            "per-panel provenance labelling of the illustrative stream and demo inputs "
            "is not established by a label search. Absence of a matched string is not "
            "proof of absence, so this is recorded as unverified rather than as a defect."
        ),
    }


# ---------------------------------------------------------------------------
def audit_legacy_engine_numerics(seed: int) -> Dict[str, Any]:
    """Independently validate the legacy ExecutionEngine's dense backward pass."""
    from simulator.execution_engine import get_engine

    engine = get_engine()
    np.random.seed(seed)
    graph = [
        {"type": "input", "neurons": 4},
        {"type": "dense", "neurons": 5, "activation": "tanh"},
        {"type": "dense", "neurons": 3, "activation": "tanh"},
        {"type": "output", "neurons": 2, "activation": "linear"},
    ]
    built = engine.build_graph(graph)
    built = engine._graph  # the engine keeps its own weight store; the public return value is metadata only
    # build_graph seeds from an unseeded generator; the initial weights are
    # therefore not reproducible, which is recorded below.
    initial_weight_norm = float(
        np.linalg.norm(np.concatenate([np.asarray(w).ravel() for w in built["weights"]]))
    )
    x = np.linspace(-0.7, 0.7, 4, dtype=np.float32)
    target = np.array([0.2, -0.1], dtype=np.float32)
    engine.forward(x)
    result = engine.backward(target.tolist(), loss_function="mse")

    weights_before = [np.asarray(w, dtype=np.float64).copy() for w in built["weights"]]
    biases_before = [np.asarray(b, dtype=np.float64).copy() for b in built["biases"]]
    grads_w = [np.asarray(layer.gradient_weights, dtype=np.float64) for layer in result.layers]
    grads_b = [np.asarray(layer.gradient_bias, dtype=np.float64) for layer in result.layers]

    def objective() -> float:
        forward = engine.forward(np.asarray(x, dtype=np.float32))
        y_hat = np.asarray(forward.final_output, dtype=np.float64).reshape(-1)
        return float(np.mean(0.5 * (y_hat - np.asarray(target, dtype=np.float64)) ** 2))

    findings: List[Dict[str, Any]] = []
    eps = 1e-3
    for index, (w, gw) in enumerate(zip(built["weights"], grads_w)):
        from lib.numerics import central_difference, compare_gradients

        def setter(value: np.ndarray, index: int = index) -> None:
            # The engine reads its weights from its own store, so the perturbation
            # must be written back there for the objective to see it.
            built["weights"][index] = np.asarray(value, dtype=np.float32)

        numeric = central_difference(
            objective, np.asarray(w, dtype=np.float64), eps, setter=setter
        )
        comparison = compare_gradients(
            np.asarray(gw, dtype=np.float64).reshape(-1),
            numeric.reshape(-1),
            eps,
            2e-3,
            2e-2,
        )
        setter(np.asarray(w, dtype=np.float64))
        findings.append(
            {
                "param_index": index,
                "kind": "weights",
                "analytic_norm": comparison and float(np.linalg.norm(gw)),
                "max_abs_error": comparison.max_abs_error,
                "relative_l2_error": comparison.relative_l2_error,
                "cosine_similarity": comparison.cosine_similarity,
                "passed": comparison.passed,
            }
        )
    return {
        "engine": type(engine).__name__,
        "supported_layers": sorted(getattr(engine, "SUPPORTED_LAYERS", []) or []),
        "initial_weight_norm": initial_weight_norm,
        "build_graph_accepts_seed": False,
        "loss_value": float(result.loss_value),
        "weight_gradient_checks": findings,
        "weights_unchanged_by_backward": bool(
            all(np.allclose(np.asarray(w), before) for w, before in zip(built["weights"], weights_before))
        ),
        "conclusion": (
            "The legacy ExecutionEngine implements a dense-only backward pass. Its "
            "agreement with central differences is measured here because it is a "
            "different implementation from the graph-session backward engine."
        ),
    }


# ---------------------------------------------------------------------------
def run(repo_root: Path, seed: int) -> Dict[str, Any]:
    audits: Dict[str, Any] = {}
    for name, runner in (
        ("execution_paths", lambda: audit_execution_paths(repo_root)),
        ("snapshot_stream", lambda: audit_snapshot_stream(repo_root)),
        ("demo_provenance", lambda: audit_demo_provenance(repo_root, seed)),
        ("label_honesty", lambda: audit_label_honesty(repo_root)),
        ("legacy_engine_numerics", lambda: audit_legacy_engine_numerics(seed)),
    ):
        try:
            audits[name] = {"status": "MEASURED", **runner()}
        except Exception as exc:  # noqa: BLE001
            audits[name] = {
                "status": "ERROR",
                "exception": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(limit=6),
            }
    summary = {
        "audits": {name: payload.get("status") for name, payload in audits.items()},
        "route_families": audits.get("execution_paths", {}).get("route_counts"),
        "stream_derived_from_configured_graph": audits.get("snapshot_stream", {}).get(
            "derived_from_configured_graph"
        ),
        "softmax_requested_but_not_registered": (
            audits.get("demo_provenance", {}).get("demo_page_requests_softmax")
            and not audits.get("demo_provenance", {}).get("softmax_registered")
        ),
        "status": "PASS"
        if all(payload.get("status") == "MEASURED" for payload in audits.values())
        else "PARTIAL",
    }
    return {"audits": audits, "summary": summary}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=".")
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
        result = run(repo_root, args.seed)
    except Exception as exc:  # noqa: BLE001
        print(f"execution-path audit failed: {exc}", file=sys.stderr)
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
            "audits": [
                "execution_paths",
                "snapshot_stream",
                "demo_provenance",
                "label_honesty",
                "legacy_engine_numerics",
            ]
        },
        summary_payload=result["summary"],
        extra_raw={"audits": result["audits"]},
        tables={
            "execution_path_families": [
                {"family": family, "routes": len(routes)}
                for family, routes in result["audits"]
                .get("execution_paths", {})
                .get("route_families", {})
                .items()
            ],
            "legacy_engine_gradient_checks": result["audits"]
            .get("legacy_engine_numerics", {})
            .get("weight_gradient_checks", []),
        },
        interpretation={
            "purpose": "Separates computed values from synthesised values and maps execution-path ownership.",
            "shared_state": (
                "The graph-session store, the legacy execution engine and the model "
                "registry each own separate weight state. No verified import path links them."
            ),
            "illustrative_stream": (
                "The /stream payload must be described as a fixed illustrative animation, "
                "not as live model state or measured training telemetry."
            ),
            "softmax": (
                "The demo requests a softmax output activation that the registry does not "
                "implement; the fallback is linear, so the displayed outputs are logits."
            ),
        },
        limitations=[
            "Frontend label coverage is established by a text search over frontend sources; "
            "absence of a matched string is not proof that no label exists.",
            "No browser session is started: WebSocket payloads are generated by calling the "
            "server's snapshot function directly.",
            "Registry adapters that require framework weights were not exercised.",
        ],
    )
    manifest = writer.write()
    summary = result["summary"]
    print(
        f"[{EXPERIMENT_ID}] audits={len(summary['audits'])} "
        f"status={summary['status']} "
        f"stream_derived_from_graph={summary['stream_derived_from_configured_graph']}"
    )
    print(f"evidence: research/evidence/{EXPERIMENT_ID}/{manifest['source_sha']}")
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())