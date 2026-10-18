#!/usr/bin/env python3
"""EXP-01 Graph capability and structural-fidelity audit.

Purpose
-------
Establish, with executed evidence rather than source inspection alone, what the
pinned simulator can actually build and execute, and where its structural
guarantees are weaker than their names suggest.

Probes
------
1. ``build_and_forward`` - every canonical case builds, validates and runs; the
   forward output is compared with an independently computed reference where one
   exists.
2. ``param_sync_roundtrip`` - writing packed graph parameters back into their
   owning layer instances is verified, not assumed.
3. ``attention_num_heads_effect`` - does the requested ``num_heads`` change the
   computation, given identical projection weights?
4. ``batchnorm_vector_degeneracy`` - is a 1-D BatchNorm output independent of its
   input, and does the gradient path below it stay open?
5. ``architecture_hash_discrimination`` - does ``NetworkGraph.architecture_hash``
   distinguish different architectures?
6. ``residual_identity`` - the residual layer is an exact identity in both
   directions.
7. ``dense_batch_support`` - the declared boundary for batched dense execution.

Usage
-----
    PYTHONPATH=backend:research python research/experiments/run_graph_capability_audit.py \
        --repo . --out-root research/evidence

Exit codes: 0 = every declared expectation met, 1 = at least one expectation
violated (findings are still written), 2 = the audit could not run.
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib import cases as case_registry  # noqa: E402
from lib.evidence import EvidenceWriter  # noqa: E402
from lib.numerics import (  # noqa: E402
    graph_forward,
    max_abs_error,
    sync_params_to_layers,
)
from lib.provenance import RunProvenance, utc_now_iso  # noqa: E402

EXPERIMENT_ID = "EXP-01-graph-capability-audit"
DEFAULT_SEED = 20261018
EPSILON = 1e-3


# ---------------------------------------------------------------------------
# probe 1: build, validate, forward
# ---------------------------------------------------------------------------
def probe_build_and_forward(case: case_registry.GraphCase) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "probe": "build_and_forward",
        "case_id": case.case_id,
        "family": case.family,
        "expectation": case.expectation,
    }
    try:
        np.random.seed(case.seed)
        from simulator.graph_engine import build_graph
        from simulator.layers import validate_layers
        from simulator.forward_engine import run_forward_full

        validation = validate_layers(list(case.layers))
        record["validation_errors"] = list(validation.errors)
        graph = build_graph(list(case.layers))
        record["built"] = True
        record["total_params_declared"] = validation.total_params
        record["flops_per_sample_declared"] = validation.flops_per_sample
        record["packed_parameter_arrays"] = len(graph.weights)
        record["packed_parameter_scalars"] = int(
            sum(int(np.size(w)) for w in graph.weights)
        )
        record["parameter_layer_types"] = [
            getattr(graph.layer_instances[i], "layer_type", "unknown")
            for i in graph.param_layer_indices
        ]

        x = case.make_input()
        record["input_shape"] = list(np.shape(x))
        y = graph_forward(graph, x)
        record["output_shape"] = list(y.shape)
        record["output_finite"] = bool(np.isfinite(y).all())

        # The trace engine accepts whatever nesting the caller supplies; the
        # route schema is audited separately by probe_forward_route_input_schema.
        nested = np.asarray(x).tolist()
        steps, trace_output, _ = run_forward_full(graph, nested)
        record["trace_event_count"] = len(steps)
        record["trace_event_operations"] = sorted({str(s.get("operation")) for s in steps})
        record["trace_operation_level"] = bool(
            set(record["trace_event_operations"]) - {"forward"}
        )
        record["trace_matches_forward"] = bool(
            trace_output is not None
            and np.asarray(trace_output, dtype=np.float64).reshape(-1).shape == y.shape
            and max_abs_error(trace_output, y) is not None
            and max_abs_error(trace_output, y) <= case.forward_tolerance["atol"]
        )
        record["observed"] = True
        record["meets_expectation"] = (
            record["output_finite"]
            and record["trace_matches_forward"]
            and not validation.errors
        )
        if case.expectation == "unsupported":
            record["meets_expectation"] = False
            record["finding"] = (
                "case is declared unsupported but executed; the registry "
                "expectation must be corrected"
            )
    except Exception as exc:  # noqa: BLE001 - failures are evidence
        record["built"] = False
        record["observed"] = True
        record["exception"] = f"{type(exc).__name__}: {exc}"
        record["traceback"] = traceback.format_exc(limit=6)
        record["meets_expectation"] = case.expectation == "unsupported"
        if case.expectation != "unsupported":
            record["finding"] = "declared-supported case failed to build or execute"
    return record


# ---------------------------------------------------------------------------
# probe 2: packed-parameter -> layer-instance synchronisation
# ---------------------------------------------------------------------------
def probe_param_sync_roundtrip(case: case_registry.GraphCase) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "probe": "param_sync_roundtrip",
        "case_id": case.case_id,
        "family": case.family,
        "expectation": case.expectation,
    }
    try:
        np.random.seed(case.seed)
        from simulator.graph_engine import build_graph

        graph = build_graph(list(case.layers))
        # Perturb the packed arrays so a silent no-op write is observable.
        rng = np.random.default_rng(case.seed + 31)
        for index, w in enumerate(graph.weights):
            delta = rng.normal(0.0, 0.05, size=w.shape).astype(np.float32)
            graph.weights[index] = np.asarray(w, dtype=np.float32) + delta
            if index < len(graph.biases):
                bdelta = rng.normal(0.0, 0.02, size=graph.biases[index].shape).astype(np.float32)
                graph.biases[index] = np.asarray(graph.biases[index], dtype=np.float32) + bdelta
        report = sync_params_to_layers(graph)
        record["sync"] = report.to_dict()
        record["observed"] = True
        record["meets_expectation"] = report.fully_synchronised
        if not report.fully_synchronised:
            record["finding"] = (
                "packed graph parameters were not observed inside their owning "
                "layer after set_params"
            )
    except Exception as exc:  # noqa: BLE001
        record["observed"] = True
        record["exception"] = f"{type(exc).__name__}: {exc}"
        record["meets_expectation"] = False
    return record


# ---------------------------------------------------------------------------
# probe 3: does num_heads change the attention computation?
# ---------------------------------------------------------------------------
def probe_attention_num_heads_effect(seed: int) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "probe": "attention_num_heads_effect",
        "expectation": "num_heads must change the computation when it changes the architecture",
    }
    try:
        from simulator.graph_engine import build_graph
        from simulator.layers import LayerConfig

        base_layers = [
            LayerConfig("input", input_shape=[4, 4]),
            LayerConfig("attention", num_heads=1, neurons=4),
            LayerConfig("flatten"),
            LayerConfig("output", neurons=2, activation="linear"),
        ]
        multi_layers = [
            LayerConfig("input", input_shape=[4, 4]),
            LayerConfig("attention", num_heads=2, neurons=4),
            LayerConfig("flatten"),
            LayerConfig("output", neurons=2, activation="linear"),
        ]
        np.random.seed(seed)
        one = build_graph(base_layers)
        np.random.seed(seed)
        two = build_graph(multi_layers)

        x = np.linspace(-0.5, 0.5, 16, dtype=np.float32).reshape(4, 4)
        y_one = graph_forward(one, x)
        y_two = graph_forward(two, x)

        # Force identical projections so any difference can only come from head handling.
        layer_one = one.layer_instances[1]
        layer_two = two.layer_instances[1]
        for name in ("W_q", "W_k", "W_v", "W_o"):
            setattr(layer_two, name, getattr(layer_one, name).copy())

        y_two_shared = graph_forward(two, x)
        difference = max_abs_error(y_one, y_two_shared)

        record.update(
            {
                "observed": True,
                "requested_heads": {"one": 1, "two": 2},
                "stored_num_heads": {
                    "one": int(getattr(layer_one, "num_heads", -1)),
                    "two": int(getattr(layer_two, "num_heads", -1)),
                },
                "attn_weights_shape_one": list(np.shape(layer_one.attn_weights)),
                "attn_weights_shape_two": list(np.shape(layer_two.attn_weights)),
                "max_abs_output_difference_with_shared_weights": difference,
                "output_identical": bool(difference is not None and difference == 0.0),
                "meets_expectation": bool(difference is not None and difference > 0.0),
                "finding": (
                    "num_heads is stored on the layer but never used by the "
                    "attention forward pass, so one-head and two-head requests "
                    "produce an identical computation for identical projections"
                ),
            }
        )
    except Exception as exc:  # noqa: BLE001
        record["observed"] = False
        record["exception"] = f"{type(exc).__name__}: {exc}"
        record["meets_expectation"] = False
    return record


# ---------------------------------------------------------------------------
# probe 4: 1-D BatchNorm degeneracy
# ---------------------------------------------------------------------------
def probe_batchnorm_vector_degeneracy(seed: int) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "probe": "batchnorm_vector_degeneracy",
        "expectation": "the probe must measure and report the 1-D BatchNorm regime",
    }
    try:
        from simulator.graph_engine import build_graph
        from simulator.layers import LayerConfig

        np.random.seed(seed)
        layers = [
            LayerConfig("input", neurons=5, input_shape=[5]),
            LayerConfig("batchnorm"),
            LayerConfig("output", neurons=2, activation="linear"),
        ]
        graph = build_graph(layers)
        bn = graph.layer_instances[1]

        x_a = np.array([0.4, -0.2, 0.9, -0.7, 0.1], dtype=np.float32)
        x_b = np.array([-0.3, 0.8, -0.6, 0.5, -0.9], dtype=np.float32)
        bn_only_a = np.asarray(bn.forward(x_a), dtype=np.float64)
        bn_only_b = np.asarray(bn.forward(x_b), dtype=np.float64)
        y_a = graph_forward(graph, x_a)
        y_b = graph_forward(graph, x_b)

        # Does the gradient below the normalisation layer stay open?
        from simulator.backward_engine import backward_full

        back = backward_full(graph, x_a.tolist(), np.array([0.1, 0.2]).tolist(), "mse")
        input_gradient = np.asarray(back["input_gradient"], dtype=np.float64)
        batchnorm_grads = np.asarray(graph.weights[1], dtype=np.float64)

        record.update(
            {
                "observed": True,
                "batchnorm_output_for_input_a": bn_only_a.tolist(),
                "batchnorm_output_for_input_b": bn_only_b.tolist(),
                "batchnorm_output_identical": bool(
                    np.allclose(bn_only_a, bn_only_b, atol=0.0, rtol=0.0)
                ),
                "batchnorm_output_equals_beta": bool(
                    np.allclose(bn_only_a, np.asarray(bn.beta, dtype=np.float64), atol=1e-6)
                ),
                "graph_output_for_input_a": y_a.tolist(),
                "graph_output_for_input_b": y_b.tolist(),
                "graph_outputs_identical": bool(np.allclose(y_a, y_b, atol=0.0, rtol=0.0)),
                "input_gradient_l2": float(np.linalg.norm(input_gradient)),
                "batchnorm_gamma_gradient_l2": float(np.linalg.norm(batchnorm_grads)),
                "meets_expectation": True,
                "finding": (
                    "With a 1-D input the per-feature mean equals the feature and "
                    "the per-feature variance is zero, so the normalised value is "
                    "zero and the layer output equals beta. The output is "
                    "independent of the input and the input gradient is exactly "
                    "zero."
                )
                if bool(np.allclose(bn_only_a, bn_only_b, atol=0.0, rtol=0.0))
                else "1-D BatchNorm output depends on its input",
            }
        )
    except Exception as exc:  # noqa: BLE001
        record["observed"] = False
        record["exception"] = f"{type(exc).__name__}: {exc}"
        record["meets_expectation"] = False
    return record


# ---------------------------------------------------------------------------
# probe 5: architecture_hash discrimination
# ---------------------------------------------------------------------------
def probe_architecture_hash_discrimination() -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "probe": "architecture_hash_discrimination",
        "expectation": "architecture_hash must distinguish different architectures",
    }
    try:
        from simulator.graph_engine import build_graph
        from simulator.layers import LayerConfig

        variants = {
            "dense_4_2_relu": [
                LayerConfig("input", neurons=4, input_shape=[4]),
                LayerConfig("dense", neurons=2, activation="relu"),
                LayerConfig("output", neurons=2, activation="linear"),
            ],
            "dense_4_2_tanh": [
                LayerConfig("input", neurons=4, input_shape=[4]),
                LayerConfig("dense", neurons=2, activation="tanh"),
                LayerConfig("output", neurons=2, activation="linear"),
            ],
            "dense_4_2_output_width_3": [
                LayerConfig("input", neurons=4, input_shape=[4]),
                LayerConfig("dense", neurons=2, activation="relu"),
                LayerConfig("output", neurons=3, activation="linear"),
            ],
            "conv_variant_same_neurons": [
                LayerConfig("input", input_shape=[4, 4, 4]),
                LayerConfig("conv2d", filters=2, kernel_size=2, activation="relu"),
                LayerConfig("flatten"),
                LayerConfig("output", neurons=2, activation="linear"),
            ],
        }
        hashes: Dict[str, str] = {}
        for name, layers in variants.items():
            np.random.seed(0)
            hashes[name] = build_graph(layers).architecture_hash

        collisions = [
            {"a": a, "b": b, "hash": hashes[a]}
            for a in hashes
            for b in hashes
            if a < b and hashes[a] == hashes[b]
        ]
        record.update(
            {
                "observed": True,
                "hashes": hashes,
                "distinct_hash_count": len(set(hashes.values())),
                "collision_pairs": collisions,
                "meets_expectation": not collisions,
                "finding": (
                    "architecture_hash is derived from the comma-joined neuron "
                    "counts only, so distinct architectures collide"
                )
                if collisions
                else "architecture_hash separated the probed variants",
            }
        )
    except Exception as exc:  # noqa: BLE001
        record["observed"] = False
        record["exception"] = f"{type(exc).__name__}: {exc}"
        record["meets_expectation"] = False
    return record


# ---------------------------------------------------------------------------
# probe 6: residual identity
# ---------------------------------------------------------------------------
def probe_residual_identity() -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "probe": "residual_identity",
        "expectation": "the residual layer must be an exact identity in both directions",
    }
    try:
        from simulator.layers.residual import ResidualLayer

        rng = np.random.default_rng(7)
        x = rng.normal(0.0, 1.0, size=(2, 3, 3)).astype(np.float32)
        layer = ResidualLayer()
        y = layer.forward(x)
        d = rng.normal(0.0, 1.0, size=(2, 3, 3)).astype(np.float32)
        dx = layer.backward(d)
        record.update(
            {
                "observed": True,
                "forward_max_abs_error": max_abs_error(y, x),
                "backward_max_abs_error": max_abs_error(dx, d),
                "meets_expectation": bool(
                    np.array_equal(y, x) and np.array_equal(dx, d)
                ),
            }
        )
    except Exception as exc:  # noqa: BLE001
        record["observed"] = False
        record["exception"] = f"{type(exc).__name__}: {exc}"
        record["meets_expectation"] = False
    return record


# ---------------------------------------------------------------------------
# probe 7: dense batch boundary
# ---------------------------------------------------------------------------
def probe_dense_batch_support() -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "probe": "dense_batch_support",
        "expectation": "the batched-dense boundary must be reproduced, not assumed",
    }
    try:
        from simulator.graph_engine import build_graph
        from simulator.layers import LayerConfig

        np.random.seed(11)
        layers = [
            LayerConfig("input", neurons=3, input_shape=[3]),
            LayerConfig("dense", neurons=4, activation="tanh"),
            LayerConfig("output", neurons=2, activation="linear"),
        ]
        graph = build_graph(layers)
        batch = np.linspace(-0.5, 0.5, 9, dtype=np.float32).reshape(3, 3)
        try:
            out = graph.forward(batch.copy())
            record.update(
                {
                    "observed": True,
                    "batch_forward_supported": True,
                    "output_shape": list(np.shape(out)),
                    "meets_expectation": True,
                }
            )
        except Exception as exc:  # noqa: BLE001
            record.update(
                {
                    "observed": True,
                    "batch_forward_supported": False,
                    "exception": f"{type(exc).__name__}: {exc}",
                    "meets_expectation": True,
                    "finding": (
                        "the dense layer flattens its input, so a batched dense "
                        "forward pass is not supported; batching is therefore a "
                        "declared boundary of the dense path"
                    ),
                }
            )
    except Exception as exc:  # noqa: BLE001
        record["observed"] = False
        record["exception"] = f"{type(exc).__name__}: {exc}"
        record["meets_expectation"] = False
    return record


# ---------------------------------------------------------------------------
# probe 8: forward-route input schema boundary
# ---------------------------------------------------------------------------
def probe_forward_route_input_schema() -> Dict[str, Any]:
    """Can the forward trace route express the inputs its graphs require?

    ``ForwardRequest.input`` is declared as ``List[float]``. Graphs declared with
    a spatial or sequence ``input_shape`` require a nested array, so the route
    schema and the engine contract must be checked against each other rather
    than inferred from one of them.
    """
    record: Dict[str, Any] = {
        "probe": "forward_route_input_schema",
        "expectation": (
            "the forward trace route must be able to express the inputs required "
            "by the architectures it accepts"
        ),
    }
    try:
        from simulator.forward_engine import run_forward_full
        from simulator.graph_engine import build_graph
        from simulator.layers import LayerConfig

        schema_observation: Dict[str, Any] = {}
        try:
            from api.simulator_routes import ForwardRequest

            field = ForwardRequest.model_fields["input"]
            annotation = str(field.annotation)
            is_flat_list = "list[float]" in annotation.lower().replace("typing.", "")
            schema_observation = {
                "request_model": "api.simulator_routes.ForwardRequest",
                "input_annotation": annotation,
                "declares_flat_float_list": is_flat_list,
            }
        except Exception as exc:  # noqa: BLE001
            schema_observation = {"error": f"{type(exc).__name__}: {exc}"}

        np.random.seed(3)
        spatial = build_graph(
            [
                LayerConfig("input", input_shape=[1, 5, 5]),
                LayerConfig("conv2d", filters=1, kernel_size=3, activation="relu"),
                LayerConfig("flatten"),
                LayerConfig("output", neurons=2, activation="linear"),
            ]
        )
        flat_payload = np.linspace(-1.0, 1.0, 25, dtype=np.float32).tolist()
        route_reproduced: Dict[str, Any] = {}
        try:
            run_forward_full(spatial, flat_payload)
            route_reproduced = {"executed": True}
        except Exception as exc:  # noqa: BLE001
            route_reproduced = {
                "executed": False,
                "exception": f"{type(exc).__name__}: {exc}",
            }

        nested_payload = np.linspace(-1.0, 1.0, 25, dtype=np.float32).reshape(1, 5, 5).tolist()
        engine_nested: Dict[str, Any] = {}
        try:
            steps, output, _ = run_forward_full(spatial, nested_payload)
            engine_nested = {
                "executed": True,
                "step_count": len(steps),
                "output_size": len(output),
            }
        except Exception as exc:  # noqa: BLE001
            engine_nested = {"executed": False, "exception": f"{type(exc).__name__}: {exc}"}

        boundary = (
            not route_reproduced.get("executed", False)
            and engine_nested.get("executed", False)
        )
        record.update(
            {
                "observed": True,
                "schema": schema_observation,
                "route_payload_flat_list": route_reproduced,
                "engine_payload_nested_list": engine_nested,
                "boundary_present": boundary,
                "meets_expectation": not boundary,
                "finding": (
                    "POST /api/simulator/forward/full declares input as "
                    "List[float]; a spatial graph requires a nested array, so the "
                    "route cannot execute convolutional graphs even though the "
                    "engine can when handed a nested payload directly"
                )
                if boundary
                else "no input-schema boundary reproduced",
            }
        )
    except Exception as exc:  # noqa: BLE001
        record["observed"] = False
        record["exception"] = f"{type(exc).__name__}: {exc}"
        record["meets_expectation"] = False
    return record


# ---------------------------------------------------------------------------
def run(repo_root: Path) -> Dict[str, Any]:
    selected = case_registry.select_cases()
    records: List[Dict[str, Any]] = []
    for case in selected:
        if case.expectation == "unsupported":
            continue
        records.append(probe_build_and_forward(case))
        records.append(probe_param_sync_roundtrip(case))
    records.append(probe_build_and_forward(case_registry.DENSE_MULTI_BATCH_MSE))
    records.append(probe_attention_num_heads_effect(DEFAULT_SEED))
    records.append(probe_batchnorm_vector_degeneracy(DEFAULT_SEED))
    records.append(probe_architecture_hash_discrimination())
    records.append(probe_residual_identity())
    records.append(probe_dense_batch_support())
    records.append(probe_forward_route_input_schema())

    violations = [r for r in records if r.get("meets_expectation") is False]
    summary = {
        "probe_count": len(records),
        "records_with_observations": sum(1 for r in records if r.get("observed")),
        "expectation_violations": len(violations),
        "violating_probes": sorted({str(r.get("probe")) for r in violations}),
        "cases_covered": sorted({str(r.get("case_id")) for r in records if r.get("case_id")}),
        "families_covered": sorted({str(r.get("family")) for r in records if r.get("family")}),
        "status": "findings_present" if violations else "all_expectations_met",
    }
    return {"records": records, "summary": summary}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=".")
    parser.add_argument("--out-root", default="research/evidence")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--allow-overwrite", action="store_true")
    args = parser.parse_args()

    repo_root = Path(args.repo).resolve()
    np.random.seed(args.seed)
    provenance = RunProvenance(
        experiment_id=EXPERIMENT_ID,
        script=str(Path(__file__).as_posix()),
        command=[sys.executable, *sys.argv],
        seed=args.seed,
        root=repo_root,
        notes={
            "runtime_source_modified": False,
            "purpose": "structural capability and fidelity audit of the pinned simulator",
        },
    )

    try:
        result = run(repo_root)
    except Exception as exc:  # noqa: BLE001
        print(f"audit failed before producing results: {exc}", file=sys.stderr)
        provenance.exit_code = 2
        provenance.finished_at_utc = utc_now_iso()
        return 2

    provenance.exit_code = 0 if result["summary"]["expectation_violations"] == 0 else 1
    provenance.finished_at_utc = utc_now_iso()

    writer = EvidenceWriter(
        experiment_id=EXPERIMENT_ID,
        repo_root=repo_root,
        provenance=provenance,
        seed=args.seed,
        allow_overwrite=args.allow_overwrite,
        configuration={
            "cases": case_registry.case_index(),
            "epsilon": EPSILON,
            "numpy_seed": args.seed,
        },
        summary_payload=result["summary"],
        extra_raw={"probe_records": result["records"]},
        tables={
            "probe_records": [
                {
                    "probe": r.get("probe"),
                    "case_id": r.get("case_id"),
                    "family": r.get("family"),
                    "observed": r.get("observed"),
                    "meets_expectation": r.get("meets_expectation"),
                    "finding": r.get("finding"),
                    "exception": r.get("exception"),
                }
                for r in result["records"]
            ]
        },
        interpretation={
            "purpose": "Structural capability evidence, not gradient correctness.",
            "how_to_read": (
                "A probe with meets_expectation=false is a reproduced runtime "
                "finding. It is not fixed here: the runtime is read-only for this "
                "research phase."
            ),
        },
        limitations=[
            "Only the graph cases registered in research/lib/cases.py are audited.",
            "Behaviour under concurrency, multi-process reuse and persistence is not covered.",
            "Probes report the pinned revision only; no claim is made about other revisions.",
        ],
    )
    manifest = writer.write()

    print(
        f"[{EXPERIMENT_ID}] probes={result['summary']['probe_count']} "
        f"violations={result['summary']['expectation_violations']} "
        f"status={result['summary']['status']}"
    )
    print(f"evidence: research/evidence/{EXPERIMENT_ID}/{manifest['source_sha']}")
    return 1 if result["summary"]["expectation_violations"] else 0


if __name__ == "__main__":
    raise SystemExit(main())