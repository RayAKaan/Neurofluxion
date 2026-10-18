#!/usr/bin/env python3
"""EXP-08 Runtime defect reproduction and severity register.

Purpose
-------
Reproduce, quantify and document defects the experiments above exposed, without
modifying any runtime source. Each defect is recorded with a minimal
reproduction, its measured impact, its effect on paper claims, whether a runtime
fix is required, and a recommended remediation that is *described*, not applied.

A defect is recorded with a severity derived from its measured effect on the
simulator's own outputs, not from how inconvenient it is.

Usage
-----
    PYTHONPATH=backend:research python research/experiments/run_defect_register.py --repo .
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib import cases as case_registry  # noqa: E402
from lib import stats as stat  # noqa: E402
from lib.evidence import EvidenceWriter  # noqa: E402
from lib.numerics import graph_forward, max_abs_error, sync_params_to_layers  # noqa: E402
from lib.provenance import RunProvenance, utc_now_iso  # noqa: E402

EXPERIMENT_ID = "EXP-08-defect-register"
DEFAULT_SEED = 20261018

SEVERITY_ORDER = {"BLOCKING": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3, "INFORMATIONAL": 4}


def defect(
    defect_id: str,
    title: str,
    severity: str,
    component: str,
    reproduction: str,
    expectation: str,
    runner: Callable[[], Dict[str, Any]],
    impact: str,
    claim_effect: str,
    remediation: str,
    requires_runtime_change: bool = True,
) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "defect_id": defect_id,
        "title": title,
        "severity": severity,
        "component": component,
        "minimal_reproduction": reproduction,
        "expectation": expectation,
        "measured_impact": impact,
        "effect_on_paper_claims": claim_effect,
        "recommended_remediation": remediation,
        "requires_runtime_change": requires_runtime_change,
    }
    try:
        record["measurements"] = runner()
        record["reproduced"] = True
        record["status"] = "REPRODUCED"
    except Exception as exc:  # noqa: BLE001
        record["measurements"] = {"exception": f"{type(exc).__name__}: {exc}"}
        record["traceback"] = traceback.format_exc(limit=6)
        record["reproduced"] = False
        record["status"] = "NOT_REPRODUCED_IN_THIS_REVISION"
    return record


# ---------------------------------------------------------------------------
def measure_attention_parameter_freeze(seed: int) -> Dict[str, Any]:
    """Attention projections are updated in the packed array but not in the layer."""
    from simulator.graph_engine import build_graph
    from simulator.training_engine import TrainingConfig, TrainingEngine

    case = case_registry.get_case("embedding_attention_rnn_dense")
    np.random.seed(seed)
    graph = build_graph(list(case.layers))
    engine = TrainingEngine(
        graph,
        TrainingConfig(
            epochs=1,
            batch_size=1,
            learning_rate=0.05,
            optimizer="sgd",
            loss_function=case.loss_fn,
            shuffle=False,
            dropout_rate=0.0,
        ),
    )
    x = case.make_input()
    target = case.make_target(1)
    layer = graph.layer_instances[2]
    projection_before = {
        name: getattr(layer, name).copy() for name in ("W_q", "W_k", "W_v", "W_o")
    }
    packed_before = graph.weights[1].copy()
    engine.train_batch([(x.copy(), target.copy())])
    packed_after = graph.weights[1].copy()
    projection_after = {
        name: getattr(layer, name) for name in ("W_q", "W_k", "W_v", "W_o")
    }
    deltas = {
        name: float(np.max(np.abs(projection_after[name] - projection_before[name])))
        for name in projection_before
    }
    return {
        "packed_array_max_abs_change": float(np.max(np.abs(packed_after - packed_before))),
        "layer_projection_max_abs_change": deltas,
        "layer_projection_changed": bool(max(deltas.values()) > 0.0),
        "packed_changed": bool(max_abs_error(packed_after, packed_before) > 0.0),
        "sync_report": sync_params_to_layers(graph).to_dict(),
        "consequence": (
            "After any number of optimizer steps the attention projections used by "
            "the forward pass are unchanged, so the attention block does not train."
        ),
    }


def measure_attention_head_count(seed: int) -> Dict[str, Any]:
    """num_heads is accepted and stored but never changes the computation."""
    from simulator.graph_engine import build_graph
    from simulator.layers import LayerConfig

    def make(heads: int):
        np.random.seed(seed)
        return build_graph(
            [
                LayerConfig("input", input_shape=[4, 4]),
                LayerConfig("attention", num_heads=heads, neurons=4),
                LayerConfig("flatten"),
                LayerConfig("output", neurons=2, activation="linear"),
            ]
        )

    one, two = make(1), make(2)
    x = np.linspace(-0.5, 0.5, 16, dtype=np.float32).reshape(4, 4)
    y_one = graph_forward(one, x)
    layer_two = two.layer_instances[1]
    for name in ("W_q", "W_k", "W_v", "W_o"):
        setattr(layer_two, name, getattr(one.layer_instances[1], name).copy())
    y_two = graph_forward(two, x)
    return {
        "stored_num_heads": {
            "one": int(one.layer_instances[1].num_heads),
            "two": int(two.layer_instances[1].num_heads),
        },
        "max_abs_output_difference_with_identical_projections": max_abs_error(y_one, y_two),
        "attn_weights_shape": list(np.shape(two.layer_instances[1].attn_weights)),
        "consequence": (
            "A user selecting two or more heads receives single-head computation "
            "with no warning. Multi-head attention is not implemented."
        ),
    }


def measure_vector_batchnorm_degeneracy(seed: int) -> Dict[str, Any]:
    from simulator.graph_engine import build_graph
    from simulator.layers import LayerConfig

    np.random.seed(seed)
    graph = build_graph(
        [
            LayerConfig("input", neurons=5, input_shape=[5]),
            LayerConfig("batchnorm"),
            LayerConfig("output", neurons=2, activation="linear"),
        ]
    )
    layer = graph.layer_instances[1]
    x_a = np.array([0.4, -0.2, 0.9, -0.7, 0.1], dtype=np.float32)
    x_b = np.array([-0.3, 0.8, -0.6, 0.5, -0.9], dtype=np.float32)
    out_a = np.asarray(layer.forward(x_a), dtype=np.float64)
    out_b = np.asarray(layer.forward(x_b), dtype=np.float64)
    from simulator.backward_engine import backward_full

    back = backward_full(graph, x_a.tolist(), [0.1, 0.2], "mse")
    input_gradient = np.asarray(back["input_gradient"], dtype=np.float64)
    return {
        "output_for_two_different_inputs": {
            "a": out_a.tolist(),
            "b": out_b.tolist(),
            "max_abs_difference": float(np.max(np.abs(out_a - out_b))),
        },
        "output_equals_beta": bool(np.allclose(out_a, np.asarray(layer.beta, dtype=np.float64))),
        "input_gradient_l2": float(np.linalg.norm(input_gradient)),
        "consequence": (
            "With a vector input the normalisation is degenerate: the layer output "
            "equals beta for every input, the input gradient is exactly zero, and no "
            "parameter below the layer can receive gradient through it."
        ),
    }


def measure_forward_route_schema(seed: int) -> Dict[str, Any]:
    from simulator.forward_engine import run_forward_full
    from simulator.graph_engine import build_graph
    from simulator.layers import LayerConfig

    schema: Dict[str, Any] = {}
    try:
        from api.simulator_routes import ForwardRequest

        annotation = str(ForwardRequest.model_fields["input"].annotation)
        schema = {
            "model": "api.simulator_routes.ForwardRequest",
            "input_annotation": annotation,
            "declares_flat_list": "list[float]" in annotation.lower().replace("typing.", ""),
        }
    except Exception as exc:  # noqa: BLE001
        schema = {"error": f"{type(exc).__name__}: {exc}"}

    np.random.seed(seed)
    spatial = build_graph(
        [
            LayerConfig("input", input_shape=[1, 5, 5]),
            LayerConfig("conv2d", filters=1, kernel_size=3, activation="relu"),
            LayerConfig("flatten"),
            LayerConfig("output", neurons=2, activation="linear"),
        ]
    )
    flat = np.linspace(-1.0, 1.0, 25, dtype=np.float32).tolist()
    nested = np.linspace(-1.0, 1.0, 25, dtype=np.float32).reshape(1, 5, 5).tolist()
    route_result: Dict[str, Any] = {}
    try:
        run_forward_full(spatial, flat)
        route_result = {"executed": True}
    except Exception as exc:  # noqa: BLE001
        route_result = {"executed": False, "exception": f"{type(exc).__name__}: {exc}"}
    engine_result: Dict[str, Any] = {}
    try:
        steps, _, _ = run_forward_full(spatial, nested)
        engine_result = {"executed": True, "step_count": len(steps)}
    except Exception as exc:  # noqa: BLE001
        engine_result = {"executed": False, "exception": f"{type(exc).__name__}: {exc}"}
    return {
        "request_schema": schema,
        "route_payload_flat_list": route_result,
        "engine_payload_nested_list": engine_result,
        "consequence": (
            "The forward trace route cannot express the nested inputs that spatial and "
            "sequence graphs require, so those graphs are not executable through the "
            "route even though the engine accepts them directly."
        ),
    }


def measure_architecture_hash_collisions(seed: int) -> Dict[str, Any]:
    from simulator.graph_engine import build_graph
    from simulator.layers import LayerConfig

    variants = {
        "relu_hidden": [
            LayerConfig("input", neurons=4, input_shape=[4]),
            LayerConfig("dense", neurons=2, activation="relu"),
            LayerConfig("output", neurons=2, activation="linear"),
        ],
        "tanh_hidden": [
            LayerConfig("input", neurons=4, input_shape=[4]),
            LayerConfig("dense", neurons=2, activation="tanh"),
            LayerConfig("output", neurons=2, activation="linear"),
        ],
        "three_outputs": [
            LayerConfig("input", neurons=4, input_shape=[4]),
            LayerConfig("dense", neurons=2, activation="relu"),
            LayerConfig("output", neurons=3, activation="linear"),
        ],
        "conv_same_neuron_counts": [
            LayerConfig("input", input_shape=[4, 4, 4]),
            LayerConfig("conv2d", filters=2, kernel_size=2, activation="relu"),
            LayerConfig("flatten"),
            LayerConfig("output", neurons=2, activation="linear"),
        ],
    }
    hashes = {}
    for name, layers in variants.items():
        np.random.seed(seed)
        hashes[name] = build_graph(layers).architecture_hash
    collisions = [
        {"a": a, "b": b, "hash": hashes[a]}
        for a in hashes
        for b in hashes
        if a < b and hashes[a] == hashes[b]
    ]
    return {
        "hashes": hashes,
        "distinct_hashes": len(set(hashes.values())),
        "variants": len(hashes),
        "collisions": collisions,
        "consequence": (
            "architecture_hash cannot identify an architecture: it collides across "
            "distinct layer types, activations and output widths that share a neuron "
            "count signature."
        ),
    }


def measure_dense_batch_boundary(seed: int) -> Dict[str, Any]:
    from simulator.graph_engine import build_graph
    from simulator.layers import LayerConfig

    np.random.seed(seed)
    graph = build_graph(
        [
            LayerConfig("input", neurons=3, input_shape=[3]),
            LayerConfig("dense", neurons=4, activation="tanh"),
            LayerConfig("output", neurons=2, activation="linear"),
        ]
    )
    batch = np.linspace(-0.5, 0.5, 9, dtype=np.float32).reshape(3, 3)
    try:
        out = graph.forward(batch.copy())
        return {"batch_supported": True, "output_shape": list(np.shape(out))}
    except Exception as exc:  # noqa: BLE001
        return {
            "batch_supported": False,
            "exception": f"{type(exc).__name__}: {exc}",
            "consequence": (
                "The dense layer flattens its input, so the batch dimension is not "
                "supported anywhere on the dense path. Reported as a declared boundary."
            ),
        }


def measure_same_stride_padding_semantics(seed: int) -> Dict[str, Any]:
    from lib import reference
    from simulator.graph_engine import build_graph
    from simulator.layers import LayerConfig

    np.random.seed(seed)
    graph = build_graph(
        [
            LayerConfig("input", input_shape=[1, 6, 6]),
            LayerConfig("conv2d", filters=2, kernel_size=3, stride=2, padding="same", activation="tanh"),
            LayerConfig("flatten"),
            LayerConfig("output", neurons=1, activation="linear"),
        ]
    )
    x = np.random.default_rng(seed).uniform(-0.8, 0.8, size=(1, 6, 6)).astype(np.float32)
    simulated = graph_forward(graph, x)
    conv_layer = graph.layer_instances[1]
    simulator_spatial = list(np.shape(conv_layer.Z))
    # Only the convolution is compared here: with the conventional geometry the
    # flattened width differs, so the rest of the graph cannot be built.
    standard_spatial = list(
        np.shape(
            reference.conv2d(
                np.asarray(x, dtype=np.float64),
                np.asarray(conv_layer.K, dtype=np.float64),
                np.asarray(conv_layer.b, dtype=np.float64),
                stride=conv_layer.cfg.stride,
                padding="same",
                activation=conv_layer.cfg.activation,
            )
        )
    )
    same_geometry = reference.reference_forward(graph, x, strict_padding=False).output.reshape(-1)
    return {
        "simulator_conv_output_shape": simulator_spatial,
        "standard_same_stride_conv_output_shape": standard_spatial,
        "reference_with_simulator_padding_max_abs_error": max_abs_error(simulated, same_geometry),
        "consequence": (
            "With stride greater than one the simulator's 'same' padding does not "
            "produce the conventional ceil(in/stride) output size, so the declared "
            "padding mode does not mean what a framework user expects."
        ),
    }


# ---------------------------------------------------------------------------
def build_defects(seed: int) -> List[Dict[str, Any]]:
    return [
        defect(
            "D-01",
            "Attention projection updates never reach the attention layer",
            "HIGH",
            "backend/simulator/layers/attention.py",
            "Build Embedding->Attention(1 head)->RNN->sigmoid Dense with a fixed seed, run "
            "one TrainingEngine.train_batch with SGD at lr=0.05, and compare the layer's "
            "W_q/W_k/W_v/W_o before and after.",
            "A parameter that the optimizer updates must change in the layer that consumes it.",
            lambda: measure_attention_parameter_freeze(seed),
            "The packed graph array changes while all four attention projections are "
            "unchanged, so attention contributes nothing to the learned function.",
            "Blocks any claim that the attention path trains, and weakens the existing "
            "one-batch attention training contract, which observes the packed array only.",
            "Correct AttentionLayer.set_params to accept the rank-3 packed layout its own "
            "params() exposes (or have params() return the rank-4 layout set_params "
            "expects), and add a regression test asserting the projections move.",
        ),
        defect(
            "D-02",
            "num_heads is accepted but has no effect",
            "MEDIUM",
            "backend/simulator/layers/attention.py",
            "Build two graphs differing only in num_heads (1 versus 2), copy identical "
            "projections into both, and compare outputs.",
            "A requested architecture parameter must change the computation.",
            lambda: measure_attention_head_count(seed),
            "Outputs are identical and the attention weight matrix has a single head's shape.",
            "The system must not be described as supporting multi-head attention.",
            "Either implement head splitting and per-head concatenation, or reject "
            "num_heads > 1 in validate_layers with an explicit error.",
        ),
        defect(
            "D-03",
            "Batch normalisation degenerates on vector inputs",
            "HIGH",
            "backend/simulator/layers/batchnorm.py",
            "Build Input(5)->BatchNorm->Output(2), feed two different 5-vectors, and compare "
            "the BatchNorm output and the backward input gradient.",
            "A normalisation layer should transform its input and propagate gradient.",
            lambda: measure_vector_batchnorm_degeneracy(seed),
            "The output equals beta for every input and the input gradient is exactly zero, "
            "so no parameter below the layer can train through it.",
            "Any vector-input BatchNorm graph must be reported as degenerate; the "
            "parameter-count and FLOP estimates in validate_layers remain valid but the "
            "layer is functionally an identity-plus-bias in that regime.",
            "Reject vector-input BatchNorm at validation time, or implement a running-statistics "
            "inference mode so the layer is usable outside a spatial context.",
        ),
        defect(
            "D-04",
            "Forward trace route cannot express spatial or sequence inputs",
            "BLOCKING",
            "backend/api/simulator_routes.py, backend/simulator/forward_engine.py",
            "POST /api/simulator/forward/full with a List[float] payload for a graph whose "
            "input_shape is [1,5,5]; then call run_forward_full directly with a nested payload.",
            "A route must be able to express the inputs of the architectures it accepts.",
            lambda: measure_forward_route_schema(seed),
            "The route raises an unpacking error while the engine accepts the same graph "
            "with a nested payload, so the visible forward-execution path is limited to "
            "flat vector inputs.",
            "The paper must not claim that the simulator's forward/trace workflow covers "
            "convolutional or recurrent architectures through the application route.",
            "Change the request model to accept a nested array (or accept an explicit "
            "input_shape alongside a flat vector), and add a route test per input rank.",
        ),
        defect(
            "D-05",
            "architecture_hash collides across distinct architectures",
            "MEDIUM",
            "backend/simulator/graph_engine.py",
            "Build four architectures that share a neuron-count signature but differ in "
            "activation, output width or layer type, and compare architecture_hash.",
            "An architecture identifier must distinguish architectures.",
            lambda: measure_architecture_hash_collisions(seed),
            "Distinct architectures produce the same 12-character hash.",
            "The identifier must not be described as an architecture fingerprint or used "
            "as an experiment identity.",
            "Hash a canonical serialisation of the full layer configuration, including "
            "layer types, activations and non-neuron fields.",
        ),
        defect(
            "D-06",
            "Dense path has no batch dimension",
            "INFORMATIONAL",
            "backend/simulator/layers/dense.py",
            "Call graph.forward with a (3, 3) array on Input(3)->Dense(4)->Output(2).",
            "A leading batch dimension is either supported or explicitly unsupported.",
            lambda: measure_dense_batch_boundary(seed),
            "The dense layer reshapes its input to one dimension and the matmul fails.",
            "Reported as a declared boundary rather than a defect; no batching claim is made.",
            "Document the boundary in the API contract, or reshape explicitly to "
            "(batch, features) and propagate batched gradients.",
            requires_runtime_change=False,
        ),
        defect(
            "D-07",
            "'same' padding with stride greater than one is non-standard",
            "MEDIUM",
            "backend/simulator/layers/conv2d.py",
            "Compare a stride-2 'same' convolution against an independent float64 "
            "reference using both the conventional and the simulator's geometry.",
            "A declared padding mode should match the conventional meaning of that mode.",
            lambda: measure_same_stride_padding_semantics(seed),
            "The simulator produces a 5x5 output where the conventional rule gives 3x3; "
            "with the simulator's own geometry the arithmetic matches the reference.",
            "Convolution configurations using 'same' with stride > 1 are not comparable "
            "with framework implementations and must not be presented as such.",
            "Adopt the framework convention (ceil(in/stride)) or rename the mode so the "
            "non-standard geometry is explicit.",
        ),
    ]


def run(repo_root: Path, seed: int) -> Dict[str, Any]:
    defects = build_defects(seed)
    for record in defects:
        if record["status"] == "REPRODUCED":
            expectation = record["expectation"]
            measured = record["measurements"]
            if record["defect_id"] == "D-01":
                record["expectation_met"] = bool(measured["layer_projection_changed"])
            elif record["defect_id"] == "D-02":
                record["expectation_met"] = bool(
                    (measured["max_abs_output_difference_with_identical_projections"] or 0.0) > 0.0
                )
            elif record["defect_id"] == "D-03":
                record["expectation_met"] = not bool(
                    measured["output_equals_beta"] and measured["input_gradient_l2"] == 0.0
                )
            elif record["defect_id"] == "D-04":
                record["expectation_met"] = bool(measured["route_payload_flat_list"].get("executed"))
            elif record["defect_id"] == "D-05":
                record["expectation_met"] = not measured["collisions"]
            elif record["defect_id"] == "D-06":
                record["expectation_met"] = bool(measured["batch_supported"])
            elif record["defect_id"] == "D-07":
                record["expectation_met"] = bool(
                    measured["simulator_conv_output_shape"]
                    == measured["standard_same_stride_conv_output_shape"]
                )
            record["defect_confirmed"] = bool(record.get("expectation_met") is False)
        else:
            record["defect_confirmed"] = False
    confirmed = [d for d in defects if d.get("defect_confirmed")]
    summary = {
        "defects_total": len(defects),
        "defects_reproduced": sum(1 for d in defects if d["status"] == "REPRODUCED"),
        "defects_confirmed": len(confirmed),
        "confirmed_ids": [d["defect_id"] for d in confirmed],
        "severity_counts": {
            level: sum(1 for d in confirmed if d["severity"] == level)
            for level in ("BLOCKING", "HIGH", "MEDIUM", "LOW", "INFORMATIONAL")
        },
        "confirmed_requiring_runtime_change": [
            d["defect_id"] for d in confirmed if d["requires_runtime_change"]
        ],
        "status": "REGISTERED" if confirmed else "NO_DEFECTS_CONFIRMED",
    }
    return {"defects": defects, "summary": summary}


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
        notes={
            "runtime_source_modified": False,
            "policy": "runtime defects are reproduced, quantified and documented; none are fixed here",
        },
    )
    try:
        result = run(repo_root, args.seed)
    except Exception as exc:  # noqa: BLE001
        print(f"defect register failed: {exc}", file=sys.stderr)
        traceback.print_exc()
        provenance.exit_code = 2
        provenance.finished_at_utc = utc_now_iso()
        return 2

    provenance.exit_code = 0
    provenance.finished_at_utc = utc_now_iso()

    writer = EvidenceWriter(
        experiment_id=EXPERIMENT_ID,
        repo_root=repo_root,
        provenance=provenance,
        seed=args.seed,
        allow_overwrite=args.allow_overwrite,
        configuration={
            "severity_scale": ["BLOCKING", "HIGH", "MEDIUM", "LOW", "INFORMATIONAL"],
            "policy": "research-only; runtime source is read-only for this phase",
        },
        summary_payload=result["summary"],
        extra_raw={"defects": result["defects"]},
        tables={
            "defect_register": [
                {
                    "defect_id": d["defect_id"],
                    "title": d["title"],
                    "severity": d["severity"],
                    "component": d["component"],
                    "status": d["status"],
                    "defect_confirmed": d.get("defect_confirmed"),
                    "requires_runtime_change": d["requires_runtime_change"],
                    "effect_on_paper_claims": d["effect_on_paper_claims"],
                    "recommended_remediation": d["recommended_remediation"],
                }
                for d in result["defects"]
            ]
        },
        interpretation={
            "purpose": "Reproduced defect register with measured impact, claim effect and remediation.",
            "policy": (
                "No runtime source was modified. Every entry is a reproduced observation "
                "on the pinned revision with a minimal reproduction, not a source-inspection guess."
            ),
            "severity": (
                "Severity is assigned from the measured effect on the simulator's own outputs: "
                "BLOCKING prevents a user-facing path from working for a supported architecture."
            ),
        },
        limitations=[
            "Defects are recorded for this revision only.",
            "A defect listed here is not fixed by this research phase; fixing it requires "
            "explicit authorisation to modify runtime source.",
            "Defects outside the simulator, for example in the frontend or the model registry, "
            "are covered by separate audits.",
        ],
    )
    manifest = writer.write()
    summary = result["summary"]
    print(
        f"[{EXPERIMENT_ID}] defects={summary['defects_total']} "
        f"confirmed={summary['defects_confirmed']} "
        f"ids={','.join(summary['confirmed_ids'])}"
    )
    print(f"evidence: research/evidence/{EXPERIMENT_ID}/{manifest['source_sha']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())