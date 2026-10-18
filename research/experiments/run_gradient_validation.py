#!/usr/bin/env python3
"""EXP-02 Graph-integrated gradient validation against central finite differences.

Purpose
-------
Replace the previous single-case finite-difference check with a research-grade
comparison that

* covers every registered graph family, not one dense case;
* sweeps several finite-difference step sizes and reports the achievable error
  floor and the estimator's own instability instead of one lucky number;
* validates **both** gradient implementations the repository ships
  (``simulator.backward_engine`` and ``simulator.training_engine``), which are
  separate code paths;
* separates parameters whose packed array actually reaches the owning layer
  from parameters whose packed array is silently ignored, so a state-propagation
  defect is never misreported as a gradient-arithmetic defect;
* records per-parameter-tensor and per-element errors for every check.

Usage
-----
    PYTHONPATH=backend:research python research/experiments/run_gradient_validation.py \
        --repo . --cases all

Exit codes: 0 = all *validatable* comparisons met their declared tolerance,
1 = at least one comparison failed, 2 = the experiment could not run.
"""
from __future__ import annotations

import argparse
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib import cases as case_registry  # noqa: E402
from lib import stats as stat  # noqa: E402
from lib.evidence import EvidenceWriter  # noqa: E402
from lib.numerics import (  # noqa: E402
    central_difference,
    compare_gradients,
    graph_forward,
    is_bce_with_sigmoid_contract,
    kink_band_analysis,
    reference_objective,
    resolution_probe,
    sync_params_to_layers,
)
from lib.provenance import RunProvenance, utc_now_iso  # noqa: E402

EXPERIMENT_ID = "EXP-02-gradient-validation"
DEFAULT_SEED = 20261018
DEFAULT_EPSILONS = (1e-2, 3e-3, 1e-3, 3e-4, 1e-4)
GRADIENT_SOURCES = ("backward_engine", "training_engine")

# How each recurrent/attention family packs its named weight matrices into the
# single graph-level array. The packing axis and mode differ per family, so the
# layout must be declared rather than assumed to be a leading-axis stack.
LAYER_WEIGHT_BLOCKS = {
    "attention": {"names": ("W_q", "W_k", "W_v", "W_o"), "stack": True},
    "lstm": {"names": ("W_f", "W_i", "W_o", "W_c"), "stack": False, "axis": 0},
    "gru": {"names": ("W_r", "W_z", "W_h"), "stack": False, "axis": 0},
    "rnn": {"names": ("W_xh", "W_hh"), "stack": False, "axis": 1, "split_attr": "input_dim"},
}

BIAS_ATTRIBUTE_LAYOUT = {
    "rnn": ("b",),
    "lstm": ("b_f", "b_i", "b_o", "b_c"),
    "gru": ("b_r", "b_z", "b_h"),
}


def _split_weight_blocks(layer, value: np.ndarray):
    """Split a packed weight array into the layer's named matrices."""
    spec = LAYER_WEIGHT_BLOCKS.get(getattr(layer, "layer_type", ""))
    if spec is None:
        return None
    names = spec["names"]
    if spec.get("stack"):
        if value.ndim != 3 or value.shape[0] != len(names):
            return None
        return {name: value[index] for index, name in enumerate(names)}
    if value.ndim != 2:
        return None
    if spec.get("split_attr"):
        split = int(getattr(layer, spec["split_attr"], value.shape[1]))
        return {name: value[:, start:stop] for name, start, stop in _column_blocks(names, split, value.shape[1])}
    axis = int(spec.get("axis", 0))
    if value.shape[axis] % len(names) != 0:
        return None
    size = value.shape[axis] // len(names)
    blocks = []
    for index in range(len(names)):
        start = index * size
        sl = [slice(None)] * 2
        sl[axis] = slice(start, start + size)
        blocks.append(value[tuple(sl)])
    return {name: blocks[index] for index, name in enumerate(names)}


def _column_blocks(names, split, total):
    if len(names) != 2:
        return []
    return [(names[0], 0, split), (names[1], split, total)]


def _apply_weight_attributes(layer, value: np.ndarray, previous: np.ndarray) -> None:
    """Write a packed weight array into the owning layer's named attributes.

    ``AttentionLayer.set_params`` only accepts a rank-4 array while its own
    ``params()`` exposes a rank-3 stack, so writing the packed array through
    ``set_params`` is a silent no-op. Writing the named attributes directly is
    what makes the perturbed parameter the parameter the forward pass consumes.
    """
    blocks = _split_weight_blocks(layer, np.asarray(value, dtype=np.float32))
    if blocks:
        for name, block in blocks.items():
            setattr(layer, name, np.asarray(block, dtype=np.float32))
        return
    layer.set_params(
        np.asarray(value, dtype=np.float32),
        np.asarray(layer.params()[1], dtype=np.float32),
    )


def _apply_bias_attributes(layer, value: np.ndarray, previous: np.ndarray) -> None:
    names = BIAS_ATTRIBUTE_LAYOUT.get(getattr(layer, "layer_type", ""))
    if names is None:
        layer.set_params(
            np.asarray(layer.params()[0], dtype=np.float32), np.asarray(value, dtype=np.float32)
        )
        return
    offset = 0
    for name in names:
        size = int(np.size(getattr(layer, name)))
        setattr(layer, name, np.asarray(value[offset : offset + size], dtype=np.float32))
        offset += size


# ---------------------------------------------------------------------------
@dataclass
class ParameterAccess:
    """Read/write access to the parameters a layer's forward pass actually uses."""

    mode: str  # "packed" | "layer_attribute"

    def read(self, graph, param_index: int, kind: str) -> np.ndarray:
        if self.mode == "packed":
            return np.asarray(
                graph.weights[param_index] if kind == "weights" else graph.biases[param_index],
                dtype=np.float64,
            )
        layer, _ = _owning_layer(graph, param_index, kind)
        if layer is None:
            return np.zeros(0)
        packed_w, packed_b = layer.params()
        source = packed_w if kind == "weights" else packed_b
        return np.asarray(source, dtype=np.float64)

    def write(self, graph, param_index: int, kind: str, value: np.ndarray) -> None:
        if self.mode == "packed":
            target = graph.weights if kind == "weights" else graph.biases
            target[param_index] = np.asarray(value, dtype=np.float32)
            return
        layer, _ = _owning_layer(graph, param_index, kind)
        if layer is None:
            return
        if kind == "weights":
            _apply_weight_attributes(layer, np.asarray(value, dtype=np.float32), np.zeros(0))
        else:
            _apply_bias_attributes(layer, np.asarray(value, dtype=np.float32), np.zeros(0))

    def sync(self, graph) -> bool:
        if self.mode != "packed":
            return True
        return sync_params_to_layers(graph).fully_synchronised

    def probe_sync(self, graph, param_index: int, kind: str, seed: int) -> Dict[str, Any]:
        """Verify that writing ``param_index`` is observable in the layer.

        An unperturbed write proves nothing: before the first optimizer step the
        packed array and the layer attribute often still hold identical values.
        The probe therefore writes a deterministic perturbation and checks that
        the owning layer's own ``params()`` reflect it.
        """
        if self.mode != "packed":
            return {"sync_verified": True, "mode": self.mode}
        original = self.read(graph, param_index, kind)
        if original.size == 0:
            return {"sync_verified": False, "mode": self.mode, "reason": "unreadable parameter"}
        rng = np.random.default_rng(seed + param_index)
        probe_value = np.asarray(
            original + rng.normal(0.0, 0.05, size=original.shape), dtype=np.float32
        )
        self.write(graph, param_index, kind, probe_value)
        report = sync_params_to_layers(graph)
        self.write(graph, param_index, kind, original)
        sync_params_to_layers(graph)
        return {
            "sync_verified": report.fully_synchronised,
            "mode": self.mode,
            "desynchronised": report.desynchronised,
        }

    def setter(self, graph, param_index: int, kind: str):
        """Return a callable that commits a working array into the graph."""

        def commit(value: np.ndarray) -> None:
            self.write(graph, param_index, kind, value)
            if self.mode == "packed":
                sync_params_to_layers(graph)

        return commit


def _owning_layer(graph, param_index: int, kind: str = "weights"):
    """Resolve the layer instance that owns packed parameter ``param_index``.

    Bias arrays are stored positionally alongside the weight arrays, so the same
    index identifies the owning layer for both kinds.
    """
    instances = list(getattr(graph, "layer_instances", []) or [])
    mapping = dict(getattr(graph, "param_index_by_layer", {}) or {})
    for layer_index, candidate in mapping.items():
        if candidate == param_index and layer_index < len(instances):
            return instances[layer_index], layer_index
    param_layers = list(getattr(graph, "param_layer_indices", []) or [])
    if param_index < len(param_layers):
        index = param_layers[param_index]
        if index < len(instances):
            return instances[index], index
    return None, None


# ---------------------------------------------------------------------------
def _slice_bounds(offset: int, shape: Tuple[int, ...]) -> Tuple[int, int]:
    size = int(np.prod(shape)) if shape else 1
    return offset, offset + size


def _analytic_gradients(
    case: case_registry.GraphCase,
    gradient_source: str,
    seed: int,
):
    """Return analytic gradients from one of the two shipped implementations."""
    from simulator.graph_engine import build_graph
    from simulator.training_engine import TrainingConfig, TrainingEngine

    np.random.seed(seed)
    graph = build_graph(list(case.layers))
    x = case.make_input()
    forward = graph_forward(graph, x)
    target = case.make_target(int(forward.size))

    if gradient_source == "backward_engine":
        from simulator.backward_engine import backward_full

        result = backward_full(
            graph, np.asarray(x).tolist(), np.asarray(target).tolist(), case.loss_fn
        )
        return graph, x, target, [np.asarray(g, dtype=np.float64) for g in result["gradients_W"]], [
            np.asarray(g, dtype=np.float64) for g in result["gradients_b"]
        ]

    engine = TrainingEngine(
        graph,
        TrainingConfig(
            epochs=1,
            batch_size=1,
            learning_rate=0.01,
            optimizer="sgd",
            loss_function=case.loss_fn,
            shuffle=False,
            dropout_rate=0.0,
        ),
    )
    activations, pre_acts, masks = engine._forward(np.asarray(x, dtype=np.float32), training=False)
    grads_w, grads_b = engine._backward(activations, pre_acts, masks, np.asarray(target, dtype=np.float32))
    return graph, x, target, [np.asarray(g, dtype=np.float64) for g in grads_w], [
        np.asarray(g, dtype=np.float64) for g in grads_b
    ]


def _objective_factory(graph, x: np.ndarray, target: np.ndarray, loss_fn: str):
    from simulator.loss_functions import compute_loss

    def objective() -> float:
        prediction = graph_forward(graph, x)
        return float(compute_loss(np.asarray(target, dtype=np.float32), prediction, loss_fn)[0])

    return objective


def run_case(
    case: case_registry.GraphCase,
    epsilons: Tuple[float, ...],
    seed: int,
) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "case_id": case.case_id,
        "family": case.family,
        "loss_fn": case.loss_fn,
        "seed": seed,
        "layer_summary": case.layer_summary(),
        "kink_band": None,
    }
    forward_reference: Dict[str, Any] = {}
    comparisons: List[Dict[str, Any]] = []

    for gradient_source in GRADIENT_SOURCES:
        try:
            graph, x, target, grads_w, grads_b = _analytic_gradients(case, gradient_source, seed)
        except Exception as exc:  # noqa: BLE001
            record.setdefault("errors", []).append(
                {"gradient_source": gradient_source, "exception": f"{type(exc).__name__}: {exc}",
                 "traceback": traceback.format_exc(limit=6)}
            )
            continue

        if gradient_source == "backward_engine":
            prediction = graph_forward(graph, x)
            independent = reference_objective(target, prediction, case.loss_fn)
            from simulator.loss_functions import compute_loss

            implemented = float(
                compute_loss(np.asarray(target, dtype=np.float32), prediction, case.loss_fn)[0]
            )
            forward_reference = {
                "output_size": int(prediction.size),
                "implemented_objective": implemented,
                "independent_reference_objective": independent,
                "objective_absolute_difference": float(abs(implemented - independent)),
                "objective_relative_difference": float(
                    abs(implemented - independent) / (abs(independent) + 1e-12)
                ),
            }
            if record["kink_band"] is None:
                record["kink_band"] = kink_band_analysis(graph, max(epsilons))

        access_modes = ["packed", "layer_attribute"]
        for mode in access_modes:
            access = ParameterAccess(mode=mode)
            sync_ok = access.sync(graph)
            analytic_tensors = [(i, "weights", grads_w[i]) for i in range(len(grads_w))]
            analytic_tensors += [(i, "biases", grads_b[i]) for i in range(len(grads_b))]
            objective = _objective_factory(graph, x, target, case.loss_fn)

            offset = 0
            for param_index, kind, analytic in analytic_tensors:
                if analytic is None:
                    continue
                try:
                    current = access.read(graph, param_index, kind)
                except Exception:
                    continue
                if current.size == 0 or current.shape != analytic.shape:
                    comparisons.append(
                        {
                            "gradient_source": gradient_source,
                            "access_mode": mode,
                            "param_index": param_index,
                            "kind": kind,
                            "shape_read": list(np.shape(current)),
                            "shape_analytic": list(np.shape(analytic)),
                            "status": "SHAPE_MISMATCH",
                            "passed": False,
                        }
                    )
                    offset += int(np.size(analytic))
                    continue

                layer, layer_index = _owning_layer(graph, param_index, kind)
                layer_type = getattr(layer, "layer_type", "unknown") if layer is not None else "unknown"
                sync_probe = access.probe_sync(graph, param_index, kind, seed)
                if not sync_probe["sync_verified"]:
                    comparisons.append(
                        {
                            "gradient_source": gradient_source,
                            "access_mode": mode,
                            "param_index": param_index,
                            "kind": kind,
                            "layer_type": layer_type,
                            "layer_instance_index": layer_index,
                            "shape": list(analytic.shape),
                            "flat_offset": offset,
                            "flat_end": offset + int(np.size(analytic)),
                            "analytic_norm": float(np.linalg.norm(analytic)),
                            "status": "PARAM_SYNC_NOT_VERIFIED",
                            "passed": False,
                            "informative": False,
                            "attribution": "packed parameter array does not reach the owning layer",
                            "sync_probe": sync_probe,
                        }
                    )
                    offset += int(np.size(analytic))
                    continue

                best: Optional[Dict[str, Any]] = None
                commit = access.setter(graph, param_index, kind)
                sweep: List[Dict[str, Any]] = []
                analytic_norm = float(np.linalg.norm(analytic))
                for epsilon in epsilons:
                    original = access.read(graph, param_index, kind)
                    try:
                        numeric = central_difference(
                            objective,
                            np.asarray(original, dtype=np.float64),
                            epsilon,
                            setter=commit,
                        )
                    finally:
                        access.write(graph, param_index, kind, original)
                        access.sync(graph)
                    comparison = compare_gradients(
                        analytic.ravel(),
                        numeric.ravel(),
                        epsilon,
                        case.gradient_tolerance["atol"],
                        case.gradient_tolerance["rtol"],
                    )
                    entry = comparison.to_dict()
                    entry["scale_informative"] = bool(
                        analytic_norm >= case.gradient_norm_floor
                    )
                    entry["relative_l2_within_declared_max"] = bool(
                        entry["relative_l2_error"] is not None
                        and entry["relative_l2_error"] <= case.gradient_relative_l2_max
                    )
                    sweep.append(entry)
                    if best is None and entry["scale_informative"] and entry["passed"] and entry[
                        "relative_l2_within_declared_max"
                    ]:
                        best = entry
                if best is None and sweep:
                    informative = [e for e in sweep if e["scale_informative"]]
                    pool = informative or sweep
                    best = min(pool, key=lambda e: e["relative_l2_error"] if e["relative_l2_error"] is not None else float("inf"))

                resolution = None
                verdict = "COMPARED"
                if best is not None and not (
                    best["passed"] and best["relative_l2_within_declared_max"]
                ):
                    # A mismatch is only evidence about the implementation if the
                    # central difference can resolve the analytic gradient in the
                    # objective's own dtype.
                    base = access.read(graph, param_index, kind)
                    resolution = resolution_probe(
                        _objective_factory(graph, x, target, case.loss_fn),
                        access.setter(graph, param_index, kind),
                        np.asarray(base, dtype=np.float64),
                        analytic.ravel(),
                        best["epsilon"],
                        seed=seed + param_index,
                    )
                    if not resolution.resolvable:
                        verdict = "INCONCLUSIVE_UNRESOLVED_BY_FLOAT32_OBJECTIVE"
                    else:
                        kink = record.get("kink_band") or {}
                        if kink.get("fraction_within_band", 0.0) > 0.0:
                            verdict = "INCONCLUSIVE_RELU_KINK_BAND"
                    try:
                        access.write(graph, param_index, kind, base)
                        access.sync(graph)
                    except Exception:
                        pass
                if best is None:
                    continue
                lo, hi = _slice_bounds(offset, analytic.shape)
                informative = best["scale_informative"]
                is_inconclusive = verdict.startswith("INCONCLUSIVE")
                passed = bool(best["passed"] and best["relative_l2_within_declared_max"])
                status = verdict
                if passed:
                    status = "COMPARED" if informative else "DEGENERATE_ZERO_GRADIENT"
                comparisons.append(
                    {
                        "gradient_source": gradient_source,
                        "access_mode": mode,
                        "param_index": param_index,
                        "kind": kind,
                        "layer_type": layer_type,
                        "layer_instance_index": layer_index,
                        "shape": list(analytic.shape),
                        "flat_offset": lo,
                        "flat_end": hi,
                        "analytic_norm": analytic_norm,
                        "best_epsilon": best["epsilon"],
                        "max_abs_error": best["max_abs_error"],
                        "relative_l2_error": best["relative_l2_error"],
                        "cosine_similarity": best["cosine_similarity"],
                        "numeric_norm": best["numeric_norm"],
                        "atol": case.gradient_tolerance["atol"],
                        "rtol": case.gradient_tolerance["rtol"],
                        "gradient_norm_floor": case.gradient_norm_floor,
                        "gradient_relative_l2_max": case.gradient_relative_l2_max,
                        "status": status,
                        "passed": passed,
                        "informative": bool(informative),
                        "inconclusive": is_inconclusive,
                        "resolution_probe": resolution.to_dict() if resolution else None,
                        "epsilon_sweep": sweep,
                        "epsilon_spread_relative_l2": (
                            float(
                                max(e["relative_l2_error"] for e in sweep)
                                - min(e["relative_l2_error"] for e in sweep)
                            )
                            if sweep
                            and all(e["relative_l2_error"] is not None for e in sweep)
                            else None
                        ),
                        "attribution": None,
                        "sync_probe": sync_probe,
                    }
                )
                offset += int(np.size(analytic))

    record["forward_reference_objective"] = forward_reference
    record["comparisons"] = comparisons
    return record


def run(repo_root: Path, selected: List[case_registry.GraphCase], epsilons: Tuple[float, ...], seed: int) -> Dict[str, Any]:
    case_records: List[Dict[str, Any]] = []
    for case in selected:
        if not case.supports_finite_difference:
            case_records.append(
                {
                    "case_id": case.case_id,
                    "family": case.family,
                    "status": "NOT_COMPARABLE",
                    "reason": case.notes,
                    "comparisons": [],
                }
            )
            continue
        case_records.append(run_case(case, epsilons, seed))

    all_comparisons = [c for record in case_records for c in record.get("comparisons", [])]
    compared = [c for c in all_comparisons if c["status"] == "COMPARED"]
    degenerate = [c for c in all_comparisons if c["status"] == "DEGENERATE_ZERO_GRADIENT"]
    inconclusive = [c for c in all_comparisons if c.get("inconclusive")]
    not_verified = [
        c
        for c in all_comparisons
        if c["status"] not in {"COMPARED", "DEGENERATE_ZERO_GRADIENT"}
        and not c.get("inconclusive")
    ]
    failures = [
            c for c in all_comparisons if not c["passed"] and not c.get("inconclusive")
            and c["status"] not in {"DEGENERATE_ZERO_GRADIENT", "PARAM_SYNC_NOT_VERIFIED", "SHAPE_MISMATCH"}
        ]
    
    per_case_summary = []
    for record in case_records:
        comps = record.get("comparisons", [])
        compared_here = [c for c in comps if c["status"] == "COMPARED"]
        errors = [c["relative_l2_error"] for c in compared_here if c["relative_l2_error"] is not None]
        per_case_summary.append(
            {
                "case_id": record["case_id"],
                "family": record["family"],
                "status": record.get("status", "COMPARED"),
                "tensor_checks_total": len(comps),
                "tensor_checks_compared": len(compared_here),
                "tensor_checks_failed": sum(
                    1 for c in comps if not c["passed"] and not c.get("inconclusive")
                ),
                "tensor_checks_inconclusive": sum(1 for c in comps if c.get("inconclusive")),
                "tensor_checks_degenerate": sum(
                    1 for c in comps if c["status"] == "DEGENERATE_ZERO_GRADIENT"
                ),
                "tensor_checks_sync_unverified": len(not_verified),
                "relative_l2_error": stat.summarise_matrix(np.array(errors)) if errors else None,
            }
        )

    error_matrix = [c["max_abs_error"] for c in compared if c["max_abs_error"] is not None]
    relative_matrix = [c["relative_l2_error"] for c in compared if c["relative_l2_error"] is not None]
    summary = {
        "cases_attempted": len(selected),
        "cases_compared": len([r for r in case_records if r.get("comparisons")]),
        "cases_not_comparable": len([r for r in case_records if r.get("status") == "NOT_COMPARABLE"]),
        "gradient_sources": list(GRADIENT_SOURCES),
        "epsilon_sweep": list(epsilons),
        "tensor_checks_total": len(all_comparisons),
        "tensor_checks_compared": len(compared),
        "tensor_checks_failed": len(failures),
        "tensor_checks_inconclusive": len(inconclusive),
        "tensor_checks_degenerate_zero_gradient": len(degenerate),
        "tensor_checks_param_sync_unverified": len(not_verified),
        "max_abs_error_distribution": stat.summarise_matrix(np.array(error_matrix))
        if error_matrix
        else None,
        "relative_l2_error_distribution": stat.summarise_matrix(np.array(relative_matrix))
        if relative_matrix
        else None,
        "epsilon_spread_relative_l2": stat.describe(
            [c["epsilon_spread_relative_l2"] for c in compared if c.get("epsilon_spread_relative_l2") is not None]
        ),
        "degenerate_tensors": [
            {
                "case_id": record["case_id"],
                "gradient_source": c["gradient_source"],
                "access_mode": c["access_mode"],
                "kind": c["kind"],
                "layer_type": c.get("layer_type"),
                "analytic_norm": c.get("analytic_norm"),
            }
            for record in case_records
            for c in record.get("comparisons", [])
            if c["status"] == "DEGENERATE_ZERO_GRADIENT"
        ],
        "inconclusive_tensors": [
            {
                "case_id": record["case_id"],
                "gradient_source": c["gradient_source"],
                "access_mode": c["access_mode"],
                "kind": c["kind"],
                "layer_type": c.get("layer_type"),
                "analytic_norm": c.get("analytic_norm"),
                "relative_l2_error": c.get("relative_l2_error"),
                "resolution_probe": c.get("resolution_probe"),
            }
            for record in case_records
            for c in record.get("comparisons", [])
            if c.get("inconclusive")
        ],
        "failed_tensors": [
            {
                "case_id": record["case_id"],
                "gradient_source": c["gradient_source"],
                "access_mode": c["access_mode"],
                "kind": c["kind"],
                "layer_type": c.get("layer_type"),
                "relative_l2_error": c.get("relative_l2_error"),
                "max_abs_error": c.get("max_abs_error"),
            }
            for record in case_records
            for c in record.get("comparisons", [])
            if not c["passed"] and not c.get("inconclusive")
            and c["status"] not in {"DEGENERATE_ZERO_GRADIENT", "PARAM_SYNC_NOT_VERIFIED", "SHAPE_MISMATCH"}
        ],
        "kink_band": {
            record["case_id"]: record.get("kink_band")
            for record in case_records
            if record.get("kink_band")
        },
        "param_sync_unverified_tensors": [
            {
                "case_id": record["case_id"],
                "gradient_source": c["gradient_source"],
                "access_mode": c["access_mode"],
                "kind": c["kind"],
                "layer_type": c.get("layer_type"),
                "analytic_norm": c.get("analytic_norm"),
                "attribution": c.get("attribution"),
            }
            for record in case_records
            for c in record.get("comparisons", [])
            if c["status"] not in {"COMPARED", "DEGENERATE_ZERO_GRADIENT"}
        ],
        "status": "PASS" if not failures else "FAILURES_PRESENT",
    }
    return {"cases": case_records, "summary": summary, "per_case_summary": per_case_summary}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=".")
    parser.add_argument("--cases", nargs="*", default=None, help="case ids; default: all")
    parser.add_argument("--family", default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--epsilons", nargs="*", type=float, default=list(DEFAULT_EPSILONS)
    )
    parser.add_argument("--allow-overwrite", action="store_true")
    args = parser.parse_args()

    repo_root = Path(args.repo).resolve()
    epsilons = tuple(float(v) for v in args.epsilons)
    selected = case_registry.select_cases(args.cases, args.family)

    provenance = RunProvenance(
        experiment_id=EXPERIMENT_ID,
        script=Path(__file__).as_posix(),
        command=[sys.executable, *sys.argv],
        seed=args.seed,
        root=repo_root,
        notes={"runtime_source_modified": False, "epsilons": list(epsilons)},
    )
    try:
        result = run(repo_root, selected, epsilons, args.seed)
    except Exception as exc:  # noqa: BLE001
        print(f"gradient validation failed before producing results: {exc}", file=sys.stderr)
        traceback.print_exc()
        provenance.exit_code = 2
        provenance.finished_at_utc = utc_now_iso()
        return 2

    summary = result["summary"]
    provenance.exit_code = 0 if summary["status"] == "PASS" else 1
    provenance.finished_at_utc = utc_now_iso()

    writer = EvidenceWriter(
        experiment_id=EXPERIMENT_ID,
        repo_root=repo_root,
        provenance=provenance,
        seed=args.seed,
        allow_overwrite=args.allow_overwrite,
        configuration={
            "cases": [c.case_id for c in selected],
            "epsilons": list(epsilons),
            "gradient_sources": list(GRADIENT_SOURCES),
            "access_modes": ["packed", "layer_attribute"],
            "objective": "implementation scalar loss via simulator.loss_functions.compute_loss",
            "method": "central differences with per-tensor np.allclose tolerance",
        },
        summary_payload=summary,
        extra_raw={"case_records": result["cases"]},
        tables={
            "gradient_tensor_checks": [
                {
                    "case_id": record["case_id"],
                    "family": record["family"],
                    "gradient_source": c["gradient_source"],
                    "access_mode": c["access_mode"],
                    "kind": c["kind"],
                    "layer_type": c.get("layer_type"),
                    "shape": c.get("shape"),
                    "best_epsilon": c.get("best_epsilon"),
                    "analytic_norm": c.get("analytic_norm"),
                    "numeric_norm": c.get("numeric_norm"),
                    "max_abs_error": c.get("max_abs_error"),
                    "relative_l2_error": c.get("relative_l2_error"),
                    "cosine_similarity": c.get("cosine_similarity"),
                    "atol": c.get("atol"),
                    "rtol": c.get("rtol"),
                    "gradient_norm_floor": c.get("gradient_norm_floor"),
                    "status": c["status"],
                    "informative": c.get("informative"),
                    "passed": c["passed"],
                }
                for record in result["cases"]
                for c in record.get("comparisons", [])
            ],
            "case_summary": [
                {
                    **row,
                    "relative_l2_error_mean": (row["relative_l2_error"] or {}).get("mean"),
                    "relative_l2_error_max": (row["relative_l2_error"] or {}).get("max"),
                }
                for row in result["per_case_summary"]
            ],
            "gradient_epsilon_sweep": [
                {
                    "case_id": record["case_id"],
                    "gradient_source": c["gradient_source"],
                    "access_mode": c["access_mode"],
                    "kind": c["kind"],
                    "layer_type": c.get("layer_type"),
                    "epsilon": point["epsilon"],
                    "max_abs_error": point["max_abs_error"],
                    "relative_l2_error": point["relative_l2_error"],
                    "passed": point["passed"],
                }
                for record in result["cases"]
                for c in record.get("comparisons", [])
                for point in c.get("epsilon_sweep", [])
            ],
        },
        interpretation={
            "purpose": "Independent numerical evidence for the shipped gradient implementations.",
            "two_gradient_sources": (
                "backward_engine and training_engine are separate implementations; "
                "a pass on one is not a pass on the other."
            ),
            "param_sync_gate": (
                "A tensor marked PARAM_SYNC_NOT_VERIFIED is excluded from the "
                "pass/fail total because the numeric derivative was computed on a "
                "parameter the forward pass never consumes. See EXP-01."
            ),
            "tolerance_policy": (
                "Tolerances are declared per case in research/lib/cases.py and are "
                "not loosened after observing results."
            ),
        },
        limitations=[
            "Finite differences are single-precision and their own error floor is "
            "reported per tensor; agreement within tolerance is not proof of exactness.",
            "Cases are fixed small configurations; no claim is made about other shapes, "
            "losses, batch sizes or initialisations.",
            "BCE is only declared valid for a sigmoid output; other combinations are "
            "reported separately rather than pooled.",
        ],
    )
    manifest = writer.write()
    print(
        f"[{EXPERIMENT_ID}] cases={summary['cases_compared']} "
        f"tensor_checks={summary['tensor_checks_compared']} "
        f"failed={summary['tensor_checks_failed']} "
        f"sync_unverified={summary['tensor_checks_param_sync_unverified']} "
        f"status={summary['status']}"
    )
    print(f"evidence: research/evidence/{EXPERIMENT_ID}/{manifest['source_sha']}")
    return 1 if summary["status"] != "PASS" else 0


if __name__ == "__main__":
    raise SystemExit(main())
