#!/usr/bin/env python3
"""EXP-04 Training-update and optimizer-contract validation.

Purpose
-------
Demonstrate that a training step performs the documented parameter update, and
measure *where* the update actually lands. A finite loss and a changed
graph-level array are not sufficient evidence: the simulator keeps a packed
graph array and a per-layer copy, and an update can be applied to one without
reaching the computation.

For every case the experiment records

* the parameter value before the update,
* the gradient the optimizer consumed,
* the learning rate actually used,
* the update predicted by the optimizer's documented reference equation,
* the parameter value after the update,
* the achieved update magnitude and relative difference,
* whether the owning layer instance observed the update,
* the loss before and after, and whether it decreased.

A separate, explicitly labelled section runs multi-step optimisation so that
parameter-update correctness and convergence are never conflated.

Usage
-----
    PYTHONPATH=backend:research python research/experiments/run_training_update_validation.py --repo .

Exit codes: 0 = all declared update contracts met, 1 = findings, 2 = harness error.
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib import cases as case_registry  # noqa: E402
from lib import stats as stat  # noqa: E402
from lib.evidence import EvidenceWriter  # noqa: E402
from lib.numerics import graph_forward, max_abs_error, sync_params_to_layers  # noqa: E402
from lib.provenance import RunProvenance, utc_now_iso  # noqa: E402

EXPERIMENT_ID = "EXP-04-training-update-validation"
DEFAULT_SEED = 20261018

OPTIMIZERS = ("sgd", "sgd_momentum", "rmsprop", "adam")
UPDATE_ATOL = 1e-6
UPDATE_RTOL = 1e-4
# An expected update whose norm is at or below this floor is not measurable in a
# float32 simulation (roughly ten times the float32 machine epsilon), so its
# relative error is reported as undefined rather than as a large ratio.
UPDATE_NORM_FLOOR = 1e-6

MOMENTUM = 0.9
BETA1 = 0.9
BETA2 = 0.999
EPS = 1e-8


# ---------------------------------------------------------------------------
def reference_update(
    optimizer: str,
    w: np.ndarray,
    b: np.ndarray,
    gw: np.ndarray,
    gb: np.ndarray,
    lr: float,
    state: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Independent implementation of the documented optimizer equations.

    This mirrors ``research/VALIDATION_MATRIX.md``'s declared semantics: plain
    vanilla SGD, momentum with the ``m <- momentum*m + g`` accumulation, RMSProp
    using ``beta2`` as the decay constant, and bias-corrected Adam with the
    epsilon applied outside the square root.
    """
    state = dict(state or {})
    w = np.asarray(w, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    gw = np.asarray(gw, dtype=np.float64)
    gb = np.asarray(gb, dtype=np.float64)
    key = (optimizer or "sgd").lower()

    if key == "sgd":
        return w - lr * gw, b - lr * gb, state

    if key in {"sgd_momentum", "momentum"}:
        mw = MOMENTUM * np.asarray(state.get("mw", np.zeros_like(w))) + gw
        mb = MOMENTUM * np.asarray(state.get("mb", np.zeros_like(b))) + gb
        state.update({"mw": mw, "mb": mb})
        return w - lr * mw, b - lr * mb, state

    if key == "rmsprop":
        vw = BETA2 * np.asarray(state.get("vw", np.zeros_like(w))) + (1 - BETA2) * gw ** 2
        vb = BETA2 * np.asarray(state.get("vb", np.zeros_like(b))) + (1 - BETA2) * gb ** 2
        state.update({"vw": vw, "vb": vb})
        return w - lr * gw / (np.sqrt(vw) + EPS), b - lr * gb / (np.sqrt(vb) + EPS), state

    if key == "adam":
        t = int(state.get("t", 0)) + 1
        mw = BETA1 * np.asarray(state.get("mw", np.zeros_like(w))) + (1 - BETA1) * gw
        vw = BETA2 * np.asarray(state.get("vw", np.zeros_like(w))) + (1 - BETA2) * gw ** 2
        mb = BETA1 * np.asarray(state.get("mb", np.zeros_like(b))) + (1 - BETA1) * gb
        vb = BETA2 * np.asarray(state.get("vb", np.zeros_like(b))) + (1 - BETA2) * gb ** 2
        state.update({"t": t, "mw": mw, "vw": vw, "mb": mb, "vb": vb})
        mwh = mw / (1 - BETA1 ** t)
        vwh = vw / (1 - BETA2 ** t)
        mbh = mb / (1 - BETA1 ** t)
        vbh = vb / (1 - BETA2 ** t)
        return (
            w - lr * mwh / (np.sqrt(vwh) + EPS),
            b - lr * mbh / (np.sqrt(vbh) + EPS),
            state,
        )

    raise ValueError(f"unsupported optimizer '{optimizer}'")


# ---------------------------------------------------------------------------
def _layer_snapshot(graph) -> Dict[int, np.ndarray]:
    snapshot: Dict[int, np.ndarray] = {}
    for index, layer in enumerate(getattr(graph, "layer_instances", []) or []):
        if not getattr(layer, "has_params", False):
            continue
        try:
            packed_w, packed_b = layer.params()
            snapshot[index] = np.asarray(packed_w, dtype=np.float64).copy()
        except Exception:
            continue
    return snapshot


def _scalar_loss(graph, x: np.ndarray, target: np.ndarray, loss_fn: str) -> float:
    from simulator.loss_functions import compute_loss

    prediction = graph_forward(graph, x)
    return float(
        compute_loss(np.asarray(target, dtype=np.float32), prediction, loss_fn)[0]
    )


def run_update_case(case, optimizer: str, lr: float, seed: int) -> Dict[str, Any]:
    from simulator.graph_engine import build_graph
    from simulator.loss_functions import compute_loss
    from simulator.training_engine import TrainingConfig, TrainingEngine

    record: Dict[str, Any] = {
        "case_id": case.case_id,
        "family": case.family,
        "optimizer": optimizer,
        "learning_rate": lr,
        "loss_fn": case.loss_fn,
        "seed": seed,
    }
    np.random.seed(seed)
    graph = build_graph(list(case.layers))
    x = case.make_input()
    prediction = graph_forward(graph, x)
    target = case.make_target(int(prediction.size))

    engine = TrainingEngine(
        graph,
        TrainingConfig(
            epochs=1,
            batch_size=1,
            learning_rate=lr,
            optimizer=optimizer,
            loss_function=case.loss_fn,
            shuffle=False,
            dropout_rate=0.0,
            l2_lambda=0.0,
        ),
    )
    loss_before = _scalar_loss(graph, x, target, case.loss_fn)
    weights_before = [np.asarray(w, dtype=np.float64).copy() for w in graph.weights]
    biases_before = [np.asarray(b, dtype=np.float64).copy() for b in graph.biases]
    layer_before = _layer_snapshot(graph)

    batch_loss, accuracy, gradient_norm = engine.train_batch([(x.copy(), target.copy())])

    loss_after = _scalar_loss(graph, x, target, case.loss_fn)
    gradients_w = [np.asarray(g, dtype=np.float64) for g in engine.last_gradients["dW"]]
    gradients_b = [np.asarray(g, dtype=np.float64) for g in engine.last_gradients["db"]]

    tensors: List[Dict[str, Any]] = []
    state: Dict[str, Any] = {}
    mismatches: List[Dict[str, Any]] = []
    layer_propagation: List[Dict[str, Any]] = []

    for index in range(len(graph.weights)):
        expected_w, expected_b, state = reference_update(
            optimizer,
            weights_before[index],
            biases_before[index],
            gradients_w[index],
            gradients_b[index],
            lr,
            state,
        )
        achieved_w = np.asarray(graph.weights[index], dtype=np.float64)
        achieved_b = np.asarray(graph.biases[index], dtype=np.float64)
        for kind, expected, achieved, before in (
            ("weights", expected_w, achieved_w, weights_before[index]),
            ("biases", expected_b, achieved_b, biases_before[index]),
        ):
            delta = achieved - before
            expected_delta = expected - before
            ok, error = max_abs_error(achieved, expected) <= UPDATE_ATOL, max_abs_error(achieved, expected)
            entry = {
                "param_index": index,
                "kind": kind,
                "shape": list(np.shape(before)),
                "pre_update_norm": float(np.linalg.norm(before)),
                "gradient_norm": float(
                    np.linalg.norm(gradients_w[index] if kind == "weights" else gradients_b[index])
                ),
                "expected_update_norm": float(np.linalg.norm(expected_delta)),
                "actual_update_norm": float(np.linalg.norm(delta)),
                "max_abs_update_error": error,
                "changed": bool(np.linalg.norm(delta) > 0.0),
                "passed": bool(ok),
            }
            if before.size and float(np.linalg.norm(expected_delta)) > UPDATE_NORM_FLOOR:
                denom = float(np.linalg.norm(expected_delta))
                entry["update_relative_error"] = float(
                    np.linalg.norm(achieved - expected) / (denom + 1e-15)
                )
                entry["update_relative_error_defined"] = True
            else:
                # A tensor whose expected update is numerically zero has no
                # meaningful relative error; reporting one would manufacture an
                # arbitrarily large ratio.
                entry["update_relative_error"] = None
                entry["update_relative_error_defined"] = False
                entry["expected_update_below_floor"] = True
            tensors.append(entry)
            if not entry["passed"]:
                mismatches.append(entry)

    layer_after = _layer_snapshot(graph)
    for layer_index, before in layer_before.items():
        after = layer_after.get(layer_index)
        observed = after is not None and max_abs_error(after, before) > 0.0
        packed_index = None
        for candidate_index, candidate in dict(getattr(graph, "param_index_by_layer", {})).items():
            if candidate_index == layer_index:
                packed_index = candidate
        packed_changed = None
        if packed_index is not None and packed_index < len(graph.weights):
            packed_changed = bool(
                max_abs_error(graph.weights[packed_index], weights_before[packed_index]) > 0.0
            )
        layer_propagation.append(
            {
                "layer_instance_index": layer_index,
                "packed_param_index": packed_index,
                "layer_type": getattr(graph.layer_instances[layer_index], "layer_type", "unknown"),
                "layer_observed_update": bool(observed),
                "packed_array_changed": packed_changed,
                "consistent": None if packed_changed is None else bool(observed == packed_changed),
            }
        )

    record.update(
        {
            "batch_loss_reported": float(batch_loss),
            "batch_accuracy": float(accuracy),
            "gradient_norm_reported": float(gradient_norm),
            "loss_before_update": loss_before,
            "loss_after_update": loss_after,
            "loss_decreased": bool(loss_after < loss_before),
            "optimizer_state_initialised": engine.optimizer_state is not None,
            "learning_rate_used": float(
                engine.weight_history[-1].get("learning_rate", lr)
                if engine.weight_history
                else lr
            ),
            "parameter_tensors": tensors,
            "layer_propagation": layer_propagation,
            "update_contract_passed": not mismatches,
            "loss_reduction_passed": bool(loss_after < loss_before),
            "status": "MEASURED",
        }
    )
    record["status_detail"] = {
        "all_layers_observed_updates": all(
            bool(item["layer_observed_update"]) for item in layer_propagation
        ),
        "inconsistent_layers": [item for item in layer_propagation if item["consistent"] is False],
    }
    return record


# ---------------------------------------------------------------------------
def run_optimization_trajectory(case, optimizer: str, lr: float, steps: int, seed: int) -> Dict[str, Any]:
    """A short optimisation trajectory, reported separately from one-step correctness."""
    from simulator.graph_engine import build_graph
    from simulator.training_engine import TrainingConfig, TrainingEngine

    np.random.seed(seed)
    graph = build_graph(list(case.layers))
    x = case.make_input()
    prediction = graph_forward(graph, x)
    target = case.make_target(int(prediction.size))
    engine = TrainingEngine(
        graph,
        TrainingConfig(
            epochs=1,
            batch_size=1,
            learning_rate=lr,
            optimizer=optimizer,
            loss_function=case.loss_fn,
            shuffle=False,
            dropout_rate=0.0,
        ),
    )
    losses: List[float] = []
    for _ in range(steps):
        losses.append(_scalar_loss(graph, x, target, case.loss_fn))
        engine.train_batch([(x.copy(), target.copy())])
    losses.append(_scalar_loss(graph, x, target, case.loss_fn))
    return {
        "case_id": case.case_id,
        "optimizer": optimizer,
        "learning_rate": lr,
        "steps": steps,
        "loss_series": losses,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "loss_reduced": bool(losses[-1] < losses[0]),
        "monotone_non_increasing": bool(all(b <= a + 1e-12 for a, b in zip(losses, losses[1:]))),
        "relative_reduction": float((losses[0] - losses[-1]) / (abs(losses[0]) + 1e-12)),
    }


# ---------------------------------------------------------------------------
def run(repo_root: Path, selected, optimizers, lr: float, steps: int, seed: int) -> Dict[str, Any]:
    update_records: List[Dict[str, Any]] = []
    for case in selected:
        if not case.supports_training_update:
            continue
        for optimizer in optimizers:
            try:
                update_records.append(run_update_case(case, optimizer, lr, seed))
            except Exception as exc:  # noqa: BLE001
                update_records.append(
                    {
                        "case_id": case.case_id,
                        "optimizer": optimizer,
                        "status": "ERROR",
                        "exception": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(limit=6),
                    }
                )

    trajectories: List[Dict[str, Any]] = []
    for case in selected:
        if case.case_id not in {r["case_id"] for r in update_records}:
            continue
        for optimizer in optimizers:
            try:
                trajectories.append(run_optimization_trajectory(case, optimizer, lr, steps, seed))
            except Exception as exc:  # noqa: BLE001
                trajectories.append(
                    {
                        "case_id": case.case_id,
                        "optimizer": optimizer,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

    measured = [r for r in update_records if r.get("status") == "MEASURED"]
    failed = [r for r in measured if not r["update_contract_passed"]]
    inconsistent_layers = [
        item for r in measured for item in r.get("status_detail", {}).get("inconsistent_layers", [])
    ]
    relative_errors = [
        t["update_relative_error"]
        for r in measured
        for t in r.get("parameter_tensors", [])
        if t.get("update_relative_error") is not None
    ]
    summary = {
        "cases_with_training_support": len({r["case_id"] for r in update_records}),
        "optimizers": list(optimizers),
        "learning_rate": lr,
        "update_measurements": len(measured),
        "update_measurements_failed": len(failed),
        "parameter_tensor_checks": sum(len(r.get("parameter_tensors", [])) for r in measured),
        "parameter_tensor_checks_with_zero_expected_update": sum(
            1
            for r in measured
            for t in r.get("parameter_tensors", [])
            if t.get("expected_update_below_floor")
        ),
        "update_relative_error_distribution": stat.summarise_matrix(np.array(relative_errors))
        if relative_errors
        else None,
        "loss_reduction_measurements": sum(1 for r in measured if r.get("loss_reduction_passed")),
        "layer_propagation_inconsistencies": inconsistent_layers,
        "failed_measurements": [
            {
                "case_id": r["case_id"],
                "optimizer": r["optimizer"],
                "failing_tensors": [
                    {
                        "kind": t["kind"],
                        "param_index": t["param_index"],
                        "max_abs_update_error": t["max_abs_update_error"],
                        "update_relative_error": t["update_relative_error"],
                    }
                    for t in r.get("parameter_tensors", [])
                    if not t["passed"]
                ],
            }
            for r in failed
        ],
        "trajectories": {
            "count": len(trajectories),
            "steps_each": steps,
            "loss_reduced": sum(1 for t in trajectories if t.get("loss_reduced")),
            "monotone_non_increasing": sum(
                1 for t in trajectories if t.get("monotone_non_increasing")
            ),
            "relative_reduction": stat.describe(
                [t["relative_reduction"] for t in trajectories if "relative_reduction" in t]
            ),
        },
        "status": "PASS" if not failed else "FINDINGS_PRESENT",
    }
    return {"updates": update_records, "trajectories": trajectories, "summary": summary}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=".")
    parser.add_argument("--cases", nargs="*", default=None)
    parser.add_argument("--family", default=None)
    parser.add_argument("--optimizers", nargs="*", default=list(OPTIMIZERS))
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--allow-overwrite", action="store_true")
    args = parser.parse_args()

    repo_root = Path(args.repo).resolve()
    selected = case_registry.select_cases(args.cases, args.family)
    provenance = RunProvenance(
        experiment_id=EXPERIMENT_ID,
        script=Path(__file__).as_posix(),
        command=[sys.executable, *sys.argv],
        seed=args.seed,
        root=repo_root,
        notes={"runtime_source_modified": False},
    )
    try:
        result = run(repo_root, selected, args.optimizers, args.learning_rate, args.steps, args.seed)
    except Exception as exc:  # noqa: BLE001
        print(f"training-update validation failed: {exc}", file=sys.stderr)
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
            "cases": [c.case_id for c in selected],
            "optimizers": list(args.optimizers),
            "learning_rate": args.learning_rate,
            "trajectory_steps": args.steps,
            "reference_equations": {
                "sgd": "w <- w - lr * g",
                "sgd_momentum": "m <- momentum*m + g; w <- w - lr*m",
                "rmsprop": "v <- beta2*v + (1-beta2)*g^2; w <- w - lr*g/(sqrt(v)+eps)",
                "adam": "bias-corrected m, v; w <- w - lr*mhat/(sqrt(vhat)+eps)",
            },
            "hyperparameters": {"momentum": MOMENTUM, "beta1": BETA1, "beta2": BETA2, "eps": EPS},
        },
        summary_payload=result["summary"],
        extra_raw={
            "update_records": result["updates"],
            "trajectories": result["trajectories"],
        },
        tables={
            "update_tensor_checks": [
                {
                    "case_id": r["case_id"],
                    "family": r.get("family"),
                    "optimizer": r["optimizer"],
                    "param_index": t["param_index"],
                    "kind": t["kind"],
                    "shape": t["shape"],
                    "pre_update_norm": t["pre_update_norm"],
                    "gradient_norm": t["gradient_norm"],
                    "expected_update_norm": t["expected_update_norm"],
                    "actual_update_norm": t["actual_update_norm"],
                    "max_abs_update_error": t["max_abs_update_error"],
                    "update_relative_error": t["update_relative_error"],
                    "changed": t["changed"],
                    "passed": t["passed"],
                }
                for r in result["updates"]
                if r.get("status") == "MEASURED"
                for t in r.get("parameter_tensors", [])
            ],
            "layer_propagation": [
                {"case_id": r["case_id"], "optimizer": r["optimizer"], **item}
                for r in result["updates"]
                if r.get("status") == "MEASURED"
                for item in r.get("layer_propagation", [])
            ],
            "optimization_trajectories": [
                {
                    "case_id": t["case_id"],
                    "optimizer": t["optimizer"],
                    "steps": t.get("steps"),
                    "initial_loss": t.get("initial_loss"),
                    "final_loss": t.get("final_loss"),
                    "loss_reduced": t.get("loss_reduced"),
                    "monotone_non_increasing": t.get("monotone_non_increasing"),
                    "relative_reduction": t.get("relative_reduction"),
                }
                for t in result["trajectories"]
            ],
        },
        interpretation={
            "update_contract": (
                "A passing update contract shows the optimizer applied the declared "
                "equation to the supplied gradient. It does not show convergence."
            ),
            "convergence_scope": (
                "Trajectories are single-batch optimisations of a fixed deterministic "
                "task and are reported separately from one-step correctness. They "
                "establish loss reduction for the tested configuration only."
            ),
            "layer_propagation": (
                "layer_propagation records whether the owning layer instance observed "
                "the update. A packed array that changes while its layer does not is a "
                "silent state divergence, not a successful update."
            ),
        },
        limitations=[
            "Single-batch updates only; no minibatch accumulation, no shuffling, no dropout.",
            "Convergence is evaluated on fixed deterministic synthetic tasks, not on any dataset.",
            "L2 regularisation is disabled in this experiment.",
        ],
    )
    manifest = writer.write()
    summary = result["summary"]
    print(
        f"[{EXPERIMENT_ID}] measurements={summary['update_measurements']} "
        f"tensor_checks={summary['parameter_tensor_checks']} "
        f"failed={summary['update_measurements_failed']} "
        f"layer_inconsistencies={len(summary['layer_propagation_inconsistencies'])} "
        f"status={summary['status']}"
    )
    print(f"evidence: research/evidence/{EXPERIMENT_ID}/{manifest['source_sha']}")
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())