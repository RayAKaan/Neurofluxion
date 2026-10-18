#!/usr/bin/env python3
"""EXP-03 Forward and trace fidelity audit against an independent reference.

Purpose
-------
Establish, with executed evidence, whether the values the simulator reports are
the values its computation actually produces, at two different granularities:

* **Forward fidelity** - the simulator's final output and every intermediate
  layer output are compared with an independent float64 reference
  (``research/lib/reference.py``) written from the standard mathematical
  definition of each operator.
* **Trace fidelity** - each record emitted by ``run_forward_full`` is
  reconstructed from the values and parameter snapshots it carries, and each
  property is classified as ``EXACTLY_VALIDATED``, ``PARTIALLY_VALIDATED``,
  ``OBSERVATIONALLY_VALIDATED`` or ``UNSUPPORTED``.

"Trace exists" is never treated as "trace is semantically correct".

Usage
-----
    PYTHONPATH=backend:research python research/experiments/run_trace_fidelity_audit.py --repo .

Exit codes: 0 = every declared property met its declared expectation,
1 = at least one finding, 2 = the experiment could not run.
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib import cases as case_registry  # noqa: E402
from lib import reference  # noqa: E402
from lib import stats as stat  # noqa: E402
from lib.evidence import EvidenceWriter  # noqa: E402
from lib.numerics import graph_forward, max_abs_error, passes_allclose  # noqa: E402
from lib.provenance import RunProvenance, utc_now_iso  # noqa: E402

EXPERIMENT_ID = "EXP-03-trace-fidelity-audit"
DEFAULT_SEED = 20261018

EXACT = "EXACTLY_VALIDATED"
PARTIAL = "PARTIALLY_VALIDATED"
OBSERVATIONAL = "OBSERVATIONALLY_VALIDATED"
UNSUPPORTED = "UNSUPPORTED"

ACTIVATION_REFERENCE = {
    "relu": lambda z: np.maximum(0.0, z),
    "sigmoid": reference.sigmoid,
    "tanh": np.tanh,
    "leaky_relu": lambda z: np.where(z > 0, z, 0.01 * z),
    "linear": lambda z: z,
}


# ---------------------------------------------------------------------------
def _reconstruct_dense_event(event: Dict[str, Any]) -> Dict[str, Any]:
    """Rebuild a dense-path trace event from only the values it carries.

    The returned ``expected`` array is what an independent reader of the record
    can compute; ``passed`` is then a plain tolerance comparison rather than a
    separate reconstruction of the same formula.
    """
    operation = event.get("operation")
    inputs = np.asarray(event.get("input_values", []), dtype=np.float64)
    if operation == "matmul":
        weights = np.asarray(event.get("weights_snapshot", []), dtype=np.float64)
        if weights.size == 0:
            return {"reconstructable": False, "reason": "no weight snapshot"}
        expected = weights @ inputs
    elif operation == "bias_add":
        bias = np.asarray(event.get("bias_snapshot", []), dtype=np.float64)
        if bias.size == 0:
            return {"reconstructable": False, "reason": "no bias snapshot"}
        expected = inputs + bias
    elif operation == "activation":
        name = str(event.get("activation_name", "")).lower()
        if name not in ACTIVATION_REFERENCE:
            return {
                "reconstructable": False,
                "reason": f"activation '{name}' has no independent reference",
            }
        expected = ACTIVATION_REFERENCE[name](inputs)
    else:
        return {"reconstructable": False, "reason": f"operation '{operation}' has no dense reference"}
    return {
        "reconstructable": True,
        "expected": expected,
        "reconstructed_norm": float(np.linalg.norm(expected)),
    }


def _audit_dense_trace(steps: List[Dict[str, Any]], atol: float, rtol: float) -> Dict[str, Any]:
    events: List[Dict[str, Any]] = []
    for event in steps:
        result = _reconstruct_dense_event(event)
        passed: Optional[bool] = None
        error: Optional[float] = None
        if result.get("reconstructable"):
            passed, error = passes_allclose(
                event.get("output_values", []), result["expected"], atol, rtol
            )
        events.append(
            {
                "step_index": event.get("step_index"),
                "layer_index": event.get("layer_index"),
                "operation": event.get("operation"),
                "reconstructable": result.get("reconstructable"),
                "reason": result.get("reason"),
                "max_abs_error": error,
                "reconstructed_norm": result.get("reconstructed_norm"),
                "passed": bool(passed) if passed is not None else None,
            }
        )
    checked = [e for e in events if e["passed"] is not None]
    return {
        "events": events,
        "events_total": len(events),
        "events_reconstructed": len(checked),
        "events_passed": sum(1 for e in checked if e["passed"]),
        "classification": EXACT if checked and all(e["passed"] for e in checked) else PARTIAL,
    }


def _audit_layer_trace(steps, reference_steps, atol, rtol) -> Dict[str, Any]:
    events: List[Dict[str, Any]] = []
    by_layer = {s.layer_index: s for s in reference_steps}
    for event in steps:
        layer_index = event.get("layer_index")
        ref = by_layer.get(int(layer_index) + 1) if layer_index is not None else None
        if ref is None:
            events.append(
                {
                    "step_index": event.get("step_index"),
                    "layer_index": layer_index,
                    "operation": event.get("operation"),
                    "comparable": False,
                    "reason": "no reference step for this layer",
                }
            )
            continue
        reported = np.asarray(event.get("output_values", []), dtype=np.float64).reshape(-1)
        comparison = reference.compare_forward(reported, ref.values.reshape(-1), atol, rtol)
        events.append(
            {
                "step_index": event.get("step_index"),
                "layer_index": layer_index,
                "operation": event.get("operation"),
                "reference_layer_type": ref.layer_type,
                "comparable": comparison.get("comparable"),
                "max_abs_error": comparison.get("max_abs_error"),
                "passed": comparison.get("passed"),
                "reason": comparison.get("reason"),
            }
        )
    comparable = [e for e in events if e.get("comparable")]
    return {
        "events": events,
        "events_total": len(events),
        "events_comparable": len(comparable),
        "events_passed": sum(1 for e in comparable if e["passed"]),
        "classification": EXACT if comparable and all(e["passed"] for e in comparable) else PARTIAL,
    }


# ---------------------------------------------------------------------------
def audit_case(case: case_registry.GraphCase, seed: int) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "case_id": case.case_id,
        "family": case.family,
        "expectation": case.expectation,
        "properties": [],
    }
    try:
        from simulator.forward_engine import run_forward_full, run_forward_step
        from simulator.graph_engine import build_graph

        np.random.seed(seed)
        graph = build_graph(list(case.layers))
        x = case.make_input()

        # --- forward fidelity -------------------------------------------------
        simulated = graph_forward(graph, x)
        ref_run = reference.reference_forward(graph, x, strict_padding=False)
        final = reference.compare_forward(
            simulated, ref_run.output.reshape(-1), case.forward_tolerance["atol"], case.forward_tolerance["rtol"]
        )
        per_layer = []
        for step in ref_run.steps:
            if step.layer_type == "input":
                continue
            graph.activations  # ensure populated state exists
            layer_index = step.layer_index
            actual = graph.activations[layer_index] if layer_index < len(graph.activations) else None
            if actual is None:
                continue
            per_layer.append(
                {
                    "layer_index": layer_index,
                    "layer_type": step.layer_type,
                    **reference.compare_forward(
                        np.asarray(actual, dtype=np.float64).reshape(-1),
                        step.values.reshape(-1),
                        case.forward_tolerance["atol"],
                        case.forward_tolerance["rtol"],
                    ),
                }
            )
        record["forward_final_output"] = final
        record["forward_per_layer"] = per_layer

        # --- strict-geometry probe for padding semantics ------------------------
        try:
            strict_run = reference.reference_forward(graph, x, strict_padding=True)
            strict = reference.compare_forward(
                simulated, strict_run.output.reshape(-1), case.forward_tolerance["atol"], case.forward_tolerance["rtol"]
            )
            record["forward_standard_padding_geometry"] = {
                **strict,
                "note": (
                    "Compared against the standard same/stride output geometry; a "
                    "shape mismatch means the simulator's padding rule differs from "
                    "the standard convention."
                ),
            }
        except Exception as exc:  # noqa: BLE001
            record["forward_standard_padding_geometry"] = {
                "comparable": False,
                "exception": f"{type(exc).__name__}: {exc}",
            }

        # --- trace fidelity ---------------------------------------------------
        steps, trace_output, layer_outputs = run_forward_full(graph, np.asarray(x).tolist())
        trace_dense_only = all(
            layer.layer_type in {"input", "dense", "output"} for layer in graph.layers
        )
        atol, rtol = case.forward_tolerance["atol"], case.forward_tolerance["rtol"]
        if trace_dense_only:
            trace_audit = _audit_dense_trace(steps, atol, rtol)
            trace_audit["granularity"] = "operation_level (matmul, bias_add, activation)"
        else:
            trace_audit = _audit_layer_trace(steps, ref_run.steps, atol, rtol)
            trace_audit["granularity"] = "layer_level (one forward record per layer)"
        record["trace_audit"] = trace_audit

        # Trace final output versus the independent reference.
        record["trace_final_output"] = reference.compare_forward(
            np.asarray(trace_output, dtype=np.float64).reshape(-1),
            ref_run.output.reshape(-1),
            atol,
            rtol,
        )

        # --- step selection consistency --------------------------------------
        if steps:
            indices = [0, len(steps) // 2, len(steps) - 1]
            step_records = []
            for index in sorted(set(indices)):
                selected = run_forward_step(graph, np.asarray(x).tolist(), index)
                matches_full = selected["step"].get("operation") == steps[index].get("operation") and selected[
                    "step"
                ].get("layer_index") == steps[index].get("layer_index")
                step_records.append(
                    {
                        "step_index": index,
                        "operation": selected["step"].get("operation"),
                        "matches_full_trace_event": bool(matches_full),
                    }
                )
            record["step_selection"] = {
                "events": step_records,
                "all_match": all(r["matches_full_trace_event"] for r in step_records),
                "classification": EXACT
                if all(r["matches_full_trace_event"] for r in step_records)
                else PARTIAL,
                "note": (
                    "run_forward_step regenerates the complete forward trace and then "
                    "selects one record; the selection is consistent but is not "
                    "incremental execution."
                ),
            }

        # --- ordering ---------------------------------------------------------
        indices = [int(s.get("step_index")) for s in steps]
        contiguous = indices == list(range(len(indices)))
        record["trace_ordering"] = {
            "step_indices": indices,
            "contiguous_from_zero": bool(contiguous),
            "classification": EXACT if contiguous else PARTIAL,
        }

        # --- trace value completeness ---------------------------------------
        record["trace_value_completeness"] = _trace_completeness(steps, graph)

        record["errors"] = []
        record["status"] = "AUDITED"
    except Exception as exc:  # noqa: BLE001
        record["exception"] = f"{type(exc).__name__}: {exc}"
        record["traceback"] = traceback.format_exc(limit=6)
        record["status"] = "DECLARED_UNSUPPORTED" if case.expectation == "unsupported" else "FAILED"
    return record


def _trace_completeness(steps, graph) -> Dict[str, Any]:
    """Do trace records carry every value the computation produced?"""
    truncated = []
    for event in steps:
        layer_index = event.get("layer_index")
        if layer_index is None or layer_index + 1 >= len(graph.layer_instances):
            continue
        layer = graph.layer_instances[layer_index + 1]
        cached = getattr(layer, "last_output", None)
        reported = np.asarray(event.get("output_values", []), dtype=np.float64).reshape(-1)
        produced = getattr(layer, "last_input", None)
        if produced is None:
            continue
        expected = int(np.size(produced))
        if expected and reported.size != expected:
            truncated.append(
                {
                    "step_index": event.get("step_index"),
                    "layer_index": layer_index,
                    "reported_values": int(reported.size),
                    "input_elements_available": expected,
                }
            )
    return {
        "records_with_shape_truncation": truncated,
        "classification": EXACT if not truncated else PARTIAL,
        "note": (
            "Layer-level trace records flatten their output. A record whose value "
            "count does not match the tensor it came from would be a truncated "
            "payload; no truncation flag is emitted by the engine."
        ),
    }


def run(repo_root: Path, selected, seed: int) -> Dict[str, Any]:
    records = [audit_case(case, seed) for case in selected]
    supported = [r for r in records if r.get("expectation") != "unsupported"]
    declared_unsupported = [r for r in records if r.get("expectation") == "unsupported"]
    unsupported_behaved_as_declared = [
        r
        for r in declared_unsupported
        if r.get("status") != "DECLARED_UNSUPPORTED"
    ]
    final_errors = [
        r["forward_final_output"]["max_abs_error"]
        for r in supported
        if r.get("forward_final_output", {}).get("max_abs_error") is not None
    ]
    layer_errors = [
        layer["max_abs_error"]
        for r in supported
        for layer in r.get("forward_per_layer", [])
        if layer.get("max_abs_error") is not None
    ]
    trace_errors = [
        r["trace_final_output"]["max_abs_error"]
        for r in supported
        if r.get("trace_final_output", {}).get("max_abs_error") is not None
    ]
    failures = [r for r in supported if r.get("status") != "AUDITED"]
    forward_failures = [
        r["case_id"] for r in supported if r.get("forward_final_output", {}).get("passed") is False
    ]
    trace_failures = [
        r["case_id"] for r in supported if r.get("trace_final_output", {}).get("passed") is False
    ]
    summary = {
        "cases_audited": len(records),
        "cases_supported": len(supported),
        "cases_declared_unsupported": len(declared_unsupported),
        "cases_declared_unsupported_that_behaved_as_declared": len(
            declared_unsupported
        )
        - len(unsupported_behaved_as_declared),
        "cases_failed_to_run": len(failures),
        "forward_final_output_failures": forward_failures,
        "trace_final_output_failures": trace_failures,
        "forward_final_max_abs_error_distribution": stat.summarise_matrix(np.array(final_errors))
        if final_errors
        else None,
        "forward_per_layer_max_abs_error_distribution": stat.summarise_matrix(np.array(layer_errors))
        if layer_errors
        else None,
        "trace_final_max_abs_error_distribution": stat.summarise_matrix(np.array(trace_errors))
        if trace_errors
        else None,
        "trace_property_classification": {
            r["case_id"]: {
                "granularity": r.get("trace_audit", {}).get("granularity"),
                "trace": r.get("trace_audit", {}).get("classification"),
                "step_selection": r.get("step_selection", {}).get("classification"),
                "ordering": r.get("trace_ordering", {}).get("classification"),
                "completeness": r.get("trace_value_completeness", {}).get("classification"),
            }
            for r in records
        },
        "status": "PASS"
        if not failures and not forward_failures and not trace_failures and not unsupported_behaved_as_declared
        else "FINDINGS_PRESENT",
    }
    return {"records": records, "summary": summary}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=".")
    parser.add_argument("--cases", nargs="*", default=None)
    parser.add_argument("--family", default=None)
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
        result = run(repo_root, selected, args.seed)
    except Exception as exc:  # noqa: BLE001
        print(f"trace fidelity audit failed: {exc}", file=sys.stderr)
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
            "reference": "research/lib/reference.py (independent float64 implementations)",
            "forward_tolerance": "per case in research/lib/cases.py",
            "classification_vocabulary": [EXACT, PARTIAL, OBSERVATIONAL, UNSUPPORTED],
        },
        summary_payload=result["summary"],
        extra_raw={"case_records": result["records"]},
        tables={
            "trace_property_classification": [
                {"case_id": case_id, **values}
                for case_id, values in result["summary"]["trace_property_classification"].items()
            ],
            "trace_events": [
                {
                    "case_id": r["case_id"],
                    "granularity": r.get("trace_audit", {}).get("granularity"),
                    "step_index": e.get("step_index"),
                    "layer_index": e.get("layer_index"),
                    "operation": e.get("operation"),
                    "reference_layer_type": e.get("reference_layer_type"),
                    "max_abs_error": e.get("max_abs_error"),
                    "passed": e.get("passed"),
                    "reason": e.get("reason"),
                }
                for r in result["records"]
                for e in r.get("trace_audit", {}).get("events", [])
            ],
            "forward_per_layer": [
                {"case_id": r["case_id"], **layer}
                for r in result["records"]
                for layer in r.get("forward_per_layer", [])
            ],
        },
        interpretation={
            "purpose": "Trace fidelity and forward-pass fidelity against an independent reference.",
            "classification": {
                EXACT: "every checked record reproduced from its own recorded values or the reference",
                PARTIAL: "some records are reconstructable; the remainder are layer-level summaries",
                OBSERVATIONAL: "the property was observed but could not be reconstructed independently",
                UNSUPPORTED: "the trace carries no evidence for this property",
            },
            "heterogeneity": (
                "Dense-only graphs emit matmul/bias_add/activation records; every other "
                "graph family emits one layer-level forward record per layer. Operation-level "
                "completeness is therefore a property of the architecture, not of the system."
            ),
        },
        limitations=[
            "The reference implements the standard operator definitions; where the simulator "
            "deviates (for example 'same' padding with stride greater than one) the difference is "
            "reported rather than reconciled.",
            "Layer-level trace records do not expose intra-layer arithmetic, so only the layer "
            "boundary is validated for non-dense graphs.",
            "No claim is made about WebSocket transport, UI rendering or asynchronous state.",
        ],
    )
    manifest = writer.write()
    summary = result["summary"]
    print(
        f"[{EXPERIMENT_ID}] cases={summary['cases_audited']} "
        f"forward_failures={len(summary['forward_final_output_failures'])} "
        f"trace_failures={len(summary['trace_final_output_failures'])} "
        f"status={summary['status']}"
    )
    print(f"evidence: research/evidence/{EXPERIMENT_ID}/{manifest['source_sha']}")
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())