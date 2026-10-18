#!/usr/bin/env python3
"""Build every planned figure and table from recorded evidence.

Purpose
-------
Produce manuscript-ready figure source data and rendered figures *only* from the
committed evidence bundles. Nothing is drawn from memory or from a value typed
into a script: if the evidence directory for an experiment is absent, the figure
is skipped and reported as unavailable rather than approximated.

For every figure the script emits a figure manifest recording its id, purpose,
source experiment, source data file, units, axis definitions, statistical
treatment, caption notes and limitations, so the eventual manuscript can cite
exactly what each figure shows.

Usage
-----
    PYTHONPATH=backend:research python research/experiments/build_figures_and_tables.py --repo .
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib import stats as stat  # noqa: E402
from lib.evidence import write_csv, write_json  # noqa: E402
from lib.provenance import utc_now_iso  # noqa: E402

FIGURE_SPECS = (
    {
        "figure_id": "FIG-01-trace-overhead-decomposition",
        "purpose": "Attribute the cost of producing an execution trace to its components.",
        "experiment": "EXP-05-performance-benchmark",
        "table": "component_medians",
        "kind": "grouped_bar",
        "x": "workload",
        "y": "median milliseconds",
        "series": ["forward", "trace_total", "trace_json"],
        "units": "milliseconds",
        "statistical_treatment": "median of repeated samples with t-based 95% interval on the mean",
        "caption_notes": "The JSON serialisation component dominates for dense graphs; "
        "recurrent workloads emit one layer-level record per layer.",
        "limitations": "Single machine; tracemalloc peak memory is Python allocations only.",
    },
    {
        "figure_id": "FIG-02-trace-overhead-ratio",
        "purpose": "Show the paired trace/forward ratio with its confidence interval per workload.",
        "experiment": "EXP-05-performance-benchmark",
        "table": "trace_overhead",
        "kind": "errorbar",
        "x": "workload",
        "y": "paired trace/forward time ratio (dimensionless)",
        "series": ["paired_ratio_median"],
        "units": "ratio",
        "statistical_treatment": "median of per-trial paired ratios; 95% t interval on the mean ratio",
        "caption_notes": "A ratio of two independently summarised statistics is reported "
        "separately and is not used for the interval.",
        "limitations": "Paths perform different amounts of work by construction.",
    },
    {
        "figure_id": "FIG-03-scaling-depth-and-width",
        "purpose": "Show how forward and trace cost grow with depth and with hidden width.",
        "experiment": "EXP-06-scalability-study",
        "table": "scaling_points",
        "kind": "line",
        "x": "value",
        "y": "median milliseconds",
        "series": ["forward_median_ms", "trace_median_ms"],
        "units": "milliseconds",
        "statistical_treatment": "median of repeated samples; fitted exponents reported in a separate table",
        "caption_notes": "Two dense sweeps: hidden depth (1-10) and hidden width (8-256). "
        "Log-log fitted exponents are descriptive for the tested range only.",
        "limitations": "One factor varied at a time; interactions are not measured.",
    },
    {
        "figure_id": "FIG-04-scaling-exponents",
        "purpose": "Report the fitted scaling exponents with their goodness of fit and tested range.",
        "experiment": "EXP-06-scalability-study",
        "table": "scaling_exponents",
        "kind": "bar",
        "x": "sweep and measured quantity",
        "y": "log-log exponent",
        "series": ["exponent"],
        "units": "dimensionless",
        "statistical_treatment": "ordinary least squares in log-log space; R-squared reported",
        "caption_notes": "Quantities that do not vary with the sweep factor are reported as "
        "invariant and are not fitted.",
        "limitations": "Descriptive within the tested range; no extrapolation.",
    },
    {
        "figure_id": "FIG-05-gradient-error-distribution",
        "purpose": "Show the distribution of finite-difference gradient disagreement.",
        "experiment": "EXP-02-gradient-validation",
        "table": "gradient_tensor_checks",
        "kind": "box",
        "x": "gradient source",
        "y": "relative L2 error (dimensionless)",
        "series": ["relative_l2_error"],
        "units": "dimensionless",
        "statistical_treatment": "one value per parameter tensor; only informative comparisons included",
        "caption_notes": "Tensors whose packed parameters never reach their layer, and tensors "
        "whose gradient is numerically zero, are excluded and reported separately. "
        "Per-case values are in the source CSV.",
        "limitations": "ReLU configurations with pre-activations inside the kink band cannot "
        "be resolved by central differences.",
    },
    {
        "figure_id": "FIG-06-training-update-error",
        "purpose": "Show the distribution of optimizer update disagreement per family and optimizer.",
        "experiment": "EXP-04-training-update-validation",
        "table": "update_tensor_checks",
        "kind": "box",
        "x": "optimizer",
        "y": "relative update error (dimensionless)",
        "series": ["update_relative_error"],
        "units": "dimensionless",
        "statistical_treatment": "one value per parameter tensor; tensors below the float32 "
        "measurement floor are excluded from the relative statistic",
        "caption_notes": "Pass/fail uses the absolute update error against a declared tolerance.",
        "limitations": "Single-batch updates with dropout and shuffling disabled.",
    },
    {
        "figure_id": "FIG-07-trace-fidelity-error",
        "purpose": "Show forward and trace fidelity error against the independent reference.",
        "experiment": "EXP-03-trace-fidelity-audit",
        "table": "forward_per_layer",
        "kind": "bar",
        "x": "graph case",
        "y": "maximum absolute error against the float64 reference",
        "series": ["max_abs_error"],
        "units": "absolute error (dimensionless activations)",
        "statistical_treatment": "maximum over all validated layers of the case",
        "caption_notes": "Errors are at the level of float32 representation error.",
        "limitations": "The reference implements the standard operator definitions.",
    },
    {
        "figure_id": "FIG-08-literature-corpus",
        "purpose": "Show the composition of the literature discovery corpus by year and source.",
        "experiment": "EXP-10-literature-screening",
        "table": "screening_dataset",
        "kind": "bar",
        "x": "publication year",
        "y": "record count",
        "series": ["record_count"],
        "units": "records",
        "statistical_treatment": "counts from the deduplicated corpus; no inference",
        "caption_notes": "Discovery set only. It is not a screened or included study set.",
        "limitations": "Metadata completeness is partial; see the metadata audit.",
    },
)

TABLE_SPECS = (
    {
        "table_id": "TAB-01-operator-coverage",
        "purpose": "Per-family coverage of construction, forward, gradient, update, trace and UI evidence.",
        "experiment": "EXP-01-graph-capability-audit",
        "source": "cases",
    },
    {
        "table_id": "TAB-02-gradient-validation",
        "purpose": "Per-case finite-difference agreement counts and error statistics.",
        "experiment": "EXP-02-gradient-validation",
        "source": "case_summary",
    },
    {
        "table_id": "TAB-03-trace-property-classification",
        "purpose": "Per-case trace property classification with granularity.",
        "experiment": "EXP-03-trace-fidelity-audit",
        "source": "trace_property_classification",
    },
    {
        "table_id": "TAB-04-training-update",
        "purpose": "Per-case, per-optimizer update agreement and layer propagation.",
        "experiment": "EXP-04-training-update-validation",
        "source": "update_tensor_checks",
    },
    {
        "table_id": "TAB-05-performance-components",
        "purpose": "Per-workload component medians, dispersion and peak traced bytes.",
        "experiment": "EXP-05-performance-benchmark",
        "source": "component_medians",
    },
    {
        "table_id": "TAB-06-scaling-points",
        "purpose": "Per-sweep-point medians, payload sizes and graph measures.",
        "experiment": "EXP-06-scalability-study",
        "source": "scaling_points",
    },
    {
        "table_id": "TAB-07-determinism",
        "purpose": "Per-case in-process and cross-process determinism results.",
        "experiment": "EXP-07-determinism-study",
        "source": "determinism_summary",
    },
    {
        "table_id": "TAB-08-defect-register",
        "purpose": "Reproduced defects with severity, claim effect and remediation.",
        "experiment": "EXP-08-defect-register",
        "source": "defect_register",
    },
    {
        "table_id": "TAB-09-related-work-comparison",
        "purpose": "Source-linked capability matrix over the declared related-work set.",
        "experiment": "EXP-10-literature-screening",
        "source": "comparison_matrix",
    },
    {
        "table_id": "TAB-10-claim-evidence-ledger",
        "purpose": "Every research claim with its evidence, status, limitation and allowed wording.",
        "experiment": "EXP-11-claim-audit",
        "source": "claim_evidence_ledger",
    },
)


def find_revision(root: Path, experiment: str) -> Optional[Path]:
    base = Path(root) / "research" / "evidence" / experiment
    if not base.exists():
        return None
    revisions = sorted(p for p in base.iterdir() if p.is_dir())
    return revisions[-1] if revisions else None


def read_table(revision: Path, name: str) -> List[Dict[str, str]]:
    path = revision / "tables" / f"{name}.csv"
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_summary(revision: Path) -> Dict[str, Any]:
    path = revision / "summary" / "summary.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _num(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def build_figure(spec: Dict[str, Any], rows: List[Dict[str, str]], out_dir: Path) -> Dict[str, Any]:
    """Emit figure source data. Rendering is delegated to matplotlib when present."""
    series_values: Dict[str, List[float]] = {}
    labels: List[str] = []
    for row in rows:
        labels.append(str(row.get(spec["x"], "")))
        for series in spec["series"]:
            value = _num(row.get(series))
            if value is not None:
                series_values.setdefault(series, []).append(value)

    data = {
        "figure_id": spec["figure_id"],
        "generated_at_utc": utc_now_iso(),
        "labels": labels,
        "series": {name: values for name, values in series_values.items()},
        "specification": spec,
    }
    write_json(out_dir / "figures" / f"{spec['figure_id']}.data.json", data)
    write_csv(
        out_dir / "tables" / f"{spec['figure_id']}.csv",
        [{spec["x"]: row.get(spec["x"], ""), **{s: row.get(s, "") for s in spec["series"]}} for row in rows],
    )

    rendered = False
    render_note = "matplotlib is not available; figure source data was written without rendering"
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure, axis = plt.subplots(figsize=(7.2, 3.6), dpi=200)
        if spec["kind"] in {"grouped_bar", "bar"}:
            width = 0.8 / max(1, len(series_values))
            for index, (name, values) in enumerate(series_values.items()):
                positions = [i + index * width for i in range(len(values))]
                axis.bar(positions, values, width=width, label=name)
            axis.set_xticks([i + 0.4 - width / 2 for i in range(len(labels))])
            axis.set_xticklabels(labels, rotation=60, ha="right", fontsize=6)
            axis.legend(fontsize=6)
        elif spec["kind"] == "line":
            for name, values in series_values.items():
                xs = [_num(label) for label in labels]
                pairs = [(x, v) for x, v in zip(xs, values) if x is not None and v > 0]
                axis.plot([p[0] for p in pairs], [p[1] for p in pairs], marker="o", label=name)
            # Only use log axes when every plotted point is positive; a zero
            # measurement would otherwise silently drop a series.
            plotted = [v for values in series_values.values() for v in values]
            if plotted and min(plotted) > 0:
                axis.set_xscale("log", base=2)
                axis.set_yscale("log")
            axis.legend(fontsize=6)
        elif spec["kind"] == "errorbar":
            xs = list(range(len(labels)))
            for name, values in series_values.items():
                axis.plot(xs, values, marker="o", label=name)
            axis.set_xticks(xs)
            axis.set_xticklabels(labels, rotation=60, ha="right", fontsize=6)
            axis.legend(fontsize=6)
        elif spec["kind"] == "box":
            groups = list(series_values.values())
            axis.boxplot(groups, tick_labels=[f"n={len(g)}" for g in groups])
            axis.set_ylabel(spec["y"])
            axis.set_xlabel(f"{spec['x']} (all values pooled; see source CSV for per-case breakdown)")
            axis.set_yscale("log")
        axis.set_ylabel(spec["y"])
        axis.set_title(spec["figure_id"], fontsize=8)
        axis.grid(True, which="both", linewidth=0.3, alpha=0.4)
        figure.tight_layout()
        figure.savefig(out_dir / "figures" / f"{spec['figure_id']}.png")
        plt.close(figure)
        rendered = True
        render_note = "rendered with matplotlib"
    except Exception as exc:  # noqa: BLE001
        render_note = f"rendering unavailable ({type(exc).__name__}: {exc}); source data written"
    return {
        "figure_id": spec["figure_id"],
        "status": "GENERATED",
        "rendered": rendered,
        "render_note": render_note,
        "rows": len(rows),
        "series_points": {name: len(values) for name, values in series_values.items()},
    }


def build_year_histogram(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    counts: Dict[str, int] = {}
    for row in rows:
        year = str(row.get("year", "")).strip()
        if year.isdigit():
            counts[year] = counts.get(year, 0) + 1
    return [{"year": year, "record_count": count} for year, count in sorted(counts.items())]


def pivot_components(rows: List[Dict[str, str]], components: List[str]) -> List[Dict[str, str]]:
    """Turn per-(workload, component) rows into one row per workload."""
    labels: List[str] = []
    values: Dict[str, Dict[str, str]] = {name: {} for name in components}
    for row in rows:
        label = str(row.get("workload_id", ""))
        component = str(row.get("component", ""))
        if component not in values:
            continue
        if label not in labels:
            labels.append(label)
        values[component][label] = row.get("median_ms", "")
    return [
        {"workload": label, **{name: values[name].get(label, "") for name in components}}
        for label in labels
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=".")
    args = parser.parse_args()

    root = Path(args.repo).resolve()
    out_dir = root / "research" / "artifacts" / "figures"
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)

    figure_results: List[Dict[str, Any]] = []
    for spec in FIGURE_SPECS:
        revision = find_revision(root, spec["experiment"])
        if revision is None:
            figure_results.append(
                {
                    "figure_id": spec["figure_id"],
                    "status": "UNAVAILABLE",
                    "reason": f"no evidence revision for {spec['experiment']}",
                }
            )
            continue
        rows = read_table(revision, spec["table"])
        if spec["figure_id"] == "FIG-08-literature-corpus":
            rows = build_year_histogram(rows)
            spec = dict(spec, x="year", series=["record_count"])
        if spec["figure_id"] == "FIG-01-trace-overhead-decomposition":
            rows = pivot_components(rows, spec["series"])
        if spec["figure_id"] == "FIG-03-scaling-depth-and-width":
            # Only the two dense sweeps carry both a numeric sweep value and a
            # measured median; boundary probes have no measurement to plot.
            rows = [
                row
                for row in rows
                if row.get("sweep") in {"depth", "width"}
                and str(row.get(spec["x"], "")).strip() not in {"", "None"}
            ]
        if not rows:
            figure_results.append(
                {
                    "figure_id": spec["figure_id"],
                    "status": "UNAVAILABLE",
                    "reason": f"table '{spec['table']}' has no rows in {revision.name}",
                }
            )
            continue
        result = build_figure(spec, rows, out_dir)
        result["source_experiment"] = spec["experiment"]
        result["source_revision"] = revision.name
        result["source_table"] = spec["table"]
        figure_results.append(result)

    table_results: List[Dict[str, Any]] = []
    for spec in TABLE_SPECS:
        revision = find_revision(root, spec["experiment"])
        if revision is None:
            table_results.append({**spec, "status": "UNAVAILABLE", "reason": "no evidence revision"})
            continue
        rows = read_table(revision, spec["source"])
        if not rows:
            table_results.append({**spec, "status": "UNAVAILABLE", "reason": "empty table"})
            continue
        target = out_dir / "tables" / f"{spec['table_id']}.csv"
        write_csv(target, rows)
        write_csv(
            out_dir / "tables" / f"{spec['table_id']}.markdown.md",
            rows,
        )
        table_results.append(
            {
                **spec,
                "status": "GENERATED",
                "rows": len(rows),
                "source_revision": revision.name,
                "path": str(target.relative_to(root)).replace("\\", "/"),
            }
        )

    manifest = {
        "schema": "neurofluxion.research.figure_manifest",
        "schema_version": 1,
        "generated_at_utc": utc_now_iso(),
        "figures": figure_results,
        "tables": table_results,
        "policy": (
            "Every figure and table is generated from a committed evidence bundle. "
            "A missing bundle produces an UNAVAILABLE entry, never an approximation."
        ),
        "limitations": [
            "Figures are evidence visualisations, not manuscript artwork; layout, caption "
            "and numbering are the responsibility of the manuscript.",
            "Rendered output requires matplotlib; when it is absent the machine-readable "
            "source data is still produced.",
        ],
    }
    write_json(out_dir / "figure_manifest.json", manifest)

    unavailable = [f for f in figure_results if f["status"] != "GENERATED"]
    print(
        f"[build_figures_and_tables] figures={len(figure_results)} "
        f"generated={len(figure_results) - len(unavailable)} "
        f"tables={len(table_results)} "
        f"unavailable={len(unavailable)}"
    )
    for entry in unavailable:
        print(f"  unavailable: {entry['figure_id']} ({entry.get('reason')})")
    print(f"output: {out_dir}")
    return 0 if not unavailable else 1


if __name__ == "__main__":
    raise SystemExit(main())