#!/usr/bin/env python3
"""EXP-10 Literature screening, deduplication audit and comparison matrix.

Purpose
-------
Turn a raw two-source metadata export into an auditable screening dataset and a
source-linked comparison matrix, while keeping automated triage clearly separate
from human screening.

Stages
------
1. ``metadata_audit``      completeness of titles, abstracts, years, venues, DOIs
2. ``dedup_audit``         verification of the deduplication rule: false merges
                           (one DOI mapped to different titles) and residual
                           duplicates that the rule failed to merge
3. ``automated_triage``    a versioned, transparent keyword rule that marks
                           records as priority for human review. This is *not* a
                           screening decision and never writes one.
4. ``comparison_matrix``   a source-linked capability matrix for a declared
                           included set. Capabilities are recorded only where a
                           cited source supports them.

Outputs are written to an evidence bundle so the counts are reproducible and
traceable to the export they were produced from.

Usage
-----
    python research/experiments/run_literature_screening.py \
        --repo . --export H:/temp-lit/export3
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import traceback
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib import stats as stat  # noqa: E402
from lib.evidence import EvidenceWriter, write_csv, write_json  # noqa: E402
from lib.provenance import RunProvenance, sha256_file, utc_now_iso  # noqa: E402

EXPERIMENT_ID = "EXP-10-literature-screening"
TRIAGE_VERSION = "triage-1.0.0"

# Terms that place a record in the neighbourhood of interactive neural-network
# inspection. Each term is recorded so the rule is reproducible and auditable.
PRIMARY_TERMS = (
    "neural network",
    "deep learning",
    "deep neural",
    "convolutional neural",
    "recurrent neural",
    "transformer",
)
SECONDARY_TERMS = (
    "visual analytics",
    "visualization",
    "visualisation",
    "interpretability",
    "explainab",
    "inspect",
    "debug",
    "trace",
    "provenance",
    "graph",
    "activation",
    "simulation",
    "simulator",
    "teaching",
    "education",
    "interactive",
    "interpretable",
)
EXCLUSION_TERMS = (
    "molecular",
    "protein",
    "medical imaging",
    "wireless sensor",
    "power system",
    "traffic flow",
    "recommendation",
    "large language model",
)


def normalize_title(value: str) -> str:
    text = unicodedata.normalize("NFKD", value or "").lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def triage_score(title: str, abstract: str, venue: str) -> Dict[str, Any]:
    haystack = " ".join([title or "", abstract or "", venue or ""]).lower()
    primary = [term for term in PRIMARY_TERMS if term in haystack]
    secondary = [term for term in SECONDARY_TERMS if term in haystack]
    exclusions = [term for term in EXCLUSION_TERMS if term in haystack]
    score = 2 * len(primary) + len(secondary)
    return {
        "triage_version": TRIAGE_VERSION,
        "primary_terms": primary,
        "secondary_terms": secondary,
        "exclusion_terms": exclusions,
        "triage_score": score,
        "triage_priority": (
            "HIGH" if score >= 5 else "MEDIUM" if score >= 3 else "LOW"
        ),
        "triage_is_screening_decision": False,
    }


# ---------------------------------------------------------------------------
def audit_metadata(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    titles_blank = sum(1 for r in records if not (r.get("title") or "").strip())
    abstracts_blank = sum(1 for r in records if not (r.get("abstract") or "").strip())
    dois_blank = sum(1 for r in records if not (r.get("doi") or "").strip())
    years = [r.get("year") for r in records if str(r.get("year") or "").strip()]
    numeric_years = sorted(int(y) for y in years if str(y).isdigit())
    venues = [(r.get("venue") or "").strip() for r in records]
    return {
        "records": len(records),
        "blank_title": titles_blank,
        "blank_abstract": abstracts_blank,
        "blank_doi": dois_blank,
        "abstract_completeness": 1.0 - (abstracts_blank / max(1, len(records))),
        "doi_completeness": 1.0 - (dois_blank / max(1, len(records))),
        "year_range": [numeric_years[0], numeric_years[-1]] if numeric_years else None,
        "year_histogram": stat.describe([float(y) for y in numeric_years]) if numeric_years else None,
        "distinct_venues": len({v for v in venues if v}),
        "blank_venue": sum(1 for v in venues if not v),
        "records_by_source_database": _count_by(
            records, lambda r: "; ".join(r.get("source_databases", []) or []) or "unknown"
        ),
        "multi_source_records": sum(
            1 for r in records if len(r.get("source_databases", []) or []) > 1
        ),
    }


def _count_by(records: Sequence[Dict[str, Any]], key) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for record in records:
        value = key(record)
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def audit_deduplication(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Check the deduplication result for false merges and residual duplicates."""
    by_doi: Dict[str, List[str]] = {}
    by_title_year: Dict[Tuple[str, str], List[str]] = {}
    for record in records:
        doi = (record.get("doi") or "").strip().lower()
        if doi:
            by_doi.setdefault(doi, []).append(normalize_title(record.get("title", "")))
        key = (normalize_title(record.get("title", "")), str(record.get("year") or ""))
        by_title_year.setdefault(key, []).append(doi or record.get("record_key", ""))

    false_merges = [
        {"doi": doi, "distinct_normalised_titles": sorted(set(titles)), "count": len(titles)}
        for doi, titles in by_doi.items()
        if len(set(titles)) > 1
    ]
    residual_duplicates = [
        {
            "normalised_title": key[0],
            "year": key[1],
            "keys": sorted(set(values)),
        }
        for key, values in by_title_year.items()
        if key[0] and len(set(values)) > 1
    ]
    return {
        "records": len(records),
        "records_with_doi": len(by_doi),
        "distinct_dois": len(by_doi),
        "duplicate_doi_groups": sum(1 for titles in by_doi.values() if len(set(titles)) > 1),
        "false_merge_candidates": len(false_merges),
        "false_merge_examples": false_merges[:20],
        "residual_duplicate_title_year_groups": len(residual_duplicates),
        "residual_duplicate_examples": residual_duplicates[:20],
        "note": (
            "A false-merge candidate is one DOI that maps to more than one "
            "normalised title. It is a review flag, not a confirmed error: "
            "pre-prints, conference and journal versions of the same work share a DOI prefix."
        ),
    }


# ---------------------------------------------------------------------------
COMPARISON_DIMENSIONS = (
    ("execution_tracing", "records intermediate operations of a model execution"),
    ("tensor_inspection", "exposes individual tensor/activation values"),
    ("graph_visualization", "renders the model or computation graph"),
    ("operation_level_inspection", "explains or exposes individual operator semantics"),
    ("recurrent_inspection", "exposes hidden or gate state over time"),
    ("interactive_navigation", "supports user-driven navigation of a computation"),
    ("numerical_transparency", "shows the arithmetic/equation behind a value"),
    ("training_visibility", "shows behaviour during training"),
    ("provenance", "ties a displayed value to an execution or parameter revision"),
    ("independent_numerical_oracle", "validates displayed values against an independent reference"),
    ("browser_accessible", "runs in a browser without installation"),
    ("local_execution", "executes locally without a hosted service"),
)

# Declared included set. Every capability value is tied to a cited primary source
# that was read for this review; nothing here is inferred from a title alone.
INCLUDED_SET: List[Dict[str, Any]] = [
    {
        "key": "activis",
        "title": "ActiVis: Visual Exploration of Industry-Scale Deep Neural Network Models",
        "venue": "IEEE TVCG",
        "year": 2018,
        "doi": "10.1109/TVCG.2017.2744718",
        "category": "direct_competitor",
        "capabilities": {
            "execution_tracing": "no",
            "tensor_inspection": "yes",
            "graph_visualization": "yes",
            "operation_level_inspection": "partial",
            "recurrent_inspection": "no",
            "interactive_navigation": "yes",
            "numerical_transparency": "partial",
            "training_visibility": "partial",
            "provenance": "not_stated",
            "independent_numerical_oracle": "no",
            "browser_accessible": "no",
            "local_execution": "unknown",
        },
        "evidence_note": "Coordinated architecture, activation and instance views with deployment case studies.",
    },
    {
        "key": "cnn_explainer",
        "title": "CNN Explainer: Learning to Explain a Single CNN",
        "venue": "IEEE TVCG",
        "year": 2021,
        "doi": "10.1109/TVCG.2020.3030418",
        "category": "direct_competitor",
        "capabilities": {
            "execution_tracing": "partial",
            "tensor_inspection": "yes",
            "graph_visualization": "yes",
            "operation_level_inspection": "yes",
            "recurrent_inspection": "no",
            "interactive_navigation": "yes",
            "numerical_transparency": "yes",
            "training_visibility": "no",
            "provenance": "not_stated",
            "independent_numerical_oracle": "no",
            "browser_accessible": "yes",
            "local_execution": "yes",
        },
        "evidence_note": "Multi-level interactive explanation linking low-level convolution operations to architecture.",
    },
    {
        "key": "lstmvis",
        "title": "LSTMVis: A Systematic Tool for Visual Analysis of Hidden State Dynamics in Recurrent Networks",
        "venue": "IEEE TVCG",
        "year": 2019,
        "doi": "10.1109/TVCG.2017.2744158",
        "category": "direct_competitor",
        "capabilities": {
            "execution_tracing": "no",
            "tensor_inspection": "yes",
            "graph_visualization": "no",
            "operation_level_inspection": "partial",
            "recurrent_inspection": "yes",
            "interactive_navigation": "yes",
            "numerical_transparency": "partial",
            "training_visibility": "no",
            "provenance": "not_stated",
            "independent_numerical_oracle": "no",
            "browser_accessible": "no",
            "local_execution": "unknown",
        },
        "evidence_note": "Hidden-state pattern search and matching for recurrent networks.",
    },
    {
        "key": "tf_graph_visualizer",
        "title": "Visualizing Dataflow in TensorFlow Graphs",
        "venue": "IEEE TVCG",
        "year": 2018,
        "doi": "10.1109/TVCG.2017.2744878",
        "category": "adjacent",
        "capabilities": {
            "execution_tracing": "no",
            "tensor_inspection": "no",
            "graph_visualization": "yes",
            "operation_level_inspection": "no",
            "recurrent_inspection": "no",
            "interactive_navigation": "yes",
            "numerical_transparency": "no",
            "training_visibility": "no",
            "provenance": "not_stated",
            "independent_numerical_oracle": "no",
            "browser_accessible": "yes",
            "local_execution": "yes",
        },
        "evidence_note": "Hierarchical exploration of TensorFlow dataflow graphs; visualisation, not numerical validation.",
    },
    {
        "key": "deepeyes",
        "title": "DeepEyes: Progressive Visual Analytics for Designing, Training, and Diagnosing Deep Neural Networks",
        "venue": "IEEE TVCG",
        "year": 2018,
        "doi": "10.1109/TVCG.2017.2744358",
        "category": "direct_competitor",
        "capabilities": {
            "execution_tracing": "no",
            "tensor_inspection": "partial",
            "graph_visualization": "yes",
            "operation_level_inspection": "partial",
            "recurrent_inspection": "no",
            "interactive_navigation": "yes",
            "numerical_transparency": "partial",
            "training_visibility": "yes",
            "provenance": "not_stated",
            "independent_numerical_oracle": "no",
            "browser_accessible": "no",
            "local_execution": "unknown",
        },
        "evidence_note": "Progressive model design and diagnosis during training.",
    },
    {
        "key": "nnvisbuilder",
        "title": "NNVisBuilder: A Toolkit for Building Customizable Neural Network Visual Analytics Systems",
        "venue": "IEEE TVCG",
        "year": 2024,
        "doi": "10.1109/TVCG.2023.3326575",
        "category": "adjacent",
        "capabilities": {
            "execution_tracing": "partial",
            "tensor_inspection": "yes",
            "graph_visualization": "yes",
            "operation_level_inspection": "no",
            "recurrent_inspection": "partial",
            "interactive_navigation": "yes",
            "numerical_transparency": "partial",
            "training_visibility": "yes",
            "provenance": "not_stated",
            "independent_numerical_oracle": "no",
            "browser_accessible": "no",
            "local_execution": "yes",
        },
        "evidence_note": "Toolkit for prototyping visualisation systems with PyTorch integration.",
    },
    {
        "key": "in_situ_va",
        "title": "An In-Situ Visual Analytics Framework for Deep Neural Networks",
        "venue": "IEEE TVCG",
        "year": 2024,
        "doi": "10.1109/TVCG.2023.3339585",
        "category": "direct_competitor",
        "capabilities": {
            "execution_tracing": "partial",
            "tensor_inspection": "yes",
            "graph_visualization": "partial",
            "operation_level_inspection": "no",
            "recurrent_inspection": "no",
            "interactive_navigation": "yes",
            "numerical_transparency": "partial",
            "training_visibility": "yes",
            "provenance": "not_stated",
            "independent_numerical_oracle": "no",
            "browser_accessible": "no",
            "local_execution": "unknown",
        },
        "evidence_note": "Visual analytics integrated with DNN training.",
    },
    {
        "key": "channel_explorer",
        "title": "ChannelExplorer: Exploring Class Separability Through Activation Channel Visualization",
        "venue": "IEEE TVCG",
        "year": 2026,
        "doi": "10.1109/TVCG.2026.3669148",
        "category": "direct_competitor",
        "capabilities": {
            "execution_tracing": "no",
            "tensor_inspection": "yes",
            "graph_visualization": "partial",
            "operation_level_inspection": "no",
            "recurrent_inspection": "no",
            "interactive_navigation": "yes",
            "numerical_transparency": "no",
            "training_visibility": "no",
            "provenance": "not_stated",
            "independent_numerical_oracle": "no",
            "browser_accessible": "no",
            "local_execution": "unknown",
        },
        "evidence_note": "Dataset-to-example drilldown with activation-channel, similarity and heatmap views.",
    },
    {
        "key": "graph_of_trace",
        "title": "Graph of Trace: Visualizing Execution Traces of Scientific Agents",
        "venue": "ACL System Demonstrations",
        "year": 2026,
        "doi": "10.18653/v1/2026.acl-demo.29",
        "category": "adjacent",
        "capabilities": {
            "execution_tracing": "yes",
            "tensor_inspection": "no",
            "graph_visualization": "yes",
            "operation_level_inspection": "no",
            "recurrent_inspection": "no",
            "interactive_navigation": "yes",
            "numerical_transparency": "no",
            "training_visibility": "no",
            "provenance": "partial",
            "independent_numerical_oracle": "no",
            "browser_accessible": "unknown",
            "local_execution": "unknown",
        },
        "evidence_note": "Fine-grained agent workflow event traces rendered as an evolving directed graph.",
    },
    {
        "key": "dlprov",
        "title": "DLProv: a suite of provenance services for deep learning workflow analyses",
        "venue": "Software Impacts",
        "year": 2025,
        "doi": "10.1016/j.simpa.2024.100730",
        "category": "adjacent",
        "capabilities": {
            "execution_tracing": "no",
            "tensor_inspection": "no",
            "graph_visualization": "partial",
            "operation_level_inspection": "no",
            "recurrent_inspection": "no",
            "interactive_navigation": "no",
            "numerical_transparency": "no",
            "training_visibility": "no",
            "provenance": "yes",
            "independent_numerical_oracle": "no",
            "browser_accessible": "no",
            "local_execution": "yes",
        },
        "evidence_note": "Workflow-level lineage and W3C PROV-compatible provenance graphs.",
    },
    {
        "key": "yprov4ml",
        "title": "yProv4ML: Effortless provenance tracking for machine learning systems",
        "venue": "SoftwareX",
        "year": 2025,
        "doi": "10.1016/j.softx.2025.102298",
        "category": "adjacent",
        "capabilities": {
            "execution_tracing": "partial",
            "tensor_inspection": "no",
            "graph_visualization": "no",
            "operation_level_inspection": "no",
            "recurrent_inspection": "no",
            "interactive_navigation": "no",
            "numerical_transparency": "no",
            "training_visibility": "no",
            "provenance": "yes",
            "independent_numerical_oracle": "no",
            "browser_accessible": "no",
            "local_execution": "yes",
        },
        "evidence_note": "PROV-JSON capture of ML execution provenance with minimal code change.",
    },
    {
        "key": "tf_playground",
        "title": "TensorFlow Playground",
        "venue": "open-source software",
        "year": 2016,
        "doi": "",
        "category": "educational_software",
        "capabilities": {
            "execution_tracing": "partial",
            "tensor_inspection": "partial",
            "graph_visualization": "yes",
            "operation_level_inspection": "partial",
            "recurrent_inspection": "no",
            "interactive_navigation": "yes",
            "numerical_transparency": "partial",
            "training_visibility": "yes",
            "provenance": "no",
            "independent_numerical_oracle": "no",
            "browser_accessible": "yes",
            "local_execution": "yes",
        },
        "evidence_note": "Browser small-network construction, training and decision-boundary exploration; not peer reviewed.",
    },
    {
        "key": "neurofluxion",
        "title": "Neurofluxion (this work)",
        "venue": "implementation study",
        "year": 2026,
        "doi": "",
        "category": "this_work",
        "capabilities": {
            "execution_tracing": "yes",
            "tensor_inspection": "yes",
            "graph_visualization": "yes",
            "operation_level_inspection": "partial",
            "recurrent_inspection": "partial",
            "interactive_navigation": "yes",
            "numerical_transparency": "yes",
            "training_visibility": "yes",
            "provenance": "partial",
            "independent_numerical_oracle": "yes",
            "browser_accessible": "yes",
            "local_execution": "yes",
        },
        "evidence_note": (
            "Values in this row are the claims this repository's own experiments "
            "support; they are not asserted for any other system."
        ),
    },
]


def build_comparison_matrix() -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for entry in INCLUDED_SET:
        row = {
            "key": entry["key"],
            "title": entry["title"],
            "venue": entry["venue"],
            "year": entry["year"],
            "doi": entry["doi"],
            "category": entry["category"],
            "evidence_note": entry["evidence_note"],
        }
        for dimension, _ in COMPARISON_DIMENSIONS:
            row[dimension] = entry["capabilities"].get(dimension, "not_stated")
        rows.append(row)
    dimension_summary = {
        dimension: {
            value: sum(1 for row in rows if row[dimension] == value)
            for value in sorted({row[dimension] for row in rows})
        }
        for dimension, _ in COMPARISON_DIMENSIONS
    }
    return {
        "dimensions": [
            {"key": key, "definition": description} for key, description in COMPARISON_DIMENSIONS
        ],
        "value_vocabulary": {
            "yes": "the cited source states or demonstrates the capability",
            "partial": "the cited source demonstrates a narrower form of the capability",
            "no": "the cited source does not present the capability",
            "not_stated": "the cited source does not address the capability",
            "unknown": "the cited source does not establish deployment detail",
        },
        "rows": rows,
        "dimension_summary": dimension_summary,
        "interpretation_limits": [
            "Values are recorded from the cited primary source at the granularity that "
            "source presents; absence of a stated capability is not proof of absence.",
            "Only the neurofluxion row is supported by this repository's own executed "
            "experiments; the other rows come from reading the cited papers, not from "
            "reimplementing them.",
            "No system in the set is re-implemented and benchmarked here, so no performance "
            "or superiority comparison is implied.",
        ],
    }


# ---------------------------------------------------------------------------
def run(repo_root: Path, export_dir: Path) -> Dict[str, Any]:
    export_dir = Path(export_dir)
    raw = json.loads((export_dir / "raw_records.json").read_text(encoding="utf-8"))
    dedup = json.loads((export_dir / "deduplicated_records.json").read_text(encoding="utf-8"))
    manifest = json.loads((export_dir / "search_manifest.json").read_text(encoding="utf-8"))

    triage_rows: List[Dict[str, Any]] = []
    for record in dedup:
        triage = triage_score(record.get("title", ""), record.get("abstract", ""), record.get("venue", ""))
        triage_rows.append({**record, **triage})

    summary = {
        "export_script_version": manifest.get("script_version"),
        "export_generated_at_utc": manifest.get("generated_at_utc"),
        "export_queries": manifest.get("queries"),
        "export_databases": manifest.get("databases"),
        "export_pagination_mode": manifest.get("pagination_mode"),
        "export_raw_records": manifest.get("raw_record_count"),
        "export_deduplicated_records": manifest.get("deduplicated_record_count"),
        "export_failures": len(manifest.get("failures", [])),
        "export_complete_two_source": (
            len(manifest.get("failures", [])) == 0
            and set(manifest.get("databases", [])) == {"OpenAlex", "Crossref"}
        ),
        "metadata_audit": audit_metadata(dedup),
        "dedup_audit": audit_deduplication(dedup),
        "triage": {
            "triage_version": TRIAGE_VERSION,
            "records_triaged": len(triage_rows),
            **_count_by(triage_rows, lambda r: r["triage_priority"]),
            "records_with_exclusion_term": sum(1 for r in triage_rows if r["exclusion_terms"]),
            "screening_decisions_made": 0,
            "screening_decisions_note": (
                "The triage rule is an automated prioritisation aid. It produces no "
                "include/exclude decision and must not be reported as screening."
            ),
        },
        "comparison_matrix": {
            "systems": len(INCLUDED_SET),
            "dimensions": len(COMPARISON_DIMENSIONS),
            "direct_competitors": sum(1 for e in INCLUDED_SET if e["category"] == "direct_competitor"),
        },
        "status": "COMPLETE" if not manifest.get("failures") else "PARTIAL_EXPORT",
    }
    return {
        "triage_rows": triage_rows,
        "raw_record_count": len(raw),
        "dedup_record_count": len(dedup),
        "export_manifest": manifest,
        "comparison_matrix": build_comparison_matrix(),
        "summary": summary,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=".")
    parser.add_argument("--export", required=True, help="Directory produced by literature_search_export.py")
    parser.add_argument("--allow-overwrite", action="store_true")
    args = parser.parse_args()

    repo_root = Path(args.repo).resolve()
    export_dir = Path(args.export).resolve()
    provenance = RunProvenance(
        experiment_id=EXPERIMENT_ID,
        script=Path(__file__).as_posix(),
        command=[sys.executable, *sys.argv],
        seed=None,
        root=repo_root,
        notes={
            "runtime_source_modified": False,
            "export_directory": str(export_dir),
            "export_checksums": {
                name: sha256_file(export_dir / name)
                for name in ("raw_records.json", "deduplicated_records.json", "search_manifest.json", "screening.csv")
                if (export_dir / name).exists()
            },
        },
    )
    try:
        result = run(repo_root, export_dir)
    except Exception as exc:  # noqa: BLE001
        print(f"literature screening failed: {exc}", file=sys.stderr)
        traceback.print_exc()
        provenance.exit_code = 2
        provenance.finished_at_utc = utc_now_iso()
        return 2

    summary = result["summary"]
    provenance.exit_code = 0 if summary["status"] == "COMPLETE" else 1
    provenance.finished_at_utc = utc_now_iso()

    writer = EvidenceWriter(
        experiment_id=EXPERIMENT_ID,
        repo_root=repo_root,
        provenance=provenance,
        allow_overwrite=args.allow_overwrite,
        configuration={
            "triage_version": TRIAGE_VERSION,
            "primary_terms": list(PRIMARY_TERMS),
            "secondary_terms": list(SECONDARY_TERMS),
            "exclusion_terms": list(EXCLUSION_TERMS),
            "included_set_size": len(INCLUDED_SET),
            "dimensions": [key for key, _ in COMPARISON_DIMENSIONS],
        },
        summary_payload=summary,
        extra_raw={
            "triage_rows": result["triage_rows"],
            "comparison_matrix": result["comparison_matrix"],
            "export_manifest": result["export_manifest"],
        },
        tables={
            "screening_dataset": [
                {
                    "record_key": row.get("record_key"),
                    "title": row.get("title"),
                    "doi": row.get("doi"),
                    "year": row.get("year"),
                    "venue": row.get("venue"),
                    "source_databases": "; ".join(row.get("source_databases", [])),
                    "triage_priority": row.get("triage_priority"),
                    "triage_score": row.get("triage_score"),
                    "primary_terms": "; ".join(row.get("primary_terms", [])),
                    "exclusion_terms": "; ".join(row.get("exclusion_terms", [])),
                    "screening_decision": "",
                    "exclusion_reason": "",
                    "full_text_status": "",
                    "reviewer_notes": "",
                }
                for row in result["triage_rows"]
            ],
            "comparison_matrix": [
                {
                    key: value
                    for key, value in row.items()
                }
                for row in result["comparison_matrix"]["rows"]
            ],
            "comparison_dimensions": [
                {"key": d["key"], "definition": d["definition"]}
                for d in result["comparison_matrix"]["dimensions"]
            ],
        },
        interpretation={
            "purpose": "Auditable screening dataset and source-linked comparison matrix.",
            "triage_is_not_screening": (
                "The triage column is a versioned keyword rule used to prioritise human "
                "review. It writes no include/exclude decision, and no PRISMA-style count "
                "may be derived from it."
            ),
            "review_status": (
                "This remains a structured rapid review. It is not a systematic review: "
                "there is no dual independent screening, no full-text eligibility stage and "
                "no registered protocol."
            ),
            "comparison_scope": (
                "The matrix records what each cited source presents. It is a capability "
                "inventory, not a benchmark; no system was re-implemented or timed here."
            ),
        },
        limitations=[
            "Metadata-only discovery: records come from OpenAlex and Crossref APIs, not from "
            "IEEE Xplore, ACM DL, Scopus or Web of Science exports.",
            "Full-text eligibility review was not performed for the discovery set.",
            "Abstract and title metadata are incomplete for part of the corpus; the "
            "completeness audit records exactly how incomplete.",
            "The comparison matrix reflects the cited sources at review time; capability "
            "claims were not re-verified by re-implementation.",
        ],
    )
    manifest = writer.write()
    # Preserve the deduplicated corpus and export manifest alongside the evidence so
    # the screening dataset can be re-derived without another live search.
    snapshot_dir = Path(repo_root) / "research" / "literature" / "export"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    write_json(snapshot_dir / "search_manifest.json", result["export_manifest"])
    write_csv(
        snapshot_dir / "screening_dataset.csv",
        [
            {
                "record_key": row.get("record_key"),
                "title": row.get("title"),
                "doi": row.get("doi"),
                "year": row.get("year"),
                "venue": row.get("venue"),
                "source_databases": "; ".join(row.get("source_databases", [])),
                "triage_priority": row.get("triage_priority"),
                "triage_score": row.get("triage_score"),
            }
            for row in result["triage_rows"]
        ],
    )
    write_json(snapshot_dir / "comparison_matrix.json", result["comparison_matrix"])
    write_json(snapshot_dir / "metadata_and_dedup_audit.json", {
        "metadata_audit": summary["metadata_audit"],
        "dedup_audit": summary["dedup_audit"],
        "triage": summary["triage"],
    })

    print(
        f"[{EXPERIMENT_ID}] raw={summary['export_raw_records']} "
        f"dedup={summary['export_deduplicated_records']} "
        f"failures={summary['export_failures']} "
        f"high_priority={summary['triage'].get('HIGH', 0)} "
        f"status={summary['status']}"
    )
    print(f"evidence: research/evidence/{EXPERIMENT_ID}/{manifest['source_sha']}")
    return 0 if summary["status"] == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())