#!/usr/bin/env python3
"""EXP-11 Overclaim audit and machine-readable claim/evidence ledger.

Purpose
-------
Scan the repository's own prose for language that asserts more than the evidence
supports, and maintain a machine-readable ledger in which every research claim is
bound to the experiments that support, weaken or leave it untested.

The scan is a *candidate generator*, not a verdict: it reports every occurrence
with its file and context so a human adjudicates each one. Nothing is silently
rewritten.

Usage
-----
    PYTHONPATH=backend:research python research/experiments/run_claim_audit.py --repo .
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.evidence import EvidenceWriter  # noqa: E402
from lib.provenance import RunProvenance, utc_now_iso  # noqa: E402

EXPERIMENT_ID = "EXP-11-claim-audit"

SCAN_DIRECTORIES = ("research", "paper", "backend", "frontend", ".github")
SCAN_SUFFIXES = (".md", ".tex", ".py", ".ts", ".tsx", ".yml", ".yaml")
SKIP_PATH_PARTS = ("node_modules", "dist", "build", "evidence", ".git", "__pycache__")
# The auditor's own term table would otherwise dominate the candidate list.
SKIP_FILES = ("research/experiments/run_claim_audit.py",)

# Terms that assert more than a bounded, executed experiment can establish.
RISK_TERMS: Dict[str, str] = {
    "first": "priority claim",
    "novel": "priority claim",
    "state-of-the-art": "priority claim",
    "state of the art": "priority claim",
    "real-time": "latency claim",
    "realtime": "latency claim",
    "scalable": "scalability claim",
    "accurate": "accuracy claim",
    "exact": "exactness claim",
    "lossless": "exactness claim",
    "universal": "generality claim",
    "arbitrary": "generality claim",
    "all operators": "generality claim",
    "all architectures": "generality claim",
    "superior": "comparative claim",
    "production-ready": "maturity claim",
    "fully reproducible": "reproducibility claim",
    "complete": "completeness claim",
    "exhaustive": "completeness claim",
    "outperforms": "comparative claim",
    "significantly": "statistical claim",
    "proves": "proof claim",
    "guarantee": "guarantee claim",
}

# Explicit negations/self-limits are not overclaims.
SAFE_CONTEXT_PATTERNS = (
    re.compile(r"\bno\b", re.IGNORECASE),
    re.compile(r"\bnot\b", re.IGNORECASE),
    re.compile(r"\bwithout\b", re.IGNORECASE),
    re.compile(r"\bnever\b", re.IGNORECASE),
    re.compile(r"\bdo(?:es)? not\b", re.IGNORECASE),
    re.compile(r"\bavoid\b", re.IGNORECASE),
    re.compile(r"\bdoes not claim\b", re.IGNORECASE),
    re.compile(r"\bmust not\b", re.IGNORECASE),
    re.compile(r"\bno claim\b", re.IGNORECASE),
    re.compile(r"\bonly\b", re.IGNORECASE),
    re.compile(r"\bremains?\b", re.IGNORECASE),
)

# Claims with their evidence binding. `evidence` names experiment ids and the
# artifacts they produced; `status` is the auditable disposition.
LEDGER: List[Dict[str, Any]] = [
    {
        "claim_id": "C-01",
        "claim": "The repository contains separate execution paths: a NumPy graph simulator, a legacy execution engine, a framework-backed Lab trace path, a model registry and an illustrative stream.",
        "claim_type": "architecture",
        "evidence": ["EXP-09"],
        "evidence_artifacts": ["research/evidence/EXP-09-execution-path-audit"],
        "measured": "106 routes classified into 6 families; state ownership verified for the graph session store, the legacy engine and the registry.",
        "status": "SUPPORTED",
        "confidence": "high",
        "limitation": "Route classification is by path prefix; no runtime call-graph trace is recorded.",
        "allowed_wording": "Describe the paths separately and name the engine that produced each value.",
    },
    {
        "claim_id": "C-02",
        "claim": "The simulator's forward computation agrees with an independent reference implementation across all registered graph families.",
        "claim_type": "numerical fidelity",
        "evidence": ["EXP-03"],
        "evidence_artifacts": ["research/evidence/EXP-03-trace-fidelity-audit"],
        "measured": "20 supported cases; maximum final-output absolute error 1.11e-07 and maximum per-layer absolute error 4.51e-07 against a float64 reference.",
        "status": "SUPPORTED",
        "confidence": "high",
        "limitation": "Covers the registered cases only; the reference implements the standard operator definitions.",
        "allowed_wording": "Agreement is demonstrated for the registered fixed cases at float32 precision.",
    },
    {
        "claim_id": "C-03",
        "claim": "Analytic gradients from both shipped gradient implementations agree with central finite differences.",
        "claim_type": "numerical fidelity",
        "evidence": ["EXP-02", "EXP-09"],
        "evidence_artifacts": ["research/evidence/EXP-02-gradient-validation", "research/evidence/EXP-09-execution-path-audit"],
        "measured": (
            "364 comparable parameter-tensor checks across 20 cases and both the backward "
            "engine and the training engine; median relative L2 error 3.68e-06, maximum "
            "3.52e-02. The legacy ExecutionEngine dense backward adds 3 checks at "
            "relative L2 2.1e-05 to 7.4e-05."
        ),
        "status": "PARTIALLY_SUPPORTED",
        "confidence": "high",
        "limitation": (
            "18 checks are marked PARAM_SYNC_NOT_VERIFIED and 14 are inconclusive; "
            "24 have a numerically zero gradient and carry no information. "
            "ReLU configurations whose pre-activations occupy the kink band cannot be "
            "validated by central differences."
        ),
        "allowed_wording": "Gradients agree with central differences for the tested tensors; "
        "the validation is scoped by case, parameter, dtype, step size and tolerance.",
    },
    {
        "claim_id": "C-04",
        "claim": "Trace records faithfully represent the computation.",
        "claim_type": "trace fidelity",
        "evidence": ["EXP-03"],
        "evidence_artifacts": ["research/evidence/EXP-03-trace-fidelity-audit"],
        "measured": (
            "Dense-only graphs emit matmul/bias_add/activation records that are each "
            "reconstructed from their own recorded values; every other family emits one "
            "layer-level record per layer, validated at the layer boundary against the "
            "independent reference. Ordering and step selection were exactly validated."
        ),
        "status": "PARTIALLY_SUPPORTED",
        "confidence": "high",
        "limitation": (
            "Operation-level reconstruction exists only for dense-only graphs; "
            "non-dense graphs expose no intra-layer arithmetic."
        ),
        "allowed_wording": "Report operation-level fidelity for dense graphs and "
        "layer-boundary fidelity for other families; do not generalise one to the other.",
    },
    {
        "claim_id": "C-05",
        "claim": "A training step performs the documented optimizer update.",
        "claim_type": "training",
        "evidence": ["EXP-04"],
        "evidence_artifacts": ["research/evidence/EXP-04-training-update-validation"],
        "measured": (
            "100 parameter-tensor update checks across 19 cases and 4 optimizers; "
            "median relative update error 6.4e-08, maximum 1.4e-03; all 19 measured "
            "updates reduced the loss on the fixed task."
        ),
        "status": "PARTIALLY_SUPPORTED",
        "confidence": "high",
        "limitation": (
            "Single-batch updates only. The attention family additionally shows a state "
            "divergence: the packed array changes while the layer does not."
        ),
        "allowed_wording": "The optimizer applied the declared equation to the supplied "
        "gradient for the tested configurations.",
    },
    {
        "claim_id": "C-06",
        "claim": "The simulator converges on tasks it is given.",
        "claim_type": "training",
        "evidence": ["EXP-04"],
        "evidence_artifacts": ["research/evidence/EXP-04-training-update-validation"],
        "measured": "76 short single-batch trajectories all reduced the loss; median relative loss reduction 0.975.",
        "status": "PARTIALLY_SUPPORTED",
        "confidence": "medium",
        "limitation": (
            "Fixed deterministic synthetic tasks, single batch, 25 steps, no shuffling, "
            "no dropout, no dataset. This is loss reduction on a fixed task, not "
            "general convergence, generalisation or optimisation behaviour."
        ),
        "allowed_wording": "Loss decreased for the tested fixed configurations.",
    },
    {
        "claim_id": "C-07",
        "claim": "Inspection has a measurable, characterised runtime cost that grows with model size.",
        "claim_type": "performance",
        "evidence": ["EXP-05", "EXP-06"],
        "evidence_artifacts": ["research/evidence/EXP-05-performance-benchmark", "research/evidence/EXP-06-scalability-study"],
        "measured": (
            "14 workloads with 60 samples per component after 10 warm-ups. The paired "
            "trace/forward ratio ranges from 1.07x (recurrent) to 23.0x (dense 128-256-64) "
            "with 95% intervals, and the residual is dominated by JSON serialisation of the "
            "record payload rather than by the traced arithmetic. Depth and width sweeps "
            "give log-log exponents with reported R-squared and tested ranges."
        ),
        "status": "SUPPORTED",
        "confidence": "high",
        "limitation": (
            "Single machine, single process, tracemalloc rather than resident memory, no "
            "HTTP transport or browser rendering, and hosted-CI timings must not be pooled "
            "with these."
        ),
        "allowed_wording": "Report the decomposed components, the paired ratio with its "
        "interval, and the environment; do not present the ratio as a universal tracing cost.",
    },
    {
        "claim_id": "C-08",
        "claim": "Execution is deterministic and reproducible.",
        "claim_type": "reproducibility",
        "evidence": ["EXP-07"],
        "evidence_artifacts": ["research/evidence/EXP-07-determinism-study"],
        "measured": (
            "20 cases; 5 repeats each for parameter construction, forward output, serialised "
            "trace and a short training trajectory, all bit-identical. 3 cases repeated across "
            "5 fresh interpreters produced identical parameter and trace digests."
        ),
        "status": "SUPPORTED",
        "confidence": "high",
        "limitation": "One machine and one interpreter build; cross-machine determinism is not claimed.",
        "allowed_wording": "Repeatable within a process and across processes on the recorded environment.",
    },
    {
        "claim_id": "C-09",
        "claim": "Every supported operator is correct end to end.",
        "claim_type": "generality",
        "evidence": ["EXP-01", "EXP-02", "EXP-03", "EXP-08"],
        "evidence_artifacts": ["research/evidence/EXP-01-graph-capability-audit", "research/evidence/EXP-08-defect-register"],
        "measured": "7 confirmed defects, including a blocking route/schema boundary and a state divergence in the attention path.",
        "status": "INVALIDATED",
        "confidence": "high",
        "limitation": "The claim is contradicted by reproduced defects.",
        "allowed_wording": "Remove the claim. Replace with the operator-family coverage table and the defect register.",
    },
    {
        "claim_id": "C-10",
        "claim": "Displayed values are tied to the selected graph, execution and parameter revision.",
        "claim_type": "provenance",
        "evidence": ["EXP-01", "EXP-08"],
        "evidence_artifacts": ["research/evidence/EXP-08-defect-register", "research/ADVERSARIAL_UI_PROVENANCE.md"],
        "measured": "An adversarial browser test reproduces stale inspection state after a graph switch and after an out-of-order response; those tests remain expected failures.",
        "status": "INVALIDATED",
        "confidence": "high",
        "limitation": "The defect is reproduced and not fixed; fixing it requires runtime authorisation.",
        "allowed_wording": "State the reproduced provenance defect as a limitation.",
    },
    {
        "claim_id": "C-11",
        "claim": "Multi-head attention is supported.",
        "claim_type": "generality",
        "evidence": ["EXP-01", "EXP-08"],
        "evidence_artifacts": ["research/evidence/EXP-08-defect-register"],
        "measured": "num_heads is stored but unused; one-head and two-head graphs with identical projections produce identical outputs.",
        "status": "INVALIDATED",
        "confidence": "high",
        "limitation": "Only single-head attention is implemented.",
        "allowed_wording": "Describe single-head attention only.",
    },
    {
        "claim_id": "C-12",
        "claim": "Attention parameters train.",
        "claim_type": "training",
        "evidence": ["EXP-04", "EXP-08"],
        "evidence_artifacts": ["research/evidence/EXP-08-defect-register"],
        "measured": "After a full optimizer step the packed attention array changes while all four projections used by the forward pass are unchanged.",
        "status": "INVALIDATED",
        "confidence": "high",
        "limitation": "The defect is reproduced and not fixed.",
        "allowed_wording": "Remove the claim; report the reproduced divergence.",
    },
    {
        "claim_id": "C-13",
        "claim": "The system improves learning, usability or debugging outcomes.",
        "claim_type": "user benefit",
        "evidence": [],
        "evidence_artifacts": [],
        "measured": "No controlled study exists.",
        "status": "UNSUPPORTED",
        "confidence": "n/a",
        "limitation": "Requires a human-subject protocol that has not been run.",
        "allowed_wording": "Make no user-benefit claim.",
    },
    {
        "claim_id": "C-14",
        "claim": "Neurofluxion is the first, novel or superior system for this task.",
        "claim_type": "priority",
        "evidence": ["EXP-10"],
        "evidence_artifacts": ["research/evidence/EXP-10-literature-screening", "research/literature/export/comparison_matrix.json"],
        "measured": (
            "A 2,076-record two-source discovery set with 0 export failures and 13 systems "
            "in a source-linked capability matrix. The review is a structured rapid review "
            "without dual screening or a full-text eligibility stage."
        ),
        "status": "UNSUPPORTED",
        "confidence": "n/a",
        "limitation": "No systematic review supports a priority claim.",
        "allowed_wording": "Avoid priority and superiority language entirely.",
    },
    {
        "claim_id": "C-15",
        "claim": "The simulator's computed values can be mistaken for a framework model's activations.",
        "claim_type": "architecture",
        "evidence": ["EXP-01", "EXP-09"],
        "evidence_artifacts": ["research/evidence/EXP-09-execution-path-audit"],
        "measured": "No verified import path links simulator graphs to registry weights; each path owns separate state.",
        "status": "SUPPORTED",
        "confidence": "high",
        "limitation": "Absence of a link was verified for the inspected code paths, not by an exhaustive import search.",
        "allowed_wording": "Keep the paths explicitly separate; no shared-weights claim is made.",
    },
    {
        "claim_id": "C-16",
        "claim": "Some user-visible values are synthesised rather than computed.",
        "claim_type": "provenance",
        "evidence": ["EXP-09"],
        "evidence_artifacts": ["research/evidence/EXP-09-execution-path-audit"],
        "measured": (
            "The /stream payload uses a fixed 3-4-2 topology, a hard-coded weight table and "
            "closed-form loss/accuracy curves of a counter, and reads no configured graph. "
            "The Simulator demo requests a softmax output that the activation registry does "
            "not implement, so the displayed outputs are logits."
        ),
        "status": "SUPPORTED",
        "confidence": "high",
        "limitation": "Frontend label coverage was checked by text search; absence of a match is not proof of absence.",
        "allowed_wording": "Describe the stream as illustrative and the demo output as logits.",
    },
    {
        "claim_id": "C-17",
        "claim": "Convolution configurations in the simulator are equivalent to framework implementations.",
        "claim_type": "compatibility",
        "evidence": ["EXP-03", "EXP-08"],
        "evidence_artifacts": ["research/evidence/EXP-08-defect-register"],
        "measured": "'same' padding with stride 2 produces a 5x5 output where the conventional rule gives 3x3; with the simulator's own geometry the arithmetic matches the reference to 3.5e-08.",
        "status": "INVALIDATED",
        "confidence": "high",
        "limitation": "Only the stride > 1 'same' configuration was probed.",
        "allowed_wording": "Do not claim framework equivalence; state the padding deviation.",
    },
]


def _looks_like_prose(file_path: str, line: str) -> bool:
    """Separate manuscript/research prose from code identifiers and literals.

    Technical code legitimately contains the words "first", "complete" and
    "exact" as identifiers, status values and messages. Flagging those as
    overclaims would bury the occurrences that matter, so they are classified
    separately instead of being silently dropped.
    """
    if file_path.endswith((".md", ".tex")):
        return True
    stripped = line.strip()
    if stripped.startswith(("#", "//", "*", '"""', "'''")):
        return True
    code_signals = (
        "def ", "class ", "self.", "return ", "assert ", "import ", "=>", "::",
        "raise ", "=", "{", "}", "(", ")", "[", "]",
    )
    return not any(signal in stripped for signal in code_signals)


def scan_overclaims(root: Path) -> Dict[str, Any]:
    findings: List[Dict[str, Any]] = []
    files_scanned = 0
    for directory in SCAN_DIRECTORIES:
        base = root / directory
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in SCAN_SUFFIXES:
                continue
            relative = path.relative_to(root).as_posix()
            if any(part in relative.split("/") for part in SKIP_PATH_PARTS):
                continue
            if relative in SKIP_FILES:
                continue
            files_scanned += 1
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            for number, line in enumerate(text.splitlines(), 1):
                lowered = line.lower()
                for term, kind in RISK_TERMS.items():
                    if term not in lowered:
                        continue
                    negated = any(pattern.search(line) for pattern in SAFE_CONTEXT_PATTERNS)
                    prose = _looks_like_prose(relative, line)
                    findings.append(
                        {
                            "file": relative,
                            "line": number,
                            "term": term,
                            "claim_kind": kind,
                            "context_class": "prose" if prose else "code_identifier_or_literal",
                            "negation_or_scope_marker_detected": negated,
                            "requires_human_adjudication": bool(prose and not negated),
                            "text": line.strip()[:400],
                        }
                    )
    by_term: Dict[str, int] = {}
    for finding in findings:
        by_term[finding["term"]] = by_term.get(finding["term"], 0) + 1
    prose_only = [f for f in findings if f["context_class"] == "prose"]
    return {
        "files_scanned": files_scanned,
        "occurrences_total": len(findings),
        "occurrences_prose": len(prose_only),
        "occurrences_code": len(findings) - len(prose_only),
        "occurrences_requiring_adjudication": sum(
            1 for f in findings if f["requires_human_adjudication"]
        ),
        "occurrences_by_term": dict(sorted(by_term.items(), key=lambda kv: (-kv[1], kv[0]))),
        "findings": findings,
        "method_note": (
            "The detector is a keyword scan over repository prose and source comments. "
            "Occurrences are classified as prose or code identifier/literal; only un-negated "
            "prose occurrences are flagged for human adjudication. It produces candidates, "
            "not verdicts."
        ),
    }


def run(repo_root: Path) -> Dict[str, Any]:
    scan = scan_overclaims(root=repo_root)
    status_counts: Dict[str, int] = {}
    for entry in LEDGER:
        status_counts[entry["status"]] = status_counts.get(entry["status"], 0) + 1
    summary = {
        "ledger_entries": len(LEDGER),
        "ledger_status_counts": status_counts,
        "claims_supported": [c["claim_id"] for c in LEDGER if c["status"] == "SUPPORTED"],
        "claims_partially_supported": [c["claim_id"] for c in LEDGER if c["status"] == "PARTIALLY_SUPPORTED"],
        "claims_unsupported": [c["claim_id"] for c in LEDGER if c["status"] == "UNSUPPORTED"],
        "claims_invalidated": [c["claim_id"] for c in LEDGER if c["status"] == "INVALIDATED"],
        "claims_not_tested": [c["claim_id"] for c in LEDGER if c["status"] in {"NOT_TESTED", "REQUIRES_MORE_EVIDENCE"}],
        "overclaim_scan": {
            "files_scanned": scan["files_scanned"],
            "occurrences_total": scan["occurrences_total"],
            "occurrences_prose": scan["occurrences_prose"],
            "occurrences_code": scan["occurrences_code"],
            "occurrences_requiring_adjudication": scan["occurrences_requiring_adjudication"],
            "occurrences_by_term": scan["occurrences_by_term"],
        },
        "status": "COMPLETE",
    }
    return {"scan": scan, "ledger": LEDGER, "summary": summary}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=".")
    parser.add_argument("--allow-overwrite", action="store_true")
    args = parser.parse_args()

    repo_root = Path(args.repo).resolve()
    provenance = RunProvenance(
        experiment_id=EXPERIMENT_ID,
        script=Path(__file__).as_posix(),
        command=[sys.executable, *sys.argv],
        seed=None,
        root=repo_root,
        notes={"runtime_source_modified": False, "audit_scope": "repository prose and source comments"},
    )
    try:
        result = run(repo_root)
    except Exception as exc:  # noqa: BLE001
        print(f"claim audit failed: {exc}", file=sys.stderr)
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
        allow_overwrite=args.allow_overwrite,
        configuration={
            "risk_terms": RISK_TERMS,
            "scan_directories": list(SCAN_DIRECTORIES),
            "scan_suffixes": list(SCAN_SUFFIXES),
            "skipped_path_parts": list(SKIP_PATH_PARTS),
            "skipped_files": list(SKIP_FILES),
            "status_vocabulary": [
                "SUPPORTED",
                "PARTIALLY_SUPPORTED",
                "UNSUPPORTED",
                "INVALIDATED",
                "REQUIRES_MORE_EVIDENCE",
                "NOT_TESTED",
            ],
        },
        summary_payload=result["summary"],
        extra_raw={
            "overclaim_findings": result["scan"]["findings"],
            "claim_ledger": result["ledger"],
        },
        tables={
            "claim_evidence_ledger": [
                {
                    "claim_id": entry["claim_id"],
                    "claim": entry["claim"],
                    "claim_type": entry["claim_type"],
                    "status": entry["status"],
                    "confidence": entry["confidence"],
                    "evidence_experiments": "; ".join(entry["evidence"]),
                    "evidence_artifacts": "; ".join(entry["evidence_artifacts"]),
                    "measured": entry["measured"],
                    "limitation": entry["limitation"],
                    "allowed_wording": entry["allowed_wording"],
                }
                for entry in result["ledger"]
            ],
            "overclaim_candidates": [
                {
                    "file": f["file"],
                    "line": f["line"],
                    "term": f["term"],
                    "claim_kind": f["claim_kind"],
                    "context_class": f["context_class"],
                    "text": f["text"],
                }
                for f in result["scan"]["findings"]
                if f["requires_human_adjudication"]
            ],
        },
        interpretation={
            "purpose": "Bind every research claim to its evidence and surface overclaim candidates.",
            "detector": (
                "The keyword detector finds candidates; it does not decide. A term inside a "
                "negated or scope-limited sentence is reported but not flagged for adjudication."
            ),
            "ledger_rule": (
                "A claim is SUPPORTED only when an executed experiment in this repository "
                "measures it. Claims without evidence are UNSUPPORTED; claims contradicted "
                "by a reproduced defect are INVALIDATED."
            ),
        },
        limitations=[
            "The detector is English-language and keyword based; it will miss rhetorical "
            "overstatement and will flag harmless technical uses of words such as 'complete'.",
            "Candidate occurrences are not automatically defects and require human review.",
        ],
    )
    manifest = writer.write()
    summary = result["summary"]
    print(
        f"[{EXPERIMENT_ID}] ledger_entries={summary['ledger_entries']} "
        f"invalidated={len(summary['claims_invalidated'])} "
        f"unsupported={len(summary['claims_unsupported'])} "
        f"overclaim_candidates={summary['overclaim_scan']['occurrences_requiring_adjudication']}"
    )
    print(f"evidence: research/evidence/{EXPERIMENT_ID}/{manifest['source_sha']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())