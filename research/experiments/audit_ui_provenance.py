#!/usr/bin/env python3
"""Source-level provenance audit for the Neurofluxion UI and execution streams.

This deliberately reports detected gaps instead of treating a clean script exit
as proof that the UI is provenance-safe. It makes no runtime source changes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def commit_for(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def audit(root: Path) -> dict:
    paths = {
        "computation_store": root / "frontend/src/store/computationStore.ts",
        "simulator_store": root / "frontend/src/store/simulatorStore.ts",
        "simulator_page": root / "frontend/src/pages/SimulatorPage.tsx",
        "stream": root / "backend/api/stream_ws.py",
        "simulator_api": root / "frontend/src/hooks/useSimulatorApi.ts",
    }
    missing = [key for key, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required audit sources missing: {missing}")

    source = {key: path.read_text(encoding="utf-8-sig") for key, path in paths.items()}
    comp = source["computation_store"]
    sim_store = source["simulator_store"]
    page = source["simulator_page"]
    stream = source["stream"]
    api = source["simulator_api"]

    has_async_full_forward = "async runFullForward(input)" in comp and "await simulatorApi.forwardFull(graphId, input)" in comp
    writes_after_await = "set({ steps: res.steps, layerOutputs: res.layer_outputs })" in comp
    has_graph_identity_guard = any(token in comp for token in (
        "getState().graphId !== graphId",
        "requestGraphId !== graphId",
        "requestId !== latestRequestId",
        "AbortController",
    ))
    graph_change_resets_computation = (
        "setGraphId: (id)" in sim_store
        and ("useComputationStore" in sim_store or "steps: []" in sim_store or "reset()" in sim_store)
    )
    randomized_demo_input = "Math.random() * 2 - 1" in page
    explicit_demo_provenance_label = any(token in page for token in (
        "Demo-generated input", "Demo input — synthetic", "provenanceBadge", "data-provenance",
    ))
    stream_is_synthesized = "_build_snapshot(epoch)" in stream and "math.sin(epoch" in stream
    stream_has_graph_correlation = any(token in stream for token in (
        "graph_id", "graphId", "execution_id", "run_id", "parameter_revision",
    ))
    api_uses_simulator_routes = '"/api/simulator/forward/full"' in api

    checks = [
        {
            "id": "P1_STALE_FORWARD_RESPONSE_GUARD",
            "status": "PASS" if has_graph_identity_guard else "GAP",
            "finding": "The async forward store has an explicit stale-request/graph identity guard."
            if has_graph_identity_guard else
            "The inspected full-forward action awaits a response and writes it to shared state without an obvious request/graph identity guard; an older response may overwrite newer graph state.",
            "source": "frontend/src/store/computationStore.ts",
        },
        {
            "id": "P2_GRAPH_SWITCH_CLEARS_INSPECTION_STATE",
            "status": "PASS" if graph_change_resets_computation else "GAP",
            "finding": "Graph changes visibly reset or invalidate computation inspection state."
            if graph_change_resets_computation else
            "The graph store's setGraphId action does not visibly clear computation steps/inspection state; stale values may remain until another action replaces them.",
            "source": "frontend/src/store/simulatorStore.ts",
        },
        {
            "id": "P3_DEMO_INPUT_PROVENANCE",
            "status": "PASS" if not randomized_demo_input or explicit_demo_provenance_label else "GAP",
            "finding": "Random demo inputs have an explicit UI provenance label."
            if explicit_demo_provenance_label else
            "SimulatorPage generates random demo inputs. A dataset metadata name alone is not verified here as a persistent, visible provenance label beside displayed values.",
            "source": "frontend/src/pages/SimulatorPage.tsx",
        },
        {
            "id": "P4_STREAM_GRAPH_RUN_CORRELATION",
            "status": "PASS" if not stream_is_synthesized or stream_has_graph_correlation else "GAP",
            "finding": "The stream is correlated to an explicit graph/execution/parameter revision."
            if stream_has_graph_correlation else
            "The inspected stream constructs epoch-varying values from a fixed illustrative topology and exposes no obvious graph/run/parameter-revision correlation.",
            "source": "backend/api/stream_ws.py",
        },
        {
            "id": "P5_SIMULATOR_ROUTE_IDENTITY",
            "status": "PASS" if api_uses_simulator_routes else "REVIEW",
            "finding": "The frontend API wrapper includes the simulator forward route."
            if api_uses_simulator_routes else
            "The expected simulator forward route string was not found; route wiring requires manual review.",
            "source": "frontend/src/hooks/useSimulatorApi.ts",
        },
        {
            "id": "P6_ASYNC_FULL_FORWARD_PATTERN",
            "status": "GAP" if has_async_full_forward and writes_after_await and not has_graph_identity_guard else "REVIEW",
            "finding": "A specific async full-forward write-after-await pattern was detected without a recognized identity guard."
            if has_async_full_forward and writes_after_await and not has_graph_identity_guard else
            "The expected async full-forward pattern changed; review this detector and inspect the store manually.",
            "source": "frontend/src/store/computationStore.ts",
        },
    ]
    return {
        "schema_version": 1,
        "audit_kind": "source-inspection; not a runtime/browser test",
        "metadata": {
            "commit": commit_for(root),
            "platform": platform.platform(),
            "source_sha256": {key: sha(path) for key, path in paths.items()},
        },
        "summary": {
            "checks": len(checks),
            "gaps": sum(item["status"] == "GAP" for item in checks),
            "review_required": sum(item["status"] == "REVIEW" for item in checks),
            "pass": sum(item["status"] == "PASS" for item in checks),
        },
        "checks": checks,
        "interpretation": (
            "A successful process exit means the audit report was generated, not that all checks passed. "
            "GAP entries are source-level risks requiring runtime tests or a separately authorized implementation change."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=".")
    parser.add_argument("--out", default="research/artifacts/ui-provenance-audit.json")
    args = parser.parse_args()
    root = Path(args.repo).resolve()
    report = audit(root)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), **report["summary"], "commit": report["metadata"]["commit"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
