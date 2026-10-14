# UI Provenance and Stale-State Audit

**Audit type:** source inspection plus a generated machine-readable report. This is not a browser-driven test and does not prove runtime behavior.

## Why this audit exists

Neurofluxion contains separate computation surfaces: the NumPy-backed simulator, a framework-backed Lab path, and an illustrative WebSocket snapshot stream. A displayed value must not be described as current model state unless it is tied to the graph, execution, and parameter revision that produced it.

## Source-level findings to validate

The audit script inspects a frozen checkout and emits one JSON record per check. Its exit code indicates whether the report was generated; GAP findings do not fail the process. Never interpret a green workflow as a provenance-safety pass.

| Check | Risk being investigated | Evidence required to close |
|---|---|---|
| P1 stale forward response | An async forward response may update shared state after the selected graph/request has changed | Browser/store race test that delays request A, selects graph B, resolves B then A, and verifies A cannot overwrite B |
| P2 graph switch reset | Previous steps, layer outputs, or inspection panels may remain visible after changing graph | Test graph A inspection, switch to graph B, and assert old steps and inspection payloads are invalidated |
| P3 demo input provenance | The Simulator page creates random demo input on mount | Browser assertion that synthetic/demo inputs are explicitly labelled wherever displayed; metadata hidden elsewhere is insufficient |
| P4 stream correlation | The snapshot stream uses a fixed illustrative topology and epoch-varying values | Either label it persistently as illustrative, or provide and test graph/run/parameter-revision correlation and arithmetic consistency |
| P5 API route identity | The simulator page should use simulator endpoints, not silently mix Lab/legacy execution state | Route interception or API-level test matching displayed output to the exact response and graph ID |

## Current handling

This repository workstream is research-only. It does not change React stores, UI components, or backend runtime behavior. If a source-level gap is confirmed, preserve it in the paper and evidence ledger; implementation fixes require separate authorization and a separate PR.

## Required adversarial browser scenarios

1. **Out-of-order responses:** delay forward request A, switch graph, issue request B, resolve B first and A last.
2. **Graph replacement:** inspect graph A, build graph B, confirm no A steps/weights/activations remain visible as B state.
3. **Unmount/remount:** navigate away while demo initialization is pending; confirm late responses do not mutate the new page's stores.
4. **Failure after success:** make a second request fail and verify the UI does not present old output as the new request's result.
5. **Stream honesty:** assert the stream UI explicitly says illustrative/synthetic unless a matching graph/run identity is present.
6. **Demo origin:** assert generated random inputs and one-hot targets are marked as demo-generated, not measured dataset results.

## Interpretation

A source audit can identify risky patterns but cannot prove a race occurred in a particular browser schedule. Conversely, the absence of a matched string is not proof of a defect. The report therefore distinguishes source findings from browser reproduction, and each gap must be closed by an adversarial runtime test or kept as an explicit limitation.

## First source-audit result (2026-10-14)

The CI report from [workflow run 38571759236](https://github.com/RayAKaan/Neurofluxion/actions/runs/38571759236) inspected checkout `da3529586a31e8d0d3d838279a87d560b33e2fa2`. It recorded five source-level gaps and one route-identity check pass:

| ID | Result | Finding |
|---|---|---|
| P1 | GAP | Full-forward async response writes shared state without an obvious request/graph identity guard. |
| P2 | GAP | Graph selection does not visibly invalidate computation steps/inspection state in the simulator store action. |
| P3 | GAP | Random demo inputs are generated; a persistent visible provenance label next to displayed values was not verified. |
| P4 | GAP | The epoch-varying fixed-topology stream has no obvious graph/run/parameter-revision correlation. |
| P5 | PASS | The frontend API wrapper includes the simulator forward route. |
| P6 | GAP | The write-after-await pattern was detected without a recognized stale-request guard. |

Artifact: [neurofluxion-ui-provenance-audit](https://github.com/RayAKaan/Neurofluxion/actions/runs/38571759236/artifacts/11896823883).

These findings are source-inspection results, not browser-reproduced failures. P1/P2 need delayed-response and graph-switch browser/store tests; P3/P4 need a UI assertion or explicit documented contract. This research PR does not implement fixes in runtime code.

A second source-audit execution, [run 38572250704](https://github.com/RayAKaan/Neurofluxion/actions/runs/38572250704), on checkout `3d62726c236e762fe370e13ae6d11fa8803ea573` reproduced the same classification: 5 GAP entries and 1 PASS entry. Its artifact is [here](https://github.com/RayAKaan/Neurofluxion/actions/runs/38572250704/artifacts/11897898614). This supports repeatability of the source-pattern detector, not proof of a browser-level race.
