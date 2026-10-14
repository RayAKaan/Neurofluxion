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
