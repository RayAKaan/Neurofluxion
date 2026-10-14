# Adversarial UI Provenance Browser Test

The test in frontend/e2e/stale-forward-provenance.spec.mjs delays a forward response, builds a replacement graph, then releases the old response and asserts that the stale sentinel is not displayed.

## Current state

The test is marked as an expected failure while the known stale-response gap remains. This keeps the failure visible in CI without introducing a knowingly red required check. Playwright treats an unexpected pass as a failure, which will signal that the expectation marker should be removed after a verified fix.

The test uses a deliberately synthetic sentinel payload, so it does not depend on output values coincidentally matching across two graphs. It verifies a UI state invariant, not numerical correctness of the simulator.

## Runtime scope

This PR adds a browser test and CI workflow only. It does not modify the application stores or components. The separate runtime change needed to invalidate or correlate outstanding requests remains unauthorized under the current research-only scope.

## CI

Workflow: .github/workflows/adversarial-ui-provenance.yml. It starts the actual backend and frontend and runs the single adversarial Playwright scenario. On failure, Playwright artifacts are retained for inspection.

## First browser reproduction (2026-10-14)

The dedicated workflow [38574488415](https://github.com/RayAKaan/Neurofluxion/actions/runs/38574488415) completed successfully with the adversarial test reported as an expected failure. The test delayed the first Run-tab forward response, built a replacement graph, released a synthetic sentinel response, and observed that the stale sentinel was rendered. This reproduces the selected-graph/stale-response UI defect in the current implementation.

The workflow is green because the known defect is marked as an expected failure, not because the invariant passed. The artifact is retained at [neurofluxion-adversarial-browser-report](https://github.com/RayAKaan/Neurofluxion/actions/runs/38574488415/artifacts/11898532514).

Next action to close the defect requires a separately authorized runtime change that correlates responses to graph/request identity and invalidates obsolete inspection state. Until then, keep this test as an explicit expected failure and do not claim provenance integrity.
