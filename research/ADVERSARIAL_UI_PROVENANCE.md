# Adversarial UI Provenance Browser Test

The test in frontend/e2e/stale-forward-provenance.spec.mjs delays a forward response, builds a replacement graph, then releases the old response and asserts that the stale sentinel is not displayed.

## Current state

The test is marked as an expected failure while the known stale-response gap remains. This keeps the failure visible in CI without introducing a knowingly red required check. Playwright treats an unexpected pass as a failure, which will signal that the expectation marker should be removed after a verified fix.

The test uses a deliberately synthetic sentinel payload, so it does not depend on output values coincidentally matching across two graphs. It verifies a UI state invariant, not numerical correctness of the simulator.

## Runtime scope

This PR adds a browser test and CI workflow only. It does not modify the application stores or components. The separate runtime change needed to invalidate or correlate outstanding requests remains unauthorized under the current research-only scope.

## CI

Workflow: .github/workflows/adversarial-ui-provenance.yml. It starts the actual backend and frontend and runs the single adversarial Playwright scenario. On failure, Playwright artifacts are retained for inspection.
