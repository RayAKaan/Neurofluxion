import { test, expect } from "@playwright/test";

// Expected failure while the current RunInspectTab handler can commit an old
// forward response after a graph replacement. Remove test.fail only after the
// runtime fix makes this adversarial scenario pass.
test("stale forward response is not shown after the selected graph changes", async ({ page }) => {
  test.fail(true, "Known provenance gap: late forward responses are not correlated with the currently selected graph.");

  let releaseDelayedResponse;
  let signalRequestStarted;
  const delayedResponseGate = new Promise((resolve) => {
    releaseDelayedResponse = resolve;
  });
  const requestStarted = new Promise((resolve) => {
    signalRequestStarted = resolve;
  });
  let delayNextForward = true;

  await page.goto("/");
  await expect(
    page.getByRole("complementary", { name: "Primary navigation" }).getByText("Backend ready", { exact: true })
  ).toBeVisible({ timeout: 30_000 });
  await page.getByRole("button", { name: /Simulator/ }).click();
  await expect(page).toHaveTitle("Simulator · Neurofluxion");
  await expect(page.getByRole("note")).toContainText("ran one random forward pass");
  await page.getByRole("tab", { name: "Run" }).click();

  await page.route("**/backend/api/simulator/forward/full", async (route) => {
    if (!delayNextForward) {
      await route.continue();
      return;
    }
    delayNextForward = false;
    signalRequestStarted();
    await delayedResponseGate;
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        steps: [{
          step_index: 0,
          layer_index: 0,
          operation: "activation",
          description: "STALE_GRAPH_RESPONSE_SENTINEL",
          output_values: [9.99],
        }],
        layer_outputs: {},
        total_steps: 1,
      }),
    });
  });

  await page.getByRole("button", { name: /Forward Pass/ }).click();
  await requestStarted;

  // Build a new graph while the previous graph's forward request is unresolved.
  await page.getByRole("tab", { name: "Build" }).click();
  const buildResponse = page.waitForResponse((response) =>
    response.url().includes("/backend/api/simulator/architecture/build") &&
    response.request().method() === "POST" &&
    response.status() === 200
  );
  await page.getByRole("button", { name: "Build", exact: true }).click();
  await buildResponse;

  releaseDelayedResponse();
  await page.getByRole("tab", { name: "Run" }).click();
  await expect(page.getByText("STALE_GRAPH_RESPONSE_SENTINEL", { exact: true })).not.toBeVisible();
});


// This second scenario isolates graph-switch invalidation from the out-of-order
// response race above: the forward response completes before graph replacement.
test("inspection from the previous graph is cleared after graph replacement", async ({ page }) => {
  test.fail(true, "Known provenance gap: graph replacement may leave previous inspection state visible.");

  await page.goto("/");
  await expect(
    page.getByRole("complementary", { name: "Primary navigation" }).getByText("Backend ready", { exact: true })
  ).toBeVisible({ timeout: 30_000 });
  await page.getByRole("button", { name: /Simulator/ }).click();
  await expect(page).toHaveTitle("Simulator · Neurofluxion");
  await expect(page.getByRole("note")).toContainText("ran one random forward pass");
  await page.getByRole("tab", { name: "Run" }).click();

  await page.route("**/backend/api/simulator/forward/full", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        steps: [{
          step_index: 0,
          layer_index: 0,
          operation: "activation",
          description: "PREVIOUS_GRAPH_INSPECTION_SENTINEL",
          output_values: [7.77],
        }],
        layer_outputs: {},
        total_steps: 1,
      }),
    });
  });

  const forwardResponse = page.waitForResponse((response) =>
    response.url().includes("/backend/api/simulator/forward/full") && response.status() === 200
  );
  await page.getByRole("button", { name: /Forward Pass/ }).click();
  await forwardResponse;
  await expect(page.getByText("PREVIOUS_GRAPH_INSPECTION_SENTINEL", { exact: true })).toBeVisible();

  await page.getByRole("tab", { name: "Build" }).click();
  const buildResponse = page.waitForResponse((response) =>
    response.url().includes("/backend/api/simulator/architecture/build") &&
    response.request().method() === "POST" &&
    response.status() === 200
  );
  await page.getByRole("button", { name: "Build", exact: true }).click();
  await buildResponse;
  await page.getByRole("tab", { name: "Run" }).click();

  await expect(page.getByText("PREVIOUS_GRAPH_INSPECTION_SENTINEL", { exact: true })).not.toBeVisible();
});
