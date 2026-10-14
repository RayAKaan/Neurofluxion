import { test, expect } from "@playwright/test";

test("Simulator displays the exact backend trace returned for its current run", async ({ page }) => {
  const forwardResponses = [];
  page.on("response", (response) => {
    if (
      response.url().includes("/backend/api/simulator/forward/full") &&
      response.request().method() === "POST"
    ) {
      forwardResponses.push(response);
    }
  });

  await page.goto("/");
  await expect(page.getByText("Backend ready", { exact: true })).toBeVisible({ timeout: 30_000 });
  await page.getByRole("button", { name: /Simulator/ }).click();
  await expect(page).toHaveTitle("Simulator · Neurofluxion");
  await expect(page.getByRole("note")).toContainText("ran one random forward pass");

  const responsesBeforeRunTab = forwardResponses.length;
  await page.getByRole("tab", { name: "Run" }).click();
  await expect.poll(() => forwardResponses.length, { timeout: 15_000 }).toBeGreaterThan(responsesBeforeRunTab);
  await page.waitForLoadState("networkidle");
  await page.waitForTimeout(250);

  // Run tab can trigger a second forward pass with a fresh random input. Compare
  // the visible trace to the latest response, not the earlier demo initialization.
  const payloads = await Promise.all(forwardResponses.map((response) => response.json()));
  const payload = payloads[payloads.length - 1];
  expect(Array.isArray(payload.steps)).toBeTruthy();
  expect(payload.steps.length).toBeGreaterThan(0);
  await expect(page.locator(".pass-step")).toHaveCount(payload.steps.length);
  await expect(page.locator(".pass-step-title").first()).toHaveText(payload.steps[0].description);
  const firstOutput = payload.steps[0].output_values.slice(0, 6).map((value) => value.toFixed(3)).join(", ");
  await expect(page.locator(".pass-step-out").first()).toHaveText(`Out: ${firstOutput}`);
});
