import { test, expect } from "@playwright/test";

test("Simulator displays the exact backend trace returned for its demo run", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByRole("status").getByText(/Backend ready/)).toBeVisible({ timeout: 30_000 });

  const forwardResponsePromise = page.waitForResponse((response) =>
    response.url().includes("/backend/api/simulator/forward/full") &&
    response.request().method() === "POST"
  );
  await page.getByRole("button", { name: /Simulator/ }).click();
  const forwardResponse = await forwardResponsePromise;
  expect(forwardResponse.ok()).toBeTruthy();

  const payload = await forwardResponse.json();
  expect(Array.isArray(payload.steps)).toBeTruthy();
  expect(payload.steps.length).toBeGreaterThan(0);
  await expect(page).toHaveTitle("Simulator · Neurofluxion");
  await expect(page.getByRole("note")).toContainText("ran one random forward pass");

  await page.getByRole("tab", { name: "Run" }).click();
  await expect(page.locator(".pass-step")).toHaveCount(payload.steps.length);
  await expect(page.locator(".pass-step-title").first()).toHaveText(payload.steps[0].description);
  const firstOutput = payload.steps[0].output_values.slice(0, 6).map((value) => value.toFixed(3)).join(", ");
  await expect(page.locator(".pass-step-out").first()).toHaveText(`Out: ${firstOutput}`);
});
