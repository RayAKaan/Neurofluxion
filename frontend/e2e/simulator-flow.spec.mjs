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
  // The app also mirrors this status in an aria-live region; scope to the visible
  // navigation status to avoid Playwright strict-mode ambiguity.
  await expect(
    page.getByRole("complementary", { name: "Primary navigation" }).getByText("Backend ready", { exact: true })
  ).toBeVisible({ timeout: 30_000 });

  await page.getByRole("button", { name: /Simulator/ }).click();
  await expect(page).toHaveTitle("Simulator · Neurofluxion");
  await expect(page.getByRole("note")).toContainText("ran one random forward pass");
  await expect.poll(() => forwardResponses.length, { timeout: 15_000 }).toBeGreaterThan(0);
  await page.waitForLoadState("networkidle");

  // The demo bootstrap performs the forward request and stores its trace. Opening
  // Run reveals that existing trace; it does not itself issue another request.
  const payloads = await Promise.all(forwardResponses.map((response) => response.json()));
  const validPayloads = payloads.filter((payload) => Array.isArray(payload.steps) && payload.steps.length > 0);
  expect(validPayloads.length).toBeGreaterThan(0);

  await page.getByRole("tab", { name: "Run" }).click();

  // Bootstrap can issue more than one forward request. Do not assume the last
  // response is the one retained by the UI; match the entire rendered trace to
  // one captured backend payload instead.
  await expect.poll(async () => {
    const renderedSteps = await page.locator(".pass-step").evaluateAll((elements) =>
      elements.map((element) => ({
        title: element.querySelector(".pass-step-title")?.textContent?.trim() ?? "",
        output: element.querySelector(".pass-step-out")?.textContent?.trim() ?? "",
      }))
    );

    return validPayloads.some((payload) => {
      if (renderedSteps.length !== payload.steps.length) return false;
      return payload.steps.every((step, index) => {
        const output = step.output_values.slice(0, 6).map((value) => value.toFixed(3)).join(", ");
        return (
          renderedSteps[index].title === step.description &&
          renderedSteps[index].output === `Out: ${output}`
        );
      });
    });
  }, { timeout: 10_000 }).toBe(true);
});
