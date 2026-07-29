import { test, type Page } from "@playwright/test";
import { PNG } from "pngjs";
import * as fs from "fs";
import * as path from "path";

/**
 * Ad-hoc capture, not a regression test — `npx playwright test shot.spec.ts`.
 * Writes a settled PNG somewhere readable so the render can be looked at directly
 * instead of described. Driven by env vars so the same file serves every look:
 *
 *   SHOT_OUT=... SHOT_CAM=hud SHOT_T=6 SHOT_EVAL='__vrmLook({subInt:0})'
 *
 * Shares openFrozen/waitUntilStill semantics with scene.spec.ts; see that file's
 * header for why both steps are needed.
 */

const OUT = process.env.SHOT_OUT
  || path.join(process.env.TEMP || ".", "claude", "shot.png");
const NOISE_PCT = 0.3;

function diffPct(a: Buffer, b: Buffer): number {
  const pa = PNG.sync.read(a);
  const pb = PNG.sync.read(b);
  let n = 0;
  for (let i = 0; i < pa.data.length; i += 4) {
    if (pa.data[i] !== pb.data[i] || pa.data[i + 1] !== pb.data[i + 1]
      || pa.data[i + 2] !== pb.data[i + 2]) n += 1;
  }
  return (n / (pa.width * pa.height)) * 100;
}

async function waitUntilStill(page: Page, timeoutMs = 25_000) {
  const c = page.locator("canvas").first();
  const deadline = Date.now() + timeoutMs;
  let prev = await c.screenshot();
  let last = 100;
  while (Date.now() < deadline) {
    await page.waitForTimeout(500);
    const cur = await c.screenshot();
    last = diffPct(prev, cur);
    if (last < NOISE_PCT) return last;
    prev = cur;
  }
  console.warn(`[shot] never settled, last diff ${last.toFixed(3)}%`);
  return last;
}

test("capture", async ({ page }) => {
  test.skip(!process.env.SHOT, "set SHOT=1 to capture");
  const t = Number(process.env.SHOT_T || 6);
  await page.setViewportSize({ width: 1080, height: 1920 });
  await page.goto(`/?debug=1&t=${t}&seed=1`);
  await page.waitForFunction(() => Boolean((window as any).__AI_DEBUG__), null, { timeout: 60_000 });
  await page.evaluate(() => (window as any).__AI_DEBUG__.ready());
  await page.evaluate((cam) => {
    (window as any).__AI_DEBUG__.setCameraPreset(cam);
    (window as any).__AI_DEBUG__.setHudPreset("secretary");
  }, process.env.SHOT_CAM || "front");
  if (process.env.SHOT_EVAL) {
    // Treated as an EXPRESSION and its value printed: in-page console.log does not
    // reach node, so returning is the only way to read a measurement back out.
    // eslint-disable-next-line no-new-func
    const r = await page.evaluate((src) => new Function(`return (${src})`)(), process.env.SHOT_EVAL);
    console.log("EVAL", JSON.stringify(r));
  }
  await waitUntilStill(page);
  fs.mkdirSync(path.dirname(OUT), { recursive: true });
  fs.writeFileSync(OUT, await page.locator("canvas").first().screenshot());
  console.log("WROTE", OUT);
});
