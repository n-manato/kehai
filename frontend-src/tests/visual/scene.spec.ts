import { test, expect, type Page } from "@playwright/test";
import { PNG } from "pngjs";

/**
 * Visual regression for the VRM + holographic-console scene.
 *
 * The contract these tests rely on is `window.__AI_DEBUG__` (see
 * src/renderer/src/utils/debug/ai-debug.ts). Without it a screenshot of this app
 * is not reproducible: the character blinks, breathes and shifts its gaze, the
 * idle clip advances, and every `uTime` shader rolls forward, so two captures of
 * IDENTICAL code differ. `?debug=1&t=<sec>&seed=<n>` replays from zero in exact
 * 1/60 steps and freezes, which is reproducible across reloads and machines.
 *
 * Freezing the clock is necessary but NOT sufficient, and that surprised us:
 * measured on this scene, consecutive captures still differed on 8.6% of pixels
 * immediately after the clock froze, then 0.39%, then settled to ~0.05% about two
 * seconds later. The clock reaching `t` says nothing about textures finishing
 * upload, materials being upgraded or the shadow map converging. So readiness here
 * is defined the only way that actually holds — successive frames agreeing — and
 * `waitUntilStill` is what every capture goes through.
 *
 * Even settled, a WebGL capture is not byte-reproducible: the GPU/compositor path
 * alone accounts for ~0.05% of pixels. Every comparison below is a tolerance.
 *
 * Deliberately captures the CANVAS, not the page: the chat UI carries live
 * conversation state that has nothing to do with the render.
 */

const VIEWPORT = { width: 1080, height: 1920 }; // the panel's 2160x3840, halved

/** Ceiling for "the scene did not change", from the measured settled noise floor
 *  (~0.02-0.1%) with headroom. A real animation moves far more than this. */
const NOISE_PCT = 0.3;

async function openFrozen(page: Page, opts: { t?: number; seed?: number } = {}) {
  const t = opts.t ?? 2;
  const seed = opts.seed ?? 1;
  await page.setViewportSize(VIEWPORT);
  await page.goto(`/?debug=1&t=${t}&seed=${seed}`);

  // The viewer mounts asynchronously (VRM download + parse), so the handle does
  // not exist at navigation time.
  await page.waitForFunction(() => Boolean((window as any).__AI_DEBUG__), null, { timeout: 60_000 });
  // Resolves only once the 1/60 replay has reached t and the clock is frozen.
  await page.evaluate(() => (window as any).__AI_DEBUG__.ready());

  const state = await page.evaluate(() => (window as any).__AI_DEBUG__.state());
  // If the clock is not actually frozen the screenshots below are noise, and a
  // baseline written from one would be worse than no test at all.
  expect(state.frozen, "scene must be frozen before capture").toBe(true);
  expect(state.seeking).toBe(false);
  expect(state.time).toBeCloseTo(t, 1);
  await waitUntilStill(page);
  return state;
}

function canvas(page: Page) {
  return page.locator("canvas").first();
}

/**
 * Block until the render stops changing. This is the real "safe to screenshot"
 * signal — see the file header for the measurements that forced it. Throws rather
 * than returning a maybe-settled frame, because a baseline captured mid-settle is
 * silently wrong forever after.
 */
async function waitUntilStill(page: Page, timeoutMs = 20_000) {
  const c = canvas(page);
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
  throw new Error(`render never settled: last frame-to-frame diff ${last.toFixed(3)}%`);
}

/** Percentage of pixels that differ at all between two PNG buffers. */
function diffPct(a: Buffer, b: Buffer): number {
  const pa = PNG.sync.read(a);
  const pb = PNG.sync.read(b);
  let n = 0;
  for (let i = 0; i < pa.data.length; i += 4) {
    if (
      pa.data[i] !== pb.data[i]
      || pa.data[i + 1] !== pb.data[i + 1]
      || pa.data[i + 2] !== pb.data[i + 2]
    ) n += 1;
  }
  return (n / (pa.width * pa.height)) * 100;
}

test.describe("VRM scene", () => {
  test("debug clock holds the scene still", async ({ page }) => {
    await openFrozen(page, { t: 2 });
    // The proof Phase 1 works: 1.5 s of real time must not advance the render.
    // Everything else in this file is built on that holding.
    const a = await canvas(page).screenshot();
    await page.waitForTimeout(1500);
    const b = await canvas(page).screenshot();
    const pct = diffPct(a, b);
    expect(pct, `frozen scene drifted on ${pct.toFixed(3)}% of pixels`).toBeLessThan(NOISE_PCT);
  });

  test("clock resumes on play()", async ({ page }) => {
    await openFrozen(page, { t: 2 });
    const a = await canvas(page).screenshot();
    await page.evaluate(() => (window as any).__AI_DEBUG__.play());
    await page.waitForTimeout(1200);
    const b = await canvas(page).screenshot();
    // Guards the inverse failure: a freeze that can never be undone would make
    // the "still" test above pass for the wrong reason.
    expect(diffPct(a, b), "play() must let the scene move again").toBeGreaterThan(NOISE_PCT);
  });

  test("renderer reports a live scene", async ({ page }) => {
    await openFrozen(page, { t: 2 });
    const info = await page.evaluate(() => (window as any).__AI_DEBUG__.getRendererInfo());
    // Guards the case where the canvas is present but nothing is drawn — a blank
    // screenshot would otherwise pass a pixel comparison forever.
    expect(info.calls, "draw calls").toBeGreaterThan(0);
    expect(info.triangles, "triangles").toBeGreaterThan(0);
    const scene = await page.evaluate(() => (window as any).__AI_DEBUG__.getSceneInfo());
    expect(scene.visibleMeshes).toBeGreaterThan(0);
  });

  test("no shader or three.js errors on load", async ({ page }) => {
    const errors: string[] = [];
    page.on("console", (m) => m.type() === "error" && errors.push(m.text()));
    page.on("pageerror", (e) => errors.push(String(e)));
    await openFrozen(page, { t: 1 });
    // Headless has no microphone and no backend conversation, so VAD and the
    // Live2D model fetch fail by definition here. Shader/WebGL errors do not.
    const ignorable = /websocket|ws:\/\/|net::ERR|favicon|Failed to start VAD|model3\.json|status of 404/i;
    const real = errors.filter((e) => !ignorable.test(e));
    expect(real, real.join("\n")).toHaveLength(0);
  });

  test("secretary full view", async ({ page }) => {
    // Longer than the behavioural tests on purpose. The twintails and sleeves are
    // damped spring/rigid-body sims: at t=2 they are still settling, and where
    // they settle depends on the path, so consecutive runs differed on ~3% of
    // pixels — all of it hair. A damped system under a static pose converges to a
    // single equilibrium, so the fix is to seek far enough for it to get there.
    await openFrozen(page, { t: 6 });
    await page.evaluate(() => {
      (window as any).__AI_DEBUG__.setCameraPreset("front");
      (window as any).__AI_DEBUG__.setHudPreset("secretary");
    });
    await waitUntilStill(page);
    await expect(canvas(page)).toHaveScreenshot("secretary-full.png");
  });

  test("hud console framing", async ({ page }) => {
    await openFrozen(page, { t: 6 }); // hair settling — see "secretary full view"
    await page.evaluate(() => {
      (window as any).__AI_DEBUG__.setCameraPreset("hud");
      (window as any).__AI_DEBUG__.setHudPreset("secretary");
    });
    await waitUntilStill(page);
    await expect(canvas(page)).toHaveScreenshot("hud-console.png");
  });
});
