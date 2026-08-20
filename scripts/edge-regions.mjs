/**
 * The controlled experiment that identified the flicker, re-runnable.
 *
 *   bash scripts/gpu-chrome.sh
 *   node scripts/edge-regions.mjs --sweep taa=0,1 --hold superSample=1,taaSettle=0
 *
 * WHY THIS EXISTS SEPARATELY FROM render-metrics.mjs
 *   render-metrics reports ONE number for the whole canvas, and the canvas is
 *   mostly empty backdrop. The measurement that actually diagnosed the problem
 *   split the frame up first:
 *
 *       outline pixels  5.4%     internal edges  8.6%     flat  0.2%
 *
 *   and the conclusion — that the outline was a victim rather than the cause, so
 *   the fix had to be scene-wide — is only visible in that split. Averaged over
 *   the frame it dilutes roughly 40x into noise, which is why a whole-frame
 *   reading can move by a fifth while the thing being complained about does not
 *   move at all. Success is defined against the INTERNAL EDGE number.
 *
 * HOW THE CLASSES ARE DECIDED
 *   Not by hand-drawn rectangles: those would have to be redrawn every time the
 *   framing changes, and picking them by eye after seeing the result is how a
 *   measurement stops being one. Each pixel is classified by the local luminance
 *   gradient of a REFERENCE frame, and the reference is captured ONCE, from the
 *   first setting in the sweep. Every setting is then scored against exactly the
 *   same set of pixels, so the comparison cannot drift with the thing it is
 *   measuring.
 *
 *     edge     steep local gradient — a boundary of some kind
 *     outline  edge AND dark: the ink line is drawn near-black (0.03,0.02,0.05)
 *     inner    edge AND not dark: hair strands, folds, features — the CONTROL
 *              GROUP, and the number that has to come down
 *     flat     almost no gradient: the noise floor
 *
 *   "Changed" is pixelmatch at threshold 0.1, the same definition
 *   render-metrics.mjs uses, so the two scripts' numbers mean the same thing.
 */
import { createRequire } from "node:module";
import process from "node:process";

const require = createRequire("/home/ai-ccore/Documents/DEV/KEHAI/frontend-src/package.json");
const { chromium } = require("playwright-core");
const { PNG } = require("pngjs");
const pixelmatch = require("pixelmatch").default ?? require("pixelmatch");

const arg = (name, dflt) => {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 && process.argv[i + 1] ? process.argv[i + 1] : dflt;
};

const CDP = arg("cdp", "http://127.0.0.1:9222");
const APP = arg("app", "http://127.0.0.1:12394");
const FRAMES = Number(arg("frames", "20"));
const T0 = Number(arg("t", "6"));
const CAM = arg("cam", "front");
const SWEEP = arg("sweep", "taa=0,1");
const HOLD = arg("hold", "");
const STATIC = process.argv.includes("--static");
/** Rows below this are the chat footer, not the render. */
const UI_BOTTOM = Number(arg("uiBottom", "1790"));
/** Frames of MOTION stepped before the first capture, so the temporal history is
 *  in its playback steady state rather than cold or converged-on-a-held-pose.
 *  16 = two full jitter cycles. */
const WARMUP = Number(arg("warmup", "16"));

const [KEY, LIST] = SWEEP.split("=");
const VALUES = LIST.split(",").map(Number);

const lum = (d, i) => 0.2126 * d[i] + 0.7152 * d[i + 1] + 0.0722 * d[i + 2];

/**
 * Per-pixel class ids from a reference frame. Central differences rather than a
 * full Sobel: the question is only "is there a step here", and a 3x3 kernel
 * smears a one-pixel line — which is exactly the feature of interest — across its
 * neighbours.
 */
function classify(png) {
  const { width: w, height: h, data: d } = png;
  const cls = new Uint8Array(w * h); // 0 none, 1 flat, 2 inner edge, 3 outline
  const counts = [0, 0, 0, 0];
  for (let y = 1; y < h - 1; y += 1) {
    if (y >= UI_BOTTOM) break;
    for (let x = 1; x < w - 1; x += 1) {
      const i = (y * w + x) * 4;
      const L = lum(d, i);
      const gx = Math.abs(lum(d, i + 4) - lum(d, i - 4));
      const gy = Math.abs(lum(d, i + w * 4) - lum(d, i - w * 4));
      const g = Math.max(gx, gy);
      let c = 0;
      if (g >= 24) c = L <= 90 ? 3 : 2;
      else if (g <= 4) c = 1;
      cls[y * w + x] = c;
      counts[c] += 1;
    }
  }
  return { cls, counts, w, h };
}

/** Share of each class whose pixels changed between two frames. */
function changedByClass(a, b, model) {
  const A = PNG.sync.read(a);
  const B = PNG.sync.read(b);
  const { cls, counts, w, h } = model;
  if (A.width !== w || A.height !== h) return null;
  const out = Buffer.alloc(w * h * 4);
  pixelmatch(A.data, B.data, out, w, h, { threshold: 0.1 });
  const hit = [0, 0, 0, 0];
  for (let p = 0; p < w * h; p += 1) {
    // pixelmatch paints differing pixels red and leaves matching ones with alpha
    // low; the red channel is the reliable flag.
    if (out[p * 4] > 0 && out[p * 4 + 1] === 0) hit[cls[p]] += 1;
  }
  return [1, 2, 3].map((c) => (counts[c] ? (hit[c] / counts[c]) * 100 : NaN));
}

const median = (xs) => {
  const s = xs.filter((x) => Number.isFinite(x)).sort((x, y) => x - y);
  return s.length ? s[Math.floor(s.length / 2)] : NaN;
};

const browser = await chromium.connectOverCDP(CDP);
const ctx = browser.contexts()[0] ?? (await browser.newContext());
const page = await ctx.newPage();
await page.setViewportSize({ width: 1080, height: 1920 });

const renderer = await page.evaluate(() => {
  const c = document.createElement("canvas");
  const g = c.getContext("webgl2") || c.getContext("webgl");
  const d = g && g.getExtension("WEBGL_debug_renderer_info");
  return g && d ? g.getParameter(d.UNMASKED_RENDERER_WEBGL) : "unknown";
});
if (!/NVIDIA/i.test(renderer)) {
  console.error(`REFUSING: not on the GPU (${renderer}). Run scripts/gpu-chrome.sh first.`);
  process.exit(1);
}
console.log(`renderer : ${renderer}`);
console.log(`mode     : ${STATIC
  ? "static (noise floor / convergence), clock held"
  : `moving, one render per frame, ${WARMUP}-frame warm-up, `
    + `t=${(T0 + WARMUP / 60).toFixed(3)}..${(T0 + (WARMUP + FRAMES) / 60).toFixed(3)}`}`);

const canvas = page.locator("canvas").first();
let model = null;
const rows = [];

for (const v of VALUES) {
  // START FROM THE SHIPPED DEFAULTS. __space() persists to localStorage, so
  // whatever any previous run or hand-tuning session left behind is load-bearing
  // here — a stale `cameraDrift: 0` silently removed most of the scene's motion
  // and cut the taa=0 BASELINE from 2.05% to 0.68%, which made TAA look like a
  // regression. Clearing is the only way a baseline means anything.
  await page.goto(`${APP}/`, { waitUntil: "domcontentloaded" });
  await page.evaluate(() => localStorage.removeItem("vrmSpace1"));
  // Reload per setting, same t for everyone — see the note in render-metrics.mjs.
  await page.goto(`${APP}/?debug=1&t=${T0}&seed=1`, { waitUntil: "domcontentloaded" });
  await page.waitForFunction(() => Boolean(window.__AI_DEBUG__), null, { timeout: 90_000 });
  await page.evaluate(() => window.__AI_DEBUG__.ready());
  await page.evaluate((c) => window.__AI_DEBUG__.setCameraPreset(c), CAM);
  if (HOLD) {
    const held = HOLD.split(",").map((p) => p.split("="));
    await page.evaluate(
      (pairs) => window.__space(Object.fromEntries(pairs.map(([k, val]) => [k, Number(val)]))),
      held,
    );
  }
  await page.evaluate(([k, val]) => window.__space({ [k]: val }), [KEY, v]);
  await page.waitForTimeout(1500);

  // Hand the loop over to single-stepping BEFORE the first capture, so the
  // reference frame and every frame after it are produced the same way.
  //
  // WARM UP UNDER MOTION FIRST. The wait above let the temporal history converge
  // on ONE held pose, which no live frame ever gets; starting cold instead would
  // measure the first-frame transient. Neither is the steady state. Stepping a
  // couple of jitter cycles of real motion puts the history exactly where it sits
  // during playback. Costs nothing when taa is off — there is no state to warm.
  if (!STATIC) {
    for (let i = 0; i < WARMUP; i += 1) {
      // eslint-disable-next-line no-await-in-loop
      await page.evaluate(() => window.__AI_DEBUG__.stepFrame());
    }
  }
  let prev = await canvas.screenshot();
  if (!model) {
    model = classify(PNG.sync.read(prev));
    const [, flat, inner, outline] = model.counts;
    console.log(
      `classes  : outline ${outline}px  inner-edge ${inner}px  flat ${flat}px  `
      + `(reference: ${KEY}=${v})`,
    );
  }
  const per = [[], [], []];
  for (let i = 1; i <= FRAMES; i += 1) {
    if (!STATIC) {
      // ONE rendered frame per capture. setTime() would freeze the clock and let
      // the loop keep redrawing the frozen pose while this script round-trips over
      // CDP, which lets the temporal history converge on a pose the live app
      // renders exactly once — see stepFrame() in utils/debug/ai-debug.ts.
      // eslint-disable-next-line no-await-in-loop
      await page.evaluate(() => window.__AI_DEBUG__.stepFrame());
    }
    // eslint-disable-next-line no-await-in-loop
    const cur = await canvas.screenshot();
    const r = changedByClass(prev, cur, model);
    if (r) r.forEach((x, k) => per[k].push(x));
    prev = cur;
  }
  const [flat, inner, outline] = per.map(median);
  rows.push({ v, flat, inner, outline });
  console.log(
    `${KEY}=${String(v).padEnd(5)} outline ${outline.toFixed(3).padStart(7)}%  `
    + `inner-edge ${inner.toFixed(3).padStart(7)}%  flat ${flat.toFixed(3).padStart(7)}%`,
  );
}

const base = rows[0];
console.log("\n--- 相対（先頭を基準） ---");
for (const r of rows) {
  const d = (a, b) => (b ? `${((a - b) / b) * 100 >= 0 ? "+" : ""}${(((a - b) / b) * 100).toFixed(0)}%` : "n/a");
  console.log(
    `${KEY}=${String(r.v).padEnd(5)} アウトライン ${d(r.outline, base.outline).padStart(6)}   `
    + `内部エッジ ${d(r.inner, base.inner).padStart(6)}   平坦 ${d(r.flat, base.flat).padStart(6)}`,
  );
}

await page.close();
await browser.close();
