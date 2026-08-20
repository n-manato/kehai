/**
 * Measure a render change objectively, on the real GPU.
 *
 *   bash scripts/gpu-chrome.sh
 *   node scripts/render-metrics.mjs --sweep superSample=1,1.25,1.5,2
 *   node scripts/render-metrics.mjs --sweep msaaSamples=4,8 --frames 40
 *
 * WHAT IT REPORTS
 *   edge%   share of pixels whose luminance changes between CONSECUTIVE frames.
 *           This is the "crawling outline" number: a line that re-lands on a
 *           different pixel each frame drives it up. Lower is steadier.
 *   ms      real frame time (median), from __AI_DEBUG__.getFrameStats().
 *
 * WHY IT IS TRUSTWORTHY, WHERE AN AD-HOC rAF LOOP IS NOT
 *   It drives the scene through __AI_DEBUG__'s deterministic clock: ?debug=1
 *   freezes the render, suppresses film grain and pins the pose, then setTime()
 *   advances in exact 1/60 steps. Without that, three things move the number on
 *   their own — grain alone was measured at 9.4% frame-to-frame, idle-clip
 *   roulette at 42%, and unsettled spring hair at ~3%. Those swamp the effect
 *   being measured.
 *
 *   It also attaches to the browser started by scripts/gpu-chrome.sh rather than
 *   launching its own, because headless Chrome without ANGLE flags renders on
 *   SwiftShader without saying so.
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
const FRAMES = Number(arg("frames", "30"));
const T0 = Number(arg("t", "6"));
const CAM = arg("cam", "front");
// --sweep superSample=1,1.5        → __space({superSample: v})
// --sweep look:outlineWidth=1.2,0  → __vrmLook({outlineWidth: v})
// --hold  superSample=1            → pinned for every row (e.g. measure the raw
//                                    artifact with the supersample mitigation off)
// --hold  superSample=1,taaSettle=0  several at once, comma separated. Needed
//   because more than one setting can invalidate a measurement on its own:
//   taaSettle latches the temporal history on a still scene, which makes a
//   --static run report a perfect 0.000% no matter what the accumulation does.
const SWEEP = arg("sweep", "superSample=1,1.5");
const HOLD = arg("hold", "");
const STATIC = process.argv.includes("--static");

const [RAWKEY, LIST] = SWEEP.split("=");
const IS_LOOK = RAWKEY.startsWith("look:");
const KEY = IS_LOOK ? RAWKEY.slice(5) : RAWKEY;
const VALUES = LIST.split(",").map(Number);

/** Share of pixels that changed, ignoring near-identical ones. */
function changedPct(a, b) {
  const A = PNG.sync.read(a);
  const B = PNG.sync.read(b);
  if (A.width !== B.width || A.height !== B.height) return NaN;
  // threshold 0.1 is Playwright's own default ballpark: below it, two captures of
  // a genuinely static frame already differ from dithering alone.
  const n = pixelmatch(A.data, B.data, null, A.width, A.height, { threshold: 0.1 });
  return (n / (A.width * A.height)) * 100;
}

const median = (xs) => {
  const s = [...xs].sort((x, y) => x - y);
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

const canvas = page.locator("canvas").first();
const rows = [];
let original = null; // the value found before the sweep touched anything

for (const v of VALUES) {
  // RELOAD per setting, rather than carrying on from where the last one stopped.
  // setTime is forward-only by design, so continuing would measure each setting
  // over a DIFFERENT stretch of the animation — and the spring hair is still
  // moving at these timescales, so that difference would land in the result and
  // be read as an effect of the setting. Same t for everyone, or the comparison
  // is not one. ?debug=1 is load-time only anyway: it is what suppresses grain
  // and pins the pose, and freeze() by hand does neither.
  await page.goto(`${APP}/?debug=1&t=${T0}&seed=1`, { waitUntil: "domcontentloaded" });
  await page.waitForFunction(() => Boolean(window.__AI_DEBUG__), null, { timeout: 90_000 });
  await page.evaluate(() => window.__AI_DEBUG__.ready());
  await page.evaluate((c) => window.__AI_DEBUG__.setCameraPreset(c), CAM);

  // Only LOOK is captured for restore. `__vrmLook()` returns the whole LOOK
  // object, and its values are the character's APPEARANCE — leaving a sweep's
  // last value behind would silently redecorate her. `__space()` returns a
  // describe() report rather than the config, so there is nothing to read back
  // there; its keys are render-budget settings, which the next run sets
  // explicitly anyway.
  if (IS_LOOK && original === null) {
    original = await page.evaluate((k) => window.__vrmLook()[k], KEY);
    console.log(`(restore note) ${KEY} was ${original}`);
  }

  if (HOLD) {
    const held = HOLD.split(",").map((p) => p.split("="));
    await page.evaluate(
      (pairs) => window.__space(Object.fromEntries(pairs.map(([k, val]) => [k, Number(val)]))),
      held,
    );
  }
  // BOTH of these persist to localStorage, so whatever the last row set is what
  // the next page load starts from. Every row therefore sets its value
  // explicitly rather than assuming a clean slate, and the original is restored
  // at the end — otherwise a sweep silently redecorates the character.
  await page.evaluate(
    ([k, val, isLook]) => (isLook ? window.__vrmLook({ [k]: val }) : window.__space({ [k]: val })),
    [KEY, v, IS_LOOK],
  );
  await page.waitForTimeout(1500);

  const backing = await page.evaluate(() => {
    const c = document.querySelector("canvas");
    return `${c.width}x${c.height}`;
  });

  // PASS 1 — stability. Step the deterministic clock, photograph every frame.
  //
  // --static holds the clock instead of advancing it. That is the CONTROL: with
  // nothing moving, any difference between frames is the renderer failing to
  // reproduce itself, so it calibrates the noise floor. Without it there is no
  // way to tell "the outline is crawling" from "her hair is legitimately
  // moving", and both land in the same number.
  const diffs = [];
  let prev = await canvas.screenshot();
  for (let i = 1; i <= FRAMES; i += 1) {
    if (!STATIC) {
      await page.evaluate((t) => window.__AI_DEBUG__.setTime(t), T0 + i / 60);
    }
    const cur = await canvas.screenshot();
    diffs.push(changedPct(prev, cur));
    prev = cur;
  }

  // PASS 2 — cost, measured SEPARATELY and deliberately.
  //
  // Frame time cannot be read during pass 1. There, every frame is bracketed by
  // a CDP setTime round-trip and a canvas screenshot, so what gets timed is the
  // harness, not the render. Measured that way the numbers came out backwards —
  // 2x the pixels reported FASTER than 1x — which is the tell that the render
  // was not what was being timed. So: let the clock run, stop touching the page,
  // and time frames the renderer is producing on its own.
  await page.evaluate(() => {
    window.__AI_DEBUG__.play();
    window.__AI_DEBUG__.resetFrameStats();
  });
  await page.waitForTimeout(4000);
  const fs = await page.evaluate(() => window.__AI_DEBUG__.getFrameStats());
  rows.push({ v, backing, edge: median(diffs), ms: fs.msMedian, fps: fs.fps });
  console.log(
    `${KEY}=${String(v).padEnd(5)} backing ${backing.padEnd(11)} ` +
    `edge ${median(diffs).toFixed(3).padStart(7)}%  ` +
    `${String(fs.msMedian).padStart(6)}ms  ${String(fs.fps).padStart(5)}fps`,
  );
}

const base = rows[0];
console.log("\n--- 相対（先頭を基準） ---");
for (const r of rows) {
  const de = ((r.edge - base.edge) / base.edge) * 100;
  const dm = ((r.ms - base.ms) / base.ms) * 100;
  console.log(
    `${KEY}=${String(r.v).padEnd(5)} ちらつき ${(de >= 0 ? "+" : "") + de.toFixed(0)}%   ` +
    `コスト ${(dm >= 0 ? "+" : "") + dm.toFixed(0)}%`,
  );
}

if (IS_LOOK && original !== null && original !== undefined) {
  await page.evaluate(([k, v]) => window.__vrmLook({ [k]: v }), [KEY, original]);
  console.log(`\n${KEY} restored to ${original}`);
}

await page.close();
await browser.close();
