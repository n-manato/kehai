/**
 * Puncture counter — the instrument that finally caught the "cyan blade pokes
 * out between the thighs" artefact (the necktie flap, Bullet group 7, popping
 * through the hip sphere; see mmd-hair-physics.ts).
 *
 *   bash scripts/gpu-chrome.sh          # real-GPU headless chrome on :9222
 *   node scripts/puncture-check.mjs <outDir> [seconds]
 *
 * LIVE mode (real clock, real idle motion) at the under-skirt peek camera, an
 * in-page rAF detector counting "was navy-dark last frame, is white/cyan now"
 * pixels at 60fps — Playwright screenshots (~8fps) cannot catch a 1-2 frame
 * event, which is why every earlier attempt missed it. Spikes save a rolling
 * before/during/after full-res triplet into <outDir>.
 *
 * The camera has to be RE-PINNED every 400ms with fitBody:false and the drift
 * knobs zeroed: live-mode websocket events (model/config) re-run the framing
 * solver, which otherwise steals the camera back mid-run — that one cost an
 * afternoon. Baseline before the mask fix: ~5 events / 75s of plain idling;
 * pass line after: 0 events across three 75s runs.
 */
import { createRequire } from "node:module";
import fs from "node:fs";
const require = createRequire("/home/ai-ccore/Documents/DEV/KEHAI/frontend-src/package.json");
const { chromium } = require("playwright-core");
const SP = process.argv[2];
const SECONDS = Number(process.argv[3] ?? 75);
const browser = await chromium.connectOverCDP("http://127.0.0.1:9222");
const ctx = browser.contexts()[0] ?? (await browser.newContext());
for (const p of ctx.pages().slice(1)) { try { await p.close(); } catch { /* ignore */ } }
const page = await ctx.newPage();
try {
  await page.setViewportSize({ width: 1080, height: 900 });
  await page.goto("http://127.0.0.1:12394/", { waitUntil: "domcontentloaded" });
  await page.evaluate(() => localStorage.removeItem("vrmSpace1"));
  await page.goto("http://127.0.0.1:12394/", { waitUntil: "domcontentloaded" });
  await page.waitForFunction(
    () => Boolean(window.__AI_DEBUG__ && window.__vrmObj && window.__vrmObj() && window.__space),
    null, { timeout: 90_000 },
  );
  await page.evaluate(() => window.__space({ fitBody: false }));
  await page.evaluate(() => window.__AI_DEBUG__.setCamera([0, 0.35, 0.12], [0, 1.7, -0.05]));
  await page.waitForTimeout(1500);
  await page.evaluate(() => {
    const src = document.querySelector("canvas");
    const W = 192; const H = 192;
    const small = document.createElement("canvas"); small.width = W; small.height = H;
    const sctx = small.getContext("2d", { willReadFrequently: true });
    const FW = 540; const FH = 450;
    const mk = () => { const c2 = document.createElement("canvas"); c2.width = FW; c2.height = FH; return c2; };
    const ring = [mk(), mk(), mk()];
    let ri = 0;
    const st = { frames: 0, base: [], events: [], prev: null };
    window.__lp = st;
    const L = (d, i) => 0.2126 * d[i] + 0.7152 * d[i + 1] + 0.0722 * d[i + 2];
    const step = () => {
      requestAnimationFrame(step);
      st.frames += 1;
      ri = (ri + 1) % 3;
      sctx.drawImage(src, 0, 0, W, H);
      ring[ri].getContext("2d").drawImage(src, 0, 0, FW, FH);
      const d = sctx.getImageData(0, 0, W, H).data;
      if (st.prev) {
        let n = 0;
        let navyPrev = 0;
        for (let p = 0; p < W * H; p += 1) {
          const i = p * 4;
          const wasNavy = L(st.prev, i) < 85 && st.prev[i + 2] >= st.prev[i];
          if (wasNavy) navyPrev += 1;
          const isBright = L(d, i) > 150;
          if (wasNavy && isBright) n += 1;
        }
        // A frame that was MOSTLY navy is the camera inside swinging cloth;
        // its reveal is not a puncture. Only sparse dark->bright counts.
        if (navyPrev > W * H * 0.4) n = 0;
        st.base.push(n);
        if (st.base.length > 240) st.base.shift();
        const med = [...st.base].sort((a, b) => a - b)[Math.floor(st.base.length / 2)];
        if (st.base.length > 90 && n > Math.max(med * 4, med + 25) && st.events.length < 5
          && (!st.events.length || st.frames - st.events[st.events.length - 1].f > 40)) {
          st.events.push({
            f: st.frames, n, med,
            images: [ring[(ri + 1) % 3].toDataURL("image/png"), ring[(ri + 2) % 3].toDataURL("image/png"), ring[ri].toDataURL("image/png")],
          });
        }
      }
      st.prev = new Uint8ClampedArray(d);
    };
    requestAnimationFrame(step);
  });
  // Something in live mode (websocket model/config events) re-frames the
  // camera; pin it by re-asserting every 400ms for the whole run.
  const pin = setInterval(() => {
    page.evaluate(() => {
      window.__space({ fitBody: false, cameraDrift: 0, cameraDriftEnabled: false });
      window.__AI_DEBUG__.setCamera([0, 0.35, 0.12], [0, 1.7, -0.05]);
    }).catch(() => {});
  }, 400);
  await page.waitForTimeout(SECONDS * 1000);
  clearInterval(pin);
  console.log("final camera:", JSON.stringify((await page.evaluate(() => window.__AI_DEBUG__.getSceneInfo())).camera.position));
  const report = await page.evaluate(() => ({
    frames: window.__lp.frames,
    events: window.__lp.events.map((e) => ({ f: e.f, n: e.n, med: e.med })),
  }));
  console.log(JSON.stringify(report));
  const evs = await page.evaluate(() => window.__lp.events);
  evs.forEach((e, i) => e.images.forEach((u, j) => {
    fs.writeFileSync(`${SP}/lp-${i}-${j}.png`, Buffer.from(u.split(",")[1], "base64"));
  }));
  console.log(`${evs.length} events saved`);
} finally {
  await page.evaluate(() => localStorage.removeItem("vrmSpace1")).catch(() => {});
  await page.close(); await browser.close();
}
