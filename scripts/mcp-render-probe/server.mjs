#!/usr/bin/env node
/**
 * kehai-render-probe — an MCP stdio server that lets a coding agent verify this
 * project's 3D rendering NUMERICALLY instead of by eyeballing screenshots.
 *
 * Born from the TAA/skirt investigation (see ~/.claude/plans/ui-sprightly-rain.md):
 * every measurement pattern in here was first a scratchpad script, and every
 * guard encodes a trap that actually burned a measurement once:
 *
 *   - SwiftShader gate ......... headless Chrome silently falls back to software
 *                                rendering; screenshots look right, numbers mean
 *                                nothing. Refuse unless the renderer is NVIDIA.
 *   - localStorage hygiene ..... __space() persists to localStorage. A leftover
 *                                cameraDrift:0 once cut the taa=0 baseline from
 *                                2.05% to 0.68% and made TAA look like a
 *                                regression. Clear vrmSpace1 before AND after;
 *                                snapshot/restore vrmLook8 (user-owned look).
 *   - taaSettle forced to 0 .... the determinism latch replays a HELD frame, so
 *                                a latched scene reports perfect stability — and
 *                                ignores live material edits entirely (a hide-
 *                                material sweep once measured "0.00% change"
 *                                because the latch was holding the image).
 *   - fitBody before setCamera . the framing solver rewrites the camera every
 *                                frame; an exact camera only sticks with it off.
 *   - sequential, never parallel background tabs get no requestAnimationFrame,
 *                                so two lockstepped pages deadlock. The debug
 *                                clock is deterministic; frame i of run A is
 *                                comparable with frame i of run B.
 *   - stepFrame, not setTime ... setTime freezes the clock but the loop keeps
 *                                redrawing while the harness round-trips, letting
 *                                temporal accumulation converge on a pose the
 *                                live app renders exactly once.
 *   - pages always closed ...... leaked pages exhaust WebGL contexts (8 leaked
 *                                pages once wedged the whole browser).
 *
 * Protocol: newline-delimited JSON-RPC 2.0 on stdin/stdout (the MCP stdio
 * transport). stdout carries ONLY protocol messages; all logging goes to stderr.
 * No SDK on purpose — @modelcontextprotocol/sdk is not vendored here, and the
 * five methods Claude Code needs are tiny.
 */
import { createRequire } from "node:module";
import readline from "node:readline";
import path from "node:path";
import fs from "node:fs";

const REPO = "/home/ai-ccore/Documents/DEV/KEHAI";
const CDP = process.env.KEHAI_CDP ?? "http://127.0.0.1:9222";
const APP = process.env.KEHAI_APP ?? "http://127.0.0.1:12394";
const OUT = path.join(REPO, "logs/render-probe");
const FLASH = path.join(REPO, "logs/flash");

const require = createRequire(path.join(REPO, "frontend-src/package.json"));
const { chromium } = require("playwright-core");
const { PNG } = require("pngjs");
const pixelmatch = require("pixelmatch").default ?? require("pixelmatch");

const log = (...a) => console.error("[render-probe]", ...a);

/* -------------------------------------------------------------------------- */
/* Browser plumbing                                                           */
/* -------------------------------------------------------------------------- */

async function preflight() {
  try {
    const r = await fetch(`${CDP}/json/version`, { signal: AbortSignal.timeout(3000) });
    if (!r.ok) throw new Error(String(r.status));
  } catch {
    throw new Error(
      `gpu-chrome is not running on ${CDP}. Start it:  bash scripts/gpu-chrome.sh`,
    );
  }
  try {
    const r = await fetch(`${APP}/`, { signal: AbortSignal.timeout(3000) });
    if (!r.ok) throw new Error(String(r.status));
  } catch {
    throw new Error(
      `KEHAI app server is not running on ${APP}. Start it:  bash start_server.sh`
      + " (from the repo root — NOT python run_server.py, which comes up mute)",
    );
  }
}

/** Serialize every tool call. Parallel pages deadlock (background-tab rAF),
 *  and the deterministic clock makes sequential runs directly comparable. */
let chain = Promise.resolve();
const mutex = (fn) => {
  const p = chain.then(fn, fn);
  chain = p.catch(() => {});
  return p;
};

const withTimeout = (p, ms, what) => Promise.race([
  p,
  new Promise((_, rej) => {
    const t = setTimeout(() => rej(new Error(`${what} timed out after ${ms / 1000}s`)), ms);
    if (typeof t.unref === "function") t.unref();
  }),
]);

async function withPage(fn) {
  await preflight();
  const browser = await chromium.connectOverCDP(CDP);
  const ctx = browser.contexts()[0] ?? (await browser.newContext());
  const page = await ctx.newPage();
  try {
    await page.setViewportSize({ width: 1080, height: 1920 });
    const renderer = await page.evaluate(() => {
      const c = document.createElement("canvas");
      const g = c.getContext("webgl2") || c.getContext("webgl");
      const d = g && g.getExtension("WEBGL_debug_renderer_info");
      return g && d ? g.getParameter(d.UNMASKED_RENDERER_WEBGL) : "unknown";
    });
    if (!/NVIDIA/i.test(renderer)) {
      throw new Error(
        `REFUSING: browser is not on the GPU (${renderer}) — measurements from a`
        + " software rasteriser are worthless. Run: bash scripts/gpu-chrome.sh",
      );
    }
    return await fn(page, renderer);
  } finally {
    await page.close().catch(() => {});
    await browser.close().catch(() => {}); // disconnects; does not kill gpu-chrome
  }
}

/** localStorage hygiene around a run. Space settings are cleared (they are
 *  render-budget knobs the tools set explicitly); the look is the character's
 *  APPEARANCE and user-owned, so it is snapshotted and restored byte-for-byte. */
/** EVERY look key, not just "vrmLook8". The viewer stores the look PER MODEL as
 *  `vrmLook8:${modelKey}` (vrm-viewer.tsx, LS_LOOK) and keeps the bare
 *  "vrmLook8" only as a legacy fallback — so snapshotting the bare key restored
 *  nothing and every `look` patch a tool passed was written to the user's real
 *  saved look PERMANENTLY. 2026-08-13: an `armOpaque:1, skirtOpaque:1` A/B run
 *  this way left the character's coat opaque for good, and a `hairColorMix:1,
 *  hairColor:0xff0000` run left her hair stored as bright red. Both were then
 *  mistaken for pre-existing bugs and chased for hours. Snapshot the whole
 *  namespace and put it back exactly, including deleting keys the run created. */
const LOOK_KEY_RE = /^vrmLook\d*(:|$)/;

async function storageOpen(page) {
  await page.goto(`${APP}/`, { waitUntil: "domcontentloaded" });
  const look = await page.evaluate((src) => {
    const re = new RegExp(src);
    const out = {};
    for (let i = 0; i < localStorage.length; i++) {
      const k = localStorage.key(i);
      if (re.test(k)) out[k] = localStorage.getItem(k);
    }
    return out;
  }, LOOK_KEY_RE.source);
  await page.evaluate(() => localStorage.removeItem("vrmSpace1"));
  return look;
}

async function storageClose(page, look) {
  try {
    await page.goto(`${APP}/`, { waitUntil: "domcontentloaded" });
    await page.evaluate(([saved, src]) => {
      localStorage.removeItem("vrmSpace1");
      const re = new RegExp(src);
      // drop any look key the run introduced, then restore the snapshot verbatim
      for (let i = localStorage.length - 1; i >= 0; i--) {
        const k = localStorage.key(i);
        if (re.test(k) && !(k in saved)) localStorage.removeItem(k);
      }
      Object.entries(saved).forEach(([k, v]) => localStorage.setItem(k, v));
    }, [look, LOOK_KEY_RE.source]);
  } catch (e) {
    log("storageClose failed (non-fatal):", e.message);
  }
}

async function gotoApp(page, { live = false, t = 6 } = {}) {
  const url = live ? `${APP}/` : `${APP}/?debug=1&t=${t}&seed=1`;
  await page.goto(url, { waitUntil: "domcontentloaded" });
  await page.waitForFunction(
    () => Boolean(window.__AI_DEBUG__ && window.__space),
    null, { timeout: 90_000 },
  );
  if (!live) await page.evaluate(() => window.__AI_DEBUG__.ready());
}

async function applyView(page, { preset, camera, target }) {
  if (camera) {
    // fitBody's framing solver rewrites the camera every frame; stand it down
    // or setCamera lasts exactly one frame.
    await page.evaluate(() => window.__space({ fitBody: false, cameraDrift: 0 }));
    const r = await page.evaluate(
      ([p, tg]) => window.__AI_DEBUG__.setCamera(p, tg),
      [camera, target ?? [0, 1.05, 0]],
    );
    if (r?.error) throw new Error(`setCamera: ${r.error}`);
  } else if (preset) {
    const r = await page.evaluate((n) => window.__AI_DEBUG__.setCameraPreset(n), preset);
    if (r?.error) throw new Error(`${r.error}; available: ${r.available?.join(", ")}`);
  }
}

async function applySettings(page, { space, look }, { forceNoSettle = true } = {}) {
  // The determinism latch replays a held frame: great for byte-stable test
  // captures, poison for measurements and for seeing live changes at all.
  const patch = { ...(forceNoSettle ? { taaSettle: 0 } : {}), ...(space ?? {}) };
  if (Object.keys(patch).length) await page.evaluate((s) => window.__space(s), patch);
  if (look && Object.keys(look).length) await page.evaluate((l) => window.__vrmLook(l), look);
  await page.waitForTimeout(1500);
}

function savePng(buffer, label) {
  fs.mkdirSync(OUT, { recursive: true });
  const stamp = new Date().toISOString().replace(/[:.]/g, "-");
  const name = `${stamp}-${String(label).replace(/[^A-Za-z0-9_.-]/g, "_")}.png`;
  const file = path.join(OUT, name);
  fs.writeFileSync(file, buffer);
  const png = PNG.sync.read(buffer);
  return { path: file, width: png.width, height: png.height };
}

/* -------------------------------------------------------------------------- */
/* Measurement primitives (ported from scripts/edge-regions.mjs)              */
/* -------------------------------------------------------------------------- */

const lum = (d, i) => 0.2126 * d[i] + 0.7152 * d[i + 1] + 0.0722 * d[i + 2];

/** Classify each pixel of a reference frame by local luminance gradient:
 *  outline (steep + dark), inner edge (steep + bright — the control group of
 *  the original flicker investigation), flat (no gradient). The reference is
 *  captured ONCE from the first row, so every row is scored on the same pixels. */
function classify(png, uiBottom) {
  const { width: w, height: h, data: d } = png;
  const cls = new Uint8Array(w * h);
  const counts = [0, 0, 0, 0];
  for (let y = 1; y < Math.min(h - 1, uiBottom); y += 1) {
    for (let x = 1; x < w - 1; x += 1) {
      const i = (y * w + x) * 4;
      const L = lum(d, i);
      const g = Math.max(
        Math.abs(lum(d, i + 4) - lum(d, i - 4)),
        Math.abs(lum(d, i + w * 4) - lum(d, i - w * 4)),
      );
      let c = 0;
      if (g >= 24) c = L <= 90 ? 3 : 2;
      else if (g <= 4) c = 1;
      cls[y * w + x] = c;
      counts[c] += 1;
    }
  }
  return { cls, counts, w, h };
}

function changedByClass(a, b, model) {
  const A = PNG.sync.read(a);
  const B = PNG.sync.read(b);
  const { cls, counts, w, h } = model;
  if (A.width !== w || A.height !== h) return null;
  const out = Buffer.alloc(w * h * 4);
  pixelmatch(A.data, B.data, out, w, h, { threshold: 0.1 });
  const hit = [0, 0, 0, 0];
  for (let p = 0; p < w * h; p += 1) {
    if (out[p * 4] > 0 && out[p * 4 + 1] === 0) hit[cls[p]] += 1;
  }
  return [1, 2, 3].map((c) => (counts[c] ? (hit[c] / counts[c]) * 100 : NaN));
}

const median = (xs) => {
  const s = xs.filter(Number.isFinite).sort((a, b) => a - b);
  return s.length ? s[Math.floor(s.length / 2)] : NaN;
};

/* -------------------------------------------------------------------------- */
/* Tools                                                                      */
/* -------------------------------------------------------------------------- */

const CAMERA_PROPS = {
  preset: {
    type: "string",
    enum: ["front", "final", "fullbody", "face", "hud"],
    description: "Named camera. Ignored when `camera` is given.",
  },
  camera: {
    type: "array", items: { type: "number" }, minItems: 3, maxItems: 3,
    description: "Exact camera position [x,y,z] metres. Forces fitBody:false.",
  },
  target: {
    type: "array", items: { type: "number" }, minItems: 3, maxItems: 3,
    description: "Look-at target, default [0,1.05,0]. Only with `camera`.",
  },
};

const TOOLS = [
  {
    name: "render_screenshot",
    description:
      "Screenshot the KEHAI scene on the real GPU. Deterministic by default"
      + " (?debug=1&t=…&seed=1: frozen clock, grain suppressed, reproducible to"
      + " the byte); live:true uses the real clock and idle motion. Returns the"
      + " saved PNG's absolute path.",
    inputSchema: {
      type: "object",
      required: ["label"],
      properties: {
        label: { type: "string", description: "Basename fragment for the PNG." },
        t: { type: "number", default: 6, description: "Deterministic clock time (s)." },
        live: { type: "boolean", default: false },
        ...CAMERA_PROPS,
        space: { type: "object", description: "__space() patch, e.g. {taa:0}." },
        look: { type: "object", description: "__vrmLook() patch (restored after)." },
      },
    },
    handler: (a) => withPage(async (page, renderer) => {
      const look = await storageOpen(page);
      try {
        await gotoApp(page, { live: a.live, t: a.t ?? 6 });
        await applyView(page, a);
        await applySettings(page, a);
        if (!a.live) await page.evaluate(() => window.__AI_DEBUG__.stepFrame());
        const shot = await page.locator("canvas").first().screenshot();
        return { ...savePng(shot, a.label), renderer, mode: a.live ? "live" : `debug t=${a.t ?? 6}` };
      } finally {
        await storageClose(page, look);
      }
    }),
  },
  {
    name: "render_flicker",
    description:
      "Temporal-stability metric: share of pixels changing between CONSECUTIVE"
      + " rendered frames (one stepFrame per capture), split into outline /"
      + " inner-edge / flat pixel classes from a reference-frame gradient"
      + " classification. Lower is steadier. Optionally sweeps one __space key"
      + " across values (fresh page load per value, same t for all).",
    inputSchema: {
      type: "object",
      properties: {
        t: { type: "number", default: 6 },
        frames: { type: "integer", default: 20, minimum: 2, maximum: 60 },
        warmup: { type: "integer", default: 16, description: "Motion frames before measuring." },
        static: { type: "boolean", default: false, description: "Hold the clock: measures the renderer's own noise floor." },
        sweepKey: { type: "string", description: "__space key to sweep (prefix 'look:' for __vrmLook)." },
        sweepValues: { type: "array", items: { type: "number" } },
        hold: { type: "object", description: "__space values pinned for every row. taaSettle:0 is always injected." },
        uiBottom: { type: "integer", default: 1790, description: "Rows below this are chat UI, excluded." },
        ...CAMERA_PROPS,
      },
    },
    handler: (a) => withPage(async (page) => {
      const look = await storageOpen(page);
      try {
        const values = a.sweepKey ? (a.sweepValues ?? [0, 1]) : [null];
        const isLook = (a.sweepKey ?? "").startsWith("look:");
        const key = isLook ? a.sweepKey.slice(5) : a.sweepKey;
        let model = null;
        const rows = [];
        for (const v of values) {
          await page.evaluate(() => localStorage.removeItem("vrmSpace1"));
          await gotoApp(page, { t: a.t ?? 6 });
          await applyView(page, a);
          await applySettings(page, { space: a.hold });
          if (v !== null) {
            await page.evaluate(
              ([k, val, lk]) => (lk ? window.__vrmLook({ [k]: val }) : window.__space({ [k]: val })),
              [key, v, isLook],
            );
            await page.waitForTimeout(1200);
          }
          if (!a.static) {
            for (let i = 0; i < (a.warmup ?? 16); i += 1) {
              await page.evaluate(() => window.__AI_DEBUG__.stepFrame());
            }
          }
          const canvas = page.locator("canvas").first();
          let prev = await canvas.screenshot();
          if (!model) model = classify(PNG.sync.read(prev), a.uiBottom ?? 1790);
          const per = [[], [], []];
          for (let i = 0; i < (a.frames ?? 20); i += 1) {
            if (!a.static) await page.evaluate(() => window.__AI_DEBUG__.stepFrame());
            else await page.waitForTimeout(120);
            const cur = await canvas.screenshot();
            const r = changedByClass(prev, cur, model);
            if (r) r.forEach((x, k) => per[k].push(x));
            prev = cur;
          }
          await page.evaluate(() => window.__AI_DEBUG__.endStep()).catch(() => {});
          rows.push({
            value: v,
            flatPct: +median(per[0]).toFixed(3),
            innerPct: +median(per[1]).toFixed(3),
            outlinePct: +median(per[2]).toFixed(3),
          });
        }
        const base = rows[0];
        const rel = (x, b) => (b ? `${x >= b ? "+" : ""}${(((x - b) / b) * 100).toFixed(0)}%` : "n/a");
        return {
          classes: { outlinePx: model.counts[3], innerPx: model.counts[2], flatPx: model.counts[1] },
          rows,
          relativeToFirst: rows.map((r) => ({
            value: r.value,
            outline: rel(r.outlinePct, base.outlinePct),
            inner: rel(r.innerPct, base.innerPct),
            flat: rel(r.flatPct, base.flatPct),
          })),
        };
      } finally {
        await storageClose(page, look);
      }
    }),
    budgetMs: 300_000,
  },
  {
    name: "render_compare",
    description:
      "A/B image comparison under two settings patches, same deterministic"
      + " instant and camera (sequential runs; the fixed clock makes frame i"
      + " of each run the same moment). Returns %differing pixels, mean/max"
      + " channel delta, the densest-diff region, and A/B/diff PNG paths.",
    inputSchema: {
      type: "object",
      required: ["settingsA", "settingsB"],
      properties: {
        settingsA: { type: "object", properties: { space: { type: "object" }, look: { type: "object" } } },
        settingsB: { type: "object", properties: { space: { type: "object" }, look: { type: "object" } } },
        t: { type: "number", default: 6 },
        warmup: { type: "integer", default: 16 },
        label: { type: "string", default: "compare" },
        ...CAMERA_PROPS,
      },
    },
    handler: (a) => withPage(async (page) => {
      const look = await storageOpen(page);
      try {
        const capture = async (settings) => {
          await page.evaluate(() => localStorage.removeItem("vrmSpace1"));
          await gotoApp(page, { t: a.t ?? 6 });
          await applyView(page, a);
          await applySettings(page, settings ?? {});
          for (let i = 0; i < (a.warmup ?? 16); i += 1) {
            await page.evaluate(() => window.__AI_DEBUG__.stepFrame());
          }
          const shot = await page.locator("canvas").first().screenshot();
          await page.evaluate(() => window.__AI_DEBUG__.endStep()).catch(() => {});
          return shot;
        };
        const bufA = await capture(a.settingsA);
        const bufB = await capture(a.settingsB);
        const A = PNG.sync.read(bufA);
        const B = PNG.sync.read(bufB);
        if (A.width !== B.width || A.height !== B.height) {
          throw new Error(`size mismatch ${A.width}x${A.height} vs ${B.width}x${B.height}`);
        }
        const w = A.width; const h = A.height;
        const diff = new PNG({ width: w, height: h });
        const n = pixelmatch(A.data, B.data, diff.data, w, h, { threshold: 0.1 });
        let sum = 0; let max = 0;
        const CELL = 128;
        const grid = new Map();
        for (let y = 0; y < h; y += 1) {
          for (let x = 0; x < w; x += 1) {
            const i = (y * w + x) * 4;
            const d = Math.max(
              Math.abs(A.data[i] - B.data[i]),
              Math.abs(A.data[i + 1] - B.data[i + 1]),
              Math.abs(A.data[i + 2] - B.data[i + 2]),
            );
            sum += d;
            if (d > max) max = d;
            if (d > 14) {
              const k = `${Math.floor(x / CELL)},${Math.floor(y / CELL)}`;
              grid.set(k, (grid.get(k) ?? 0) + 1);
            }
          }
        }
        let densest = { x: 0, y: 0, count: 0 };
        for (const [k, count] of grid) {
          if (count > densest.count) {
            const [gx, gy] = k.split(",").map(Number);
            densest = { x: gx * CELL, y: gy * CELL, count };
          }
        }
        const pa = savePng(bufA, `${a.label ?? "compare"}-A`);
        const pb = savePng(bufB, `${a.label ?? "compare"}-B`);
        const pd = savePng(PNG.sync.write(diff), `${a.label ?? "compare"}-diff`);
        return {
          pctDiff: +((n / (w * h)) * 100).toFixed(3),
          meanAbsDelta: +(sum / (w * h)).toFixed(3),
          maxAbsDelta: max,
          densestRegion: { ...densest, w: CELL, h: CELL },
          paths: { a: pa.path, b: pb.path, diff: pd.path },
        };
      } finally {
        await storageClose(page, look);
      }
    }),
    budgetMs: 300_000,
  },
  {
    name: "render_framestats",
    description:
      "Frame cost with the harness's hands off the page: lets the clock run"
      + " free for N seconds, then reads the app's own rolling frame-time"
      + " window. Median (one GC pause must not decide the verdict) plus p95.",
    inputSchema: {
      type: "object",
      properties: {
        seconds: { type: "number", default: 4, minimum: 1, maximum: 30 },
        t: { type: "number", default: 6 },
        space: { type: "object" },
        ...CAMERA_PROPS,
      },
    },
    handler: (a) => withPage(async (page, renderer) => {
      const look = await storageOpen(page);
      try {
        await gotoApp(page, { t: a.t ?? 6 });
        await applyView(page, a);
        await applySettings(page, a);
        const backing = await page.evaluate(() => {
          const c = document.querySelector("canvas");
          return `${c.width}x${c.height}`;
        });
        await page.evaluate(() => {
          window.__AI_DEBUG__.play();
          window.__AI_DEBUG__.resetFrameStats();
        });
        await page.waitForTimeout((a.seconds ?? 4) * 1000);
        const fsr = await page.evaluate(() => window.__AI_DEBUG__.getFrameStats());
        return { ...fsr, backing, renderer };
      } finally {
        await storageClose(page, look);
      }
    }),
  },
  {
    name: "flash_box",
    description:
      "The artefact black-box drop-box (logs/flash/): frames uploaded by the"
      + " in-app flash recorder (hotkey 9, or its one-frame-pop auto-detector)."
      + " list returns names/sizes/mtimes; clear deletes all captures.",
    inputSchema: {
      type: "object",
      required: ["action"],
      properties: { action: { type: "string", enum: ["list", "clear"] } },
    },
    handler: async (a) => {
      fs.mkdirSync(FLASH, { recursive: true });
      const files = fs.readdirSync(FLASH).filter((f) => f.endsWith(".png"));
      if (a.action === "clear") {
        files.forEach((f) => fs.unlinkSync(path.join(FLASH, f)));
        return { cleared: files.length };
      }
      return {
        dir: FLASH,
        files: files
          .map((f) => {
            const st = fs.statSync(path.join(FLASH, f));
            return { name: f, path: path.join(FLASH, f), bytes: st.size, mtime: st.mtime.toISOString() };
          })
          .sort((x, y) => x.mtime.localeCompare(y.mtime)),
      };
    },
  },
];

/* -------------------------------------------------------------------------- */
/* JSON-RPC / MCP plumbing                                                    */
/* -------------------------------------------------------------------------- */

const KNOWN_PROTOCOLS = ["2025-06-18", "2025-03-26", "2024-11-05"];
const send = (msg) => process.stdout.write(`${JSON.stringify(msg)}\n`);

async function handle(msg) {
  const { id, method, params } = msg;
  const isNotification = id === undefined || id === null;
  try {
    if (method === "initialize") {
      const v = params?.protocolVersion;
      send({
        jsonrpc: "2.0",
        id,
        result: {
          protocolVersion: KNOWN_PROTOCOLS.includes(v) ? v : KNOWN_PROTOCOLS[0],
          capabilities: { tools: { listChanged: false } },
          serverInfo: { name: "kehai-render-probe", version: "1.0.0" },
        },
      });
      return;
    }
    if (isNotification) return; // notifications/initialized, notifications/cancelled, ...
    if (method === "ping") {
      send({ jsonrpc: "2.0", id, result: {} });
      return;
    }
    if (method === "tools/list") {
      send({
        jsonrpc: "2.0",
        id,
        result: {
          tools: TOOLS.map(({ name, description, inputSchema }) => ({ name, description, inputSchema })),
        },
      });
      return;
    }
    if (method === "tools/call") {
      const tool = TOOLS.find((t) => t.name === params?.name);
      if (!tool) {
        send({ jsonrpc: "2.0", id, error: { code: -32602, message: `unknown tool: ${params?.name}` } });
        return;
      }
      inflight += 1;
      try {
        const result = await withTimeout(
          mutex(() => tool.handler(params?.arguments ?? {})),
          tool.budgetMs ?? 120_000,
          tool.name,
        );
        send({
          jsonrpc: "2.0",
          id,
          result: { content: [{ type: "text", text: JSON.stringify(result, null, 1) }], isError: false },
        });
      } catch (e) {
        // Execution failures are tool RESULTS, not protocol errors — the model
        // needs to read the remediation text ("start gpu-chrome with ...").
        send({
          jsonrpc: "2.0",
          id,
          result: { content: [{ type: "text", text: String(e?.message ?? e) }], isError: true },
        });
      } finally {
        inflight -= 1;
        maybeExit();
      }
      return;
    }
    send({ jsonrpc: "2.0", id, error: { code: -32601, message: `method not found: ${method}` } });
  } catch (e) {
    log("handler crash:", e);
    if (!isNotification) {
      send({ jsonrpc: "2.0", id, error: { code: -32603, message: String(e?.message ?? e) } });
    }
  }
}

const rl = readline.createInterface({ input: process.stdin, terminal: false });
rl.on("line", (line) => {
  const s = line.trim();
  if (!s) return;
  let msg;
  try {
    msg = JSON.parse(s);
  } catch {
    send({ jsonrpc: "2.0", id: null, error: { code: -32700, message: "parse error" } });
    return;
  }
  void handle(msg);
});
/** In-flight tools/call count and the shutdown latch. Exiting must wait for the
 *  RESPONSE to be written, not just for the tool promise to settle — draining
 *  the mutex chain alone raced process.exit against the awaiting handler's own
 *  send() on the same microtask turn, and the last response of a piped test run
 *  was lost about half the time. The counter is decremented AFTER send(). */
let inflight = 0;
let closing = false;
const maybeExit = () => {
  if (closing && inflight === 0) process.exit(0);
};

rl.on("close", () => {
  closing = true;
  maybeExit();
});
log(`ready (CDP ${CDP}, APP ${APP})`);
