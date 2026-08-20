#!/usr/bin/env bash
# Launch a headless Chrome that renders on the REAL GPU, with a CDP port open, so
# both chrome-devtools-mcp and Playwright can attach to the SAME browser.
#
#   bash scripts/gpu-chrome.sh          # start (idempotent)
#   bash scripts/gpu-chrome.sh stop
#   bash scripts/gpu-chrome.sh status
#
# WHY THIS EXISTS
# ---------------
# chrome-devtools-mcp launches its own Chrome and gives no way to pass ANGLE
# flags. Headless Chrome without them silently falls back to SwiftShader — it
# renders, it screenshots, nothing errors, and every measurement taken from it is
# worthless for judging a GPU render. A whole afternoon of "measurements" of the
# outline flicker was taken that way before anyone noticed. So the browser is
# started HERE with verified flags, and the MCP server attaches over
# --browserUrl instead of starting its own.
#
# Measured on this box (2026-08-10), same page, same flags:
#   --use-angle=vulkan                → ANGLE (NVIDIA, Vulkan 1.4.303, RTX 3090)  MAX_SAMPLES=8
#   --use-gl=angle --use-angle=gl-egl → ANGLE (NVIDIA, RTX 3090, OpenGL ES 3.2)   MAX_SAMPLES=8
#   --use-gl=egl alone                → no WebGL context at all
# SwiftShader reports MAX_SAMPLES=4, which is also why the MSAA cap looked like 4.
set -uo pipefail

PORT="${GPU_CHROME_PORT:-9222}"
PROFILE="${GPU_CHROME_PROFILE:-/tmp/kehai-gpu-chrome}"
CHROME="${CHROME_BIN:-/usr/bin/google-chrome}"
# GPU0 is normally full (vLLM holds ~23.7 of 24 GB); GPU1 has room. Chrome only
# needs a few hundred MB, but on a full device it fails into SwiftShader rather
# than erroring — exactly the silent failure this script exists to prevent.
export CUDA_VISIBLE_DEVICES="${GPU_CHROME_DEVICE:-1}"

is_up() { curl -s -m 2 "http://127.0.0.1:${PORT}/json/version" >/dev/null 2>&1; }

case "${1:-start}" in
  stop)
    pkill -f "remote-debugging-port=${PORT}" 2>/dev/null && echo "stopped" || echo "not running"
    exit 0
    ;;
  status)
    if is_up; then
      curl -s "http://127.0.0.1:${PORT}/json/version" | head -c 400; echo
    else
      echo "not running on :${PORT}"; exit 1
    fi
    exit 0
    ;;
esac

if is_up; then
  echo "already running on :${PORT}"
  exit 0
fi

mkdir -p "$PROFILE"
setsid --fork nohup "$CHROME" \
  --headless=new \
  --remote-debugging-port="${PORT}" \
  --remote-allow-origins='*' \
  --user-data-dir="$PROFILE" \
  --no-first-run --no-default-browser-check \
  --no-sandbox --disable-gpu-sandbox \
  --use-angle=vulkan \
  --ignore-gpu-blocklist \
  --enable-gpu-rasterization \
  --hide-scrollbars \
  --autoplay-policy=no-user-gesture-required \
  about:blank \
  >/tmp/kehai-gpu-chrome.log 2>&1 < /dev/null

for _ in $(seq 40); do is_up && break; sleep 0.25; done
if ! is_up; then
  echo "FAILED: Chrome did not open :${PORT} — see /tmp/kehai-gpu-chrome.log" >&2
  exit 1
fi

# THE GATE. Ask the browser what it is actually rendering with, and refuse to
# hand back a browser that is not on the GPU. A warning would be ignored; this is
# the one failure mode that quietly invalidates everything downstream.
SRCDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../frontend-src" && pwd)"
# playwright-core is already installed and `connectOverCDP` attaches to an
# EXISTING browser, so this needs no downloaded browser binary — which is also
# why the whole loop can run without `npx playwright install`.
RENDERER=$(cd "$SRCDIR" && node -e '
(async () => {
  const { chromium } = require("playwright-core");
  const b = await chromium.connectOverCDP("http://127.0.0.1:" + process.argv[1]);
  const ctx = b.contexts()[0] || (await b.newContext());
  const p = await ctx.newPage();
  const out = await p.evaluate(() => {
    const c = document.createElement("canvas");
    const g = c.getContext("webgl2") || c.getContext("webgl");
    if (!g) return "NO-WEBGL";
    const d = g.getExtension("WEBGL_debug_renderer_info");
    return (d ? g.getParameter(d.UNMASKED_RENDERER_WEBGL) : "NO-EXT")
         + " | MAX_SAMPLES=" + g.getParameter(g.MAX_SAMPLES);
  });
  console.log(out);
  await p.close();
  await b.close();
})().catch((e) => { console.log("PROBE-FAILED: " + e.message); });
' "$PORT" 2>/dev/null | tail -1)

echo "renderer: ${RENDERER:-（取得失敗）}"
case "$RENDERER" in
  *NVIDIA*) echo "OK — real GPU. CDP: http://127.0.0.1:${PORT}" ;;
  *)
    echo "" >&2
    echo "FAILED: not rendering on the GPU (got: ${RENDERER:-nothing})." >&2
    echo "Anything measured through this browser would be SwiftShader and meaningless." >&2
    echo "Check: nvidia-smi (VRAM on device ${CUDA_VISIBLE_DEVICES}), /tmp/kehai-gpu-chrome.log" >&2
    exit 1
    ;;
esac
