#!/usr/bin/env bash
# Render host: the KEHAI page running in Chrome on this box's RTX 3090, on a
# virtual display that the stream captures. Viewers open /render-view
# (see src/kehai/render_stream.py).
#
#   scripts/render-host.sh start|stop|status
#
# The pieces are the measured working set, not a guess:
#   Xvfb :57 at RENDER_SIZE        a display to capture without root or a monitor
#                                  (the box's real X is someone else's session).
#   --use-angle=vulkan             WebGL on the 3090 even though Xvfb itself is
#                                  software: ANGLE renders through Vulkan and only
#                                  presents into the X framebuffer. GL/EGL on Xvfb
#                                  falls back to llvmpipe; Vulkan under HEADLESS
#                                  crashes the GPU process (exit 139).
#   --autoplay-policy=...          the tab plays TTS with no user gesture.
#   DRI_PRIME=pci-0000_1a_00_0     pins Vulkan to GPU0 through Mesa's device_select
#                                  layer. Left alone it picks GPU1, which already
#                                  carries GSV/gesture/ActionPlan and is ~95% full —
#                                  rendering there would compete with TTS.
# Capture + NVENC run in the server (render_stream.py) against $RENDER_DISPLAY.
set -u
PORT="${RENDER_CDP_PORT:-9222}"
DISP="${RENDER_DISPLAY:-:57}"
SIZE="${RENDER_SIZE:-2160x3840}"
PROFILE="${RENDER_PROFILE:-$HOME/.cache/kehai-render-chrome}"
URL="${RENDER_URL:-http://127.0.0.1:12394/?render=1}"

# Match on the process NAME, not just the command line: `pgrep -f` with the
# port string also matches any shell whose command line mentions it — including
# the one that called `stop` — and kills it.
chrome_pids() {
  ps -eo pid=,comm=,args= | awk -v p="--remote-debugging-port=$PORT" '$2 ~ /^chrome/ && index($0, p) { print $1 }'
}
xvfb_pids() {
  ps -eo pid=,comm=,args= | awk -v d="$DISP" '$2 == "Xvfb" && $4 == d { print $1 }'
}

case "${1:-status}" in
  start)
    if [ -n "$(chrome_pids)" ]; then echo "already running on :$PORT"; exit 0; fi
    mkdir -p "$PROFILE"
    if [ -z "$(xvfb_pids)" ]; then
      setsid --fork Xvfb "$DISP" -screen 0 "${SIZE}x24" -nolisten tcp </dev/null >"$PROFILE/xvfb.log" 2>&1
      sleep 1
    fi
    W="${SIZE%x*}"; H="${SIZE#*x}"
    DISPLAY="$DISP" DRI_PRIME="${RENDER_DRI_PRIME:-pci-0000_1a_00_0}" setsid --fork google-chrome --user-data-dir="$PROFILE" \
      --remote-debugging-port="$PORT" --no-first-run --ignore-gpu-blocklist \
      --use-angle=vulkan --enable-features=Vulkan,VulkanFromANGLE,DefaultANGLEVulkan \
      --autoplay-policy=no-user-gesture-required \
      --window-position=0,0 --window-size="$W,$H" --kiosk "$URL" </dev/null >"$PROFILE/chrome.log" 2>&1
    for _ in $(seq 20); do
      curl -sf -o /dev/null "http://127.0.0.1:$PORT/json/version" && { echo "render host up: $SIZE on $DISP, CDP :$PORT -> $URL"; exit 0; }
      sleep 1
    done
    echo "render host did NOT come up — see $PROFILE/chrome.log"; exit 1
    ;;
  stop)
    c="$(chrome_pids)"; x="$(xvfb_pids)"
    [ -z "$c$x" ] && { echo "not running"; exit 0; }
    [ -n "$c" ] && kill $c
    [ -n "$x" ] && kill $x
    echo "stopped"
    ;;
  status)
    if [ -n "$(chrome_pids)" ]; then
      echo "running: CDP :$PORT, display $DISP ($( [ -n "$(xvfb_pids)" ] && echo up || echo DOWN ))"
    else
      echo "NOT running"
    fi
    ;;
  *) echo "usage: $0 {start|stop|status}"; exit 2 ;;
esac
