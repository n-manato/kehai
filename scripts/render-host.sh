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
#
# GPU X SERVER (2026-11-05, "そもそも 60 FPS にしたい"): on Xvfb every 4K frame is read back
# from the GPU into the software framebuffer — SwapBuffers 29 ms/frame, so the page could not
# pass 30 fps whatever it drew (a bare full-screen shader: 15-30 fps at 2160x3840). An Xorg
# on GPU0 with no display device (/etc/X11/kehai-render-x.conf, see scripts/pcb-render-x/)
# lets Chrome present on the GPU: the same shader 60 fps, and x11grab + NVENC still 60 fps.
# ANGLE on Vulkan there (--use-angle=vulkan ONLY): it holds 60 fps through a reply, while
# native GL (--use-angle=gl) also made 60 idle but fell to 1-5 fps for ~5 s whenever TTS +
# GestureLSM loaded GPU0 (eglMakeCurrent/SwapBuffers waiting up to 0.9 s behind CUDA).
# Adding VulkanFromANGLE/DefaultANGLEVulkan (the Xvfb set) sends presentation back through
# the readback path (83 ms/frame). Picked automatically when that config is installed; RENDER_X=xvfb = the old
# Xvfb :57 + Vulkan. The display used is written to .render-display for start_server.sh.
set -u
PORT="${RENDER_CDP_PORT:-9222}"
XCONF="kehai-render-x.conf"
# Default gpu when the config is installed (2026-11-05 14:00: 60 fps through replies, with
# the TTS on piecewise graphs, CUDA MPS for GPU0's services, half-res AO, the viewer-link
# guards in render_stream.py). RENDER_X=xvfb = the 30 fps Xvfb path.
if [ -z "${RENDER_X:-}" ]; then
  if [ -f "/etc/X11/$XCONF" ]; then RENDER_X=gpu; else RENDER_X=xvfb; fi
fi
if [ "$RENDER_X" = gpu ]; then DISP="${RENDER_DISPLAY:-:58}"; else DISP="${RENDER_DISPLAY:-:57}"; fi
STATE="$(cd "$(dirname "$0")/.." && pwd)/.render-display"
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
# The GPU X server runs as root through Xorg.wrap, but its REAL uid is ours, so we may
# signal it. Matched on name + display like Xvfb.
xorg_pids() {
  ps -eo pid=,comm=,args= | awk -v d="$DISP" '$2 == "Xorg" && $4 == d { print $1 }'
}
display_pids() { if [ "$RENDER_X" = gpu ]; then xorg_pids; else xvfb_pids; fi; }

case "${1:-status}" in
  start)
    if [ -n "$(chrome_pids)" ]; then echo "already running on :$PORT"; exit 0; fi
    mkdir -p "$PROFILE"
    if [ "$RENDER_X" = gpu ]; then
      if [ -z "$(xorg_pids)" ]; then
        # -sharevts/-novtswitch on vt8: the box's monitor is GDM's :0 on vt2 (on this same
        # GPU) and must never be switched away from; no input devices (see the config).
        setsid --fork Xorg "$DISP" -config "$XCONF" -sharevts -novtswitch -nolisten tcp -noreset vt8 \
          </dev/null >"$PROFILE/xorg.log" 2>&1
        for _ in $(seq 30); do [ -S "/tmp/.X11-unix/X${DISP#:}" ] && break; sleep 0.5; done
      fi
      # DrDc: the compositor's GPU work on its own thread, not behind WebGL's on GPU-main;
      # asyncCommandQueue: ANGLE submits to the GPU from a worker. Under a full CUDA load on
      # the same GPU these took a 4K page from ~33-46 to ~49-52 fps (2026-11-05).
      GPU_FLAGS=(--use-angle=vulkan --enable-features=EnableDrDc --enable-angle-features=asyncCommandQueue)
    else
      if [ -z "$(xvfb_pids)" ]; then
        setsid --fork Xvfb "$DISP" -screen 0 "${SIZE}x24" -nolisten tcp </dev/null >"$PROFILE/xvfb.log" 2>&1
        sleep 1
      fi
      GPU_FLAGS=(--use-angle=vulkan --enable-features=Vulkan,VulkanFromANGLE,DefaultANGLEVulkan)
    fi
    echo "$DISP" > "$STATE"
    W="${SIZE%x*}"; H="${SIZE#*x}"
    DISPLAY="$DISP" DRI_PRIME="${RENDER_DRI_PRIME:-pci-0000_1a_00_0}" setsid --fork google-chrome --user-data-dir="$PROFILE" \
      --remote-debugging-port="$PORT" --no-first-run --ignore-gpu-blocklist \
      "${GPU_FLAGS[@]}" \
      --autoplay-policy=no-user-gesture-required \
      --window-position=0,0 --window-size="$W,$H" --kiosk "$URL" </dev/null >"$PROFILE/chrome.log" 2>&1
    for _ in $(seq 20); do
      curl -sf -o /dev/null "http://127.0.0.1:$PORT/json/version" && { echo "render host up: $SIZE on $DISP ($RENDER_X), CDP :$PORT -> $URL"; exit 0; }
      sleep 1
    done
    echo "render host did NOT come up — see $PROFILE/chrome.log"; exit 1
    ;;
  stop)
    c="$(chrome_pids)"; x="$(display_pids)"
    [ -z "$c$x" ] && { echo "not running"; exit 0; }
    [ -n "$c" ] && kill $c
    [ -n "$x" ] && kill $x
    # Wait until both are really gone: a `start` right after found the dying X server,
    # skipped starting a new one, and Chrome then failed with "Missing X server".
    for _ in $(seq 40); do [ -z "$(chrome_pids)$(display_pids)" ] && break; sleep 0.25; done
    echo "stopped"
    ;;
  status)
    if [ -n "$(chrome_pids)" ]; then
      echo "running: CDP :$PORT, display $DISP $RENDER_X ($( [ -n "$(display_pids)" ] && echo up || echo DOWN ))"
    else
      echo "NOT running"
    fi
    ;;
  *) echo "usage: $0 {start|stop|status}"; exit 2 ;;
esac
