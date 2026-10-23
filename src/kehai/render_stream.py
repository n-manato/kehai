"""Server-side rendering: the app runs in a GPU Chrome on this box, a viewer watches.

The weak device (a laptop now, the PROTO later) opens `/render-view`. That page
does no 3D at all: it shows frames, plays sound, and sends back input and its
microphone. The actual app is a tab in a headless Chrome on this box's RTX 3090
(`scripts/render-host.sh`), opened with `?render=1`.

Everything rides one WebSocket per side through this server, never WebRTC:

    viewer  <-> /render-stream/view    <-> hub <-> CDP (pixels out, input in)
                                             <-> /render-stream/source (sound)

Why not WebRTC: the only path into this box is a TCP port forward (and later a
Cloudflare Tunnel), and neither carries UDP, so WebRTC would need a TURN relay.
A WebSocket goes through both as-is.

Pixels come one of two ways:
  x11  (the 4K path) the render host runs on a virtual display (Xvfb, see
       scripts/render-host.sh); FFmpeg grabs it and encodes H.264 on NVENC.
       Measured at 2160x3840: 60 fps capture+encode.
  cdp  (fallback, no display) CDP screencast JPEG. Fine at 1080p (60 fps,
       ~30 ms) but ~30 fps / ~90 ms at 4K, because the JPEGs are made on the CPU.
Input is injected through CDP either way, so it is real input (trusted events,
IME text included).

Binary frames, first byte is the kind:
    1 audio  source -> viewer   Int16 LE mono 48 kHz PCM
    2 video  hub    -> viewer   float64 LE server timestamp + JPEG
    3 mic    viewer -> source   Int16 LE mono 48 kHz PCM
    4 h264   hub    -> viewer   float64 LE server timestamp + u8 keyframe + Annex B AU
    5 cam    viewer -> source   JPEG frame of the viewer's camera (head tracking)
Text frames are JSON control messages (mode, size, input, ping, and the mic panel:
mic / mic-set / mic-reset / mic-monitor / mic-level, see vad/mic_tuning.py).
"""

import asyncio
import base64
import itertools
import json
import os
import struct
import time
import urllib.request
from pathlib import Path

import websockets
from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from loguru import logger

from .vad import mic_tuning

CDP_HTTP = os.environ.get("RENDER_CDP", "http://127.0.0.1:9222")
VIEW_HTML = Path(__file__).with_name("render_view.html")
KIND_AUDIO, KIND_VIDEO, KIND_MIC, KIND_H264, KIND_CAM = 1, 2, 3, 4, 5

# The 4K path. CDP screencast tops out near 30 fps / ~90 ms at 2160x3840 (CPU
# JPEG), so when the render host runs on a virtual display the server grabs that
# display and encodes on the 3090's NVENC instead. FFmpeg 8 specifically: 4.2's
# single-threaded pipeline serialises grab and encode and caps at ~55 fps at 4K;
# 8.x overlaps them and holds 60.
X_DISPLAY = os.environ.get("RENDER_DISPLAY", ":57")
X_SIZE = os.environ.get("RENDER_SIZE", "2160x3840")
FFMPEG = os.environ.get("RENDER_FFMPEG", str(Path.home() / "opt/ffmpeg8/ffmpeg"))
NVENC_GPU = os.environ.get("RENDER_NVENC_GPU", "0")
# Send a portrait display as a landscape stream (rotated 90° clockwise; the viewer
# rotates it back). Hardware H.264 decoders that take 3840x2160 routinely refuse
# 2160x3840 — measured: Chrome on the Windows viewer rejected it outright and its
# software fallback managed ~10 fps. The CPU transpose costs nothing visible:
# capture+transpose+NVENC still holds 60 fps at 4K.
# Off by default (user's call): set RENDER_ROTATE=auto to rotate portrait output.
ROTATE = os.environ.get("RENDER_ROTATE", "0")
AUD = b"\x00\x00\x00\x01\x09"


def _rotated() -> bool:
    w, h = (int(v) for v in X_SIZE.split("x"))
    return ROTATE == "1" or (ROTATE == "auto" and h > w)


def _stream_size() -> str:
    w, h = X_SIZE.split("x")
    return f"{h}x{w}" if _rotated() else X_SIZE


def _x11_available() -> bool:
    return Path(f"/tmp/.X11-unix/X{X_DISPLAY.lstrip(':')}").exists() and Path(FFMPEG).exists()


def _is_key(au: bytes) -> bool:
    """True if the access unit holds an IDR slice (NAL type 5)."""
    i = au.find(b"\x00\x00\x01")
    while i != -1 and i + 3 < len(au):
        if au[i + 3] & 0x1F == 5:
            return True
        i = au.find(b"\x00\x00\x01", i + 3)
    return False


class _NvencCapture:
    """x11grab -> h264_nvenc, split into access units on the AUD the encoder emits."""

    def __init__(self, on_au):
        self.on_au = on_au
        self.proc: asyncio.subprocess.Process | None = None
        self.reader: asyncio.Task | None = None

    async def start(self, mbps: int):
        w, h = X_SIZE.split("x")
        rotate = ["-vf", "transpose=clock"] if _rotated() else []
        args = [
            FFMPEG, "-hide_banner", "-loglevel", "error", "-nostdin", "-filter_threads", "4",
            "-f", "x11grab", "-framerate", "60", "-video_size", f"{w}x{h}", "-draw_mouse", "0",
            "-i", f"{X_DISPLAY}.0", *rotate,
            "-pix_fmt", "bgr0", "-c:v", "h264_nvenc", "-gpu", NVENC_GPU,
            "-preset", "p1", "-tune", "ull", "-bf", "0", "-g", "60", "-rc", "cbr",
            "-b:v", f"{mbps}M", "-maxrate", f"{mbps}M", "-bufsize", f"{max(1, mbps // 15)}M",
            "-aud", "1", "-bsf:v", "dump_extra=freq=keyframe", "-f", "h264", "-",
        ]
        self.proc = await asyncio.create_subprocess_exec(
            *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        self.reader = asyncio.create_task(self._read())
        logger.info(
            f"[render] NVENC capture {w}x{h}@60 -> {_stream_size()}{' (rotated)' if rotate else ''} "
            f"{mbps} Mbps from {X_DISPLAY} (GPU{NVENC_GPU})"
        )

    async def _read(self):
        buf = b""
        try:
            while True:
                chunk = await self.proc.stdout.read(1 << 20)
                if not chunk:
                    break
                buf += chunk
                # Emit every complete AU: the bytes from one AUD up to the next.
                while True:
                    nxt = buf.find(AUD, 1)
                    if nxt == -1:
                        break
                    au, buf = buf[:nxt], buf[nxt:]
                    self.on_au(au)
        finally:
            err = (await self.proc.stderr.read()).decode(errors="replace").strip() if self.proc else ""
            if err:
                logger.warning(f"[render] ffmpeg: {err[-400:]}")

    async def stop(self):
        if self.proc and self.proc.returncode is None:
            self.proc.terminate()
            try:
                await asyncio.wait_for(self.proc.wait(), 3)
            except asyncio.TimeoutError:
                self.proc.kill()
        if self.reader:
            self.reader.cancel()
        self.proc = None


class _Cdp:
    """Minimal CDP client for the render tab: fire-and-forget commands + frame events."""

    def __init__(self, on_frame):
        self.on_frame = on_frame
        self.ws = None
        self.ids = itertools.count(1)
        self.reader: asyncio.Task | None = None

    async def connect(self) -> bool:
        targets = await asyncio.to_thread(
            lambda: json.load(urllib.request.urlopen(f"{CDP_HTTP}/json", timeout=3))
        )
        page = next(
            (t for t in targets if t.get("type") == "page" and "render=1" in t.get("url", "")),
            None,
        )
        if not page:
            return False
        self.ws = await websockets.connect(page["webSocketDebuggerUrl"], max_size=None)
        self.reader = asyncio.create_task(self._read())
        return True

    async def send(self, method: str, params: dict | None = None):
        if self.ws:
            await self.ws.send(json.dumps({"id": next(self.ids), "method": method, "params": params or {}}))

    async def _read(self):
        try:
            async for raw in self.ws:
                msg = json.loads(raw)
                if msg.get("method") == "Page.screencastFrame":
                    p = msg["params"]
                    # Ack first: Chrome withholds the next frame until it is acked.
                    await self.send("Page.screencastFrameAck", {"sessionId": p["sessionId"]})
                    self.on_frame(base64.b64decode(p["data"]))
                elif "error" in msg:
                    logger.warning(f"[render] CDP error: {msg['error']}")
        except websockets.ConnectionClosed:
            pass

    async def close(self):
        if self.reader:
            self.reader.cancel()
        if self.ws:
            await self.ws.close()
        self.ws = None


class _Hub:
    def __init__(self):
        self.viewer: WebSocket | None = None
        # Video (pump), audio (source_ws), pong and the opening messages all write to
        # the viewer from different tasks. Two concurrent sends trip the websockets
        # drain assertion ("assert waiter is None or waiter.cancelled()"), which killed
        # view_ws and dropped the viewer — 60 times in one server log. Serialize them.
        self.viewer_lock = asyncio.Lock()
        self.source: WebSocket | None = None
        self.cdp: _Cdp | None = None
        self.latest: bytes | None = None
        self.fresh = asyncio.Event()
        self.size = (1080, 1920)
        self.quality = 70
        self.frames = 0
        self.mode = "cdp"
        self.nvenc: _NvencCapture | None = None
        self.aus: list[bytes] = []
        self.need_key = True
        self.sent = 0
        self.dropped = 0
        self.viewer_stat: dict = {}
        self.mic_q_ms = None
        self.injecting = False
        self.audio_samples = 0  # PCM samples forwarded to the viewer (should be 48k/s)

    async def send_viewer(self, ws: WebSocket, data: bytes | str) -> None:
        """Every write to the viewer goes through here (see viewer_lock)."""
        async with self.viewer_lock:
            if isinstance(data, str):
                await ws.send_text(data)
            else:
                await ws.send_bytes(data)

    def on_frame(self, jpeg: bytes):
        # Keep only the newest: a slow link drops frames instead of building lag.
        self.latest = struct.pack("<Bd", KIND_VIDEO, time.time()) + jpeg
        self.frames += 1
        self.fresh.set()

    def on_au(self, au: bytes):
        # H.264 cannot skip arbitrary frames like JPEG can: if the viewer falls
        # behind, drop the backlog and resume at the next keyframe (<= 1 s, -g 60).
        key = _is_key(au)
        if len(self.aus) > 20:
            self.dropped += len(self.aus)
            self.aus.clear()
            self.need_key = True
        if self.need_key and not key:
            self.dropped += 1
            return
        self.need_key = False
        self.aus.append(struct.pack("<BdB", KIND_H264, time.time(), 1 if key else 0) + au)
        self.frames += 1
        self.fresh.set()

    async def start_video(self, mbps: int):
        if self.mode == "x11" and not self.nvenc:
            self.need_key = True
            self.aus.clear()
            self.nvenc = _NvencCapture(self.on_au)
            await self.nvenc.start(mbps)

    async def stop_video(self):
        if self.nvenc:
            await self.nvenc.stop()
            self.nvenc = None

    async def start_cdp(self) -> bool:
        if self.cdp and self.cdp.ws:
            return True
        cdp = _Cdp(self.on_frame)
        try:
            if not await cdp.connect():
                return False
        except Exception as e:
            logger.warning(f"[render] CDP unreachable at {CDP_HTTP}: {e}")
            return False
        self.cdp = cdp
        self.mode = "x11" if _x11_available() else "cdp"
        if self.mode == "cdp":
            await self.apply_size()
        return True

    async def apply_size(self):
        # The x11 path renders at the display's fixed size (the PROTO's 4K); only
        # the screencast fallback follows the viewer's window.
        if not self.cdp or self.mode != "cdp":
            return
        w, h = self.size
        await self.cdp.send("Page.stopScreencast")
        await self.cdp.send(
            "Emulation.setDeviceMetricsOverride",
            {"width": w, "height": h, "deviceScaleFactor": 1, "mobile": False},
        )
        await self.cdp.send(
            "Page.startScreencast",
            {"format": "jpeg", "quality": self.quality, "maxWidth": w, "maxHeight": h, "everyNthFrame": 1},
        )

    async def stop_cdp(self):
        if self.cdp:
            await self.cdp.send("Page.stopScreencast")
            await self.cdp.close()
            self.cdp = None

    async def input(self, m: dict):
        if not self.cdp:
            return
        t = m.get("t")
        if t == "mouse":
            await self.cdp.send("Input.dispatchMouseEvent", {
                "type": m["type"], "x": m["x"], "y": m["y"], "button": m.get("button", "none"),
                "buttons": m.get("buttons", 0), "clickCount": m.get("clickCount", 0),
                "modifiers": m.get("modifiers", 0),
            })
        elif t == "wheel":
            await self.cdp.send("Input.dispatchMouseEvent", {
                "type": "mouseWheel", "x": m["x"], "y": m["y"],
                "deltaX": m.get("dx", 0), "deltaY": m.get("dy", 0), "modifiers": m.get("modifiers", 0),
            })
        elif t == "key":
            p = {
                "type": m["type"], "key": m.get("key", ""), "code": m.get("code", ""),
                "windowsVirtualKeyCode": m.get("keyCode", 0), "modifiers": m.get("modifiers", 0),
            }
            if m["type"] == "keyDown":
                # Enter has to carry its text or the chat box never submits; every
                # other forwarded key is non-text (text arrives via insertText) and
                # must be rawKeyDown, or Chrome types its `key` into the field.
                if m.get("key") == "Enter":
                    p["text"] = "\r"
                else:
                    p["type"] = "rawKeyDown"
            await self.cdp.send("Input.dispatchKeyEvent", p)
        elif t == "text":
            await self.cdp.send("Input.insertText", {"text": m.get("text", "")})
            logger.debug(f"[render] text input {m.get('text', '')[:20]!r}")


hub = _Hub()


def init_render_routes() -> APIRouter:
    router = APIRouter()

    @router.post("/render-stream/reload")
    async def reload_viewer(request: Request):
        """Make the connected viewer load the latest render_view.html. A unit
        like the PROTO has no keyboard, so an edited page never reached it."""
        if not request.client or request.client.host not in ("127.0.0.1", "::1"):
            return JSONResponse({"ok": False, "error": "loopback only"}, status_code=403)
        if not hub.viewer:
            return {"ok": False, "error": "no viewer connected"}
        await hub.send_viewer(hub.viewer, json.dumps({"t": "reload"}))
        return {"ok": True}

    @router.get("/render-view")
    async def render_view():
        # no-store: the PROTO kept running a stale copy after an update.
        return HTMLResponse(VIEW_HTML.read_text(encoding="utf-8"), headers={"Cache-Control": "no-store"})

    @router.websocket("/render-stream/source")
    async def source_ws(ws: WebSocket):
        await ws.accept()
        hub.source = ws
        logger.info("[render] source tab connected")
        try:
            while True:
                msg = await ws.receive()
                if msg["type"] == "websocket.disconnect":
                    break
                if msg.get("text"):
                    m = json.loads(msg["text"])
                    if m.get("t") == "micq":
                        hub.mic_q_ms = m.get("ms")
                    continue
                data = msg.get("bytes") or b""
                if data[:1] == bytes([KIND_AUDIO]):
                    hub.audio_samples += (len(data) - 1) // 2
                if data[:1] == bytes([KIND_AUDIO]) and hub.viewer:
                    try:
                        await hub.send_viewer(hub.viewer, data)
                    except Exception:
                        pass
        except WebSocketDisconnect:
            pass
        finally:
            if hub.source is ws:
                hub.source = None

    @router.websocket("/render-stream/mic-inject")
    async def mic_inject_ws(ws: WebSocket):
        """Test hook: feed mic PCM (kind 3 frames) to the render tab exactly as
        a viewer would, without taking the viewer slot. Loopback only."""
        if not ws.client or ws.client.host not in ("127.0.0.1", "::1"):
            await ws.close(code=4003)
            return
        await ws.accept()
        # Two mics into one queue would interleave (and the render tab's
        # catch-up logic would thin both out), so the viewer's is muted meanwhile.
        hub.injecting = True
        try:
            while True:
                data = await ws.receive_bytes()
                # Camera frames too, so head tracking can be exercised on a box
                # with no camera (render_view.html sends kind 5 the same way).
                if data[:1] in (bytes([KIND_MIC]), bytes([KIND_CAM])) and hub.source:
                    await hub.source.send_bytes(data)
        except WebSocketDisconnect:
            pass
        finally:
            hub.injecting = False

    @router.websocket("/render-stream/view")
    async def view_ws(ws: WebSocket):
        await ws.accept()
        if hub.viewer:
            try:
                await hub.viewer.close(code=4000, reason="replaced by a newer viewer")
            except Exception:
                pass
        hub.viewer = ws
        logger.info("[render] viewer connected")

        async def pump():
            while True:
                await hub.fresh.wait()
                hub.fresh.clear()
                frame, hub.latest = hub.latest, None
                if frame:
                    await hub.send_viewer(ws, frame)
                aus, hub.aus = hub.aus, []
                for au in aus:
                    await hub.send_viewer(ws, au)
                    hub.sent += 1

        async def stats():
            while True:
                n0, s0, d0, a0 = hub.frames, hub.sent, hub.dropped, hub.audio_samples
                await asyncio.sleep(10)
                size = X_SIZE if hub.mode == "x11" else hub.size
                v = hub.viewer_stat
                logger.info(
                    f"[render] {hub.mode} {size}: made {(hub.frames - n0) / 10:.1f} fps, sent {(hub.sent - s0) / 10:.1f}, "
                    f"dropped {hub.dropped - d0} | viewer: in {v.get('inFps')} drawn {v.get('drawn')} fps, "
                    f"{v.get('mbps')} Mbps, rtt {v.get('rtt')} ms, {v.get('codec')} dec={v.get('dec')} q={v.get('q')} "
                    f"| audio {(hub.audio_samples - a0) / 10000:.1f} kS/s in, viewer ctx {v.get('ctxRate')} Hz "
                    f"| audio queues: mic {hub.mic_q_ms} ms (render side), play {v.get('playQ')}"
                    f"/{v.get('jbTarget')} ms target, {v.get('starves')} starves "
                    f"+ device {v.get('baseLatency')} ms (viewer) "
                    f"{v.get('err') or ''} | {v.get('ua', '')[:70]}"
                )

        if not await hub.start_cdp():
            await hub.send_viewer(ws, json.dumps({"t": "error", "msg": "render host tab (?render=1) not found"}))
        await hub.send_viewer(ws, json.dumps({
            "t": "mode", "mode": hub.mode, "size": X_SIZE,
            "stream": _stream_size(), "rotate": 90 if _rotated() else 0,
        }))
        async def mic_monitor():
            # The VAD's live level and decisions, 10x a second, while the panel is open.
            seen = mic_tuning.MONITOR.seq
            while True:
                await asyncio.sleep(0.1)
                mon = mic_tuning.MONITOR
                ev = mon.since(seen)
                seen = mon.seq
                await hub.send_viewer(ws, json.dumps({
                    "t": "mic-level", "p": round(mon.prob, 3), "db": round(mon.db, 1),
                    "st": mon.state, "ev": ev,
                }))

        # The viewer applies gain / noise suppression to its own capture, so it needs
        # the stored setup before it opens the mic.
        await hub.send_viewer(ws, json.dumps({"t": "mic", **mic_tuning.snapshot()}))
        tasks = [asyncio.create_task(pump()), asyncio.create_task(stats())]
        monitor: asyncio.Task | None = None
        try:
            while True:
                msg = await ws.receive()
                if msg["type"] == "websocket.disconnect":
                    break
                if msg.get("bytes"):
                    data = msg["bytes"]
                    if data[:1] == bytes([KIND_MIC]) and hub.source and not hub.injecting:
                        try:
                            await hub.source.send_bytes(data)
                        except Exception:
                            pass
                    elif data[:1] == bytes([KIND_CAM]) and hub.source:
                        # The viewer's camera, for the render tab's head tracker.
                        try:
                            await hub.source.send_bytes(data)
                        except Exception:
                            pass
                    continue
                m = json.loads(msg.get("text") or "{}")
                if m.get("t") == "size":
                    hub.size = (int(m["w"]), int(m["h"]))
                    hub.quality = int(m.get("q", hub.quality))
                    if await hub.start_cdp():
                        await hub.apply_size()
                        await hub.start_video(int(m.get("mbps", 30)))
                elif m.get("t") == "ping":
                    await hub.send_viewer(ws, json.dumps({"t": "pong", "c": m["c"], "s": time.time()}))
                elif m.get("t") == "stat":
                    hub.viewer_stat = m
                elif m.get("t") == "caps":
                    logger.info(f"[render] viewer hardware decode: {m.get('caps')}")
                elif m.get("t") == "mic-set":
                    snap = mic_tuning.update(m.get("values") or {})
                    await hub.send_viewer(ws, json.dumps({"t": "mic", **snap}))
                elif m.get("t") == "mic-reset":
                    await hub.send_viewer(ws, json.dumps({"t": "mic", **mic_tuning.reset()}))
                elif m.get("t") == "mic-monitor":
                    if m.get("on") and not monitor:
                        monitor = asyncio.create_task(mic_monitor())
                    elif not m.get("on") and monitor:
                        monitor.cancel()
                        monitor = None
                else:
                    await hub.input(m)
        except WebSocketDisconnect:
            pass
        finally:
            for t in tasks:
                t.cancel()
            if monitor:
                monitor.cancel()
            if hub.viewer is ws:
                hub.viewer = None
                await hub.stop_video()
                await hub.stop_cdp()
            logger.info("[render] viewer disconnected")

    return router
