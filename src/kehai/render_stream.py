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
    5 cam    viewer -> server   JPEG frame of the viewer's camera (head tracking).
                                The server finds the faces in it (face_detect.py)
                                and sends the render tab only {"t": "faces", ...};
                                without a detector the frame goes on to the
                                render tab, which then detects for itself.
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
from collections import deque
from pathlib import Path

import numpy as np
import websockets
from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, Response
from loguru import logger
from scipy.signal import resample_poly

import av

from . import active_speaker, face_detect
from .utils import asr_capture
from .vad import mic_tuning
from . import voice_direction

CDP_HTTP = os.environ.get("RENDER_CDP", "http://127.0.0.1:9222")
VIEW_HTML = Path(__file__).with_name("render_view.html")
KIND_AUDIO, KIND_VIDEO, KIND_MIC, KIND_H264, KIND_CAM = 1, 2, 3, 4, 5
# A face cut by the viewer from its full-resolution camera picture, for
# active_speaker: <B kind><H track id><H face px> + JPEG (112x112 grayscale).
KIND_FACE = 6
# Camera-to-sound delay the viewer cannot see (the USB camera and its driver,
# before Chrome stamps the frame), found with active_speaker's ASD_LAG_SWEEP.
_FACE_EXTRA_LAG_S = float(os.getenv("ASD_EXTRA_LAG_S", "0") or 0)
# Direct PCM path: how far ahead of real time the viewer is fed (fills its
# jitter buffer at once; its floor is 40 ms, its lag trim starts at target+40).
DIRECT_LEAD_S = float(os.getenv("VIEWER_PCM_LEAD_S", "0.04") or 0.04)

# The 4K path. CDP screencast tops out near 30 fps / ~90 ms at 2160x3840 (CPU
# JPEG), so when the render host runs on a virtual display the server grabs that
# display and encodes on the 3090's NVENC instead. FFmpeg 8 specifically: 4.2's
# single-threaded pipeline serialises grab and encode and caps at ~55 fps at 4K;
# 8.x overlaps them and holds 60.
X_DISPLAY = os.environ.get("RENDER_DISPLAY", ":57")
X_SIZE = os.environ.get("RENDER_SIZE", "2160x3840")
FFMPEG = os.environ.get("RENDER_FFMPEG", str(Path.home() / "opt/ffmpeg8/ffmpeg"))
NVENC_GPU = os.environ.get("RENDER_NVENC_GPU", "0")
# How the display is captured (2026-11-05). x11grab asks the X server for the pixels;
# on the GPU X server that readback waits until the GPU has drained what every CUDA
# client queued (Whisper's encoder/decoder, the TTS graphs, GestureLSM: 100-700 ms),
# and the single-threaded X server holds Chrome's presents meanwhile — the render
# froze at 11-15 fps under that load while drawing 3.6 ms of GPU work per frame.
# kehai-glgrab (scripts/pcb-render-x/glgrab) binds the window as a GL texture and
# feeds it to NVENC on the GPU: any wait happens in that process, never in the X
# server (test page under the same load: 60 fps, capture 60 fps).
# RENDER_CAPTURE=x11grab restores the ffmpeg path.
CAPTURE = os.environ.get("RENDER_CAPTURE", "auto")
GLGRAB = os.environ.get(
    "RENDER_GLGRAB", str(Path(__file__).resolve().parents[2] / "scripts/pcb-render-x/glgrab/kehai-glgrab")
)
# Viewer link guards (see HubState.send_viewer / pump): waiting bytes above which video is
# dropped to the next keyframe (~70 ms at 30 Mbps), and how long one send may hang.
VIEWER_VIDEO_BUF = int(os.environ.get("RENDER_VIDEO_BUF", str(256 * 1024)))
VIEWER_SEND_TIMEOUT_S = float(os.environ.get("RENDER_SEND_TIMEOUT_S", "3"))
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


def _use_glgrab() -> bool:
    return (
        CAPTURE != "x11grab" and not _rotated() and NVENC_GPU == "0"
        and os.access(GLGRAB, os.X_OK)
    )


class _NvencCapture:
    """The display as H.264 access units (split on the AUD the encoder emits):
    kehai-glgrab (GPU texture -> NVENC), else x11grab -> h264_nvenc."""

    def __init__(self, on_au, on_end=None):
        self.on_au = on_au
        self.on_end = on_end  # called when the encoder exits without stop()
        self.stopping = False
        self.proc: asyncio.subprocess.Process | None = None
        self.reader: asyncio.Task | None = None
        self.err_reader: asyncio.Task | None = None
        self.err_tail: list[str] = []

    async def start(self, mbps: int):
        w, h = X_SIZE.split("x")
        if _use_glgrab():
            self.proc = await asyncio.create_subprocess_exec(
                GLGRAB, w, h, "60", str(mbps),
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                env={**os.environ, "DISPLAY": X_DISPLAY},
            )
            self.reader = asyncio.create_task(self._read())
            self.err_reader = asyncio.create_task(self._read_err())
            logger.info(
                f"[render] GPU capture {w}x{h}@60 {mbps} Mbps from {X_DISPLAY} "
                f"(kehai-glgrab: window texture -> NVENC, no X readback)"
            )
            return
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
        self.err_reader = asyncio.create_task(self._read_err())
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
            proc = self.proc
            rc = await proc.wait() if proc else None
            if self.err_reader:
                try:
                    await asyncio.wait_for(asyncio.shield(self.err_reader), 1)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    pass
            err = " | ".join(self.err_tail[-4:])
            logger.warning(f"[render] NVENC capture ended (exit {rc}){': ' + err[-400:] if err else ''}")
            if not self.stopping and self.on_end:
                self.on_end()

    async def _read_err(self):
        """Drain stderr as it comes: kehai-glgrab reports once a minute, and an
        undrained pipe would block it after a few hours."""
        proc = self.proc
        while proc and proc.stderr:
            line = await proc.stderr.readline()
            if not line:
                break
            text = line.decode(errors="replace").strip()
            if not text:
                continue
            self.err_tail = (self.err_tail + [text])[-20:]
            if text.startswith("[glgrab]"):
                logger.info(f"[render] {text}")
            else:
                logger.warning(f"[render] capture: {text}")

    async def stop(self):
        self.stopping = True
        if self.proc and self.proc.returncode is None:
            self.proc.terminate()
            try:
                await asyncio.wait_for(self.proc.wait(), 3)
            except asyncio.TimeoutError:
                self.proc.kill()
        if self.reader:
            self.reader.cancel()
        if self.err_reader:
            self.err_reader.cancel()
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
        # The mic relay, the test hook and the face results all write to the
        # source from different tasks — the same hazard as viewer_lock.
        self.source_lock = asyncio.Lock()
        self.cdp: _Cdp | None = None
        self.latest: bytes | None = None
        self.fresh = asyncio.Event()
        self.size = (1080, 1920)
        self.quality = 70
        self.frames = 0
        self.mode = "cdp"
        self.nvenc: _NvencCapture | None = None
        self.video_mbps = 30
        self.video_revivals: list[float] = []
        self.aus: list[bytes] = []
        self.need_key = True
        self.sent = 0
        self.dropped = 0
        self.viewer_stat: dict = {}
        self.viewer_lag_max = 0  # worst main-thread lag the viewer reported this window (ms)
        self.viewer_skipped = 0  # frames the viewer skipped to catch up, this window
        self.mic_q_ms = None
        # Viewer mic audio the render tab threw away (queue over 80 ms) or
        # filled with silence (queue empty mid-stream), ms since the last stats
        # line. Either one cuts holes in the speech whisper hears.
        self.mic_drop_ms = 0
        self.mic_gap_ms = 0
        # The viewer's mic exactly as it arrived (48 kHz int16), the last minute,
        # for ASR: the render tab's copy reaches the server VAD damaged — it drops
        # and zero-fills audio to keep its queue short, resamples to 16 kHz on the
        # way, and lags ~0.1 s (measured 81-111 ms) — and whisper misheard that
        # copy where it got the original right ("Hi, how are you?" vs "Thank
        # you."). See websocket_handler._CUMULATIVE_ASR.
        self.viewer_mic: deque[tuple[float, np.ndarray]] = deque()
        # VAD_VIEWER_DIRECT: the render client's handler registers an async
        # callable(pcm 48 kHz int16 bytes) here and runs the server VAD on the
        # viewer's mic as it arrives, instead of on the render tab's round-trip
        # copy (which lags it 81-111 ms — measured by cross-correlation — before
        # the first VAD frame is even assembled). 2026-11-04.
        self.mic_sink = None
        # VIEWER_PCM_DIRECT: the reply's PCM goes to the viewer straight from the
        # server (ViewerMirror on the render client's socket) instead of being
        # played by the render tab, tapped, and shipped back here — a hop the
        # reviewer measured at 40-80 ms (ctx 5-10 + MediaStream 21-31 + Tap
        # 0-20 + dispatch 0-16) on top of the tab's preroll. While on, audio
        # the tab taps is dropped at source_ws so nothing plays twice.
        self.direct_pcm = os.getenv("VIEWER_PCM_DIRECT", "0") == "1"
        self._up = None  # stateful 32 -> 48 kHz for viewer_pcm32
        self._direct_buf = bytearray()  # 48 kHz int16 waiting for _pace_direct
        self.direct_sent_at = 0.0  # monotonic time the last direct frame went out (speaking)
        self._pacer: asyncio.Task | None = None
        self.direct_chunks = 0
        self.direct_first_wall = 0.0
        self.injecting = False
        # logs/MIC_DEAF: the viewer's mic is still recorded (asr_capture) but not
        # handed to the VAD — she hears nothing and is driven by typed text
        # (/render-stream/type). Checked once a second; no restart needed.
        self.text_sink = None
        self._deaf = False
        self._deaf_checked = 0.0
        self.audio_samples = 0  # PCM samples forwarded to the viewer (should be 48k/s)
        # Face detection on the viewer's camera: None = not loaded yet, False =
        # unavailable (frames are forwarded instead). One frame in flight.
        self.faces: face_detect.FaceDetector | None | bool = None
        self.face_busy = False
        self.face_task: asyncio.Task | None = None
        self.face_frames = 0
        self.face_ms = 0.0
        self.face_count = 0
        self._face_dec = None
        self.face_crops = 0  # viewer full-resolution face crops (KIND_FACE) since the last stats line
        self.face_age_ms: list[int] = []  # their camera-to-send ages since the last stats line
        # Latest camera JPEG, for GET /render-stream/camera.jpg. One frame in
        # memory only; never written to disk.
        self.cam_last = b""
        self.cam_at = 0.0
        self.cam_changed_at = 0.0  # when the picture last differed from the one before
        self.cam_frozen_logged = False

    @property
    def deaf(self) -> bool:
        now = time.monotonic()
        if now - self._deaf_checked > 1.0:
            self._deaf_checked = now
            on = (Path(__file__).resolve().parents[2] / "logs" / "MIC_DEAF").exists()
            if on != self._deaf:
                logger.info(f"[render] microphone -> VAD is now {'OFF (logs/MIC_DEAF): typed text only' if on else 'ON'}")
            self._deaf = on
        return self._deaf

    def remember_mic(self, pcm: bytes) -> None:
        now = time.time()
        self.viewer_mic.append((now, np.frombuffer(pcm, dtype=np.int16)))
        while self.viewer_mic and now - self.viewer_mic[0][0] > 60:
            self.viewer_mic.popleft()

    def viewer_mic_live(self) -> bool:
        return bool(self.viewer and self.viewer_mic) and time.time() - self.viewer_mic[-1][0] < 0.5

    def viewer_audio_16k(self, t0: float, t1: float) -> np.ndarray:
        """The viewer's mic between wall times t0 and t1, as 16 kHz float32
        (polyphase-filtered, so nothing aliases into the speech band)."""
        parts = []
        for t, a in list(self.viewer_mic):
            first = t - len(a) / 48000  # arrival time of the chunk's first sample
            if t < t0 or first > t1:
                continue
            i0 = max(0, int((t0 - first) * 48000))
            i1 = min(len(a), int((t1 - first) * 48000) + 1)
            parts.append(a[i0:i1])
        if not parts:
            return np.zeros(0, dtype=np.float32)
        x = np.concatenate(parts).astype(np.float32) / 32768.0
        return resample_poly(x, 1, 3).astype(np.float32)

    async def send_source(self, data: bytes | str) -> None:
        """Every write to the render tab goes through here (see source_lock)."""
        ws = self.source
        if not ws:
            return
        async with self.source_lock:
            try:
                if isinstance(data, str):
                    await ws.send_text(data)
                else:
                    await ws.send_bytes(data)
            except Exception:
                pass

    async def on_cam(self, data: bytes) -> None:
        """A camera frame (kind 5) from the viewer or the test hook.

        Never queued: while one frame is being detected the next ones are
        dropped, because a stale head position is worse than a skipped one.
        """
        now = time.time()
        if data[1:] != self.cam_last:
            if self.cam_frozen_logged:
                logger.info(f"[render] camera moving again after {now - self.cam_changed_at:.1f}s frozen")
                self.cam_frozen_logged = False
            self.cam_changed_at = now
        self.cam_last, self.cam_at = data[1:], now
        # A FROZEN feed — the viewer resending one identical picture while its
        # camera stalls (2026-10-23 on the PROTO, many seconds at a time) — shows
        # nobody, and the addressee check then dropped every utterance as "nobody
        # within 3.5 m". Feed it no frames instead: with no fresh frames the check
        # answers as if there were no camera (face_detect.Attention.judge).
        if now - self.cam_changed_at > 1.0:
            if not self.cam_frozen_logged:
                logger.warning("[render] camera frozen (identical frames) — addressee check off until it moves")
                self.cam_frozen_logged = True
            return
        if not self.source:
            return
        if self.faces is False:
            await self.send_source(data)
            return
        if self.face_busy:
            return
        self.face_busy = True
        # Held so the task is not garbage-collected mid-detection.
        self.face_task = asyncio.create_task(self._detect(data[1:]))

    async def _detect(self, jpeg: bytes) -> None:
        try:
            if self.faces is None:
                self.faces = await asyncio.to_thread(face_detect.load) or False
                if not self.faces:
                    return
                if active_speaker.ENABLED:
                    # load the lip model now, not on the first verdict
                    asyncio.get_running_loop().run_in_executor(None, active_speaker._load)
            t0 = time.perf_counter()
            result = await asyncio.to_thread(self._detect_and_track, jpeg)
            self.face_ms += (time.perf_counter() - t0) * 1000
            self.face_frames += 1
            self.face_count = len(result["faces"])
            attn = face_detect.ATTENTION.record(result["faces"])
            if face_detect.ADDRESSEE:
                # Per-frame verdict for the render tab's barge-in check, so it
                # judges on exactly the frames the turn will be judged on.
                result["attn"] = attn
            await self.send_source(json.dumps({"t": "faces", **result}))
            if self.viewer is not None and active_speaker.COLLECT:
                # the near faces, for the viewer to cut at full resolution (kind 6)
                await self.send_viewer(self.viewer, json.dumps({"t": "asd-faces", "faces": active_speaker.TRACKS.boxes()}))
        except Exception as err:
            logger.debug(f"[faces] frame skipped: {err}")
        finally:
            self.face_busy = False

    def _detect_and_track(self, jpeg: bytes) -> dict:
        """(worker thread) YuNet on the frame, then the lip crops of its near faces."""
        result = self.faces.detect(jpeg)
        gray = getattr(self.faces, "last_gray", None)
        if gray is not None:
            active_speaker.TRACKS.record(gray, result["faces"])
        return result

    def on_face(self, data: bytes) -> None:
        """A face crop from the viewer (KIND_FACE)."""
        if len(data) < 8 or not active_speaker.COLLECT:
            return
        try:
            # <u16 track><u16 px><u16 ms since the camera took it> — timed by
            # when it was taken, not when it arrived (~0.5 s later at 4K).
            tid, px, age = struct.unpack("<HHH", data[1:7])
            self.face_age_ms.append(age)
            if self._face_dec is None:
                self._face_dec = av.CodecContext.create("mjpeg", "r")
            fr = self._face_dec.decode(av.Packet(data[7:]))[0]
            g = fr.to_ndarray(format="gray")
            if g.shape != (112, 112):
                g = active_speaker._resize(g, 112, 112)
            active_speaker.TRACKS.record_hires(time.monotonic() - age / 1000.0 - _FACE_EXTRA_LAG_S, tid, g, float(px))
            self.face_crops += 1
        except Exception as err:  # noqa: BLE001
            logger.debug(f"[asd] viewer face crop skipped: {err!r}")

    async def viewer_pcm32(self, pcm: bytes) -> None:
        """Direct path: 32 kHz int16 mono PCM (the reply, as tts_manager ships
        it) -> 48 kHz -> KIND_AUDIO to the viewer. Stateful across chunks."""
        if not self.viewer or not pcm:
            return
        from .tts.qwen3tts_stream import _Upsampler  # same linear, border-safe upsampler

        if self._up is None:
            self._up = _Upsampler(32000, 48000)
        x = np.frombuffer(pcm[: len(pcm) & ~1], dtype=np.int16)
        out = self._up.feed(x)
        if out.size == 0:
            return
        if self.direct_chunks == 0:
            self.direct_first_wall = time.time()
            logger.info(
                "[render] direct PCM: first chunk to the viewer @"
                + time.strftime("%H:%M:%S", time.localtime(self.direct_first_wall))
                + f".{int(self.direct_first_wall * 1000) % 1000:03d}"
            )
        self.direct_chunks += 1
        # PACED, not pushed: Qwen3-TTS hands over 480 ms of audio every ~100 ms
        # (4x real time), and the viewer's jitter buffer reads anything queued
        # beyond its target as standing lag and trims it, and caps the queue at
        # 1 s — so pushing a whole sentence at once chopped it ("ちゃんと喋らない",
        # 2026-11-04 07:52). The tab's tap fed it at playback rate; so does this:
        # 20 ms frames on a real-time clock, DIRECT_LEAD_S sent ahead.
        self._direct_buf += out.tobytes()
        if self._pacer is None or self._pacer.done():
            self._pacer = asyncio.create_task(self._pace_direct())

    async def _pace_direct(self) -> None:
        frame = 960 * 2  # 20 ms of 48 kHz int16
        lead = int(DIRECT_LEAD_S * 48000)
        start, sent = time.monotonic(), 0
        idle_since = None
        while True:
            if len(self._direct_buf) < frame:
                if self._direct_buf:
                    # the tail of a sentence: pad to one frame
                    self._direct_buf += bytes(frame - len(self._direct_buf))
                    continue
                if idle_since is None:
                    idle_since = time.monotonic()
                elif time.monotonic() - idle_since > 1.0:
                    return  # nothing for a second: a new sentence starts a new clock
                await asyncio.sleep(0.005)
                continue
            if idle_since is not None:
                # Audio resumed after a gap: restart the clock, counting what
                # the viewer still holds. Restarting from zero sent a fresh
                # DIRECT_LEAD_S on top of it whenever the buffer ran dry for a
                # moment (the TTS's first chunks arrive just in time), and the
                # extra ~40 ms stayed as standing lag the viewer then trimmed
                # out of the speech (play 80-100 ms against a 20 ms target).
                ahead = sent / 48000 - (time.monotonic() - start)
                if ahead < -0.02 and sent:
                    # the viewer ran out before the next audio reached us: a gap the
                    # listener hears, caused upstream (TTS / sentence hand-over)
                    logger.info(f"[render] direct PCM ran dry: audio arrived {-ahead * 1000:.0f} ms after the viewer's ran out")
                backlog = max(0.0, ahead)
                start, sent, idle_since = time.monotonic(), int(backlog * 48000), None
            data = bytes(self._direct_buf[:frame])
            del self._direct_buf[:frame]
            if not self.viewer:
                continue
            t_due = start + (sent - lead) / 48000  # when this frame should leave
            t0 = time.monotonic()
            try:
                await self.send_viewer(self.viewer, bytes([KIND_AUDIO]) + data)
            except Exception:  # noqa: BLE001 — viewer gone; the tab's copy is dropped anyway
                pass
            t1 = time.monotonic()
            self.direct_sent_at = t1
            late = t1 - max(t_due, 0) if sent else 0.0
            if sent and late > 0.05:
                # Past the ~60 ms the viewer holds (lead + its jitter floor) this frame
                # lands after the viewer ran dry. Split: was the loop late to run us,
                # or did the send itself (viewer_lock behind a video frame, TCP) take it?
                logger.info(
                    f"[render] direct PCM late {late * 1000:.0f} ms "
                    f"(woke {max(0.0, t0 - t_due) * 1000:.0f} ms late, send+lock {(t1 - t0) * 1000:.0f} ms)"
                )
            sent += 960
            due = start + (sent - lead) / 48000
            delay = due - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)

    @property
    def speaking(self) -> bool:
        """Her voice is playing on the viewer now (direct path: frames go out on a
        real-time clock, so the last one sent is what plays)."""
        return bool(self._direct_buf) or time.monotonic() - self.direct_sent_at < 0.25

    async def viewer_flush(self, why: str = "") -> None:
        """Direct path: tell the viewer to drop queued audio (barge-in /
        interrupt) and restart the upsampler's state for the next reply."""
        self._up = None
        self.direct_chunks = 0
        self._direct_buf.clear()
        if self._pacer is not None and not self._pacer.done():
            self._pacer.cancel()
        self._pacer = None
        if not self.viewer:
            return
        try:
            await self.send_viewer(self.viewer, json.dumps({"t": "flush", "why": why}))
        except Exception:  # noqa: BLE001
            pass

    async def send_viewer(self, ws: WebSocket, data: bytes | str) -> None:
        """Every write to the viewer goes through here (see viewer_lock).

        Bounded: a viewer that stops reading (12:19 on 2026-11-05 one send sat 10.4 s,
        then the server sent nothing for minutes while the page never reconnected)
        must not wedge the stream. A send still pending after VIEWER_SEND_TIMEOUT_S
        closes the connection; a live viewer reconnects by itself (onclose -> retry)."""
        async with self.viewer_lock:
            try:
                if isinstance(data, str):
                    await asyncio.wait_for(ws.send_text(data), VIEWER_SEND_TIMEOUT_S)
                else:
                    await asyncio.wait_for(ws.send_bytes(data), VIEWER_SEND_TIMEOUT_S)
            except asyncio.TimeoutError:
                logger.warning(f"[render] viewer send stuck {VIEWER_SEND_TIMEOUT_S:.0f} s — closing it to reconnect")
                try:
                    await asyncio.wait_for(ws.close(code=1011, reason="send stalled"), 1.0)
                except Exception:  # noqa: BLE001
                    pass
                raise

    @staticmethod
    def ws_buffered(ws: WebSocket) -> int:
        """Bytes written to the viewer's socket but not yet sent (0 if unknown). Read off
        uvicorn's websockets protocol, which is what Starlette's send() is bound to."""
        try:
            tr = getattr(getattr(ws, "_send", None), "__self__", None).transport
            return int(tr.get_write_buffer_size())
        except Exception:  # noqa: BLE001
            return 0

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
        # A capture whose ffmpeg has exited still sat in self.nvenc, so a
        # viewer that (re)connected never got video again: 0 fps after a page
        # reload until the server was restarted (2026-11-04 18:40).
        if self.nvenc and (self.nvenc.proc is None or self.nvenc.proc.returncode is not None):
            logger.warning("[render] NVENC capture had exited; starting a new one")
            await self.nvenc.stop()
            self.nvenc = None
        self.video_mbps = mbps
        if self.mode == "x11" and not self.nvenc:
            self.need_key = True
            self.aus.clear()
            self.nvenc = _NvencCapture(self.on_au, on_end=self._video_ended)
            await self.nvenc.start(mbps)

    def _video_ended(self):
        # The encoder exited on its own (render Chrome restarted and its window was
        # gone for 5 s, a driver error...). Before, the viewer kept a frozen picture
        # until it reconnected. Start a new one while a viewer is watching — at most
        # 5 times a minute, so a capture that cannot run does not spin.
        now = time.time()
        self.video_revivals = [t for t in self.video_revivals if now - t < 60] + [now]
        if not self.viewer or len(self.video_revivals) > 5:
            if self.viewer:
                logger.error("[render] capture keeps exiting (5 restarts in a minute); not restarting")
            return

        async def revive():
            await asyncio.sleep(1.0)
            if self.viewer:
                logger.warning("[render] restarting the capture for the connected viewer")
                await self.start_video(self.video_mbps)

        asyncio.get_running_loop().create_task(revive())

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


class ViewerMirror:
    """The render client's WebSocket, wrapped (routes.py) when VIEWER_PCM_DIRECT
    is on: every audio-stream-chunk (PCM; Opus is refused for this client via
    `pcm_only`) and every backchannel clip that goes to the render tab is also
    decoded here and pushed to the viewer at once; the 'user-speech-start'
    control (barge-in) flushes the viewer's queue. GatedSocket wraps this, so a
    speculative reply reaches the viewer only when it is released. Everything
    else delegates to the real socket."""

    pcm_only = True

    def __init__(self, ws: WebSocket):
        self._ws = ws

    def __getattr__(self, name):
        return getattr(self._ws, name)

    async def send_text(self, text: str) -> None:
        await self._ws.send_text(text)
        if not hub.direct_pcm:
            return
        try:
            if '"audio-stream-start"' in text[:48]:
                hub.direct_chunks = 0  # log the next chunk as this sentence's first
            elif '"audio-stream-chunk"' in text[:48]:
                m = json.loads(text)
                if m.get("codec") == "opus":
                    return
                await hub.viewer_pcm32(base64.b64decode(m.get("data") or ""))
            elif '"backchannel"' in text[:32]:
                m = json.loads(text)
                raw = base64.b64decode(m.get("audio") or "")
                if raw[:4] == b"RIFF":
                    import io
                    import wave

                    with wave.open(io.BytesIO(raw)) as w:
                        rate, frames = w.getframerate(), w.readframes(w.getnframes())
                    if rate == 32000:
                        await hub.viewer_pcm32(frames)
            elif '"user-speech-start"' in text[:64]:
                await hub.viewer_flush("user-speech-start")
        except Exception as e:  # noqa: BLE001 — the tab still got the message
            logger.debug(f"[render] viewer mirror: {e!r}")


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

    @router.get("/render-stream/mic-level")
    async def mic_level(request: Request):
        """What the VAD hears right now (is the microphone alive?). Loopback only."""
        if not request.client or request.client.host not in ("127.0.0.1", "::1"):
            return JSONResponse({"ok": False, "error": "loopback only"}, status_code=403)
        mon = mic_tuning.MONITOR
        return {"prob": round(float(mon.prob), 3), "db": round(float(mon.db), 1), "state": mon.state, "seq": mon.seq}

    def _local(request: Request) -> bool:
        return bool(request.client) and request.client.host in ("127.0.0.1", "::1")

    @router.get("/render-stream/type")
    async def type_page(request: Request):
        """A text box that speaks to her as the visitor at the PROTO would (loopback only)."""
        if not _local(request):
            return JSONResponse({"ok": False, "error": "loopback only"}, status_code=403)
        return HTMLResponse(_TYPE_PAGE)

    @router.post("/render-stream/say")
    async def say(request: Request):
        if not _local(request):
            return JSONResponse({"ok": False, "error": "loopback only"}, status_code=403)
        text = str((await request.json()).get("text") or "").strip()
        if not text:
            return {"ok": False, "error": "empty"}
        if hub.text_sink is None:
            return {"ok": False, "error": "the render tab is not connected yet"}
        logger.info(f"[render] typed to her: {text[:120]!r}")
        await hub.text_sink(text)
        return {"ok": True}

    @router.post("/render-stream/deaf")
    async def deaf(request: Request):
        if not _local(request):
            return JSONResponse({"ok": False, "error": "loopback only"}, status_code=403)
        flag = Path(__file__).resolve().parents[2] / "logs" / "MIC_DEAF"
        body = await request.json()
        if "on" in body:
            flag.touch() if body["on"] else flag.unlink(missing_ok=True)
            hub._deaf_checked = 0.0
        return {"ok": True, "deaf": flag.exists()}

    @router.post("/render-stream/mic-probe")
    async def mic_probe(request: Request):
        """Diagnostic: the viewer records every audio input raw, in stereo, for a
        few seconds and sends it up (kind 7 -> logs/mic_probe/). Loopback only."""
        if not request.client or request.client.host not in ("127.0.0.1", "::1"):
            return JSONResponse({"ok": False, "error": "loopback only"}, status_code=403)
        if not hub.viewer:
            return {"ok": False, "error": "no viewer connected"}
        await hub.send_viewer(hub.viewer, json.dumps({"t": "mic-probe", "seconds": 4}))
        return {"ok": True}

    @router.get("/render-stream/camera.jpg")
    async def camera_frame(request: Request):
        """The last frame the viewer's camera sent — what the face detector is
        looking at. Answers "she ignores me" when the log says 0 faces in view
        (2026-10-23: speech kept being dropped as not addressed to her). Loopback
        only: the frame shows visitors' faces."""
        if not request.client or request.client.host not in ("127.0.0.1", "::1"):
            return JSONResponse({"ok": False, "error": "loopback only"}, status_code=403)
        if not hub.cam_last:
            return JSONResponse({"ok": False, "error": "no camera frame yet"}, status_code=404)
        return Response(
            hub.cam_last,
            media_type="image/jpeg",
            headers={"Cache-Control": "no-store", "X-Frame-Age": f"{time.time() - hub.cam_at:.2f}"},
        )

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
                        hub.mic_drop_ms += m.get("drop", 0)
                        hub.mic_gap_ms += m.get("gap", 0)
                    continue
                data = msg.get("bytes") or b""
                if data[:1] == bytes([KIND_AUDIO]):
                    hub.audio_samples += (len(data) - 1) // 2
                    if hub.direct_pcm:
                        continue  # the reply reaches the viewer directly (viewer_pcm32)
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
                if data[:1] == bytes([KIND_MIC]):
                    # Stands in for the viewer's mic in every respect (ASR
                    # memory, direct VAD feed, the tab's copy).
                    hub.remember_mic(data[1:])
                    if hub.mic_sink is not None:
                        await hub.mic_sink(data[1:])
                    await hub.send_source(data)
                elif data[:1] == bytes([KIND_CAM]):
                    await hub.on_cam(data)
                elif data[:1] == bytes([KIND_FACE]):
                    hub.on_face(data)
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
                for i, au in enumerate(aus):
                    # Never write video faster than the link drains: past VIEWER_VIDEO_BUF
                    # waiting bytes, drop to the next keyframe. The socket is shared with
                    # her voice (20 ms frames, ~60 ms in hand at the viewer), so a backed-up
                    # video queue delayed the voice too, and a stall followed by a burst of
                    # backlog is what overwhelmed the viewer (2026-11-05 12:19).
                    if hub.ws_buffered(ws) > VIEWER_VIDEO_BUF:
                        hub.dropped += len(aus) - i
                        hub.need_key = True
                        break
                    await hub.send_viewer(ws, au)
                    hub.sent += 1

        async def stats():
            while True:
                n0, s0, d0, a0 = hub.frames, hub.sent, hub.dropped, hub.audio_samples
                hub.mic_drop_ms = hub.mic_gap_ms = 0
                f0, fm0 = hub.face_frames, hub.face_ms
                await asyncio.sleep(10)
                # A viewer watching and no picture made for 10 s, nothing
                # running: start one. (2026-11-09: a page reload overlapped the
                # old page's socket, the capture stopped and the new page never
                # asked again — 0 fps until reloaded once more.)
                # Also when the render tab was not reachable yet as the viewer
                # connected (a server restart: the viewer reconnects before the
                # render tab does, start_cdp fails, the mode stays "cdp" and
                # nothing asks again — 0 fps, 2026-11-09 02:42).
                if hub.viewer is ws and hub.frames == n0 and (
                    hub.nvenc is None or hub.nvenc.proc is None or hub.nvenc.proc.returncode is not None
                ):
                    if hub.mode != "x11" and await hub.start_cdp() and hub.mode == "x11":
                        await hub.send_viewer(ws, json.dumps({
                            "t": "mode", "mode": hub.mode, "size": X_SIZE,
                            "stream": _stream_size(), "rotate": 90 if _rotated() else 0,
                        }))
                    if hub.mode == "x11":
                        logger.warning("[render] viewer connected but no capture running — starting one")
                        await hub.start_video(hub.video_mbps or 30)
                nf = hub.face_frames - f0
                faces = (
                    f"faces {nf / 10:.1f} fps {(hub.face_ms - fm0) / nf:.0f} ms, {hub.face_count} in view"
                    + (f", {hub.face_crops / 10:.0f} lip crops/s" if hub.face_crops else "")
                    + (f" (taken {int(np.median(hub.face_age_ms))} ms before sending)" if hub.face_age_ms else "")
                    if nf else "faces off"
                )
                hub.face_crops = 0
                hub.face_age_ms = []
                size = X_SIZE if hub.mode == "x11" else hub.size
                v = hub.viewer_stat
                lag_note = hub.viewer_lag_max
                hub.viewer_lag_max = 0
                skipped_note = hub.viewer_skipped
                hub.viewer_skipped = 0
                logger.info(
                    f"[render] {hub.mode} {size}: made {(hub.frames - n0) / 10:.1f} fps, sent {(hub.sent - s0) / 10:.1f}, "
                    f"dropped {hub.dropped - d0} | viewer: in {v.get('inFps')} drawn {v.get('drawn')} fps, "
                    f"{v.get('mbps')} Mbps, rtt {v.get('rtt')} ms, {v.get('codec')} dec={v.get('dec')} q={v.get('q')} "
                    f"lag {lag_note} ms skipped {skipped_note} "
                    f"| audio {(hub.audio_samples - a0) / 10000:.1f} kS/s in, viewer ctx {v.get('ctxRate')} Hz "
                    f"| audio queues: mic {hub.mic_q_ms} ms (render side, {hub.mic_drop_ms} ms dropped, "
                    f"{hub.mic_gap_ms} ms gaps), play {v.get('playQ')}"
                    f"/{v.get('jbTarget')} ms target, {v.get('starves')} starves, {v.get('trimmedMs', '?')} ms trimmed "
                    f"+ device {v.get('baseLatency')} ms (viewer) | {faces} "
                    f"{v.get('err') or ''} | {v.get('ua', '')[:70]}"
                )

        try:
            if not await hub.start_cdp():
                logger.warning("[render] viewer connected before the render tab was reachable — will retry")
                await hub.send_viewer(ws, json.dumps({"t": "error", "msg": "render host tab (?render=1) not found"}))
            await hub.send_viewer(ws, json.dumps({
                "t": "mode", "mode": hub.mode, "size": X_SIZE,
                "stream": _stream_size(), "rotate": 90 if _rotated() else 0,
            }))
            await hub.send_viewer(ws, json.dumps({"t": "mic", **mic_tuning.snapshot()}))
        except Exception as e:  # noqa: BLE001
            # The page gave up while the server was still starting (its 4 s
            # watchdog closed with 4001, 2026-11-09 20:45:53): this raised out of
            # the handler before its cleanup, as an ASGI error, and left
            # hub.viewer pointing at the dead socket. The page reconnects itself.
            logger.info(f"[render] viewer left during the hello ({e!r})")
            if hub.viewer is ws:
                hub.viewer = None
                await hub.stop_cdp()
            return
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
        tasks = [asyncio.create_task(pump()), asyncio.create_task(stats())]
        monitor: asyncio.Task | None = None
        try:
            while True:
                msg = await ws.receive()
                if msg["type"] == "websocket.disconnect":
                    break
                if msg.get("bytes"):
                    data = msg["bytes"]
                    if data[:1] == bytes([KIND_MIC]) and not hub.injecting:
                        asr_capture.record("viewer", data[1:])
                        hub.remember_mic(data[1:])
                        if hub.mic_sink is not None and not hub.deaf:
                            await hub.mic_sink(data[1:])
                        await hub.send_source(data)
                    elif data[:1] == bytes([KIND_CAM]) and not hub.injecting:
                        # The viewer's camera, for the render tab's head tracker and
                        # the addressee checks. While the test hook injects (omni_probe
                        # --cam) its pictures stand in for the camera, as its mic does.
                        await hub.on_cam(data)
                    elif data[:1] == bytes([KIND_FACE]) and not hub.injecting:
                        hub.on_face(data)
                    elif data[:1] == bytes([8]):
                        # Raw stereo beside the mic, for the voice's direction. Not
                        # while a test injects its own audio: the room's real sound
                        # would be placed in the injected turn's stead (20:15 probe:
                        # dropped as "from the side").
                        if not hub.injecting:
                            voice_direction.feed(data[2:])
                    elif data[:1] == bytes([7]):
                        d = Path(__file__).resolve().parents[2] / "logs" / "mic_probe"
                        d.mkdir(parents=True, exist_ok=True)
                        (d / f"input{data[1]}.s16le").write_bytes(data[2:])
                        logger.info(f"[render] mic probe: input {data[1]}, {len(data) - 2} bytes")
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
                    # worst over the log window (stats() resets it), not just the last second
                    hub.viewer_lag_max = max(hub.viewer_lag_max, int(m.get("lagMax") or 0))
                    hub.viewer_skipped += int(m.get("skipped") or 0)
                elif m.get("t") == "caps":
                    logger.info(f"[render] viewer hardware decode: {m.get('caps')}")
                elif m.get("t") == "cam-res":
                    # The camera's real resolution / rate (lip crops need ~25 fps).
                    logger.info(f"[render] viewer camera runs at {m.get('w')}x{m.get('h')} {m.get('fps', '')} fps {m.get('note', '')}".rstrip())
                elif m.get("t") == "cams":
                    # Which camera the viewer chose and why (render_view.html pickCam).
                    logger.info(
                        f"[render] viewer camera: {m.get('chosen')} (dark {m.get('dark')}) "
                        f"tried {m.get('tried')}"
                    )
                    try:
                        from . import face_detect  # lazy: optional subsystem

                        face_detect.set_viewer_camera(str(m.get("chosen") or ""))
                    except Exception as e:  # noqa: BLE001 — never break the viewer socket
                        logger.warning(f"[render] camera -> addressee distance: {e!r}")
                elif m.get("t") == "dir-mic":
                    logger.info(f"[render] viewer stereo mic for direction: {m}")
                elif m.get("t") == "mic-probe":
                    logger.info(f"[render] mic probe: {m}")
                elif m.get("t") == "mics":
                    # Which microphone the viewer chose (render_view.html: full-band pick).
                    logger.info(
                        f"[render] viewer mic: {m.get('using')} {m.get('settings')} "
                        f"asked {m.get('asked')} constraints {m.get('constraints')} "
                        f"inputs {m.get('inputs')}"
                    )
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


_TYPE_PAGE = """<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>KEHAI — type to her</title>
<style>body{font:16px system-ui;max-width:720px;margin:24px auto;padding:0 12px;background:#111;color:#eee}
input[type=text]{width:100%;font-size:20px;padding:12px;box-sizing:border-box;background:#222;color:#fff;border:1px solid #555;border-radius:6px}
#log div{padding:4px 0;border-bottom:1px solid #333}label{display:block;margin:14px 0}.err{color:#f88}</style>
<h3>Type to her (Enter to send)</h3>
<form id=f><input type=text id=t autofocus autocomplete=off></form>
<label><input type=checkbox id=d> Microphone off for her (the mic is still recorded)</label>
<div id=log></div>
<script>
const $=id=>document.getElementById(id), post=(u,b)=>fetch(u,{method:"POST",headers:{"content-type":"application/json"},body:JSON.stringify(b)}).then(r=>r.json());
post("/render-stream/deaf",{}).then(r=>$("d").checked=r.deaf);
$("d").onchange=()=>post("/render-stream/deaf",{on:$("d").checked}).then(r=>$("d").checked=r.deaf);
$("f").onsubmit=async e=>{e.preventDefault();const text=$("t").value.trim();if(!text)return;$("t").value="";
  const r=await post("/render-stream/say",{text}).catch(e=>({ok:false,error:String(e)}));
  const d=document.createElement("div");d.textContent=(r.ok?"":"[not sent: "+r.error+"] ")+text;if(!r.ok)d.className="err";$("log").prepend(d);};
</script>"""
