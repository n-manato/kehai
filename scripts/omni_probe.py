"""Spoken-turn probe: play a WAV into KEHAI's server VAD and watch the reply.

    .venv/bin/python scripts/omni_probe.py question.wav [more.wav ...] [--window 40] [--gain 1.0]
                                           [--cam picture.jpg] [--expect-ignored]

Unlike e2e_probe.py (typed text), this exercises the MIC path: the file is
streamed as `raw-audio-data` frames (16 kHz float32, 512 samples = 32 ms, in
real time) exactly as the browser's VAD context sends them, then 1.5 s of
silence so the server's Silero VAD closes the turn. In omni mode the server
attaches this audio to the LLM request; in cascade mode Whisper's text is all
the LLM gets — the SAME probe measures both, so the two modes can be compared
on the same utterances (what she heard = `user-input-transcription`, what she
said = audio-stream-start display_text).

Prints: the transcript the server settled on, time from end-of-speech to the
first audio, what was spoken, motion clip ids and their fetch status.
Exit 0 = a reply with audio arrived.

Loudness: the server VAD opens a turn only above its onset gate (68 dB, see
conf.yaml); synthetic files are normalised to peak 0.9 by default. --gain
multiplies after that (0.4 is a handy "too quiet to count" control).
"""

import asyncio
import os
import json
import sys
import time
import urllib.request
from collections import Counter

import numpy as np
import soundfile as sf
import websockets
from scipy.signal import resample_poly

HOST = "127.0.0.1:12394"
URL = f"ws://{HOST}/client-ws"
FRAME = 512  # samples @16 kHz = 32 ms, what vad-context.tsx sends
TAIL_SILENCE_S = 1.5  # > end_misses (22) * 32 ms = 0.7 s + grace


def load_16k(path: str, gain: float) -> np.ndarray:
    x, sr = sf.read(path, dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != 16000:
        from math import gcd

        g = gcd(sr, 16000)
        x = resample_poly(x, 16000 // g, sr // g).astype(np.float32)
    peak = float(np.max(np.abs(x))) or 1.0
    x = x * (0.9 / peak) * gain
    # Every run a slightly different signal (noise ~-90 dBFS, inaudible, under
    # every gate): replaying the SAME samples hit the server's caches — the
    # Thinker's transcript (omni_addressee.HEARD, keyed by the audio) and vLLM's
    # prefix cache — and the second run of the latency bench came out 0.1-0.2 s
    # faster than any real visitor could (2026-11-08). PROBE_SAME_AUDIO=1 = old way.
    if os.getenv("PROBE_SAME_AUDIO") != "1":
        x = x + np.random.default_rng().normal(0, 3e-5, len(x)).astype(np.float32)
    return np.clip(x, -1, 1).astype(np.float32)


async def fetch_clip(mid: str, results: dict) -> None:
    loop = asyncio.get_running_loop()
    url = f"http://{HOST}/motion/clip-smpl/{mid}"
    for _ in range(40):
        try:

            def _get():
                with urllib.request.urlopen(url, timeout=10) as r:
                    return r.status, r.read()

            status, body = await loop.run_in_executor(None, _get)
            if status == 200:
                data = json.loads(body)
                frames = data.get("joints") or data.get("frames") or []
                results[mid] = f"200, {len(frames)} frames"
                return
        except Exception as e:  # noqa: BLE001 — 202 surfaces as HTTPError
            if "202" not in str(e):
                results[mid] = f"ERROR {e}"
                return
        await asyncio.sleep(0.5)
    results[mid] = "still 202 after 20s"


async def stream_audio_viewer(inj, audio: np.ndarray) -> float:
    """--via-viewer: the PROTO path. The 16 kHz wav goes out as the viewer's mic
    would arrive at the server — 48 kHz int16, kind 3, 20 ms frames on
    /render-stream/mic-inject — so the render-tab round trip (or the direct VAD
    feed, VAD_VIEWER_DIRECT) is part of what is measured. The reply then belongs
    to the render tab's session, not to this probe's, so the first audio is read
    from the server log ([TTFA] ... first_pcm@HH:MM:SS.mmm); this returns the
    WALL time the last speech frame was sent and prints it."""
    from scipy.signal import resample_poly

    x48 = resample_poly(audio, 3, 1).astype(np.float32)  # 16 -> 48 kHz, band-limited like a real mic
    pad = np.zeros(int(TAIL_SILENCE_S * 48000), dtype=np.float32)
    frames = np.concatenate([x48, pad])
    n_frame = 960
    n = (len(frames) + n_frame - 1) // n_frame
    speech_frames = (len(x48) + n_frame - 1) // n_frame
    t_start = time.monotonic()
    w_end = time.time()
    for i in range(n):
        chunk = frames[i * n_frame : (i + 1) * n_frame]
        if len(chunk) < n_frame:
            chunk = np.pad(chunk, (0, n_frame - len(chunk)))
        pcm = (np.clip(chunk, -1, 1) * 32767).astype(np.int16).tobytes()
        await inj.send(bytes([3]) + pcm)
        if i == speech_frames - 1:
            w_end = time.time()
        due = t_start + (i + 1) * n_frame / 48000
        delay = due - time.monotonic()
        if delay > 0:
            await asyncio.sleep(delay)
    print(f"  speech_end_wall={w_end:.3f} ({time.strftime('%H:%M:%S', time.localtime(w_end))}.{int(w_end * 1000) % 1000:03d})")
    return w_end


async def stream_audio(ws, audio: np.ndarray) -> float:
    """Send the file + trailing silence in real time; return the monotonic
    time the last speech frame went out (= end of speech, TTFA's t0)."""
    pad = np.zeros(int(TAIL_SILENCE_S * 16000), dtype=np.float32)
    frames = np.concatenate([audio, pad])
    n = (len(frames) + FRAME - 1) // FRAME
    speech_frames = (len(audio) + FRAME - 1) // FRAME
    t_start = time.monotonic()
    t_end_speech = t_start
    for i in range(n):
        chunk = frames[i * FRAME : (i + 1) * FRAME]
        if len(chunk) < FRAME:
            chunk = np.pad(chunk, (0, FRAME - len(chunk)))
        await ws.send(json.dumps({"type": "raw-audio-data", "audio": chunk.tolist()}))
        if i == speech_frames - 1:
            t_end_speech = time.monotonic()
        # pace to wall clock
        due = t_start + (i + 1) * FRAME / 16000
        delay = due - time.monotonic()
        if delay > 0:
            await asyncio.sleep(delay)
    return t_end_speech


async def one_turn(ws, audio: np.ndarray, window: float) -> bool:
    kinds: Counter[str] = Counter()
    heard = None
    first_audio = None
    said: list[str] = []
    clips: dict[str, str] = {}
    pollers: list[asyncio.Task] = []
    done = asyncio.Event()

    async def listen():
        nonlocal heard, first_audio
        try:
            while True:
                raw = await ws.recv()
                try:
                    m = json.loads(raw)
                except Exception:  # noqa: BLE001
                    continue
                t = m.get("type", "?")
                kinds[t] += 1
                if t == "user-input-transcription":
                    heard = m.get("text")
                elif t == "audio-stream-chunk" and first_audio is None:
                    first_audio = time.monotonic()
                elif t == "audio-stream-start":
                    dt = (m.get("display_text") or {}).get("text")
                    if dt:
                        said.append(dt)
                    gm = (m.get("actions") or {}).get("motion") or {}
                    if gm.get("id"):
                        pollers.append(asyncio.create_task(fetch_clip(gm["id"], clips)))
                elif t == "audio-stream-end":
                    gm = (m.get("actions") or {}).get("motion") or {}
                    if gm.get("upgrade_id"):
                        pollers.append(asyncio.create_task(fetch_clip(gm["upgrade_id"], clips)))
                elif t == "backend-synth-complete":
                    done.set()
        except asyncio.CancelledError:
            pass

    listener = asyncio.create_task(listen())
    t_end = await stream_audio(ws, audio)
    try:
        await asyncio.wait_for(done.wait(), timeout=window)
    except asyncio.TimeoutError:
        pass
    await asyncio.sleep(1.0)  # trailing frames
    listener.cancel()
    if pollers:
        await asyncio.gather(*pollers, return_exceptions=True)

    print(f"  heard   : {heard!r}")
    if first_audio is not None:
        print(f"  first audio {first_audio - t_end:.2f}s after end of speech")
    else:
        print("  first audio: NONE")
    print(f"  spoken  : {' '.join(said)!r}")
    for mid, st in clips.items():
        print(f"  motion {mid[:8]} -> {st}")
    print("  msgs    : " + ", ".join(f"{k} x{v}" for k, v in sorted(kinds.items())))
    return first_audio is not None and kinds.get("audio-stream-chunk", 0) > 0


async def cam_feed(path: str) -> None:
    """Send `path` as the viewer camera at ~2.5 fps for as long as we run."""
    jpeg = open(path, "rb").read()
    cam_url = URL.rsplit("/", 1)[0] + "/render-stream/mic-inject"
    n = 0
    try:
        async with websockets.connect(cam_url, max_size=None) as cam:
            while True:
                # Trailing bytes after the JPEG EOI are ignored by decoders but
                # make consecutive frames differ (render_stream's frozen check).
                await cam.send(bytes([5]) + jpeg + bytes([n % 7 + 1]))
                n += 1
                await asyncio.sleep(0.4)
    except asyncio.CancelledError:
        pass
    except Exception as e:  # noqa: BLE001
        print(f"  cam feed stopped: {e!r}")


async def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    window = 40.0
    gain = 1.0
    if "--window" in sys.argv:
        window = float(sys.argv[sys.argv.index("--window") + 1])
        args = [a for a in args if a != str(window) and a != sys.argv[sys.argv.index("--window") + 1]]
    if "--gain" in sys.argv:
        g = sys.argv[sys.argv.index("--gain") + 1]
        gain = float(g)
        args = [a for a in args if a != g]
    # --cam <jpeg>: feed this picture as the viewer camera for the whole run
    # (kind 5 frames on the mic-inject hook, ~2.5 fps, each one made to differ
    # from the last so render_stream does not flag the feed as frozen). This is
    # what the omni addressee check (omni_addressee.py) judges the turns by:
    # an empty room must be ignored, a person facing the camera answered.
    cam_path = None
    if "--cam" in sys.argv:
        cam_path = sys.argv[sys.argv.index("--cam") + 1]
        args = [a for a in args if a != cam_path]
    # --expect-ignored: the turns must NOT be answered (an empty room, people
    # talking among themselves) — PASS is silence, FAIL is any audio.
    expect_ignored = "--expect-ignored" in sys.argv
    # --via-viewer: send the wav as the PROTO viewer's mic (48 kHz kind-3 frames
    # on /render-stream/mic-inject). The reply goes to the render tab's session,
    # so only speech_end_wall is printed here; scripts/omni_latency_bench.py
    # --via-viewer pairs it with the server log's [TTFA] first_pcm@ wall stamp.
    via_viewer = "--via-viewer" in sys.argv
    if not args:
        print(__doc__)
        return 2
    ok_all = True
    cam_task = asyncio.create_task(cam_feed(cam_path)) if cam_path else None
    if cam_task:
        await asyncio.sleep(1.5)  # a few frames in before the first turn
    if via_viewer:
        inj_url = URL.rsplit("/", 1)[0] + "/render-stream/mic-inject"
        async with websockets.connect(inj_url, max_size=None) as inj:
            for path in args:
                audio = load_16k(path, gain)
                print(f">>> {path} ({len(audio) / 16000:.2f}s, gain {gain}) via viewer")
                await stream_audio_viewer(inj, audio)
                # keep the hook open (and the viewer's own mic aside) while she answers
                silence = (np.zeros(960, dtype=np.int16)).tobytes()
                for _ in range(int(min(window, 9.0) / 0.02)):
                    await inj.send(bytes([3]) + silence)
                    await asyncio.sleep(0.02)
                print()
        if cam_task:
            cam_task.cancel()
        print("RESULT: SENT (read the server log for [TTFA] first_pcm@)")
        return 0
    async with websockets.connect(URL, max_size=None) as ws:
        try:  # drain the opening burst (set-model, config, history, ...)
            while True:
                await asyncio.wait_for(ws.recv(), timeout=3.0)
        except asyncio.TimeoutError:
            pass
        for path in args:
            audio = load_16k(path, gain)
            print(f">>> {path} ({len(audio) / 16000:.2f}s, gain {gain})")
            ok = await one_turn(ws, audio, window)
            if expect_ignored:
                ok = not ok
                print("  ignored : " + ("yes (as expected)" if ok else "NO — she answered"))
            ok_all = ok_all and ok
            print()
            await asyncio.sleep(2.0)
    if cam_task:
        cam_task.cancel()
    print("RESULT:", "PASS" if ok_all else "FAIL")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
