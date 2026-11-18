"""Who-is-talking probe: play a video's picture AND sound into KEHAI as the PROTO
viewer would, and read back whether she took the turn.

    .venv/bin/python scripts/asd_probe.py clip.mp4 START DUR [--crop x0,y0,x1,y1]
                                          [--audio other.wav] [--expect answer|ignore]

The picture goes out as kind-5 camera frames (640x360 JPEG, 15 fps) and the sound
as kind-3 mic frames (48 kHz int16, 20 ms) on /render-stream/mic-inject, in real
time and in step, then 1.5 s of silence with the picture still running. The
server's active-speaker check (src/kehai/active_speaker.py) sees the faces in the
picture move — or not — with the voice. The reply belongs to the render tab's
session, so the verdict is read from the server log ([addressee] lines).

--crop keeps only part of the picture (normalised to the frame), scaled back to
640x360: e.g. only a person who is listening while someone off-picture talks —
the TV case — which must be ignored. --audio replaces the clip's sound.
--before Q.wav --gap S: first ask Q (sound only, no picture) so she is answering,
then S seconds later play the clip over her reply: a barge-in test. Expect
"answer" = she stops for it, "ignore" = she talks on.

Test clip: TaoRuijie/TalkNet-ASD demo/001.avi (public). Its 0-20 s is one person
on the right talking to camera, one on the left listening.
"""

import asyncio
import fractions
import re
import sys
import time

import av
import numpy as np
import websockets

HOST = "127.0.0.1:12394"
INJ = f"ws://{HOST}/render-stream/mic-inject"
LOG = "logs/server.log"


def load(path: str, start: float, dur: float, crop, audio_path: str | None):
    c = av.open(path)
    frames = []
    for f in c.decode(video=0):
        t = float(f.pts * f.time_base)
        if t < start:
            continue
        if t >= start + dur:
            break
        img = f.to_ndarray(format="rgb24")
        if crop:
            h, w = img.shape[:2]
            x0, y0, x1, y1 = crop
            img = img[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]
        frames.append((t - start, av.VideoFrame.from_ndarray(np.ascontiguousarray(img), format="rgb24").reformat(width=640, height=360)))
    src = av.open(audio_path or path)
    rs = av.AudioResampler(format="s16", layout="mono", rate=48000)
    pcm = []
    for f in src.decode(audio=0):
        for g in rs.resample(f):
            pcm.append(g.to_ndarray().reshape(-1))
    a = np.concatenate(pcm).astype(np.float32) / 32768
    off = 0 if audio_path else int(start * 48000)
    a = a[off:off + int(dur * 48000)]
    a = a * (0.9 / (np.max(np.abs(a)) or 1.0))
    return frames, a


async def run(frames, audio, before=None, gap: float = 2.5) -> float:
    enc = av.CodecContext.create("mjpeg", "w")
    enc.width, enc.height, enc.pix_fmt = 640, 360, "yuvj420p"
    enc.time_base = fractions.Fraction(1, 25)
    jpgs = [(t, bytes(enc.encode(f.reformat(format="yuvj420p"))[0])) for t, f in frames]
    tail = np.zeros(int(1.5 * 48000), np.float32)
    x = np.concatenate([audio, tail])
    async with websockets.connect(INJ, max_size=None) as inj:
        if before is not None:
            q = np.concatenate([before, np.zeros(int(gap * 48000), np.float32)])
            tq = time.monotonic()
            for i in range(len(q) // 960):
                await inj.send(bytes([3]) + (np.clip(q[i * 960:(i + 1) * 960], -1, 1) * 32767).astype(np.int16).tobytes())
                d = tq + (i + 1) * 960 / 48000 - time.monotonic()
                if d > 0:
                    await asyncio.sleep(d)
        # a second of picture first: the faces are found before the voice starts
        t0 = time.monotonic() + 1.0
        for k in range(15):
            await inj.send(bytes([5]) + jpgs[0][1] + bytes([k % 7 + 1]))
            await asyncio.sleep(1 / 15)
        n = len(x) // 960
        next_cam, cam_i = 0.0, 0
        for i in range(n):
            now = (i * 960) / 48000
            while now >= next_cam:
                # 15 fps, from the clip while it lasts, then its last frame
                j = min(range(len(jpgs)), key=lambda q: abs(jpgs[q][0] - next_cam)) if next_cam < jpgs[-1][0] else len(jpgs) - 1
                await inj.send(bytes([5]) + jpgs[j][1] + bytes([cam_i % 7 + 1]))
                cam_i += 1
                next_cam += 1 / 15
            pcm = (np.clip(x[i * 960:(i + 1) * 960], -1, 1) * 32767).astype(np.int16).tobytes()
            await inj.send(bytes([3]) + pcm)
            delay = t0 + (i + 1) * 960 / 48000 - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
        # keep the hook (and the camera) going while she decides
        end = time.monotonic() + 6.0
        while time.monotonic() < end:
            await inj.send(bytes([3]) + np.zeros(960, np.int16).tobytes())
            await asyncio.sleep(0.02)
    return t0


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opt = {sys.argv[i]: sys.argv[i + 1] for i in range(1, len(sys.argv) - 1) if sys.argv[i].startswith("--")}
    args = [a for a in args if a not in opt.values()]
    if len(args) < 3:
        print(__doc__)
        return 2
    path, start, dur = args[0], float(args[1]), float(args[2])
    crop = tuple(float(v) for v in opt["--crop"].split(",")) if "--crop" in opt else None
    expect = opt.get("--expect")
    frames, audio = load(path, start, dur, crop, opt.get("--audio"))
    before = None
    if "--before" in opt:
        import soundfile as sf
        from scipy.signal import resample_poly

        b, sr = sf.read(opt["--before"], dtype="float32", always_2d=False)
        b = b.mean(axis=1) if b.ndim > 1 else b
        b = resample_poly(b, 48000, sr).astype(np.float32)
        before = b * (0.9 / (np.max(np.abs(b)) or 1.0))
    log_from = sum(1 for _ in open(LOG, errors="replace"))
    print(f">>> {path} {start:.1f}+{dur:.1f}s crop={crop} audio={opt.get('--audio') or 'own'}: {len(frames)} frames")
    asyncio.run(run(frames, audio, before, float(opt.get("--gap", 2.5))))
    lines = open(LOG, errors="replace").read().splitlines()[log_from:]
    took = None
    for ln in lines:
        m = re.search(r"\[(addressee|asd|spec|turn|server-vad)\].*", ln)
        if m and any(k in ln for k in ("[addressee]", "[asd]", "FAST HIT", "words UNFINISHED", "words finished", "rapid follow-up", "interrupted", "[spec] skip")):
            print("  " + m.group(0)[:300])
            if before is not None:
                if "barge-in confirmed" in ln or "she stops" in ln:
                    took = True
                elif "no barge-in" in ln or "[spec] skip: not to her" in ln:
                    took = False if took is None else took
            elif "FAST HIT" in ln or re.search(r"\[addressee\] to her \(", ln):
                took = True
            elif re.search(r"not to her .*(dropped|ignored)", ln):
                took = False
    print(f"  => {'TAKEN' if took else 'IGNORED' if took is False else 'no verdict seen'}")
    if expect:
        ok = (expect == "answer") == bool(took)
        print("RESULT:", "PASS" if ok else "FAIL")
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
