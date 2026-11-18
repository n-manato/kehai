"""Who is talking: is a face in front of the PROTO moving its lips WITH the voice?

Why (2026-11-08): whether speech was meant for her was being decided from what
was said (a few-shot "is this to Iris?" judge on the transcript). Measured on
public data that has nothing to do with the PROTO room, that judge passed only
41% of ordinary requests to an assistant (MASSIVE en-US: "should I bring a
jacket tonight", "compose email") — tuned on one day's TV lines and one
visitor's sentences, it did not carry over to other people or other topics. Words
cannot carry it: a TV character, a podcast guest and two visitors chatting say
the same kinds of sentences a visitor says to her. What separates them is the
SOURCE: a visitor talking to her is a face in front of her whose mouth moves
with the voice; the TV, a phone, someone behind the camera is not.

That is active speaker detection, an audio-visual task with public models
trained on thousands of speakers: LR-ASD (asd_model.py, 0.84 M params, MIT;
AVA-ActiveSpeaker mAP 94.5, Columbia F1 96.4 with these TalkSet weights). Its
input, per face track: 112x112 grayscale crops at 25 fps and the 13 MFCCs of the
16 kHz audio at 100 Hz; output a logit per video frame, > 0 = this face is
speaking. CPU, a few ms per second of speech.

Crops: `record()` is given every camera frame with the faces YuNet found
(face_detect.py) and keeps a crop per near face, in the reference geometry
(Columbia_test.py crop_video: a square of 2x the box's half-size around its
centre, extended down/sideways by 40%, padded with 110, resized to 224, centre
112). Faces are tracked across frames by IoU. Higher-resolution crops sent by the
viewer itself (kind 6, render_view.html) replace the ones cut from the 640x360
frame when they arrive — at 2 m a face in the 640-wide frame is only ~25 px.

The verdict (`speaking()`) for the span of an utterance:
  ("speaking", track) — a near face's lips moved with the voice
  ("silent", ...)     — near faces were seen well enough, and none spoke
  (None, why)         — not enough to tell (no camera, no face, too few frames,
                        face too small): the caller decides as it did without it

Thresholds, from the TalkNet demo clip (public, two people at a table; the
speaker vs the listener at 25 fps): with faces of 120-150 px the speaker's mean
logit over 2 s windows was +1.1..+2.7 and the listener's -0.5..-1.1. Shrunk to
the PROTO frame's size the listener drifted up: 36-46 px faces at 15 fps gave the
listener +0.7/+0.9 in 2 of 10 windows, 25-30 px in 7 of 9. So a verdict is given
only on faces of ASD_MIN_FACE_PX (56) or more in the picture they were cut from
(at 55-80 px the speaker was +1.5..+3.1 and the listener at most +0.31; the
PROTO camera, ~100 degrees wide, gives a face ~120/d px at 1920 wide and ~240/d px
at 3840, ~40/d px in the 640-wide frame — so the viewer's full-resolution crops
are what make it work beyond ~1 m), "speaking" at +ASD_SPEAK (0.6) or more,
"silent" at ASD_SILENT (-0.3) or less, nothing in between.

Env: ASD=0 turns it off; ASD_MODEL (default models/asd/finetuning_TalkSet.model);
ASD_SPEAK; ASD_SILENT; ASD_MIN_FACE_PX; ASD_THREADS (4).
"""

from __future__ import annotations

import math
import os
import threading
import time
from collections import deque
from functools import lru_cache

import numpy as np
from loguru import logger

ENABLED = os.getenv("ASD", "1") == "1"
# Face crops are kept (Tracks) for the lip model above OR for the visual-only lip
# movement check (lip_motion.py, LIP=1).
COLLECT = ENABLED or os.getenv("LIP", "0") == "1"
MODEL_PATH = os.getenv("ASD_MODEL", "models/asd/finetuning_TalkSet.model")
SPEAK = float(os.getenv("ASD_SPEAK", "0.6") or 0.6)
SILENT = float(os.getenv("ASD_SILENT", "-0.3") or -0.3)
MIN_FACE_PX = float(os.getenv("ASD_MIN_FACE_PX", "56") or 56)
# Faces further than this are not in a conversation with her (face_detect.Attention.MAX_M).
MAX_M = float(os.getenv("ADDRESSEE_MAX_M", "2.5"))  # metres at face_detect.HFOV_DEG 100
FPS = 25
CROP_SCALE = 0.40  # Columbia_test.py cropScale
KEEP_S = 20.0


# --- MFCC: python_speech_features.mfcc(audio, 16000, numcep=13, winlen=0.025,
# winstep=0.010) with its defaults (nfilt 26, nfft 512, preemph 0.97, ceplifter 22,
# appendEnergy, rectangular window), in NumPy. The model was trained on int16-scale
# samples (scipy wavfile.read), so the audio is scaled to that range first.
def _hz2mel(hz):
    return 2595 * np.log10(1 + hz / 700.0)


def _mel2hz(mel):
    return 700 * (10 ** (mel / 2595.0) - 1)


@lru_cache(maxsize=1)
def _fbank(nfilt=26, nfft=512, sr=16000) -> np.ndarray:
    mel = np.linspace(_hz2mel(0), _hz2mel(sr / 2), nfilt + 2)
    b = np.floor((nfft + 1) * _mel2hz(mel) / sr)
    fb = np.zeros((nfilt, nfft // 2 + 1))
    for j in range(nfilt):
        for i in range(int(b[j]), int(b[j + 1])):
            fb[j, i] = (i - b[j]) / (b[j + 1] - b[j])
        for i in range(int(b[j + 1]), int(b[j + 2])):
            fb[j, i] = (b[j + 2] - i) / (b[j + 2] - b[j + 1])
    return fb


@lru_cache(maxsize=1)
def _dct_lifter(nfilt=26, numcep=13, L=22) -> np.ndarray:
    n = np.arange(nfilt)
    k = np.arange(numcep)[:, None]
    d = np.cos(np.pi * k * (2 * n + 1) / (2 * nfilt)) * np.sqrt(2.0 / nfilt)
    d[0] *= 1 / np.sqrt(2)
    lift = 1 + (L / 2.0) * np.sin(np.pi * np.arange(numcep) / L)
    return (d * lift[:, None]).T  # (nfilt, numcep)


def mfcc(audio16k: np.ndarray) -> np.ndarray:
    """(N, 13) at 100 Hz from float audio in [-1, 1]."""
    x = np.asarray(audio16k, dtype=np.float64) * 32768.0
    x = np.append(x[0], x[1:] - 0.97 * x[:-1])
    flen, step = 400, 160
    n = 1 if len(x) <= flen else 1 + int(math.ceil((len(x) - flen) / step))
    pad = np.zeros((n - 1) * step + flen)
    pad[: len(x)] = x
    idx = np.arange(flen)[None, :] + step * np.arange(n)[:, None]
    frames = pad[idx]
    pspec = (np.abs(np.fft.rfft(frames, 512)) ** 2) / 512
    energy = pspec.sum(1)
    energy[energy == 0] = np.finfo(float).eps
    feat = pspec @ _fbank().T
    feat[feat == 0] = np.finfo(float).eps
    out = np.log(feat) @ _dct_lifter()
    out[:, 0] = np.log(energy)
    return out.astype(np.float32)


# --- face tracks and their crops --------------------------------------------
def _iou(a, b) -> float:
    ix = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / (a[2] * a[3] + b[2] * b[3] - inter + 1e-9)


def crop_face(gray: np.ndarray, box_px) -> tuple[np.ndarray, float]:
    """The reference crop (Columbia_test.py crop_video + evaluate_network) of one
    face from a grayscale frame: (112x112 uint8, face size in px)."""
    x, y, w, h = box_px
    bs = max(w, h) / 2
    my, mx = y + h / 2, x + w / 2
    cs = CROP_SCALE
    bsi = int(bs * (1 + 2 * cs))
    padded = np.pad(gray, bsi, mode="constant", constant_values=110)
    my, mx = my + bsi, mx + bsi
    y0, y1 = int(my - bs), int(my + bs * (1 + 2 * cs))
    x0, x1 = int(mx - bs * (1 + cs)), int(mx + bs * (1 + cs))
    face = padded[max(0, y0):y1, max(0, x0):x1]
    if face.size == 0:
        return np.full((112, 112), 110, np.uint8), 2 * bs
    big = _resize(face, 224, 224)
    return big[56:168, 56:168].copy(), 2 * bs


def _resize(img: np.ndarray, w: int, h: int) -> np.ndarray:
    """Bilinear resize without OpenCV."""
    ih, iw = img.shape
    ys = (np.arange(h) + 0.5) * ih / h - 0.5
    xs = (np.arange(w) + 0.5) * iw / w - 0.5
    y0 = np.clip(np.floor(ys).astype(int), 0, ih - 1)
    x0 = np.clip(np.floor(xs).astype(int), 0, iw - 1)
    y1 = np.clip(y0 + 1, 0, ih - 1)
    x1 = np.clip(x0 + 1, 0, iw - 1)
    wy = np.clip(ys - y0, 0, 1)[:, None]
    wx = np.clip(xs - x0, 0, 1)[None, :]
    f = img.astype(np.float32)
    top = f[y0][:, x0] * (1 - wx) + f[y0][:, x1] * wx
    bot = f[y1][:, x0] * (1 - wx) + f[y1][:, x1] * wx
    return np.clip(top * (1 - wy) + bot * wy, 0, 255).astype(np.uint8)


class _Track:
    __slots__ = ("id", "box", "last", "crops", "hires_until", "facing", "d", "emb", "final_upto")

    def __init__(self, tid: int, box, t: float):
        self.id, self.box, self.last = tid, box, t
        # (time, 112x112 crop, face px, facing, distance m, source)
        self.crops: deque = deque()
        self.hires_until = 0.0
        self.facing, self.d = False, None
        # Visual embeddings on the 25 fps grid (index i = monotonic time i/25,
        # shared by all tracks so one audio encoding serves them all), kept
        # between verdicts: the visual encoder is ~90% of the cost.
        self.emb: dict[int, np.ndarray] = {}
        # Embeddings up to this grid index had their full right context; later
        # ones are provisional (redone on the next pass, used as they are by a
        # verdict that cannot wait — they are the frames after the speech).
        self.final_upto = -1


DT = 1.0 / FPS
CTX = 10  # grid frames of temporal context either side (the encoder sees +-9)


class Tracks:
    """Face tracks over the last KEEP_S seconds, with a crop per frame. One
    camera, process-wide (like face_detect.ATTENTION)."""

    def __init__(self):
        self._lock = threading.Lock()
        self._tracks: list[_Track] = []
        self._next = 1

    def record(self, gray: np.ndarray, faces: list[dict], t: float | None = None) -> None:
        """One camera frame (grayscale, full size) and its YuNet faces
        (face_detect.FaceDetector.detect: box normalised to the frame)."""
        if not COLLECT:
            return
        t = time.monotonic() if t is None else t
        h, w = gray.shape
        with self._lock:
            for f in faces:
                if f.get("d", 99) > MAX_M:
                    continue
                bx = f["box"]
                box = (bx[0] * w, bx[1] * h, bx[2] * w, bx[3] * h)
                tr = max(self._tracks, key=lambda k: _iou(k.box, box), default=None)
                if tr is None or _iou(tr.box, box) < 0.3 or t - tr.last > 1.0:
                    tr = _Track(self._next, box, t)
                    self._next += 1
                    self._tracks.append(tr)
                tr.box, tr.last = box, t
                tr.facing, tr.d = bool(f.get("facing")), f.get("d")
                if t < tr.hires_until:
                    continue  # the viewer is sending this face at full resolution
                crop, px = crop_face(gray, box)
                tr.crops.append((t, crop, px, tr.facing, tr.d, "frame"))
            self._trim(t)

    def boxes(self) -> list[dict]:
        """Near faces now, for the viewer to cut at full resolution."""
        now = time.monotonic()
        with self._lock:
            live = [tr for tr in self._tracks if now - tr.last < 0.5]
            live.sort(key=lambda tr: tr.d if tr.d is not None else 99)
            return [{"id": tr.id, "box": [round(v, 4) for v in _norm_box(tr.box)]} for tr in live[:2]]

    def record_hires(self, t: float, tid: int, crop112: np.ndarray, px: float) -> None:
        """A crop the viewer cut from its full-resolution camera picture (kind 6),
        for the face it was sent as `tid` (boxes())."""
        if not COLLECT:
            return
        with self._lock:
            tr = next((k for k in self._tracks if k.id == tid), None)
            if tr is None:
                return
            if t >= tr.hires_until:
                # switching sources: drop the low-resolution crops still ahead of this one
                while tr.crops and tr.crops[-1][0] >= t - 0.2 and tr.crops[-1][5] == "frame":
                    tr.crops.pop()
            tr.hires_until = t + 0.5
            tr.crops.append((t, crop112, px, tr.facing, tr.d, "viewer"))

    def _trim(self, now: float) -> None:
        for tr in self._tracks:
            while tr.crops and now - tr.crops[0][0] > KEEP_S:
                tr.crops.popleft()
            if tr.emb:
                lo = int((now - KEEP_S) / DT)
                for i in [i for i in tr.emb if i < lo]:
                    del tr.emb[i]
        self._tracks = [tr for tr in self._tracks if now - tr.last < KEEP_S]

    def snapshot(self, t0: float, t1: float) -> list[tuple[_Track, list]]:
        with self._lock:
            return [(tr, [c for c in tr.crops if t0 <= c[0] <= t1]) for tr in self._tracks]


_FRAME_W, _FRAME_H = 640.0, 360.0


def _norm_box(box_px):
    return (box_px[0] / _FRAME_W, box_px[1] / _FRAME_H, box_px[2] / _FRAME_W, box_px[3] / _FRAME_H)


TRACKS = Tracks()


# --- the model -----------------------------------------------------------------
_model = None
_model_lock = threading.Lock()
_failed = False
# The visual encoder exported to ONNX (models/asd/lrasd_visual.onnx, from the same
# weights; torch.onnx.export, opset 17, dynamic length): ONNX Runtime runs it in
# ~93 ms per 3 s of video on 4 threads, torch ~140-190 ms. Checked against torch
# on load; torch is used when the file is missing or disagrees.
VISUAL_ONNX = os.getenv("ASD_VISUAL_ONNX", "models/asd/lrasd_visual.onnx")
_ort = None


def _load():
    global _model, _failed
    if _model is not None or _failed:
        return _model
    with _model_lock:
        if _model is None and not _failed:
            try:
                import torch

                from . import asd_model

                # (KEHAI's only other torch user is the Silero VAD, a tiny model.)
                torch.set_num_threads(int(os.getenv("ASD_THREADS", "4") or 4))
                _model = asd_model.load(MODEL_PATH)
                _load_onnx(_model)
                logger.info(f"[asd] loaded {MODEL_PATH} (visual encoder: {'onnxruntime' if _ort else 'torch'})")
            except Exception as e:  # noqa: BLE001
                _failed = True
                logger.warning(f"[asd] active speaker detection unavailable: {e!r}")
    return _model


def _load_onnx(m) -> None:
    global _ort
    if not os.path.exists(VISUAL_ONNX):
        return
    try:
        import onnxruntime as ort
        import torch

        o = ort.SessionOptions()
        o.intra_op_num_threads = int(os.getenv("ASD_THREADS", "4") or 4)
        # Idle ORT threads spin by default, and starved the torch audio encoder
        # that runs right after (7 -> 42 ms).
        o.add_session_config_entry("session.intra_op.allow_spinning", "0")
        sess = ort.InferenceSession(VISUAL_ONNX, o, providers=["CPUExecutionProvider"])
        x = (np.random.default_rng(0).random((1, 30, 112, 112)) * 255).astype(np.float32)
        with torch.no_grad():
            ref = m.visual(torch.from_numpy(x)).numpy()
        diff = float(np.abs(sess.run(None, {"v": x})[0] - ref).max())
        if diff > 1e-3:
            logger.warning(f"[asd] {VISUAL_ONNX} disagrees with the weights (max diff {diff:.3g}); using torch")
            return
        _ort = sess
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[asd] onnxruntime visual encoder unavailable ({e!r}); using torch")


def _visual(frames: np.ndarray) -> np.ndarray:
    """(T, 112, 112) float32 -> (T, 128)."""
    if _ort is not None:
        return _ort.run(None, {"v": frames[None]})[0][0]
    import torch

    with torch.no_grad():
        return _load().visual(torch.from_numpy(frames).unsqueeze(0))[0].numpy()


def _grid(crops: list, times: np.ndarray) -> np.ndarray:
    """The crop nearest each grid time -> (len(times), 112, 112) uint8."""
    ts = np.array([c[0] for c in crops])
    j = np.clip(np.searchsorted(ts, times), 0, len(ts) - 1)
    jm = np.clip(j - 1, 0, len(ts) - 1)
    j = np.where(np.abs(ts[jm] - times) < np.abs(ts[j] - times), jm, j)
    return np.stack([crops[i][1] for i in j])


def _embed(tr: _Track, crops: list, i0: int, i1: int, right_ok: int) -> None:
    """Visual embeddings for grid frames i0..i1-1 of a track, with CTX frames of
    context each side (right context only up to grid index right_ok)."""

    m = _load()
    if m is None or i1 <= i0 or not crops:
        return
    a, b = i0 - CTX, min(i1 + CTX, right_ok)
    times = np.arange(a, b) * DT
    out = _visual(_grid(crops, times).astype(np.float32))
    for k in range(i0, i1):
        tr.emb[k] = out[k - a]


def prepare(now: float | None = None) -> None:
    """While someone is talking (websocket_handler calls this every ~0.5 s of an
    utterance, in a thread): encode the frames of the near faces that now have
    their full context, so a verdict at the pause has only the last half second
    left to do. (Encoding in chunks WITHOUT the context after each chunk pushed
    a listener's score from +0.16 to +0.72 — over the "speaking" line — on the
    TalkNet clip; with it the result is the same as encoding the whole span.)"""
    if not ENABLED or _load() is None:
        return
    now = time.monotonic() if now is None else now
    for tr, crops in sorted(TRACKS.snapshot(now - 3.0, now), key=lambda u: -(u[1][-1][2] if u[1] else 0))[:3]:
        if not _usable(crops):
            continue
        last = int(crops[-1][0] / DT)
        start = max(tr.final_upto + 1, int((now - 3.0) / DT))
        end = last - CTX  # last frame with CTX real frames after it
        if end - start >= 5:
            _embed(tr, crops, start, end + 1, last + 1)
            tr.final_upto = max(tr.final_upto, end)


def _usable(crops: list) -> bool:
    return len(crops) >= 4 and float(np.median([c[2] for c in crops])) >= MIN_FACE_PX


def _span(start: float, end: float) -> tuple[int, int]:
    return int(math.ceil(start / DT)), int(end / DT)


def _visual_span(tr: _Track, crops: list, i0: int, i1: int) -> np.ndarray:
    """(i1-i0, 128) visual embeddings of a track. Each verdict on it (every pause
    of an utterance, the endpoint, the barge-in probe) reuses what earlier ones
    computed: only new frames, and the last CTX that lacked their right context
    then, are encoded again."""
    right_ok = max(int(crops[-1][0] / DT) + 1, i1)
    need = [i for i in range(i0, i1) if i not in tr.emb or i > tr.final_upto]
    if need:
        _embed(tr, crops, need[0], i1, right_ok)
        tr.final_upto = max(tr.final_upto, min(i1, right_ok - CTX) - 1)
    return np.stack([tr.emb[i] for i in range(i0, i1)])


def _lag_sweep(m, tid: int, v: np.ndarray, i0: int, i1: int, audio16k: np.ndarray, t0: float) -> None:
    """ASD_LAG_SWEEP=1: the face's mean logit with the audio taken d seconds
    EARLIER than the frames' arrival times (d > 0 = the picture arrives later
    than the sound), to find the camera path's delay against the mic path."""
    import torch

    out = []
    for d in (-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0):
        a0 = int(round((i0 * DT - d - t0) * 16000))
        lo = max(0, -a0 // 640 + (1 if -a0 % 640 else 0)) if a0 < 0 else 0
        a0 += lo * 640
        n = min(i1 - i0 - lo, (len(audio16k) - a0 - 400) // 640)
        if n < 15:
            out.append(f"{d:+.1f}:-")
            continue
        mf = mfcc(audio16k[a0:a0 + n * 640 + 400])[: n * 4]
        with torch.no_grad():
            a = m.audio(torch.from_numpy(mf).unsqueeze(0))
            lg = m.backend(a, torch.from_numpy(v[lo:lo + n]).unsqueeze(0)).numpy()
        out.append(f"{d:+.1f}:{float(np.mean(lg)):+.2f}")
    logger.info(f"[asd] lag sweep #{tid} (audio earlier by d s): {' '.join(out)}")


def _dump(tid: int, crops: list, i0: int, i1: int, audio16k: np.ndarray, t0: float) -> None:
    """ASD_DUMP=<dir>: the frames a verdict looked at (every 4th, tiled, PNG) and
    its audio, to see what the lip model saw. Contains faces — delete after use."""
    try:
        import av

        d = os.getenv("ASD_DUMP")
        os.makedirs(d, exist_ok=True)
        fr = _grid(crops, np.arange(i0, i1) * DT)[::4][:16]
        rows = [np.concatenate(list(fr[k:k + 8]) + [np.zeros((112, 112), np.uint8)] * (8 - len(fr[k:k + 8])), 1) for k in range(0, len(fr), 8)]
        img = np.concatenate(rows, 0)
        stamp = time.strftime("%H%M%S")
        c = av.CodecContext.create("png", "w")
        c.width, c.height, c.pix_fmt = img.shape[1], img.shape[0], "gray"
        pk = c.encode(av.VideoFrame.from_ndarray(img, format="gray"))
        open(os.path.join(d, f"{stamp}_t{tid}.png"), "wb").write(b"".join(bytes(p) for p in pk))
        ts = np.array([c_[0] for c_ in crops])
        np.save(os.path.join(d, f"{stamp}_t{tid}_meta.npy"), np.array([i0 * DT - t0, len(audio16k) / 16000, ts[0] - t0, ts[-1] - t0, len(crops)]))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"[asd] dump failed: {e!r}")


def speaking(t0: float, t1: float, audio16k: np.ndarray, why: str = "") -> tuple[str | None, str, float | None]:
    """Did a near face speak the audio of t0..t1 (monotonic; audio16k covers that
    span)? ("speaking" | "silent" | None, why, share of the speaking face's frames
    in which it faced her). ~47 ms of CPU per second of new video per face (4
    threads); call in a thread."""
    if not ENABLED:
        return None, "asd off", None
    if t1 - t0 < 0.4:
        return None, "asd: too short", None
    usable, why_not = [], []
    for tr, crops in TRACKS.snapshot(t0 - 0.5, t1):
        if len(crops) < 0.4 * (t1 - t0) * 10:  # under ~4 frames/s of the span: not seen
            why_not.append(f"#{tr.id} {len(crops)} frames")
            continue
        px = float(np.median([c[2] for c in crops]))
        if px < MIN_FACE_PX:
            why_not.append(f"#{tr.id} {px:.0f} px")
            continue
        usable.append((tr, crops, px))
    if not usable:
        return None, "asd: no face seen well enough" + (f" ({', '.join(why_not)})" if why_not else ""), None
    if _load() is None:
        return None, "asd unavailable", None
    import torch

    m = _load()
    # the nearest (largest) three: each costs ~30 ms per second of new video
    usable.sort(key=lambda u: -u[2])
    tc = time.perf_counter()
    # One audio encoding on the shared grid, sliced per face.
    t_last = t0 + len(audio16k) / 16000
    spans = {}
    for tr, crops, px in usable[:3]:
        i0, i1 = _span(max(t0, crops[0][0]), min(t1, crops[-1][0] + 0.1, t_last - 0.03))
        if i1 - i0 >= 10:
            spans[tr.id] = (i0, i1)
    if not spans:
        return None, "asd: spans too short", None
    I0, I1 = min(v[0] for v in spans.values()), max(v[1] for v in spans.values())
    a0 = int(round((I0 * DT - t0) * 16000))
    aud_all = audio16k[max(0, a0):a0 + (I1 - I0) * 640 + 400]
    if a0 < 0:
        aud_all = np.concatenate([np.zeros(-a0, np.float32), aud_all])
    mf = mfcc(aud_all)
    n_all = min(I1 - I0, len(mf) // 4)
    with torch.no_grad():
        a_all = m.audio(torch.from_numpy(mf[: n_all * 4]).unsqueeze(0))
    best = None
    lines = []
    for tr, crops, px in usable[:3]:
        if tr.id not in spans:
            continue
        i0, i1 = spans[tr.id]
        i1 = min(i1, I0 + n_all)
        if i1 - i0 < 10:
            continue
        v = _visual_span(tr, crops, i0, i1)
        if os.getenv("ASD_DUMP"):
            _dump(tr.id, crops, i0, i1, audio16k, t0)
        if os.getenv("ASD_LAG_SWEEP") == "1":
            _lag_sweep(m, tr.id, v, i0, i1, audio16k, t0)
        with torch.no_grad():
            logits = m.backend(a_all[:, i0 - I0:i1 - I0], torch.from_numpy(v).unsqueeze(0)).numpy()
        aud = aud_all[(i0 - I0) * 640:(i1 - I0) * 640]
        start, end = i0 * DT, i1 * DT
        # The speaking part only: the turn's audio starts before the speech and
        # ends after it; mean over frames whose audio is voiced.
        e = np.array([np.sqrt(np.mean(aud[i * 640:(i + 1) * 640] ** 2) + 1e-12) for i in range(len(logits))])
        voiced = e >= max(e.max() * 0.15, 1e-4)
        lg = logits[voiced] if voiced.sum() >= 4 else logits
        mean = float(np.mean(lg))
        frac = float(np.mean(lg > 0))
        sel = [c for c in crops if start <= c[0] <= end]
        facing = float(np.mean([c[3] for c in sel])) if sel else 0.0
        src = "viewer" if any(c[5] == "viewer" for c in sel) else "frame"
        lines.append(f"#{tr.id} {mean:+.2f} ({frac:.0%} of {len(lg)} voiced frames, {px:.0f} px {src}, facing {facing:.0%})")
        if best is None or mean > best[1]:
            best = (tr.id, mean, facing)
    ms = (time.perf_counter() - tc) * 1000
    if best is None:
        return None, "asd: spans too short", None
    detail = f"{'; '.join(lines)} [{ms:.0f} ms{', ' + why if why else ''}]"
    if best[1] >= SPEAK:
        return "speaking", f"lips moved with the voice: {detail}", best[2]
    if best[1] <= SILENT:
        return "silent", f"no face in front spoke it: {detail}", None
    return None, f"asd undecided: {detail}", None
