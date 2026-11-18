"""Where does the voice come from? Records only — nothing decides on it yet.

The small PROTO (Proto M2, from 2026-11-09) has a camera with two microphones
("ProtoCam1 Analog Stereo"): opened raw, the left and right channels differ
(correlation 0.67-0.87; the first PROTO's were sample-identical). A voice from
straight ahead reaches both at once; a voice from the side reaches one up to
~12 samples (0.25 ms at 48 kHz) earlier. In the first recordings the loud
speech in front sat at +0.2..+2 samples and the quieter sounds of the room at
+-9..12.

The viewer sends that raw stereo beside its normal (processed, mono) capture
as kind 8; each block gets a lag by GCC-PHAT (300-6000 Hz) and a level. Per
turn the [direction] log line gives the lag of the voiced blocks, so that
"in front" can be set from real turns before anything acts on it.

VOICE_DIRECTION=0 turns it off.
"""

from __future__ import annotations

import os
import time
from collections import deque

import numpy as np

ENABLED = os.getenv("VOICE_DIRECTION", "1") == "1"
RATE = 48000
MAX_LAG = 24  # samples either way
FRONT = float(os.getenv("VOICE_FRONT_LAG", "4"))  # |lag| at or under this counts as in front (provisional)
SHE = lambda: False  # noqa: E731 — set by websocket_handler: is her voice playing now
_blocks: deque = deque(maxlen=int(120 * RATE / 4096))  # (t, lag, level dBFS, peak sharpness, she speaks)
_band = None
_same = 0.0  # share of recent blocks whose two channels are one signal


def feed(raw: bytes) -> None:
    """One block of interleaved int16 stereo at 48 kHz, as it arrived."""
    global _band
    if not ENABLED or len(raw) < 4 * 1024:
        return
    x = np.frombuffer(raw[: len(raw) // 4 * 4], np.int16).reshape(-1, 2).astype(np.float32) / 32768.0
    left, right = x[:, 0] - x[:, 0].mean(), x[:, 1] - x[:, 1].mean()
    n = 2 * len(left)
    spec = np.fft.rfft(left, n) * np.conj(np.fft.rfft(right, n))
    if _band is None or len(_band) != len(spec):
        f = np.fft.rfftfreq(n, 1 / RATE)
        _band = (f >= 300) & (f <= 6000)
    spec = np.where(_band, spec / (np.abs(spec) + 1e-12), 0)
    cc = np.fft.irfft(spec, n)
    cc = np.concatenate([cc[-MAX_LAG:], cc[: MAX_LAG + 1]])
    k = int(np.argmax(cc))
    lag = float(k - MAX_LAG)
    if 0 < k < len(cc) - 1:  # parabola through the peak: a fraction of a sample
        a, b, c = cc[k - 1], cc[k], cc[k + 1]
        if a - 2 * b + c:
            lag += 0.5 * float(a - c) / float(a - 2 * b + c)
    sharp = float(cc[k] / (np.std(cc) + 1e-9))
    db = float(20 * np.log10(np.sqrt(np.mean(left**2)) + 1e-9))
    # One microphone presented as two channels (the first PROTO's "HD Audio
    # Analog Stereo": left == right to the sample) would read as lag 0 — "in
    # front" — for every sound. Such blocks say nothing about direction.
    global _same
    side = float(20 * np.log10(np.sqrt(np.mean((left - right) ** 2)) + 1e-9))
    # Digital silence (both channels 0, -180 dBFS) says neither: counted as "two
    # signals" it held _same under 0.5 on the first PROTO, whose stream idles at
    # exact zero — every voice then read lag 0, "100 % in front" (2026-11-12 23:20).
    if db > -100:
        # 0.1: a 1 s utterance (12 blocks) settles it; real stereo never reads
        # 40 dB of L-R under the signal, so a fast flip cannot misfire there.
        _same = 0.9 * _same + 0.1 * (1.0 if side < db - 40 else 0.0)
    _blocks.append((time.monotonic(), lag, db, sharp, bool(SHE())))


def features(t0: float, t1: float) -> dict | None:
    if _same > 0.5:
        return None  # not two microphones
    rows = [b for b in _blocks if t0 <= b[0] <= t1 and not b[4]]  # her own voice is not the visitor's
    if len(rows) < 2:
        return None
    floor = float(np.percentile([b[2] for b in _blocks], 20))
    # The voice that opened the turn is the loudest thing in it: its blocks are
    # those within 12 dB of the turn's loudest. (Was "6 dB over the room's
    # floor": in a loud room — floor -43 dBFS at 20:03 — a visitor at -38 had
    # no block over it, no verdict, and the lip rule decided again.)
    top = max(b[2] for b in rows)
    if top < floor + 6:
        return None  # nothing over the room's own level: no voice to place (a lag read from noise is no verdict)
    loud = [b for b in rows if b[2] >= top - 12]
    lags = np.array([b[1] for b in loud])
    w = np.array([10 ** (b[2] / 20) for b in loud])
    return {
        "n": len(loud),
        "median": float(np.median(lags)),
        "front": float(np.mean(np.abs(lags) <= FRONT)),
        "front_w": float(np.sum(w[np.abs(lags) <= FRONT]) / (np.sum(w) + 1e-12)),
        "db": float(np.median([b[2] for b in loud])),
        "floor": floor,
        "lags": [round(float(v), 1) for v in lags[:40]],
    }


def line(t0: float, t1: float) -> str:
    x = features(t0, t1)
    if x is None:
        n = sum(1 for b in _blocks if t0 <= b[0] <= t1)
        why = "one microphone on two channels" if _same > 0.5 else "she was speaking, or nothing rose over the room's level" if n >= 2 else "the stereo stream is not arriving" if not _blocks or time.monotonic() - _blocks[-1][0] > 1.0 else "too short"
        return f"no verdict for this turn ({n} blocks in it: {why})"
    return (
        f"{x['front']:.0%} of {x['n']} voiced blocks in front (|lag| <= {FRONT:g}), {x['front_w']:.0%} by loudness, "
        f"median lag {x['median']:+.1f}, level {x['db']:.0f} dBFS (floor {x['floor']:.0f}) | lags {x['lags']}"
    )


def last_front_voice(t_from: float, t_turn_end: float | None = None) -> tuple[bool, float | None]:
    """(the stereo stream is alive, when a voice from in front was last heard
    since t_from — her own voice left out). For the unfinished-turn hold: has
    the visitor gone quiet, or are they starting again under the onset gate?

    A voice from in front is two blocks in a row from there, not more than
    15 dB under the turn's own loudest: in a room where others talk, 10-30% of
    THEIR blocks read as "in front" one at a time (2026-11-09 21:29-22:49, turns
    with nobody near), and a single such block every second kept the hold from
    ever ending early there."""
    now = time.monotonic()
    if not _blocks or now - _blocks[-1][0] > 1.0 or _same > 0.5:
        return False, None
    floor = float(np.percentile([b[2] for b in _blocks], 20))
    rows = [b for b in _blocks if b[0] >= t_from]
    turn = [b[2] for b in rows if not b[4] and (t_turn_end is None or b[0] <= t_turn_end)]
    thr = max(floor + 4, (max(turn) - 15) if turn else -200.0)
    ok = [(not b[4]) and abs(b[1]) <= FRONT and b[2] >= thr for b in rows]
    for i in range(len(rows) - 1, 0, -1):
        if ok[i] and ok[i - 1]:
            return True, rows[i][0]
    return True, None


def she_line(since_s: float = 60.0) -> str | None:
    """Where her own voice (the PROTO's speaker) reads on the two microphones —
    log only, to learn whether a voice cutting in over her could be told from
    it at all. None when there is nothing to report."""
    now = time.monotonic()
    rows = [b for b in _blocks if b[4] and now - b[0] <= since_s]
    if len(rows) < 10 or _same > 0.5:
        return None
    lags = np.array([b[1] for b in rows])
    return (f"{len(rows)} blocks while she spoke: median lag {np.median(lags):+.1f} "
            f"(p10 {np.percentile(lags, 10):+.1f}, p90 {np.percentile(lags, 90):+.1f}), "
            f"{np.mean(np.abs(lags) <= FRONT):.0%} within |lag| <= {FRONT:g}, level {np.median([b[2] for b in rows]):.0f} dBFS")

