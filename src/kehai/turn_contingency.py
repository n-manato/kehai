"""Does this voice take turns with her? Records only — nothing decides on it yet.

A person talking to her waits while she speaks and answers within a second or
two of her finishing; the robot next to the PROTO and the TV talk on regardless
— over her replies, and long before anyone walked up. That is told from timing
alone: no words, no voice quality (the PROTO's microphone is cut at 4 kHz), so
it holds for any visitor in any language.

Per turn (the [contingency] log line):
  gap      — seconds from the end of her last speech to this voice's start
  over her — share of her last speaking time (20 s back) with a voice over it
  before   — share of the 6 s before this voice began (her own speech left
             out) that already had a voice: a talker that never stops
  barge    — share of this utterance spoken while she was speaking
  step     — this voice's level minus the level of the voice in the 6 s before:
             a visitor starting to talk over a running loudspeaker adds to it

TURN_CONTINGENCY=0 turns it off.
"""

from __future__ import annotations

import os
import time
from collections import deque

ENABLED = os.getenv("TURN_CONTINGENCY", "1") == "1"
SHE = lambda: False  # noqa: E731 — set by websocket_handler: is her voice playing now
ALONE = lambda: False  # noqa: E731 — set by websocket_handler: the camera works and nobody is near
_frames: deque = deque(maxlen=int(90 / 0.032))  # (t, voice, she) per 32 ms VAD frame

# The room's own voices (2026-11-09, the small PROTO is carried from place to
# place, so a fixed onset level cannot be right everywhere). Whatever speaks
# while nobody stands in front of her and she is silent is the room: the robot
# next door, a TV, people talking further off. The onset gate sits just above
# the loud end of that (95th percentile of the last 10 minutes + ROOM_MARGIN),
# within ROOM_MIN..ROOM_MAX; in a room with no such voices it rests at ROOM_MIN.
# VAD_ROOM_GATE=0 keeps the gate at the mic panel's value.
ROOM_GATE = os.getenv("VAD_ROOM_GATE", "1") == "1"
ROOM_MIN = float(os.getenv("VAD_ROOM_MIN_DB", "66"))
ROOM_MAX = float(os.getenv("VAD_ROOM_MAX_DB", "72"))
ROOM_MARGIN = float(os.getenv("VAD_ROOM_MARGIN_DB", "1.5"))
ROOM_KEEP_S = 600.0
_room: deque = deque()  # (t, db) of voice frames heard with nobody in front


def frame(prob: float, db: float, prob_min: float, db_min: float) -> None:
    now = time.monotonic()
    voice = prob >= prob_min and db >= db_min
    she = bool(SHE())
    if ENABLED:
        _frames.append((now, voice, she, float(db)))
    if ROOM_GATE and voice and not she and ALONE():
        _room.append((now, float(db)))
    while _room and now - _room[0][0] > ROOM_KEEP_S:
        _room.popleft()


def room_gate() -> tuple[float, float | None, int]:
    """(onset gate, the room voices' loud end or None, frames it rests on)."""
    if len(_room) < 100:  # ~3 s of voice: too little to call it the room's
        return ROOM_MIN, None, len(_room)
    v = sorted(d for _, d in _room)
    top = v[int(0.95 * (len(v) - 1))]
    return min(max(ROOM_MIN, top + ROOM_MARGIN), ROOM_MAX), top, len(v)


def _share(rows) -> float | None:
    rows = list(rows)
    return sum(1 for r in rows if r[1]) / len(rows) if len(rows) >= 15 else None


def features(t_on: float, t_end: float) -> dict:
    fr = list(_frames)
    she_before = [f[0] for f in fr if f[2] and f[0] < t_on]
    gap = t_on - she_before[-1] if she_before and t_on - she_before[-1] < 60 else None
    lvl = lambda rows: (lambda v: sum(v) / len(v) if len(v) >= 15 else None)([r[3] for r in rows if r[1] and r[3] > 0])  # noqa: E731
    now_db = lvl(f for f in fr if t_on <= f[0] <= t_end)
    was_db = lvl(f for f in fr if not f[2] and t_on - 6.3 <= f[0] < t_on - 0.3)
    return {
        "step": now_db - was_db if now_db is not None and was_db is not None else None,
        "gap": gap,
        "over_her": _share(f for f in fr if f[2] and t_on - 20 <= f[0] < t_on),
        "before": _share(f for f in fr if not f[2] and t_on - 6.3 <= f[0] < t_on - 0.3),
        "barge": (lambda u: sum(1 for f in u if f[2]) / len(u) if u else None)([f for f in fr if t_on <= f[0] <= t_end]),
    }


# A busy room: voices ran over this share of her own recent speech. Quiet days
# (2026-11-09, one visitor at a time) read 0 % at the median and 22 % at the 90th
# percentile; the school visit of 2026-11-13 read 56 % at the median and 81 % of
# its turns were over 40 %. There a voice over her reply is the room, not someone
# cutting in — and nothing on the one microphone tells the two apart.
BUSY_OVER_HER = float(os.getenv("TURN_BUSY_OVER_HER", "0.4") or 0)  # 0 = never busy
_BUSY_MIN_FRAMES = 62  # 2 s of her speech in the last 90 s, or there is nothing to go by


def busy() -> tuple[bool, float | None]:
    """(is the room talking over her, the share of her recent speech with a voice on it)."""
    if not ENABLED or BUSY_OVER_HER <= 0:
        return False, None
    rows = [f for f in _frames if f[2]]
    if len(rows) < _BUSY_MIN_FRAMES:
        return False, None
    share = sum(1 for f in rows if f[1]) / len(rows)
    if share >= BUSY_OVER_HER:
        global _busy_at
        _busy_at = time.monotonic()
    return share >= BUSY_OVER_HER, share


_busy_at = 0.0
BUSY_HOLD_S = float(os.getenv("TURN_BUSY_HOLD_S", "300") or 0)


def busy_room() -> bool:
    """The room was busy (busy()) now or within BUSY_HOLD_S: the state outlasts
    the 90 s of frames, because it can only be measured while she speaks."""
    return busy()[0] or (_busy_at > 0 and time.monotonic() - _busy_at < BUSY_HOLD_S)


def agreement(a: str, b: str) -> float:
    """Word overlap (F1) of two transcripts of one utterance, 0..1."""
    import re
    from collections import Counter

    wa, wb = (re.sub(r"[^a-z0-9' ]", " ", x.lower()).split() for x in (a, b))
    if not wa or not wb:
        return 0.0
    return 2 * sum((Counter(wa) & Counter(wb)).values()) / (len(wa) + len(wb))


def line(t_on: float, t_end: float) -> str:
    x = features(t_on, t_end)
    pct = lambda v: "n/a" if v is None else f"{v:.0%}"  # noqa: E731
    gap = "n/a" if x["gap"] is None else f"{x['gap']:+.1f} s"
    step = "n/a" if x["step"] is None else f"{x['step']:+.1f} dB"
    return f"level step {step} over the voice before, gap {gap} after her speech, voice over her {pct(x['over_her'])}, voice in the 6 s before {pct(x['before'])}, barge {pct(x['barge'])}"
