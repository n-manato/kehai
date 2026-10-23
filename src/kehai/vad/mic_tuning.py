"""Live microphone tuning for the exhibition floor.

The VAD thresholds live in conf.yaml and are read once at startup, which is fine at
a desk and useless in a hall: the room is only known when you are standing in it,
it changes through the day, and nobody there can edit YAML and restart the server.
So the viewer (render_view.html) has a mic panel, and what it sets lands here:

  * applied to the running VAD on its next 32 ms frame (StateMachine polls VERSION)
  * kept in mic_tuning.json, so a restart keeps the hall's calibration
  * anything never touched falls back to conf.yaml (set_vad_defaults)

Two of the values never reach the server's VAD — gain_db and the browser's
noise_suppression / auto_gain switches act on the viewer's own capture — but they
are stored here too, so one file holds the whole setup and a viewer that reloads
(the PROTO does, after every update) comes back the same.

MONITOR is the other direction: the VAD's live level and its decisions, which the
panel draws while it is open, so a threshold is set against what the VAD actually
hears rather than guessed.
"""

import json
import math
import threading
import time
from collections import deque
from pathlib import Path

from loguru import logger

PATH = Path(__file__).resolve().parents[3] / "mic_tuning.json"

# name -> (kind, min, max). Ranges are what makes sense to offer on a slider, and
# also what a bad file or a bad message is clamped to.
FIELDS: dict[str, tuple[type, float, float]] = {
    "gain_db": (float, -12, 24),
    "noise_suppression": (bool, 0, 1),
    "auto_gain": (bool, 0, 1),
    "prob_threshold": (float, 0.1, 0.95),
    "onset_db": (int, 30, 95),
    "sustain_db": (int, 20, 90),
    "segment_mean_db": (int, 0, 85),
    "onset_ms": (int, 64, 640),
}

# What the viewer captured with before this panel existed.
_defaults: dict = {"gain_db": 0.0, "noise_suppression": True, "auto_gain": True}
_overrides: dict = {}
_lock = threading.Lock()
VERSION = 0


def _clamp(name: str, value):
    kind, lo, hi = FIELDS[name]
    if kind is bool:
        return bool(value)
    v = float(value)
    if not math.isfinite(v):
        raise ValueError(name)
    v = min(hi, max(lo, v))
    return int(round(v)) if kind is int else round(v, 3)


def _load() -> None:
    global _overrides
    try:
        raw = json.loads(PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return
    except Exception as e:
        logger.warning(f"[mic-tuning] ignoring unreadable {PATH.name}: {e!r}")
        return
    _overrides = {}
    for k, v in raw.items():
        if k in FIELDS:
            try:
                _overrides[k] = _clamp(k, v)
            except (TypeError, ValueError):
                pass
    if _overrides:
        logger.info(f"[mic-tuning] loaded {PATH.name}: {_overrides}")


def _save() -> None:
    tmp = PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(_overrides, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(PATH)


def set_vad_defaults(**values) -> None:
    """The conf.yaml values, registered by the VAD when it is built."""
    global VERSION
    with _lock:
        _defaults.update({k: _clamp(k, v) for k, v in values.items() if k in FIELDS})
        VERSION += 1


def effective() -> dict:
    with _lock:
        out = {**_defaults, **_overrides}
    # Sustain above onset would let a turn start that cannot keep itself alive.
    if "onset_db" in out and "sustain_db" in out and out["sustain_db"] > out["onset_db"]:
        out["onset_db"] = out["sustain_db"]
    return out


def snapshot() -> dict:
    with _lock:
        custom = sorted(_overrides)
        defaults = dict(_defaults)
    return {"values": effective(), "defaults": defaults, "custom": custom}


def update(changes: dict) -> dict:
    global VERSION
    with _lock:
        for k, v in (changes or {}).items():
            if k not in FIELDS:
                continue
            try:
                _overrides[k] = _clamp(k, v)
            except (TypeError, ValueError):
                continue
        VERSION += 1
        _save()
    logger.info(f"[mic-tuning] set {changes} -> {effective()}")
    return snapshot()


def reset() -> dict:
    global VERSION
    with _lock:
        _overrides.clear()
        VERSION += 1
        _save()
    logger.info("[mic-tuning] reset to conf.yaml defaults")
    return snapshot()


class _Monitor:
    """Latest VAD frame and a short log of its decisions, for the mic panel."""

    def __init__(self):
        self.prob = 0.0
        self.db = 0.0
        self.state = "idle"
        self.seq = 0
        self.events: deque = deque(maxlen=32)

    def frame(self, prob: float, db: float, state: str) -> None:
        self.prob = float(prob)
        self.db = float(db) if math.isfinite(db) else 0.0
        self.state = state

    def event(self, kind: str, **info) -> None:
        self.seq += 1
        self.events.append({"n": self.seq, "k": kind, "t": time.time(), **info})

    def since(self, n: int) -> list:
        return [e for e in self.events if e["n"] > n]


MONITOR = _Monitor()
_load()
