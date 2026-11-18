"""Debug capture of the voice path, to find WHERE speech gets damaged.

Speech from the PROTO passes four stages before whisper sees it (viewer mic ->
render tab -> server VAD segment -> ASR), and a misheard word could come from
any of them. While ``logs/asr_debug/ENABLED`` exists this records the same
speech at three points so they can be transcribed side by side:

  viewer.rec    the PROTO's mic as it reached the server (48 kHz int16 mono)
  tab.rec       what the render tab streamed to the server VAD (16 kHz)
  seg-*.wav     each segment handed to ASR; its transcript goes to segments.log

``*.rec`` files are records of [f64 wall time][u32 byte length][payload]. They
hold the voices of whoever stands at the PROTO: enable only while testing, then
delete the folder. Off (no flag file) it costs one stat() per second.
"""

from __future__ import annotations

import os
import struct
import time
import wave
from pathlib import Path

DIR = Path(__file__).resolve().parents[3] / "logs" / "asr_debug"
_flag_checked = 0.0
_flag_on = False


def on() -> bool:
    global _flag_checked, _flag_on
    now = time.monotonic()
    if now - _flag_checked > 1.0:
        _flag_checked = now
        _flag_on = (DIR / "ENABLED").exists()
    return _flag_on


def record(name: str, payload: bytes) -> None:
    if not on() or not payload:
        return
    with open(DIR / f"{name}.rec", "ab") as f:
        f.write(struct.pack("<dI", time.time(), len(payload)) + payload)


def segment(pcm_int16: bytes) -> str | None:
    """Save one 16 kHz segment; returns its file name (None when off)."""
    if not on():
        return None
    name = f"seg-{time.time():.3f}.wav"
    with wave.open(str(DIR / name), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(pcm_int16)
    return name


def transcript(name: str, text: str) -> None:
    with open(DIR / "segments.log", "a", encoding="utf-8") as f:
        f.write(f"{name}\t{text}\n")


def note(text: str) -> None:
    """A line in segments.log (turn transcripts, so segments can be grouped)."""
    if on():
        with open(DIR / "segments.log", "a", encoding="utf-8") as f:
            f.write(f"{time.time():.3f}\t{text}\n")


os.makedirs(DIR, exist_ok=True)
