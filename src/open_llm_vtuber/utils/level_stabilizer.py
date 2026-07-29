"""Streaming loudness stabiliser for the GSV stream ("しゃべってるときの音量が安定しない").

GPT-SoVITS synthesises a reply FRAGMENT BY FRAGMENT (cut5), and each fragment is
normalised on its own — so the level STEPS at fragment boundaries. Measured on the live
stream: up to 10.2 dB of step INSIDE a single sentence (mean spread 6.5 dB), which is
plainly audible as the voice getting louder/quieter mid-speech.

Fix: follow the voiced level with two envelopes — a SHORT one (~0.5 s, tracks the
current fragment) and a LONG one (~8 s, the character's natural loudness) — and apply
`long / short` as a gain. Fragment offsets, which persist for seconds, are corrected;
syllable-rate dynamics (~0.1-0.2 s) pass through untouched, so the delivery keeps its
expression. The gain slews slowly while voiced and SNAPS during silence (inaudible
there — and fragment boundaries are exactly where the silence is), which is what keeps
it from pumping.

Streaming-safe and latency-free: no lookahead, output length == input length, so TTFA
and every downstream timing (lip-sync, co-speech clocks) are unchanged. int16 mono 32k.
Disable with TTS_LEVEL_STABILIZE=0.
"""

from __future__ import annotations

import numpy as np

SR = 32000
FRAME = 640  # 20 ms
_VOICE_FLOOR = 300.0  # int16 RMS (~0.009 fs) — below this the frame is silence
# Clamp. Measured artefact steps reach ~10dB, so 8 covers nearly all of them. A genuinely
# quiet passage that lasts is NOT fought forever: the 8s follower drifts down to meet it,
# so expression survives while a 1-3s fragment artefact gets corrected.
_MAX_DB = 8.0
_SLEW_DB_PER_S = 5.0  # steady-state: slow enough that it cannot pump
_SLEW_FAST_DB_PER_S = 45.0  # just after a pause: land the new fragment's level at once
_TAU_SHORT = 0.6  # steady-state fragment-level follower
_TAU_SHORT_FAST = 0.06  # right after a pause: lock onto the new fragment fast
_TAU_LONG = 8.0  # the voice's own loudness
_SIL_FRAMES_BOUNDARY = 2  # >=40ms of silence => treat the next audio as a new fragment
_FAST_FRAMES = 10  # ~200ms of fast adaptation after each boundary
_LIMIT = 32000.0  # headroom guard before int16 clip


class LevelStabilizer:
    """Feed int16-LE PCM, get the same PCM with fragment-to-fragment level steps
    ironed out. One instance per turn (state carries across sentences so sentence
    boundaries are smoothed too)."""

    def __init__(self) -> None:
        self._rem = b""  # partial frame carried to the next feed()
        self._short = 0.0
        self._long = 0.0
        self._desired_db = 0.0
        self._gain_db = 0.0
        self._prev_gain = 1.0
        self._dt = FRAME / SR
        self._a_short = 1.0 - float(np.exp(-self._dt / _TAU_SHORT))
        self._a_short_fast = 1.0 - float(np.exp(-self._dt / _TAU_SHORT_FAST))
        self._a_long = 1.0 - float(np.exp(-self._dt / _TAU_LONG))
        self._sil = _SIL_FRAMES_BOUNDARY  # start as if after a pause
        self._fast = _FAST_FRAMES

    def feed(self, pcm: bytes) -> bytes:
        if not pcm:
            return pcm
        buf = self._rem + pcm
        n_frames = len(buf) // (FRAME * 2)
        if n_frames == 0:
            self._rem = buf
            return b""
        used = n_frames * FRAME * 2
        self._rem = buf[used:]
        x = np.frombuffer(buf[:used], dtype="<i2").astype(np.float32).reshape(n_frames, FRAME)

        out = np.empty_like(x)
        for i in range(n_frames):
            f = x[i]
            rms = float(np.sqrt((f * f).mean()))
            if rms > _VOICE_FLOOR:
                # A pause just ended => this is a NEW fragment, possibly at a completely
                # different level. Adapt fast for ~200ms so it is matched from its first
                # syllable (waiting for the slow follower took >1s = the audible slide).
                if self._sil >= _SIL_FRAMES_BOUNDARY:
                    self._fast = _FAST_FRAMES
                self._sil = 0
                fast = self._fast > 0
                if fast:
                    self._fast -= 1
                if self._short <= 0.0:  # very first voiced frame: seed both followers
                    self._short = self._long = rms
                else:
                    self._short += (rms - self._short) * (
                        self._a_short_fast if fast else self._a_short
                    )
                    self._long += (rms - self._long) * self._a_long
                self._desired_db = float(
                    np.clip(20.0 * np.log10(self._long / max(self._short, 1.0)), -_MAX_DB, _MAX_DB)
                )
                step = (_SLEW_FAST_DB_PER_S if fast else _SLEW_DB_PER_S) * self._dt
                d = self._desired_db - self._gain_db
                self._gain_db += float(np.clip(d, -step, step))
            else:
                self._sil += 1
                # Silence: no one can hear the gain move — take the correction now, so
                # the next fragment starts already matched instead of sliding into level.
                self._gain_db = self._desired_db
            g1 = float(10.0 ** (self._gain_db / 20.0))
            # per-sample ramp: a stepped gain would click ("zipper noise")
            ramp = np.linspace(self._prev_gain, g1, FRAME, dtype=np.float32)
            y = f * ramp
            peak = float(np.abs(y).max())
            if peak > _LIMIT:  # keep headroom; only bites on an already-hot fragment
                y *= _LIMIT / peak
            out[i] = y
            self._prev_gain = g1
        return np.clip(out, -32768, 32767).astype("<i2").tobytes()

    def flush(self) -> bytes:
        """Emit the sub-frame remainder (end of turn) — ungained, it is <20 ms."""
        r, self._rem = self._rem, b""
        return r
