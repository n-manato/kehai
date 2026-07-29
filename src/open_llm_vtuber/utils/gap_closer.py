"""Streaming mid-word silence closer for the GSV Rem voice.

The fine-tuned RemGSV model STOCHASTICALLY bakes 40-170ms silences into the
middle of words (measured: same text/params, ~1 in 6 renders; with slower
reference clips every render). Acoustically such a gap inside a word reads as a
geminate ("大好きです" -> "大っ好きです" feel) — the residual 促音 the user
kept hearing after every transport-level cause was fixed. Reference-clip swaps
made it WORSE (prosody leak), so the fix is a post-process: close those gaps in
the streamed PCM before it reaches the WebRTC track.

Real geminates must survive: a genuine っ closure is ~60-100ms of silence. The
caller passes the sentence TEXT — if it contains っ/ッ we only close gaps
longer than 110ms; otherwise any 40-170ms mid-word gap is fair game. Gaps are
shrunk (not zeroed) to 24ms with short crossfades so the join stays natural.

Streaming-safe: `feed()` emits audio 220ms behind the newest input (lookahead
to see a gap's far edge); the first 0.4s passes through untouched so TTFA is
unaffected (synthesis outruns playback, so the lag is absorbed by the buffer).
`flush()` drains the tail at sentence end. int16 mono 32k throughout.
"""

from __future__ import annotations

import numpy as np

SR = 32000
# 340 (~0.010 fs): the model's baked gaps are often not DIGITAL silence but a
# low-level noise floor — the strict 200 threshold missed those (tap analysis
# found 40-53ms dips surviving at 0.012 fs while none existed at 0.004 fs).
_SIL_TH = 340
_VOICED_RMS = 500.0  # neighbor RMS above this = voiced context (~0.015 fs)
_MIN_GAP_MS = 40
_MAX_GAP_MS = 190
_SOKUON_MIN_GAP_MS = 110  # text has っ/ッ: only close clearly-too-long gaps
# English: stop consonants (p/t/k) carry a REAL 50-100ms closure inside words —
# the aggressive 40ms floor would eat them. Only close clearly-model-baked gaps.
_EN_MIN_GAP_MS = 110
_KEEP_MS = 24  # closed gaps are shrunk to this, not removed
_FADE_MS = 4
_NEIGHBOR_MS = 50
_HOLDBACK_MS = 220  # lookahead so a gap's far edge is visible before emitting
_PASSTHROUGH_MS = 250  # sentence head: emit immediately (protect TTFA)


def _ms(n: int) -> int:
    return int(SR * n / 1000)


class GapCloser:
    def __init__(self, text: str = "", lang: str = "ja") -> None:
        has_sokuon = ("っ" in text) or ("ッ" in text)
        if lang and lang != "ja":
            min_gap_ms = _EN_MIN_GAP_MS
        elif has_sokuon:
            min_gap_ms = _SOKUON_MIN_GAP_MS
        else:
            min_gap_ms = _MIN_GAP_MS
        self._min_gap = _ms(min_gap_ms)
        self._max_gap = _ms(_MAX_GAP_MS)
        self._buf = np.zeros(0, dtype=np.int16)
        self._emitted = 0
        self.closed = 0  # gaps closed (diagnostics)
        # GSV HTTP chunks split int16 samples across boundaries — carry the odd
        # trailing byte to the next feed (same fix as TTSAudioTrack._inbuf;
        # without it np.frombuffer raises and the sentence DIES after the first
        # odd chunk — "最初の2-3文字しか読まない").
        self._inbuf = b""

    def _close_gaps(self, seg: np.ndarray) -> np.ndarray:
        """Close eligible silence runs fully inside `seg` (caller guarantees
        enough context on both sides)."""
        sil = np.abs(seg.astype(np.int32)) < _SIL_TH
        out_parts = []
        pos = 0
        i = 0
        n = len(seg)
        nb = _ms(_NEIGHBOR_MS)
        keep = _ms(_KEEP_MS)
        fade = _ms(_FADE_MS)
        while i < n:
            if not sil[i]:
                i += 1
                continue
            j = i
            while j < n and sil[j]:
                j += 1
            run = j - i
            # only mid-word runs: voiced on BOTH sides within the neighbor window
            if self._min_gap <= run <= self._max_gap and i >= nb and j + nb <= n:
                pre = seg[i - nb : i].astype(np.float32)
                post = seg[j : j + nb].astype(np.float32)
                if (
                    np.sqrt(np.mean(pre**2)) > _VOICED_RMS
                    and np.sqrt(np.mean(post**2)) > _VOICED_RMS
                ):
                    # emit up to gap start, then a shortened gap with fades
                    out_parts.append(seg[pos:i])
                    mid = np.zeros(keep, dtype=np.int16)
                    # short fade-out of the last samples before the gap and
                    # fade-in after it, to kill any residual click
                    if len(out_parts[-1]) >= fade:
                        tail = out_parts[-1][-fade:].astype(np.float32)
                        out_parts[-1] = out_parts[-1].copy()
                        out_parts[-1][-fade:] = (
                            tail * np.linspace(1.0, 0.6, fade)
                        ).astype(np.int16)
                    out_parts.append(mid)
                    head = seg[j : j + fade].astype(np.float32)
                    faded = (head * np.linspace(0.6, 1.0, fade)).astype(np.int16)
                    out_parts.append(faded)
                    pos = j + fade
                    self.closed += 1
            i = j
        out_parts.append(seg[pos:])
        return np.concatenate(out_parts) if len(out_parts) > 1 else seg[pos:]

    def feed(self, pcm: bytes) -> bytes:
        """Add streamed PCM; return the processed audio that is safe to emit."""
        if pcm:
            data = self._inbuf + pcm
            usable = (len(data) // 2) * 2
            self._inbuf = data[usable:]
            if usable:
                self._buf = np.concatenate(
                    [self._buf, np.frombuffer(data[:usable], dtype=np.int16)]
                )
        # sentence head: pass through untouched so first audio isn't delayed
        if self._emitted < _ms(_PASSTHROUGH_MS):
            take = min(len(self._buf), _ms(_PASSTHROUGH_MS) - self._emitted)
            out = self._buf[:take]
            self._buf = self._buf[take:]
            self._emitted += take
            return out.tobytes()
        hold = _ms(_HOLDBACK_MS)
        if len(self._buf) <= hold:
            return b""
        emit_len = len(self._buf) - hold
        seg = self._close_gaps(self._buf[: emit_len + hold])
        # everything but the trailing holdback (post-processing may have
        # shortened the segment; keep the same absolute holdback from the end)
        cut = max(0, len(seg) - hold)
        out = seg[:cut]
        self._buf = seg[cut:]
        self._emitted += len(out)
        return out.tobytes()

    def flush(self) -> bytes:
        """Sentence end: process and drain whatever is held back."""
        if len(self._buf) == 0:
            return b""
        seg = self._close_gaps(self._buf)
        self._buf = np.zeros(0, dtype=np.int16)
        self._emitted += len(seg)
        return seg.tobytes()
