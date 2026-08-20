"""Pre-synthesized backchannel (相槌): instant voice onset.

The felt response gap is dominated by fixed costs (endpoint + LLM + TTS first
chunk ≈ 1.0-1.3s). ChatGPT-Voice masks this with an immediate vocal
acknowledgment. Here: a few short Rem-voice phrases are synthesized ONCE via
the normal GSV streaming route, cached as raw 32k PCM in memory (and on disk
under cache/backchannel/), and pushed straight onto the WebRTC track the moment
a user turn is finalized — zero GSV round trip, voice within ~0.3s.

Disable with BACKCHANNEL=0. The cache primes lazily in the background on first
use, so the very first turn after a server start has no backchannel.
"""

from __future__ import annotations

import asyncio
import os
import random
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from loguru import logger

from ..tts.gsv_stream import GSVStreamConfig, stream_sentence_pcm

PHRASES = ["うん!", "はい!", "ええ。"]

_CACHE_DIR = Path("cache/backchannel")
_pcm: Dict[str, bytes] = {}
_prime_task: Optional[asyncio.Task] = None
# Rate limit: a rapid-follow-up merge finalizes twice in quick succession and
# played TWO acknowledgments back to back ("うん!…はい!") with the cached
# leading silence between them — heard as the 促音 gap. One per this window.
_MIN_INTERVAL_S = 4.0
_last_played = 0.0


def _trim_silence(pcm: bytes, sr: int = 32000, keep_ms: int = 30) -> bytes:
    """Strip leading/trailing silence baked into the GSV render (the leading
    ~100-150ms silence created an artifact-band gap when the phrase landed
    adjacent to other audio), keeping a small natural margin."""
    d = np.frombuffer(pcm, dtype=np.int16)
    nz = np.where(np.abs(d.astype(np.int32)) > 130)[0]
    if len(nz) == 0:
        return pcm
    keep = int(sr * keep_ms / 1000)
    a = max(0, int(nz[0]) - keep)
    b = min(len(d), int(nz[-1]) + keep)
    a -= a % 2  # even sample alignment for int16 buffers
    return d[a:b].tobytes()


def enabled() -> bool:
    # Default OFF: the user didn't want the extra acknowledgment utterances.
    return os.getenv("BACKCHANNEL", "0") == "1"


async def _prime() -> None:
    cfg = GSVStreamConfig(
        base_url=os.getenv("GSV_STREAM_URL", GSVStreamConfig.base_url)
    )
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for ph in PHRASES:
        if ph in _pcm:
            continue
        f = _CACHE_DIR / f"{abs(hash(ph))}.pcm"
        if f.exists():
            _pcm[ph] = f.read_bytes()
            continue
        try:
            parts = []
            async for chunk in stream_sentence_pcm(ph, cfg):
                parts.append(chunk)
            pcm = _trim_silence(b"".join(parts))
            if len(pcm) > 6400:  # >0.1s — sanity
                _pcm[ph] = pcm
                f.write_bytes(pcm)
                logger.info(
                    f"[backchannel] cached {ph!r} ({len(pcm) / 64000:.2f}s)"
                )
        except Exception as e:  # noqa: BLE001 — backchannel is best-effort
            logger.warning(f"[backchannel] prime failed for {ph!r}: {e!r}")


def kick_prime() -> None:
    """Start the lazy cache fill (idempotent; call from the request path)."""
    global _prime_task
    if not enabled():
        return
    if len(_pcm) >= len(PHRASES):
        return
    if _prime_task is None or _prime_task.done():
        try:
            _prime_task = asyncio.get_running_loop().create_task(_prime())
        except RuntimeError:
            pass


def get_random() -> Optional[bytes]:
    """A random cached acknowledgment PCM, or None if not primed/disabled/
    rate-limited (max one per _MIN_INTERVAL_S — merges finalize twice)."""
    global _last_played
    if not enabled() or not _pcm:
        return None
    now = time.monotonic()
    if now - _last_played < _MIN_INTERVAL_S:
        return None
    _last_played = now
    return _pcm[random.choice(list(_pcm.keys()))]
