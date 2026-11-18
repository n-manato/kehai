"""Backchannel (相槌): a short acknowledgment the moment a turn ends.

The felt gap after you stop talking is endpoint + LLM + TTS first chunk. A
voice that answers "Mm-hm." straight away reads as listening, and makes the
real reply that follows feel immediate. The clip is synthesized ONCE per
character, held in memory as base64, and sent to the client as a `backchannel`
message — over the WebSocket, so it reaches the render-host page too (the
earlier version pushed onto the WebRTC track only, and was Japanese-only).

SAME VOICE AS THE REPLIES — i.e. the streaming backend tts_manager uses for
them (GSV_STREAM=1: the GSV fine-tune with the character's own reference clip,
Qwen3-TTS in omni mode, ElevenLabs), not the engine's non-streaming route.
That route (openai_tts -> gateway /v1/audio/speech) synthesizes with the
gateway's FIXED reference clip and Japanese prompt, so each reply opened with
an "Mm-hm." in a voice that was not hers and then continued in hers — heard
on the PROTO as "she starts in someone else's voice" (2026-11-03). The GSV
weight swap is awaited first: priming at connect used to outrun it and bake
the previous character's fine-tune into the clips.

History: tried once and removed by request ("the extra utterances weren't
wanted"); re-enabled on request 2026-10-22, so it stays opt-in — BACKCHANNEL=1.
Rate-limited to one per _MIN_INTERVAL_S so a merged follow-up turn can't stack
two acknowledgments, and the first turn after start only primes the cache.
"""

from __future__ import annotations

import asyncio
import base64
import io
import os
import random
import time
import wave
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from loguru import logger

PHRASES = {
    "en": ["Mm-hm.", "Okay.", "Hmm."],
    "ja": ["うん。", "はい。", "ええ。"],
}
_MIN_INTERVAL_S = 4.0

_clips: Dict[str, List[str]] = {}  # conf_uid -> base64 audio files
_priming: Dict[str, asyncio.Task] = {}
_last_played = 0.0


def enabled() -> bool:
    return os.getenv("BACKCHANNEL", "0") == "1"


def _trim_wav(data: bytes, keep_ms: int = 30) -> bytes:
    """Cut the silence TTS bakes around a phrase. A clip that runs 0.7 s instead
    of 0.3 s is still playing when the reply's first words arrive."""
    try:
        with wave.open(io.BytesIO(data)) as w:
            params, frames = w.getparams(), w.readframes(w.getnframes())
        if params.sampwidth != 2:
            return data
        pcm = np.frombuffer(frames, dtype=np.int16).reshape(-1, params.nchannels)
        loud = np.where(np.abs(pcm.astype(np.int32)).max(axis=1) > 300)[0]
        if len(loud) == 0:
            return data
        keep = int(params.framerate * keep_ms / 1000)
        pcm = pcm[max(0, loud[0] - keep): loud[-1] + keep]
        out = io.BytesIO()
        with wave.open(out, "wb") as w:
            w.setparams(params)
            w.writeframes(pcm.tobytes())
        return out.getvalue()
    except Exception:  # noqa: BLE001 — not WAV: send as synthesized
        return data


def _pcm_to_wav(pcm: bytes, rate: int) -> bytes:
    out = io.BytesIO()
    with wave.open(out, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)
    return out.getvalue()


async def _synthesize(context, phrase: str) -> bytes:
    """WAV bytes of `phrase` in the character's REPLY voice (module docstring).
    Streaming off, or the backend yielding nothing: the engine, as before."""
    if os.getenv("GSV_STREAM", "0") == "1" and hasattr(context, "gsv_stream_config"):
        from ..conversations.tts_manager import (
            PCM_SAMPLE_RATE,
            _get_stream_client,
            stream_backend,
        )
        from ..tts.gsv_stream import GSVStreamConfig

        cfg = context.gsv_stream_config()
        fn, where = stream_backend(cfg)
        if isinstance(cfg, GSVStreamConfig):
            await context.ensure_gsv_weights()
        pcm = b"".join([c async for c in fn(phrase, cfg, client=_get_stream_client())])
        if pcm:
            return _pcm_to_wav(pcm, PCM_SAMPLE_RATE)
        logger.warning(f"[backchannel] {where} gave no audio for {phrase!r}; using the engine")
    path = await context.tts_engine.async_generate_audio(
        phrase, file_name_no_ext=f"backchannel_{abs(hash(phrase))}"
    )
    data = Path(path).read_bytes()
    Path(path).unlink(missing_ok=True)
    return data


async def _prime(conf_uid: str, context, lang: str) -> None:
    where = "engine"
    if os.getenv("GSV_STREAM", "0") == "1" and hasattr(context, "gsv_stream_config"):
        from ..conversations.tts_manager import stream_backend

        where = stream_backend(context.gsv_stream_config())[1]
    clips = []
    for phrase in PHRASES.get(lang, PHRASES["en"]):
        try:
            data = _trim_wav(await _synthesize(context, phrase))
            if len(data) > 2000:
                clips.append(base64.b64encode(data).decode())
        except Exception as e:  # noqa: BLE001 — best-effort
            logger.warning(f"[backchannel] could not synthesize {phrase!r}: {e!r}")
    if clips:
        _clips[conf_uid] = clips
        logger.info(f"[backchannel] {len(clips)} clip(s) ready for {conf_uid} via {where}")


def prime(context) -> None:
    """Synthesize this character's clips in the background now, while nobody is
    talking. Left to the first turn that wants one, the three clips went to the
    one GSV worker at the same moment as that turn's reply, which queued behind
    them: first audio 3.1 s after the endpoint (2026-10-23). Called when a
    client's conversation is set up; a no-op once the clips exist."""
    if not enabled():
        return
    conf_uid = context.character_config.conf_uid
    if _clips.get(conf_uid):
        return
    task = _priming.get(conf_uid)
    if task is not None and not task.done():
        return
    lang = getattr(context.character_config, "language", "en") or "en"
    try:
        _priming[conf_uid] = asyncio.get_running_loop().create_task(
            _prime(conf_uid, context, lang)
        )
    except RuntimeError:
        pass  # no loop (sync context) — the next get_clip primes instead


async def get_clip(context) -> Optional[str]:
    """A base64 audio clip in the character's voice, or None (disabled,
    still priming, or rate-limited)."""
    global _last_played
    if not enabled():
        return None
    conf_uid = context.character_config.conf_uid
    clips = _clips.get(conf_uid)
    if not clips:
        prime(context)
        return None
    now = time.monotonic()
    if now - _last_played < _MIN_INTERVAL_S:
        return None
    _last_played = now
    return random.choice(clips)
