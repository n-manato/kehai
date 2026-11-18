"""Streaming Qwen3-TTS (vLLM-Omni) : text -> raw PCM chunks, yielded as they arrive.

A drop-in for gsv_stream.stream_sentence_pcm, down to the 32 kHz s16 mono PCM
the rest of the streaming path is built around. The voice is a zero-shot clone
of the character's GPT-SoVITS reference clip (same file, same transcript — see
service_context.gsv_stream_config), so switching the backend changes the engine,
not the character setup.

Why: GPT-SoVITS needs 0.36-1.16 s to its first chunk; vLLM-Omni's Qwen3-TTS
server (`talker_first_audio`, codec streaming) returned its first PCM in
0.11-0.13 s on this box (2026-11-03, RTX 3090, 1.7B Base) at 4x real time.

The server speaks 24 kHz; the stream is upsampled to 32 kHz here (stateful
linear interpolation across chunk borders) so tts_manager, the Opus encoder,
the gesture windows and the WebRTC feed are untouched.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import AsyncIterator

import httpx
import numpy as np
from loguru import logger

PCM_SAMPLE_RATE = 32000
_TIMING = os.getenv("TTS_TIMING", "1") == "1"
SERVER_RATE = 24000

_LANG = {"en": "English", "ja": "Japanese", "zh": "Chinese", "ko": "Korean",
         "de": "German", "fr": "French", "es": "Spanish", "pt": "Portuguese",
         "it": "Italian", "ru": "Russian"}


@dataclass
class Qwen3TTSStreamConfig:
    """Voice identity (the GSV reference clip + its transcript) and the server."""

    base_url: str = field(
        default_factory=lambda: os.getenv(
            "QWEN3TTS_URL", "http://127.0.0.1:8809/v1/audio/speech"
        )
    )
    model: str = "qwen3-tts"
    ref_audio_path: str = ""
    ref_text: str = ""
    text_lang: str = "en"
    # Only the speaker embedding (no transcript): faster prompt, looser clone.
    x_vector_only: bool = False
    # A speaker baked into the served checkpoint by fine-tuning (finetune/
    # sft_12hz_pcb.py --speaker_name): CustomVoice request, no reference clip.
    # The voice is then as fixed as the GSV fine-tune was.
    speaker: str | None = field(default_factory=lambda: os.getenv("QWEN3TTS_SPEAKER") or None)
    # Fixed sampling seed. Unseeded, the talker re-rolls the voice every sentence
    # (median f0 of one sentence rendered 3x: 179 / 183 / 190 Hz; seeded: 207 x3)
    # — heard as "someone else's voice mixed in" mid-reply (2026-11-03).
    seed: int | None = field(
        default_factory=lambda: int(os.getenv("QWEN3TTS_SEED", "1234") or 0) or None
    )


# Sentence-final slowdown (2026-11-04, "文末が早口になる"). The fine-tuned
# Aura voice keeps its pace, or speeds up, to the end of a sentence, where the
# speaker herself slows down: on the same 16 sentences, second-half / first-half
# speaking rate was 0.92 in Aura's recordings and 1.15 from the TTS (faster in
# 15 of 16; an energy-peak syllable count agreed: 0.91 vs 1.05). Neither the
# talker's temperature, its repetition penalty, nor two more epochs on her
# recordings alone moved it. So the later part of each sentence is stretched,
# pitch kept, from QWEN3TTS_SLOW_FROM of its expected length (characters /
# QWEN3TTS_CPS) up to QWEN3TTS_SLOW_END at the end. QWEN3TTS_SLOW_END=1 = off.
_SLOW_END = float(os.getenv("QWEN3TTS_SLOW_END", "1.4") or 1.0)
_SLOW_FROM = float(os.getenv("QWEN3TTS_SLOW_FROM", "0.4") or 0.4)
_CPS = float(os.getenv("QWEN3TTS_CPS", "18") or 18.0)


class _Slowdown:
    """Streaming WSOLA time stretch with a rising ratio (1.0 -> end).

    10 ms synthesis hop, 20 ms periodic-Hann frames (they sum to 1 at 50 %
    overlap); each frame is taken within +-5 ms of its nominal place where it
    best continues the previous one, so periods line up and the pitch stays.
    Starts on 10 ms of silence (the first window's rise) and ends on the last
    window's fall, so a sentence no longer stops on a hard edge."""

    def __init__(self, sr: int, expect_s: float, end: float, start_frac: float):
        self.hs = max(1, int(0.010 * sr))
        self.n = 2 * self.hs
        self.tol = self.hs // 2
        self.win = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(self.n) / self.n)).astype(np.float32)
        self.x = np.zeros(self.hs, dtype=np.float32)  # leading silence under the first rise
        self.x0 = 0
        self.real_end = self.hs
        self.pa = 0.0
        self.prev: int | None = None
        self.acc = np.zeros(self.n, dtype=np.float32)
        self.expect = max(0.5, expect_s) * sr
        self.end = max(1.0, end)
        self.start_frac = min(0.95, max(0.0, start_frac))

    def _ratio(self, pos: float) -> float:
        prog = pos / self.expect
        if prog <= self.start_frac:
            return 1.0
        return 1.0 + (self.end - 1.0) * min(1.0, (prog - self.start_frac) / (1.0 - self.start_frac))

    def feed(self, pcm: np.ndarray) -> np.ndarray:
        if pcm.size:
            self.x = np.concatenate([self.x, pcm.astype(np.float32)])
            self.real_end += pcm.size
        return self._run(final=False)

    def flush(self) -> np.ndarray:
        return self._run(final=True)

    def _run(self, final: bool) -> np.ndarray:
        out: list[np.ndarray] = []
        while True:
            pa = int(round(self.pa))
            if final and pa >= self.real_end:
                break
            lo, hi = (pa - self.tol, pa + self.tol) if self.prev is not None else (pa, pa)
            lo = max(lo, self.x0)
            need = hi + self.n
            if self.prev is not None:
                need = max(need, self.prev + self.hs + self.n)
            have = self.x0 + len(self.x)
            if need > have:
                if not final:
                    break
                self.x = np.concatenate([self.x, np.zeros(need - have, dtype=np.float32)])
            if self.prev is None:
                best = pa
            else:
                nat = self.prev + self.hs - self.x0
                ref = self.x[nat: nat + self.n]
                seg = self.x[lo - self.x0: hi - self.x0 + self.n]
                c = np.correlate(seg, ref, mode="valid")
                best = lo + int(np.argmax(c)) if c.size else pa
            b = best - self.x0
            self.acc += self.x[b: b + self.n] * self.win
            out.append(self.acc[: self.hs].copy())
            self.acc[: self.hs] = self.acc[self.hs:]
            self.acc[self.hs:] = 0.0
            self.prev = best
            self.pa += self.hs / self._ratio(self.pa)
            keep = min(self.prev, int(self.pa) - self.tol) - self.x0
            if keep > 4 * self.n:
                self.x = self.x[keep:]
                self.x0 += keep
        if final:
            out.append(self.acc[: self.hs].copy())
            self.acc[:] = 0.0
        if not out:
            return np.zeros(0, dtype=np.int16)
        return np.clip(np.concatenate(out), -32768, 32767).astype(np.int16)


class _Upsampler:
    """24 kHz -> 32 kHz, linear, keeps the last sample so chunk borders are seamless."""

    def __init__(self, src: int, dst: int):
        self.ratio = dst / src
        self.last: float | None = None
        self.phase = 0.0  # output position (in source samples) past `last`

    def feed(self, pcm: np.ndarray) -> np.ndarray:
        if pcm.size == 0:
            return pcm
        x = pcm.astype(np.float32)
        if self.last is not None:
            x = np.concatenate(([self.last], x))
            start = self.phase
        else:
            start = 0.0
        step = 1.0 / self.ratio
        n = int(np.floor((len(x) - 1 - start) / step)) + 1
        if n <= 0:
            self.last = float(x[-1])
            self.phase = start - (len(x) - 1)
            return np.zeros(0, dtype=np.int16)
        pos = start + np.arange(n) * step
        out = np.interp(pos, np.arange(len(x)), x)
        self.phase = pos[-1] + step - (len(x) - 1)
        self.last = float(x[-1])
        return np.clip(out, -32768, 32767).astype(np.int16)


async def stream_sentence_pcm(
    text: str,
    cfg: Qwen3TTSStreamConfig,
    client: httpx.AsyncClient | None = None,
) -> AsyncIterator[bytes]:
    """Yield raw PCM (int16 mono 32 kHz) for ``text`` as the server produces it.
    A backend hiccup is logged and ends the stream (a silent sentence), never raised."""
    owns = client is None
    if client is None:
        client = httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=5.0))
    body = {
        "model": cfg.model,
        "input": text,
        "task_type": "Base",
        "language": _LANG.get(cfg.text_lang, "English"),
        "response_format": "pcm",
        "stream_format": "audio",
    }
    if cfg.seed is not None:
        body["seed"] = cfg.seed
    # Bound a runaway. A fine-tuned CustomVoice checkpoint can fail to emit the
    # codec EOS and babble to the server's 680-token budget (54 s of loud
    # speech for "Hello.", 2026-11-03 epoch-9). Speech runs ~12.5 codec tokens
    # per second and ~14 characters per second, so ~0.9 tokens per character;
    # allow 2.5x that plus a floor, so a runaway costs a couple of seconds at
    # most, and no normal sentence is cut (100 chars -> 240 tokens = 19 s).
    body["max_new_tokens"] = int(min(680, 15 + 2.3 * len(text)))
    if cfg.speaker:
        body["task_type"] = "CustomVoice"
        body["voice"] = cfg.speaker
    elif cfg.ref_audio_path:
        body["ref_audio"] = f"file://{cfg.ref_audio_path}"
        if cfg.ref_text and not cfg.x_vector_only:
            body["ref_text"] = cfg.ref_text
        else:
            body["x_vector_only_mode"] = True
    up = _Upsampler(SERVER_RATE, PCM_SAMPLE_RATE)
    carry = b""
    try:
        _t_send = __import__("time").monotonic()
        if _TIMING:
            from ..utils.turn_timing import wall_ms

            logger.info(f"[timing] tts request sent @{wall_ms()}: {text[:30]!r}")
        async with client.stream("POST", cfg.base_url, json=body) as r:
            if r.status_code != 200:
                err = (await r.aread())[:300]
                logger.warning(f"[qwen3tts] {r.status_code} @ {cfg.base_url}: {err!r}")
                return
            rate = int(r.headers.get("x-audio-sample-rate", SERVER_RATE))
            if rate != SERVER_RATE:
                up = _Upsampler(rate, PCM_SAMPLE_RATE)
            slow = _Slowdown(rate, len(text) / _CPS, _SLOW_END, _SLOW_FROM) if _SLOW_END > 1.0 else None
            async for chunk in r.aiter_raw():
                if not chunk:
                    continue
                buf = carry + chunk
                even = len(buf) & ~1
                carry = buf[even:]
                if even:
                    pcm = np.frombuffer(buf[:even], dtype=np.int16)
                    if slow is not None:
                        pcm = slow.feed(pcm)
                    out = up.feed(pcm)
                    if out.size:
                        yield out.tobytes()
            if slow is not None:
                out = up.feed(slow.flush())
                if out.size:
                    yield out.tobytes()
    except Exception as e:  # noqa: BLE001 — degrade to a silent sentence
        logger.warning(f"[qwen3tts] stream failed for {text[:30]!r}: {e!r}")
    finally:
        if owns:
            await client.aclose()
