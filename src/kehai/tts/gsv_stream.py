"""Streaming GPT-SoVITS client: text -> raw PCM chunks, yielded as they arrive.

Unlike `openai_tts` (which downloads a whole sentence's WAV to a file before the
frontend can play it), this streams GSV's raw PCM as it is generated so the
frontend can start playing the first phrase ~1s sooner. Two GSV knobs make the
stream INCREMENTAL rather than one big burst at the end:

  * ``parallel_infer=false`` + ``batch_size=1`` — synthesize segments SEQUENTIALLY
    so each finishes (and streams) before the next starts. With the default
    parallel batching GSV computes every segment first, then streams (measured:
    first PCM at ~1.6s vs ~1.0s sequential).
  * ``text_split_method=cut5`` — split the sentence on punctuation so a sentence
    with internal commas streams phrase-by-phrase instead of as one segment.

Output is raw PCM: signed 16-bit little-endian, mono, 32000 Hz (GSV v2's rate).
``Accept-Encoding: identity`` is REQUIRED — otherwise httpx buffers the whole
body to gunzip it and the stream collapses back to one end-of-synthesis burst.
"""

from __future__ import annotations

import asyncio
import os
from typing import AsyncIterator
from dataclasses import dataclass, field

import httpx
from loguru import logger

# GSV raw PCM format (fixed by the GPT-SoVITS v2 model).
PCM_SAMPLE_RATE = 32000
PCM_SAMPLE_WIDTH = 2  # bytes (int16)
PCM_CHANNELS = 1

_STREAM_HEADERS = {"Accept-Encoding": "identity"}  # keep the body un-gzipped => real streaming


@dataclass
class GSVStreamConfig:
    """Voice identity + sampler knobs sent to GSV /tts. Mirrors the values the
    gateway/voice_service already use for the Rem voice (so streamed audio sounds
    identical to the non-streaming path)."""

    # Nexus gateway streaming route: forwards every param below to GPT-SoVITS
    # (:9880) /tts and pipes the raw PCM straight back — no SSH tunnel. Overridable
    # via GSV_STREAM_URL (e.g. a direct http://127.0.0.1:9880/tts for LAN testing).
    base_url: str = "https://gpu.aiccore-uno.ai/v1/audio/stream"
    # Bearer key for the gateway (start_server.sh exports MOTION_GATEWAY_KEY from
    # .motion_gateway_key). Empty => no auth header (direct GSV, no gateway).
    api_key: str = field(default_factory=lambda: os.getenv("MOTION_GATEWAY_KEY", ""))
    ref_audio_path: str = "/home/ai-ccore/Documents/DEV/voice/refs/f02_025.wav"
    prompt_text: str = (
        "今日はレーって始まってムーで終わる名前の女の子と一緒にいるといいことがありますよ。"
    )
    text_lang: str = "ja"
    prompt_lang: str = "ja"
    # Sampler defaults (anti-runaway values from the project notes).
    # repetition_penalty: 1.35 measurably SWALLOWS the sentence-final vowel
    # (same text synthesized ~60-120ms shorter; "ね" -> "ねっ" 促音感 — the AR
    # model hits EOS early because the trailing vowel's continuation tokens get
    # penalized). 1.15 keeps some runaway protection; per-sentence texts are
    # short so runaway risk is low. Revert live via GSV_REP_PEN=1.35.
    tuning: dict = field(
        default_factory=lambda: {
            "top_k": "10",
            "top_p": "0.8",
            "temperature": "0.6",
            "repetition_penalty": os.getenv("GSV_REP_PEN", "1.15"),
            "speed_factor": "1.05",
        }
    )
    # Streaming-specific: sequential synth + punctuation split => incremental PCM.
    split_method: str = "cut5"
    # Silence GSV inserts between cut5 fragments (streaming). This silence is
    # INSIDE the synthesized audio, so the track's 0.3s minimum-gap floor can't
    # shape it. 0.15 measured 0.28-0.30s TYPICAL comma gaps but the fragment's
    # trailing decay varies — tap analysis caught joins down to 150-250ms,
    # which sits in the glottal-stop percept band (≤~250ms = っ, ~300ms+ = 間).
    # 0.22 pushes every comma gap safely above 0.3s. Tune via GSV_FRAG_INTERVAL.
    fragment_interval: str = field(
        default_factory=lambda: os.getenv("GSV_FRAG_INTERVAL", "0.22")
    )


def _params(text: str, cfg: GSVStreamConfig, split_method: str | None = None) -> dict:
    return {
        "text": text,
        "text_lang": cfg.text_lang,
        "ref_audio_path": cfg.ref_audio_path,
        "prompt_lang": cfg.prompt_lang,
        "prompt_text": cfg.prompt_text,
        "text_split_method": split_method or cfg.split_method,
        "batch_size": "1",
        "parallel_infer": "false",
        "split_bucket": "false",
        "media_type": "raw",
        # GSV api_v2 streaming modes: "true"/1 = best quality, but the first
        # chunk only leaves when the WHOLE sentence is synthesized (0.2 s for
        # "Hello.", 0.8 s for a 7-word line, 1.6 s for a 20-word one — measured
        # 2026-11-03 against :9880 directly); 3 = chunk-by-chunk out of the AR
        # decoder, first chunk 0.18-0.22 s whatever the length, "lower quality"
        # per GSV's own doc (chunk joins). Try GSV_STREAMING_MODE=3 by ear.
        "streaming_mode": os.getenv("GSV_STREAMING_MODE", "true"),
        "fragment_interval": cfg.fragment_interval,
        **cfg.tuning,
    }


async def stream_sentence_pcm(
    text: str,
    cfg: GSVStreamConfig,
    client: httpx.AsyncClient | None = None,
    chunk_bytes: int = 8192,
    split_method: str | None = None,
) -> AsyncIterator[bytes]:
    """Yield raw PCM (int16 mono 32kHz) for ``text`` as GSV produces it.

    Never raises to the caller for a backend hiccup — logs and stops the stream so
    the conversation degrades to "this sentence was silent" instead of crashing.
    """
    owns_client = client is None
    if client is None:
        client = httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0))
    # base_url may be the LAN gateway tunnel (fast path); keep the PUBLIC route
    # as an automatic fallback so a tunnel drop degrades to +0.2s latency, not
    # to a silent sentence. Only retried if NOTHING was yielded yet — a stream
    # that died mid-sentence can't be restarted without replaying audio.
    fallback = os.getenv(
        "GSV_STREAM_FALLBACK", "https://gpu.aiccore-uno.ai/v1/audio/stream"
    )
    urls = [cfg.base_url]
    if fallback and fallback != cfg.base_url:
        urls.append(fallback)
    try:
        headers = dict(_STREAM_HEADERS)
        if cfg.api_key:
            headers["Authorization"] = f"Bearer {cfg.api_key}"
        # Up to 3 rounds over the URL list with a short pause between rounds: a
        # transient 429 ("max concurrency reached" — e.g. the tail of the
        # previous turn still releasing its slot) must delay the sentence by
        # ~0.6s, NOT silently drop it (a dropped mid-reply sentence is heard as
        # "the speech cut off").
        for round_i in range(3):
            if round_i:
                await asyncio.sleep(0.6)
            for i, url in enumerate(urls):
                yielded = False
                # Tunnel (127.0.0.1:13000) fails over FAST: a half-dead tunnel
                # (TCP connects, PC-B silent after a VPN blip) must drop to the
                # public route in seconds, not block the sentence for the
                # client's 120s read timeout ("しゃべっても反応しない").
                is_tunnel = "127.0.0.1" in url or "localhost" in url
                stream_kw: dict = {
                    "params": _params(text, cfg, split_method),
                    "headers": headers,
                }
                if is_tunnel:
                    stream_kw["timeout"] = httpx.Timeout(8.0, connect=2.0)
                try:
                    async with client.stream("GET", url, **stream_kw) as r:
                        if r.status_code != 200:
                            body = (await r.aread())[:200]
                            logger.warning(
                                f"[gsv_stream] {r.status_code} @ {url}: {body!r}"
                            )
                            continue  # next URL / next round
                        async for chunk in r.aiter_raw():
                            if chunk:
                                yielded = True
                                yield chunk
                    return
                except Exception as e:  # noqa: BLE001
                    if yielded:  # died mid-sentence — can't cleanly restart
                        logger.warning(
                            f"[gsv_stream] died mid-stream for {text[:30]!r}: {e!r}"
                        )
                        return
                    more = " — trying fallback" if i + 1 < len(urls) else ""
                    logger.warning(f"[gsv_stream] {url} failed: {e!r}{more}")
        logger.warning(
            f"[gsv_stream] ALL attempts failed — sentence dropped: {text[:40]!r}"
        )
    finally:
        if owns_client:
            await client.aclose()
