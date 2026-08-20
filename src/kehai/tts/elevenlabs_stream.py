"""Streaming ElevenLabs TTS: text -> raw PCM chunks, yielded as they arrive.

A drop-in for gsv_stream.stream_sentence_pcm, deliberately down to the sample
rate. The app's whole streaming path is built around 32 kHz s16 mono PCM — the
WAV writer, the sample_rate the client is told, the gesture-timing windows that
divide byte counts by it, and the WebRTC feed — so a different rate would mean
touching nine call sites to gain nothing.

`pcm_32000` is undocumented (ElevenLabs' published PCM rates are 16/22.05/24/
44.1 kHz) but the API accepts it and returns genuine 32 kHz: probed against the
real endpoint, `pcm_32000` and `pcm_24000` of the same sentence came back as
96595 and 72446 samples — 3.02 s either way, to the millisecond. Verified rather
than assumed because the failure mode is silent: a rate mismatch does not error,
it just plays the voice at the wrong pitch.

Uses the HTTP streaming endpoint, not the WebSocket. The WebSocket exists for
feeding a half-written sentence as an LLM emits tokens; this app already splits
into whole sentences upstream and calls this once per sentence, so the
WebSocket's buffering would add latency in exchange for a feature not needed
here. Its `alignment` payload is the reason to revisit that — see
ELEVEN_ALIGNMENT below.
"""

from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass, field
from typing import AsyncIterator

import httpx
from loguru import logger

# Matches gsv_stream so the two are interchangeable.
PCM_SAMPLE_RATE = 32000
PCM_SAMPLE_WIDTH = 2
PCM_CHANNELS = 1

BASE_URL = "https://api.elevenlabs.io/v1/text-to-speech"


@dataclass
class ElevenStreamConfig:
    """Voice identity + sampler knobs.

    Defaults are deliberately None so ElevenLabs' own per-voice settings apply:
    the voice was tuned in their UI (the chancellor avatar ships stability 0.6,
    similarity 0.85, style 0.1, speed 0.89) and duplicating those numbers here
    would silently override the tuning the moment either side changed.
    """

    voice_id: str = ""
    model_id: str = "eleven_flash_v2_5"
    api_key: str = ""
    language_code: str = ""
    # Mirrors GSVStreamConfig.text_lang. tts_manager reads `text_lang` off
    # whichever config it holds — for the gap-closer's language and for a
    # Japanese-specific pause rule — so a backend that omits it does not fall
    # back gracefully, it raises mid-sentence and the reply goes silent. Kept in
    # sync with language_code below rather than left for the caller to remember.
    text_lang: str = "en"
    # None = use the voice's saved setting.
    stability: float | None = None
    similarity_boost: float | None = None
    style: float | None = None
    speed: float | None = None
    use_speaker_boost: bool | None = None
    # 0..4; 3+ trades pronunciation of numbers/dates for latency, so the default
    # stays conservative.
    optimize_streaming_latency: int = 0
    # A pronunciation dictionary by NAME, not by id+version. The dictionary is
    # edited in ElevenLabs' own UI, and every edit mints a NEW version id — so a
    # pinned id would keep applying the rules as they were on the day it was
    # pasted here, silently, with the config still looking correct. Resolving the
    # name to its latest version each run means adding a rule in the UI is enough.
    pronunciation_dictionary: str = ""
    # Escape hatch: explicit {pronunciation_dictionary_id, version_id} locators,
    # for pinning a version on purpose.
    pronunciation_dictionary_locators: list = field(default_factory=list)

    @property
    def key(self) -> str:
        return self.api_key or os.getenv("ELEVENLABS_API_KEY", "")


# name -> locator, resolved once per process. `None` records a lookup that failed
# so a missing dictionary costs one request per run rather than one per sentence.
_DICT_CACHE: dict[str, dict | None] = {}


async def _resolve_dictionary(
    name: str, key: str, client: httpx.AsyncClient
) -> dict | None:
    """Look a pronunciation dictionary up by name and return its latest version.

    Never raises: mispronouncing a proper noun is a far better outcome than a
    silent turn, so any failure here degrades to "no dictionary".
    """
    if name in _DICT_CACHE:
        return _DICT_CACHE[name]
    loc = None
    try:
        r = await client.get(
            "https://api.elevenlabs.io/v1/pronunciation-dictionaries",
            headers={"xi-api-key": key},
            params={"page_size": 100},
            timeout=15.0,
        )
        if r.status_code == 200:
            for d in r.json().get("pronunciation_dictionaries") or []:
                if d.get("name") == name and not d.get("archived_time_unix"):
                    loc = {
                        "pronunciation_dictionary_id": d["id"],
                        "version_id": d["latest_version_id"],
                    }
                    logger.info(
                        f"[eleven_stream] pronunciation dictionary {name!r}: "
                        f"{d.get('latest_version_rules_num')} rules"
                    )
                    break
            if loc is None:
                logger.warning(
                    f"[eleven_stream] no pronunciation dictionary named {name!r} — "
                    "names are case-sensitive and must match the ElevenLabs UI"
                )
        else:
            logger.warning(f"[eleven_stream] dictionary list {r.status_code}")
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[eleven_stream] dictionary lookup failed: {e!r}")
    _DICT_CACHE[name] = loc
    return loc


async def _locators(cfg: ElevenStreamConfig, key: str, client: httpx.AsyncClient):
    locs = list(cfg.pronunciation_dictionary_locators or [])
    if cfg.pronunciation_dictionary:
        loc = await _resolve_dictionary(cfg.pronunciation_dictionary, key, client)
        if loc:
            locs.append(loc)
    return locs


async def _request(
    text: str, cfg: ElevenStreamConfig, key: str, client: httpx.AsyncClient
) -> tuple[dict, dict]:
    """The query params and JSON body, built ONCE for both stream endpoints.

    These were two near-identical copies, which is how `text_lang` came to be
    read on a path no short test sentence reached. Anything added to the request
    now reaches the plain and the with-timestamps endpoint by construction rather
    than by remembering to edit both.
    """
    params = {"output_format": f"pcm_{PCM_SAMPLE_RATE}"}
    if cfg.optimize_streaming_latency:
        params["optimize_streaming_latency"] = str(cfg.optimize_streaming_latency)
    body: dict = {"text": text, "model_id": cfg.model_id}
    if cfg.language_code:
        body["language_code"] = cfg.language_code
    vs = _voice_settings(cfg)
    if vs:
        body["voice_settings"] = vs
    locs = await _locators(cfg, key, client)
    if locs:
        body["pronunciation_dictionary_locators"] = locs
    return params, body


def _voice_settings(cfg: ElevenStreamConfig) -> dict | None:
    vs = {
        "stability": cfg.stability,
        "similarity_boost": cfg.similarity_boost,
        "style": cfg.style,
        "speed": cfg.speed,
        "use_speaker_boost": cfg.use_speaker_boost,
    }
    vs = {k: v for k, v in vs.items() if v is not None}
    return vs or None


async def stream_sentence_pcm(
    text: str,
    cfg: ElevenStreamConfig,
    client: httpx.AsyncClient | None = None,
    chunk_bytes: int = 8192,
    split_method: str | None = None,  # accepted for signature parity; unused
    on_visemes=None,
) -> AsyncIterator[bytes]:
    """Yield raw PCM (int16 mono 32 kHz) for ``text`` as ElevenLabs produces it.

    Never raises to the caller: a backend hiccup logs and ends the stream, so the
    turn degrades to "this sentence was silent" rather than taking the
    conversation down. Same contract as the GSV path, for the same reason.

    `on_visemes(cues)` opts into the `stream/with-timestamps` endpoint, which
    returns the same audio plus per-character timings. Optional, and the audio
    path does not depend on it: if the callback raises, or the timings are
    missing, the sentence still speaks. Mouth shapes are a nicety; the voice is
    not.
    """
    del split_method
    text = (text or "").strip()
    if not text:
        return
    if not cfg.voice_id:
        logger.warning("[eleven_stream] no voice_id configured — sentence skipped")
        return
    key = cfg.key
    if not key:
        logger.warning("[eleven_stream] no ELEVENLABS_API_KEY — sentence skipped")
        return

    owns_client = client is None
    if client is None:
        client = httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0))

    if on_visemes is not None:
        async for chunk in _stream_with_timestamps(
            text, cfg, client, key, on_visemes, owns_client
        ):
            yield chunk
        return

    url = f"{BASE_URL}/{cfg.voice_id}/stream"
    params, body = await _request(text, cfg, key, client)

    try:
        async with client.stream(
            "POST",
            url,
            params=params,
            json=body,
            headers={"xi-api-key": key},
        ) as resp:
            if resp.status_code != 200:
                detail = (await resp.aread())[:300]
                logger.warning(f"[eleven_stream] {resp.status_code}: {detail!r}")
                return
            # Chunks arrive at whatever size the transport gives; the consumer
            # reassembles across boundaries, and an odd trailing byte would shift
            # every later sample, so only whole frames are emitted.
            tail = b""
            async for raw in resp.aiter_bytes(chunk_bytes):
                if not raw:
                    continue
                buf = tail + raw
                usable = (len(buf) // PCM_SAMPLE_WIDTH) * PCM_SAMPLE_WIDTH
                tail = buf[usable:]
                if usable:
                    yield buf[:usable]
            if tail:
                logger.debug(f"[eleven_stream] dropped {len(tail)} trailing byte(s)")
    except Exception as e:  # noqa: BLE001 — a silent sentence beats a dead turn
        logger.warning(f"[eleven_stream] stream failed: {e!r}")
    finally:
        if owns_client:
            await client.aclose()


async def _stream_with_timestamps(
    text: str,
    cfg: ElevenStreamConfig,
    client: httpx.AsyncClient,
    key: str,
    on_visemes,
    owns_client: bool,
) -> AsyncIterator[bytes]:
    """Same audio, plus per-character timings, from `stream/with-timestamps`.

    The response is newline-delimited JSON — each line an `audio_base64` chunk
    with the `alignment` for exactly that chunk — so audio still arrives
    progressively; it is only wrapped. Chosen over the WebSocket `/stream-input`
    (which also carries alignment) because this keeps the plain
    request/response shape the rest of this module already handles, and the
    WebSocket's value is feeding half-written sentences, which this app never
    does: it splits into whole sentences upstream.

    Timings are per-SENTENCE, starting at 0. Placing a sentence on the turn's
    timeline is the caller's job — it is the only party that knows when this
    sentence's audio actually starts playing.
    """
    from .visemes import cues_from_alignment

    url = f"{BASE_URL}/{cfg.voice_id}/stream/with-timestamps"
    params, body = await _request(text, cfg, key, client)

    chars: list = []
    starts: list = []
    ends: list = []
    try:
        async with client.stream(
            "POST", url, params=params, json=body, headers={"xi-api-key": key}
        ) as resp:
            if resp.status_code != 200:
                detail = (await resp.aread())[:300]
                logger.warning(f"[eleven_stream] ts {resp.status_code}: {detail!r}")
                return
            async for line in resp.aiter_lines():
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except Exception:  # noqa: BLE001 — a partial line is not fatal
                    continue
                b64 = msg.get("audio_base64")
                if b64:
                    pcm = base64.b64decode(b64)
                    usable = (len(pcm) // PCM_SAMPLE_WIDTH) * PCM_SAMPLE_WIDTH
                    if usable:
                        yield pcm[:usable]
                al = msg.get("alignment") or {}
                if al:
                    chars += al.get("characters") or []
                    starts += al.get("character_start_times_seconds") or []
                    ends += al.get("character_end_times_seconds") or []
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[eleven_stream] ts stream failed: {e!r}")
    finally:
        if owns_client:
            await client.aclose()

    # Emitted once the sentence is complete: the cues are only useful as a whole
    # (they are scheduled against playback), and a failure here must not have
    # been able to interrupt the audio above.
    if chars:
        try:
            on_visemes(cues_from_alignment(chars, starts, ends))
        except Exception as e:  # noqa: BLE001 — the mouth must not break the voice
            logger.debug(f"[eleven_stream] viseme callback failed: {e!r}")
