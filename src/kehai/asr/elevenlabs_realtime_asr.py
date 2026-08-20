"""ElevenLabs Scribe v2 Realtime — the same segments, over a kept-open socket.

A drop-in for elevenlabs_asr: the server VAD still decides what counts as speech
and still hands over one finished segment, and this still returns its text. Only
the transport changes.

WHY THAT IS WORTH DOING. Measured against the real endpoint, same 2.86 s of
Japanese, same account:

    batch HTTP (elevenlabs_asr)          1051 ms
    realtime WS, socket already open      213-281 ms
    realtime WS, cold socket              240 ms + 184 ms handshake

The ~800 ms is not transcription — it is the per-request cost of a multipart
upload over a fresh TLS connection to a CDN edge. Holding one socket open pays
that once per process instead of once per segment.

WHY NOT STREAM THE MICROPHONE STRAIGHT IN. The endpoint will do that, and it is
faster still: fed in real time it emits partial transcripts DURING speech and
commits 138 ms after the last audio. But it would mean sending the raw mic
instead of what the VAD passed, and this app's turn-taking depends on the VAD
being the gate — the segment mean-level test exists because room noise was
reaching whisper and coming back as a hallucinated "Thank you." that the avatar
then answered. Streaming everything would hand that same noise to a new
recogniser, and bill for it. The gate stays; the transport gets faster.

NO DIARISATION HERE. The realtime API does not label speakers, so a character
that needs `diarize` (several people around the unit) must stay on the batch
engine. That is the one reason to prefer the slower route.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import threading
from typing import List, Optional

import numpy as np
from loguru import logger

from .asr_interface import ASRInterface

DEFAULT_URL = "wss://api.elevenlabs.io/v1/speech-to-text/realtime"

# Sent per frame. 0.5 s keeps the number of JSON frames small for a typical
# segment while staying well inside any message-size limit.
CHUNK_SAMPLES = 8000


class VoiceRecognition(ASRInterface):
    SAMPLE_RATE = 16000
    NUM_CHANNELS = 1
    SAMPLE_WIDTH = 2

    def __init__(
        self,
        api_key: str = "",
        base_url: str = DEFAULT_URL,
        model_id: str = "scribe_v2_realtime",
        language_code: str = "",
        keyterms: Optional[List[str]] = None,
        filter_background_audio: bool = True,
        timeout: float = 30.0,
    ):
        self.base_url = base_url or DEFAULT_URL
        self.api_key = api_key or os.getenv("ELEVENLABS_API_KEY", "")
        self.model_id = model_id
        self.language_code = language_code
        self.keyterms = [t[:50] for t in (keyterms or [])][:1000]
        # The endpoint's own non-speech suppression. On by default here for the
        # same reason tag_audio_events is off in the batch engine: anything the
        # recogniser invents for a door closing becomes something the user
        # "said".
        self.filter_background_audio = filter_background_audio
        self.timeout = timeout
        self._ws = None
        self._ws_loop = None
        # Segments overlap in this app — several can be in flight while the user
        # keeps talking — but one socket carries one transcript at a time, so
        # they are serialised here. They are still concurrent with the SPEECH,
        # which is where the overlap actually pays.
        self._lock: asyncio.Lock | None = None
        # Private loop for the sync entry point (see transcribe_np).
        self._sync_loop: asyncio.AbstractEventLoop | None = None
        self._sync_thread: threading.Thread | None = None
        self._sync_start = threading.Lock()
        if not self.api_key:
            logger.warning(
                "[eleven_rt_asr] no API key (api_key empty and ELEVENLABS_API_KEY "
                "unset) — transcription will be rejected"
            )

    # -- socket ------------------------------------------------------------

    def _url(self) -> str:
        q = [f"model_id={self.model_id}"]
        if self.language_code:
            q.append(f"language_code={self.language_code}")
        if self.filter_background_audio:
            q.append("filter_background_audio=true")
        for t in self.keyterms:
            q.append(f"keyterms={t}")
        return f"{self.base_url}?{'&'.join(q)}"

    async def _connect(self):
        import websockets

        ws = await websockets.connect(
            self._url(),
            additional_headers={"xi-api-key": self.api_key},
            open_timeout=10,
            ping_interval=20,
        )
        # The server opens with session_started; reading it here means a
        # rejected session surfaces now rather than as a silent empty segment.
        first = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
        if first.get("message_type") != "session_started":
            await ws.close()
            raise RuntimeError(f"session not started: {str(first)[:200]}")
        logger.info(f"[eleven_rt_asr] session {first.get('session_id', '?')[:8]} open")
        return ws

    async def _ensure(self):
        # A websocket belongs to the event loop that created it. Reusing one from
        # a different loop raises "Event loop is closed" on the first send — which
        # the retry below survives, but at the cost of a whole round trip
        # (measured 478 ms per call instead of 256 ms) and a warning per segment
        # that reads like a fault. So the loop is part of the socket's identity.
        loop = asyncio.get_running_loop()
        if self._ws is not None and self._ws_loop is loop:
            try:
                if self._ws.state.name == "OPEN":
                    return self._ws
            except Exception:  # noqa: BLE001 — treat anything odd as "reconnect"
                pass
        self._ws = await self._connect()
        self._ws_loop = loop
        return self._ws

    # -- transcription -----------------------------------------------------

    async def _once(self, pcm: bytes) -> str:
        ws = await self._ensure()
        for i in range(0, len(pcm), CHUNK_SAMPLES * self.SAMPLE_WIDTH):
            frame = pcm[i : i + CHUNK_SAMPLES * self.SAMPLE_WIDTH]
            await ws.send(
                json.dumps(
                    {
                        "message_type": "input_audio_chunk",
                        "audio_base_64": base64.b64encode(frame).decode(),
                        "sample_rate": self.SAMPLE_RATE,
                    }
                )
            )
        # An empty chunk with commit=true closes the utterance without adding
        # audio, so the transcript covers exactly the segment the VAD handed over.
        await ws.send(
            json.dumps(
                {
                    "message_type": "input_audio_chunk",
                    "audio_base_64": "",
                    "sample_rate": self.SAMPLE_RATE,
                    "commit": True,
                }
            )
        )
        text = ""
        while True:
            msg = await asyncio.wait_for(ws.recv(), timeout=self.timeout)
            d = json.loads(msg)
            mt = d.get("message_type")
            if mt == "committed_transcript":
                return (d.get("text") or "").strip()
            if mt == "partial_transcript":
                # Kept as the fallback answer: if the socket dies before the
                # commit lands, the best partial beats losing the turn.
                text = (d.get("text") or "").strip() or text
            elif mt in ("input_error", "error"):
                raise RuntimeError(str(d)[:200])
            elif mt == "session_ended":
                self._ws = None
                return text

    async def async_transcribe_np(self, audio: np.ndarray) -> str:
        if audio is None or len(audio) == 0:
            return ""
        if self._lock is None:
            self._lock = asyncio.Lock()
        pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
        async with self._lock:
            # One retry on a FRESH socket. A kept-open connection is the whole
            # point, and the cost of that is that it can be stale in ways only
            # discovered on send — a reconnect turns "she stopped hearing me"
            # into an extra ~200 ms.
            for attempt in (1, 2):
                try:
                    return await self._once(pcm)
                except Exception as e:  # noqa: BLE001 — never take the turn down
                    logger.warning(f"[eleven_rt_asr] attempt {attempt} failed: {e!r}")
                    try:
                        if self._ws is not None:
                            await self._ws.close()
                    except Exception:  # noqa: BLE001
                        pass
                    self._ws = None
            return ""

    def _sync_submit(self, coro):
        """Run `coro` on this engine's own long-lived loop, from any thread.

        Not `asyncio.run`: that builds and destroys a loop per call, and a socket
        cannot outlive its loop — so every sync call found a dead socket, paid a
        failed send plus a reconnect, and logged a warning that read like a
        fault (measured 478 ms per call instead of 256 ms). One loop for the
        lifetime of the engine keeps the socket warm on this path too.
        """
        if self._sync_loop is None:
            with self._sync_start:
                if self._sync_loop is None:
                    loop = asyncio.new_event_loop()
                    t = threading.Thread(
                        target=loop.run_forever, name="eleven-rt-asr", daemon=True
                    )
                    t.start()
                    self._sync_loop, self._sync_thread = loop, t
        fut = asyncio.run_coroutine_threadsafe(coro, self._sync_loop)
        return fut.result(timeout=self.timeout + 15)

    def transcribe_np(self, audio: np.ndarray) -> str:
        """Sync entry point, for callers outside the event loop — the shadow A/B
        runs its challenger on a plain thread."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return self._sync_submit(self.async_transcribe_np(audio))
        # Inside a loop the async path is the one to call; blocking on a future
        # from here would deadlock, so this is a programming error rather than a
        # runtime one.
        raise RuntimeError(
            "transcribe_np called from inside an event loop — use async_transcribe_np"
        )

    def transcribe(self, audio_path: str) -> str:  # pragma: no cover - unused
        raise NotImplementedError("eleven_rt_asr transcribes numpy audio only")
