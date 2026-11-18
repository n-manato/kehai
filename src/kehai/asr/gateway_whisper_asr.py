"""Gateway STT client: POSTs the utterance to the Nexus gateway
``/v1/audio/transcriptions`` (faster-whisper running on PC-B's RTX 3090) and
returns the transcription text.

Keeps PC-A's GPU idle and matches how LLM / TTS / gesture are already routed
(public gateway, Bearer key). Measured warm: ~1.4s for a ~3s JA utterance
(large-v3 via the public route); the first request after a PC-B reboot pays the
model load (~30s) — start_server.sh's warmup covers that.
"""

import io
import os
import time
import wave
from typing import Optional

import httpx
import numpy as np
from loguru import logger

from .asr_interface import ASRInterface
from ..utils.sentence_spacing import unglue


class VoiceRecognition(ASRInterface):
    def __init__(
        self,
        base_url: str = "https://gpu.aiccore-uno.ai/v1/audio/transcriptions",
        model: str = "faster-whisper-large-v3",
        api_key: str = "",
        timeout: float = 60.0,
        language: str = "",
        beam_size: int = 1,
        initial_prompt: str = "",
        vad_filter: bool = False,
    ) -> None:
        self.base_url = base_url
        # Live overrides (start_server.sh, omni mode): there the LLM hears the
        # audio itself and this transcript only gates the turn and fills the
        # history, so the faster turbo model with greedy decoding is enough —
        # and with a 0.6 s endpoint the text has to be back before the endpoint,
        # or the speculative reply sits released-but-waiting on it (2026-11-03).
        self.model = os.getenv("ASR_MODEL_OVERRIDE", "") or model
        # Whisper language hint; empty = per-utterance auto-detect. The PC-B
        # sidecar forwards it to WhisperModel.transcribe(language=...).
        self.language = language
        # ACCENTED-SPEECH aids (a Japanese speaker's English here). beam_size>1
        # recovers a lot of what greedy decoding mangles; initial_prompt biases
        # the decoder toward the names and terms this deployment actually uses,
        # so they stop turning into similar-sounding everyday words.
        self.beam_size = int(os.getenv("ASR_BEAM_OVERRIDE", "") or beam_size)
        self.initial_prompt = initial_prompt
        # Strip non-speech spans before decoding: whisper otherwise "hears"
        # breath and room tone as words and reports them with full confidence.
        # Measured free (0.55s with, 0.55s without).
        self.vad_filter = vad_filter
        # Logged once per change, not per turn: the point is to notice a silent
        # substitution, not to narrate every transcription.
        self._last_model_seen: Optional[str] = None
        # start_server.sh exports MOTION_GATEWAY_KEY from .motion_gateway_key —
        # an empty api_key here falls back to it so the key stays out of configs.
        self.api_key = api_key or os.getenv("MOTION_GATEWAY_KEY", "")
        self._client = httpx.Client(timeout=httpx.Timeout(timeout, connect=10.0))
        if not self.api_key:
            logger.warning(
                "[gateway_asr] no Bearer key (api_key empty and MOTION_GATEWAY_KEY "
                "unset) — gateway requests will be rejected"
            )

    def _wav_bytes(self, audio: np.ndarray) -> bytes:
        """float32 mono @SAMPLE_RATE -> 16-bit PCM WAV bytes (in memory)."""
        pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(self.NUM_CHANNELS)
            w.setsampwidth(self.SAMPLE_WIDTH)
            w.setframerate(self.SAMPLE_RATE)
            w.writeframes(pcm)
        return buf.getvalue()

    def _ogg_opus_bytes(self, audio: np.ndarray) -> Optional[bytes]:
        """float32 mono @16k -> Ogg/Opus bytes (~15x smaller than WAV, so the
        Cloudflare upload leg of every segment's transcription shrinks). Returns
        None if encoding is unavailable — the caller falls back to WAV."""
        try:
            import av  # aiortc already pulls PyAV in

            pcm16 = (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2")
            buf = io.BytesIO()
            out = av.open(buf, mode="w", format="ogg")
            stream = out.add_stream("libopus", rate=self.SAMPLE_RATE)
            frame_len = 320  # 20ms @16k — a size libopus accepts directly
            pts = 0
            for i in range(0, len(pcm16), frame_len):
                chunk = pcm16[i : i + frame_len]
                frame = av.AudioFrame(format="s16", layout="mono", samples=len(chunk))
                frame.sample_rate = self.SAMPLE_RATE
                frame.pts = pts
                pts += len(chunk)
                frame.planes[0].update(chunk.tobytes())
                for pkt in stream.encode(frame):
                    out.mux(pkt)
            for pkt in stream.encode(None):
                out.mux(pkt)
            out.close()
            return buf.getvalue()
        except Exception as e:  # noqa: BLE001 — any encode hiccup -> WAV fallback
            logger.debug(f"[gateway_asr] opus encode unavailable ({e!r}); using WAV")
            return None

    # Public route used when base_url (usually the SSH LAN tunnel) is down —
    # a dead tunnel must degrade to +~0.35s latency, not to deaf ears.
    # ASR_PUBLIC_FALLBACK="" turns it off: on PC-B the public gateway is this
    # same box behind Cloudflare, so it cannot help, and the detour cost a turn
    # 35 s on 2026-10-23 21:22 (5 s local timeout, then a 30 s wait for a 502).
    _PUBLIC_URL = os.getenv(
        "ASR_PUBLIC_FALLBACK", "https://gpu.aiccore-uno.ai/v1/audio/transcriptions"
    )

    async def async_transcribe_fast(self, audio: np.ndarray) -> str:
        """The quick model, greedy (ASR_FAST_MODEL, default the turbo id): for a
        transcript that has to be back before the turn's endpoint (~280 ms after
        the pause) — the busy-room check against the Thinker's own transcript.
        large-v3 with beam 5 takes 370-1300 ms; this ~180-260 ms."""
        import asyncio

        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        return await asyncio.to_thread(self.transcribe_np, audio, True)

    def transcribe_np(self, audio: np.ndarray, fast: bool = False) -> str:
        # Opus only pays off on a real network leg. On loopback (PC-B: the
        # gateway is on this box) it is pure cost: ~25ms to encode here, ~8ms
        # to decode in the sidecar, and a lossy copy of the speech whisper
        # then has to recognize. Loopback gets the raw WAV — except PC-A's
        # :13000, which is an SSH tunnel and so a network leg after all.
        _tw = time.monotonic()
        wav_files = {"file": ("utterance.wav", self._wav_bytes(audio), "audio/wav")}
        _wav_ms = (time.monotonic() - _tw) * 1000
        ogg_files = None

        def files_for(url: str) -> dict:
            nonlocal ogg_files
            local = "127.0.0.1" in url or "localhost" in url
            if local and ":13000" not in url:
                return wav_files
            if ogg_files is None:
                ogg = self._ogg_opus_bytes(audio)
                ogg_files = (
                    {"file": ("utterance.ogg", ogg, "audio/ogg")}
                    if ogg is not None
                    else wav_files
                )
            return ogg_files

        data = {"model": os.getenv("ASR_FAST_MODEL", "faster-whisper-large-v3") if fast else self.model}
        if self.language:
            data["language"] = self.language
        if self.beam_size and self.beam_size > 1 and not fast:
            data["beam_size"] = str(self.beam_size)
        if self.initial_prompt:
            data["initial_prompt"] = self.initial_prompt
        if self.vad_filter:
            data["vad_filter"] = "1"
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        urls = [self.base_url]
        if self._PUBLIC_URL and self._PUBLIC_URL not in urls:
            urls.append(self._PUBLIC_URL)
        for i, url in enumerate(urls):
            # The tunnel (127.0.0.1:13000) gets a SHORT timeout: a half-dead
            # tunnel (VPN blip — TCP connects but PC-B never answers) must fail
            # over to the public route in seconds, not block up to `timeout`
            # (60s) which reads as "しゃべっても反応しない". Public keeps the long
            # timeout (a real transcription can take a while).
            # Only when there IS a fallback to fail over to: with none, cutting a
            # slow loopback transcription short just loses the turn.
            is_tunnel = ("127.0.0.1" in url or "localhost" in url) and i + 1 < len(urls)
            per_to = (
                httpx.Timeout(5.0, connect=2.0) if is_tunnel else self._client.timeout
            )
            try:
                t0 = time.monotonic()
                r = self._client.post(
                    url, headers=headers, files=files_for(url), data=data, timeout=per_to
                )
                if r.status_code != 200:
                    logger.warning(
                        f"[gateway_asr] {r.status_code} @ {url}: {r.text[:200]}"
                    )
                    continue
                body = r.json()
                # Report the model the backend ACTUALLY ran, not the one we asked
                # for. They differ: the sidecar silently serves "large-v3" with
                # large-v3-turbo. That substitution cost accuracy on accented
                # speech for weeks and was invisible from here, because this
                # client used to read only `text` off the response.
                used = body.get("model")
                if used and used != self._last_model_seen:
                    self._last_model_seen = used
                    logger.info(
                        f"[gateway_asr] backend model = {used!r} "
                        f"(requested {self.model!r})"
                    )
                logger.debug(
                    f"[gateway_asr] {used or '?'} {(time.monotonic() - t0) * 1000:.0f}ms"
                )
                if fast:  # where a quick transcript's time goes (it has to beat the endpoint)
                    self.last_fast = (_wav_ms, (time.monotonic() - t0) * 1000, r.elapsed.total_seconds() * 1000)
                # The sidecar joins whisper's segments without a space between
                # them: "Bye.All you." — see sentence_spacing.
                return unglue((body.get("text") or "").strip())
            except Exception as e:  # noqa: BLE001 — try fallback, else raise
                more = " — trying public fallback" if i + 1 < len(urls) else ""
                logger.warning(f"[gateway_asr] {url} failed: {e!r}{more}")
        # A failure is not silence. Returned as "", the cumulative ASR took it as
        # "nothing more was said" and answered the previous, shorter transcript —
        # the sentence cut off where the last pause had been.
        raise RuntimeError("gateway ASR failed on every route")
