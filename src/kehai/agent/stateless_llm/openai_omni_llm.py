"""Qwen3-Omni (served by vLLM-Omni) as the conversation LLM.

The user's SPEECH goes in as audio — an `audio_url` data: part that the agent
adds to the user turn (basic_memory_agent._to_messages) — and text comes out
exactly as from any OpenAI-compatible model, so sentence splitting, the
emotion / gesture tags, tools, memory and interrupt handling are untouched.
Whisper keeps running beside it for the TURN CONTROL (segmenting, semantic
endpointing, the speculative match, history); only the understanding moves to
the audio, which is what removes the ASR word-mangling from the reply.

With audio_output=true the Talker's speech rides the same stream as "audio"
chunks. They are taken off the text stream here and handed to the turn's
audio sink; tts/omni_stream.py reads that sink in place of GPT-SoVITS, so
tts_manager (framing, motion, subtitles, TTFA) never knows the difference.

Only ever constructed in `omni` mode (LLM_PROVIDER_OVERRIDE=openai_omni_llm,
exported by start_server.sh from .mode); the cascade never imports this.
"""

import asyncio
import base64
import io
from typing import Any, Dict, Optional

import numpy as np
from loguru import logger

from .openai_compatible_llm import AsyncLLM as _OpenAIAsyncLLM

try:  # only needed to decode Talker audio (audio_output=true)
    import soundfile as sf
except Exception:  # noqa: BLE001
    sf = None


class AsyncOmniLLM(_OpenAIAsyncLLM):
    # Read by basic_memory_agent._to_messages: attach the utterance audio.
    accepts_audio = True

    def __init__(
        self,
        *,
        speaker: Optional[str] = None,
        audio_output: bool = False,
        top_k: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.top_k = int(top_k) if top_k else None
        # The base __init__ writes the instance default (False); re-assert.
        self.accepts_audio = True
        self.speaker = speaker or None
        self.audio_output = bool(audio_output)
        # The turn's audio deltas as (float32 mono, sample_rate); None marks the
        # end of one LLM call. Installed per turn by the consumer (new_turn()).
        self.audio_sink: Optional[asyncio.Queue] = None
        self._audio_chunks = 0
        logger.info(
            f"[omni] audio_output={self.audio_output} "
            f"speaker={self.speaker or 'default'} "
            f"sampling temperature={self.temperature} top_p={self.top_p} top_k={self.top_k}"
        )

    # "Heard" step (2026-11-04, "うまく会話が成立しない"). Answering straight
    # from the audio under the 7.5k-char persona, the Thinker misread what it
    # could transcribe correctly when asked to: "I want to get information about
    # nearby supermarket or market" was answered with a campus blackout, "I want
    # to know about tomorrow's weather" with "I am not aware of tomorrow's
    # weather". Transcribing first (97-168 ms) and handing that text in beside
    # the audio made 8 of 9 such replies fit. Its transcript was also closer than
    # Whisper's on that speaker (band-limited PROTO mic, accented English).
    # OMNI_HEARD=0 turns it off; OMNI_HEARD_LANG names the language to write in.
    _HEARD_SYSTEM = "Transcribe the speech exactly, in {lang}. Output only the words."

    async def transcribe(self, wav_b64: str, timeout: float = 1.5) -> Optional[str]:
        # One request per utterance: the sentence-completeness check
        # (omni_addressee.COMPLETE) asks for the same audio at the same pause, and
        # omni_addressee.HEARD shares it.
        from ... import omni_addressee

        return await omni_addressee.HEARD.get(wav_b64, timeout)

    def new_turn(self) -> asyncio.Queue:
        """Fresh sink for the turn about to start; the consumer keeps the handle."""
        self.audio_sink = asyncio.Queue()
        self._audio_chunks = 0
        return self.audio_sink

    # --- hooks of openai_compatible_llm.AsyncLLM ---------------------------

    def _extra_request_kwargs(self) -> Dict[str, Any]:
        kw: Dict[str, Any] = {}
        if self.audio_output:
            kw["modalities"] = ["text", "audio"]
        extra: Dict[str, Any] = {}
        if self.top_k:
            extra["top_k"] = self.top_k
        if self.speaker and self.audio_output:
            # vLLM logs and ignores fields it does not know, so a wrong name here
            # costs nothing but the voice falling back to the default.
            extra["speaker"] = self.speaker
        if extra:
            kw["extra_body"] = extra
        return kw

    def _consume_side_channel(self, chunk) -> bool:
        # vLLM-Omni tags every streamed chunk with `modality` ("text" | "audio");
        # the openai SDK keeps unknown fields as pydantic extras.
        modality = getattr(chunk, "modality", None)
        if modality is None:
            extra = getattr(chunk, "model_extra", None)
            if isinstance(extra, dict):
                modality = extra.get("modality")
        if modality != "audio":
            return False
        if not chunk.choices:
            return True
        data = chunk.choices[0].delta.content
        if not data:
            return True
        sink = self.audio_sink
        if sink is None:
            return True  # text-only consumer this turn: drop the audio quietly
        try:
            raw = base64.b64decode(data)
            if sf is not None:
                pcm, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
            else:  # bare PCM16 at the Talker's native rate
                pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                sr = 24000
            if getattr(pcm, "ndim", 1) > 1:
                pcm = pcm.mean(axis=1)
            self._audio_chunks += 1
            if self._audio_chunks == 1:
                logger.info(
                    f"[omni] first audio chunk {len(pcm) / sr * 1000:.0f} ms @ {sr} Hz"
                )
            sink.put_nowait((np.asarray(pcm, dtype=np.float32), int(sr)))
        except Exception as e:  # noqa: BLE001 — a bad chunk must not kill the text
            logger.warning(f"[omni] audio chunk dropped: {e!r}")
        return True

    def _on_stream_end(self) -> None:
        sink = self.audio_sink
        if sink is not None:
            sink.put_nowait(None)
            if self._audio_chunks:
                logger.debug(f"[omni] call ended after {self._audio_chunks} audio chunks")
