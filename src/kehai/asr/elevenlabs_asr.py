"""ElevenLabs Scribe speech-to-text.

Same shape as gateway_whisper_asr: the server VAD hands over one finished
segment as float32 mono @16k, this returns its text. Deliberately a drop-in for
that engine rather than a new pipeline — the turn-taking around it (proximity
gate, segment mean-level gate, semantic endpointing) is heavily tuned, and
swapping the recogniser is the only way to measure the recogniser.

Three things it does that the whisper route could not:

* `keyterms` is a real parameter, so the domain vocabulary stops being smuggled
  in through `initial_prompt`. That workaround has bitten before: whisper ECHOED
  the prompt back as a user turn on quiet audio, and the avatar answered it.
  A dedicated field cannot be echoed because it is not part of the prompt.

* `tag_audio_events` is turned OFF. It defaults to ON and annotates non-speech
  with "(laughter)", "(footsteps)" and the like — which is precisely the class
  of text that must never reach the LLM as something the user said. The whisper
  equivalent of this problem was hallucinated "Thank you." on room noise.

* `diarize` labels each word with a `speaker_id`. With several people around the
  unit, loudness alone cannot separate "talking to her" from "talking near her";
  speaker identity can. Note this is BATCH-only — the realtime WebSocket API has
  no diarisation — so it is a reason to stay on this endpoint.
"""

import io
import os
import wave
from typing import List, Optional

import httpx
import numpy as np
from loguru import logger

from .asr_interface import ASRInterface

DEFAULT_URL = "https://api.elevenlabs.io/v1/speech-to-text"


class VoiceRecognition(ASRInterface):
    SAMPLE_RATE = 16000
    NUM_CHANNELS = 1
    SAMPLE_WIDTH = 2

    def __init__(
        self,
        api_key: str = "",
        base_url: str = DEFAULT_URL,
        model_id: str = "scribe_v2",
        language_code: str = "",
        keyterms: Optional[List[str]] = None,
        diarize: bool = False,
        num_speakers: int = 0,
        tag_audio_events: bool = False,
        timeout: float = 30.0,
    ):
        self.base_url = base_url or DEFAULT_URL
        # Key from config, else the environment — same convention as the gateway
        # engine, so a key never has to be written into a tracked file.
        self.api_key = api_key or os.getenv("ELEVENLABS_API_KEY", "")
        self.model_id = model_id
        self.language_code = language_code
        # Max 1000 terms, 50 chars each (API limit); trimmed rather than sent and
        # rejected, because a 422 here would read as "she stopped hearing me".
        self.keyterms = [t[:50] for t in (keyterms or [])][:1000]
        self.diarize = diarize
        self.num_speakers = num_speakers
        self.tag_audio_events = tag_audio_events
        self._client = httpx.Client(timeout=httpx.Timeout(timeout, connect=5.0))
        if not self.api_key:
            logger.warning(
                "[elevenlabs_asr] no API key (api_key empty and ELEVENLABS_API_KEY "
                "unset) — transcription requests will be rejected"
            )

    def _wav_bytes(self, audio: np.ndarray) -> bytes:
        """float32 mono @16k -> 16-bit PCM WAV bytes.

        WAV rather than the raw `pcm_s16le_16` the API also accepts: it is
        self-describing, so a sample-rate change upstream cannot silently
        produce chipmunk audio that still transcribes to plausible words.
        """
        pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(self.NUM_CHANNELS)
            w.setsampwidth(self.SAMPLE_WIDTH)
            w.setframerate(self.SAMPLE_RATE)
            w.writeframes(pcm)
        return buf.getvalue()

    def _form(self) -> dict:
        data = {
            "model_id": self.model_id,
            # Off on purpose — see the module docstring.
            "tag_audio_events": "true" if self.tag_audio_events else "false",
            # Word timings are not used by the conversation path and only make
            # the response bigger.
            "timestamps_granularity": "word" if self.diarize else "none",
        }
        if self.language_code:
            data["language_code"] = self.language_code
        if self.diarize:
            data["diarize"] = "true"
            if self.num_speakers and self.num_speakers > 1:
                data["num_speakers"] = str(self.num_speakers)
        return data

    def transcribe_np(self, audio: np.ndarray) -> str:
        if audio is None or len(audio) == 0:
            return ""
        files = [("file", ("utterance.wav", self._wav_bytes(audio), "audio/wav"))]
        data = self._form()
        # httpx sends repeated keys for a list field; keyterms is an array param.
        for t in self.keyterms:
            files.append(("keyterms", (None, t)))
        headers = {"xi-api-key": self.api_key} if self.api_key else {}
        try:
            r = self._client.post(
                self.base_url, headers=headers, files=files, data=data
            )
        except Exception as e:  # noqa: BLE001 — never take the turn down
            logger.warning(f"[elevenlabs_asr] request failed: {e!r}")
            return ""
        if r.status_code != 200:
            logger.warning(f"[elevenlabs_asr] {r.status_code}: {r.text[:300]}")
            return ""
        try:
            js = r.json()
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[elevenlabs_asr] bad JSON: {e!r}")
            return ""

        text = (js.get("text") or "").strip()
        if self.diarize:
            self._log_speakers(js, text)
        return text

    def _log_speakers(self, js: dict, text: str) -> None:
        """Report who spoke, without acting on it yet.

        Diarisation is measured before it is trusted: whether speaker ids are
        stable enough across segments to filter on is a question about this room
        and this microphone, and the log is how that gets answered. Acting on an
        unverified id would trade a known problem for a subtler one.
        """
        words = js.get("words") or []
        counts: dict = {}
        for w in words:
            if w.get("type") not in (None, "word"):
                continue
            sid = w.get("speaker_id") or "?"
            counts[sid] = counts.get(sid, 0) + 1
        if counts:
            spread = ", ".join(f"{k}:{v}" for k, v in sorted(counts.items()))
            logger.info(f"[elevenlabs_asr] speakers {spread} | {text[:60]!r}")

    def transcribe(self, audio_path: str) -> str:  # pragma: no cover - unused path
        raise NotImplementedError("elevenlabs_asr transcribes numpy audio only")
