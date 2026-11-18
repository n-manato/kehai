"""Run two recognisers on the same audio; answer with one, record both.

"Is ElevenLabs more accurate than whisper?" cannot be settled by a benchmark or
by one clean sample — it is a question about THIS microphone, THIS room and a
Japanese speaker's English, and the only honest source of truth is the audio
that actually arrives. Measured on one 2.9s file the two were a coin flip and
whisper was marginally better, which is exactly why a real sample is needed
before switching the thing that decides what the avatar hears.

So: the PRIMARY engine answers the turn, unchanged and unslowed. The CHALLENGER
runs on a background thread and both transcripts go to a JSONL file for later
comparison. Nothing about the conversation changes while this is on, which is
the point — a trial that alters behaviour cannot be left running long enough to
collect anything.

Three properties this must have, and each is a way it could quietly go wrong:

* NO ADDED LATENCY. The challenger is never awaited. `transcribe_np` returns as
  soon as the primary does.
* NO ADDED FAILURE. Any challenger error is logged and dropped. A dead API key
  or an exhausted quota must not cost a single turn.
* NO UNBOUNDED WORK. If the challenger falls behind, its queue drops the oldest
  entries rather than growing. A backlog would keep spending credits on audio
  from minutes ago.
"""

import json
import os
import queue
import threading
import time
from typing import Optional

import numpy as np
from loguru import logger

from .asr_interface import ASRInterface


class VoiceRecognition(ASRInterface):
    def __init__(
        self,
        primary: str = "gateway_whisper_asr",
        challenger: str = "elevenlabs_asr",
        log_path: str = "logs/asr_shadow.jsonl",
        enabled: bool = True,
        max_queue: int = 8,
        _parent: Optional[dict] = None,
    ):
        from .asr_factory import ASRFactory

        parent = _parent or {}
        self.primary_name = primary
        self.challenger_name = challenger
        self.log_path = log_path
        # Env override so the comparison can be stopped without a config edit and
        # a restart — it costs money per turn.
        env = os.getenv("ASR_SHADOW", "")
        self.enabled = (env == "1") if env else enabled

        def build(name: str):
            sub = parent.get(name)
            if sub is None:
                raise ValueError(
                    f"shadow_asr: no '{name}' section in asr_config to build from"
                )
            return ASRFactory.get_asr_system(
                name, **{k: v for k, v in sub.items() if v is not None}
            )

        self._primary = build(primary)
        self._challenger = build(challenger) if self.enabled else None

        self._q: "queue.Queue[tuple]" = queue.Queue(maxsize=max_queue)
        self._worker: Optional[threading.Thread] = None
        if self.enabled:
            os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)
            self._worker = threading.Thread(target=self._run, daemon=True)
            self._worker.start()
            logger.info(
                f"[asr-shadow] {primary} (answers) vs {challenger} (recorded) "
                f"-> {self.log_path}"
            )
        else:
            logger.info(f"[asr-shadow] disabled; {primary} only")

    # --- the turn path -------------------------------------------------------

    def transcribe_np(self, audio: np.ndarray) -> str:
        t0 = time.time()
        text = self._primary.transcribe_np(audio)
        ms = int((time.time() - t0) * 1000)
        if self.enabled and audio is not None and len(audio):
            try:
                # copy(): the caller owns its buffer and may reuse it, and the
                # worker reads it after we return.
                self._q.put_nowait((audio.copy(), text, ms, time.time()))
            except queue.Full:
                # Drop the OLDEST so the newest audio is the one compared —
                # stale audio is worth less and still costs credits.
                try:
                    self._q.get_nowait()
                    self._q.put_nowait((audio.copy(), text, ms, time.time()))
                except Exception:  # noqa: BLE001
                    pass
        return text

    async def async_transcribe_fast(self, audio: np.ndarray) -> str:
        """The primary's quick model, when it has one (gateway_whisper_asr). This
        wrapper did not pass the method through, so "the quick transcript" ran
        large-v3/beam 5 after all (2026-11-13, two replays measured the wrong thing)."""
        quick = getattr(self._primary, "async_transcribe_fast", None)
        if quick is None:
            return await self.async_transcribe_np(audio)
        return await quick(audio)

    @property
    def last_fast(self):
        return getattr(self._primary, "last_fast", (0, 0, 0))

    # --- the background comparison ------------------------------------------

    def _run(self) -> None:
        while True:
            try:
                audio, primary_text, primary_ms, ts = self._q.get()
            except Exception:  # noqa: BLE001
                continue
            try:
                t0 = time.time()
                other = self._challenger.transcribe_np(audio)
                other_ms = int((time.time() - t0) * 1000)
                self._record(audio, ts, primary_text, primary_ms, other, other_ms)
            except Exception as e:  # noqa: BLE001 — a trial must never bite
                logger.debug(f"[asr-shadow] challenger failed: {e!r}")

    @staticmethod
    def _norm(s: str) -> str:
        """Compare on words, not punctuation: the two engines punctuate
        differently and that difference is not an accuracy difference."""
        keep = [c.lower() for c in s if c.isalnum() or c.isspace() or ord(c) > 0x3000]
        return " ".join("".join(keep).split())

    def _record(self, audio, ts, a_text, a_ms, b_text, b_ms) -> None:
        a_n, b_n = self._norm(a_text), self._norm(b_text)
        row = {
            "t": round(ts, 3),
            "dur_s": round(len(audio) / 16000.0, 2),
            "primary": {"engine": self.primary_name, "text": a_text, "ms": a_ms},
            "challenger": {"engine": self.challenger_name, "text": b_text, "ms": b_ms},
            "same": a_n == b_n,
            # Both empty is agreement but says nothing about accuracy — flagged so
            # it can be excluded when scoring.
            "both_empty": not a_n and not b_n,
        }
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        except Exception as e:  # noqa: BLE001
            logger.debug(f"[asr-shadow] log write failed: {e!r}")
        if not row["both_empty"] and not row["same"]:
            logger.info(
                f"[asr-shadow] differ ({row['dur_s']}s) "
                f"{self.primary_name}={a_text[:60]!r} | "
                f"{self.challenger_name}={b_text[:60]!r}"
            )

    def transcribe(self, audio_path: str) -> str:  # pragma: no cover
        return self._primary.transcribe(audio_path)
