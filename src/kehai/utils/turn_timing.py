"""Per-turn TTFA instrumentation: one log line per voice turn showing where the
time between "user stopped speaking" and "first TTS audio" actually went.

    [TTFA] asr=xxxms llm_first_sentence=xxxms tts_first_chunk=xxxms
           TOTAL(endpoint->first_pcm)=xxxms (+ client preroll: 0.05s render tab / 0.25s browser)

Marks are idempotent (first write wins) so only the FIRST sentence of a turn is
measured. Singleton — the VAD handler and tts_manager share it without wiring.
"""

import time

from loguru import logger


class TurnTiming:
    def __init__(self) -> None:
        self._t: dict = {}
        self._active = False

    def start(self) -> None:
        """Call at the moment the turn's endpoint fires (TURN_END)."""
        self._t = {"turn_end": time.monotonic()}
        self._active = True

    def mark(self, name: str) -> None:
        if self._active and name not in self._t:
            self._t[name] = time.monotonic()

    def finish_first_audio(self) -> None:
        """Call when the first PCM chunk of the turn's first sentence arrives."""
        if not self._active:
            return
        self._active = False
        t = self._t

        def d(a: str, b: str) -> str:
            if a in t and b in t:
                return f"{(t[b] - t[a]) * 1000:.0f}ms"
            return "?"

        logger.info(
            "[TTFA] asr={} | llm_first_sentence={} | tts_first_chunk={} | "
            "TOTAL endpoint->first_pcm={} (+ client preroll: 0.05s render tab / 0.25s browser)".format(
                d("turn_end", "asr_done"),
                d("asr_done", "first_sentence"),
                d("first_sentence", "first_pcm"),
                d("turn_end", "first_pcm"),
            )
        )


turn_timing = TurnTiming()
