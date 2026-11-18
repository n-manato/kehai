"""Per-turn TTFA instrumentation: one log line per voice turn showing where the
time between "user stopped speaking" and "first TTS audio" actually went.

    [TTFA] asr=xxxms llm_first_sentence=xxxms tts_first_chunk=xxxms
           TOTAL(endpoint->first_pcm)=xxxms (+ client preroll: 0.05s render tab / 0.25s browser)

Marks are idempotent (first write wins) so only the FIRST sentence of a turn is
measured. Singleton — the VAD handler and tts_manager share it without wiring.
"""

import time

from loguru import logger


def wall_ms(t: float | None = None) -> str:
    """HH:MM:SS.mmm for log lines that the latency bench lines up across stages."""
    t = time.time() if t is None else t
    return time.strftime("%H:%M:%S", time.localtime(t)) + f".{int(t * 1000) % 1000:03d}"


class TurnTiming:
    def __init__(self) -> None:
        self._t: dict = {}
        self._wall = 0.0
        self._active = False

    def start(self) -> None:
        """Call at the moment the turn's endpoint fires (TURN_END)."""
        self._t = {"turn_end": time.monotonic()}
        self._wall = time.time()
        self._active = True

    def pending(self) -> bool:
        """A turn has started and its first voice has not been timed yet."""
        return self._active

    def mark(self, name: str) -> None:
        if self._active and name not in self._t:
            self._t[name] = time.monotonic()

    def finish_first_audio(self, note: str = "") -> None:
        """Call when the first PCM chunk of the turn's first sentence arrives."""
        if not self._active:
            return
        self._active = False
        t = self._t

        def d(a: str, b: str) -> str:
            if a in t and b in t:
                return f"{(t[b] - t[a]) * 1000:.0f}ms"
            return "?"

        def w(name: str) -> str:
            if name in t:
                ts = self._wall + (t[name] - t["turn_end"])
                return time.strftime("%H:%M:%S", time.localtime(ts)) + f".{int(ts * 1000) % 1000:03d}"
            return "?"

        logger.info(
            "[TTFA] asr={} | llm_first_sentence={} | tts_first_chunk={} | "
            "TOTAL endpoint->first_pcm={} (+ client preroll: localStorage.streamPreroll, else 0.05s render tab / 0.25s browser){} | wall endpoint@{} first_pcm@{}".format(
                d("turn_end", "asr_done"),
                d("asr_done", "first_sentence"),
                d("first_sentence", "first_pcm"),
                d("turn_end", "first_pcm"),
                f" [{note}]" if note else "",
                w("turn_end"),
                w("first_pcm"),
            )
        )


turn_timing = TurnTiming()
