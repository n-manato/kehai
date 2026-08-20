"""Estimate how long a sentence takes to speak, BEFORE TTS synthesis.

Used to size the bespoke body-motion generation request to the sentence's length
(the motion clip should span the spoken sentence). This is fired in parallel with
TTS in tts_manager.speak(), so the real audio duration isn't known yet — a
text-based estimate is all we have. The frontend later uses the ACTUAL
audio.duration for playback sizing; this estimate only sizes the generation
request. HY-Motion caps clips at 5s, so we clamp to [MIN_DUR, MAX_DUR].
"""

import re

# Rough speaking rates. Japanese/Chinese TTS reads ~5 characters/sec; Latin text
# ~15 chars/sec (~2.7 words/sec — EDIGSV measured 14.7 cps on real renders).
# Module-level so Phase-4 tuning is one place.
CPS_CJK = 5.0
CPS_LATIN = 15.0
# Fixed per-sentence overhead for LATIN speech: onset breath + final lengthening
# that chars/cps misses on SHORT English sentences ("Good morning." estimated
# 0.9s vs ~1.3s actual) — the drift made motion pacing run ahead of the voice.
# CJK keeps its original (user-approved) tuning with no overhead term.
LATIN_SENT_OVERHEAD = 0.35
MIN_DUR = 1.0
MAX_DUR = 5.0  # HY-Motion max clip length (motion-service/config.py MAX_DURATION_SEC)

# Same punctuation the TTS empty-text guard strips (tts_manager.speak).
_STRIP = re.compile(r"[\s.,!?，。！？'\"』」）】、～~…·・:：;；]+")
_CJK = re.compile(r"[぀-ヿ一-鿿ｦ-ﾟ㐀-䶿]")


def estimate_speech_duration(text: str, lang: str | None = None) -> float:
    """Return an estimated spoken duration (seconds), clamped to [MIN_DUR, MAX_DUR]."""
    core = _STRIP.sub("", text or "")
    n = len(core)
    if n == 0:
        return MIN_DUR
    if lang:
        cjk = lang.lower().startswith(("ja", "zh", "ko"))
    else:
        # Auto-detect: majority CJK characters -> CJK rate.
        cjk = len(_CJK.findall(core)) * 2 >= n
    cps = CPS_CJK if cjk else CPS_LATIN
    dur = n / cps
    if not cjk:
        dur += LATIN_SENT_OVERHEAD
    return max(MIN_DUR, min(MAX_DUR, dur))
