"""Audio tags must reach the voice intact — or not at all. Never as spoken words.

`keep_audio_tags` lets ElevenLabs v3 direction like "[laughs]" survive the TTS
text pipeline. The dangerous failure is not "the tag was dropped", it is "the
brackets were dropped and the word was left" — the avatar then says the word
"laughs" out loud, which is worse than the feature being off.

Two passes destroy a tag left in the text, and both are silent:
  * speechify strips "[" and "]" as unspeakable characters
  * the Japanese latin->kana pass transliterates any latin run

Both were hit for real while building this. The pipeline now cuts tags out
before any pass runs, and this pins that: for every case, no tag word may appear
outside brackets in the result.

Run: .venv/Scripts/python.exe scripts/check_audio_tags.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from kehai.utils.tts_preprocessor import tts_filter  # noqa: E402

# Words that must never be spoken. Anything in AUDIO_TAGS could leak this way;
# these are the ones a real reply is most likely to contain.
LEAK_WORDS = ("laughs", "whispers", "sighs", "clears", "throat", "zqaudiotag")

CASES = [
    # (text, language, expected with keep_audio_tags=True)
    ("[joy] Well [laughs] that is fine.", "en", "Well [laughs] that is fine."),
    ("[laughs] hi", "en", "[laughs] hi"),
    ("hi [whispers] there", "en", "hi [whispers] there"),
    ("[sighs] a [clears throat] b", "en", "[sighs] a [clears throat] b"),
    # the app's own emotion protocol still wins — the face is driven from these
    ("[surprise_shock] oh", "en", "oh"),
    ("[neutral] fine", "en", "fine"),
    # unknown bracket content is dropped exactly as before
    ("[stage left] go", "en", "go"),
    ("no tags at all here.", "en", "no tags at all here."),
    # the latin->kana pass is the one that ate a placeholder scheme
    ("[laughs] こんにちは", "ja", "[laughs] こんにちは"),
]


def run(text: str, language: str, keep: bool) -> str:
    return tts_filter(
        text,
        remove_special_char=True,
        ignore_brackets=True,
        ignore_parentheses=True,
        ignore_asterisks=True,
        ignore_angle_brackets=True,
        language=language,
        keep_audio_tags=keep,
    )


def main() -> int:
    failures = []
    for text, lang, want in CASES:
        got = run(text, lang, keep=True)
        if got != want:
            failures.append(f"{text!r} ({lang}) -> {got!r}, expected {want!r}")
            continue
        outside = re.sub(r"\[[^\]]*\]", "", got).lower()
        for w in LEAK_WORDS:
            if w in outside:
                failures.append(f"{text!r} -> {got!r}: {w!r} would be SPOKEN")
        print(f"  ok  {text!r} -> {got!r}")

    # Off is the default, and must behave exactly as it did before any of this.
    off = run("[joy] Well [laughs] that is fine.", "en", keep=False)
    if off != "Well that is fine.":
        failures.append(f"keep_audio_tags=False changed behaviour: {off!r}")
    else:
        print(f"  ok  keep_audio_tags=False unchanged -> {off!r}")

    for f in failures:
        print("FAIL:", f, file=sys.stderr)
    if failures:
        return 1
    print("\nPASS: audio tags survive intact, and never leak as speech")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
