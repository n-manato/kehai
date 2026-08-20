"""Turn ElevenLabs character timings into mouth shapes.

The avatar's lip sync currently opens one jaw morph by how loud the audio is.
That is a volume meter, not a mouth: "eee" and "ooo" look identical, and a
loud consonant opens the jaw wider than a quiet vowel. Meanwhile the Chancelor
model ships the full Oculus viseme set — aa, E, ih, oh, ou, PP, FF, TH, DD, kk,
CH, SS, nn, RR — and nothing drives any of it.

ElevenLabs returns, alongside the audio, exactly what is missing: which
character is being spoken and when. Mapping character -> viseme is enough to get
real mouth shapes, and it costs nothing extra because the timings ride along
with the audio request.

WHY GRAPHEMES AND NOT PHONEMES. A proper job would run the text through a
pronouncer: English spelling is not pronunciation ("though", "tough"). But a
phonemiser is a dictionary, a language dependency and a whole class of new
failures, and at 24-30 mouth shapes a second nobody reads the mouth closely
enough for "gh" to matter. What people DO notice is a mouth that moves in step
with the voice, opens on vowels and closes on stops — which graphemes give for
free. Digraphs (th, ch, sh, ph) are handled because those are the ones that look
wrong often enough to see.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

# The Oculus/Meta viseme names, which is what avatar pipelines (Ready Player Me,
# Avaturn, Character Creator) ship. `sil` is the closed/rest mouth.
VISEMES = (
    "aa",
    "E",
    "ih",
    "oh",
    "ou",
    "PP",
    "FF",
    "TH",
    "DD",
    "kk",
    "CH",
    "SS",
    "nn",
    "RR",
    "sil",
)

# Two-character sequences first — "th" must not be read as t + h.
_DIGRAPHS = {
    "th": "TH",
    "ch": "CH",
    "sh": "CH",
    "ph": "FF",
    "wh": "ou",
    "ck": "kk",
    "ng": "nn",
    "qu": "kk",
}

_SINGLES = {
    # vowels
    "a": "aa",
    "e": "E",
    "i": "ih",
    "o": "oh",
    "u": "ou",
    "y": "ih",
    # bilabials — the mouth visibly CLOSES; getting these right is most of the
    # difference between "talking" and "chewing"
    "p": "PP",
    "b": "PP",
    "m": "PP",
    # labiodentals
    "f": "FF",
    "v": "FF",
    # alveolar stops / nasals
    "t": "DD",
    "d": "DD",
    "n": "nn",
    "l": "nn",
    # velars
    "k": "kk",
    "g": "kk",
    "c": "kk",
    "x": "kk",
    # sibilants
    "s": "SS",
    "z": "SS",
    "j": "CH",
    # approximants
    "r": "RR",
    "w": "ou",
    "h": "E",
}


# AUDIO TAGS. eleven_v3 takes inline direction like "[laughs] Well, ..." and does
# NOT say the bracketed word — verified by transcribing the output, which came
# back as "Well, that is one way to put it" for both v3 and flash.
#
# The alignment, however, DOES contain those characters, with real time spans
# (measured: `[laughs]` held 0.52 s at the head of the clip, which is the laugh
# itself). Mapping them as graphemes would have the avatar visibly mouth the word
# "laughs" over the sound of laughing — so the span is replaced by ONE cue.
#
# Which cue depends on what the tag is, because the two kinds are not alike:
#   * tags that ARE a sound (a laugh, a sigh, a gasp) own that stretch of audio,
#     and a shut mouth over an audible laugh reads as broken. They get an open
#     mouth for the whole span.
#   * tags that direct DELIVERY ([whispers], [cheerfully]) make no sound of their
#     own; the span is a beat before the line. They get `sil`.
# An unknown tag is treated as delivery — a still mouth is the safer wrong guess.
_VOCAL_TAGS = (
    "laugh",
    "chuckl",
    "giggl",
    "sigh",
    "gasp",
    "cough",
    "gulp",
    "throat",
    "cry",
    "sob",
    "snort",
    "yawn",
    "hum",
    "breath",
    "exhale",
    "inhale",
)


def _tag_cue(tag: str) -> str:
    """`sil` or an open mouth for the whole of a `[...]` span."""
    t = tag.lower()
    return "aa" if any(w in t for w in _VOCAL_TAGS) else "sil"


@dataclass
class VisemeCue:
    """One mouth shape: `v` from `t0` to `t1`, seconds from the clip start."""

    v: str
    t0: float
    t1: float


def _viseme_for(ch: str, nxt: str) -> str | None:
    two = (ch + nxt).lower()
    if two in _DIGRAPHS:
        return _DIGRAPHS[two]
    c = ch.lower()
    if c in _SINGLES:
        return _SINGLES[c]
    if c.isspace() or not c.isalnum():
        return "sil"
    # Digits and anything else are spoken but unmapped; a neutral open beats a
    # frozen mouth.
    return "aa" if c.isdigit() else None


def cues_from_alignment(
    characters: List[str],
    starts: List[float],
    ends: List[float],
    min_dur: float = 0.045,
) -> List[VisemeCue]:
    """Character timings -> viseme cues, merged and floored.

    Two things are done to the raw per-character stream, both because a mouth
    that changes shape every 30 ms reads as a flutter rather than as speech:

    * RUNS ARE MERGED. "ss" or a vowel spanning several characters is one shape
      held, not the same shape retriggered.
    * SHORT CUES ARE FLOORED to `min_dur`. Below roughly 45 ms a morph cannot
      visibly reach its target before the next one starts, so it only adds
      noise. The floor borrows time from the following cue rather than shifting
      everything, so the mouth never drifts out of sync with the audio.
    """
    n = min(len(characters), len(starts), len(ends))
    raw: List[VisemeCue] = []
    i = 0
    while i < n:
        ch = characters[i]

        # An audio tag is direction, not speech (see _VOCAL_TAGS): swallow the
        # whole `[...]` run and stand one cue in its place. An unclosed bracket is
        # left to fall through and be spelled, since it is then just a character.
        if ch == "[":
            close = -1
            for j in range(i + 1, min(n, i + 40)):
                if characters[j] == "]":
                    close = j
                    break
            if close > i:
                tag = "".join(characters[i + 1 : close])
                t0, t1 = float(starts[i]), float(ends[close])
                if t1 > t0:
                    raw.append(VisemeCue(_tag_cue(tag), t0, t1))
                i = close + 1
                continue

        nxt = characters[i + 1] if i + 1 < n else ""
        v = _viseme_for(ch, nxt)
        i += 1
        if v is None:
            continue
        t0, t1 = float(starts[i - 1]), float(ends[i - 1])
        if t1 <= t0:
            continue
        if raw and raw[-1].v == v and t0 - raw[-1].t1 < 0.02:
            raw[-1] = VisemeCue(v, raw[-1].t0, t1)  # merge the run
        else:
            raw.append(VisemeCue(v, t0, t1))

    out: List[VisemeCue] = []
    for i, c in enumerate(raw):
        t1 = c.t1
        if t1 - c.t0 < min_dur:
            # Extend, but never past where the next cue starts — the shapes must
            # stay in the order and the places the voice put them.
            limit = raw[i + 1].t0 if i + 1 < len(raw) else c.t0 + min_dur
            t1 = min(max(t1, c.t0 + min_dur), max(limit, c.t0 + 0.001))
        out.append(VisemeCue(c.v, round(c.t0, 4), round(t1, 4)))
    return out


def cues_to_payload(cues: List[VisemeCue], offset: float = 0.0) -> List[dict]:
    """Compact wire form. `offset` shifts a sentence onto the turn's timeline."""
    return [
        {"v": c.v, "t": round(c.t0 + offset, 4), "d": round(c.t1 - c.t0, 4)}
        for c in cues
    ]
