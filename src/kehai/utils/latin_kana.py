"""Latin -> katakana for TTS: GSV (the Rem voice) silently SKIPS latin text —
an English-heavy sentence measured 0.5s of audio for ~8s of content, which the
user heard as "字幕には出るのに調べた内容をしゃべらない". The LLM is instructed
to write katakana, but it still leaks brand names (Qualcomm, STM32...), so this
converts whatever remains, deterministically, JUST for the spoken text (the
subtitle keeps the original spelling).

Strategy per word:
  1. alkana dictionary (120k English words -> katakana), case-insensitive;
  2. short/ALL-CAPS tokens -> letter-by-letter spelling (SoC -> エスオーシー);
  3. camel/compound split, each part via 1/2;
  4. unknown parts -> romaji-style reading (qualcomm -> クアルコム近似): rough
     but AUDIBLE, which beats silence.
Digits are left alone (GSV reads them natively in a Japanese sentence).
"""

from __future__ import annotations

import re

try:
    import alkana  # dictionary lookup; optional at runtime
except Exception:  # noqa: BLE001 — degrade to romaji/letter fallbacks
    alkana = None

_LETTER = {
    "a": "エー", "b": "ビー", "c": "シー", "d": "ディー", "e": "イー",
    "f": "エフ", "g": "ジー", "h": "エイチ", "i": "アイ", "j": "ジェー",
    "k": "ケー", "l": "エル", "m": "エム", "n": "エヌ", "o": "オー",
    "p": "ピー", "q": "キュー", "r": "アール", "s": "エス", "t": "ティー",
    "u": "ユー", "v": "ブイ", "w": "ダブリュー", "x": "エックス",
    "y": "ワイ", "z": "ゼット",
}

# Romaji-ish syllable tables for the unknown-word fallback.
_YOUON = {
    "kya": "キャ", "kyu": "キュ", "kyo": "キョ", "sha": "シャ", "shu": "シュ",
    "sho": "ショ", "cha": "チャ", "chu": "チュ", "cho": "チョ", "nya": "ニャ",
    "nyu": "ニュ", "nyo": "ニョ", "hya": "ヒャ", "hyu": "ヒュ", "hyo": "ヒョ",
    "mya": "ミャ", "myu": "ミュ", "myo": "ミョ", "rya": "リャ", "ryu": "リュ",
    "ryo": "リョ", "gya": "ギャ", "gyu": "ギュ", "gyo": "ギョ", "ja": "ジャ",
    "ju": "ジュ", "jo": "ジョ", "bya": "ビャ", "byu": "ビュ", "byo": "ビョ",
    "pya": "ピャ", "pyu": "ピュ", "pyo": "ピョ",
    "tsu": "ツ", "shi": "シ", "chi": "チ", "thi": "ティ", "dhi": "ディ",
    "fa": "ファ", "fi": "フィ", "fe": "フェ", "fo": "フォ",
    "va": "ヴァ", "vi": "ヴィ", "ve": "ヴェ", "vo": "ヴォ",
    "wi": "ウィ", "we": "ウェ", "wo": "ウォ", "qua": "クア", "que": "クエ",
    "qui": "クイ", "quo": "クオ",
}
_BASE = {
    "k": "カキクケコ", "s": "サシスセソ", "t": "タチツテト", "n": "ナニヌネノ",
    "h": "ハヒフヘホ", "m": "マミムメモ", "y": "ヤイユイェヨ", "r": "ラリルレロ",
    "w": "ワウィウウェヲ", "g": "ガギグゲゴ", "z": "ザジズゼゾ",
    "d": "ダヂヅデド", "b": "バビブベボ", "p": "パピプペポ", "f": "ファフィフフェフォ",
    "v": "ヴァヴィヴヴェヴォ", "j": "ジャジジュジェジョ", "c": "カシクセコ",
    "l": "ラリルレロ", "x": "クサクシクスクセクソ",
}
_VOWEL = {"a": 0, "i": 1, "u": 2, "e": 3, "o": 4}
_VOWEL_KANA = "アイウエオ"
# default vowel used when a consonant has no following vowel ("m" -> ム)
_TRAIL = {
    "k": "ク", "s": "ス", "t": "ト", "n": "ン", "h": "フ", "m": "ム",
    "r": "ル", "g": "グ", "z": "ズ", "d": "ド", "b": "ブ", "p": "プ",
    "f": "フ", "v": "ヴ", "j": "ジ", "c": "ク", "l": "ル", "w": "ウ",
    "y": "イ", "q": "ク", "x": "クス",
}


def _romaji_word(w: str) -> str:
    """Read an unknown word AS IF it were romaji — rough but audible."""
    s = w.lower()
    out = []
    i = 0
    n = len(s)
    while i < n:
        # doubled consonant -> ッ (nasals just merge: comm -> コム)
        if i + 1 < n and s[i] == s[i + 1] and s[i].isalpha() and s[i] not in "aeiou":
            if s[i] in "mn":
                i += 1  # drop the first nasal silently
            else:
                out.append("ッ")
                i += 1
            continue
        for ln in (3, 2):  # youon / digraphs first
            if s[i : i + ln] in _YOUON:
                out.append(_YOUON[s[i : i + ln]])
                i += ln
                break
        else:
            ch = s[i]
            if ch in _VOWEL:
                out.append(_VOWEL_KANA[_VOWEL[ch]])
                i += 1
            elif ch == "n" and (i + 1 >= n or s[i + 1] not in "aeioy"):
                out.append("ン")
                i += 1
            elif ch in _BASE and i + 1 < n and s[i + 1] in _VOWEL:
                row = _BASE[ch]
                # rows may hold multi-char entries (fa フ ァ...): index by kana
                if ch in ("f", "v", "j", "w", "x"):
                    pairs = {
                        "f": ["ファ", "フィ", "フ", "フェ", "フォ"],
                        "v": ["ヴァ", "ヴィ", "ヴ", "ヴェ", "ヴォ"],
                        "j": ["ジャ", "ジ", "ジュ", "ジェ", "ジョ"],
                        "w": ["ワ", "ウィ", "ウ", "ウェ", "ヲ"],
                        "x": ["クサ", "クシ", "クス", "クセ", "クソ"],
                    }[ch]
                    out.append(pairs[_VOWEL[s[i + 1]]])
                else:
                    out.append(row[_VOWEL[s[i + 1]]])
                i += 2
            elif ch.isalpha():
                out.append(_TRAIL.get(ch, ""))
                i += 1
            else:
                out.append(ch)  # digits/symbols pass through
                i += 1
    return "".join(out)


def _dict_kana(w: str) -> str | None:
    if alkana is None:
        return None
    return alkana.get_kana(w) or alkana.get_kana(w.lower())


def _word_kana(w: str) -> str:
    k = _dict_kana(w)
    if k:
        return k
    # short or ALL-CAPS token: spell the letters (SoC -> エスオーシー, AI -> エーアイ)
    letters_only = re.sub(r"[^A-Za-z]", "", w)
    if w.isupper() or len(letters_only) <= 3:
        return "".join(_LETTER.get(c.lower(), c) for c in w)
    # camel / compound split, convert each part
    parts = re.findall(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+", w)
    if len(parts) > 1:
        return "".join(
            p if p.isdigit() else (_dict_kana(p) or _word_kana(p)) for p in parts
        )
    return _romaji_word(w)


_LATIN_RE = re.compile(r"[A-Za-z][A-Za-z'\-]*")


def latin_to_kana(text: str) -> str:
    """Replace every latin word in `text` with a katakana reading."""
    return _LATIN_RE.sub(lambda m: _word_kana(m.group(0)), text)
