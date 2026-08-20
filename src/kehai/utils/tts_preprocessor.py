import os
import re
import unicodedata
from loguru import logger
from ..translate.translate_interface import TranslateInterface

# ---- speechify: FORCE the spoken text into a TTS-readable form ------------
# The system prompt asks the LLM for plain speech, but that is advisory — a
# model can still emit markdown, URLs, code or the wrong script. This pass is
# the deterministic guarantee. Subtitles/memory keep the original text.

_CODE_FENCE_RE = re.compile(r"```.*?(```|$)", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`([^`\n]*)`")
_URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")
# Markdown line furniture: headings, bullets, numbered lists, quotes, rules.
_MD_LINE_RE = re.compile(
    r"^\s*(#{1,6}\s+|[-*+•・●]\s+|\d{1,2}[.)]\s+|>\s+|[-=_*]{3,}\s*$)",
    re.MULTILINE,
)
_MD_EMPHASIS_RE = re.compile(r"(\*\*|__|~~)")
_CJK_RE = re.compile(r"[぀-ヿ一-鿿ｦ-ﾟ㐀-䶿]+")
# Characters no TTS frontend speaks, which SURVIVE the category-based special
# char filter because Unicode classes them as punctuation: kaomoji debris
# (＼(^o^)／ -> "\/" after NFKC + paren-stripping), slashes, carets, tildes,
# backticks, braces. Nuked outright from the SPOKEN text.
_UNSPEAKABLE_RE = re.compile(r"[\\/^~_|<>*`{}\[\]§†‡•¤¶]+")
# Orphan emoticon leftovers once their partner chars are stripped: a colon or
# semicolon floating between spaces / at line edges (":)" -> " : ").
_ORPHAN_PUNCT_RE = re.compile(r"(?:(?<=\s)|^)[:;'\"-]+(?=\s|$)")

# Symbols the GSV frontends read badly or skip — verbalize the common ones.
_SYMBOLS = {
    "ja": {
        "%": "パーセント",
        "％": "パーセント",
        "&": " アンド ",
        "℃": "度",
        "+": " プラス ",
        "=": " イコール ",
    },
    "en": {
        "%": " percent",
        "％": " percent",
        "&": " and ",
        "℃": " degrees",
        "+": " plus ",
        "=": " equals ",
    },
}


# Proper nouns the TTS frontends mangle. Applied to the SPOKEN text only, so subtitles,
# chat history and memory keep the real spelling. The Japanese latin->kana gate runs AFTER
# speechify and would otherwise spell the name out letter by letter:
#   "AI-CCORE" -> エーアイ-シーシーオーアールイー  ("A.I. see-see-oh-ar-ee")
# Matches AI-CCORE / AICCORE / AI CORE and any case.
#
# The model likes to gloss the name with its reading — "AI-CCORE（エーアイ・シーコア）は…".
# Left alone the gloss is spoken too, so the corrected pronunciation is immediately
# followed by the wrong one ("eeai-koa, eeai-shiikoa"); in English it also left a
# stranded "( )" behind once the CJK strip emptied it. Swallow a parenthetical that
# is nothing but kana/中黒 right after the name.
_GLOSS = r"(?:\s*[（(][぀-ヿー・\s]+[）)])?"
_SAY_AS = [
    (
        re.compile(r"AI[\s\-‐-―_]*C{1,2}ORE" + _GLOSS, re.IGNORECASE),
        {"ja": "エーアイコア", "en": "AI core"},
    ),
]


def say_as(text: str, language: str = "ja") -> str:
    """Rewrite proper nouns into their intended PRONUNCIATION."""
    for rx, by_lang in _SAY_AS:
        rep = by_lang.get(language, by_lang.get("en", ""))
        text = rx.sub(rep, text)
    return text


def speechify(text: str, language: str = "ja") -> str:
    """Deterministically convert arbitrary LLM output into speakable text:
    drop code, name URLs/emails, strip markdown furniture, verbalize common
    symbols, and (en) remove CJK runs the English GSV frontend cannot speak."""
    # Pronunciation of proper nouns, before anything else can split them up.
    text = say_as(text, language)
    # Code first (fences may span lines; their content is never speakable).
    text = _CODE_FENCE_RE.sub(" ", text)
    text = _INLINE_CODE_RE.sub(r"\1", text)  # keep inline-code WORDS, drop ticks
    # URLs / emails become a plain spoken word instead of letter salad.
    link_word = "リンク" if language == "ja" else "the link"
    mail_word = "メールアドレス" if language == "ja" else "the email address"
    text = _URL_RE.sub(link_word, text)
    text = _EMAIL_RE.sub(mail_word, text)
    # Markdown furniture.
    text = _MD_LINE_RE.sub("", text)
    text = _MD_EMPHASIS_RE.sub("", text)
    text = text.replace("|", " ").replace("#", " ")
    # Symbol verbalization (conservative set).
    for sym, word in _SYMBOLS.get(language, _SYMBOLS["ja"]).items():
        if sym in text:
            text = text.replace(sym, word)
    # The English GSV frontend silently SKIPS CJK — drop it from the SPOKEN
    # text (subtitles keep it) so a stray 日本語 leak doesn't become silence.
    if language != "ja":
        text = _CJK_RE.sub(" ", text)
    # Unspeakable leftovers (kaomoji debris, slashes, carets...) and orphan
    # emoticon punctuation. Runs LAST so it catches what other passes expose.
    text = _UNSPEAKABLE_RE.sub(" ", text)
    text = _ORPHAN_PUNCT_RE.sub(" ", text)
    # Collapse whatever whitespace all of the above left behind.
    text = re.sub(r"[ \t]*\n[ \t\n]*", " ", text)
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text


def tts_filter(
    text: str,
    remove_special_char: bool,
    ignore_brackets: bool,
    ignore_parentheses: bool,
    ignore_asterisks: bool,
    ignore_angle_brackets: bool,
    translator: TranslateInterface | None = None,
    language: str = "ja",
    keep_audio_tags: bool = False,
) -> str:
    """Filter the text before TTS. Subtitles and LLM memory keep the original.

    With `keep_audio_tags`, ElevenLabs v3 direction like "[laughs]" is CUT OUT
    of the text before any of the cleaning passes run and spliced back
    afterwards, so no pass ever sees it.

    That is deliberate rather than defensive. Two passes here destroy a tag left
    in place, and both do it silently:
      * `speechify` strips "[" and "]" as unspeakable, which turns "[laughs]"
        into the bare word `laughs` — no longer acted, and now SPOKEN.
      * the Japanese latin->kana pass transliterates any latin run, so even a
        placeholder came back as ズクアウヂオタグ0ゼットキュー and got read out.
    A tag that never enters the pipeline cannot be mangled by a pass added later
    either, which a placeholder scheme could not promise.
    """
    if keep_audio_tags:
        parts = _BRACKET_RE.split(text)  # [text, inner, text, inner, ...]
        out: list[str] = []
        for i, part in enumerate(parts):
            is_tag_body = i % 2 == 1
            if is_tag_body and part.strip().lower() in AUDIO_TAGS:
                out.append(f"[{part}]")
                continue
            # Anything else goes through the normal pipeline — including a
            # bracket body that is NOT an audio tag, which the bracket filter
            # then removes exactly as it always did.
            piece = part if not is_tag_body else f"[{part}]"
            out.append(
                tts_filter(
                    piece,
                    remove_special_char,
                    ignore_brackets,
                    ignore_parentheses,
                    ignore_asterisks,
                    ignore_angle_brackets,
                    translator,
                    language,
                    keep_audio_tags=False,
                )
            )
        return re.sub(r"\s{2,}", " ", " ".join(p for p in out if p)).strip()

    if ignore_asterisks:
        try:
            text = filter_asterisks(text)
        except Exception as e:
            logger.warning(f"Error ignoring asterisks: {e}")
            logger.warning(f"Text: {text}")
            logger.warning("Skipping...")

    if ignore_brackets:
        try:
            text = filter_brackets(text)
        except Exception as e:
            logger.warning(f"Error ignoring brackets: {e}")
            logger.warning(f"Text: {text}")
            logger.warning("Skipping...")
    if ignore_parentheses:
        try:
            text = filter_parentheses(text)
        except Exception as e:
            logger.warning(f"Error ignoring parentheses: {e}")
            logger.warning(f"Text: {text}")
            logger.warning("Skipping...")
    if ignore_angle_brackets:
        try:
            text = filter_angle_brackets(text)
        except Exception as e:
            logger.warning(f"Error ignoring angle brackets: {e}")
            logger.warning(f"Text: {text}")
            logger.warning("Skipping...")
    if remove_special_char:
        try:
            text = remove_special_characters(text)
        except Exception as e:
            logger.warning(f"Error removing special characters: {e}")
            logger.warning(f"Text: {text}")
            logger.warning("Skipping...")
    if translator:
        try:
            logger.info("Translating...")
            text = translator.translate(text)
            logger.info(f"Translated: {text}")
        except Exception as e:
            logger.critical(f"Error translating: {e}")
            logger.critical(f"Text: {text}")
            logger.warning("Skipping...")

    # FORCE speakability regardless of what the LLM emitted: code/markdown/
    # URLs/symbols/wrong-script runs never reach the voice. Deterministic —
    # the prompt rules are advisory, this is the guarantee. TTS_SPEECHIFY=0
    # disables for debugging.
    if os.getenv("TTS_SPEECHIFY", "1") == "1":
        try:
            cleaned = speechify(text, language=language)
            if cleaned != text:
                logger.info(f"[speechify] {text!r} -> {cleaned!r}")
                text = cleaned
        except Exception as e:  # noqa: BLE001 — never break TTS over this
            logger.warning(f"Error speechifying text: {e}")

    # Latin -> katakana for the SPOKEN text only (subtitles/memory keep the
    # original): GSV silently skips latin words, so English brand names were
    # simply missing from the voice ("スキップされる。しゃべらない").
    # Japanese voices only — an English voice reads latin natively, and this
    # pass would katakana-ize every word. Disable with TTS_LATIN_KANA=0.
    if language == "ja" and os.getenv("TTS_LATIN_KANA", "1") == "1":
        try:
            from .latin_kana import latin_to_kana

            converted = latin_to_kana(text)
            if converted != text:
                logger.info(f"[latin-kana] {text!r} -> {converted!r}")
                text = converted
        except Exception as e:  # noqa: BLE001 — never break TTS over this
            logger.warning(f"Error converting latin to kana: {e}")

    logger.debug(f"Filtered text: {text}")
    return text


def remove_special_characters(text: str) -> str:
    """
    Filter text to remove all non-letter, non-number, and non-punctuation characters.

    Args:
        text (str): The text to filter.

    Returns:
        str: The filtered text.
    """
    normalized_text = unicodedata.normalize("NFKC", text)

    def is_valid_char(char: str) -> bool:
        category = unicodedata.category(char)
        return (
            category.startswith("L")
            or category.startswith("N")
            or category.startswith("P")
            or char.isspace()
        )

    filtered_text = "".join(char for char in normalized_text if is_valid_char(char))
    return filtered_text


def _filter_nested(text: str, left: str, right: str) -> str:
    """
    Generic function to handle nested symbols.

    Args:
        text (str): The text to filter.
        left (str): The left symbol (e.g. '[' or '(').
        right (str): The right symbol (e.g. ']' or ')').

    Returns:
        str: The filtered text.
    """
    if not isinstance(text, str):
        raise TypeError("Input must be a string")
    if not text:
        return text

    result = []
    depth = 0
    for char in text:
        if char == left:
            depth += 1
        elif char == right:
            if depth > 0:
                depth -= 1
        else:
            if depth == 0:
                result.append(char)
    filtered_text = "".join(result)
    filtered_text = re.sub(r"\s+", " ", filtered_text).strip()
    return filtered_text


# ElevenLabs v3 takes inline direction in square brackets — "[laughs] Well, ..."
# — and performs it instead of reading it out. That collides head-on with this
# app, which uses square brackets for its OWN protocol: the LLM emits [joy] /
# [surprise_shock] to drive the avatar's face, and `ignore_brackets` exists to
# keep those out of the voice.
#
# So brackets cannot simply be kept or dropped wholesale. This is the allow-list
# that separates the two: NON-VERBAL REACTIONS and DELIVERY DIRECTION, which the
# voice can act, and which deliberately excludes every plain emotion word
# ([sad], [angry], [curious]). Those overlap with the emotionMap keys, and the
# app's own protocol has to win there — the face is driven from them, and v3
# infers emotion from the words anyway.
AUDIO_TAGS = frozenset(
    {
        # reactions that make a sound of their own
        "laughs",
        "laughing",
        "laughs harder",
        "chuckles",
        "giggles",
        "sighs",
        "gasps",
        "gulps",
        "clears throat",
        "coughs",
        "snorts",
        "sniffs",
        "exhales",
        "inhales",
        "breathes deeply",
        "yawns",
        "hums",
        # how the line is delivered
        "whispers",
        "whispering",
        "shouts",
        "shouting",
        "sings",
        "mutters",
        "cheerfully",
        "flatly",
        "deadpan",
        "playfully",
        "sarcastically",
        "warmly",
        "softly",
        "quickly",
        "slowly",
        # cognitive beats
        "pauses",
        "hesitates",
        "stammers",
        "stutters",
        "long pause",
        "short pause",
    }
)

_BRACKET_RE = re.compile(r"\[([^\[\]]*)\]")


def filter_brackets(text: str, keep_audio_tags: bool = False) -> str:
    """
    Filter text to remove all text within brackets, handling nested cases.

    Args:
        text (str): The text to filter.
        keep_audio_tags (bool): Keep bracketed spans that name a known ElevenLabs
            audio tag (see AUDIO_TAGS) and drop the rest. Only worth enabling for
            a voice on a model that acts them — anything else silently swallows
            them, which is harmless but pointless.

    Returns:
        str: The filtered text.
    """
    if not keep_audio_tags:
        return _filter_nested(text, "[", "]")
    if not isinstance(text, str):
        raise TypeError("Input must be a string")

    # Protect the tags worth keeping, run the normal nested strip over the rest,
    # then put them back. Going through the ordinary path matters: it is what
    # handles nesting and the whitespace left behind, and a second implementation
    # would drift from it.
    kept: list[str] = []

    def stash(m: re.Match) -> str:
        if m.group(1).strip().lower() in AUDIO_TAGS:
            kept.append(m.group(0))
            return f"\x00{len(kept) - 1}\x00"
        return m.group(0)

    out = _filter_nested(_BRACKET_RE.sub(stash, text), "[", "]")
    for i, tag in enumerate(kept):
        out = out.replace(f"\x00{i}\x00", tag)
    return re.sub(r"\s{2,}", " ", out).strip()


def filter_parentheses(text: str) -> str:
    """
    Filter text to remove all text within parentheses, handling nested cases.

    Args:
        text (str): The text to filter.

    Returns:
        str: The filtered text.
    """
    return _filter_nested(text, "(", ")")


def filter_angle_brackets(text: str) -> str:
    """
    Filter text to remove all text within angle brackets, handling nested cases.

    Args:
        text (str): The text to filter.

    Returns:
        str: The filtered text.
    """
    return _filter_nested(text, "<", ">")


def filter_asterisks(text: str) -> str:
    """
    Removes text enclosed within asterisks of any length (*, **, ***, etc.) from a string.

    Args:
        text: The input string.

    Returns:
        The string with asterisk-enclosed text removed.
    """
    # Handle asterisks of any length (*, **, ***, etc.)
    filtered_text = re.sub(r"\*{1,}((?!\*).)*?\*{1,}", "", text)

    # Clean up any extra spaces
    filtered_text = re.sub(r"\s+", " ", filtered_text).strip()

    return filtered_text
