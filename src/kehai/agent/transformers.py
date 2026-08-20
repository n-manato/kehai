import re
from typing import AsyncIterator, Tuple, Callable, List, Union, Dict, Any
from functools import wraps
from .output_types import Actions, SentenceOutput, DisplayText
from ..utils.tts_preprocessor import tts_filter as filter_text
from ..motion.motion_lib import EMOTE_TAGS, is_emote as _is_emote
from ..motion.prompts import INTENT_KEYS
from ..live2d_model import Live2dModel
from ..config_manager import TTSPreprocessorConfig
from ..utils.sentence_divider import SentenceDivider
from ..utils.sentence_divider import SentenceWithTags, TagState
from loguru import logger

# Content gesture tag the LLM emits per sentence, e.g. "[g:point]". Has a colon,
# so it is NOT caught by the generic [tag] strip in actions_extractor — extract +
# remove it there.
_GESTURE_TAG = re.compile(r"\[g:([a-z_]+)\]")
# The local LLM (qwen3) reliably EMITS gesture tags but usually DROPS the persona's
# "g:" prefix, writing a bare [key] (e.g. [wave], [thumbsup]). Accept [key] too, but
# ONLY for the curated KAWAII emote tags — they don't overlap the emotion keys, so an
# emotion tag like [joy] is never mistaken for a gesture. Longest-first for safe alt.
_BARE_GESTURE_TAG = re.compile(
    r"\[(" + "|".join(sorted(EMOTE_TAGS, key=len, reverse=True)) + r")\]"
)

# Characters that count as a sentence ending. Used to decide whether a "。"
# needs to be restored after a trailing emoji (which LLMs sometimes use in
# place of a full stop) is stripped out.
END_PUNCTUATION_CHARS = {
    ".",
    "!",
    "?",
    "。",
    "！",
    "？",
    "…",
    "、",
    "，",
    ",",
    "～",
    "~",
    "‼",
    "⁉",
    "❗",
    "❓",
    "♪",
    "」",
    "』",
    ")",
    "）",
}

# Emotion keyword prefixes considered "negative". A sentence with no emotion of
# its own inherits the previous sentence's emotion so the body motion stays
# stable — but a *negative* emotion (a frown, a glare) must NOT linger onto the
# following, often neutral or warm, sentences. LLMs rarely tag their closing
# line, so without this the last sentence of a reply keeps whatever sad/angry
# face came before it. Negative emotions therefore relax back to neutral instead
# of being carried over; positive/neutral ones still carry for motion continuity.
NEGATIVE_EMOTION_PREFIXES = ("anger", "sadness", "disgust", "fear")


def _is_negative_emotion(keyword: str) -> bool:
    return keyword.lower().startswith(NEGATIVE_EMOTION_PREFIXES)


# Minimum amount of spoken text (a proxy for elapsed time, ~chars) that must go
# by before another *filler* body gesture is triggered while the same emotion
# continues. Previously a filler fired on EVERY sentence; Japanese TTS sentences
# are short, so gestures restarted every ~1-2s and the body read as hurried even
# though each blend was smooth ("too fast"). Holding ~this many characters of
# speech lets the current gesture actually land before the next one starts. A
# genuine emotion *change* still triggers immediately — only same-emotion fillers
# are throttled. Tune higher for calmer movement, lower for busier.
MOTION_DWELL_CHARS = 28


def sentence_divider(
    faster_first_response: bool = True,
    segment_method: str = "pysbd",
    valid_tags: List[str] = None,
):
    """
    Decorator that transforms token stream into sentences with tags

    Args:
        faster_first_response: bool - Whether to enable faster first response
        segment_method: str - Method for sentence segmentation
        valid_tags: List[str] - List of valid tags to process
    """

    def decorator(
        func: Callable[
            ..., AsyncIterator[Union[str, Dict[str, Any]]]
        ],  # Expects str or dict
    ) -> Callable[
        ..., AsyncIterator[Union[SentenceWithTags, Dict[str, Any]]]
    ]:  # Yields SentenceWithTags or dict
        @wraps(func)
        async def wrapper(
            *args, **kwargs
        ) -> AsyncIterator[Union[SentenceWithTags, Dict[str, Any]]]:
            divider = SentenceDivider(
                faster_first_response=faster_first_response,
                segment_method=segment_method,
                valid_tags=valid_tags or [],
            )
            stream_from_func = func(*args, **kwargs)

            # Process the mixed stream using the updated SentenceDivider
            async for item in divider.process_stream(stream_from_func):
                if isinstance(item, SentenceWithTags):
                    logger.debug(f"sentence_divider yielding sentence: {item}")
                elif isinstance(item, dict):
                    logger.debug(f"sentence_divider yielding dict: {item}")
                yield item  # Yield either SentenceWithTags or dict
            # Flushing is handled within divider.process_stream

        return wrapper

    return decorator


def actions_extractor(live2d_model: Live2dModel):
    """
    Decorator that extracts actions from sentences, passing through dicts.
    """

    def decorator(
        func: Callable[
            ..., AsyncIterator[Union[SentenceWithTags, Dict[str, Any]]]
        ],  # Input type hint
    ) -> Callable[
        ..., AsyncIterator[Union[Tuple[SentenceWithTags, Actions], Dict[str, Any]]]
    ]:  # Output type hint
        @wraps(func)
        async def wrapper(
            *args, **kwargs
        ) -> AsyncIterator[
            Union[Tuple[SentenceWithTags, Actions], Dict[str, Any]]
        ]:  # Yield type hint
            stream = func(*args, **kwargs)
            # Track the emotion we last EMITTED. A new expression + motion is
            # only sent when the emotion actually changes, so a multi-sentence
            # reply of the same emotion plays one motion instead of restarting
            # it on every sentence. Live2D holds the expression until changed,
            # so the face stays correct even while we send no new actions.
            # Keywords (not numeric indices) are tracked because several
            # emotions can share the same emotionMap index, which would make
            # reverse lookup ambiguous.
            last_emotion_keywords: list = []
            # Track the motion groups last SENT so the body motion only restarts
            # when the emotion (and thus the gesture) actually changes, rather
            # than on every sentence. Keeps movement calm instead of frantic.
            last_motion_groups: list = []
            # While the same emotion runs across several sentences the body would
            # otherwise freeze (one motion looping) even though the mouth keeps
            # moving — which reads as "talking but stiff". To add lifelike
            # "talking" gestures we slip in short filler motions (a nod, a glance)
            # on the sentences between emotion changes. ``filler_idx`` cycles
            # through ``FILLER_MOTIONS`` so the gestures vary instead of repeating.
            filler_idx = 0
            # Same-emotion filler gestures. "Look" is intentionally NOT here —
            # it's reserved as the "suspicious / doubtful glance" gesture (driven
            # by the 🤨/🧐 cue), so using it as a generic filler would dilute that
            # meaning. Idle (the varied talk beats) and Idle_nod keep the body
            # alive between gestures instead.
            FILLER_MOTIONS = ["Idle_nod", "Idle", "Idle_nod"]
            # Accumulated spoken characters since the last body-motion trigger.
            # Used to throttle same-emotion filler gestures (see MOTION_DWELL_CHARS)
            # so the body doesn't restart a gesture on every short sentence.
            chars_since_motion = 0
            # Whether a "shy" emotion is currently in effect for this reply.
            # Shyness (blush) reads as the *state* of the whole utterance, but
            # LLMs sprinkle joy emojis (❤️😊) on nearly every sentence, which
            # would wipe the blush within seconds of it appearing. A blushing
            # smile is still a blush — so while shy is active, joy-family cues
            # KEEP the shy face instead of replacing it. Any other emotion
            # family (sadness, anger, surprise...) ends the shy state normally.
            shy_active = False
            async for item in stream:
                if isinstance(item, SentenceWithTags):
                    sentence = item
                    actions = Actions()
                    # Only extract emotions for non-tag text
                    if not any(
                        tag.state in [TagState.START, TagState.END]
                        for tag in sentence.tags
                    ):
                        # Content GESTURE intent, e.g. [g:point] -> a sentence-
                        # matching body clip. Extract + strip (its colon dodges the
                        # generic [tag] strip below).
                        gm = _GESTURE_TAG.search(sentence.text)  # [g:key]
                        if gm:
                            key = gm.group(1)
                            # TWO vocabularies arrive here. The kawaii EMOTE clips
                            # (wave, thumbsup, ...) play a canned motion; the 20
                            # GESTURE_INTENTS (tilt, shrug, shush, wave_both, ...)
                            # are prompts for the GENERATED body-motion path.
                            # prompt_for_key() understands either, so both are
                            # legitimate body keys — is_emote() alone is not the
                            # right test for "is this a gesture".
                            if _is_emote(key) or key in INTENT_KEYS:
                                actions.gesture = key
                            # Strip UNCONDITIONALLY, including keys neither table
                            # knows (the LLM invents them: [g:hand_on_hippo]).
                            # The removal used to sit INSIDE the is_emote branch,
                            # so every GESTURE_INTENTS tag survived into
                            # display_text — and display_text is what the subtitle
                            # shows, what chat history stores, and what
                            # set_memory_from_history replays to the LLM, which
                            # then copies the tag as a few-shot example. Measured
                            # 2026-08-13: ~3000 leaked tags across chat_history/.
                            # The generic [tag] strip further down cannot catch
                            # them — its character class has no ":".
                            sentence.text = _GESTURE_TAG.sub("", sentence.text)
                        else:
                            # bare [key] (qwen3's usual form) for a known gesture key
                            bm = _BARE_GESTURE_TAG.search(sentence.text)
                            if bm:
                                actions.gesture = bm.group(1)
                                sentence.text = _BARE_GESTURE_TAG.sub("", sentence.text)
                        emotion_keywords = live2d_model.extract_emotion_keywords(
                            sentence.text
                        )
                        # Emojis the LLM emits naturally also drive motion. They
                        # are collected in addition to explicit [tags] so the
                        # body gestures even when the model writes no tags.
                        emoji_emotions = live2d_model.extract_emoji_emotions(
                            sentence.text
                        )
                        emotion_keywords = emotion_keywords + emoji_emotions
                        # Shy stickiness (see shy_active above): joy-family cues
                        # while shy is active keep the blushing face instead of
                        # replacing it; any other family ends the shy state.
                        if any(k.startswith("shy") for k in emotion_keywords):
                            shy_active = True
                        elif emotion_keywords and not all(
                            k.startswith(("joy", "shy")) for k in emotion_keywords
                        ):
                            shy_active = False
                        if (
                            shy_active
                            and emotion_keywords
                            and all(
                                k.startswith(("joy", "shy")) for k in emotion_keywords
                            )
                        ):
                            emotion_keywords = ["shy"]
                        # Strip EVERY bracketed tag-like token from the spoken
                        # text — not only known emotions. This prevents tags the
                        # LLM invents (e.g. "[anger_thinking]") from leaking into
                        # the subtitle / TTS audio.
                        sentence.text = re.sub(
                            r"\[[a-zA-Z_][a-zA-Z0-9_]*\]", "", sentence.text
                        )
                        # Emojis are motion triggers only — remove them from the
                        # text so they appear in neither the subtitle nor the
                        # TTS audio (the spoken/displayed words are unchanged).
                        # LLMs often use a trailing emoji in place of a full stop
                        # ("うれしいです😊"), so once it's removed the sentence can
                        # lose its closing punctuation. Detect "ended with emoji"
                        # robustly by comparing the stripped text's tail before
                        # and after emoji removal: if the trailing run changed,
                        # an emoji was there, and we restore a "。" so subtitles
                        # don't end abruptly.
                        before = sentence.text.rstrip()
                        cleaned = live2d_model.remove_emojis(sentence.text).rstrip()
                        ended_with_emoji = bool(before) and (
                            not cleaned or before[-1] != cleaned[-1]
                        )
                        sentence.text = cleaned.lstrip()
                        if (
                            ended_with_emoji
                            and sentence.text
                            and sentence.text[-1] not in END_PUNCTUATION_CHARS
                        ):
                            # Restore a full stop matching the sentence's script
                            # (an English sentence must not get a CJK 「。」).
                            sentence.text += (
                                "。"
                                if re.search(r"[぀-ヿ一-鿿]", sentence.text)
                                else "."
                            )

                        # The sentence's effective emotion: its own tag/emoji, or
                        # the carried-over emotion when it has none — except a
                        # negative emotion is NOT carried (it would leave a frown
                        # on the following, often warm, sentences). When the
                        # previous emotion was negative and this sentence has none
                        # of its own, relax the face back to neutral instead.
                        if emotion_keywords:
                            effective = emotion_keywords
                            last_emotion_keywords = emotion_keywords
                        elif last_emotion_keywords and not any(
                            _is_negative_emotion(k) for k in last_emotion_keywords
                        ):
                            effective = last_emotion_keywords
                        else:
                            effective = ["neutral"]
                            last_emotion_keywords = ["neutral"]

                        if effective:
                            # The FACE expression still updates every sentence so
                            # the look tracks the words closely. Expression NAMES
                            # (e.g. "face_egao") are sent instead of numeric
                            # indices so the frontend applies named .exp3.json
                            # expressions directly.
                            actions.expressions = live2d_model.resolve_expressions(
                                effective
                            )
                            # Carry the RAW emotion keyword(s) through every
                            # sentence too. The MMD face maps these fine-grained
                            # keywords (joy_grin, anger_pout, sadness_tearful, ...)
                            # to distinct expression presets — richer than the
                            # Live2D emotionMap collapse, and unlike `motions` it's
                            # present on every sentence (motions is sparse).
                            actions.emotions = effective
                            # The BODY motion, however, only restarts when the
                            # motion group actually changes. A run of same-emotion
                            # sentences keeps one motion playing/looping instead of
                            # re-triggering a new gesture on every sentence, which
                            # is what made the movement feel frantic. Live2D holds
                            # the last motion until a new one is sent.
                            motion_groups = live2d_model.resolve_motion_groups(
                                effective
                            )
                            # Dwell is a proxy for SPOKEN TIME: latin text packs
                            # ~3x more chars into a second of speech than CJK,
                            # so count it at 1/3 weight or English replies would
                            # fire fillers ~3x too often.
                            _sl = len(sentence.text)
                            if not re.search(r"[぀-ヿ一-鿿]", sentence.text):
                                _sl //= 3
                            chars_since_motion += _sl
                            if motion_groups != last_motion_groups:
                                # Emotion changed: play its gesture immediately and
                                # reset the filler run / dwell so we don't nod over
                                # it right away.
                                actions.motions = motion_groups
                                last_motion_groups = motion_groups
                                chars_since_motion = 0
                            elif chars_since_motion >= MOTION_DWELL_CHARS:
                                # Same emotion continuing AND enough speech has
                                # elapsed: slip in a short filler gesture (nod /
                                # glance) so the body keeps moving without restarting
                                # a gesture on every short sentence. Cycle through
                                # FILLER_MOTIONS so the gestures vary.
                                filler = FILLER_MOTIONS[
                                    filler_idx % len(FILLER_MOTIONS)
                                ]
                                filler_idx += 1
                                actions.motions = live2d_model.resolve_motion_groups(
                                    [filler]
                                )
                                chars_since_motion = 0
                            # else: within the dwell window — send no new motion so
                            # the current gesture plays out (the frontend auto-picks
                            # a fresh Idle beat once it finishes), keeping the pace
                            # natural instead of hurried.
                    yield sentence, actions  # Yield the tuple
                elif isinstance(item, dict):
                    # Pass through dictionaries
                    yield item
                else:
                    logger.warning(
                        f"actions_extractor received unexpected type: {type(item)}"
                    )

        return wrapper

    return decorator


def display_processor():
    """
    Decorator that processes text for display, passing through dicts.
    """

    def decorator(
        func: Callable[
            ..., AsyncIterator[Union[Tuple[SentenceWithTags, Actions], Dict[str, Any]]]
        ],  # Input type hint
    ) -> Callable[
        ...,
        AsyncIterator[
            Union[Tuple[SentenceWithTags, DisplayText, Actions], Dict[str, Any]]
        ],
    ]:  # Output type hint
        @wraps(func)
        async def wrapper(
            *args, **kwargs
        ) -> AsyncIterator[
            Union[Tuple[SentenceWithTags, DisplayText, Actions], Dict[str, Any]]
        ]:  # Yield type hint
            stream = func(*args, **kwargs)

            async for item in stream:
                if (
                    isinstance(item, tuple)
                    and len(item) == 2
                    and isinstance(item[0], SentenceWithTags)
                ):
                    sentence, actions = item
                    text = sentence.text
                    # Handle think tag states
                    for tag in sentence.tags:
                        if tag.name == "think":
                            if tag.state == TagState.START:
                                text = "("
                            elif tag.state == TagState.END:
                                text = ")"

                    display = DisplayText(text=text)  # Simplified DisplayText creation
                    yield sentence, display, actions  # Yield the tuple
                elif isinstance(item, dict):
                    # Pass through dictionaries
                    yield item
                else:
                    logger.warning(
                        f"display_processor received unexpected type: {type(item)}"
                    )

        return wrapper

    return decorator


def tts_filter(
    tts_preprocessor_config: TTSPreprocessorConfig = None,
):
    """
    Decorator that filters text for TTS, passing through dicts.
    Skips TTS for think tag content.
    """

    def decorator(
        func: Callable[
            ...,
            AsyncIterator[
                Union[Tuple[SentenceWithTags, DisplayText, Actions], Dict[str, Any]]
            ],
        ],  # Input type hint
    ) -> Callable[
        ..., AsyncIterator[Union[SentenceOutput, Dict[str, Any]]]
    ]:  # Output type hint
        @wraps(func)
        async def wrapper(
            *args, **kwargs
        ) -> AsyncIterator[Union[SentenceOutput, Dict[str, Any]]]:  # Yield type hint
            stream = func(*args, **kwargs)
            config = tts_preprocessor_config or TTSPreprocessorConfig()

            async for item in stream:
                if (
                    isinstance(item, tuple)
                    and len(item) == 3
                    and isinstance(item[1], DisplayText)
                ):
                    sentence, display, actions = item
                    if any(tag.name == "think" for tag in sentence.tags):
                        tts = ""
                    else:
                        tts = filter_text(
                            text=display.text,
                            remove_special_char=config.remove_special_char,
                            ignore_brackets=config.ignore_brackets,
                            ignore_parentheses=config.ignore_parentheses,
                            ignore_asterisks=config.ignore_asterisks,
                            ignore_angle_brackets=config.ignore_angle_brackets,
                            language=getattr(config, "language", "ja"),
                            keep_audio_tags=getattr(config, "keep_audio_tags", False),
                        )

                    logger.debug(f"[{display.name}] display: {display.text}")
                    logger.debug(f"[{display.name}] tts: {tts}")

                    yield SentenceOutput(
                        display_text=display,
                        tts_text=tts,
                        actions=actions,
                    )
                elif isinstance(item, dict):
                    # Pass through dictionaries
                    yield item
                else:
                    logger.warning(f"tts_filter received unexpected type: {type(item)}")

        return wrapper

    return decorator
