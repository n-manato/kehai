"""Emotion -> English motion-prompt templates for HY-Motion (PA-A backend copy).

Mirrors motion-service/prompts.py (the PC-B copy) — keep the two in sync. The
gateway's /motion/generate accepts a `prompt` (verified in Phase 0: it is honored
and changes the output), so the backend sends an emotion-appropriate phrasing per
generation, round-robining phrasings for variety.

Rules (HY-Motion best practice): English, < 30 words, motion <= 5s, person STAYS
IN PLACE and talks (the retargeter discards global translation).
"""

from __future__ import annotations

POOL_EMOTIONS = ["neutral", "happy", "sad", "thinking", "surprised", "angry", "shy"]

# A VTuber TALKING TO THE CAMERA: front-facing, upper body facing the viewer,
# actively GESTURING WITH BOTH HANDS while speaking — a lively conversation, not a
# frozen pose. Earlier prompts leaned on "hand on hip / arms crossed / cool aloof"
# which made HY-Motion output STATIC standing poses ("動きが硬い/静止的"); the base
# action is now "talking with natural moving hand gestures". The character's cool
# "jito-me" flavour lives in the FACE (mmd-expression), NOT in a stiff body. Every
# prompt still says facing-the-camera / front-facing so she keeps addressing the
# viewer instead of turning away.
EMOTION_PROMPTS: dict[str, list[str]] = {
    "neutral": [
        "facing the camera, talking to the viewer, gesturing naturally with both hands while speaking, lively and relaxed, front-facing",
        "addressing the camera, speaking with easy flowing hand gestures, hands moving as she talks, upright, front-facing",
    ],
    "happy": [
        "facing the camera, talking to the viewer cheerfully, expressive lively hand gestures, light and animated, front-facing",
        "addressing the camera, speaking brightly with buoyant moving hand gestures, upbeat, front-facing",
    ],
    "sad": [
        "facing the camera, talking to the viewer softly, small slow hand gestures, gentle and subdued, front-facing",
        "addressing the camera, speaking quietly with light downward hand movements, front-facing",
    ],
    "thinking": [
        "facing the camera, talking to the viewer, gesturing thoughtfully with the hands, one hand occasionally near the chin, front-facing",
        "addressing the camera while considering, hands moving slowly as she reasons aloud, front-facing",
    ],
    "surprised": [
        "facing the camera, talking to the viewer, animated surprised hand gestures, hands lifting expressively, front-facing",
        "addressing the camera, speaking with quick lively startled gestures, front-facing",
    ],
    "angry": [
        "facing the camera, talking to the viewer emphatically, firm pointed hand gestures, front-facing",
        "addressing the camera, speaking sharply with emphatic cutting hand gestures, front-facing",
    ],
    "shy": [
        "facing the camera, talking to the viewer a little shyly, small tentative hand gestures near the body, front-facing",
        "addressing the camera, speaking softly with light fidgeting hand gestures, front-facing",
    ],
}


# CONTENT gesture intents (higher resolution than the 7 emotions): the LLM tags
# each sentence with the best-matching one (see the persona's ## Gestures block
# and transformers.actions_extractor). Kept in the SAME languid/cool register as
# the character, but each describes a distinct, recognizable gesture so the body
# actually matches the words. The worker pre-generates a buffer per used intent,
# so playback is instant. Keep keys in sync with the persona's gesture list.
# Each is the same "facing the camera, talking to the viewer" base + one clear
# gesture, so the body stays front-facing while doing the gesture.
GESTURE_INTENTS: dict[str, list[str]] = {
    "greet": ["facing the camera, talking to the viewer while giving a small casual wave hello, front-facing, cool"],
    "nod": ["facing the camera, talking to the viewer and nodding slowly in agreement, front-facing, calm"],
    "shake_head": ["facing the camera, talking to the viewer and slowly shaking the head no, front-facing, cool"],
    "tilt": ["facing the camera, talking to the viewer with the head tilted slightly, one loose hand, front-facing, curious"],
    "point": ["facing the camera, talking to the viewer while casually pointing forward with one hand, front-facing, cool"],
    "point_self": ["facing the camera, talking to the viewer with a hand on the own chest, referring to self, front-facing"],
    "count": ["facing the camera, talking to the viewer while counting slowly on the fingers, front-facing, deliberate"],
    "think": ["facing the camera, talking to the viewer with one hand near the chin, considering, front-facing"],
    "shrug": ["facing the camera, talking to the viewer with a small lazy shrug, palms loose, front-facing, dunno"],
    "dismiss": ["facing the camera, talking to the viewer while waving one hand dismissively, front-facing, cool"],
    "explain": ["facing the camera, talking to the viewer while gesturing loosely with both hands to explain, front-facing"],
    "present": ["facing the camera, talking to the viewer with one open upturned hand showing something, front-facing"],
    "beckon": ["facing the camera, talking to the viewer while beckoning come-here with one hand, front-facing, teasing"],
    "cross_arms": ["facing the camera, talking to the viewer with arms loosely crossed, cool aloof, front-facing"],
    "hands_on_hips": ["facing the camera, talking to the viewer with one hand on the hip, confident, front-facing, cool"],
    "tease": ["facing the camera, talking to the viewer with a sly smirk and one hand up, teasing, front-facing"],
    "yawn": ["facing the camera, talking to the viewer with a slow lazy stretch covering a yawn, front-facing, bored"],
    "shush": ["facing the camera, talking to the viewer while raising one finger to the lips, quiet, front-facing, sly"],
    "facepalm": ["facing the camera, talking to the viewer with one hand to the forehead, exasperated, front-facing, cool"],
    "wave_both": ["facing the camera, talking to the viewer with a small relaxed two-hand gesture, front-facing, low-key"],
}

# Everything the worker / prompt resolver understands: 7 emotions + gesture intents.
INTENT_KEYS = list(GESTURE_INTENTS.keys())


def prompt_for_key(key: str, variant: int = 0) -> str:
    """Pick an English motion prompt for a body KEY — a gesture intent (content)
    or an emotion (fallback). Round-robins phrasings via `variant`."""
    options = (
        GESTURE_INTENTS.get(key)
        or EMOTION_PROMPTS.get(key)
        or EMOTION_PROMPTS["neutral"]
    )
    return options[variant % len(options)]


def prompt_for(emotion: str, variant: int = 0) -> str:
    """Back-compat alias: pick a phrasing for an emotion/key."""
    return prompt_for_key(emotion, variant)


def is_known_key(key: str | None) -> bool:
    return bool(key) and (key in GESTURE_INTENTS or key in POOL_EMOTIONS)


def emotion_from_keyword(keyword: str | None) -> str:
    """Collapse a fine-grained emotion keyword (actions.emotions[0], e.g.
    "joy_grin", "anger_pout") to one of the 7 pool emotions. Backend mirror of
    the frontend mmdEmotionFromKeyword (mmd-motion-player.ts) — keep in sync."""
    if not keyword:
        return "neutral"
    k = keyword.lower()
    if k.startswith("joy") or k in ("smug", "smirk") or k.startswith("love"):
        return "happy"
    if k.startswith("sad"):
        return "sad"
    if k.startswith(("anger", "angry")) or k == "disgust":
        return "angry"
    if k.startswith("surprise") or k == "fear":
        return "surprised"
    if k.startswith("think"):
        return "thinking"
    if k.startswith("shy"):
        return "shy"
    return "neutral"
