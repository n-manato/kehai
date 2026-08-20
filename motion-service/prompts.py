"""Emotion -> English motion-prompt templates for HY-Motion.

Shared design constant. The PC-A backend derives the same emotion vocabulary
from the existing motion groups (Joy_* -> happy, Sadness_* -> sad, ...), so keep
the keys here in sync with src/kehai/motion/ on PC-A.

Rules (HY-Motion best practice): English, < 30 words, motion <= 5s. Prompts
describe a person who STAYS IN PLACE and talks (the retargeter discards global
translation), so avoid "walking/running/jumping across the scene".
"""
from __future__ import annotations

# Emotions the warm pool keeps ready. Mirrors emotionFromMotionGroup() on the
# frontend (live2d/talking-motion) and the backend emotion extraction.
POOL_EMOTIONS = ["neutral", "happy", "sad", "thinking", "surprised", "angry", "shy"]

# Multiple phrasings per emotion -> variety in the pool (vary seed + phrasing).
EMOTION_PROMPTS: dict[str, list[str]] = {
    "neutral": [
        "a person talking calmly, relaxed small hand gestures, gentle weight shifts",
        "a person speaking naturally, occasional light hand movements, standing at ease",
    ],
    "happy": [
        "a cheerful person talking and gesturing with both hands, upbeat and lively",
        "an excited person speaking happily, open arm gestures, slight bounce",
    ],
    "sad": [
        "a person speaking softly and dejected, slumped shoulders, small slow gestures",
        "a downcast person talking quietly, head slightly lowered, minimal movement",
    ],
    "thinking": [
        "a person thinking while speaking, one hand near the chin, slight head tilt",
        "a thoughtful person talking, looking up briefly, slow considered gestures",
    ],
    "surprised": [
        "a surprised person reacting, quickly raised hands, leaning back slightly",
        "a startled person talking, eyes wide gesture, sudden small jump",
    ],
    "angry": [
        "a person speaking firmly and annoyed, tense sharp hand gestures",
        "an irritated person talking, crossed-then-pointing arms, stiff posture",
    ],
    "shy": [
        "a shy person talking timidly, hands close to the body, small fidgety gestures",
        "a bashful person speaking softly, fingers fidgeting, slight turn away",
    ],
}


def prompt_for(emotion: str, variant: int = 0) -> str:
    """Pick a phrasing for an emotion (round-robin via `variant`)."""
    options = EMOTION_PROMPTS.get(emotion) or EMOTION_PROMPTS["neutral"]
    return options[variant % len(options)]
