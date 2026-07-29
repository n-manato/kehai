"""TEXT-driven STREAMING co-speech gesture via the gateway `/v1/motion/stream`.

ActionPlan (ECCV'26, resident on PC-B GPU1) generates SMPL body motion from an
ENGLISH action prompt and streams it latent-by-latent: first frames ~0.35s after
the request, then ~100fps of motion — faster than the speech it accompanies.

This replaces the audio-driven GestureLSM path (`gesture_client`) when
MOTION_TEXT_STREAM=1: because the input is the LLM's sentence TEXT, the gesture
fires the moment the sentence exists — no waiting for TTS audio — so the body
moves from the very first spoken frame instead of ~2s in.

The gateway stream is NDJSON:
    {"type":"meta","fps":30,"joint_names":[...22 SMPL...],"chained":bool}
    {"type":"frames","i0":N,"joints":[[[x,y,z]x22]x4]}      # NEW frames only
    {"type":"done","frames":N,"gen_ms":T}

Continuity: the service chains consecutive segments of one `session` (the
sampler's native conditioning), so a reply's sentences form ONE continuous
motion. `reset_chain()` breaks the chain (interrupt / new conversation).

Config (shared with the other motion routes):
    MOTION_TEXT_STREAM=1                    enable this path (default off)
    MOTION_GATEWAY_BASE=https://gpu.aiccore-uno.ai/v1
    MOTION_GATEWAY_KEY / .motion_gateway_key
"""

from __future__ import annotations

import json
import os
import re
from typing import AsyncGenerator

import httpx
from loguru import logger

from .gesture_client import _key  # same Bearer key resolution

_DEFAULT_BASE = "https://gpu.aiccore-uno.ai/v1"

# ActionPlan is CLIP-text conditioned on ENGLISH HumanML3D-style prompts, so the
# Japanese sentence is mapped to an in-distribution action description: content
# keywords first (greeting/thanks/pointing...), else the sentence's emotion tag.
_EMO_PROMPT = {
    "joy": "a person talks happily, nodding with a small hand gesture",
    "anger": "a person talks firmly with a sharp small hand gesture",
    "sadness": "a person talks quietly with small subdued gestures",
    "surprise": "a person tilts their head back slightly while talking",
    "fear": "a person talks nervously, with small hesitant gestures",
    "disgust": "a person talks dismissively, waving a hand away",
    "neutral": "a person talks casually, gesturing naturally with one hand",
}
_KEYWORD_PROMPTS: list[tuple[tuple[str, ...], str]] = [
    (
        ("こんにち", "おはよ", "こんばん", "はじめまして", "やっほ", "ハロー", "hello"),
        "a person waves their hand in greeting while talking",
    ),
    (
        ("バイバイ", "またね", "さよなら", "おやすみ", "goodbye", "bye"),
        "a person waves goodbye while talking",
    ),
    (
        ("ありがと", "感謝", "thank"),
        "a person bows slightly in thanks while talking",
    ),
    (
        ("ごめん", "すみません", "申し訳", "sorry"),
        "a person bows apologetically while talking",
    ),
    (
        ("すごい", "やった", "最高", "楽しみ", "wow", "amazing"),
        "a person cheers excitedly while talking, raising both hands",
    ),
    (
        ("うーん", "どうしよ", "考え", "かなあ", "hmm"),
        "a person thinks while talking, putting a hand near their chin",
    ),
    (
        ("見て", "ほら", "こちら", "あちら", "look"),
        "a person points forward while talking",
    ),
]


def gesture_prompt(key: str | None) -> str | None:
    """English ActionPlan prompt for a [g:key] content-gesture tag the persona
    emitted (extracted upstream into actions.gesture), or None if it isn't a known
    gesture key. Delegates to the CANONICAL GESTURE_INTENTS map (motion/prompts.py)
    so the tag path and the MMD pool path share one calibrated vocabulary. Using
    this SKIPS the separate intent-LLM round-trip entirely — content-matched AND
    faster, because the LLM already chose the gesture while writing the line."""
    if not key:
        return None
    from .prompts import GESTURE_INTENTS, prompt_for_key

    k = key.strip().lower()
    return prompt_for_key(k) if k in GESTURE_INTENTS else None


def enabled() -> bool:
    return os.getenv("MOTION_TEXT_STREAM", "0") == "1" and bool(_key())


def _base() -> str:
    return os.getenv("MOTION_GATEWAY_BASE", _DEFAULT_BASE).rstrip("/")


def intent_prompt(text: str, emotion: str | None = None) -> str:
    """Japanese sentence -> English gesture prompt. CONTENT-driven only: a keyword
    match (greeting/thanks/pointing...) else a content-neutral small gesture. The
    emotion is deliberately IGNORED here — the body gesture must follow WHAT IS SAID;
    the emotion drives the FACE, not the body. (Keying the body off the emotion made
    every gesture a generic "nod/smile", and `joy_smile` fell through to the neutral
    "talks casually" template — the main reason the motion didn't match the words.)"""
    t = (text or "").lower()
    for keys, prompt in _KEYWORD_PROMPTS:
        if any(k in t for k in keys):
            return prompt
    return "a person gestures naturally with one hand while talking"


# LLM-crafted intent: the keyword/emotion templates cover a handful of cases and
# everything else degrades to "talks casually" — content-blind motion. The local
# vLLM (GPU0 — no contention with GSV/ActionPlan on GPU1) turns the actual line
# into a matching HumanML3D-style prompt. The call runs IN PARALLEL with the
# deferred-launch wait for the first TTS chunk, so it usually adds ~0 delay.
_INTENT_SYS = (
    "You write ONE short English motion prompt for a text-to-motion model "
    "trained on HumanML3D, describing how a standing person gestures while "
    "speaking the given Japanese line. MATCH THE CONTENT of the line: if it "
    "points something out, counts, shows a size/amount, greets, refuses or "
    "denies, asks/wonders, thinks, presents or offers, or dismisses — use THAT "
    "specific gesture (points forward, counts on fingers, spreads hands to show "
    "size, waves, shakes head, tilts head, hand on chin, opens a palm to "
    "present, waves a hand away). For ordinary talk, a small VARIED gesture (a "
    "nod, a head tilt, one hand opening, a light shrug). Reserve big moves "
    "(raising both arms, cheering, bowing deeply) for an explicit cheer, "
    "celebration, greeting or apology. Standing in place only — no walking, "
    "sitting, or props. NO facial expressions, NO mood adverbs. Format: 'a "
    "person <gestures> while talking', max 12 words. Output ONLY the prompt. "
    "Use ONLY simple, common upper-body gestures: point forward, open/raise one "
    "or both palms, nod, tilt head, shake head, count on fingers, shrug, wave, "
    "spread hands, hand on chin, hand to chest. Do NOT use theatrical or unusual "
    "moves (covering/cupping the face, hugging, kneeling, dancing, clapping "
    "wildly, hands on hips) — keep it a calm standing conversation."
)


async def intent_prompt_llm(
    text: str,
    emotion: str | None,
    client: httpx.AsyncClient | None = None,
    timeout: float = 2.5,
) -> str:
    """Sentence -> motion prompt via the gateway LLM; falls back to the keyword/
    emotion heuristic on any failure. Disable with MOTION_INTENT_LLM=0."""
    fb = intent_prompt(text)  # content-only fallback (emotion is a face concern)
    key = _key()
    if not key or os.getenv("MOTION_INTENT_LLM", "1") != "1":
        return fb
    body = {
        "model": os.getenv("MOTION_INTENT_MODEL", "qwen3-30b-a3b"),
        "messages": [
            {"role": "system", "content": _INTENT_SYS},
            {
                # CONTENT ONLY — no emotion. The gesture must match WHAT is said, not
                # the mood; passing the emotion made outputs "smile/soft" and dampened.
                "role": "user",
                "content": f"line: {text}",
            },
        ],
        "max_tokens": 64,  # 40 truncated ~12-word prompts mid-phrase ("...toward the")
        "temperature": 0.4,
        "stream": False,
    }
    own = client is None
    cl = client or httpx.AsyncClient(timeout=timeout)
    try:
        r = await cl.post(
            f"{_base()}/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json=body,
            timeout=timeout,
        )
        if r.status_code == 200:
            choice = r.json()["choices"][0]
            # A truncated (finish_reason=length) prompt ends mid-phrase and drives
            # ActionPlan poorly — prefer the complete heuristic instead.
            if choice.get("finish_reason") == "length":
                return fb
            c = (choice["message"]["content"] or "").strip()
            c = c.strip('"').strip().splitlines()[0].strip()
            # sanity: short English prompt, else keep the heuristic
            if c and len(c) <= 160 and re.search(r"[A-Za-z]{3}", c):
                if "person" not in c.lower():
                    c = "a person " + c
                return c
    except Exception as e:  # noqa: BLE001 — heuristic fallback, never raise
        logger.debug(f"[ap-motion] intent LLM failed ({type(e).__name__}: {e})")
    finally:
        if own:
            await cl.aclose()
    return fb


# ---- content -> gesture KEY classification (for the pre-generated clip library) --- #
# Pick ONE of the fixed gesture keys for a line, so the matching pre-generated library
# clip can play INSTANTLY (no GPU gen, no lag, no voice contention). Classifying to a
# closed list is far more reliable for the local qwen3 than emitting inline [g:] tags.
_CLASSIFY_SYS = (
    "Pick the ONE body gesture that best matches a Japanese line a character speaks, "
    "from this fixed list, or 'none' if no clear gesture fits:\n"
    "greet(waves hello) nod(agree/yes) shake_head(deny/no) tilt(puzzled/question) "
    "point(points at something) point_self(refers to self) count(counts/lists things) "
    "think(pondering) shrug(dunno) dismiss(waves it off) explain(explains with both "
    "hands) present(offers/shows) beckon(come here) cross_arms hands_on_hips(confident) "
    "tease yawn(bored) shush(be quiet) facepalm(oh no) wave_both(light both hands)\n"
    "Most ordinary lines are 'none' or a small one (nod/tilt/explain/present). "
    "Output ONLY the key, one word, lowercase."
)


async def classify_key(
    text: str,
    client: "httpx.AsyncClient | None" = None,
    timeout: float = 2.5,
) -> str | None:
    """Sentence -> one gesture key (for the library) via the gateway LLM, or None.
    Disable with MOTION_INTENT_LLM=0."""
    key = _key()
    if not key or os.getenv("MOTION_INTENT_LLM", "1") != "1":
        return None
    from .prompts import INTENT_KEYS

    body = {
        "model": os.getenv("MOTION_INTENT_MODEL", "qwen3-30b-a3b"),
        "messages": [
            {"role": "system", "content": _CLASSIFY_SYS},
            {"role": "user", "content": f"line: {text}"},
        ],
        "max_tokens": 6,
        "temperature": 0.2,
        "stream": False,
    }
    own = client is None
    cl = client or httpx.AsyncClient(timeout=timeout)
    try:
        r = await cl.post(
            f"{_base()}/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json=body,
            timeout=timeout,
        )
        if r.status_code == 200:
            c = (r.json()["choices"][0]["message"]["content"] or "").strip().lower()
            for tok in re.findall(r"[a-z_]+", c):
                if tok in INTENT_KEYS:
                    return tok
    except Exception as e:  # noqa: BLE001 — no gesture on failure
        logger.debug(f"[ap-motion] classify_key failed ({type(e).__name__}: {e})")
    finally:
        if own:
            await cl.aclose()
    return None


# ---- gesture chaining ------------------------------------------------------ #
# The service chains segments per `session` on its own (30s TTL); this flag only
# lets an interrupt / conversation switch force the NEXT segment to start fresh.
_chain_broken = False


def reset_chain() -> None:
    global _chain_broken
    _chain_broken = True


async def stream_motion(
    prompt: str,
    seconds: float,
    session: str,
    client: httpx.AsyncClient | None = None,
    # read=20s: the gap between NDJSON lines is the service's GPU-queue wait; a
    # healthy queue clears in ~1-4s, so 20s means something is wedged — give up
    # (the frontend's procedural hold covers) instead of pinning a connection.
    timeout: httpx.Timeout | float = httpx.Timeout(20.0, connect=10.0),
) -> AsyncGenerator[dict, None]:
    """POST the prompt to the gateway and yield parsed NDJSON events as they
    arrive ({"type": "meta"|"frames"|"done", ...}). Yields nothing on any
    gateway/network failure — the caller degrades to the procedural layer
    (never blocks speech)."""
    global _chain_broken
    chain = not _chain_broken
    _chain_broken = False
    key = _key()
    if not key:
        return
    body = {
        "text": prompt,
        "seconds": round(max(1.0, seconds), 2),
        "session": session,
        "chain": chain,
    }
    headers = {"Authorization": f"Bearer {key}"}
    own_client = client is None
    cl = client or httpx.AsyncClient(timeout=timeout)
    # LAN-tunnel base with the public gateway as automatic fallback (only when
    # NOTHING was yielded yet — a stream that died mid-clip stays dead).
    fb = os.getenv("MOTION_GATEWAY_FALLBACK", _DEFAULT_BASE).rstrip("/")
    bases = [_base()]
    if fb and fb != bases[0]:
        bases.append(fb)
    try:
        for i, base in enumerate(bases):
            yielded = False
            try:
                async with cl.stream(
                    "POST",
                    f"{base}/motion/stream",
                    headers=headers,
                    json=body,
                    timeout=timeout,
                ) as r:
                    if r.status_code != 200:
                        detail = (await r.aread())[:200]
                        logger.warning(
                            f"[ap-motion] gateway {r.status_code} @ {base}: {detail!r}"
                        )
                        continue  # next base
                    async for line in r.aiter_lines():
                        if not line.strip():
                            continue
                        try:
                            yielded = True
                            yield json.loads(line)
                        except json.JSONDecodeError:
                            logger.warning(
                                f"[ap-motion] bad NDJSON line: {line[:120]!r}"
                            )
                return
            except Exception as e:  # noqa: BLE001 — try fallback, else degrade
                if yielded:
                    logger.warning(f"[ap-motion] died mid-stream: {e!r}")
                    return
                more = " — trying fallback" if i + 1 < len(bases) else ""
                logger.warning(
                    f"[ap-motion] {base} failed: {type(e).__name__}: {e}{more}"
                )
    finally:
        if own_client:
            await cl.aclose()
