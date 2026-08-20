"""Pre-generated SMPL gesture-clip LIBRARY (one clip per content-gesture key).

Co-speech gestures are GENERATED per sentence by ActionPlan (~2s on the shared PC-B
GPU1), which makes them sparse (voice-priority pacing), lagged, and coarse. Instead we
pre-generate ONE clean clip per intent key ONCE (offline, see scratchpad/generate_lib.py
-> motion-lib/*.json) and play the matching one INSTANTLY: no GPU gen -> no voice
contention, no lag, and a curated (cleaner, more distinct) motion. The sentence -> key
mapping is the [g:key] tag or actionplan_client.classify_key().
"""

from __future__ import annotations

import json
import os

from loguru import logger

# repo-root/motion-lib*  (this file is src/kehai/motion/motion_lib.py).
# motion-lib-kawaii holds the KAWAII ANIMATIONS 100 talk-loops + cute emotes
# (retargeted to the SMPL 22-joint clip format, see scratchpad/fbx_to_smpl_clip.py);
# it is scanned LAST so its curated clips win over any same-named ActionPlan clip.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
# KAWAII-only: the old realistic ActionPlan clips in motion-lib/ are retired (the
# talk-loops + cute emotes fully replace them). Re-add that dir here to bring them back.
_DIRS = [os.path.join(_ROOT, "motion-lib-kawaii")]

_clips: dict[str, dict] = {}

# Emotion -> talking-loop style (KAWAII Speak01-10). Every reply sentence plays its
# emotion's talk loop for the WHOLE sentence (seamless loop, seam≈0) so the body is
# never a statue mid-speech — the core "話してるとき違和感がない" motion. Base emotion
# prefixes (joy_*, anger_*, ...) fall through to their family below.
TALK_BY_EMOTION = {
    "neutral": "talk_normal",
    "joy": "talk_chatty",
    "joy_smile": "talk_chatty",
    "joy_big": "talk_excited",
    "joy_grin": "talk_excited",
    "happy": "talk_chatty",
    "excited": "talk_excited",
    "shy": "talk_shy",
    "anger": "talk_scold",
    "anger_rage": "talk_scold",
    "anger_pout": "talk_pout",
    "angry": "talk_scold",
    "sadness": "talk_calm",
    "sad": "talk_calm",
    "fear": "talk_calm",
    "surprise": "talk_excited",
    "surprised": "talk_excited",
    "thinking": "talk_explain",
    "look": "talk_serious",
    "calm": "talk_calm",
}


def talk_key_for_emotion(emotion: str | None) -> str:
    e = (emotion or "neutral").strip().lower()
    if e in TALK_BY_EMOTION:
        return TALK_BY_EMOTION[e]
    return TALK_BY_EMOTION.get(e.split("_")[0], "talk_normal")


def is_talk(key: str | None) -> bool:
    return bool(key) and key.strip().lower().startswith("talk_")


def is_emote(key: str | None) -> bool:
    """A KAWAII one-shot pose clip (not a talk loop, not silence)."""
    k = (key or "").strip().lower()
    return bool(k) and k in _clips and not k.startswith("talk_")


# Curated set of KAWAII emote keys the LLM may tag inline as `[g:<key>]` (an ACCENT
# pose played once over the talking body — see the persona's ジェスチャータグ section).
# A subset of the emote clips: the ones with clear, common conversational meaning.
EMOTE_TAGS = [
    "wave",
    "thumbsup",
    "yay",
    "cheer",
    "laugh",
    "surprised",
    "shy",
    "idea",
    "bow",
    "present",
    "pointing",
    "hand_on_hip",
    "finger_snap",
    "idol_pose",
    "cute_armup",
]

# Deterministic JP keyword -> emote. The local LLM emits [g:emote] tags unreliably, so
# this fires the cute accent from unambiguous phrasing instead (no LLM, no latency).
# HIGH-PRECISION only (strong, rare cues) so emotes stay OCCASIONAL accents over the
# always-on talk loop — checked first-match, order = priority.
_EMOTE_CUES: list[tuple[str, tuple[str, ...]]] = [
    (
        "wave",
        (
            "こんにち",
            "こんばん",
            "おはよ",
            "はじめまして",
            "やっほ",
            "ばいばい",
            "バイバイ",
            "またね",
            "じゃあね",
            "さような",
            "また会",
            "hello",
            "hi there",
            "good morning",
            "good evening",
            "goodbye",
            "see you later",
            "nice to meet you",
        ),
    ),
    (
        "bow",
        (
            "ありがと",
            "感謝",
            "ごめん",
            "すみません",
            "すいません",
            "申し訳",
            "よろしくお願",
            "thank you",
            "thanks",
            "my apologies",
            "i apologize",
            "sorry",
        ),
    ),
    (
        "surprised",
        (
            "えっ",
            "ええっ",
            "まさか",
            "びっくり",
            "驚き",
            "驚い",
            "本当ですか",
            "ほんとですか",
            "うそっ",
            "ウソ",
            "no way",
            "really?",
            "that is surprising",
            "unexpected",
        ),
    ),
    (
        "thumbsup",
        (
            "いいですね",
            "いいと思",
            "賛成",
            "その通り",
            "もちろん",
            "だいじょうぶ",
            "大丈夫",
            "ばっちり",
            "バッチリ",
            "任せて",
            "good idea",
            "sounds good",
            "i agree",
            "of course",
            "absolutely",
            "no problem",
            "leave it to me",
        ),
    ),
    (
        "yay",
        (
            "やった",
            "わーい",
            "わあい",
            "嬉し",
            "うれし",
            "楽しみ",
            "最高",
            "hooray",
            "wonderful",
            "excellent",
            "great news",
            "i'm glad",
        ),
    ),
    (
        "laugh",
        (
            "あはは",
            "うふふ",
            "えへへ",
            "面白",
            "おもしろ",
            "haha",
            "that was a joke",
            "amusing",
            "very funny",
        ),
    ),
    ("shy", ("恥ずかし", "照れ", "embarrassing", "blushing")),
    (
        "idea",
        (
            "そうだ！",
            "ひらめ",
            "いいことを思",
            "なるほど",
            "思いつ",
            "i have an idea",
            "i know what to do",
            "that explains it",
            "i see now",
        ),
    ),
    (
        "cheer",
        (
            "がんばって",
            "ファイト",
            "応援して",
            "good luck",
            "you can do it",
            "hang in there",
        ),
    ),
    (
        "present",
        (
            "どうぞ",
            "こちらです",
            "ご覧くださ",
            "見てくださ",
            "here you go",
            "here it is",
            "take a look",
            "as you can see",
        ),
    ),
]


def emote_for_text(text: str | None) -> str | None:
    """First high-precision emote cue found in the sentence (or None).
    Matched on the lowercased sentence so the English cues fire regardless of
    capitalization (JA cues are unaffected by lower())."""
    if not text:
        return None
    lowered = text.lower()
    for key, cues in _EMOTE_CUES:
        if key in _clips and any(c in lowered for c in cues):
            return key
    return None


# OPENER BRIDGE: a pre-generated GestureLSM clip (motion-lib-bridge/opener.json) that
# covers the turn's first ~1.5s. GestureLSM is audio-driven, so the real clip cannot
# exist until the voice does and then costs ~1.0-1.7s of inference — measured, and
# essentially irreducible. Playing this GestureLSM-STYLED clip from t=0 masks that gap
# far better than the KAWAII loop, and the real clip crossfades over it when it lands.
# Chosen as the medoid of 100 samples that passed a quality gate (see
# scratchpad/bridge_select.py). Kept OUT of _clips so it can't be tagged as an emote.
_BRIDGE_DIR = os.path.join(_ROOT, "motion-lib-bridge")
_bridge: dict | None = None


def bridge_clip() -> dict | None:
    """A ready-to-serve payload for the opener bridge (None if not installed)."""
    if _bridge is None:
        return None
    return {
        "joint_names": _bridge["joint_names"],
        "joints": _bridge["joints"],
        "fps": _bridge.get("fps", 30),
        "partial": False,
        "chained": False,
        "seed_overlap": 0,
        "gesturelsm": True,  # generated by GestureLSM (see gesture_client)
    }


# IDLE clips: what she does STANDING THERE between turns — the procedural breath-and-sway
# alone reads as a mannequin. `motion-lib-catalog/` holds EVERY standing clip from the
# pack (auditionable in the 🎬 UI); `active.json` in that folder is the subset actually
# used in the rotation. Played one at a time, weighted-random, never twice in a row.
# All are KAWAII (22-joint), so they never block a talk clip.
_IDLE_DIR = os.path.join(_ROOT, "motion-lib-catalog")
_ACTIVE_FILE = os.path.join(_IDLE_DIR, "active.json")
_idle: dict[str, dict] = {}
_idle_stats: dict[str, dict] = {}  # per-clip audit numbers, shown in the picker UI
# Default rotation: the clips that passed the planted-feet audit (knee <=35 deg, foot
# lift <=0.12m, no hip drop). The avatar's ankles are IK-pinned, so an animation that
# walks/lifts a foot/crouches gets its legs fought by the IK — bent knees and thighs
# through the skirt. The user can override the set from the UI (writes active.json).
_DEFAULT_ACTIVE = [
    "idle01_breathing",
    "idle09_waiting",
    "idle02_lookleftandright",
    "idle50_standingtalk1_1",
    "idle51_standingtalk1_2",
    "idle03_lookathands",
    "idle08_comeupwithanidea",
    "idle43_handonhip",
    "idle39_cutearmup",
    "idle76_motivatedpose",
]
_IDLE_WEIGHT = {  # unlisted -> 1
    "idle01_breathing": 5,
    "idle09_waiting": 4,
    "idle02_lookleftandright": 3,
    "idle03_lookathands": 2,
    "idle50_standingtalk1_1": 3,
    "idle51_standingtalk1_2": 3,
    "idle08_comeupwithanidea": 2,
    "idle43_handonhip": 2,
    # chan_* = Mixamo, converted by scripts/fbx_to_smpl_clip.mjs, for the chancellor avatar.
    # Weighted so that the great majority of the time she is simply STANDING there:
    # the reactions (a thank-you, a bow) are seasoning, and at equal weight a
    # chancellor who reacts to nobody every few seconds reads as a nervous tic.
    #
    # EVERY clip here was picked by MEASURED FOOT ORIENTATION, not by name and not by
    # stance width. "足が開きすぎ" turned out to be TOE-OUT, not how far apart the feet
    # are: the retarget re-solves stance width against the avatar's own legs (a source
    # clip at +3.1 deg thigh splay and one at -1.1 drew 0.03 apart) but takes the ankle
    # orientation verbatim, so a clip with the left foot at +28 deg drew at +25. Around
    # 10-15 deg of toe-out is a normal human stance; what looks wrong is one foot far
    # further out than the other. These are the Mixamo standing clips whose feet are
    # both within 15 deg and symmetric to within 8 — 5 out of 49 measured.
    # Length matters as much as content here: the idle->idle crossfade is 1.0s, so a 2s
    # clip leaves the body permanently mid-blend, which reads as restless bouncing rather
    # than as waiting. Only clips comfortably longer than the fade are in the rotation.
    "chan_idle01_weightshift": 5,  # 16.6s
    "chan_idle02_female": 4,  # 10.0s
    "chan_idle03_thinking": 2,  # 4.2s
}
# The rotation is PER CHARACTER. It used to be one global list, which meant Miku and
# Chancelor shared it: a set of anime-styled KAWAII idles reads as a different person
# on a photoreal chancellor, and a set of restrained standing idles reads as a dead
# Miku — there is no single list that is right for both. `_active_by_conf` is keyed by
# character_config.conf_uid; `_active_default` is what a character with no set of its
# own gets (and what the pre-split active.json migrates into).
_active_default: list[str] = []
_active_by_conf: dict[str, list[str]] = {}
# Never-twice-in-a-row is also per character, or switching models would suppress the
# first clip of the new one for no reason.
_idle_last: dict[str, str] = {}


def _conf(conf: str | None) -> str:
    return (conf or "").strip() or "_default"


def idle_keys() -> list[str]:
    return sorted(_idle.keys())


def active_keys(conf: str | None = None) -> list[str]:
    """The rotation this character actually plays: its own set, else the default."""
    own = _active_by_conf.get(_conf(conf))
    return list(own if own is not None else _active_default)


def _save_active() -> None:
    try:
        with open(_ACTIVE_FILE, "w") as f:
            json.dump(
                {"default": _active_default, "by_conf": _active_by_conf}, f, indent=1
            )
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[motion_lib] could not save active idle set: {e}")


def set_active(keys: list[str], conf: str | None = None) -> list[str]:
    """Replace one character's idle rotation (UI picker) and persist it next to the
    clips. No conf -> the default set, which is what characters without one use."""
    picked = [k for k in dict.fromkeys(k.strip().lower() for k in keys) if k in _idle]
    c = _conf(conf)
    if c == "_default":
        global _active_default
        _active_default = picked
    else:
        _active_by_conf[c] = picked
    _save_active()
    logger.info(f"[motion_lib] idle rotation for {c} set to {len(picked)} clips")
    return list(picked)


def _measure(d: dict) -> dict:
    """Cheap audit of a clip's own SMPL joints: how much it moves the feet and bends the
    knees. Surfaced in the picker so the risky ones are obvious BEFORE playing them."""
    try:
        import math

        jn = d["joint_names"]
        J = d["joints"]
        ix = {n: i for i, n in enumerate(jn)}
        need = ("Pelvis", "L_Ankle", "R_Ankle", "L_Hip", "R_Hip", "L_Knee", "R_Knee")
        if any(k not in ix for k in need):
            return {}
        pv = ix["Pelvis"]
        lift = travel = 0.0
        for a in ("L_Ankle", "R_Ankle"):
            j = ix[a]
            ys = [f[j][1] - f[pv][1] for f in J]
            xs = [f[j][0] - f[pv][0] for f in J]
            zs = [f[j][2] - f[pv][2] for f in J]
            lift = max(lift, max(ys) - min(ys))
            travel = max(travel, math.hypot(max(xs) - min(xs), max(zs) - min(zs)))
        knee = 0.0
        for hip, kn, an in (
            ("L_Hip", "L_Knee", "L_Ankle"),
            ("R_Hip", "R_Knee", "R_Ankle"),
        ):
            h, k, a = ix[hip], ix[kn], ix[an]
            for f in J:
                u = [f[h][i] - f[k][i] for i in range(3)]
                w = [f[a][i] - f[k][i] for i in range(3)]
                du = math.dist(f[h], f[k]) or 1.0
                dw = math.dist(f[a], f[k]) or 1.0
                dot = sum(u[i] * w[i] for i in range(3)) / (du * dw)
                knee = max(
                    knee, 180 - math.degrees(math.acos(max(-1.0, min(1.0, dot))))
                )
        hip_drop = max(f[pv][1] for f in J) - min(f[pv][1] for f in J)
        # NB: these are the SOURCE animation's numbers, not the retargeted result — the
        # avatar's knee is decided by the planted-ankle IK fighting the clip, so they
        # only WARN, they don't certify. (Measured counter-examples both ways.) Judging
        # is what the picker's preview is for; `verified` marks the ones actually checked
        # on the avatar with scratchpad/audit_idle.js.
        return {
            "footLift": round(lift, 3),
            "footTravel": round(travel, 3),
            "kneeMax": round(knee),
            "hipDrop": round(hip_drop, 3),
            "warn": bool(lift > 0.15 or knee > 70),  # walks/lifts a foot/crouches
        }
    except Exception:  # noqa: BLE001
        return {}


def catalog(conf: str | None = None) -> dict:
    """Everything auditionable, with duration + audit numbers + the active rotation.
    `inherited` tells the picker it is showing the shared default rather than a set
    this character owns, so ticking a box creates one instead of editing everyone's."""
    return {
        "conf": _conf(conf),
        "inherited": _conf(conf) not in _active_by_conf,
        "active": active_keys(conf),
        "clips": [
            {
                "key": k,
                "frames": len(_idle[k]["joints"]),
                "secs": round(
                    len(_idle[k]["joints"]) / float(_idle[k].get("fps", 30)), 1
                ),
                # checked ON THE AVATAR (not just the source numbers) and looked right
                "verified": k in _DEFAULT_ACTIVE,
                **_idle_stats.get(k, {}),
            }
            for k in sorted(_idle)
        ],
    }


def _idle_payload(key: str) -> dict:
    c = _idle[key]
    return {
        "joint_names": c["joint_names"],
        "joints": c["joints"],
        "fps": c.get("fps", 30),
        "partial": False,
        "chained": False,
        "seed_overlap": 0,
        "idle": True,  # frontend: play once, then ask for the next one
    }


def get_idle(key: str) -> tuple[str, dict] | None:
    """A SPECIFIC idle clip by key (debug/audit; the app uses random_idle)."""
    k = (key or "").strip().lower()
    return (k, _idle_payload(k)) if k in _idle else None


def random_idle(conf: str | None = None) -> tuple[str, dict] | None:
    """A weighted-random clip from this character's ACTIVE rotation, never twice in a
    row. An OWN set that is empty means "this character has no idle clips" and returns
    None — only the shared default falls back to the whole catalog, which is the
    pre-split behaviour for characters that never had a set."""
    import random

    c = _conf(conf)
    own = _active_by_conf.get(c)
    if own is not None:
        pool = [k for k in own if k in _idle]
    else:
        pool = [k for k in _active_default if k in _idle] or list(_idle)
    keys = [k for k in pool if k != _idle_last.get(c)] or pool
    if not keys:
        return None
    key = random.choices(keys, weights=[_IDLE_WEIGHT.get(k, 1) for k in keys], k=1)[0]
    _idle_last[c] = key
    return key, _idle_payload(key)


def _load_idle() -> None:
    if _idle or not os.path.isdir(_IDLE_DIR):
        return
    for fn in os.listdir(_IDLE_DIR):
        if not fn.endswith(".json") or fn == "active.json":
            continue
        try:
            with open(os.path.join(_IDLE_DIR, fn), "r") as f:
                d = json.load(f)
            if d.get("joint_names") and d.get("joints"):
                key = fn[:-5].lower()
                _idle[key] = d
                _idle_stats[key] = _measure(d)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[motion_lib] failed to load idle {fn}: {e}")
    global _active_default, _active_by_conf
    saved: list[str] = []
    by_conf: dict[str, list[str]] = {}
    if os.path.isfile(_ACTIVE_FILE):
        try:
            with open(_ACTIVE_FILE, "r") as f:
                raw = json.load(f)
            # {"active": [...]} is the pre-split file: one list shared by every
            # character. It becomes the DEFAULT set, so nothing changes for a config
            # that never had a per-character rotation.
            saved = list(raw.get("default") or raw.get("active") or [])
            by_conf = {
                str(k): [str(x).lower() for x in (v or [])]
                for k, v in (raw.get("by_conf") or {}).items()
            }
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[motion_lib] could not read active idle set: {e}")
    _active_default = [k for k in (saved or _DEFAULT_ACTIVE) if k in _idle]
    # A key that no longer exists on disk is dropped, but an EMPTY result is kept as
    # an empty list rather than deleted: "this character plays no idle clips" is a
    # real choice and must survive a restart.
    _active_by_conf = {c: [k for k in ks if k in _idle] for c, ks in by_conf.items()}
    if _idle:
        logger.info(
            f"[motion_lib] idle catalog {len(_idle)} clips, default rotation "
            f"{len(_active_default)} ({'saved' if saved else 'built-in'}): "
            f"{', '.join(_active_default)}"
        )
        for c, ks in _active_by_conf.items():
            logger.info(f"[motion_lib] idle rotation for {c}: {len(ks)} clips")


def _load_bridge() -> None:
    global _bridge
    p = os.path.join(_BRIDGE_DIR, "opener.json")
    if not os.path.isfile(p):
        return
    try:
        with open(p, "r") as f:
            d = json.load(f)
        if d.get("joint_names") and d.get("joints"):
            # This canned clip was sampled straight from GestureLSM, so it carries the
            # model's turn-away: measured -41.8 deg at frame 0, recovering over ~1.3s —
            # i.e. she started EVERY reply twisted sideways. Live clips get corrected in
            # gesture_client; this one is loaded from disk, so correct it here. keep=0:
            # a canned opener must read as facing the viewer, always.
            from . import gesture_client  # lazy: avoid an import cycle at module load

            d = gesture_client._face_front(d, keep=0.0)
            _bridge = d
            logger.info(
                f"[motion_lib] opener bridge loaded: {len(d['joints'])} frames "
                f"({len(d['joints']) / float(d.get('fps', 30)):.1f}s, "
                f"{len(d['joint_names'])} joints)"
            )
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[motion_lib] failed to load opener bridge: {e}")


def _load() -> None:
    if _clips:
        return
    for d in _DIRS:
        if not os.path.isdir(d):
            continue
        n = 0
        for fn in os.listdir(d):
            if not fn.endswith(".json"):
                continue
            key = fn[:-5].lower()
            try:
                with open(os.path.join(d, fn), "r") as f:
                    data = json.load(f)
                if data.get("joint_names") and data.get("joints"):
                    _clips[key] = data
                    n += 1
            except Exception as e:  # noqa: BLE001
                logger.warning(f"[motion_lib] failed to load {fn}: {e}")
        logger.info(f"[motion_lib] loaded {n} clips from {d}")
    logger.info(f"[motion_lib] total {len(_clips)} clips")


_load()
_load_bridge()
_load_idle()


def has(key: str | None) -> bool:
    return bool(key) and key.strip().lower() in _clips


def keys() -> list[str]:
    return sorted(_clips.keys())


def get(key: str) -> dict | None:
    """A ready-to-serve clip payload for `key` (same shape as /motion/clip-smpl):
    joints + names + fps, complete (partial=False). Returns a shallow copy so the
    caller can add per-request fields without mutating the cached clip."""
    c = _clips.get((key or "").strip().lower())
    if not c:
        return None
    return {
        "joint_names": c["joint_names"],
        "joints": c["joints"],
        "fps": c.get("fps", 30),
        "partial": False,
        "chained": False,
        "seed_overlap": 0,
    }
