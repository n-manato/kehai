"""Same-origin motion endpoint for the MMD avatar.

The browser hits these SAME-ORIGIN endpoints; this server fetches the RAW SMPL
motion from the remote HY-Motion gateway (PC-B, `?format=smpl`), retargets it to
the MMD skeleton HERE (kehai/motion/smpl_to_mmd.py, robust world-space
direction matching), and streams back the `.vmd`. Doing the retarget on this
side (instead of on PC-B) fixes the SMPL<->MMD skeleton-mismatch bugs (arms
swinging behind the body) and lets us iterate with our own render/FK verification.

If the gateway can't return SMPL (older build) it returns a `.vmd` directly,
which we pass through unchanged.

Config (resolved per request):
    key : env MOTION_GATEWAY_KEY, else file <repo-root>/.motion_gateway_key
    base: env MOTION_GATEWAY_BASE, else the public gateway
"""

import asyncio
import os
import time
from pathlib import Path

import httpx
from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import JSONResponse
from loguru import logger

from . import motion_cache
from .motion.prompts import is_known_key
from .motion.smpl_to_mmd import clip_quality, smpl_dict_to_vmd_bytes

DEFAULT_BASE = "https://gpu.aiccore-uno.ai/v1"
ROOT = Path(__file__).resolve().parents[2]
KEY_FILE = ROOT / ".motion_gateway_key"
PMX_PATH = (
    ROOT
    / "Live2D Characters"
    / "YYB Hatsune Miku_10th"
    / "YYB Hatsune Miku_10th_v1.02.pmx"
)
POOL_EMOTIONS = {"neutral", "happy", "sad", "thinking", "surprised", "angry", "shy"}
GEN_DURATION = 4.0

# WARM talk clips for gesture mode: pre-baked generic GestureLSM->Miku VMDs served
# instantly by /motion/pool so the body starts moving WITH speech at t=0, before
# the sentence's real co-speech gesture (/motion/clip) is generated + synced in.
# Round-robin across sentences for variety. Loaded once (list of bytes).
WARM_DIR = ROOT / "cache" / "warm_gestures"
_warm_clips: list[bytes] | None = None
_warm_i = 0


def _warm_clip() -> bytes | None:
    """Next pre-baked warm talk VMD (round-robin), or None if none are present."""
    global _warm_clips, _warm_i
    if _warm_clips is None:
        try:
            _warm_clips = [p.read_bytes() for p in sorted(WARM_DIR.glob("*.vmd"))]
        except OSError:
            _warm_clips = []
        logger.info(f"[motion] loaded {len(_warm_clips)} warm talk clips")
    if not _warm_clips:
        return None
    clip = _warm_clips[_warm_i % len(_warm_clips)]
    _warm_i += 1
    return clip


# Pool quality filter: HY-Motion sometimes generates self-intersecting clips
# (hands through each other / through the chest — measured ~22% of frames on a
# real clip). Clips over these thresholds are refetched up to QUALITY_TRIES;
# the best-scoring one seen is used regardless, so the conversation never blocks.
# (Small violations are also fixed at retarget time by the IK collision clamp.)
QUALITY_TRIES = 3
QUALITY_MAX_HANDS = 0.15  # max fraction of frames with hands interpenetrating
QUALITY_MAX_TORSO = 0.20  # max fraction of frames with a wrist inside the torso
QUALITY_DEADLINE_S = 8.0  # overall budget for quality REFETCHES: each retry is a
# full gateway round-trip. This path is now only the COLD-BUFFER fallback (the
# warm buffer serves instantly), so keep it tight — a slow, contended gateway
# must not stall the fallback for 20s+ (measured 24s under contention).


def _resolve_key() -> str:
    k = os.getenv("MOTION_GATEWAY_KEY", "").strip()
    if k:
        return k
    try:
        if KEY_FILE.exists():
            return KEY_FILE.read_text(encoding="utf-8").strip()
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[motion] could not read key file {KEY_FILE}: {e}")
    return ""


def _resolve_base() -> str:
    return os.getenv("MOTION_GATEWAY_BASE", DEFAULT_BASE).rstrip("/")


def init_motion_routes() -> APIRouter:
    router = APIRouter()
    if _resolve_key():
        logger.info(f"[motion] proxy enabled -> {_resolve_base()} (retarget on PA-A)")
    else:
        logger.warning(
            f"[motion] no key (env MOTION_GATEWAY_KEY or {KEY_FILE.name}); "
            "/motion/* returns 503 until configured"
        )
    if not PMX_PATH.exists():
        logger.warning(f"[motion] PMX not found at {PMX_PATH} — retarget will fail")

    def _headers(key: str) -> dict:
        return {"Authorization": f"Bearer {key}"}

    # One shared client (connection pooling): a fresh AsyncClient per request
    # paid TCP+TLS setup to the gateway on every motion fetch.
    client = httpx.AsyncClient(timeout=90.0)

    async def _fetch_smpl(emotion: str, key: str, timeout: float = 90.0):
        """Return an httpx.Response: warm pool first, then on-demand generate."""
        base = _resolve_base()
        r = await client.get(
            f"{base}/motion/pool/{emotion}?format=smpl",
            headers=_headers(key),
            timeout=timeout,
        )
        if r.status_code == 200:
            return r
        r = await client.post(
            f"{base}/motion/generate?format=smpl",
            headers=_headers(key),
            json={"emotion": emotion, "duration": GEN_DURATION},
            timeout=timeout,
        )
        return r

    @router.get("/motion/health")
    async def motion_health():
        return {
            "configured": bool(_resolve_key()),
            "base": _resolve_base(),
            "retarget": "PA-A direction-matching",
            "pmx": PMX_PATH.exists(),
            "emotions": sorted(POOL_EMOTIONS),
        }

    @router.get("/motion/debug")
    async def motion_debug():
        return motion_cache.stats()

    @router.get("/motion/clip/{motion_id}")
    async def motion_clip(motion_id: str):
        """A bespoke per-sentence clip (fired in parallel with TTS). Ready -> 200
        VMD; still generating / unknown -> 202 (the frontend's single "fall back
        to the pool buffer" signal). Never blocks — the short wait inside get_clip
        happens after the browser has already started the audio."""
        # Audio-driven gesture lands a couple seconds into the sentence (it needs
        # the finished audio + a gateway round-trip). The frontend POLLS this
        # endpoint (playGesture) until the clip is ready, so each request returns
        # fast — a short server wait lets an already-ready clip come back on the
        # first poll without holding the connection open. HY-Motion bespoke keeps
        # its own short wait.
        from .motion import gesture_client  # lazy

        wait_s = 1.0 if gesture_client.enabled() else motion_cache.CLIP_WAIT_S
        vmd = await motion_cache.get_clip(motion_id, wait_s=wait_s)
        if vmd:
            return Response(content=vmd, media_type="application/octet-stream")
        dead = motion_cache.clip_is_dead(motion_id)
        if dead:  # see the clip-smpl route: terminal, so stop polling
            return Response(status_code=410, headers={"X-Motion-Reason": dead})
        return Response(status_code=202)  # not ready -> frontend uses pool buffer

    @router.get("/motion/clip-smpl/{motion_id}")
    async def motion_clip_smpl(motion_id: str):
        """Same as /motion/clip but returns the RAW SMPL joints JSON for the VRM
        viewer (which does its own SMPL->VRM retarget). Ready -> 200 JSON; not
        ready / unknown -> 202. Polled by the frontend just like /motion/clip."""
        from .motion import gesture_client  # lazy

        wait_s = 1.0 if gesture_client.enabled() else motion_cache.CLIP_WAIT_S
        joints = await motion_cache.get_clip_joints(motion_id, wait_s=wait_s)
        if joints:
            return JSONResponse(content=joints)
        # 410, not 202, once the id is known to be finished or failed. 202 means
        # "ask again", and for an id that can never produce a clip that kept the
        # frontend polling until its own timeout — the avatar waiting on a clip
        # that was not coming instead of moving on to the procedural body.
        dead = motion_cache.clip_is_dead(motion_id)
        if dead:
            return Response(status_code=410, headers={"X-Motion-Reason": dead})
        return Response(status_code=202)

    @router.get("/motion/idle")
    async def motion_idle(
        key: str | None = None, conf: str | None = None, engaged: bool = False
    ):
        """A random IDLE clip for the frontend to play between turns (she should look
        like she's waiting, not standing frozen). The frontend asks again when the clip
        ends, so consecutive requests give different clips. 204 if none installed.
        `?key=` pins a specific clip (debugging / auditing one animation).
        `?conf=` is the character's conf_uid — the rotation is per character, because
        one list cannot suit both an anime Miku and a photoreal chancellor.
        `?engaged=1`: someone is in front of her — pick from the character's quiet
        set (motion_lib._engaged_by_conf); the reply says whether it did."""
        from .motion import motion_lib  # lazy

        got = motion_lib.get_idle(key) if key else motion_lib.random_idle(conf, engaged)
        if not got:
            return Response(status_code=204)
        key, clip = got
        attending = engaged and key in motion_lib.engaged_keys(conf)
        return JSONResponse(content={**clip, "key": key, "engaged": attending})

    @router.get("/motion/catalog")
    async def motion_catalog(conf: str | None = None):
        """Every auditionable idle clip + which ones are in THIS character's rotation.
        Drives the 🎬 picker UI so the idle set can be chosen by watching, not by
        guessing, and tells the frontend whether this character has idle clips at all."""
        from .motion import motion_lib  # lazy

        return JSONResponse(content=motion_lib.catalog(conf))

    @router.post("/motion/idle-set")
    async def motion_idle_set(body: dict):
        """Replace one character's idle rotation with the keys ticked in the picker.
        No `conf` in the body edits the shared default, which is what characters
        without a set of their own play."""
        from .motion import motion_lib  # lazy

        keys = body.get("active")
        if not isinstance(keys, list):
            return JSONResponse(
                status_code=400, content={"error": "active must be a list"}
            )
        conf = body.get("conf")
        return JSONResponse(
            content={"conf": conf, "active": motion_lib.set_active(keys, conf)}
        )

    @router.get("/motion/pool/{emotion}")
    async def motion_pool(emotion: str):
        # GESTURE-ONLY mode: HY-Motion pool/idle is retired. Return 204 so the
        # frontend's playEmotion / pumpIdle treat it as a miss and fall through to
        # the procedural idle (breathing/sway) — no HY-Motion clips ever play. The
        # body is driven solely by GestureLSM (/motion/clip). Reverts automatically
        # when gesture is disabled (GESTURE_ENABLED=0).
        from .motion import gesture_client  # lazy

        if gesture_client.enabled():
            # Serve a pre-baked WARM talk clip so the body moves WITH speech from
            # t=0; the frontend loops it and crossfades to this sentence's real
            # co-speech gesture (/motion/clip) when it lands. No warm clips present
            # -> 204 (frontend falls through to the procedural idle, prior behaviour).
            warm = _warm_clip()
            if warm is not None:
                return Response(content=warm, media_type="application/octet-stream")
            return Response(status_code=204)
        # `emotion` is a body KEY: one of the 7 pool emotions OR a gesture intent
        # (point/count/shrug/...). Unknown keys fall back to neutral rather than
        # 404 so a stale frontend never breaks the body.
        if not is_known_key(emotion):
            emotion = "neutral"
        # Hybrid fast path: serve a fresh, continuously-regenerated buffer clip
        # instantly (no gateway round-trip). Cold buffer -> fall through to a
        # direct gateway fetch below (and the worker warms this emotion for next
        # time via get_buffer_clip marking it active).
        buffered = motion_cache.get_buffer_clip(emotion)
        if buffered is not None:
            return Response(content=buffered, media_type="application/octet-stream")
        key = _resolve_key()
        if not key:
            raise HTTPException(503, "motion gateway key not configured")
        # Cold-buffer remote fallback: the gateway pool only knows the 7 emotions,
        # not gesture intents, so fetch a base emotion here (the warm buffer, which
        # DID know the gesture, is preferred above and warms for next time).
        gw_emotion = emotion if emotion in POOL_EMOTIONS else "neutral"
        try:
            r = await _fetch_smpl(gw_emotion, key)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[motion] gateway unreachable: {e}")
            raise HTTPException(502, "motion gateway unreachable")
        if r.status_code != 200:
            logger.warning(f"[motion] gateway {r.status_code} for {emotion}")
            raise HTTPException(503, "no motion")

        ctype = r.headers.get("content-type", "")
        if "json" in ctype:
            # Quality gate: score self-intersection; refetch bad clips, keep the
            # best seen so we always answer with SOMETHING.
            best, best_q, best_score = None, None, None
            t0 = time.monotonic()
            for attempt in range(QUALITY_TRIES):
                if attempt > 0:
                    remaining = QUALITY_DEADLINE_S - (time.monotonic() - t0)
                    if remaining <= 0:
                        logger.info(
                            f"[motion] {emotion} quality-retry deadline reached"
                        )
                        break
                    try:
                        r = await _fetch_smpl(
                            emotion, key, timeout=max(5.0, min(90.0, remaining))
                        )
                    except Exception:  # noqa: BLE001
                        break
                    if r.status_code != 200:
                        break
                    if "json" not in r.headers.get("content-type", ""):
                        # Old-gateway VMD fallback mid-loop: honor the module's
                        # pass-through contract. Only take it when no SMPL clip
                        # was usable — never turn a valid VMD into a 503.
                        if best is None:
                            return Response(
                                content=r.content,
                                media_type="application/octet-stream",
                            )
                        break
                try:
                    smpl = r.json()
                    q = await asyncio.to_thread(clip_quality, smpl)
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"[motion] bad SMPL payload for {emotion}: {e}")
                    continue
                if not q["frames"]:
                    logger.warning(f"[motion] empty SMPL clip for {emotion}")
                    continue
                score = q["hands_overlap"] + q["hand_in_torso"]
                if best is None or score < best_score:
                    best, best_q, best_score = smpl, q, score
                if (
                    q["hands_overlap"] <= QUALITY_MAX_HANDS
                    and q["hand_in_torso"] <= QUALITY_MAX_TORSO
                ):
                    break
                logger.info(
                    f"[motion] {emotion} clip self-intersects {q} "
                    f"(try {attempt + 1}/{QUALITY_TRIES}) -> refetch"
                )
            if best is None:
                raise HTTPException(503, "no usable motion")
            logger.info(f"[motion] {emotion} clip quality: {best_q}")
            try:
                vmd = await asyncio.to_thread(
                    smpl_dict_to_vmd_bytes, best, str(PMX_PATH)
                )
            except Exception as e:  # noqa: BLE001
                logger.warning(f"[motion] retarget failed for {emotion}: {e}")
                raise HTTPException(500, "retarget failed")
            return Response(content=vmd, media_type="application/octet-stream")
        # gateway returned a .vmd directly (no SMPL support) -> pass through
        return Response(content=r.content, media_type="application/octet-stream")

    return router
