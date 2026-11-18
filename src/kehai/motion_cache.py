"""Continuous HY-Motion generation + cache (the hybrid motion engine on PA-A).

Phase 0 measured the gateway at ~5s/clip and SINGLE-CONCURRENCY (parallel requests
get 429). So a single background WORKER serializes all gateway calls and:

  1. keeps a small fresh-clip BUFFER per active emotion (+neutral), continuously
     re-generating so the avatar always has a fresh, emotion-matched body clip
     ready for instant playback (this is the reliable path);
  2. serves best-effort BESPOKE per-sentence clips (sized to the sentence's
     estimated speech length) when tts_manager fires one — if it lands in time the
     frontend prefers it, otherwise it falls back to the buffer.

Nothing here ever blocks the conversation: generation is off-thread/off-loop work,
the payload is never awaited on it, and the frontend has an instant buffer/pool
fallback. See docs plan a-nested-volcano.md.
"""

from __future__ import annotations

import asyncio
import os
import time
import uuid
from collections import OrderedDict, defaultdict, deque

import httpx
from loguru import logger

from .motion.prompts import POOL_EMOTIONS, prompt_for_key
from .motion.smpl_to_mmd import smpl_dict_to_vmd_bytes

# --- tunables (Phase 4) ---------------------------------------------------- #
BUFFER_DEPTH = 2  # fresh clips kept ready per active emotion
DEFAULT_DUR = 4.0  # buffer clip length (s)
REFRESH_S = 45.0  # regenerate a buffer clip older than this (continuous freshness)
ACTIVE_TTL_S = 120.0  # an emotion stays "active" (kept warm) this long after use
CLIP_WAIT_S = 0.3  # bespoke get_clip short wait (audio is already playing)
BESPOKE_TTL_S = 60.0
STALE_BESPOKE_S = 8.0  # a bespoke this old: its sentence has surely played — the
# clip is redirected into the emotion buffer instead of fulfilled for nobody
MAX_BESPOKE = 64
IDLE_SLEEP = 0.5  # worker nap when nothing needs generating
GEN_TIMEOUT_S = 15.0  # per-request cap: a hung gateway call must NOT freeze the
# single worker for the client's full 120s (gen is ~5s; 15s = 3x margin)

_client = httpx.AsyncClient(timeout=120.0)
_buffers: dict[str, deque] = defaultdict(deque)  # emotion -> deque[(ts, vmd)]
_active: "OrderedDict[str, float]" = OrderedDict()  # emotion -> last-used ts
_variant: dict[str, int] = defaultdict(int)  # per-emotion prompt round-robin
_bespoke_q: "asyncio.Queue[str]" = asyncio.Queue()
_worker_task: asyncio.Task | None = None


class _Entry:
    __slots__ = (
        "event",
        "vmd",
        "joints",
        "error",
        "ts",
        "emotion",
        "duration",
        "streaming",
        "keep",
        "lazy_vmd",
    )

    def __init__(self, emotion: str, duration: float) -> None:
        self.lazy_vmd = False  # GestureLSM clip: the MMD .vmd is built on first request
        self.event = asyncio.Event()
        self.vmd: bytes | None = None
        self.joints: dict | None = None  # raw SMPL joints for the VRM viewer
        self.error = False
        self.ts = time.monotonic()
        self.emotion = emotion
        self.duration = duration
        # TEXT-STREAMING entry (ActionPlan): `joints` GROWS while generation runs
        # (joints["partial"] True until complete). get_clip_joints serves partial
        # snapshots WITHOUT consuming, so the frontend's poll loop extends the
        # playing clip live; the entry is consumed on the first post-done fetch.
        self.streaming = False
        # Non-consuming: a pre-generated LIBRARY clip is fetched TWICE (prefetch +
        # playGesture race concurrently), so get_clip_joints must serve a snapshot
        # WITHOUT popping the entry, or one fetch consumes it and the other (and the
        # actual playback) gets nothing. Evicted by _evict_bespoke, not consume-once.
        self.keep = False


_bespoke: "OrderedDict[str, _Entry]" = OrderedDict()

# GestureLSM is a single model instance on the gateway (one GPU): serialize gesture
# generations so a burst of sentences doesn't fire concurrent forwards at it (which
# contend / can error). Gen is ~2s each, so serialized they land ~every 2s during a
# reply — plenty for continuous co-speech. Tasks are kept referenced so the event
# loop doesn't GC them mid-flight.
_gesture_sem = asyncio.Semaphore(1)
_gesture_tasks: set[asyncio.Task] = set()
_CALLER_LOGGED = False


# --- config ---------------------------------------------------------------- #
_enabled_cache = {"ts": -1e9, "val": False}


def enabled() -> bool:
    # Continuous generation ON: the worker generates fresh clips with the LANGUID
    # "jito-me" EMOTION_PROMPTS (small, cool gestures) into the per-emotion buffer,
    # which /motion/pool serves. The old quality worry (big/janky gestures, loop
    # snap) is addressed by the calmer prompts + bone damping + the frontend
    # seamless-loop crossfade. Set MOTION_BESPOKE=0 to fall back to the gateway's
    # curated pool.
    now = time.monotonic()
    if now - _enabled_cache["ts"] < 30.0:
        return bool(_enabled_cache["val"])
    from .motion_routes import _resolve_key  # lazy: avoid import cycle

    val = bool(_resolve_key()) and os.getenv("MOTION_BESPOKE", "1") == "1"
    _enabled_cache.update(ts=now, val=val)
    return val


def _prompt_supported() -> bool:
    # Phase 0 confirmed the gateway honors `prompt`; env can force it off.
    return os.getenv("MOTION_PROMPT", "1") != "0"


def _next_variant(emotion: str) -> int:
    v = _variant[emotion]
    _variant[emotion] = v + 1
    return v


def _mark_active(emotion: str) -> None:
    _active[emotion] = time.monotonic()
    _active.move_to_end(emotion)


def _active_emotions() -> list[str]:
    now = time.monotonic()
    live = [e for e, ts in _active.items() if now - ts <= ACTIVE_TTL_S]
    for e in [e for e in _active if e not in live]:
        _active.pop(e, None)
    # neutral is always kept warm (idle fallback / first sentence).
    return list(dict.fromkeys(["neutral", *live]))


# --- gateway call + retarget ----------------------------------------------- #
async def _gen_one(emotion: str, duration: float, prompt: str | None) -> bytes | None:
    from .motion_routes import PMX_PATH, _resolve_base, _resolve_key  # lazy

    base, key = _resolve_base(), _resolve_key()
    if not key:
        return None
    # `emotion` here is a body KEY (a gesture intent like "point", or one of the 7
    # emotions). The gateway's emotion field must be one of its 7; the actual
    # gesture is carried by `prompt`, which the gateway honors. So pass a base
    # emotion for the field and let the prompt drive the motion.
    gw_emotion = emotion if emotion in POOL_EMOTIONS else "neutral"
    body: dict = {"emotion": gw_emotion, "duration": round(duration, 2)}
    if prompt and _prompt_supported():
        body["prompt"] = prompt
    try:
        r = await _client.post(
            f"{base}/motion/generate?format=smpl",
            headers={"Authorization": f"Bearer {key}"},
            json=body,
            timeout=GEN_TIMEOUT_S,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[motion_cache] generate failed ({emotion}): {e}")
        return None
    if r.status_code != 200 or "json" not in r.headers.get("content-type", ""):
        logger.warning(f"[motion_cache] gateway {r.status_code} for {emotion}")
        return None
    try:
        smpl = r.json()
        return await asyncio.to_thread(smpl_dict_to_vmd_bytes, smpl, str(PMX_PATH))
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[motion_cache] retarget failed ({emotion}): {e}")
        return None


# --- the single worker --------------------------------------------------- #
# Policy for the one 5s/single-slot gateway: KEEP THE FALLBACK ALIVE FIRST.
#   1. FLOOR  — any active emotion (incl. neutral) whose buffer is EMPTY gets a
#      clip first, so the per-sentence pool fallback is always instant. This must
#      never be starved (the previous "bespoke-first" policy starved it, so the
#      buffer went empty exactly during a reply and every fallback hit the slow
#      remote path).
#   2. BESPOKE (opportunistic) — the oldest still-fresh request. A generated clip
#      ALSO seeds its emotion buffer, so a bespoke that lands too late for its own
#      sentence (usual at ~5s gen) is reused, not wasted. Stale requests (their
#      sentence has long played) are dropped, not generated.
#   3. TOP-UP — fill active buffers to full depth / refresh stale ones.
def _push_buffer(emotion: str, vmd: bytes) -> None:
    dq = _buffers[emotion]
    dq.append((time.monotonic(), vmd))
    while len(dq) > BUFFER_DEPTH:
        dq.popleft()
    logger.debug(f"[motion_cache] buffered {emotion} depth={len(dq)}")


def _pick_floor() -> str | None:
    for emo in _active_emotions():
        if len(_buffers[emo]) == 0:
            return emo
    return None


def _pick_refill() -> str | None:
    now = time.monotonic()
    for emo in _active_emotions():
        dq = _buffers[emo]
        if len(dq) < BUFFER_DEPTH or (dq and now - dq[0][0] > REFRESH_S):
            return emo
    return None


def _next_bespoke() -> str | None:
    """Pop the next USABLE bespoke id, discarding entries that are gone, already
    done, or stale (sentence has played — mark error so any late fetch falls back
    and the id stops being a live request; the worker still buffers its clip)."""
    now = time.monotonic()
    while True:
        try:
            mid = _bespoke_q.get_nowait()
        except asyncio.QueueEmpty:
            return None
        ent = _bespoke.get(mid)
        if ent is None or ent.event.is_set():
            continue  # evicted / already fulfilled
        if now - ent.ts > STALE_BESPOKE_S:
            ent.error = True
            ent.event.set()  # let a late get_clip fall back cleanly
            continue  # but keep the emotion warm via top-up, not this dead job
        return mid


async def _worker() -> None:
    logger.info("[motion_cache] continuous generation worker running")
    while True:
        try:
            floor = _pick_floor()
            if floor is not None:  # 1) fallback buffer must never be empty
                vmd = await _gen_one(
                    floor, DEFAULT_DUR, prompt_for_key(floor, _next_variant(floor))
                )
                if vmd is not None:
                    _push_buffer(floor, vmd)
                else:
                    await asyncio.sleep(1.0)
                continue
            mid = _next_bespoke()
            if mid is not None:  # 2) opportunistic bespoke, reused into the buffer
                ent = _bespoke[mid]
                vmd = await _gen_one(
                    ent.emotion,
                    ent.duration,
                    prompt_for_key(ent.emotion, _next_variant(ent.emotion)),
                )
                ent.vmd, ent.error = vmd, vmd is None
                ent.event.set()
                if vmd is not None:
                    _push_buffer(ent.emotion, vmd)
                continue
            emo = _pick_refill()
            if emo is not None:  # 3) top up to full depth / refresh stale
                vmd = await _gen_one(
                    emo, DEFAULT_DUR, prompt_for_key(emo, _next_variant(emo))
                )
                if vmd is not None:
                    _push_buffer(emo, vmd)
                else:
                    await asyncio.sleep(1.0)
                continue
            await asyncio.sleep(IDLE_SLEEP)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[motion_cache] worker error: {e}")
            await asyncio.sleep(1.0)


def start_worker() -> None:
    """Start the continuous HY-Motion buffer worker (server startup hook). No-op
    when the motion subsystem is disabled (no gateway key) OR in gesture-only mode
    (the body is driven by GestureLSM; the HY-Motion pool is retired, so its
    continuous generation is pure waste). The gesture path (start_gesture/get_clip)
    is independent of this worker and keeps running."""
    global _worker_task
    from .motion import gesture_client  # lazy: avoid import cycle

    if gesture_client.enabled():
        logger.info("[motion_cache] gesture-only: HY-Motion pool worker disabled")
        return
    if not enabled():
        logger.info("[motion_cache] disabled (no gateway key or MOTION_BESPOKE=0)")
        return
    if _worker_task is None or _worker_task.done():
        _worker_task = asyncio.create_task(_worker())


# --- public API ------------------------------------------------------------ #
def _evict_bespoke() -> None:
    now = time.monotonic()
    for mid in [m for m, e in _bespoke.items() if now - e.ts > BESPOKE_TTL_S]:
        _bespoke.pop(mid, None)
    while len(_bespoke) > MAX_BESPOKE:
        _bespoke.popitem(last=False)


def start_generation(emotion: str, duration: float, prompt: str | None = None) -> str:
    """Queue a bespoke per-sentence clip; returns its id synchronously (no await,
    never blocks speak()). `prompt` is currently unused (the worker derives an
    emotion template) but kept for a future content-matching upgrade."""
    _mark_active(emotion)
    mid = uuid.uuid4().hex
    _bespoke[mid] = _Entry(emotion, duration)
    _evict_bespoke()
    try:
        _bespoke_q.put_nowait(mid)
    except Exception:  # noqa: BLE001 — queue not ready yet; entry will TTL out
        pass
    return mid


def start_gesture(
    audio_bytes: bytes,
    text: str,
    dur_override: float | None = None,
    t_audio: float | None = None,
) -> str:
    """Fire an AUDIO-DRIVEN co-speech gesture generation (GestureLSM via the
    gateway /v1/gesture) in the background; returns its id synchronously (never
    blocks speech). The clip is served through get_clip / /motion/clip like a
    bespoke clip. Independent of the HY-Motion single-slot worker — GestureLSM is a
    different service on a different GPU, so it runs concurrently. Because the
    gesture needs the finished TTS audio, this is fired from _process_tts (after
    synthesis), not from speak(); it lands a couple seconds into the sentence.

    `t_audio` (time.monotonic of this sentence's FIRST PCM chunk = the moment the voice
    can start) turns on the TTS-vs-motion GAP ledger: how far the finished clip trails
    the start of speech, split into fire-wait (audio we had to buffer before we could
    POST) and generation."""
    from .motion import gesture_client  # lazy: avoid import cycle
    from .motion_routes import PMX_PATH  # lazy

    mid = uuid.uuid4().hex
    ent = _Entry("gesture", 0.0)
    ent.lazy_vmd = os.getenv("MOTION_VMD_EAGER", "0") != "1"
    _bespoke[mid] = ent
    _evict_bespoke()

    async def _run() -> None:
        import time as _time

        t0 = _time.monotonic()
        vmd = joints = None
        try:
            async with _gesture_sem:  # one gateway forward at a time
                if ent.event.is_set():  # cancelled (interrupt) while queued
                    return
                queued = _time.monotonic() - t0

                def _publish(smpl: dict) -> None:
                    """Serve the VRM clip the instant the joints exist — the MMD .vmd
                    retarget that follows is 31-235ms the frontend must not wait for."""
                    if ent.joints is None and not ent.event.is_set():
                        ent.joints = smpl
                        ent.event.set()
                        if t_audio is not None:
                            logger.info(
                                f"[gap-ledger] {mid[:8]} joints published "
                                f"{(_time.monotonic() - t_audio) * 1000:.0f}ms after voice"
                            )

                vmd, joints = await gesture_client.generate_gesture_vmd(
                    audio_bytes,
                    text,
                    None if ent.lazy_vmd else str(PMX_PATH),
                    client=_client,
                    dur_override=dur_override,
                    on_joints=_publish,
                )
            if joints is None and vmd is None:
                logger.warning(
                    f"[gesture] {mid[:8]} produced no clip (gateway/retarget failed)"
                )
            else:
                n_frames = len((joints or {}).get("joints") or [])
                logger.info(
                    f"[gesture-ledger] {mid[:8]} READY: queued {queued:.1f}s + "
                    f"gen {(_time.monotonic() - t0 - queued):.1f}s, "
                    f"clip {n_frames / 30.0:.1f}s@30fps"
                )
                if t_audio is not None:
                    # TTS-vs-MOTION GAP: voice could start at t_audio; the clip is only
                    # usable now. fire_wait = audio we had to buffer before POSTing.
                    _gap = _time.monotonic() - t_audio
                    _fire = t0 - t_audio
                    logger.info(
                        f"[gap-ledger] {mid[:8]} TTS->GestureLSM gap {_gap * 1000:.0f}ms "
                        f"(fire_wait {_fire * 1000:.0f}ms + queue {queued * 1000:.0f}ms "
                        f"+ gen {(_time.monotonic() - t0 - queued) * 1000:.0f}ms)"
                    )
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[motion_cache] gesture gen failed: {e!r}")
        # ready if EITHER form succeeded (MMD needs vmd, VRM needs joints). Never clobber
        # joints already published by _publish (a later failure in the .vmd retarget must
        # not wipe a clip the frontend may already be playing).
        ent.vmd = vmd
        if joints is not None:
            ent.joints = joints
        ent.error = vmd is None and ent.joints is None
        ent.event.set()

    try:
        task = asyncio.get_running_loop().create_task(_run())
        _gesture_tasks.add(task)  # keep a ref so the loop can't GC it mid-flight
        task.add_done_callback(_gesture_tasks.discard)
    except RuntimeError:  # no running loop (shouldn't happen in the async app)
        ent.error = True
        ent.event.set()
    return mid


def start_gesture_text(
    text: str,
    emotion: str | None,
    seconds: float,
    session: str,
    start_event: "asyncio.Event | None" = None,
    use_intent: bool = True,
    gesture: str | None = None,
    prefer_bridge: bool = False,
    lsm_only: bool = False,
    bridge: str | None = None,
) -> str:
    """Fire a TEXT-DRIVEN STREAMING gesture (ActionPlan via /v1/motion/stream) in
    the background; returns its id synchronously (never blocks speech). The entry's
    joints GROW as latents stream in (first frames ~0.6s after launch), served as
    partial clips through /motion/clip-smpl. `start_event` (optional) DEFERS the
    network launch until the caller sets it: the id exists immediately (rides the
    audio-stream-start frame so the frontend can schedule its apply), but the GPU
    work starts only when signalled — used to hold generation until GSV's first
    TTS chunk is out, because both share PC-B GPU1 and a simultaneous launch
    measurably delayed the VOICE (tts_first_chunk 500ms -> 1109ms). Voice speed
    wins; motion starts ~0.3s behind and the clock-synced apply plays it from the
    correct position."""
    from .motion import actionplan_client, motion_lib  # lazy: avoid import cycle

    # WHO called this, and with which flag. Searching for the call sites
    # statically was wrong four times running; asking the caller to name itself
    # settles it however many paths there turn out to be. Once per process.
    global _CALLER_LOGGED
    if not _CALLER_LOGGED:
        _CALLER_LOGGED = True
        import traceback as _tb

        _stack = "".join(_tb.format_stack(limit=8)[:-1])
        logger.info(
            "[gesture] start_gesture_text lsm_only=%s; caller:\n%s", lsm_only, _stack
        )
    mid = uuid.uuid4().hex
    ent = _Entry("gesture", seconds)
    ent.streaming = True
    _bespoke[mid] = ent
    _evict_bespoke()
    # KAWAII fast-path (INSTANT, LOCAL — no GPU gen, no gateway, no lag):
    # LSM-ONLY characters skip the KAWAII library entirely: no one-shot emote, no
    # talk-loop. The BRIDGE below still applies — it is pre-generated GestureLSM
    # motion, not a different animation style — and it is the only thing covering
    # the ~1.4s before the audio-driven clip can exist. Without it the body would
    # simply hold still at the start of every sentence.
    if lsm_only:
        _b = motion_lib.bridge_clip(bridge)
        if _b is not None:
            ent.joints = _b
            ent.keep = True
            ent.event.set()
            logger.info(
                f"[gesture-tag] {mid[:8]} LSM-ONLY bridge '{bridge or 'opener'}' "
                f"({len(_b['joints'])}f) emo={emotion}"
            )
        else:
            # No bridge installed: leave the entry EMPTY rather than substituting a
            # KAWAII clip. The real clip lands shortly; until then the body holds,
            # which is what "GestureLSM only" asks for.
            logger.info(f"[gesture-tag] {mid[:8]} LSM-ONLY (no bridge) emo={emotion}")
        return mid
    # 1) an explicit [g:emote] content tag -> play that cute ONE-SHOT pose.
    gk = (gesture or "").strip().lower()
    if not motion_lib.is_emote(gk):
        # No explicit tag -> deterministic JP keyword cue (greeting/thanks/surprise...).
        # The local LLM emits [g:emote] unreliably; this fires the accent anyway.
        gk = motion_lib.emote_for_text(text) or ""
    if motion_lib.is_emote(gk):
        ent.joints = motion_lib.get(gk)  # loop off (one-shot emote)
        ent.keep = True  # non-consuming: prefetch + playGesture both fetch it
        ent.event.set()
        logger.info(f"[gesture-tag] {mid[:8]} KAWAII emote key={gk} emo={emotion}")
        return mid
    # 1.5) TURN OPENER: play the pre-generated GestureLSM BRIDGE clip. The real
    #      audio-driven clip cannot exist for ~1.0-1.7s (inference; measured), so this
    #      keeps the opening GestureLSM-styled instead of visibly starting on a KAWAII
    #      loop and switching. The real clip crossfades over it when it lands.
    if prefer_bridge:
        _b = motion_lib.bridge_clip(bridge)
        if _b is not None:
            ent.joints = _b
            ent.keep = True
            ent.event.set()
            logger.info(f"[gesture-tag] {mid[:8]} BRIDGE(opener) emo={emotion}")
            return mid
    # 2) DEFAULT: the sentence's emotion picks a KAWAII talk-loop (Speak01-10) that
    #    loops for the WHOLE sentence -> continuous, natural, kawaii co-speech motion
    #    (never a statue mid-speech). This is the common path; ActionPlan gen is retired.
    tkey = motion_lib.talk_key_for_emotion(emotion)
    if motion_lib.has(tkey):
        clip = motion_lib.get(tkey)
        clip["loop"] = True  # frontend wraps the playhead for the sentence's duration
        ent.joints = clip
        ent.keep = True
        ent.event.set()
        logger.info(f"[gesture-tag] {mid[:8]} KAWAII talk key={tkey} emo={emotion}")
        return mid
    # 3) FALLBACK (kawaii lib absent): the old ActionPlan text-stream gen path.
    prompt = actionplan_client.intent_prompt(
        text
    )  # content-only heuristic gen fallback

    async def _run() -> None:
        t0 = time.monotonic()
        frames = 0
        gen_prompt = prompt  # heuristic fallback until the intent LLM resolves
        try:
            # 1) CLASSIFY the line to a library key (GPU0, fast ~6 tokens). A hit
            #    plays the pre-generated clip INSTANTLY — no ActionPlan gen, no
            #    deferral, no GPU1/voice contention, and it lands on time.
            if use_intent:
                key = await actionplan_client.classify_key(text, client=_client)
                if ent.event.is_set():  # interrupted while classifying
                    return
                if key and motion_lib.has(key):
                    ent.joints = motion_lib.get(key)
                    ent.keep = True  # non-consuming (prefetch + playGesture both fetch)
                    ent.event.set()
                    logger.info(
                        f"[gesture-tag] {mid[:8]} LIBRARY(classified) key={key}"
                    )
                    return
            # 2) No library match -> GENERATE via ActionPlan. The content prompt
            #    (intent LLM, GPU0) runs in PARALLEL with the deferred-launch wait.
            intent_task = (
                asyncio.create_task(
                    actionplan_client.intent_prompt_llm(text, None, client=_client)
                )
                if use_intent
                else None
            )
            # Deferred launch: wait for the caller's go signal (first TTS chunk).
            if start_event is not None and not start_event.is_set():
                try:
                    await asyncio.wait_for(start_event.wait(), 15.0)
                except asyncio.TimeoutError:  # sentence never synthesized — drop
                    if intent_task is not None:
                        intent_task.cancel()
                    return
            if intent_task is not None:
                # Cap the wait for the content-matched prompt. The intent LLM is on
                # GPU0 (no voice contention) and runs DURING the deferred-launch wait,
                # so this only delays the GESTURE, never the voice. 1.2s was too tight
                # — slow responses fell back to the generic "talks casually"; 2.5s
                # (with the ~0.5s head start = ~3s total) nearly eliminates that.
                try:
                    gen_prompt = await asyncio.wait_for(intent_task, 2.5)
                except Exception:  # noqa: BLE001 — timeout/cancel: heuristic ok
                    intent_task.cancel()
            # No app-side semaphore: the service's GPU lock queues requests in
            # arrival order (fires are ~0.5-1s apart, so arrival order = speech
            # order and chaining stays intact). The old shared semaphore held
            # each slot for the FULL round trip, piling queued 2.6-3.7s of
            # delay onto mid-turn sentences.
            if ent.event.is_set():  # cancelled (interrupt) before start
                return
            queued = time.monotonic() - t0
            async for ev in actionplan_client.stream_motion(
                gen_prompt, seconds, session, client=_client
            ):  # noqa: E501 — cancel_all() cancels this task; closing the stream aborts the service-side generation
                if mid not in _bespoke:  # interrupted: stop consuming
                    break
                et = ev.get("type")
                if et == "meta":
                    ent.joints = {
                        "joint_names": ev.get("joint_names") or [],
                        "joints": [],
                        "fps": ev.get("fps", 30),
                        "partial": True,
                        # chained: frame 0 continues the previous segment's end
                        # pose — the frontend takes the short-fade path.
                        "chained": bool(ev.get("chained")),
                        "seed_overlap": 1 if ev.get("chained") else 0,
                    }
                elif et == "frames" and ent.joints is not None:
                    ent.joints["joints"].extend(ev.get("joints") or [])
                    frames = len(ent.joints["joints"])
                    if not ent.event.is_set():
                        logger.info(
                            f"[gesture-ledger] {mid[:8]} FIRST-FRAMES: queued "
                            f"{queued:.2f}s + {time.monotonic() - t0 - queued:.2f}s"
                        )
                        ent.event.set()  # serve partials from now on
                elif et == "done" and ent.joints is not None:
                    ent.joints["partial"] = False
                    logger.info(
                        f"[gesture-ledger] {mid[:8]} READY(text-stream): "
                        f"{frames / 30.0:.1f}s@30fps in "
                        f"{time.monotonic() - t0:.1f}s ({ev.get('gen_ms')}ms gen) "
                        f"prompt={gen_prompt[:70]!r}"
                    )
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[motion_cache] text gesture failed: {e!r}")
        finally:
            # A LIBRARY hit (keep=True) already served a complete clip and returned
            # here — skip the stream-cleanup, which would set ent.error = (frames==0)
            # = True (no frames streamed) and make get_clip_joints 202 forever.
            if not ent.keep:
                # Also runs on task CANCELLATION (interrupt): closing the stream
                # mid-flight is what tells the service to abort the generation.
                if ent.joints is not None:
                    ent.joints["partial"] = False  # stream ended (done/error/abort)
                ent.error = frames == 0
                ent.event.set()
                if ent.error:
                    logger.warning(
                        f"[gesture] {mid[:8]} text-stream produced no frames"
                    )

    try:
        task = asyncio.get_running_loop().create_task(_run())
        _gesture_tasks.add(task)
        task.add_done_callback(_gesture_tasks.discard)
    except RuntimeError:
        ent.error = True
        ent.event.set()
    return mid


# Ids that will NEVER produce a clip: the generation failed, or the clip was
# already handed over and the entry consumed. Kept so the routes can answer
# "stop asking" instead of the "not yet" they answer for an id still in flight.
#
# Without this, 202 means both things at once. A failed generation popped its
# entry and then looked identical to an id the frontend was simply early for, so
# the poll loop kept asking until its own timeout — measured over one session:
# 63 clips that never arrived, polled 1648 times between them, one of them 106
# times. The avatar spends that whole window waiting for a clip that cannot come
# instead of falling back to the procedural body.
#
# Bounded: this is a fixed-size tail of recent ids, not a leak.
_DEAD_MAX = 512
_dead: "OrderedDict[str, str]" = OrderedDict()


def _mark_dead(motion_id: str, reason: str) -> None:
    _dead[motion_id] = reason
    _dead.move_to_end(motion_id)
    while len(_dead) > _DEAD_MAX:
        _dead.popitem(last=False)


def clip_is_dead(motion_id: str) -> str | None:
    """The reason this id will never produce a clip, or None if it might still."""
    return _dead.get(motion_id)


async def get_clip(motion_id: str, wait_s: float = CLIP_WAIT_S) -> bytes | None:
    """Return a bespoke clip's VMD if ready (waiting at most wait_s), else None.
    The wait is safe: the frontend calls this AFTER audio.play(), so it never
    delays speech. Each id is consumed once."""
    ent = _bespoke.get(motion_id)
    if ent is None:
        return None
    if not ent.event.is_set():
        try:
            await asyncio.wait_for(ent.event.wait(), wait_s)
        except asyncio.TimeoutError:
            return None
    _bespoke.pop(motion_id, None)
    if ent.error:
        _mark_dead(motion_id, "generation failed")
        return None
    _mark_dead(motion_id, "already delivered")
    if ent.vmd is None and ent.lazy_vmd and ent.joints:
        # MMD model: build the .vmd now (see start_gesture). Off the loop as far as
        # the GIL allows; only an MMD frontend ever reaches this.
        ent.vmd = await asyncio.to_thread(_vmd_from_joints, ent.joints)
    return ent.vmd


def _vmd_from_joints(joints: dict) -> bytes | None:
    from .motion.smpl_to_mmd import smpl_dict_to_vmd_bytes  # lazy
    from .motion_routes import PMX_PATH  # lazy

    try:
        return smpl_dict_to_vmd_bytes(joints, str(PMX_PATH))
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[motion_cache] lazy .vmd retarget failed: {e!r}")
        return None


async def get_clip_joints(motion_id: str, wait_s: float = CLIP_WAIT_S) -> dict | None:
    """Like get_clip but returns the raw SMPL joints dict ({joint_names, joints, fps})
    for the VRM viewer to retarget itself. Same wait/consume-once semantics — EXCEPT
    text-streaming entries: while generation is in flight the current PARTIAL clip is
    returned (a snapshot copy, entry kept) so the frontend's poll loop can extend the
    playing clip; the entry is consumed on the first fetch after completion."""
    ent = _bespoke.get(motion_id)
    if ent is None:
        return None
    if not ent.event.is_set():
        try:
            await asyncio.wait_for(ent.event.wait(), wait_s)
        except asyncio.TimeoutError:
            return None
    if ent.streaming and not ent.error:
        j = ent.joints
        if j is None:
            return None
        # partial (growing) OR library (keep): serve a snapshot and DO NOT consume,
        # so the concurrent prefetch + playGesture fetches both succeed.
        if j.get("partial") or ent.keep:
            return {**j, "joints": list(j["joints"])}
    _bespoke.pop(motion_id, None)
    if ent.error:
        _mark_dead(motion_id, "generation failed")
        return None
    # Delivered. The entry is gone, so any later poll for this id would sit on
    # 202 until its own deadline; record it as finished instead.
    _mark_dead(motion_id, "already delivered")
    return ent.joints


def get_buffer_clip(emotion: str) -> bytes | None:
    """Pop the oldest fresh buffer clip for an emotion (rotates through the
    ready set), marking the emotion active so the worker keeps it warm. None if
    the buffer is cold — caller falls back to a direct gateway pool fetch."""
    _mark_active(emotion)
    dq = _buffers.get(emotion)
    if dq and len(dq) > 0:
        return dq.popleft()[1]
    return None


def stats() -> dict:
    """Live introspection of the motion cache (debug endpoint)."""
    return {
        "worker_alive": _worker_task is not None and not _worker_task.done(),
        "enabled": enabled(),
        "buffers": {e: len(dq) for e, dq in _buffers.items()},
        "active": list(_active.keys()),
        "bespoke_pending": sum(1 for e in _bespoke.values() if not e.event.is_set()),
        "bespoke_total": len(_bespoke),
    }


def cancel_all() -> None:
    """Drop pending bespoke work (on interrupt / turn clear). Buffers are kept —
    they're conversation-independent and still useful."""
    try:
        from .motion import actionplan_client, gesture_client  # lazy: avoid cycle

        gesture_client.reset_chain()  # a new turn starts a fresh gesture chain
        actionplan_client.reset_chain()
    except Exception:  # noqa: BLE001
        pass
    for ent in _bespoke.values():
        if not ent.event.is_set():
            ent.error = True
            ent.event.set()
    _bespoke.clear()
    # Cancel in-flight gesture tasks: closing their gateway streams makes the
    # service ABORT the queued/running generations (a dead turn's backlog was
    # starving the next turn's live sentences by 15+s).
    for t in list(_gesture_tasks):
        t.cancel()
    try:
        while True:
            _bespoke_q.get_nowait()
    except asyncio.QueueEmpty:
        pass
