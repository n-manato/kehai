"""Co-speech gesture via the AICCORE Nexus gateway `/v1/gesture`.

Sends the TTS audio + the assistant's text to GestureLSM (embedded in the Nexus
gateway on PC-B), gets back SMPL-X joint positions, and retargets them to a Miku
`.vmd` with the existing `smpl_to_mmd` pipeline. No `smplx` dependency here — the
SMPL-X forward runs server-side (GPU) and we receive plain joint positions, the
same shape the retargeter already consumes for HY-Motion.

Why audio-driven: GestureLSM generates gestures FROM the speech, so the motion is
beat-synced to what the avatar is actually saying — the thing text->HY-Motion
never did (and HY-Motion also had static fingers + a 5s/single-slot bottleneck).

Config (shared with the HY-Motion motion routes):
    GESTURE_ENABLED=1                       enable this path (default off)
    MOTION_GATEWAY_BASE=https://gpu.aiccore-uno.ai/v1
    MOTION_GATEWAY_KEY / .motion_gateway_key   Bearer key (server-side only)
    GESTURE_PRECISION=f16|f32               joints wire format (default f16 = compact)
"""

from __future__ import annotations

import base64
import io
import os
import time
import wave
from pathlib import Path

import httpx
import numpy as np
from loguru import logger

from .smpl_to_mmd import smpl_dict_to_vmd_bytes

_ROOT = Path(__file__).resolve().parents[3]
_KEY_FILE = _ROOT / ".motion_gateway_key"
_DEFAULT_BASE = "https://gpu.aiccore-uno.ai/v1"

# GestureLSM needs a minimum amount of audio: the gateway returns 502 for clips
# shorter than ~5s (measured: 1.5/2/2.5/4s -> 502, 5/6s -> 200). Most sentences
# are 2-4s, so WITHOUT padding almost every gesture failed and the body fell back
# to the neutral pool ("ほぼうまくいってない"). Pad short clips with trailing
# silence to this length; the model gestures over the speech and settles to a
# rest pose during the silent tail (both verified natural on the avatar).
GESTURE_MIN_SECONDS = 6.0


_WIN_FRAMES, _WIN_OVERLAP, _GEN_FPS = 128, 16, 30


def _window_cover_seconds(seconds: float) -> float:
    """Audio length (s) for which GestureLSM returns at least `seconds` of motion
    (see the window note in start_gesture), never under GESTURE_MIN_SECONDS."""
    import math

    need = max(_WIN_FRAMES, math.ceil(seconds * _GEN_FPS))
    k = max(1, math.ceil((need - _WIN_OVERLAP) / (_WIN_FRAMES - _WIN_OVERLAP)))
    frames = _WIN_OVERLAP + k * (_WIN_FRAMES - _WIN_OVERLAP)
    return max(GESTURE_MIN_SECONDS, (frames + 2) / _GEN_FPS)


def enabled() -> bool:
    return os.getenv("GESTURE_ENABLED", "0") == "1"


def _wav_seconds(audio_bytes: bytes) -> float:
    """Duration of a WAV in seconds, or 0.0 if it isn't parseable (caller then just
    skips the trim hint — never break the gesture path over a metadata read)."""
    try:
        with wave.open(io.BytesIO(audio_bytes), "rb") as w:
            return w.getnframes() / float(w.getframerate())
    except Exception:  # noqa: BLE001
        return 0.0


def _pad_audio_to_min(
    audio_bytes: bytes, min_seconds: float = GESTURE_MIN_SECONDS
) -> bytes:
    """Return WAV bytes padded with trailing silence to >= min_seconds. Clips that
    are already long enough are returned unchanged. Uses the stdlib `wave` module
    (no ffmpeg needed) — the TTS here (gpt_sovits) emits WAV. Degrades to the
    original bytes if the clip isn't a parseable WAV (a long clip may still work;
    a short one would 502, but that's no worse than before)."""
    try:
        with wave.open(io.BytesIO(audio_bytes), "rb") as w:
            params = w.getparams()
            fr = w.getframerate()
            n = w.getnframes()
            frames = w.readframes(n)
        if fr <= 0:
            return audio_bytes
        dur = n / fr
        if dur >= min_seconds:
            return audio_bytes  # long enough already — send as-is
        pad_frames = int((min_seconds - dur) * fr)
        silence = b"\x00" * (pad_frames * params.sampwidth * params.nchannels)
        out = io.BytesIO()
        with wave.open(out, "wb") as o:
            o.setparams(params)
            o.writeframes(frames + silence)
        logger.debug(
            f"[gesture] padded audio {dur:.2f}s -> {min_seconds}s for gesture gen"
        )
        return out.getvalue()
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[gesture] audio pad failed ({e}); sending original")
        return audio_bytes


def _base() -> str:
    return os.getenv("MOTION_GATEWAY_BASE", _DEFAULT_BASE).rstrip("/")


_rr = 0


def _bases() -> list[str]:
    """Gateway base URLs to try in order. MOTION_GATEWAY_BASE may be a COMMA-SEPARATED
    list of LAN SSH-forward ports (http://127.0.0.1:18802,...:18803,...:18804) — one per
    parallel gateway instance on PC-B. We ROUND-ROBIN across them so a burst of lines
    spreads over the instances (parallel gen + parallel transfer over separate tunnels)
    instead of queueing on one. The LAN is ~1.5s/gesture vs the internet tunnel's ~2.8s
    (the tunnel adds ~2s relay latency and strips gzip). MOTION_GATEWAY_FALLBACK (the
    public tunnel) is appended last, so a dropped SSH tunnel still works."""
    global _rr
    raw = os.getenv("MOTION_GATEWAY_BASE", _DEFAULT_BASE)
    lan = [b.strip().rstrip("/") for b in raw.split(",") if b.strip()]
    if not lan:
        lan = [_DEFAULT_BASE]
    if len(lan) > 1:
        _rr = (_rr + 1) % len(lan)
        lan = lan[_rr:] + lan[:_rr]  # rotate the try-order per call
    fb = os.getenv("MOTION_GATEWAY_FALLBACK", "").rstrip("/")
    bases = list(lan)
    if fb and fb not in bases:
        bases.append(fb)
    return bases


def _precision() -> str:
    """Requested joints wire format: "f16" (default; base64 float16 buffer, ~7.5x
    smaller than the nested JSON arrays — 3.4MB -> ~460KB measured) or "f32" (the
    legacy format). Set GESTURE_PRECISION=f32 (or 0/off) to revert."""
    p = os.getenv("GESTURE_PRECISION", "f16").strip().lower()
    return "f32" if p in ("f32", "0", "off") else "f16"


def _decode_joints(payload: dict) -> list | None:
    """Joints from EITHER gateway wire format, as the nested [T][J][3] float list
    every downstream consumer (retarget, the /motion/clip-smpl JSON route, the
    debug dump) already expects. f16 form: base64 little-endian float16, C-order,
    reshaped per "shape"; legacy form: the plain "joints" array. Units (meters)
    and joint_names order are identical in both. None if neither is usable."""
    try:
        b64 = payload.get("joints_f16_b64")
        if payload.get("dtype") == "float16" and b64:
            arr = np.frombuffer(base64.b64decode(b64), dtype="<f2")
            return arr.reshape(payload["shape"]).astype(np.float32).tolist()
    except Exception as e:  # noqa: BLE001 — fall through to the legacy key if present
        logger.warning(f"[gesture] f16 joints decode failed: {e!r}")
    return payload.get("joints") or None


def _key() -> str:
    k = os.getenv("MOTION_GATEWAY_KEY", "").strip()
    if not k and _KEY_FILE.exists():
        try:
            k = _KEY_FILE.read_text(encoding="utf-8").strip()
        except OSError:
            k = ""
    return k


def _normalize_upright(smpl: dict) -> dict:
    """Lock the LOWER body to an upright standing pose. Co-speech motion from
    GestureLSM is full-body — the legs spread, the pelvis leans/turns, the weight
    shifts. On a VTuber avatar that's planted (fixed root, no foot IK) that reads as
    a contorted torso + raised shoulders ("胴体が変に曲がる/肩が上がる"). This keeps the
    UPPER-body gesture but stands the avatar up straight: re-express every frame in a
    FIXED upright pelvis frame (removes the root turn/lean/drift) and FREEZE the leg
    joints at the first frame. Verified offline: pelvis rotation 26°→0, feet planted,
    wrist/head gesture preserved. Applied at generation time so MMD + VRM + the SMPL
    sample viewer all get the same upright motion. Toggle: GESTURE_UPRIGHT=0."""
    try:
        names = smpl["joint_names"]
        J = np.asarray(smpl["joints"], dtype=float)
        n = {name: i for i, name in enumerate(names)}
        if J.ndim != 3 or any(
            k not in n for k in ("Pelvis", "Spine1", "L_Hip", "R_Hip")
        ):
            return smpl

        def _nrm(v):
            m = np.linalg.norm(v)
            return v / m if m > 1e-9 else v

        def _pelvis_frame(f):
            up = _nrm(J[f, n["Spine1"]] - J[f, n["Pelvis"]])
            fwd = _nrm(np.cross(J[f, n["L_Hip"]] - J[f, n["R_Hip"]], up))
            right = _nrm(np.cross(up, fwd))
            return np.stack([right, up, fwd], axis=1)  # world axes as columns

        r0 = _pelvis_frame(0)
        fwd_h = _nrm(np.array([r0[0, 2], 0.0, r0[2, 2]]))  # frame-0 forward, flattened
        up_ref = np.array([0.0, 1.0, 0.0])
        right_ref = _nrm(np.cross(up_ref, fwd_h))
        r_ref = np.stack([right_ref, up_ref, np.cross(right_ref, up_ref)], axis=1)
        pelvis0 = J[0, n["Pelvis"]].copy()
        out = np.empty_like(J)
        for f in range(J.shape[0]):
            c = (
                r_ref @ _pelvis_frame(f).T
            )  # rotate this frame's pelvis onto the upright ref
            out[f] = (c @ (J[f] - J[f, n["Pelvis"]]).T).T + pelvis0
        legs = [
            n[k]
            for k in (
                "L_Hip",
                "L_Knee",
                "L_Ankle",
                "L_Foot",
                "R_Hip",
                "R_Knee",
                "R_Ankle",
                "R_Foot",
            )
            if k in n
        ]
        out[:, legs] = out[0, legs]  # freeze the legs at the standing pose

        return {**smpl, "joints": out.tolist()}
    except Exception as e:  # noqa: BLE001 — never break the gesture path
        logger.warning(f"[gesture] upright normalize failed: {e!r}")
        return smpl


def _face_front(smpl: dict, keep: float | None = None) -> dict:
    """Keep the torso facing the camera through a spoken turn.

    GestureLSM's co-speech steadily TURNS the body: measured over one reply the shoulder
    line drifts from -6 deg to -43 deg and stays there ("話しながら傾いてく"). For a
    front-facing VTuber that reads as her slowly turning away. Rotate the upper body
    (spine and above, arms and fingers with it) back around vertical by (1-keep) of its
    own yaw, pivoting at the spine base — the GESTURE is untouched, only its facing.

    Measured from the CLAVICLE line, not the shoulder joints: those swing with the arms
    and made the correction jitter (twitchy hands). Temporally low-passed for the same
    reason. Runs on every GestureLSM clip, independent of the (off by default) upright
    leg-freeze. Tune: GESTURE_TORSO_YAW (1 = don't correct, 0 = hard lock to front).
    """
    if keep is None:
        keep = float(os.getenv("GESTURE_TORSO_YAW", "0.12"))
    if keep >= 0.999:
        return smpl
    try:
        names = smpl["joint_names"]
        J = np.asarray(smpl["joints"], dtype=float)
        n = {name: i for i, name in enumerate(names)}
        if J.ndim != 3 or any(k not in n for k in ("L_Collar", "R_Collar", "Spine1")):
            return smpl

        upper = [
            n[k]
            for k in ("Spine1", "Spine2", "Spine3", "Neck", "Head", "L_Collar", "R_Collar",
                      "L_Shoulder", "R_Shoulder", "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist",
                      "L_Eye", "R_Eye")
            if k in n
        ]
        upper += [
            i for name, i in n.items()
            if any(t in name for t in ("Thumb", "Index", "Middle", "Ring", "Pinky", "Little"))
        ]
        upper = sorted(set(upper))
        out = J.copy()
        piv_i = n["Spine1"]
        # Reference = the AVATAR's absolute sideways axis (SMPL here is X-side, Y-up,
        # Z-front), NOT the clip's own first frame. Per-clip referencing only removed the
        # drift WITHIN a clip; since each clip is chained from the previous one's end
        # pose, the turn-away still accumulated across a reply (measured: still -32 deg
        # by the end). An absolute reference cannot accumulate.
        sr0 = out[0, n["L_Collar"]] - out[0, n["R_Collar"]]
        if abs(sr0[0]) < 1e-6:
            return smpl
        ref = np.array([1.0 if sr0[0] > 0 else -1.0, 0.0, 0.0])
        # VECTORISED (2026-11-05): the per-frame x per-joint loop took ~22 ms for a 5 s
        # clip on the event loop, which the 20 ms audio frames paced to the PROTO viewer
        # share. Same result: each frame's yaw is measured on its own unrotated collars,
        # low-passed in order (cheap scalars), then every upper joint of every frame is
        # turned in one batched product.
        sr = out[:, n["L_Collar"]] - out[:, n["R_Collar"]]
        m = np.hypot(sr[:, 0], sr[:, 2])
        ok = m >= 1e-6
        cx = np.where(ok, sr[:, 0] / np.where(ok, m, 1.0), 0.0)
        cz = np.where(ok, sr[:, 2] / np.where(ok, m, 1.0), 0.0)
        ang = np.arctan2(ref[2] * cx - ref[0] * cz, ref[0] * cx + ref[2] * cz)
        corr = np.zeros(out.shape[0])
        yaw_s, have = 0.0, False
        for f in range(out.shape[0]):
            if not ok[f]:
                continue  # no rotation for this frame; the low-pass state carries over
            yaw_s = ang[f] if not have else (0.85 * yaw_s + 0.15 * ang[f])  # temporal low-pass
            have = True
            corr[f] = -yaw_s * (1.0 - keep)
        ca, sa = np.cos(corr)[:, None], np.sin(corr)[:, None]
        piv = out[:, piv_i][:, None, :]  # (F,1,3)
        d = out[:, upper] - piv
        out[:, upper, 0] = ca * d[..., 0] + sa * d[..., 2] + piv[..., 0]
        out[:, upper, 2] = -sa * d[..., 0] + ca * d[..., 2] + piv[..., 2]
        # 0.1 mm: the wire values came in as float16 (~3 significant digits), and the
        # full float64 repr made the JSON of an 8 s clip ~2x slower to encode and 2.3x larger.
        out = np.round(out, 4)
        return {**smpl, "joints": out.tolist()}
    except Exception as e:  # noqa: BLE001 — never break the gesture path
        logger.warning(f"[gesture] face-front failed: {e!r}")
        return smpl


# ---- gesture chaining -----------------------------------------------------------
# The gateway can seed each generation with the PREVIOUS clip's final latents
# (latent inpainting on PC-B), so consecutive sentences produce ONE continuous
# motion instead of independent clips that each restart from neutral. The seed is
# an opaque base64 latent tail echoed through us; it expires quickly so a chain
# never spans unrelated replies.
_chain_seed: str | None = None
_chain_ts = 0.0
_CHAIN_TTL_S = 20.0


def reset_chain() -> None:
    """Forget the gesture-chain seed (interrupt / conversation switch)."""
    global _chain_seed
    _chain_seed = None


async def generate_gesture_vmd(
    audio_bytes: bytes,
    text: str,
    pmx_path: str,
    client: httpx.AsyncClient | None = None,
    timeout: float = 120.0,
    dur_override: float | None = None,
    on_joints=None,
) -> tuple[bytes | None, dict | None]:
    """POST the spoken audio (+ text) to the gateway, retarget the returned joints
    to a Miku `.vmd`. Returns `(vmd_bytes, smpl_dict)`: the MMD `.vmd` (or None on
    retarget failure) AND the raw SMPL joints dict `{joint_names, joints, fps}` (for
    the VRM viewer, which retargets on its own). Both are None on a gateway/network
    failure so the caller degrades to the procedural idle (never blocks speech).

    `on_joints(smpl)` (optional) is invoked the INSTANT the SMPL joints are ready —
    before the MMD `.vmd` retarget (31ms warm / 235ms on the first call, which loads the
    PMX). The VRM frontend only needs the joints, so publishing them first takes the
    retarget off the TTS-vs-motion critical path."""
    key = _key()
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    # Pad short sentences up so GestureLSM doesn't 502 on <5s of audio.
    # Tell the gateway the REAL speech length BEFORE padding. We pad up to
    # GESTURE_MIN_SECONDS so GestureLSM won't 502 on a short clip, but the padded tail is
    # silence — generating and shipping gesture frames for it is pure latency (a 1s
    # "こんにちは" was costing a 6s / ~860KB clip). The gateway trims to `dur`.
    _tt0 = time.monotonic()
    real_seconds = _wav_seconds(audio_bytes)
    # GestureLSM makes motion in 128-frame windows (30 fps) that overlap by 16:
    # n frames of audio give 16 + 112 * floor((n - 16) / 112) frames, the rest is
    # dropped. Padded to 6 s that is ONE window, 4.27 s, whatever `dur` asked for —
    # every longer sentence ran out of motion and froze (2026-11-05). Pad to the
    # window count that covers the requested length.
    want = max(real_seconds, dur_override or 0.0)
    pad_to = _window_cover_seconds(want)
    audio_bytes = _pad_audio_to_min(audio_bytes, pad_to)
    _tt_pad = time.monotonic()
    files = {"audio": ("speech.wav", audio_bytes, "audio/wav")}
    data = {"text": text or "", "format": "joints"}
    # Compact response: float16 joints instead of nested JSON arrays. A gateway
    # that predates `precision` ignores the field and answers with plain "joints",
    # which _decode_joints also accepts — so this stays backward compatible.
    if _precision() == "f16":
        data["precision"] = "f16"
    if real_seconds > 0:
        # dur_override: the streaming path fires clips from SHORT audio windows
        # (0.7-2.8s) but needs each clip to keep the body moving until the NEXT
        # clip chains in (~spacing seconds later). Requesting a longer dur keeps
        # the silent-tail motion (model settles naturally — verified) instead of
        # trimming the clip to the window and freezing in a hold. Capped at the
        # padded audio length so the gateway never trims beyond what it has.
        eff = max(real_seconds, dur_override or 0.0)
        eff = min(eff, max(real_seconds, pad_to))
        data["dur"] = f"{eff:.2f}"
    global _chain_seed, _chain_ts
    chain_seed = _chain_seed if (time.time() - _chain_ts) < _CHAIN_TTL_S else None
    if chain_seed:
        data["seed_b64"] = chain_seed

    async def _post(cl: httpx.AsyncClient, base: str) -> httpx.Response:
        return await cl.post(
            f"{base}/gesture", headers=headers, files=files, data=data, timeout=timeout
        )

    # Try the LAN base first, then the tunnel fallback; within each, the shared pooled
    # client first then a FRESH connection: if the gateway (PC-B) service restarted,
    # the pooled keep-alive connections go stale and every reuse fails (connection
    # error / spurious non-200) while a new connection succeeds — which silently killed
    # all gestures until an app restart. A refused LAN connection (SSH tunnel down)
    # fails fast, so the tunnel fallback adds negligible latency.
    payload = None
    payload_size = 0
    for base in _bases():
        for use_shared in [True, False] if client is not None else [False]:
            retrying = use_shared and client is not None
            try:
                if use_shared:
                    r = await _post(client, base)
                else:
                    async with httpx.AsyncClient(timeout=timeout) as c:
                        r = await _post(c, base)
                if r.status_code == 200:
                    # Body is JSON in both wire formats: legacy {"joints":[T][J][3],...}
                    # or f16 {"dtype":"float16","shape":[T,J,3],"joints_f16_b64":...}.
                    # Nexus may label it octet-stream; httpx parses regardless.
                    payload = r.json()
                    payload_size = len(r.content)
                    break
                logger.warning(
                    f"[gesture] gateway {r.status_code} @ {base}: {r.text[:160]!r}"
                    f"{' — retrying with a fresh connection' if retrying else ''}"
                )
            except Exception as e:  # noqa: BLE001  (network/parse — degrade to fallback)
                logger.warning(
                    f"[gesture] gateway call failed @ {base} ({type(e).__name__}: {e})"
                    f"{' — retrying with a fresh connection' if retrying else ''}"
                )
        if payload is not None:
            break
    if payload is None:
        return None, None

    _tt_post = time.monotonic()
    joints = _decode_joints(payload)
    _tt_dec = time.monotonic()
    names = payload.get("joint_names")
    if not joints or not names:
        return None, None
    wire = "f16" if payload.get("dtype") == "float16" else "f32"
    logger.debug(
        f"[gesture] payload {payload_size / 1024:.0f}KB ({wire}, {len(joints)} frames)"
    )
    # keep the chain alive: this clip's final latents seed the NEXT sentence
    if payload.get("seed_out"):
        _chain_seed = payload["seed_out"]
        _chain_ts = time.time()
    overlap = int(payload.get("seed_overlap") or 0)
    smpl = {
        "joint_names": names,
        "joints": joints,
        "fps": payload.get("fps", 30),
        # chained: this clip was generated seeded with the PREVIOUS clip's latents, so
        # its motion CONTINUES the prior clip (seamless). The gateway now does this via
        # latent continuity with seed_overlap=0 (no explicit frame REPLAY), so key off
        # the seed we sent, NOT overlap>0 (that stale check dropped every seam back to
        # matchStart -> a visible pose JUMP = "継ぎ目でがくがく").
        "chained": bool(chain_seed),
        "seed_overlap": overlap,
        # Explicit source tag. The frontend used to infer "this is GestureLSM" from the
        # joint count (>=40), which broke the moment the KAWAII clips also carried
        # fingers. Tag it instead of counting.
        "gesturelsm": True,
    }
    # Optional lower-body upright/freeze (default OFF): the frontend now plants the
    # feet with IK and lets the pelvis follow the motion, which reads better than a
    # frozen lower body, so we serve the raw joints. Flip GESTURE_UPRIGHT=1 to freeze.
    if os.getenv("GESTURE_UPRIGHT", "0") == "1":
        smpl = _normalize_upright(smpl)
    # Always: undo the slow turn-away the model builds up over a reply (see _face_front).
    smpl = _face_front(smpl)
    # DEBUG: capture the exact live gateway output (raw joints + text) so it can be
    # re-rendered / re-retargeted offline to diagnose what the user actually sees.
    if os.getenv("MOTION_DEBUG_SAVE") == "1":
        try:
            import datetime
            import json as _json

            dbg = _ROOT / "cache" / "gesture_debug"
            dbg.mkdir(parents=True, exist_ok=True)
            stamp = datetime.datetime.now().strftime("%H%M%S_%f")
            (dbg / f"g_{stamp}.json").write_text(
                _json.dumps({**smpl, "text": text or ""}), encoding="utf-8"
            )
        except Exception:  # noqa: BLE001
            pass
    # Publish the joints NOW (VRM path) — the .vmd retarget below is MMD-only.
    if on_joints is not None:
        try:
            on_joints(smpl)
        except Exception:  # noqa: BLE001 — never let a consumer break generation
            logger.warning("[gesture] on_joints callback failed", exc_info=False)
    # pmx_path=None: no MMD retarget here (motion_cache builds it on the first
    # /motion/clip request instead). It is 60-170 ms of pure Python per clip, ran on
    # the event loop for every clip, and only an MMD model ever asks for it.
    vmd = None
    if pmx_path:
        try:
            vmd = smpl_dict_to_vmd_bytes(smpl, pmx_path)
        except Exception:  # noqa: BLE001
            vmd = None
    _tt_end = time.monotonic()
    logger.info(
        "[gesture-prof] pad {:.0f}ms + post {:.0f}ms + decode {:.0f}ms + vmd {:.0f}ms "
        "= {:.0f}ms (audio {:.1f}s)".format(
            (_tt_pad - _tt0) * 1000,
            (_tt_post - _tt_pad) * 1000,
            (_tt_dec - _tt_post) * 1000,
            (_tt_end - _tt_dec) * 1000,
            (_tt_end - _tt0) * 1000,
            real_seconds,
        )
    )
    return vmd, smpl
