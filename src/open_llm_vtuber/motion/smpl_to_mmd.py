"""SMPL -> MMD (.pmx yyb_miku) retargeting on PA-A, via WORLD-SPACE DIRECTION
MATCHING (robust to local-axis / rest-pose sign mismatches).

Why this approach: PC-B's local-rotation copy keeps producing the "arm swings
behind the body on the settle" defect because it assumes the SMPL and MMD bone
local axes line up (they don't). Here we instead AIM each MMD bone so its
rest-direction (to its child) points along the corresponding SMPL limb segment's
WORLD direction. Aiming in world space is independent of each skeleton's local
axis convention, so front/back can't silently invert. (Generalises the existing
`aimBoneDown` arm-rest trick to a per-frame, per-bone retarget.)

Pipeline:  SMPL joint world positions [T, J, 3]  --aim-->  MMD local bone
quaternions per frame  -->  .vmd

Status: core direction-match + VMD writer + PMX parse are implemented. The SMPL
input loader + the global SMPL->MMD coordinate basis are marked
`# FINALISE(sample)` — lock them once PC-B sends one real sample clip, then
verify with the headless render + FK + QA tooling.

CLI:  python -m open_llm_vtuber.motion.smpl_to_mmd <smpl.json|.npz> <pmx> -o out.vmd
"""

from __future__ import annotations

import json
import math
import os
import struct
import sys


# --------------------------------------------------------------------------- #
# vector / quaternion helpers (quat = (x, y, z, w))
# --------------------------------------------------------------------------- #
def vsub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def vadd(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def vscale(a, s):
    return (a[0] * s, a[1] * s, a[2] * s)


def vlen(a):
    return math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])


def vnorm(a):
    L = vlen(a)
    return (a[0] / L, a[1] / L, a[2] / L) if L > 1e-9 else (0.0, 0.0, 0.0)


def vcross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def vdot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def qmul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def qconj(q):
    return (-q[0], -q[1], -q[2], q[3])


def qrot(q, v):
    x, y, z, w = q
    vx, vy, vz = v
    tx = 2 * (y * vz - z * vy)
    ty = 2 * (z * vx - x * vz)
    tz = 2 * (x * vy - y * vx)
    return (
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    )


def nlerp(a, b, t):
    """Shortest-path normalized lerp between quaternions (approx slerp). t=0 -> a,
    t=1 -> b. Used to DAMPEN a bone toward rest (b = the source rotation, a =
    identity, t = fraction of motion kept)."""
    d = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]
    if d < 0.0:
        b = (-b[0], -b[1], -b[2], -b[3])
    x = a[0] + (b[0] - a[0]) * t
    y = a[1] + (b[1] - a[1]) * t
    z = a[2] + (b[2] - a[2]) * t
    w = a[3] + (b[3] - a[3]) * t
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    return (x / n, y / n, z / n, w / n)


def q_from_to(a, b):
    """Minimal quaternion rotating unit vector a -> unit vector b."""
    a = vnorm(a)
    b = vnorm(b)
    d = vdot(a, b)
    if d >= 1.0 - 1e-8:
        return (0.0, 0.0, 0.0, 1.0)
    if d <= -1.0 + 1e-8:  # anti-parallel: rotate 180 about any perpendicular axis
        ax = vcross((1, 0, 0), a)
        if vlen(ax) < 1e-6:
            ax = vcross((0, 1, 0), a)
        ax = vnorm(ax)
        return (ax[0], ax[1], ax[2], 0.0)
    c = vcross(a, b)
    w = 1.0 + d
    n = math.sqrt(c[0] ** 2 + c[1] ** 2 + c[2] ** 2 + w * w)
    return (c[0] / n, c[1] / n, c[2] / n, w / n)


def axis_angle_to_quat(x, y, z):
    a = math.sqrt(x * x + y * y + z * z)
    if a < 1e-9:
        return (0.0, 0.0, 0.0, 1.0)
    s = math.sin(a / 2) / a
    return (x * s, y * s, z * s, math.cos(a / 2))


def quat_about(axis, angle):
    s = math.sin(angle / 2)
    return (axis[0] * s, axis[1] * s, axis[2] * s, math.cos(angle / 2))


def vperp(v, axis):
    """Component of v perpendicular to unit `axis`, normalised."""
    d = vdot(v, axis)
    return vnorm((v[0] - d * axis[0], v[1] - d * axis[1], v[2] - d * axis[2]))


def signed_angle(a, b, axis):
    """Signed angle to rotate a -> b about `axis` (a, b ~perp to axis)."""
    return math.atan2(vdot(vcross(a, b), axis), vdot(a, b))


def twist_angle_about(q, axis):
    """Swing-twist decomposition: signed angle of q's TWIST about unit `axis`.
    (Projects q's rotation-vector onto axis -> the roll component about it.)"""
    d = q[0] * axis[0] + q[1] * axis[1] + q[2] * axis[2]  # dot(qvec, axis)
    if d * d < 1e-18 and q[3] * q[3] < 1e-18:
        return 0.0
    return 2.0 * math.atan2(d, q[3])


def twist_about(q, axis):
    """Swing-twist decomposition: the TWIST quaternion of q about unit `axis`
    (rotation purely about axis). q == qmul(twist_about(q,a), swing) recovers the
    swing as qmul(qconj(twist), q). Used to route a joint's roll to a twist bone."""
    d = q[0] * axis[0] + q[1] * axis[1] + q[2] * axis[2]  # dot(qvec, axis)
    tq = (axis[0] * d, axis[1] * d, axis[2] * d, q[3])
    n = math.sqrt(tq[0] ** 2 + tq[1] ** 2 + tq[2] ** 2 + tq[3] ** 2)
    if n < 1e-9:  # q is a 180deg swing perpendicular to axis: no defined twist
        return (0.0, 0.0, 0.0, 1.0)
    return (tq[0] / n, tq[1] / n, tq[2] / n, tq[3] / n)


def mat3_to_quat(m):
    """3x3 rotation matrix (row-major) -> quaternion (x, y, z, w)."""
    t = m[0][0] + m[1][1] + m[2][2]
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        return (
            (m[2][1] - m[1][2]) / s,
            (m[0][2] - m[2][0]) / s,
            (m[1][0] - m[0][1]) / s,
            0.25 * s,
        )
    if m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        s = math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2
        return (
            0.25 * s,
            (m[0][1] + m[1][0]) / s,
            (m[0][2] + m[2][0]) / s,
            (m[2][1] - m[1][2]) / s,
        )
    if m[1][1] > m[2][2]:
        s = math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2
        return (
            (m[0][1] + m[1][0]) / s,
            0.25 * s,
            (m[1][2] + m[2][1]) / s,
            (m[0][2] - m[2][0]) / s,
        )
    s = math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2
    return (
        (m[0][2] + m[2][0]) / s,
        (m[1][2] + m[2][1]) / s,
        0.25 * s,
        (m[1][0] - m[0][1]) / s,
    )


def hand_basis(wrist, mid, index, pinky):
    """Orthonormal hand frame from 4 joint positions: x = wrist->middle (where
    the hand points), z = palm normal = x X (index->pinky), y = z X x."""
    x = vnorm(vsub(mid, wrist))
    z = vnorm(vcross(x, vsub(pinky, index)))
    y = vcross(z, x)
    return x, y, z


# --------------------------------------------------------------------------- #
# Collision guard: HY-Motion clips SELF-INTERSECT (measured on real output:
# min inter-wrist 4.7cm < hand width, wrist 7.3cm from the torso axis = inside
# the chest; ~22% of frames). A faithful retarget reproduces that, so the hand
# IK targets are clamped out of a torso capsule and apart to >= a hand's width.
# All operations are continuous in the inputs -> no popping.
# --------------------------------------------------------------------------- #
HAND_MIN_M = (
    0.09  # min inter-wrist distance (SMPL metres): ~ hand width (quality scoring)
)
TORSO_RADIUS_M = 0.13  # torso capsule radius: chest half-depth + clothing margin
SMPL_HALF_DEPTH_M = 0.10  # SMPL body half-depth at the chest (its own front surface)
HAND_CLEAR_M = 0.03  # margin kept between the hand and the dress front
HAND_CENTRE_OFF_M = 0.07  # wrist -> hand centre, along the hand direction
HAND_TIP_OFF_M = 0.13  # wrist -> near-fingertip, along the hand direction
SKIRT_CLEAR_M = 0.0  # margin kept between the hand and the skirt surface. 0 =
# only stop actual penetration (was 0.04 = pushing hands forward off the skirt,
# which splayed the arms unnaturally); physics deflects the rest.
HAND_CENTRE_MIN_M = 0.12  # min distance between the two hand CENTRES (hands are
# ~14cm long: wrist-only spacing still let palms overlap)
GUARD_ITERS = int(os.environ.get("GUARD_ITERS", "0"))  # FLEXIBLE (was 3): collision guard OFF — GestureLSM doesn't self-intersect like HY-Motion did. re-run the 3-point guard + separation until they agree: each
# correction can re-violate an earlier constraint (measured up to 9mm residual)
GUARD_RATE_M = 0.10  # max per-frame CHANGE of the guard correction (metres):
# keeps the guarded target continuous across frames when constraint hand-offs
# (skirt push <-> front clamp) would otherwise flip it in a single frame.
# Measured trade-off on the real clips: 0.05 halves the pop again but lets the
# lagging correction sink hands ~8cm into the dress during fast transitions;
# 0.10 keeps residual penetration ~5mm while cutting the worst elbow pop
# 1.87 -> 0.91 units/frame.


def _closest_on_seg(p, a, b):
    ab = vsub(b, a)
    L2 = vdot(ab, ab)
    t = 0.0 if L2 < 1e-12 else max(0.0, min(1.0, vdot(vsub(p, a), ab) / L2))
    return vadd(a, vscale(ab, t))


def _capsule_push(t, a, b, r):
    """Push point t out of the capsule (segment a-b, radius r). Continuous."""
    c = _closest_on_seg(t, a, b)
    d = vsub(t, c)
    L = vlen(d)
    if L >= r:
        return t
    n = vnorm(d) if L > 1e-6 else (0.0, 0.0, 1.0)  # degenerate: push to the front
    return vadd(c, vscale(n, r))


def clip_quality(smpl: dict) -> dict:
    """Self-intersection metrics of a RAW SMPL clip (units = metres). Returns the
    fraction of frames where the hands interpenetrate each other / the torso —
    used by the PA-A proxy to reject grossly bad clips before retargeting."""
    joints = smpl.get("joints") or []
    names = smpl.get("joint_names") or SMPL_NAMES
    idx = {n: i for i, n in enumerate(names)}

    def j(*alts):
        for a in alts:
            if a in idx:
                return idx[a]
        return None

    lw, rw = j("L_Wrist", "left_wrist"), j("R_Wrist", "right_wrist")
    pe, ne = j("Pelvis", "pelvis"), j("Neck", "neck")
    T = len(joints)
    if T == 0 or None in (lw, rw, pe, ne):
        # Unusable clip (empty, or joints we can't even locate): report the
        # WORST score, not a perfect one — a 0.0/0.0 here made the quality gate
        # select broken clips over merely-imperfect real ones.
        return {"frames": T, "hands_overlap": 1.0, "hand_in_torso": 1.0}
    ww_bad = wt_bad = 0
    for P in joints:
        if vlen(vsub(P[lw], P[rw])) < HAND_MIN_M:
            ww_bad += 1
        dl = vlen(vsub(P[lw], _closest_on_seg(P[lw], P[pe], P[ne])))
        dr = vlen(vsub(P[rw], _closest_on_seg(P[rw], P[pe], P[ne])))
        if min(dl, dr) < TORSO_RADIUS_M - 0.01:
            wt_bad += 1
    return {
        "frames": T,
        "hands_overlap": round(ww_bad / T, 3),
        "hand_in_torso": round(wt_bad / T, 3),
    }


# --------------------------------------------------------------------------- #
# SMPL(-H) body skeleton (first 22 joints; fingers ignored)
# --------------------------------------------------------------------------- #
SMPL_NAMES = [
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
]
SMPL_IDX = {n: i for i, n in enumerate(SMPL_NAMES)}

# Each MMD bone is aimed along the SMPL segment (parent_joint -> child_joint).
# bone : (smpl_parent, smpl_child)
# SMPL joint names follow PC-B's HY-Motion output (SMPL-H order, first 22 = body).
MMD_AIM = {
    "上半身": ("Spine1", "Spine2"),
    "上半身2": ("Spine2", "Spine3"),
    "首": ("Neck", "Head"),
    "左肩": ("L_Collar", "L_Shoulder"),
    "左腕": ("L_Shoulder", "L_Elbow"),
    "左ひじ": ("L_Elbow", "L_Wrist"),
    "右肩": ("R_Collar", "R_Shoulder"),
    "右腕": ("R_Shoulder", "R_Elbow"),
    "右ひじ": ("R_Elbow", "R_Wrist"),
    "左足": ("L_Hip", "L_Knee"),
    "左ひざ": ("L_Knee", "L_Ankle"),
    "左足首": ("L_Ankle", "L_Foot"),
    "右足": ("R_Hip", "R_Knee"),
    "右ひざ": ("R_Knee", "R_Ankle"),
    "右足首": ("R_Ankle", "R_Foot"),
    # Wrist: aim the hand along forearm->middle-finger so it BENDS at the wrist
    # (SMPL wrist bends 15-55 deg; un-retargeted it stays rigidly straight =
    # "hand looks wrong"). Roll is inherited from the forearm (already corrected).
    "左手首": ("L_Wrist", "L_Middle1"),
    "右手首": ("R_Wrist", "R_Middle1"),
    # 頭 has no clean child without face joints -> left at rest.
}

# FOREARM bones get a TWIST/ROLL correction (direction-aim leaves roll
# undetermined -> hand/cuff faces wrong way). Restricted to the forearm (ひじ):
# twisting it rolls the hand about the elbow->wrist axis (= "forearm roll").
# Shoulder/upper-arm are deliberately EXCLUDED -- twisting those swings the whole
# arm (their rest frames differ too much between SMPL and MMD to transfer safely).
ARM_TWIST = {"左ひじ", "右ひじ"}

# Wrist bones are oriented by FULL basis-matching (both pointing direction AND
# palm-facing), not just an aim -- aiming leaves the palm roll undetermined and it
# came out reversed. (mmd 4 bones, smpl 4 joints): wrist, middle, index, pinky.
PALM_REF = {
    "左手首": (
        ("左手首", "左中指１", "左人指１", "左小指１"),
        ("L_Wrist", "L_Middle1", "L_Index1", "L_Pinky1"),
    ),
    "右手首": (
        ("右手首", "右中指１", "右人指１", "右小指１"),
        ("R_Wrist", "R_Middle1", "R_Index1", "R_Pinky1"),
    ),
}

# Arms are solved by POSITION (2-bone IK to a scaled hand target) instead of by
# copying angles, so the hands land where they do relative to the SMPL body (they
# meet / clip correctly on Miku's different proportions). Key = the upper-arm bone
# that triggers the IK (which also solves its forearm); value = SMPL side prefix.
ARM_IK = {"左腕": "L", "右腕": "R"}
ARM_IK_SKIP = {"左ひじ", "右ひじ"}  # solved inside the arm IK, skip in the main loop

# SEMANTIC retarget chain (skips the model's interleaved helper/twist/breast
# bones). yyb_miku's real PMX hierarchy is e.g.
#   上半身2 -> 左肩P -> 左肩 -> 左肩C -> 左腕 -> 左腕捩 -> 左ひじ -> 左手捩 -> 左手首
# and 上半身2's FIRST child is 右胸, 左足's parent is 腰キャンセル左. Aiming with
# children[0] (helper bones) or reading the real parent's world rotation (an
# un-keyed identity helper) corrupts orientation. We instead aim each bone at its
# semantic child and read its semantic parent's world rotation; every bone BETWEEN
# them stays identity (un-keyed), so local = parent^-1 * world lands correctly.
MMD_CHAIN_CHILD = {
    "上半身": "上半身2",
    "上半身2": "首",
    "首": "頭",
    "左肩": "左腕",
    "左腕": "左ひじ",
    "左ひじ": "左手首",
    "左手首": "左中指１",
    "右肩": "右腕",
    "右腕": "右ひじ",
    "右ひじ": "右手首",
    "右手首": "右中指１",
    "左足": "左ひざ",
    "左ひざ": "左足首",
    "左足首": "左つま先",
    "右足": "右ひざ",
    "右ひざ": "右足首",
    "右足首": "右つま先",
}
MMD_CHAIN_PARENT = {
    "上半身2": "上半身",
    "首": "上半身2",
    "左肩": "上半身2",
    "左腕": "左肩",
    "左ひじ": "左腕",
    "左手首": "左ひじ",
    "右肩": "上半身2",
    "右腕": "右肩",
    "右ひじ": "右腕",
    "右手首": "右ひじ",
    "左ひざ": "左足",
    "左足首": "左ひざ",
    "右ひざ": "右足",
    "右足首": "右ひざ",
    # The legs DO hang under the keyed 下半身: the real chain is
    # 左足 <- 腰キャンセル左 <- 下半身, and 腰キャンセル's grant source (腰) is
    # never keyed, so it stays identity and 下半身's rotation FK-propagates into
    # the legs at playback. The leg IK must solve against that parent world or
    # the "planted" feet skate whenever the pelvis sways.
    "左足": "下半身",
    "右足": "下半身",
    # 上半身 is a chain root (real parent is an un-keyed helper).
}

# Legs are ALSO solved by position (2-bone IK), with the ankle target PLANTED at
# its source-mapped absolute position: the pelvis/centre translation (weight
# shift) then bends the knees instead of skating the feet across the floor.
LEG_IK = {"左足": "L", "右足": "R"}
LEG_IK_SKIP = {"左ひざ", "右ひざ"}

# Bones driven by copying the SMPL joint's WORLD ROTATION (pose FK) instead of
# aiming at a child: the head (nods/tilts -- SMPL has no joint above Head) and
# the pelvis (hip sway; MMD 下半身 has no aim child -- legs hang off 腰キャンセル).
# Both skeletons' rest world orientations are identity, so a direct copy is exact.
POSE_COPY = {"頭": "Head", "下半身": "Pelvis"}

# Dampen a few bones toward their REST orientation (keep only a fraction of the
# transferred motion). The raw SMPL is "a person looking DOWN at their own hands
# while gesturing" — for a forward-facing avatar that bows the head/neck far too
# much, and the collar lift shrugs the shoulders up. Keep expressive motion but
# pull these back toward neutral. 1.0 = full motion, 0.0 = frozen at rest.
IDENT = (0.0, 0.0, 0.0, 1.0)
HEAD_DAMP = 1.0  # FLEXIBLE: head follows the source nod/tilt fully (was 0.0 frozen)
# Damp the neck bow + shoulder shrug + SPINE lean toward rest. The raw co-speech
# Torso/neck keep-fraction toward the SOURCE aim (`TORSO_LEAN_W`: 1.0 = full source,
# 0.0 = rest/upright). MEASURED (2026-07-09): the co-speech SOURCE torso (pelvis→neck)
# is near-VERTICAL (5.8° from up, slightly BACK) — the "looking down at hands" is the
# HEAD bow, NOT a torso lean. Applying the full source aim (W=1.0) does NOT reproduce
# that — it OVER-transfers and tips the torso ~14° FORWARD (前傾), a lean the source
# doesn't have. So keep this LOW: 首 0.2 / 上半身 0.35 gives an upright 2° torso that
# actually matches the source. USER wants upright (no 前傾). Env override `TORSO_LEAN_W`
# scales all three if you ever want to dial it. Shoulders keep their light 0.4 damp.
TORSO_LEAN_W = float(os.environ.get("TORSO_LEAN_W", "1.0"))  # FLEXIBLE: 1.0 = follow source (was 0.0 frozen)
_neck = TORSO_LEAN_W
_spine = TORSO_LEAN_W
AIM_DAMP = {"首": _neck, "左肩": 1.0, "右肩": 1.0, "上半身": _spine, "上半身2": _spine}  # FLEXIBLE: shoulders 0.4->1.0
# 下半身 (pelvis) keep-fraction when derived from POSITIONS (co-speech joints path).
# Kept LOW: the pelvis barely affects the look (LBD 0.0 vs 0.6 render near-identical)
# and its source has a constant ~31° anatomical offset we don't want to over-inject.
# A little participation so the hip isn't dead, but the upright look comes from the
# 首/上半身 damp above, not from leaning the pelvis.
LOWER_BODY_DAMP = float(os.environ.get("LOWER_BODY_DAMP", "1.0"))  # FLEXIBLE (was 0.3): pelvis follows source
# Pelvis-over-feet: Miku's hip-midpoint sits ~0.15 (shoulder-widths) FORWARD of the
# planted ankles while the SOURCE's is over them — from the side the feet+neck are
# vertical but the pelvis juts forward = a "く"/banana. This weight pulls the centre
# horizontally so the hips sit above the feet (matching the source's small offset).
# 0=off, 1=full. Env-tunable.
PELVIS_OVER_FEET_W = float(os.environ.get("PELVIS_OVER_FEET_W", "1.0"))
# Straighten the VISIBLE shin to vertical (the sample's shins are ~straight, 1.6°;
# Miku's leg-IK bent them back ~6.7° = the "首から下が斜め"). Pulls the ankle toward
# directly-below-the-knee so the below-skirt shin reads vertical; the thigh keeps its
# IK angle (hidden by the skirt). 0=off, 1=fully vertical shin. Env-tunable.
SHIN_VERTICAL_W = float(os.environ.get("SHIN_VERTICAL_W", "1.0"))

# --- "stay in place" limits (a talking VTuber should not slide/float/pop) ------
# HY-Motion carries global/weight-shift motion in the pelvis+trans; applied 1:1 it
# DRIFTS the avatar sideways up to ~1.5u ("腰から横に動く" → looks like it floats
# around the waist). Keep only a fraction of the horizontal centre translation so
# she stays planted in frame; vertical is kept (it's small + stable).
CENTER_XZ_DAMP = 0.15
# Feet are planted by leg IK at rest+ankle_delta; a lifting SMPL ankle floats the
# foot up to ~0.3u ("足が地面についてない"). Damp the ankle's horizontal delta to
# match the centre (else the foot skates away from the now-planted hips) and cap
# how far a foot may rise above its rest ground height.
FOOT_XZ_DAMP = 0.15
FOOT_RISE_MAX_M = 0.06  # cap how far a foot may rise above rest ground (float)
FOOT_SINK_MAX_M = 0.06  # cap how far a foot may drop below rest ground (sink through floor)
# Arm-IK elbow POLE: computed per-frame from the SMPL bend, it can FLIP 180° between
# frames (near-straight arm → ill-defined bend plane), which teleports the elbow AND
# its child wrist ~1.5u/frame ("肘が変な位置に飛ぶ"). Limit the pole's angular change
# per frame so the bend plane turns smoothly instead of snapping.
MAX_POLE_STEP_RAD = 6.28  # FLEXIBLE (was 0.2): no elbow-pole slew cap. ~11°/frame = 340°/s at 30fps — enough for real elbow
# re-orientation, but a hard cap so an unstable bend plane can't "spin" (くるくる).
# BASE anchor of the elbow bend plane toward the natural down-out resting elbow,
# applied EVERY frame (not only when the pole points up/back). Co-speech source
# elbow azimuth is noisy → without a base anchor the elbow occasionally swings to
# a wrong direction (肘がむいてはいけない方向). A talking VTuber's elbows just hang
# down-out, so trading a little source faithfulness for a stable elbow is correct.
ELBOW_NATURAL_BIAS = float(os.environ.get("ELBOW_NATURAL_BIAS", "0.0"))  # FLEXIBLE (was 0.45): no elbow down-out anchor
# Weight of the FOLD anchor (pole -> down-out for tightly-folded arms). Env-gated so
# its interference with the source elbow AZIMUTH (forward reach) can be measured/tuned.
FOLD_ANCHOR_W = float(os.environ.get("FOLD_ANCHOR_W", "0.0"))  # FLEXIBLE (was 1.0): no fold-anchor on the elbow
# Miku's arms are LONGER relative to its (narrow) shoulders than SMPL-X's, so a
# hand target scaled by SHOULDER WIDTH (which faithfully reproduces hand PLACEMENT
# / hgap) leaves the elbow over-folded and the upper arm hanging low vs the source
# (measured: elbow +45°, shoulder -20° on the approved G=3.0 ref clip). The user
# wants gesture SHAPE (joint ANGLES) matched, so REFLEX_MATCH_W keeps each hand
# target's DIRECTION from the arm-root but resets its DISTANCE (law of cosines on
# Miku's own bone lengths) to reproduce the SOURCE elbow flex — extending reach
# OUTWARD only (safe vs the collision guard that runs next). 0=off (pure position
# match), 1=full angle match. Env-overridable for tuning.
REFLEX_MATCH_W = float(os.environ.get("REFLEX_MATCH_W", "1.0"))
# The guard-correction slew (GUARD_RATE_M×scale) is scale-multiplied — at scale≈7.5
# that is ~0.75u/frame, so a hand-off between the skirt push / front clamp can jump
# the hand ~0.6u; plus a raw gesture peaks near 0.7u. Both teleport the hand+elbow to
# a "weird position". Cap the TOTAL per-frame move of the final hand target (absolute
# units): raw talking motion is a median 0.05u/frame (p90 0.43), so 0.45 keeps normal
# gestures and only clips teleports.
HAND_TARGET_MAX_STEP_M = 99.0  # FLEXIBLE (was 0.45): no per-frame hand-move cap

# Fingers: HY-Motion outputs a STATIC natural curl (measured constant over the
# clip). Transferring it replaces the model's stiff flat rest hands. Same aim
# machinery; last phalanx of each finger has no SMPL child (no tip joint) ->
# stays at model rest. ０ helper bones (薬指０ etc.) are skipped like all helpers.
FINGER_AIM = {}
for _pre, _S in (("左", "L"), ("右", "R")):
    FINGER_AIM.update(
        {
            f"{_pre}親指０": (f"{_S}_Thumb1", f"{_S}_Thumb2"),
            f"{_pre}親指１": (f"{_S}_Thumb2", f"{_S}_Thumb3"),
            f"{_pre}人指１": (f"{_S}_Index1", f"{_S}_Index2"),
            f"{_pre}人指２": (f"{_S}_Index2", f"{_S}_Index3"),
            f"{_pre}中指１": (f"{_S}_Middle1", f"{_S}_Middle2"),
            f"{_pre}中指２": (f"{_S}_Middle2", f"{_S}_Middle3"),
            f"{_pre}薬指１": (f"{_S}_Ring1", f"{_S}_Ring2"),
            f"{_pre}薬指２": (f"{_S}_Ring2", f"{_S}_Ring3"),
            f"{_pre}小指１": (f"{_S}_Pinky1", f"{_S}_Pinky2"),
            f"{_pre}小指２": (f"{_S}_Pinky2", f"{_S}_Pinky3"),
        }
    )
    MMD_CHAIN_PARENT.update(
        {
            f"{_pre}親指０": f"{_pre}手首",
            f"{_pre}親指１": f"{_pre}親指０",
            f"{_pre}人指１": f"{_pre}手首",
            f"{_pre}人指２": f"{_pre}人指１",
            f"{_pre}中指１": f"{_pre}手首",
            f"{_pre}中指２": f"{_pre}中指１",
            f"{_pre}薬指１": f"{_pre}手首",
            f"{_pre}薬指２": f"{_pre}薬指１",
            f"{_pre}小指１": f"{_pre}手首",
            f"{_pre}小指２": f"{_pre}小指１",
        }
    )
    MMD_CHAIN_CHILD.update(
        {
            f"{_pre}親指０": f"{_pre}親指１",
            f"{_pre}親指１": f"{_pre}親指２",
            f"{_pre}人指１": f"{_pre}人指２",
            f"{_pre}人指２": f"{_pre}人指３",
            f"{_pre}中指１": f"{_pre}中指２",
            f"{_pre}中指２": f"{_pre}中指３",
            f"{_pre}薬指１": f"{_pre}薬指２",
            f"{_pre}薬指２": f"{_pre}薬指３",
            f"{_pre}小指１": f"{_pre}小指２",
            f"{_pre}小指２": f"{_pre}小指３",
        }
    )
MMD_AIM.update(FINGER_AIM)
MMD_CHAIN_PARENT["頭"] = "首"

# Co-speech (positions-only) hand data over-curls the fingers into a claw/fist on
# every frame (BEAT2 hands are noisy and the model barely articulates them). Keep
# only a fraction of the curl so the hands read as relaxed. Applied ONLY on the
# positions path (smpl_wr is None); HY-Motion's static natural curl is left alone.
FINGER_BONES = frozenset(FINGER_AIM)
FINGER_DAMP = 1.0  # FLEXIBLE (was 0.4): fingers follow the source fully

# SMPL-H body parent chain (by PC-B joint names) for world-rotation FK.
SMPL_PARENT_NAME = {
    "Pelvis": None,
    "L_Hip": "Pelvis",
    "R_Hip": "Pelvis",
    "Spine1": "Pelvis",
    "L_Knee": "L_Hip",
    "R_Knee": "R_Hip",
    "Spine2": "Spine1",
    "L_Ankle": "L_Knee",
    "R_Ankle": "R_Knee",
    "Spine3": "Spine2",
    "L_Foot": "L_Ankle",
    "R_Foot": "R_Ankle",
    "Neck": "Spine3",
    "L_Collar": "Spine3",
    "R_Collar": "Spine3",
    "Head": "Neck",
    "L_Shoulder": "L_Collar",
    "R_Shoulder": "R_Collar",
    "L_Elbow": "L_Shoulder",
    "R_Elbow": "R_Shoulder",
    "L_Wrist": "L_Elbow",
    "R_Wrist": "R_Elbow",
}
_SMPL_FK_ORDER = [
    "Pelvis",
    "Spine1",
    "Spine2",
    "Spine3",
    "Neck",
    "Head",
    "L_Collar",
    "L_Shoulder",
    "L_Elbow",
    "L_Wrist",
    "R_Collar",
    "R_Shoulder",
    "R_Elbow",
    "R_Wrist",
    "L_Hip",
    "L_Knee",
    "L_Ankle",
    "L_Foot",
    "R_Hip",
    "R_Knee",
    "R_Ankle",
    "R_Foot",
]


def smpl_world_rots(pose_row, names):
    """SMPL per-joint LOCAL axis-angle (pose_row [J*3]) -> world quats by FK.
    Result is in SMPL coords (X-left, Y-up, Z-front) == our MMD/three.js frame."""
    idx = {n: i for i, n in enumerate(names)}
    local = {}
    for n in SMPL_PARENT_NAME:
        i = idx.get(n)
        if i is not None and i * 3 + 2 < len(pose_row):  # guard short pose rows
            local[n] = axis_angle_to_quat(
                pose_row[i * 3], pose_row[i * 3 + 1], pose_row[i * 3 + 2]
            )
    world = {}
    for n in _SMPL_FK_ORDER:
        if n not in local:
            continue
        p = SMPL_PARENT_NAME[n]
        world[n] = (
            local[n] if (p is None or p not in world) else qmul(world[p], local[n])
        )
    return world


# FINALISE(sample): SMPL world axes -> MMD world axes. SMPL is Y-up; MMD (three.js
# load) is Y-up but Z is negated and X may need a sign. Lock this 3x3 basis (a
# signed-axis permutation) against one real sample by rendering + comparing.
# Provisional: keep X, up Y, flip Z (left-handed MMD).
def smpl_to_mmd_axes(v):
    return (
        v[0],
        v[1],
        v[2],
    )  # FINALISE(sample): SMPL (X-left,Y-up,Z-front) == three.js MMD frame


def pelvis_world_q_from_positions(fr, idx):
    """Pelvis (下半身) WORLD rotation derived from JOINT POSITIONS alone.

    The co-speech gesture path sends format=joints (positions, no per-joint
    rotations), so POSE_COPY — which copies a source rotation into 下半身 — falls
    back to identity, the pelvis never rotates, and the legs (aimed off 下半身)
    sit at the wrong angle while the torso looks folded at the waist ("足の角度が
    おかしい／数値が腰に行ってない"). Rebuild the pelvis frame from the hip line and
    the pelvis->spine up-axis. SMPL and MMD share the frame (smpl_to_mmd_axes is
    identity) and both rest orientations are identity, so the current pelvis basis
    IS the rest->current rotation. Returns (x,y,z,w) or None if joints are missing.
    """
    need = ("Pelvis", "Spine1", "L_Hip", "R_Hip")
    if any(n not in idx for n in need):
        return None
    P, Sp = fr[idx["Pelvis"]], fr[idx["Spine1"]]
    Lh, Rh = fr[idx["L_Hip"]], fr[idx["R_Hip"]]
    y = vnorm(vsub(Sp, P))  # up: pelvis -> lower spine
    z = vcross(vnorm(vsub(Lh, Rh)), y)  # forward = left x up (right-handed)
    if vlen(z) < 1e-6:  # spine parallel to the hip line (degenerate)
        return None
    z = vnorm(z)
    x = vnorm(vcross(y, z))  # re-orthogonalised left
    # columns of the rotation matrix are the current pelvis axes (rest = identity)
    return mat3_to_quat(
        [[x[0], y[0], z[0]], [x[1], y[1], z[1]], [x[2], y[2], z[2]]]
    )


# --------------------------------------------------------------------------- #
# PMX bone parse (rest world positions + hierarchy) -- ported & validated
# --------------------------------------------------------------------------- #
def parse_pmx_bones(path):
    d = open(path, "rb").read()
    assert d[0:4] == b"PMX ", d[0:4]
    o = 8
    gcount = d[o]
    o += 1
    g = d[o : o + gcount]
    o += gcount
    enc = g[0]
    addUV = g[1]
    vIdx, tIdx, bIdx = g[2], g[3], g[5]
    tenc = "utf-16-le" if enc == 0 else "utf-8"

    def rtext():
        nonlocal o
        ln = struct.unpack_from("<i", d, o)[0]
        o += 4
        s = d[o : o + ln].decode(tenc, "replace")
        o += ln
        return s

    def ridx(sz):
        nonlocal o
        b = d[o : o + sz]
        o += sz
        return struct.unpack({1: "<b", 2: "<h", 4: "<i"}[sz], b)[0]

    rtext()
    rtext()
    rtext()
    rtext()
    vcount = struct.unpack_from("<i", d, o)[0]
    o += 4
    vstat = 12 + 12 + 8 + addUV * 16
    for _ in range(vcount):
        o += vstat
        wt = d[o]
        o += 1
        o += {
            0: bIdx,
            1: bIdx * 2 + 4,
            2: bIdx * 4 + 16,
            3: bIdx * 2 + 4 + 36,
            4: bIdx * 4 + 16,
        }[wt]
        o += 4
    fc = struct.unpack_from("<i", d, o)[0]
    o += 4
    o += fc * vIdx
    tc = struct.unpack_from("<i", d, o)[0]
    o += 4
    for _ in range(tc):
        rtext()
    mc = struct.unpack_from("<i", d, o)[0]
    o += 4
    for _ in range(mc):
        rtext()
        rtext()
        o += 44 + 1 + 20 + tIdx + tIdx + 1
        tr = d[o]
        o += 1
        o += tIdx if tr == 0 else 1
        rtext()
        o += 4
    bc = struct.unpack_from("<i", d, o)[0]
    o += 4
    bones = []
    for _ in range(bc):
        name = rtext()
        rtext()
        rawpos = struct.unpack_from("<3f", d, o)
        o += 12
        # MMDLoader/MMDParser builds the three.js skeleton with Z NEGATED from the
        # raw PMX (verified: 左腕 raw z=-0.10 vs three.js z=+0.10), and applies VMD
        # rotations RAW onto it. So we must retarget in the THREE.JS frame: negate
        # Z here so our aimed rotations match the skeleton they will drive.
        pos = (rawpos[0], rawpos[1], -rawpos[2])
        parent = ridx(bIdx)
        o += 4
        flags = struct.unpack_from("<H", d, o)[0]
        o += 2
        o += bIdx if (flags & 0x0001) else 12
        if flags & 0x0100 or flags & 0x0200:
            o += bIdx + 4
        if flags & 0x0400:
            o += 12
        if flags & 0x0800:
            o += 24
        if flags & 0x2000:
            o += 4
        if flags & 0x0020:
            o += bIdx + 4 + 4
            lc = struct.unpack_from("<i", d, o)[0]
            o += 4
            for _ in range(lc):
                o += bIdx
                o += (1 + 24) if d[o] else 1
        bones.append({"name": name, "pos": pos, "parent": parent})
    return bones


# --------------------------------------------------------------------------- #
# SMPL input loader  -- FINALISE(sample) against PC-B's real format
# --------------------------------------------------------------------------- #
def load_smpl(path):
    """Return (joints [T][J][3] as tuples, names list, fps)."""
    if path.endswith(".json"):
        d = json.load(open(path, encoding="utf-8"))
        names = d.get("joint_names", SMPL_NAMES)
        fps = d.get("fps", 30)
        joints = [[tuple(j) for j in frame] for frame in d["joints"]]
        return joints, names, fps
    # .npz fallback
    import numpy as np  # noqa: PLC0415

    z = np.load(path, allow_pickle=True)
    arr = z["joints"] if "joints" in z else z[z.files[0]]
    names = list(z["joint_names"]) if "joint_names" in z else SMPL_NAMES
    fps = int(z["fps"]) if "fps" in z else 30
    joints = [[tuple(map(float, j)) for j in frame] for frame in arr]
    return joints, names, fps


# --------------------------------------------------------------------------- #
# RETARGET: aim each MMD bone along its SMPL segment, per frame
# --------------------------------------------------------------------------- #
def retarget(
    joints, names, pmx_bones, poses=None, front_z=None, trans=None, capture=None
):
    """capture (QA hook): a list; per frame a dict with the final clamped hand
    targets / dp / scale is appended, so external checks can re-verify the
    guard's constraints without duplicating the target pipeline."""
    idx = {n: i for i, n in enumerate(names)}
    name2bone = {b["name"]: b for b in pmx_bones}
    # Order: both collars BEFORE the arms, so the 2-bone IK (below) can read both
    # shoulder world positions to build the shared chest-centre hand target.
    bone_order = (
        [
            "下半身",
            "上半身",
            "上半身2",
            "首",
            "頭",
            "左肩",
            "右肩",
            "左腕",
            "左ひじ",
            "左手首",
            "右腕",
            "右ひじ",
            "右手首",
        ]
        + sorted(FINGER_AIM)
        + ["左足", "左ひざ", "左足首", "右足", "右ひざ", "右足首"]
    )
    bone_order = [
        b for b in bone_order if b in name2bone and (b in MMD_AIM or b in POSE_COPY)
    ]
    rest = {b["name"]: b["pos"] for b in pmx_bones}

    # rest direction of each MMD bone toward its SEMANTIC child (NOT children[0]:
    # the model interleaves twist/helper/breast bones, so children[0] points the
    # wrong way -- e.g. 上半身2's first child is 右胸, 左腕's is 左腕捩).
    rest_dir = {}
    for bn in bone_order:
        cn = MMD_CHAIN_CHILD.get(bn)
        if cn and cn in rest:
            rest_dir[bn] = vnorm(vsub(rest[cn], rest[bn]))
        else:  # fallback: first real child in the PMX hierarchy
            kids = [
                b
                for b in pmx_bones
                if b["parent"] >= 0 and pmx_bones[b["parent"]]["name"] == bn
            ]
            rest_dir[bn] = (
                vnorm(vsub(kids[0]["pos"], rest[bn])) if kids else (0.0, -1.0, 0.0)
            )

    # SMPL rest-pose world rotations (all-zero axis-angle). We transfer only the
    # roll RELATIVE to rest, so the SMPL/MMD rest-frame axis mismatch can't inject
    # a spurious spin (the absolute-reference version swung the whole arm).
    smpl_rest = (
        smpl_world_rots([0.0] * (len(names) * 3), names) if poses is not None else None
    )

    if not joints:
        return []
    # global scale + frame-0 references for the centre translation / planted feet
    f0 = joints[0]
    sw0 = (
        vlen(vsub(f0[idx["L_Shoulder"]], f0[idx["R_Shoulder"]]))
        if "L_Shoulder" in idx and "R_Shoulder" in idx
        else 0.0
    )
    miku_sw = (
        vlen(vsub(rest["左腕"], rest["右腕"]))
        if "左腕" in rest and "右腕" in rest
        else 0.0
    )
    scale_g = miku_sw / sw0 if sw0 > 1e-6 and miku_sw > 1e-6 else 8.0
    pelvis0 = f0[idx["Pelvis"]] if "Pelvis" in idx else None
    trans0 = trans[0] if trans else None
    ankle0 = {S: f0[idx[f"{S}_Ankle"]] for S in ("L", "R") if f"{S}_Ankle" in idx}

    # Skirt radial profile from the スカート_row_col bone rings (per row: centroid
    # + max radius). Arm<->skirt PHYSICS collision is disabled (it detonated the
    # 108-body skirt), so the retarget itself must keep hands out of the skirt
    # cone: hand points get pushed radially out of these rings.
    _srows: dict[int, list] = {}
    for b in pmx_bones:
        nm = b["name"]
        if nm.startswith("スカート_"):
            parts = nm.split("_")
            if len(parts) >= 3 and parts[1].isdigit():
                _srows.setdefault(int(parts[1]), []).append(b["pos"])
    skirt_prof = []
    for pts in _srows.values():
        cy = sum(p[1] for p in pts) / len(pts)
        cx = sum(p[0] for p in pts) / len(pts)
        cz = sum(p[2] for p in pts) / len(pts)
        rr = max(math.hypot(p[0] - cx, p[2] - cz) for p in pts)
        skirt_prof.append((cy, cx, cz, rr))
    skirt_prof.sort()  # ascending y: [0] = hem ring ... [-1] = waist ring

    frames = []
    prev_corr = {"L": None, "R": None}  # guard-correction slew state (per hand)
    prev_pole = {"L": None, "R": None}  # arm-IK elbow pole slew state (per hand)
    prev_ftarget = {"L": None, "R": None}  # final hand-target slew state (per hand)
    for fi, fr in enumerate(joints):
        smpl_wr = smpl_world_rots(poses[fi], names) if poses is not None else None
        local_q = {}
        world_q = {}  # world rotation of each semantic bone
        world_p = {}  # world position of each semantic bone's ORIGIN
        arm_targets = None  # both hands' clamped IK targets, computed once
        # centre (weight-shift) translation = global root motion vs frame 0.
        # HY-Motion pins the pelvis inside `joints` and carries the global motion
        # in `trans`; summing pelvis delta + trans delta handles either convention.
        if pelvis0 is not None:
            pv = vsub(fr[idx["Pelvis"]], pelvis0)
            if trans is not None and trans0 is not None and fi < len(trans):
                pv = vadd(pv, vsub(trans[fi], trans0))
            dp = vscale(pv, scale_g)
            # Keep her planted in frame: damp the horizontal drift (the source
            # walks/weight-shifts sideways up to ~1.5u), keep the small vertical.
            dp = (dp[0] * CENTER_XZ_DAMP, dp[1], dp[2] * CENTER_XZ_DAMP)
        else:
            dp = (0.0, 0.0, 0.0)
        # Pelvis-over-feet: shift the centre XZ so the hip-midpoint sits above the
        # PLANTED foot-midpoint (matching the source's small pelvis-over-ankle offset),
        # instead of the model's rest stance that juts the pelvis forward ("く" side-on).
        if (
            PELVIS_OVER_FEET_W > 0.0
            and "左足" in rest
            and "右足" in rest
            and ankle0
            and all(f"{S}_Ankle" in idx for S in ("L", "R"))
            and all(b in rest for b in ("左足首", "右足首"))
        ):
            fmid = (0.0, 0.0, 0.0)
            for S, ab in (("L", "左足首"), ("R", "右足首")):
                ad = vscale(vsub(fr[idx[f"{S}_Ankle"]], ankle0[S]), scale_g)
                fmid = vadd(
                    fmid, (rest[ab][0] + ad[0] * FOOT_XZ_DAMP, 0.0, rest[ab][2] + ad[2] * FOOT_XZ_DAMP)
                )
            fmid = vscale(fmid, 0.5)
            rest_hipmid = vscale(vadd(rest["左足"], rest["右足"]), 0.5)
            src_ankmid = vscale(vadd(fr[idx["L_Ankle"]], fr[idx["R_Ankle"]]), 0.5)
            src_off = vscale(vsub(fr[idx["Pelvis"]], src_ankmid), scale_g)  # faithful offset
            want_x, want_z = fmid[0] + src_off[0], fmid[2] + src_off[2]
            cur_x, cur_z = rest_hipmid[0] + dp[0], rest_hipmid[2] + dp[2]
            dp = (
                dp[0] + PELVIS_OVER_FEET_W * (want_x - cur_x),
                dp[1],
                dp[2] + PELVIS_OVER_FEET_W * (want_z - cur_z),
            )
        if vlen(dp) > 1e-4:
            local_q["__center__"] = dp

        def set_world_pos(bn):
            pn = MMD_CHAIN_PARENT.get(bn)
            if pn is not None and pn in world_q and pn in world_p:
                world_p[bn] = vadd(
                    world_p[pn], qrot(world_q[pn], vsub(rest[bn], rest[pn]))
                )
            else:
                world_p[bn] = vadd(rest[bn], dp)  # chain root rides the centre shift

        def child_origin(bn, child_name):
            return vadd(
                world_p[bn], qrot(world_q[bn], vsub(rest[child_name], rest[bn]))
            )

        for bn in bone_order:
            if bn in ARM_IK_SKIP or bn in LEG_IK_SKIP:
                continue  # ひじ/ひざ are solved inside their limb's IK
            # -------- world-rotation copy: head (nod/tilt), pelvis (hip sway) --
            if bn in POSE_COPY:
                set_world_pos(bn)
                pname = MMD_CHAIN_PARENT.get(bn)
                pworld = world_q.get(pname, (0, 0, 0, 1))
                jn = POSE_COPY[bn]
                if smpl_wr is not None and jn in smpl_wr:
                    wq = smpl_wr[jn]  # source rotation available (HY-Motion path)
                elif bn == "下半身":
                    # Positions-only (co-speech): rebuild the pelvis rotation from
                    # the hip/spine positions so 下半身 rotates with the body and the
                    # legs sit correctly (damped like 上半身 to stay upright).
                    pq = pelvis_world_q_from_positions(fr, idx)
                    wq = (
                        qmul(pworld, nlerp(IDENT, qmul(qconj(pworld), pq), LOWER_BODY_DAMP))
                        if pq is not None
                        else pworld
                    )
                else:
                    wq = pworld  # 頭 with no source rotation -> stays at rest
                local = qmul(qconj(pworld), wq)
                if bn == "頭":  # dampen the head bow toward the neck's orientation
                    local = nlerp(IDENT, local, HEAD_DAMP)
                    wq = qmul(pworld, local)
                world_q[bn] = wq
                local_q[bn] = local
                continue
            # -------- 2-bone IK for the leg: ankle PLANTED at its source-mapped
            # absolute position -> the centre (weight-shift) translation bends
            # the knees instead of skating the feet across the floor.
            if bn in LEG_IK:
                S = LEG_IK[bn]
                pre = bn[0]
                hip, knee, ank = f"{pre}足", f"{pre}ひざ", f"{pre}足首"
                if knee not in rest or ank not in rest:  # model without MMD legs
                    world_q[hip] = world_q.get("下半身", (0, 0, 0, 1))
                    local_q[hip] = (0, 0, 0, 1)
                    continue
                set_world_pos(hip)  # hip origin rides the keyed 下半身 rotation
                pw = world_q.get(MMD_CHAIN_PARENT.get(hip, ""), (0, 0, 0, 1))
                needj = (f"{S}_Hip", f"{S}_Knee", f"{S}_Ankle")
                if any(n not in idx for n in needj) or S not in ankle0:
                    world_q[hip] = pw
                    local_q[hip] = (0, 0, 0, 1)
                    world_p[knee] = child_origin(hip, knee)
                    world_q[knee] = pw
                    local_q[knee] = (0, 0, 0, 1)
                    world_p[ank] = child_origin(knee, ank)
                    continue
                Hp = world_p[hip]
                # Ankle PLANTED at rest + source delta. Damp the HORIZONTAL delta
                # to match the (damped) centre so the foot stays under the planted
                # hips instead of skating away, and CAP how far it may rise so a
                # lifting source ankle can't float the foot off the ground.
                ad = vscale(vsub(fr[idx[f"{S}_Ankle"]], ankle0[S]), scale_g)
                ad = (
                    ad[0] * FOOT_XZ_DAMP,
                    max(-FOOT_SINK_MAX_M, min(ad[1], FOOT_RISE_MAX_M)),
                    ad[2] * FOOT_XZ_DAMP,
                )
                target = vadd(rest[ank], ad)
                L1 = vlen(vsub(rest[knee], rest[hip]))
                L2 = vlen(vsub(rest[ank], rest[knee]))
                to = vsub(target, Hp)
                d = vlen(to)
                d = max(abs(L1 - L2) + 1e-3, min(L1 + L2 - 1e-3, d))
                dt = vnorm(to)
                sdir = vnorm(vsub(fr[idx[f"{S}_Ankle"]], fr[idx[f"{S}_Hip"]]))
                ev = vsub(fr[idx[f"{S}_Knee"]], fr[idx[f"{S}_Hip"]])
                pole = vsub(ev, vscale(sdir, vdot(ev, sdir)))
                pole = vsub(pole, vscale(dt, vdot(pole, dt)))
                if vlen(pole) < 1e-4:
                    pole = vsub((0.0, 0.0, 1.0), vscale(dt, vdot((0.0, 0.0, 1.0), dt)))
                pole = vnorm(pole)
                ca = max(-1.0, min(1.0, (L1 * L1 + d * d - L2 * L2) / (2 * L1 * d)))
                sa = math.sqrt(max(0.0, 1.0 - ca * ca))
                kneeP = vadd(Hp, vadd(vscale(dt, L1 * ca), vscale(pole, L1 * sa)))
                wq_h = qmul(
                    q_from_to(qrot(pw, rest_dir[hip]), vnorm(vsub(kneeP, Hp))), pw
                )
                world_q[hip] = wq_h
                local_q[hip] = qmul(qconj(pw), wq_h)
                if SHIN_VERTICAL_W > 0.0:
                    # Re-aim the SHIN vertical: pull the ankle toward directly below
                    # the knee. Thigh (hip->kneeP) is unchanged (skirt-hidden); only
                    # the visible below-knee shin straightens toward the sample.
                    vtar = (kneeP[0], kneeP[1] - L2, kneeP[2])
                    target = vadd(
                        vscale(target, 1.0 - SHIN_VERTICAL_W), vscale(vtar, SHIN_VERTICAL_W)
                    )
                wq_k = qmul(
                    q_from_to(qrot(wq_h, rest_dir[knee]), vnorm(vsub(target, kneeP))),
                    wq_h,
                )
                world_q[knee] = wq_k
                local_q[knee] = qmul(qconj(wq_h), wq_k)
                world_p[knee] = kneeP
                world_p[ank] = target
                continue
            # -------- 2-bone IK for the arm (upper arm + forearm) -------------
            if bn in ARM_IK:
                S = ARM_IK[bn]
                pre = bn[0]
                ua, fa, wr = f"{pre}腕", f"{pre}ひじ", f"{pre}手首"
                set_world_pos(ua)  # shoulder = arm-root origin
                need = (
                    f"{S}_Shoulder",
                    f"{S}_Elbow",
                    f"{S}_Wrist",
                    "L_Shoulder",
                    "R_Shoulder",
                    "L_Wrist",  # arm_targets is built for BOTH hands at once
                    "R_Wrist",
                )
                pw = world_q.get(MMD_CHAIN_PARENT[ua], (0, 0, 0, 1))
                if (
                    any(n not in idx for n in need)
                    or fa not in rest
                    or wr not in rest
                    or any(b2 not in world_p for b2 in ("左肩", "右肩"))
                    or any(b2 not in rest for b2 in ("左腕", "右腕"))
                ):
                    world_q[ua] = pw
                    local_q[ua] = (0, 0, 0, 1)
                    if fa in rest:
                        world_p[fa] = (
                            child_origin(ua, fa) if ua in world_p else rest[fa]
                        )
                        world_q[fa] = pw
                        local_q[fa] = (0, 0, 0, 1)
                        if wr in rest:
                            world_p[wr] = child_origin(fa, wr)
                    continue
                Sp = world_p[ua]
                if arm_targets is None:
                    Lr = child_origin("左肩", "左腕")
                    Rr = child_origin("右肩", "右腕")
                    Cm = vscale(vadd(Lr, Rr), 0.5)  # Miku chest centre
                    Cs = vscale(vadd(fr[idx["L_Shoulder"]], fr[idx["R_Shoulder"]]), 0.5)
                    sws = vlen(vsub(fr[idx["L_Shoulder"]], fr[idx["R_Shoulder"]]))
                    scale = vlen(vsub(Lr, Rr)) / sws if sws > 1e-6 else 1.0
                    # hand targets: SMPL wrists, placed relative to the shared chest
                    # centre and scaled by shoulder width -> reproduces the source
                    # hand placement on Miku's proportions... then COLLISION-CLAMPED
                    # (HY-Motion self-intersects: hands through each other / chest).
                    tL = vadd(Cm, vscale(vsub(fr[idx["L_Wrist"]], Cs), scale))
                    tR = vadd(Cm, vscale(vsub(fr[idx["R_Wrist"]], Cs), scale))
                    # match the SOURCE elbow flex on Miku's longer arms: keep each
                    # target's direction from the arm-root, reset its distance so
                    # the 2-bone arm reproduces the reference elbow angle.
                    if REFLEX_MATCH_W > 0.0:

                        def _reflex(t, root, pre, Ss):
                            ua2, fa2, wr2 = f"{pre}腕", f"{pre}ひじ", f"{pre}手首"
                            if ua2 not in rest or fa2 not in rest or wr2 not in rest:
                                return t
                            L1 = vlen(vsub(rest[fa2], rest[ua2]))
                            L2 = vlen(vsub(rest[wr2], rest[fa2]))
                            up = vsub(fr[idx[f"{Ss}_Elbow"]], fr[idx[f"{Ss}_Shoulder"]])
                            fv = vsub(fr[idx[f"{Ss}_Wrist"]], fr[idx[f"{Ss}_Elbow"]])
                            lu, lf = vlen(up), vlen(fv)
                            if lu < 1e-6 or lf < 1e-6 or L1 < 1e-6 or L2 < 1e-6:
                                return t
                            ct = max(-1.0, min(1.0, vdot(up, fv) / (lu * lf)))
                            d = (L1 * L1 + L2 * L2 + 2 * L1 * L2 * ct) ** 0.5
                            d = max(abs(L1 - L2) + 1e-3, min(L1 + L2 - 1e-3, d))
                            dv = vsub(t, root)
                            dl = vlen(dv)
                            if dl < 1e-6:
                                return t
                            tr = vadd(root, vscale(dv, d / dl))
                            return vadd(
                                vscale(t, 1.0 - REFLEX_MATCH_W),
                                vscale(tr, REFLEX_MATCH_W),
                            )

                        tL = _reflex(tL, Lr, "左", "L")
                        tR = _reflex(tR, Rr, "右", "R")
                    # hand directions (wrist->middle), for the hand-CENTRE spacing
                    hdL = (
                        vnorm(vsub(fr[idx["L_Middle1"]], fr[idx["L_Wrist"]]))
                        if "L_Middle1" in idx
                        else (0.0, 0.0, 0.0)
                    )
                    hdR = (
                        vnorm(vsub(fr[idx["R_Middle1"]], fr[idx["R_Wrist"]]))
                        if "R_Middle1" in idx
                        else (0.0, 0.0, 0.0)
                    )
                    # COLLISION GUARD at THREE points along each hand (wrist, hand
                    # centre, near-fingertip). Clamping only the wrist left the
                    # palm/fingers free to sink into the chest whenever the hand
                    # pointed AT the body ("hands go inside the torso"). Each point
                    # is pushed out of the torso capsule and, in the frontal zone,
                    # in front of the MEASURED dress surface; the wrist target is
                    # translated by every correction so the whole hand clears.
                    # Frontal gate = "is the point in front of the chest axis"
                    # (the old gate used the SOURCE's clearance, which skipped
                    # exactly the chest-touching gestures that clip the worst).
                    hand_guard = None
                    if "首" in world_p:
                        hipC = (
                            vadd(vscale(vadd(rest["左足"], rest["右足"]), 0.5), dp)
                            if "左足" in rest and "右足" in rest
                            else world_p.get("上半身", Cm)
                        )
                        neckP = world_p["首"]
                        r = TORSO_RADIUS_M * scale
                        smpl_front = Cs[2] + SMPL_HALF_DEPTH_M
                        arm_half = (
                            abs(rest["左腕"][0] - rest["右腕"][0]) * 0.5
                            if "左腕" in rest and "右腕" in rest
                            else 1.2
                        )
                        waist_y = rest["上半身"][1] if "上半身" in rest else Cm[1] - 3.0

                        def skirt_push(p):
                            """Radial push out of the skirt cone (rest rings ride
                            the centre shift). Fades below the hem and above the
                            waist ring, where other guards own the space."""
                            if not skirt_prof:
                                return p
                            y = p[1] - dp[1]  # rings are in rest space
                            lo, hi = skirt_prof[0], skirt_prof[-1]
                            ring = None
                            if y >= hi[0]:
                                ring = hi
                            elif y <= lo[0]:
                                ring = lo
                            else:
                                for a, b2 in zip(skirt_prof, skirt_prof[1:]):
                                    if a[0] <= y <= b2[0]:
                                        f = (y - a[0]) / max(b2[0] - a[0], 1e-6)
                                        ring = tuple(
                                            a[i] + f * (b2[i] - a[i]) for i in range(4)
                                        )
                                        break
                            if ring is None:
                                return p
                            w = max(0.0, min(1.0, (y - (lo[0] - 0.8)) / 0.8))
                            w *= max(0.0, min(1.0, (hi[0] + 0.5 - y) / 0.5))
                            if w <= 0.0:
                                return p
                            cx, cz = ring[1] + dp[0], ring[2] + dp[2]
                            need = ring[3] + SKIRT_CLEAR_M * scale
                            dx, dz = p[0] - cx, p[2] - cz
                            dist = math.hypot(dx, dz)
                            if dist >= need:
                                return p
                            # Push direction: radial in xz, BLENDED toward +z
                            # (front) near the ring axis. Pure radial flips sign
                            # as the hand crosses the ring centre and fights the
                            # front clamp (a bistable switch that popped the
                            # target ~2.4 units between adjacent frames).
                            k = min(1.0, dist / (need * 0.5))
                            if dist > 1e-6:
                                nx = k * (dx / dist)
                                nz = k * (dz / dist) + (1.0 - k)
                            else:
                                nx, nz = 0.0, 1.0
                            nl = math.hypot(nx, nz)
                            n = (nx / nl, nz / nl) if nl > 1e-6 else (0.0, 1.0)
                            push = w * (need - dist) * 0.5  # gentle: half the way out
                            return (p[0] + n[0] * push, p[1], p[2] + n[1] * push)

                        def guard(t, hd, sw_j):
                            # wrist point keeps the source's front clearance (the
                            # "not pushed back" fix); centre/tip just must clear.
                            c = sw_j[2] - smpl_front
                            for off, bonus in (
                                (0.0, max(c, 0.0)),
                                (HAND_CENTRE_OFF_M, 0.0),
                                (HAND_TIP_OFF_M, 0.0),
                            ):
                                p = vadd(t, vscale(hd, off * scale))
                                p2 = _capsule_push(p, hipC, neckP, r)
                                p2 = skirt_push(p2)
                                if front_z is not None:
                                    xw = max(
                                        0.0,
                                        min(
                                            1.0,
                                            (arm_half * 1.3 - abs(p[0] - Cm[0]))
                                            / (arm_half * 0.5),
                                        ),
                                    )
                                    yw = max(
                                        0.0, min(1.0, (p[1] - (waist_y - 1.0)) / 1.0)
                                    )
                                    zw = max(
                                        0.0, min(1.0, (p2[2] - Cm[2]) / (0.05 * scale))
                                    )
                                    zmin = (
                                        front_z + dp[2] + (HAND_CLEAR_M + bonus) * scale
                                    )
                                    if p2[2] < zmin:
                                        p2 = (
                                            p2[0],
                                            p2[1],
                                            p2[2] + xw * yw * zw * (zmin - p2[2]),
                                        )
                                t = vadd(t, vsub(p2, p))
                            return t

                        hand_guard = guard
                    # 3) guard + hand-CENTRE separation, ITERATED: each step's
                    # correction can re-violate an earlier constraint (the wrist
                    # clamp, then the centre/tip pushes, then the separation);
                    # a single sequential pass left hands up to ~9mm inside the
                    # dress front. A few passes converge.
                    rawL, rawR = tL, tR
                    hm = HAND_CENTRE_MIN_M * scale
                    for _ in range(GUARD_ITERS):
                        if hand_guard is not None:
                            tL = hand_guard(tL, hdL, fr[idx["L_Wrist"]])
                            tR = hand_guard(tR, hdR, fr[idx["R_Wrist"]])
                        cL = vadd(tL, vscale(hdL, HAND_CENTRE_OFF_M * scale))
                        cR = vadd(tR, vscale(hdR, HAND_CENTRE_OFF_M * scale))
                        v = vsub(cL, cR)
                        dv = vlen(v)
                        if dv < hm:
                            n = vnorm(v) if dv > 1e-6 else (1.0, 0.0, 0.0)
                            push = (hm - dv) * 0.5
                            tL = vadd(tL, vscale(n, push))
                            tR = vsub(tR, vscale(n, push))
                    # 4) temporal rate-limit on the CORRECTION (not the target):
                    # raw motion passes through 1:1, but the guard's adjustment
                    # is slewed, so a constraint hand-off (skirt push <-> front
                    # clamp) can't flip the target within a single frame.
                    step = GUARD_RATE_M * scale
                    arm_targets = {}
                    for side, raw, tt in (("L", rawL, tL), ("R", rawR, tR)):
                        corr = vsub(tt, raw)
                        pc = prev_corr[side]
                        if pc is not None:
                            dc = vsub(corr, pc)
                            dl = vlen(dc)
                            if dl > step:
                                corr = vadd(pc, vscale(dc, step / dl))
                        prev_corr[side] = corr
                        ft = vadd(raw, corr)
                        # absolute cap on the total per-frame hand-target move so
                        # neither the (scale-inflated) guard slew nor a raw gesture
                        # spike can teleport the hand+elbow to a weird position.
                        pft = prev_ftarget[side]
                        if pft is not None:
                            dft = vsub(ft, pft)
                            dl = vlen(dft)
                            if dl > HAND_TARGET_MAX_STEP_M:
                                ft = vadd(pft, vscale(dft, HAND_TARGET_MAX_STEP_M / dl))
                        prev_ftarget[side] = ft
                        arm_targets[side] = ft
                    if capture is not None:
                        capture.append(
                            {
                                "fi": fi,
                                "targets": dict(arm_targets),
                                "raw": {"L": rawL, "R": rawR},
                                "hd": {"L": hdL, "R": hdR},
                                "dp": dp,
                                "scale": scale,
                                "sw": {
                                    "L": fr[idx["L_Wrist"]],
                                    "R": fr[idx["R_Wrist"]],
                                },
                                "smpl_front": smpl_front if hand_guard else None,
                                "Cm": Cm,
                            }
                        )
                target = arm_targets[S]
                L1 = vlen(vsub(rest[fa], rest[ua]))
                L2 = vlen(vsub(rest[wr], rest[fa]))
                to = vsub(target, Sp)
                d = vlen(to)
                d = max(abs(L1 - L2) + 1e-3, min(L1 + L2 - 1e-3, d))  # clamp reach
                dt = vnorm(to)
                # elbow bend plane from the SMPL elbow (pole), so Miku's elbow points
                # the same way as the source instead of snapping to a default.
                sdir = vnorm(vsub(fr[idx[f"{S}_Wrist"]], fr[idx[f"{S}_Shoulder"]]))
                ev = vsub(fr[idx[f"{S}_Elbow"]], fr[idx[f"{S}_Shoulder"]])
                pole_s = vsub(ev, vscale(sdir, vdot(ev, sdir)))  # SMPL bend, perp sdir
                if vlen(pole_s) < 1e-4:
                    pole_s = (0.0, -1.0, 0.0)  # straight arm: default elbow-down
                pole_s = vnorm(pole_s)
                # Carry the bend direction into Miku's frame by the SAME rotation that
                # takes the SMPL aim (sdir) to Miku's target aim (dt), instead of
                # PROJECTING pole onto dt. When the hand is clamped forward (dress
                # front guard), dt points much more forward than sdir; projecting
                # then collapsed the elbow behind the body ("肘が後ろすぎる"). The
                # rotation keeps the elbow's bend angle relative to the arm, so the
                # elbow follows the hand forward and sits at the torso's side.
                pole = qrot(q_from_to(sdir, dt), pole_s)
                pole = vsub(pole, vscale(dt, vdot(pole, dt)))  # re-orthonormalise
                if vlen(pole) < 1e-4:
                    pole = vsub(
                        (0.0, -1.0, 0.0), vscale(dt, vdot((0.0, -1.0, 0.0), dt))
                    )
                pole = vnorm(pole)
                # Redirect an UNNATURAL elbow toward the natural down-and-outward
                # resting elbow (at the torso's side). A talking/resting elbow hangs
                # DOWN; the source's folded/crossed arms + the forward hand-clamp
                # push the bend plane BEHIND the back (pole.z<0) OR straight UP
                # (pole.y>0) on many frames — the elbow "swings to a direction it
                # shouldn't" (肘がむいてはいけない方向に行く). Correct both, weighted by
                # how backward/upward the pole is; a natural down/forward elbow
                # (e.g. arms crossed at the chest) is left untouched.
                bad = ELBOW_NATURAL_BIAS + max(0.0, -pole[2]) + max(0.0, pole[1])
                if bad > 0.0:
                    out_sign = 1.0 if S == "L" else -1.0
                    # Mostly DOWN with only a little out: a resting/talking elbow
                    # hangs down at the torso's side, it does NOT splay out akimbo.
                    natural = (out_sign * 0.35, -1.0, 0.2)  # down-dominant, slight out+fwd
                    natural = vsub(natural, vscale(dt, vdot(natural, dt)))
                    if vlen(natural) > 1e-4:
                        natural = vnorm(natural)
                        w = min(0.8, bad)  # base anchor + extra for up/back poles
                        pole = vadd(vscale(pole, 1.0 - w), vscale(natural, w))
                        pole = vsub(pole, vscale(dt, vdot(pole, dt)))
                        pole = vnorm(pole) if vlen(pole) > 1e-4 else natural
                # FOLD ANCHOR: a folded arm (hand near the shoulder) has an
                # ill-defined bend plane; combined with the collision guard swinging
                # the hand target, the pole rotates continuously → the elbow "spins"
                # (衝突判定により肘がくるくる). Anchor the pole toward the stable
                # down-out elbow, more strongly the more the arm folds. Extended arms
                # (reach>=0.72) keep the source pole (expressive); a tight fold
                # (reach->0.3) gets up to 0.6 anchor so the azimuth can't wander.
                reach = d / max(1e-6, L1 + L2)
                fold = max(0.0, min(1.0, (0.72 - reach) / 0.42))
                if fold > 0.0:
                    out_sign = 1.0 if S == "L" else -1.0
                    anchor = (out_sign * 0.35, -1.0, 0.2)
                    anchor = vsub(anchor, vscale(dt, vdot(anchor, dt)))
                    if vlen(anchor) > 1e-4:
                        anchor = vnorm(anchor)
                        w = 0.6 * fold * FOLD_ANCHOR_W
                        pole = vadd(vscale(pole, 1.0 - w), vscale(anchor, w))
                        pole = vsub(pole, vscale(dt, vdot(pole, dt)))
                        pole = vnorm(pole) if vlen(pole) > 1e-4 else anchor
                # Temporal smoothing: cap the pole's angular change per frame so the
                # elbow bend plane can't FLIP between frames (which teleports the
                # elbow + its child wrist ~1.5u — "肘が変な位置に飛ぶ"). A near-straight
                # source arm has an ill-defined bend plane that jitters; this holds it.
                pp = prev_pole[S]
                if pp is not None:
                    cosang = max(-1.0, min(1.0, vdot(pole, pp)))
                    ang = math.acos(cosang)
                    if ang > MAX_POLE_STEP_RAD:
                        t = MAX_POLE_STEP_RAD / ang
                        blend = vadd(vscale(pp, 1.0 - t), vscale(pole, t))
                        blend = vsub(blend, vscale(dt, vdot(blend, dt)))
                        if vlen(blend) > 1e-4:
                            pole = vnorm(blend)
                prev_pole[S] = pole
                ca = max(-1.0, min(1.0, (L1 * L1 + d * d - L2 * L2) / (2 * L1 * d)))
                sa = math.sqrt(max(0.0, 1.0 - ca * ca))
                elbow = vadd(Sp, vadd(vscale(dt, L1 * ca), vscale(pole, L1 * sa)))
                # upper arm: aim its rest child-dir toward the solved elbow
                wq_ua = qmul(
                    q_from_to(qrot(pw, rest_dir[ua]), vnorm(vsub(elbow, Sp))), pw
                )
                world_q[ua] = wq_ua
                local_q[ua] = qmul(qconj(pw), wq_ua)
                world_p[ua] = Sp
                # forearm: aim toward the wrist target as a PURE HINGE (minimal,
                # no-twist rotation). The old code added the twist of the SMPL
                # elbow's WORLD rotation about the forearm axis here — but that
                # world twist is dominated by how the whole arm is posed (measured
                # 150-270deg on real clips), not the physiological forearm roll,
                # and dumping it on the bend bone candy-wrapped the elbow mesh.
                # Hand orientation (the palm roll) is set at the wrist by the
                # PALM_REF basis-match; the forearm roll that the sleeve needs is
                # routed to the 手捩 twist bone below, keeping ひじ a clean hinge.
                fa_dir = vnorm(vsub(target, elbow))
                wq_fa = qmul(q_from_to(qrot(wq_ua, rest_dir[fa]), fa_dir), wq_ua)
                world_q[fa] = wq_fa
                local_q[fa] = qmul(qconj(wq_ua), wq_fa)
                world_p[fa] = elbow
                world_p[wr] = target
                continue
            # -------- FK aim (spine, collar, legs, feet) + wrist basis-match ---
            sp, sc = MMD_AIM[bn]
            set_world_pos(bn)
            pname = MMD_CHAIN_PARENT.get(bn)
            pworld = world_q.get(pname, (0, 0, 0, 1))
            if sp not in idx or sc not in idx:
                world_q[bn] = pworld
                local_q[bn] = (0, 0, 0, 1)
                continue
            tgt = smpl_to_mmd_axes(vsub(fr[idx[sc]], fr[idx[sp]]))
            if vlen(tgt) < 1e-6:
                world_q[bn] = pworld
                local_q[bn] = (0, 0, 0, 1)
                continue
            tgt = vnorm(tgt)
            cur_world_dir = qrot(pworld, rest_dir[bn])
            wq = qmul(q_from_to(cur_world_dir, tgt), pworld)
            if (
                smpl_wr is not None
                and bn in ARM_TWIST
                and sp in smpl_wr
                and smpl_rest is not None
                and sp in smpl_rest
            ):
                rel = qmul(smpl_wr[sp], qconj(smpl_rest[sp]))
                wq = qmul(quat_about(tgt, twist_angle_about(rel, tgt)), wq)
            # WRIST: full basis-match so the palm faces the right way.
            if bn in PALM_REF:
                (mw, mmid, mind, mpink), (sw, smid, sind, spink) = PALM_REF[bn]
                if all(j in idx for j in (sw, smid, sind, spink)) and all(
                    b in rest for b in (mw, mmid, mind, mpink)
                ):
                    xs, ys, zs = hand_basis(
                        fr[idx[sw]], fr[idx[smid]], fr[idx[sind]], fr[idx[spink]]
                    )
                    xm, ym, zm = hand_basis(
                        rest[mw], rest[mmid], rest[mind], rest[mpink]
                    )
                    R = [
                        [
                            xs[i] * xm[j] + ys[i] * ym[j] + zs[i] * zm[j]
                            for j in range(3)
                        ]
                        for i in range(3)
                    ]
                    wq = mat3_to_quat(R)
            # Dampen the neck bow / shoulder shrug toward rest (see AIM_DAMP).
            if bn in AIM_DAMP:
                wq = qmul(
                    pworld, nlerp(IDENT, qmul(qconj(pworld), wq), AIM_DAMP[bn])
                )
            # Relax the over-curled co-speech fingers toward rest (positions path).
            elif smpl_wr is None and bn in FINGER_BONES:
                wq = qmul(
                    pworld, nlerp(IDENT, qmul(qconj(pworld), wq), FINGER_DAMP)
                )
            # WRIST: distribute the forearm ROLL to the 手捩 twist bone. The palm
            # basis-match can leave a large roll of the hand relative to the (now
            # pure-hinge) forearm; keyed entirely on 手首 it wrings the sleeve at
            # the wrist. Splitting it — twist onto 手捩 (spread along the forearm),
            # swing left on 手首 — keeps both joints' meshes clean, the way MMD's
            # twist bones are meant to be used. World orientation is unchanged.
            tb = f"{bn[0]}手捩" if bn in PALM_REF else None
            if tb is not None and tb in rest:
                rel = qmul(qconj(pworld), wq)  # wrist relative to forearm
                fa_ax = vnorm(vsub(rest[bn], rest[pname]))  # forearm long axis
                tw = twist_about(rel, fa_ax)
                world_q[tb] = qmul(pworld, tw)
                local_q[tb] = tw
                world_q[bn] = wq
                local_q[bn] = qmul(qconj(tw), rel)  # residual swing on the wrist
            else:
                world_q[bn] = wq
                local_q[bn] = qmul(qconj(pworld), wq)
        frames.append(local_q)
    return frames


# --------------------------------------------------------------------------- #
# VMD writer (bone rotation keyframes; rotation-only)
# --------------------------------------------------------------------------- #
def write_vmd_bytes(frames, model="yyb_miku", fps=30) -> bytes:
    buf = bytearray()
    buf += b"Vocaloid Motion Data 0002".ljust(30, b"\x00")
    buf += model.encode("cp932", "replace")[:20].ljust(20, b"\x00")
    # VMD playback is fixed at 30fps: rescale source frame indices so non-30fps
    # clips keep their real-time speed (interp is linear, so spreading/merging
    # keyframes is safe). At fps>30 colliding indices keep the first key.
    scale_fi = 30.0 / float(fps) if fps and fps > 0 else 1.0
    seen = set()
    bone_frames = []
    for fi, lq in enumerate(frames):
        fo = int(fi * scale_fi + 0.5)
        for bn, q in lq.items():
            if bn == "__center__":
                # centre (weight-shift) translation, identity rotation
                if ("センター", fo) not in seen:
                    seen.add(("センター", fo))
                    bone_frames.append(("センター", fo, (0.0, 0.0, 0.0, 1.0), q))
                continue
            if q == (0, 0, 0, 1):
                continue
            if (bn, fo) in seen:
                continue
            seen.add((bn, fo))
            bone_frames.append((bn, fo, q, None))
    buf += struct.pack("<I", len(bone_frames))
    interp = bytes([20, 20, 0, 0] * 4 + [20, 107, 107, 107] * 0).ljust(64, b"\x14")
    for bn, fi, q, tr in bone_frames:
        buf += bn.encode("cp932", "replace")[:15].ljust(15, b"\x00")
        buf += struct.pack("<I", fi)
        # Translation: we hold it in the three.js frame; MMDParser negates Z on
        # load (leftToRightVector), so pre-negate Z here to cancel that.
        t = tr or (0.0, 0.0, 0.0)
        buf += struct.pack("<3f", t[0], t[1], -t[2])
        # We compute rotations in the three.js frame. MMDParser, on load, applies
        # leftToRightQuaternion: q.x=-q.x, q.y=-q.y. So to land on our desired
        # three.js rotation we pre-invert X,Y here (it cancels the parser flip).
        buf += struct.pack("<4f", -q[0], -q[1], q[2], q[3])
        buf += interp[:64]
    buf += struct.pack("<I", 0)  # morph count
    buf += struct.pack("<I", 0)  # camera
    buf += struct.pack("<I", 0)  # light
    buf += struct.pack("<I", 0)  # self shadow
    buf += struct.pack("<I", 0)  # ik
    return bytes(buf)


def write_vmd(frames, out_path, model="yyb_miku"):
    with open(out_path, "wb") as fh:
        fh.write(write_vmd_bytes(frames, model))


# --------------------------------------------------------------------------- #
# In-memory entry point for the PA-A proxy (fetch raw SMPL -> .vmd bytes)
# --------------------------------------------------------------------------- #
_PMX_CACHE: dict[str, list] = {}
_FRONT_CACHE: dict[str, float | None] = {}


def get_pmx_bones(pmx_path: str):
    if pmx_path not in _PMX_CACHE:
        _PMX_CACHE[pmx_path] = parse_pmx_bones(pmx_path)
    return _PMX_CACHE[pmx_path]


def parse_pmx_front_z(path, bones):
    """Measure the model's CLOTHED chest-front surface z (three.js frame, +Z =
    front) from the PMX VERTICES: p98 of z over the waist..neck band, x limited
    to the body width (excludes the twin-tails). This is how the retarget 'knows
    where the body is': hand targets are kept in front of the real dress surface
    (measured ~18cm on yyb_miku vs the 10cm SMPL body the source assumes)."""
    bp = {b["name"]: b["pos"] for b in bones}
    if "上半身" not in bp or "首" not in bp:
        return None
    y_lo, y_hi = bp["上半身"][1], bp["首"][1] + 0.5
    x_max = abs(bp["左腕"][0]) * 1.4 if "左腕" in bp else 1.7
    d = open(path, "rb").read()
    o = 8
    gcount = d[o]
    o += 1
    g = d[o : o + gcount]
    o += gcount
    addUV, bIdx = g[1], g[5]
    for _ in range(4):  # 4 header text fields
        ln = struct.unpack_from("<i", d, o)[0]
        o += 4 + ln
    vcount = struct.unpack_from("<i", d, o)[0]
    o += 4
    zs = []
    for _ in range(vcount):
        x, y, z = struct.unpack_from("<3f", d, o)
        if y_lo <= y <= y_hi and abs(x) <= x_max:
            zs.append(-z)  # three.js frame: z negated (same as bones)
        o += 12 + 12 + 8 + addUV * 16
        wt = d[o]
        o += 1
        o += {
            0: bIdx,
            1: bIdx * 2 + 4,
            2: bIdx * 4 + 16,
            3: bIdx * 2 + 4 + 36,
            4: bIdx * 4 + 16,
        }[wt]
        o += 4
    if not zs:
        return None
    zs.sort()
    return zs[min(len(zs) - 1, int(0.98 * len(zs)))]


def get_pmx_front(pmx_path: str):
    if pmx_path not in _FRONT_CACHE:
        try:
            _FRONT_CACHE[pmx_path] = parse_pmx_front_z(
                pmx_path, get_pmx_bones(pmx_path)
            )
        except Exception:  # noqa: BLE001 — front clamp is an enhancement, never fatal
            _FRONT_CACHE[pmx_path] = None
    return _FRONT_CACHE[pmx_path]


def smpl_dict_to_vmd_bytes(smpl: dict, pmx_path: str) -> bytes:
    """Retarget a raw SMPL motion dict ({joint_names, joints:[T][J][3], fps})
    straight to .vmd bytes. PMX rest skeleton is parsed once and cached."""
    names = smpl.get("joint_names") or SMPL_NAMES
    joints = [[tuple(j) for j in frame] for frame in smpl["joints"]]
    poses = smpl.get("poses")  # [T][J*3] axis-angle -> enables roll/twist correction
    frames = retarget(
        joints,
        names,
        get_pmx_bones(pmx_path),
        poses=poses,
        front_z=get_pmx_front(pmx_path),
        trans=smpl.get("trans"),
    )
    return write_vmd_bytes(frames, fps=smpl.get("fps") or 30)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("smpl")
    ap.add_argument("pmx")
    ap.add_argument("-o", "--out", default="retarget.vmd")
    a = ap.parse_args()
    if a.smpl.endswith(".json"):
        smpl = json.load(open(a.smpl, encoding="utf-8-sig"))
    else:
        import numpy as np  # noqa: PLC0415

        z = np.load(a.smpl, allow_pickle=True)
        smpl = {
            "joint_names": list(z["joint_names"]) if "joint_names" in z else SMPL_NAMES,
            "joints": (z["joints"] if "joints" in z else z[z.files[0]]).tolist(),
            "poses": z["poses"].tolist() if "poses" in z else None,
        }
    vmd = smpl_dict_to_vmd_bytes(smpl, a.pmx)
    open(a.out, "wb").write(vmd)
    print(f"retargeted {len(smpl['joints'])} frames -> {a.out}", file=sys.stderr)
