"""
Procedurally generate natural "conversational beat" Live2D motions for the Rem
model and register them in its .model3.json.

Why this exists
---------------
The model already ships rich *emotion* gestures (Joy_big, Anger_rage, ...), but
its neutral talking state (the "Idle" motion group) had only ONE clip, so while
the character just chats it loops the same motion and feels robotic.

The frontend already picks a RANDOM, non-repeating clip from whatever motion
group the backend names (and auto-plays random "Idle" clips when nothing else is
active). So simply adding many varied clips to the Idle / Idle_nod groups makes
the body move naturally during conversation with NO backend or frontend code
changes.

A .motion3.json is just keyframe curves over the model's named parameters, so we
synthesize new clips by composing small, on-model head / body / gaze movements.
Amplitudes are kept to ~30-50% of the ranges actually observed in the model's
existing motions, so nothing ever leaves the natural pose space.

Usage
-----
    python scripts/generate_talk_motions.py            # generate + register
    python scripts/generate_talk_motions.py --count 40
    python scripts/generate_talk_motions.py --clean    # remove generated clips

Re-running is safe: it removes previously generated talk_* clips (file prefix
below) and their model3.json entries first, then regenerates.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re

MODEL_DIR = os.path.join("Live2D Characters", "ac_base_rem_mizugi01")
MODEL_JSON = os.path.join(MODEL_DIR, "ac_base_rem_mizugi01.model3.json")
MOTIONS_DIR = os.path.join(MODEL_DIR, "motions")
PREFIX = "AC_rem_mizugi01_talk_"  # marks generated clips for safe cleanup

# The authored neutral-waiting motion. Used as the source of truth for the
# model's rest pose (arm/hand/shoulder positions).
REST_MOTION = os.path.join(MOTIONS_DIR, "AC_rem_mizugi01_act_normal_w.motion3.json")

FPS = 30.0
FADE = 0.8

# Conversational amplitude caps. Each is a deliberately small fraction of the
# range the parameter actually reaches in the model's authored motions, so the
# beats read as "alive while talking" rather than as emotional acting.
CAP = {
    "ParamAngleX": 12.0,      # head turn  (real range +/-24)
    "ParamAngleY": 10.0,      # head nod   (real range +/-30)
    "ParamAngleZ": 9.0,       # head tilt  (real range +/-20)
    "ParamBodyAngleX": 5.0,   # body sway  (real range +/-10)
    "ParamBodyAngleY": 3.0,   # lean fwd   (real range -2.8..6)
    "ParamBodyAngleZ": 1.5,
    "ParamEyeBallX": 0.6,     # gaze
    "ParamEyeBallY": 0.4,
    "ParamBrowLY": 0.3,       # subtle brow for emphasis
    "ParamBrowRY": 0.3,
    "ParamShoulder": 3.0,
}


def jt(t: float) -> float:
    """Small jitter on a normalized time so no two clips line up identically."""
    return min(0.98, max(0.02, t + random.uniform(-0.025, 0.025)))


def jv(v: float) -> float:
    """+/-6% jitter on a value."""
    return v * random.uniform(0.94, 1.06)


# Each template returns {param_id: [(t_norm, value), ...]} with t in [0,1].
# Every channel MUST start and end at 0 so the one-shot clip hands cleanly back
# to the next random pick / resting pose.

def tpl_nod(a: float, s: int) -> dict:
    A = CAP["ParamAngleY"] * a
    return {
        "ParamAngleY": [(0, 0), (jt(0.30), 0), (jt(0.44), -A), (jt(0.56), 0.2 * A),
                        (jt(0.70), -0.45 * A), (jt(0.84), 0), (1, 0)],
        "ParamAngleX": [(0, 0), (0.5, jv(0.18 * CAP["ParamAngleX"] * a) * s), (1, 0)],
        "ParamBrowLY": [(0, 0), (jt(0.4), 0.5 * CAP["ParamBrowLY"] * a),
                        (jt(0.7), 0), (1, 0)],
        "ParamBrowRY": [(0, 0), (jt(0.4), 0.5 * CAP["ParamBrowRY"] * a),
                        (jt(0.7), 0), (1, 0)],
    }


def tpl_tilt(a: float, s: int) -> dict:
    Z = CAP["ParamAngleZ"] * a * s
    return {
        "ParamAngleZ": [(0, 0), (jt(0.26), Z), (jt(0.74), jv(Z)), (1, 0)],
        "ParamAngleX": [(0, 0), (jt(0.30), 0.4 * CAP["ParamAngleX"] * a * s),
                        (jt(0.70), 0.4 * CAP["ParamAngleX"] * a * s), (1, 0)],
        "ParamEyeBallX": [(0, 0), (jt(0.30), 0.3 * s), (jt(0.7), 0.3 * s), (1, 0)],
    }


def tpl_glance(a: float, s: int) -> dict:
    gx = CAP["ParamEyeBallX"] * random.uniform(0.6, 1.0) * s
    return {
        # quick saccade out, hold, quick return
        "ParamEyeBallX": [(0, 0), (jt(0.26), 0), (0.31, gx), (jt(0.60), jv(gx)),
                          (0.66, 0), (1, 0)],
        "ParamAngleX": [(0, 0), (jt(0.33), 0.3 * CAP["ParamAngleX"] * a * s),
                        (jt(0.62), 0.3 * CAP["ParamAngleX"] * a * s), (1, 0)],
        "ParamAngleY": [(0, 0), (jt(0.45), -0.2 * CAP["ParamAngleY"] * a), (1, 0)],
    }


def tpl_lean(a: float, s: int) -> dict:
    bx = CAP["ParamBodyAngleX"] * a * s
    return {
        "ParamBodyAngleX": [(0, 0), (jt(0.30), bx), (jt(0.70), jv(bx)), (1, 0)],
        "ParamBodyAngleY": [(0, 0), (jt(0.4), CAP["ParamBodyAngleY"] * a * 0.6),
                            (jt(0.7), CAP["ParamBodyAngleY"] * a * 0.6), (1, 0)],
        "ParamAngleX": [(0, 0), (jt(0.32), 0.3 * CAP["ParamAngleX"] * a * s),
                        (jt(0.70), 0.3 * CAP["ParamAngleX"] * a * s), (1, 0)],
        "ParamShoulder": [(0, 0), (jt(0.35), CAP["ParamShoulder"] * a * 0.5),
                          (jt(0.7), CAP["ParamShoulder"] * a * 0.5), (1, 0)],
    }


def tpl_emphasis(a: float, s: int) -> dict:
    A = CAP["ParamAngleY"] * a
    return {
        "ParamAngleY": [(0, 0), (jt(0.20), 0), (0.29, -A), (jt(0.42), 0), (1, 0)],
        "ParamBrowLY": [(0, 0), (0.22, CAP["ParamBrowLY"] * a), (jt(0.46), 0), (1, 0)],
        "ParamBrowRY": [(0, 0), (0.22, CAP["ParamBrowRY"] * a), (jt(0.46), 0), (1, 0)],
        "ParamBodyAngleX": [(0, 0), (jt(0.3), 0.3 * CAP["ParamBodyAngleX"] * a * s),
                            (1, 0)],
    }


def tpl_settle(a: float, s: int) -> dict:
    # very low-amplitude multi-axis drift: "just breathing/alive"
    def drift(cap, k):
        return [(0, 0), (jt(0.33), jv(cap * a * 0.3 * k)),
                (jt(0.66), jv(-cap * a * 0.25 * k)), (1, 0)]
    return {
        "ParamAngleX": drift(CAP["ParamAngleX"], s),
        "ParamAngleY": drift(CAP["ParamAngleY"], 1),
        "ParamAngleZ": drift(CAP["ParamAngleZ"], -s),
        "ParamBodyAngleX": drift(CAP["ParamBodyAngleX"], s),
    }


# (template, is_nod_like) -- nod-like clips are ALSO registered under Idle_nod
TEMPLATES = [
    (tpl_nod, True),
    (tpl_tilt, False),
    (tpl_glance, False),
    (tpl_lean, False),
    (tpl_emphasis, True),
    (tpl_settle, False),
]


def load_rest_pose() -> dict[str, float]:
    """
    Extract the rest position of every arm/hand/shoulder parameter from the
    authored neutral-waiting motion (its value at t=0).

    Why: emotion motions (act_egao etc.) end with the arms away from the rest
    pose, and a finished motion's last values persist on the model. The
    original single Idle clip animated ALL parameters back to neutral, but the
    generated beats only move head/body/gaze — so without these channels a
    raised arm would stay raised through every following beat. Each beat
    therefore actively eases the arms back to (and holds them at) rest.
    """
    with open(REST_MOTION, encoding="utf-8") as fh:
        data = json.load(fh)
    pat = re.compile(r"arm|hand|shoulder", re.IGNORECASE)
    rest: dict[str, float] = {}
    for curve in data["Curves"]:
        if curve["Target"] == "Parameter" and pat.search(curve["Id"]):
            rest[curve["Id"]] = curve["Segments"][1]  # value at t=0
    return rest


def add_rest_channels(channels: dict, rest_pose: dict[str, float]) -> None:
    """Hold every rest-pose parameter the template didn't animate at its rest
    value for the whole clip. FadeIn blends the stuck pose smoothly into it."""
    for pid, value in rest_pose.items():
        if pid not in channels:
            channels[pid] = [(0, value), (1, value)]


# Anything matching this is a "face" parameter the idle beats should reset to
# neutral, so an idle beat clears whatever face an emotion motion baked in
# (eye-smile, brow shapes, mouth shapes/scales, cheek, tears, cross-eye, gaze,
# eye-highlight, sweat...). EXCLUDES eye-open (blink owns it) and mouth-open
# (lip-sync owns it).
FACE_PARAM_RE = re.compile(r"(eye|brow|mouth|cheek|namida|crosseyed|ase|teeth|tongue)", re.I)
FACE_PARAM_EXCLUDE = {"ParamEyeLOpen", "ParamEyeROpen", "ParamMouthOpenY"}


def load_face_neutral() -> dict[str, float]:
    """
    Collect EVERY face parameter any authored (non-generated) motion animates,
    and the neutral value to hold it at — taken from the authored neutral motion
    (act_normal_w) where it defines one, else 0. This guarantees an idle beat
    resets the full face the previous emotion motion left behind, not just a
    handful of params.
    """
    with open(REST_MOTION, encoding="utf-8") as fh:
        normal = json.load(fh)
    normal_vals = {
        c["Id"]: c["Segments"][1] for c in normal["Curves"] if c["Target"] == "Parameter"
    }

    face_ids: set[str] = set()
    for name in os.listdir(MOTIONS_DIR):
        if not name.endswith(".motion3.json") or name.startswith(PREFIX):
            continue
        with open(os.path.join(MOTIONS_DIR, name), encoding="utf-8") as fh:
            data = json.load(fh)
        for curve in data["Curves"]:
            pid = curve["Id"]
            if (
                curve["Target"] == "Parameter"
                and pid not in FACE_PARAM_EXCLUDE
                and FACE_PARAM_RE.search(pid)
            ):
                # Gaze (EyeBallX/Y) and eye-highlight are also reset to neutral
                # here; the live gaze/wet layers re-add their values on top
                # additively, so this just clears the emotion-motion residue.
                face_ids.add(pid)
    return {pid: normal_vals.get(pid, 0.0) for pid in face_ids}


def add_face_neutral_channels(channels: dict, face_neutral: dict[str, float]) -> None:
    """Hold face params at neutral for the whole clip (unless a template/secondary
    channel already animates that param, e.g. brows on a gesture)."""
    for pid, value in face_neutral.items():
        if pid not in channels:
            channels[pid] = [(0, value), (1, value)]


def add_secondary_noise(channels: dict, a: float) -> None:
    """Sprinkle a tiny drift onto one head axis the template didn't use, so even
    same-template clips never look identical."""
    for cand in ("ParamAngleZ", "ParamAngleX", "ParamBodyAngleX"):
        if cand not in channels:
            cap = CAP[cand]
            sgn = random.choice((-1, 1))
            channels[cand] = [(0, 0), (jt(0.4), jv(cap * a * 0.25) * sgn),
                              (jt(0.75), 0), (1, 0)]
            return


def keyframes_to_segments(kf: list[tuple[float, float]], dur: float) -> list[float]:
    """Connect keyframes with ease-in/ease-out Bezier segments (flat tangents)."""
    pts = [(round(t * dur, 3), round(v, 4)) for t, v in kf]
    segs: list[float] = [pts[0][0], pts[0][1]]
    for (t0, v0), (t1, v1) in zip(pts, pts[1:]):
        dt = t1 - t0
        c1t, c1v = round(t0 + dt / 3, 3), v0
        c2t, c2v = round(t0 + 2 * dt / 3, 3), v1
        segs += [1, c1t, c1v, c2t, c2v, t1, v1]  # 1 = Bezier
    return segs


def build_clip(idx: int, rest_pose: dict[str, float], face_neutral: dict[str, float]) -> tuple[dict, bool]:
    tpl, is_nod = random.choice(TEMPLATES)
    a = random.uniform(0.45, 1.0)        # overall amplitude scale
    s = random.choice((-1, 1))           # left/right mirror
    dur = round(random.uniform(2.2, 3.8), 3)

    channels = tpl(a, s)
    add_secondary_noise(channels, a)
    add_rest_channels(channels, rest_pose)
    add_face_neutral_channels(channels, face_neutral)

    # clamp every value to its cap, then build curves
    curves = []
    total_seg = total_pts = 0
    for pid, kf in channels.items():
        cap = CAP.get(pid)  # rest-pose channels have no cap; keep their value
        if cap is not None:
            kf = [(t, max(-cap, min(cap, v))) for t, v in kf]
        segs = keyframes_to_segments(kf, dur)
        curves.append({"Target": "Parameter", "Id": pid, "Segments": segs})
        n = len(kf)
        total_seg += n - 1
        total_pts += 1 + 3 * (n - 1)

    clip = {
        "Version": 3,
        "Meta": {
            "Duration": dur,
            "Fps": FPS,
            "Loop": False,
            "AreBeziersRestricted": True,
            "CurveCount": len(curves),
            "TotalSegmentCount": total_seg,
            "TotalPointCount": total_pts,
            "UserDataCount": 0,
            "TotalUserDataSize": 0,
        },
        "Curves": curves,
        "UserData": [],
    }
    return clip, is_nod


def clean(model: dict) -> None:
    for f in os.listdir(MOTIONS_DIR):
        if f.startswith(PREFIX):
            os.remove(os.path.join(MOTIONS_DIR, f))
    motions = model["FileReferences"]["Motions"]
    for grp in motions:
        motions[grp] = [e for e in motions[grp]
                        if PREFIX not in os.path.basename(e["File"])]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=30)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--clean", action="store_true", help="remove generated clips and exit")
    args = ap.parse_args()
    random.seed(args.seed)

    with open(MODEL_JSON, encoding="utf-8") as fh:
        model = json.load(fh)

    clean(model)  # idempotent: wipe any previous generation first
    if args.clean:
        with open(MODEL_JSON, "w", encoding="utf-8") as fh:
            json.dump(model, fh, ensure_ascii=False, indent=1)
        print("Removed generated talk clips and entries.")
        return

    motions = model["FileReferences"]["Motions"]
    idle_entries = motions.setdefault("Idle", [])
    nod_entries = motions.setdefault("Idle_nod", [])
    rest_pose = load_rest_pose()
    face_neutral = load_face_neutral()
    print(f"Rest pose: holding {len(rest_pose)} arm/hand/shoulder params at rest.")
    print(f"Face neutral: holding {len(face_neutral)} face params at neutral "
          f"(clears emotion-motion residue): {face_neutral}")

    n_idle = n_nod = 0
    for i in range(1, args.count + 1):
        clip, is_nod = build_clip(i, rest_pose, face_neutral)
        fname = f"{PREFIX}{i:02d}.motion3.json"
        with open(os.path.join(MOTIONS_DIR, fname), "w", encoding="utf-8") as fh:
            json.dump(clip, fh, ensure_ascii=False)
        entry = {"File": f"motions/{fname}", "FadeInTime": FADE, "FadeOutTime": FADE}
        idle_entries.append(entry)
        n_idle += 1
        if is_nod:
            nod_entries.append(dict(entry))
            n_nod += 1

    with open(MODEL_JSON, "w", encoding="utf-8") as fh:
        json.dump(model, fh, ensure_ascii=False, indent=1)

    print(f"Generated {args.count} conversational beats.")
    print(f"  Idle group now: {len(idle_entries)} clips (+{n_idle})")
    print(f"  Idle_nod group now: {len(nod_entries)} clips (+{n_nod})")


if __name__ == "__main__":
    main()
