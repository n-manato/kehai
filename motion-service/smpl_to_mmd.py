"""SMPL/SMPL-H -> MMD (.pmx `yyb_miku_10th`) retargeting.

THE hardest, highest-risk piece (plan risk #2). This module turns HY-Motion's
SMPL pose sequence (NPZ) into per-frame LOCAL bone quaternions for the MMD
skeleton, which `vmd_writer` then serialises to .vmd.

Status:
  - SMPL_JOINTS / SMPL_TO_MMD bone map below: DONE (real content, verify L/R in Phase 0).
  - The numeric retarget (rest-pose alignment, axis conversion, root handling):
    `# TODO(PC-B)` — must be validated on ONE clip end-to-end before automating.

Reusable references (don't reinvent):
  - GMR (General Motion Retargeting): SMPL-X -> humanoid, real time on CPU.
  - MMD_3D_POSE_Converter: 3D pose -> VMD.
  - Nuthouse01/PMX-VMD-Scripting-Tools: read .pmx rest pose + write .vmd.

Coordinate / rest-pose notes (the usual retarget headaches):
  - SMPL: Y-up, meters, T-pose rest, root = pelvis with GLOBAL translation.
  - MMD : Y-up but LEFT-handed/Z-forward conventions differ; A-pose rest; root =
          センター. yyb_miku kScale ~0.5.
  - We DISCARD SMPL global translation (in-place talker). Optionally keep a small
    vertical センター bob. Per-joint we map SMPL local rotation -> MMD local
    rotation through each skeleton's rest orientation (R_mmd = B_j * R_smpl * B_j^-1
    style change-of-basis; compute B_j from the two rest poses).
"""
from __future__ import annotations

# SMPL 24-joint order (SMPL-H adds 22 hand joints after index 21; we ignore
# fingers for now — see plan risk #4).
SMPL_JOINTS = [
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
    "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
    "neck", "left_collar", "right_collar", "head", "left_shoulder",
    "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "left_hand", "right_hand",
]

# SMPL joint -> MMD (Japanese) bone name on yyb_miku. None = drop.
# NOTE: SMPL "left_*" is the subject's left; MMD 左 is the model's left. They
# usually align, but mirror conventions bite — VERIFY in Phase 0 with one clip.
SMPL_TO_MMD: dict[str, str | None] = {
    "pelvis":         "センター",     # root; translation discarded, rotation kept
    "spine1":         "上半身",
    "spine2":         "上半身2",
    "spine3":         "上半身2",      # MMD has no spine3; fold into 上半身2
    "neck":           "首",
    "head":           "頭",
    "left_collar":    "左肩",
    "right_collar":   "右肩",
    "left_shoulder":  "左腕",
    "right_shoulder": "右腕",
    "left_elbow":     "左ひじ",
    "right_elbow":    "右ひじ",
    "left_wrist":     "左手首",
    "right_wrist":    "右手首",
    "left_hip":       "左足",
    "right_hip":      "右足",
    "left_knee":      "左ひざ",
    "right_knee":     "右ひざ",
    "left_ankle":     "左足首",
    "right_ankle":    "右足首",
    "left_foot":      "左つま先",     # often IK-driven in MMD; may drop
    "right_foot":     "右つま先",
    "left_hand":      None,           # fingers ignored for now
    "right_hand":     None,
}

# Bones whose SMPL global translation we keep a fraction of (in-place bob only).
ROOT_BONE = "センター"


def load_smpl_npz(path: str):
    """Load HY-Motion NPZ. Returns (poses, trans, fps, meta).

    TODO(PC-B): confirm the exact NPZ keys from `local_infer.py` output
    (e.g. 'poses' axis-angle [T, 24*3], 'trans' [T,3], 'mocap_frame_rate').
    """
    raise NotImplementedError("TODO(PC-B): inspect HY-Motion NPZ keys and load")


def retarget(smpl_npz_path: str, pmx_path: str):
    """SMPL sequence -> per-frame MMD local bone quaternions.

    Returns an object consumable by `vmd_writer.write_vmd`:
        { "fps": int, "frames": [ { bone_name: (qx,qy,qz,qw), ... }, ... ] }

    TODO(PC-B):
      1) read MMD rest orientations per bone from `pmx_path` (Nuthouse01).
      2) read SMPL rest (T-pose) joint orientations.
      3) per joint compute change-of-basis B_j; convert SMPL local rotation ->
         MMD local quaternion; handle Y-up/handedness + L/R mirror.
      4) discard global translation (optionally small センター vertical bob).
      5) resample to TARGET_FPS.
    """
    raise NotImplementedError("TODO(PC-B): implement SMPL->MMD numeric retarget")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="SMPL npz -> MMD .vmd (Phase-0 spike)")
    ap.add_argument("npz")
    ap.add_argument("pmx")
    ap.add_argument("-o", "--out", default="out.vmd")
    args = ap.parse_args()

    from vmd_writer import write_vmd  # local import to keep module load cheap

    motion = retarget(args.npz, args.pmx)
    write_vmd(motion, args.out)
    print(f"wrote {args.out}")
