"""Write retargeted MMD bone frames to a .vmd file.

VMD is a documented little-endian binary (header + bone keyframes + ...). The
simplest path is to reuse a vetted writer rather than hand-rolling the struct
packing and quaternion ordering:

  - Nuthouse01/PMX-VMD-Scripting-Tools  (read/write VMD in Python)
  - MMD_3D_POSE_Converter               (3D pose -> VMD reference)

`motion` is what `smpl_to_mmd.retarget` returns:
    { "fps": int, "frames": [ { bone_name(str): (qx,qy,qz,qw), ... }, ... ] }

VMD stores per-bone keyframes: frame index, position (x,y,z), rotation
quaternion (x,y,z,w), and interpolation bytes. For talking gestures we keep
position 0 (rotation-only) except optionally the センター bob.
"""
from __future__ import annotations


def write_vmd(motion: dict, out_path: str, model_name: str = "yyb_miku") -> None:
    """Serialise frames -> .vmd.

    TODO(PC-B): implement using Nuthouse01's vmd writer:
      - build a Vmd with one BoneFrame per (bone, frame): pos=(0,0,0) unless
        ROOT bob, rot=quaternion, default interpolation (20/107 linear bytes).
      - frame index = round(frame_t * 30) for 30fps VMD.
      - encode bone names as Shift-JIS (MMD convention) — Nuthouse01 handles it.
    """
    raise NotImplementedError("TODO(PC-B): write VMD via Nuthouse01 vmd writer")
