/* eslint-disable no-underscore-dangle */
import * as THREE from "three";

/**
 * SMPL / SMPL-X REFERENCE FIGURE — the "motion model's own default body".
 *
 * The co-speech motion model (ActionPlan, SMPL 22-joint; or GestureLSM, SMPL-X
 * 55-joint) outputs raw joint POSITIONS. `vrm-motion-player` retargets those onto
 * Miku's VRM humanoid; this class draws the SAME raw joints as a plain articulated
 * body (spheres at joints + cylinders for bones) so it can stand NEXT TO Miku and
 * play the identical clip in perfect sync. It is the zero-retarget ground truth:
 * whatever this figure does is exactly "the movement the motion model intends", so
 * comparing it against Miku shows whether the retarget is faithful.
 *
 * Driven every frame from `vrmMotionPlayer.getCurrentFrame()` (the interpolated,
 * crossfade-blended frame that is ALSO driving Miku this tick) — no separate fetch,
 * no separate clock, so the two bodies can never drift apart.
 *
 * Joint naming follows the SMPL convention used throughout the retarget code
 * (Pelvis, L_Hip, Spine1, L_Collar, L_Shoulder, L_Wrist, ...). Bones are drawn from
 * the kinematic PARENT map for whatever joints are actually present, so the same
 * class renders the 22-joint SMPL body AND the 55-joint SMPL-X body (fingers/face
 * appear automatically when the clip carries them).
 */

// child -> parent, by SMPL joint name. Superset: the 22 body joints plus the
// SMPL-X hand/face joints (only drawn when present in the clip).
const PARENT: Record<string, string> = {
  // spine
  Spine1: "Pelvis", Spine2: "Spine1", Spine3: "Spine2", Neck: "Spine3", Head: "Neck",
  // legs
  L_Hip: "Pelvis", R_Hip: "Pelvis",
  L_Knee: "L_Hip", R_Knee: "R_Hip",
  L_Ankle: "L_Knee", R_Ankle: "R_Knee",
  L_Foot: "L_Ankle", R_Foot: "R_Ankle",
  // arms
  L_Collar: "Spine3", R_Collar: "Spine3",
  L_Shoulder: "L_Collar", R_Shoulder: "R_Collar",
  L_Elbow: "L_Shoulder", R_Elbow: "R_Shoulder",
  L_Wrist: "L_Elbow", R_Wrist: "R_Elbow",
  // SMPL-X head extras
  Jaw: "Head", L_Eye: "Head", R_Eye: "Head",
};
// SMPL-X fingers (5 x 3 per side) attach at the wrist.
(["L", "R"] as const).forEach((s) => {
  (["Index", "Middle", "Ring", "Pinky", "Thumb"] as const).forEach((f) => {
    PARENT[`${s}_${f}1`] = `${s}_Wrist`;
    PARENT[`${s}_${f}2`] = `${s}_${f}1`;
    PARENT[`${s}_${f}3`] = `${s}_${f}2`;
  });
});

type Frame = { frame: number[][]; idx: Record<string, number> };

// A neutral standing pose (arms down) so the "default 3D model" is VISIBLE the
// moment compare mode is enabled — even before any gesture clip has played
// (e.g. the motion gateway is down). SMPL frame: X-left, Y-up, Z-front, metres.
const DEFAULT_REST: Record<string, [number, number, number]> = {
  Pelvis: [0, 0.95, 0], Spine1: [0, 1.05, 0], Spine2: [0, 1.18, 0], Spine3: [0, 1.30, 0],
  Neck: [0, 1.45, 0], Head: [0, 1.62, 0],
  L_Hip: [0.09, 0.90, 0], R_Hip: [-0.09, 0.90, 0],
  L_Knee: [0.10, 0.50, 0.02], R_Knee: [-0.10, 0.50, 0.02],
  L_Ankle: [0.10, 0.08, 0], R_Ankle: [-0.10, 0.08, 0],
  L_Foot: [0.11, 0.03, 0.12], R_Foot: [-0.11, 0.03, 0.12],
  L_Collar: [0.07, 1.42, 0], R_Collar: [-0.07, 1.42, 0],
  L_Shoulder: [0.16, 1.43, 0], R_Shoulder: [-0.16, 1.43, 0],
  L_Elbow: [0.19, 1.16, 0.02], R_Elbow: [-0.19, 1.16, 0.02],
  L_Wrist: [0.20, 0.92, 0.04], R_Wrist: [-0.20, 0.92, 0.04],
};

// Lower-body joints frozen when `lockLegs` is on: talk-in-place, the feet never
// move — only the upper body gestures (matches Miku's locked-leg co-speech).
const LOWER = new Set([
  "Pelvis", "L_Hip", "R_Hip", "L_Knee", "R_Knee", "L_Ankle", "R_Ankle", "L_Foot", "R_Foot",
]);

export class SmplReferenceFigure {
  /** Add this to the scene; move it with `.group.position` (x-offset beside Miku). */
  readonly group = new THREE.Group();

  /** Horizontal offset so the figure stands beside the avatar (metres). */
  offsetX = 0.9;

  private joints = new Map<string, THREE.Mesh>();

  private bones: { mesh: THREE.Mesh; child: string; parent: string }[] = [];

  private sig = ""; // joint-set signature; rebuild the meshes when it changes

  private baselineSet = false; // one-time vertical align (feet ~ y=0)

  private hasLive = false; // a real clip frame has driven the figure at least once

  private restFrame: Frame | null = null; // cached DEFAULT_REST as a {frame, idx}

  /** Freeze the lower body (pelvis + legs + feet) so only the upper body moves. */
  lockLegs = true;

  private lowerRest: Record<string, number[]> | null = null; // captured leg rest positions

  private pelvis0: number[] | null = null; // pelvis rest position (upper body is re-rooted here)

  private label: THREE.Sprite | null = null;

  private readonly jointGeo = new THREE.SphereGeometry(0.024, 12, 10);

  private readonly boneGeo = new THREE.CylinderGeometry(0.015, 0.015, 1, 8);

  private readonly jointMat: THREE.MeshStandardMaterial;

  private readonly boneMat: THREE.MeshStandardMaterial;

  // scratch (no per-frame allocation)
  private _p = new THREE.Vector3();

  private _c = new THREE.Vector3();

  private _dir = new THREE.Vector3();

  private _mid = new THREE.Vector3();

  private _up = new THREE.Vector3(0, 1, 0);

  private _q = new THREE.Quaternion();

  constructor(accent = 0x7fd4ff) {
    this.jointMat = new THREE.MeshStandardMaterial({
      color: 0xffffff, emissive: accent, emissiveIntensity: 0.35, roughness: 0.5, metalness: 0.0,
    });
    this.boneMat = new THREE.MeshStandardMaterial({
      color: accent, emissive: accent, emissiveIntensity: 0.25, roughness: 0.6, metalness: 0.0,
    });
    this.group.name = "smplReferenceFigure";
    this.label = this.makeLabel("参照 (モデル基準)");
    if (this.label) { this.label.position.set(0, 1.92, 0); this.group.add(this.label); }
  }

  /** (Re)build the joint spheres + bone cylinders for the given joint set. */
  private rebuild(idx: Record<string, number>) {
    for (const m of this.joints.values()) { this.group.remove(m); }
    for (const b of this.bones) { this.group.remove(b.mesh); }
    this.joints.clear();
    this.bones = [];
    const names = Object.keys(idx);
    for (const name of names) {
      const mesh = new THREE.Mesh(this.jointGeo, this.jointMat);
      mesh.name = `j_${name}`;
      this.group.add(mesh);
      this.joints.set(name, mesh);
    }
    for (const child of Object.keys(PARENT)) {
      const parent = PARENT[child];
      if (idx[child] === undefined || idx[parent] === undefined) continue;
      const mesh = new THREE.Mesh(this.boneGeo, this.boneMat);
      mesh.name = `b_${parent}_${child}`;
      this.group.add(mesh);
      this.bones.push({ mesh, child, parent });
    }
    this.sig = names.slice().sort().join(",");
    this.baselineSet = false; // re-align vertically for the new body
    this.lowerRest = null; // re-capture the leg rest for the new body
  }

  /** Build the DEFAULT_REST standing pose as a {frame, idx} (cached). */
  private getRestFrame(): Frame {
    if (!this.restFrame) {
      const names = Object.keys(DEFAULT_REST);
      const idx: Record<string, number> = {};
      const frame: number[][] = [];
      names.forEach((n, i) => { idx[n] = i; frame.push(DEFAULT_REST[n].slice()); });
      this.restFrame = { frame, idx };
    }
    return this.restFrame;
  }

  /** Pose the figure from the motion player's current raw SMPL frame. Pass the
   *  value of `vrmMotionPlayer.getCurrentFrame()`. When null: before any real clip
   *  has played, show the DEFAULT_REST standing pose so the reference body is
   *  visible immediately; after a clip has played, hold its last pose. */
  update(data: Frame | null) {
    let eff = data;
    if (!eff) {
      if (this.hasLive) return; // hold the last real pose
      eff = this.getRestFrame(); // never driven yet -> show the default standing body
    } else if (!this.hasLive) {
      this.hasLive = true;
      this.baselineSet = false; // re-align vertically to the real motion
      this.lowerRest = null; // re-capture leg rest from the real clip
    }
    const { frame, idx } = eff;
    const sig = Object.keys(idx).slice().sort().join(",");
    if (sig !== this.sig) this.rebuild(idx);

    // lockLegs: freeze the lower body to its captured rest and re-root the upper
    // body onto the fixed pelvis, so the feet never move (only the upper body
    // gestures). eposFor returns each joint's effective (possibly frozen) position.
    const pelvisI = idx.Pelvis;
    const lock = this.lockLegs && pelvisI !== undefined && !!frame[pelvisI];
    if (lock && !this.lowerRest) {
      this.lowerRest = {};
      LOWER.forEach((n) => { const i = idx[n]; if (i !== undefined && frame[i]) this.lowerRest![n] = frame[i].slice(); });
      this.pelvis0 = frame[pelvisI].slice();
    }
    let ox = 0; let oy = 0; let oz = 0;
    if (lock && this.pelvis0) {
      const pv = frame[pelvisI]; ox = this.pelvis0[0] - pv[0]; oy = this.pelvis0[1] - pv[1]; oz = this.pelvis0[2] - pv[2];
    }
    const eposFor = (name: string, i: number): number[] | null => {
      if (lock && LOWER.has(name)) return (this.lowerRest && this.lowerRest[name]) || frame[i] || null;
      const raw = frame[i];
      if (!raw) return null;
      return lock ? [raw[0] + ox, raw[1] + oy, raw[2] + oz] : raw;
    };

    this.joints.forEach((mesh, name) => {
      const i = idx[name];
      const p = i !== undefined ? eposFor(name, i) : null;
      if (!p) { mesh.visible = false; return; }
      mesh.visible = true;
      mesh.position.set(p[0], p[1], p[2]);
    });

    for (const b of this.bones) {
      const pi = idx[b.parent]; const ci = idx[b.child];
      const pp = pi !== undefined ? eposFor(b.parent, pi) : null;
      const cp = ci !== undefined ? eposFor(b.child, ci) : null;
      if (!pp || !cp) { b.mesh.visible = false; continue; }
      this._p.fromArray(pp); this._c.fromArray(cp);
      this._dir.subVectors(this._c, this._p);
      const len = this._dir.length();
      if (len < 1e-5) { b.mesh.visible = false; continue; }
      b.mesh.visible = true;
      this._mid.addVectors(this._p, this._c).multiplyScalar(0.5);
      b.mesh.position.copy(this._mid);
      this._dir.multiplyScalar(1 / len);
      this._q.setFromUnitVectors(this._up, this._dir);
      b.mesh.quaternion.copy(this._q);
      b.mesh.scale.set(1, len, 1);
    }

    // One-time vertical align: drop the group so the lowest joint (feet) sits on
    // the floor (y=0), matching Miku's planted stance. Motion (bob/weight-shift)
    // is preserved — only a constant offset is applied.
    if (!this.baselineSet) {
      let minY = Infinity;
      this.joints.forEach((m) => { if (m.visible && m.position.y < minY) minY = m.position.y; });
      if (Number.isFinite(minY)) { this.group.position.y = -minY; this.baselineSet = true; }
    }
    this.group.position.x = this.offsetX;
  }

  private makeLabel(text: string): THREE.Sprite | null {
    const canvas = document.createElement("canvas");
    canvas.width = 512; canvas.height = 128;
    const ctx = canvas.getContext("2d");
    if (!ctx) return null;
    ctx.fillStyle = "rgba(10,14,26,0.55)";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.font = "600 60px system-ui, sans-serif";
    ctx.fillStyle = "#bfe6ff";
    ctx.textAlign = "center"; ctx.textBaseline = "middle";
    ctx.fillText(text, canvas.width / 2, canvas.height / 2);
    const tex = new THREE.CanvasTexture(canvas);
    tex.colorSpace = THREE.SRGBColorSpace;
    const mat = new THREE.SpriteMaterial({ map: tex, transparent: true, depthTest: false });
    const sprite = new THREE.Sprite(mat);
    sprite.scale.set(0.5, 0.125, 1);
    return sprite;
  }

  dispose() {
    for (const m of this.joints.values()) this.group.remove(m);
    for (const b of this.bones) this.group.remove(b.mesh);
    this.joints.clear(); this.bones = [];
    if (this.label) {
      this.group.remove(this.label);
      (this.label.material as THREE.SpriteMaterial).map?.dispose();
      (this.label.material as THREE.Material).dispose();
      this.label = null;
    }
    this.jointGeo.dispose();
    this.boneGeo.dispose();
    this.jointMat.dispose();
    this.boneMat.dispose();
  }
}
