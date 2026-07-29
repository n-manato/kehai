/* eslint-disable no-underscore-dangle */
import * as THREE from "three";
import { clampHandsOutsideTorso } from "./torso-clamp";

/**
 * SMPL BODY MESH reference — the avatar the ActionPlan paper renders (render.py →
 * SMPL vertices). Skins the real SMPL body mesh to the live 22-joint SMPL motion
 * so it stands beside Miku and plays the identical clip in perfect sync — the
 * zero-retarget ground truth as a skin surface (not a stick figure).
 *
 * The gateway streams joint POSITIONS only (no SMPL pose params), so per frame we
 * DERIVE each joint's global rotation from the skeleton (swing the rest bone
 * direction onto the current bone direction — twist is dropped, fine for a body
 * silhouette), convert to local, and drive a three.js SkinnedMesh (GPU LBS).
 *
 * Asset (exported from the SMPL model, served under /Live2D%20Characters/smpl/):
 *   smpl_body_meta.json  { nverts, nfaces, njoints, jointNames[], parents[], restJoints[[x,y,z]] }
 *   smpl_body_verts.f32  Float32[nverts*3]   rest (template) vertices
 *   smpl_body_faces.i32  Int32[nfaces*3]
 *   smpl_body_skinidx.u16 Uint16[nverts*4]   top-4 bone indices / vertex
 *   smpl_body_skinw.f32  Float32[nverts*4]   top-4 weights (normalised)
 */

const BASE = "/Live2D%20Characters/smpl";

// Which child joint each joint aims at to derive its global rotation (bone j->child).
// Only used when BOTH endpoints are present in the live frame; leaves inherit parent.
const AIM_CHILD: Record<string, string> = {
  Pelvis: "Spine1", Spine1: "Spine2", Spine2: "Spine3", Spine3: "Neck", Neck: "Head",
  L_Hip: "L_Knee", L_Knee: "L_Ankle", L_Ankle: "L_Foot",
  R_Hip: "R_Knee", R_Knee: "R_Ankle", R_Ankle: "R_Foot",
  L_Collar: "L_Shoulder", L_Shoulder: "L_Elbow", L_Elbow: "L_Wrist",
  R_Collar: "R_Shoulder", R_Shoulder: "R_Elbow", R_Elbow: "R_Wrist",
};
const LEG = new Set(["L_Hip", "R_Hip", "L_Knee", "R_Knee", "L_Ankle", "R_Ankle", "L_Foot", "R_Foot"]);
// The head's true FACING isn't in the 22 joint positions, so orienting it from the
// short neck->head vector over-tilts. Damp the neck+head rotation toward upright a
// little so it reads natural (mild, stateless — no temporal feedback).
const DAMP_BONES = new Set(["Neck", "Head"]);
const DAMP_GAIN = 0.6; // 0 = fully upright, 1 = full raw tilt
// How much the head follows the neck's world orientation (rest is 0 = fully upright/
// level, forward-facing). Low, because head facing isn't in the data — keeps the head
// level like Miku instead of the chin-up artifact.
const HEAD_FOLLOW = 0.2;
// A natural arms-down standing pose (SMPL joint positions) shown when no clip is
// playing — so the idle body isn't a T-pose. The derivation swings the model's
// T-pose rest onto these to bring the arms down.
const DEFAULT_REST: Record<string, [number, number, number]> = {
  Pelvis: [0, 0.95, 0], Spine1: [0, 1.05, 0], Spine2: [0, 1.18, 0], Spine3: [0, 1.30, 0],
  Neck: [0, 1.45, 0], Head: [0, 1.62, 0],
  L_Hip: [0.09, 0.90, 0], R_Hip: [-0.09, 0.90, 0], L_Knee: [0.10, 0.50, 0.02], R_Knee: [-0.10, 0.50, 0.02],
  L_Ankle: [0.10, 0.08, 0], R_Ankle: [-0.10, 0.08, 0], L_Foot: [0.10, 0.03, 0.12], R_Foot: [-0.10, 0.03, 0.12],
  L_Collar: [0.07, 1.42, 0], R_Collar: [-0.07, 1.42, 0], L_Shoulder: [0.16, 1.43, 0], R_Shoulder: [-0.16, 1.43, 0],
  L_Elbow: [0.19, 1.16, 0.02], R_Elbow: [-0.19, 1.16, 0.02], L_Wrist: [0.20, 0.92, 0.04], R_Wrist: [-0.20, 0.92, 0.04],
};

type Frame = { frame: number[][]; idx: Record<string, number> };
type Asset = {
  meta: { nverts: number; nfaces: number; njoints: number; jointNames: string[]; parents: number[]; restJoints: number[][] };
  verts: Float32Array; faces: Uint32Array; skinIdx: Uint16Array; skinW: Float32Array;
};

export class SmplMeshFigure {
  readonly group = new THREE.Group();

  offsetX = 0.9;

  lockLegs = true;

  // Push hands/elbows out of the (thin, raw SMPL) torso. 0 disables. Live: __vrmRef.torsoRadius.
  torsoRadius = 0.14;

  private mesh: THREE.SkinnedMesh | null = null;

  private bones: THREE.Bone[] = [];

  private nameToBone: Record<string, number> = {};

  private asset: Asset | null = null;

  private ready = false;

  private hasLive = false;

  private footYOffset = 0; // group y so the template's lowest vertex (soles) sits on y=0

  private restFrame: Frame | null = null;

  // scratch
  private _restDir = new THREE.Vector3();
  private _curDir = new THREE.Vector3();
  private _up = new THREE.Vector3();
  private _side = new THREE.Vector3();
  private _fwd = new THREE.Vector3();
  private _m3 = new THREE.Matrix4();
  private _Rg: THREE.Quaternion[] = []; // global rotation per joint
  private _q = new THREE.Quaternion();
  private _qi = new THREE.Quaternion();
  private _qA = new THREE.Quaternion(); // scratch (identity target for the neck/head damp)

  constructor(private accent = 0xbfa58f) {
    this.group.name = "smplMeshFigure";
    const label = this.makeLabel("参照 (論文SMPL)");
    if (label) { label.position.set(0, 1.95, 0); this.group.add(label); }
  }

  /** Load the SMPL asset and build the SkinnedMesh. Call once; resolves ready. */
  async load(): Promise<boolean> {
    try {
      const [meta, vb, fb, ib, wb] = await Promise.all([
        fetch(`${BASE}/smpl_body_meta.json`).then((r) => r.json()),
        fetch(`${BASE}/smpl_body_verts.f32`).then((r) => r.arrayBuffer()),
        fetch(`${BASE}/smpl_body_faces.i32`).then((r) => r.arrayBuffer()),
        fetch(`${BASE}/smpl_body_skinidx.u16`).then((r) => r.arrayBuffer()),
        fetch(`${BASE}/smpl_body_skinw.f32`).then((r) => r.arrayBuffer()),
      ]);
      this.asset = {
        meta,
        verts: new Float32Array(vb),
        faces: new Uint32Array(new Int32Array(fb)), // i32 on disk -> u32 index
        skinIdx: new Uint16Array(ib),
        skinW: new Float32Array(wb),
      };
      this.build();
      this.ready = true;
      return true;
    } catch (e) {
      console.error("[SMPL-mesh] asset load failed:", e);
      return false;
    }
  }

  isReady() { return this.ready; }

  private build() {
    const a = this.asset!;
    const { jointNames, parents, restJoints, nverts } = a.meta;
    // lowest template vertex Y -> group offset so the soles rest on the floor (y=0).
    let minY = Infinity;
    for (let i = 1; i < nverts * 3; i += 3) if (a.verts[i] < minY) minY = a.verts[i];
    this.footYOffset = Number.isFinite(minY) ? -minY : 0;

    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(a.verts.slice(0, nverts * 3), 3));
    geo.setIndex(new THREE.BufferAttribute(a.faces, 1));
    geo.setAttribute("skinIndex", new THREE.BufferAttribute(a.skinIdx, 4));
    geo.setAttribute("skinWeight", new THREE.BufferAttribute(a.skinW, 4));
    geo.computeVertexNormals();

    // Build the bone hierarchy at the rest joint positions.
    this.bones = jointNames.map(() => new THREE.Bone());
    this.nameToBone = {};
    this._restLen = new Array(jointNames.length).fill(0);
    jointNames.forEach((nm, j) => {
      this.nameToBone[nm] = j;
      const p = parents[j];
      const rj = restJoints[j];
      if (p < 0) this.bones[j].position.set(rj[0], rj[1], rj[2]);
      else {
        const rp = restJoints[p];
        this.bones[j].position.set(rj[0] - rp[0], rj[1] - rp[1], rj[2] - rp[2]);
        this.bones[p].add(this.bones[j]);
      }
    });
    this._Rg = jointNames.map(() => new THREE.Quaternion());

    // Paper-style neutral clay body, strongly self-lit so it reads clearly against
    // the dark scene regardless of Miku's lighting rig / post-grade.
    const mat = new THREE.MeshStandardMaterial({
      color: 0xd9d0c4, roughness: 0.95, metalness: 0.0,
      emissive: 0x7a7266, emissiveIntensity: 0.6, // FrontSide (closed body) = lighter GPU load
    });
    const mesh = new THREE.SkinnedMesh(geo, mat);
    mesh.add(this.bones[0]);
    mesh.bind(new THREE.Skeleton(this.bones)); // bones at rest => bind pose = template
    mesh.frustumCulled = false;
    this.mesh = mesh;
    this.group.add(mesh);
  }

  /** Arms-down standing {frame, idx} shown before any live clip (not a T-pose). */
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

  /** Derive global rotation for a joint that aims at `childName`, into out. */
  private aimGlobal(name: string, childName: string, frame: number[][], idx: Record<string, number>, out: THREE.Quaternion): boolean {
    const a = this.asset!; const j = this.nameToBone[name]; const c = this.nameToBone[childName];
    const fi = idx[name]; const ci = idx[childName];
    if (fi === undefined || ci === undefined || !frame[fi] || !frame[ci]) return false;
    const rj = a.meta.restJoints[j]; const rc = a.meta.restJoints[c];
    this._restDir.set(rc[0] - rj[0], rc[1] - rj[1], rc[2] - rj[2]);
    this._curDir.set(frame[ci][0] - frame[fi][0], frame[ci][1] - frame[fi][1], frame[ci][2] - frame[fi][2]);
    if (this._restDir.lengthSq() < 1e-9 || this._curDir.lengthSq() < 1e-9) return false;
    this._restDir.normalize(); this._curDir.normalize();
    out.setFromUnitVectors(this._restDir, this._curDir);
    return true;
  }

  /** Root (pelvis) full-frame global rotation from up=Pelvis->Spine1, side=R_Hip->L_Hip. */
  private rootGlobal(frame: number[][], idx: Record<string, number>, out: THREE.Quaternion): boolean {
    const a = this.asset!;
    const nP = idx.Pelvis; const nS = idx.Spine1; const nL = idx.L_Hip; const nR = idx.R_Hip;
    if (nP === undefined || nS === undefined || nL === undefined || nR === undefined) return false;
    const restQ = this._qi; // rest frame
    const curQ = this._q;
    if (!this.frameQuat(a.meta.restJoints[this.nameToBone.Pelvis], a.meta.restJoints[this.nameToBone.Spine1],
      a.meta.restJoints[this.nameToBone.R_Hip], a.meta.restJoints[this.nameToBone.L_Hip], restQ)) return false;
    if (!this.frameQuat(frame[nP], frame[nS], frame[nR], frame[nL], curQ)) return false;
    out.copy(curQ).multiply(restQ.invert());
    return true;
  }

  /** Orthonormal orientation from up(p0->p1) and side(r0->r1) landmark axes. */
  private frameQuat(p0: number[], p1: number[], r0: number[], r1: number[], out: THREE.Quaternion): boolean {
    this._up.set(p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]);
    this._side.set(r1[0] - r0[0], r1[1] - r0[1], r1[2] - r0[2]);
    if (this._up.lengthSq() < 1e-9 || this._side.lengthSq() < 1e-9) return false;
    this._up.normalize();
    this._fwd.crossVectors(this._side, this._up);
    if (this._fwd.lengthSq() < 1e-9) return false;
    this._fwd.normalize();
    this._side.crossVectors(this._up, this._fwd).normalize();
    this._m3.makeBasis(this._side, this._up, this._fwd);
    out.setFromRotationMatrix(this._m3);
    return true;
  }

  update(data: Frame | null) {
    if (!this.ready || !this.asset || !this.mesh) return;
    let eff = data;
    if (!eff) {
      if (this.hasLive) return; // hold last pose
      eff = this.getRestFrame();
    } else if (!this.hasLive) {
      this.hasLive = true; this.baselineSet = false;
    }
    const { idx } = eff;
    // Push hands/elbows out of the (thin, raw SMPL) torso so the gesture doesn't clip.
    const frame = clampHandsOutsideTorso(eff.frame, idx, this.torsoRadius);
    const a = this.asset; const names = a.meta.jointNames; const parents = a.meta.parents;

    // 1) global rotation per joint
    for (let j = 0; j < names.length; j += 1) {
      const nm = names[j];
      if (nm === "Pelvis") {
        if (this.lockLegs || !this.rootGlobal(frame, idx, this._Rg[j])) this._Rg[j].identity();
        continue;
      }
      if (this.lockLegs && LEG.has(nm)) { this._Rg[j].identity(); continue; }
      const child = AIM_CHILD[nm];
      if (child && this.aimGlobal(nm, child, frame, idx, this._q)) this._Rg[j].copy(this._q);
      else this._Rg[j].copy(this._Rg[parents[j] >= 0 ? parents[j] : 0]); // leaf inherits parent
    }
    // 2) local rotation = parentGlobal^-1 * global ; drive the bones. Each result is
    //    validated (finite + normalized) so one degenerate frame can NEVER blow up the
    //    skinned mesh — a bad bone keeps its previous pose instead of going to NaN.
    for (let j = 0; j < names.length; j += 1) {
      const p = parents[j];
      if (p < 0) this._q.copy(this._Rg[j]);
      else this._q.copy(this._qi.copy(this._Rg[p]).invert().multiply(this._Rg[j]));
      // mild stateless damp of the neck/head toward upright (facing is only a guess)
      if (DAMP_BONES.has(names[j])) this._q.slerp(this._qA.identity(), 1 - DAMP_GAIN);
      const n = this._q.x * this._q.x + this._q.y * this._q.y + this._q.z * this._q.z + this._q.w * this._q.w;
      if (!Number.isFinite(n) || n < 1e-8) continue; // reject -> bone holds its last pose
      this._q.normalize(); // renormalize (guards drift/explosion); safe since n >= 1e-8
      this.bones[j].quaternion.copy(this._q);
    }
    // Keep the HEAD facing forward/level. The 22-joint stream has NO head-orientation
    // (no eye joints), so deriving head facing from the neck->head direction made it
    // look UP ~17° (chin-up) while Miku faces level — an artifact, not the model's
    // intent. Override the head to be MOSTLY upright (small neck follow) so it matches
    // Miku's forward-facing head. Torso/neck stay faithful.
    const nb = this.bones[this.nameToBone.Neck];
    const hb = this.bones[this.nameToBone.Head];
    if (nb && hb) {
      this.mesh.updateMatrixWorld(true);
      nb.getWorldQuaternion(this._q); // neck world
      this._qA.identity().slerp(this._q, HEAD_FOLLOW); // target head world: mostly upright
      hb.quaternion.copy(this._qi.copy(this._q).invert().multiply(this._qA)); // -> head local
    }

    // root translation: keep the pelvis at rest (lower body planted when lockLegs).
    const rj = a.meta.restJoints[this.nameToBone.Pelvis];
    this.bones[0].position.set(rj[0], rj[1], rj[2]);

    // Place beside Miku with the soles on the floor (deterministic from the template).
    this.group.position.set(this.offsetX, this.footYOffset, 0);
  }

  private makeLabel(text: string): THREE.Sprite | null {
    const canvas = document.createElement("canvas");
    canvas.width = 512; canvas.height = 128;
    const ctx = canvas.getContext("2d"); if (!ctx) return null;
    ctx.fillStyle = "rgba(10,14,26,0.55)"; ctx.fillRect(0, 0, 512, 128);
    ctx.font = "600 58px system-ui, sans-serif"; ctx.fillStyle = "#e8d9c7";
    ctx.textAlign = "center"; ctx.textBaseline = "middle";
    ctx.fillText(text, 256, 64);
    const tex = new THREE.CanvasTexture(canvas); tex.colorSpace = THREE.SRGBColorSpace;
    const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, transparent: true, depthTest: false }));
    sp.scale.set(0.52, 0.13, 1);
    return sp;
  }

  dispose() {
    if (this.mesh) {
      this.mesh.geometry.dispose();
      (this.mesh.material as THREE.Material).dispose();
      this.group.remove(this.mesh);
      this.mesh = null;
    }
    this.group.clear();
  }
}
