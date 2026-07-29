/* Real MMD hair physics for the converted VRM (Phase 2).
 *
 * The Miku VRM was converted from a PMX whose twintails are FOUR Bullet rigid-body
 * chains per side (HZ heavy spine + HA/HB/HC light strands) constrained by 6-DOF
 * spring joints and colliding with kinematic body capsules. The VRM export dropped
 * ALL of it; springbone approximations can't reproduce the look (no joint limits ->
 * "タコ踊り", no collision, no inertia). This module rebuilds the original setup with
 * ammo.wasm (the same Bullet engine MMD uses, already shipped for the MMD viewer)
 * from `public/physics/miku-hair-physics.json` (baked straight out of the PMX).
 *
 * The sim runs in PMX units/space (like MMD itself — Bullet dynamics don't scale
 * with the scene), converting bone transforms per frame through the fitted map
 *   vrm = A · pmx,  A = diag(s, s, ±s)  (s = 0.08, Z flipped for this model).
 * Per bone the constant  X = bindWorld_vrm⁻¹ · A · translate(bonePos_pmx)  makes
 *   boneForm_pmx(t) = A⁻¹ · boneWorld_vrm(t) · X
 * equal translate(bonePos_pmx) exactly at bind (PMX binds have identity rotation).
 * Kinematic bodies are driven from bones; after stepping, physics bodies write
 * back to the bones (MMDPhysics-style offset forms). Per-frame math stays in
 * three.js land — ammo.js value-returning calls (op_mul/getRotation) allocate WASM
 * heap objects and would leak in a 60fps loop.
 */
import * as THREE from "three";
import type { VRM } from "@pixiv/three-vrm";

let ammoPromise: Promise<any> | null = null;

function fetchWasmBinary(url: string): Promise<ArrayBuffer | undefined> {
  return fetch(url)
    .then((resp) => (resp.ok ? resp.arrayBuffer() : undefined))
    .catch(() => undefined);
}

function loadAmmo(): Promise<any> {
  const w = window as any;
  if (w.Ammo && typeof w.Ammo.btVector3 === "function") return Promise.resolve(w.Ammo);
  if (!ammoPromise) {
    const dirUrl = new URL("libs/ammo/", document.baseURI).href;
    ammoPromise = new Promise<any>((resolvePromise, reject) => {
      const script = document.createElement("script");
      script.src = `${dirUrl}ammo.wasm.js`;
      script.onload = async () => {
        const factory = typeof w.Ammo === "function" ? w.Ammo : null;
        if (!factory) { reject(new Error("ammo.wasm.js loaded but no Ammo() factory")); return; }
        const wasmBinary = await fetchWasmBinary(`${dirUrl}ammo.wasm.wasm`);
        factory({ locateFile: (f: string) => `${dirUrl}${f}`, ...(wasmBinary ? { wasmBinary } : {}) })
          .then((AmmoLib: any) => { w.Ammo = AmmoLib; resolvePromise(AmmoLib); }, reject);
      };
      script.onerror = () => reject(new Error(`failed to fetch ${dirUrl}ammo.wasm.js`));
      document.head.appendChild(script);
    }).catch((e) => { ammoPromise = null; throw e; });
  }
  return ammoPromise;
}

interface SimBody {
  body: any; // btRigidBody
  bone: THREE.Object3D;
  mode: number; // 0 kinematic (bone->body), 1 physics (body->bone), 2 physics rotation-only
  X: THREE.Matrix4; // boneForm_pmx = Ainv * boneWorld_vrm * X
  Xinv: THREE.Matrix4;
  Om: THREE.Matrix4; // bone -> body (bind, PMX space)
  Oinvm: THREE.Matrix4;
}

/** btTransform -> three Matrix4 without allocating on the WASM heap. */
function btToMatrix(Ammo: any, tf: any, qScratch: THREE.Quaternion, out: THREE.Matrix4): THREE.Matrix4 {
  const q = tf.getRotation(); // allocates — destroyed below
  qScratch.set(q.x(), q.y(), q.z(), q.w());
  Ammo.destroy(q);
  const o = tf.getOrigin(); // reference — no alloc
  out.makeRotationFromQuaternion(qScratch);
  out.setPosition(o.x(), o.y(), o.z());
  return out;
}

export class MMDHairPhysics {
  private Ammo: any = null;

  private world: any = null;

  private bodies: SimBody[] = [];

  private constraints: any[] = [];

  private A = new THREE.Matrix4();

  private Ainv = new THREE.Matrix4();

  // scratch (JS heap only in the hot path)
  private _m0 = new THREE.Matrix4(); private _m1 = new THREE.Matrix4(); private _m2 = new THREE.Matrix4();

  private _p = new THREE.Vector3(); private _q = new THREE.Quaternion(); private _s = new THREE.Vector3();

  private _tf: any = null; private _v3: any = null; private _q4: any = null;

  enabled = false;

  /** Bone names now owned by Bullet — their springbone joints must be removed. */
  hairBoneNames = new Set<string>();

  async init(vrm: VRM): Promise<boolean> {
    try {
      const resp = await fetch(new URL("physics/miku-hair-physics.json", document.baseURI).href);
      if (!resp.ok) return false;
      const data = await resp.json();
      const Ammo = await loadAmmo();
      this.Ammo = Ammo;
      // ---- world (MMD defaults: gravity -98 pmx-units/s^2, fixed 1/65s steps) --
      const config = new Ammo.btDefaultCollisionConfiguration();
      const dispatcher = new Ammo.btCollisionDispatcher(config);
      const broadphase = new Ammo.btDbvtBroadphase();
      const solver = new Ammo.btSequentialImpulseConstraintSolver();
      this.world = new Ammo.btDiscreteDynamicsWorld(dispatcher, broadphase, solver, config);
      this.world.setGravity(new Ammo.btVector3(data.gravity[0], data.gravity[1], data.gravity[2]));
      this._tf = new Ammo.btTransform();
      this._v3 = new Ammo.btVector3(0, 0, 0);
      this._q4 = new Ammo.btQuaternion(0, 0, 0, 1);
      const s = data.space.scale;
      this.A.makeScale(data.space.flipX ? -s : s, s, data.space.flipZ ? -s : s);
      this.Ainv.copy(this.A).invert();

      // ---- bind pose from the skeleton (current pose may already be idle-swayed)
      const bind = new Map<string, THREE.Matrix4>();
      vrm.scene.traverse((o: any) => {
        if (!o.isSkinnedMesh || !o.skeleton) return;
        const sk = o.skeleton;
        for (let i = 0; i < sk.bones.length; i += 1) {
          const nm = sk.bones[i].name;
          if (!bind.has(nm)) bind.set(nm, new THREE.Matrix4().copy(sk.boneInverses[i]).invert());
        }
      });
      vrm.scene.updateMatrixWorld(true);

      const rbFormsM: (THREE.Matrix4 | null)[] = []; // bind body forms (three, PMX space)
      const euler = new THREE.Euler(); // PMX euler = THREE 'XYZ' order (matches MMDPhysics.js
      const one = new THREE.Vector3(1, 1, 1); // — Bullet's setEulerZYX composes DIFFERENTLY and
      // crumpled every compound-angle body: the back/side skirt panels collapsed)
      for (const b of data.bodies) {
        const bone = b.vrmHumanoid
          ? (vrm.humanoid as any)?.getRawBoneNode?.(b.vrmHumanoid)
          : vrm.scene.getObjectByName(b.vrmBone);
        // bind from the skeleton by the RESOLVED node's name (humanoid bones are renamed);
        // fall back to the current matrix (scene just updated) if it's not skinned.
        const bw = bone ? (bind.get(bone.name) ?? bone.matrixWorld.clone()) : null;
        if (!bone || !bw) { rbFormsM.push(null); this.bodies.push(null as any); continue; }
        // rbFormBind (PMX): translate(pos) * euler('XYZ') — exactly MMDPhysics.js semantics
        const rbM = new THREE.Matrix4().compose(
          this._p.set(b.pos[0], b.pos[1], b.pos[2]),
          this._q.setFromEuler(euler.set(b.rot[0], b.rot[1], b.rot[2])),
          one,
        );
        rbFormsM.push(rbM);
        const X = new THREE.Matrix4().copy(bw).invert().multiply(this.A)
          .multiply(this._m0.makeTranslation(b.bonePos[0], b.bonePos[1], b.bonePos[2]));
        const Om = this._m0.makeTranslation(-b.bonePos[0], -b.bonePos[1], -b.bonePos[2]).clone().multiply(rbM);
        // ---- shape + body ------------------------------------------------------
        let shape: any;
        if (b.shape === "sphere") shape = new Ammo.btSphereShape(b.size[0]);
        else if (b.shape === "box") shape = new Ammo.btBoxShape(new Ammo.btVector3(b.size[0], b.size[1], b.size[2]));
        else shape = new Ammo.btCapsuleShape(b.size[0], b.size[1]);
        const kinematic = b.mode === 0;
        const mass = kinematic ? 0 : b.mass;
        const inertia = new Ammo.btVector3(0, 0, 0);
        if (mass > 0) shape.calculateLocalInertia(mass, inertia);
        const motion = new Ammo.btDefaultMotionState(this.matrixToNewBt(rbM));
        const info = new Ammo.btRigidBodyConstructionInfo(mass, motion, shape, inertia);
        const body = new Ammo.btRigidBody(info);
        body.setDamping(b.linDamp, b.angDamp);
        body.setFriction(b.friction);
        body.setRestitution(b.restitution);
        body.setActivationState(4); // DISABLE_DEACTIVATION
        body.setSleepingThresholds(0, 0);
        if (kinematic) body.setCollisionFlags(body.getCollisionFlags() | 2); // CF_KINEMATIC_OBJECT
        else {
          // Continuous collision detection: the skirt panels are THIN boxes — during fast
          // gesture moves they tunnel straight through the leg capsules in one discrete
          // step and the thigh shows through the skirt ("スカートの中が貫通して見える").
          // CCD sweeps a sphere between steps so thin-fast bodies can't skip contacts.
          const minDim = Math.max(0.02, Math.min(...b.size.filter((x: number) => x > 0)));
          body.setCcdMotionThreshold(0.01);
          body.setCcdSweptSphereRadius(minDim * 0.8);
        }
        // NOTE ON THE MASKS — two failed experiments, do not repeat. The bake stores the
        // PMX 非衝突グループフラグ verbatim (CLEAR bit = collides; the necktie's 0xBF7E
        // clears exactly body/self/wrists), while Bullet's filter is SET bit = collides —
        // so the raw value inverts the author's collision design. Restoring it was tried:
        //  * inverting ALL masks — the twintail chains spawn interpenetrated and their new
        //    self-collision blows them into zigzag spirals within seconds;
        //  * inverting only the skirt (group 9: self-collision ON, legs OFF, per the PMX) —
        //    the resting pleats push each other apart, visibly changing the skirt's shape
        //    and shading; rejected on sight.
        // The accidental behaviour is what this VRM has been tuned around (legs support
        // the cloth, panels layer without fighting), so the raw mask is kept deliberately.
        this.world.addRigidBody(body, 1 << b.group, b.mask);
        this.bodies.push({
          body, bone, mode: b.mode, X, Xinv: new THREE.Matrix4().copy(X).invert(),
          Om, Oinvm: new THREE.Matrix4().copy(Om).invert(),
        });
        if (b.hair && b.mode !== 0) this.hairBoneNames.add(bone.name);
      }

      // ---- constraints (btGeneric6DofSpringConstraint, MMDPhysics semantics) ---
      for (const j of data.joints) {
        const bA = this.bodies[j.a]; const bB = this.bodies[j.b];
        const fa = rbFormsM[j.a]; const fb = rbFormsM[j.b];
        if (!bA || !bB || !fa || !fb) continue;
        const jM = this._m1.compose(
          this._p.set(j.pos[0], j.pos[1], j.pos[2]),
          this._q.setFromEuler(euler.set(j.rot[0], j.rot[1], j.rot[2])),
          one,
        );
        const frameA = this.matrixToNewBt(this._m0.copy(fa).invert().multiply(jM));
        const frameB = this.matrixToNewBt(this._m2.copy(fb).invert().multiply(jM));
        const c = new Ammo.btGeneric6DofSpringConstraint(bA.body, bB.body, frameA, frameB, true);
        c.setLinearLowerLimit(new Ammo.btVector3(j.posMin[0], j.posMin[1], j.posMin[2]));
        c.setLinearUpperLimit(new Ammo.btVector3(j.posMax[0], j.posMax[1], j.posMax[2]));
        c.setAngularLowerLimit(new Ammo.btVector3(j.rotMin[0], j.rotMin[1], j.rotMin[2]));
        c.setAngularUpperLimit(new Ammo.btVector3(j.rotMax[0], j.rotMax[1], j.rotMax[2]));
        for (let i = 0; i < 3; i += 1) {
          if (j.springPos[i] !== 0) { c.enableSpring(i, true); c.setStiffness(i, j.springPos[i]); }
          if (j.springRot[i] !== 0) { c.enableSpring(i + 3, true); c.setStiffness(i + 3, j.springRot[i]); }
        }
        this.world.addConstraint(c, true);
        this.constraints.push(c);
      }
      this.enabled = true;
      console.info(`[vrm] MMD hair physics ON: ${this.bodies.length} bodies, ${this.constraints.length} joints (Bullet/ammo)`);
      return true;
    } catch (e) {
      console.error("[vrm] MMD hair physics init failed — springbone fallback stays:", e);
      this.enabled = false;
      return false;
    }
  }

  private matrixToNewBt(m: THREE.Matrix4): any {
    const Ammo = this.Ammo;
    m.decompose(this._p, this._q, this._s);
    const tf = new Ammo.btTransform();
    tf.setIdentity();
    tf.setOrigin(new Ammo.btVector3(this._p.x, this._p.y, this._p.z));
    tf.setRotation(new Ammo.btQuaternion(this._q.x, this._q.y, this._q.z, this._q.w));
    return tf;
  }

  /** Set the reusable scratch btTransform from a three matrix (no WASM allocs). */
  private setScratchBt(m: THREE.Matrix4) {
    m.decompose(this._p, this._q, this._s);
    this._v3.setValue(this._p.x, this._p.y, this._p.z);
    this._q4.setValue(this._q.x, this._q.y, this._q.z, this._q.w);
    this._tf.setIdentity();
    this._tf.setOrigin(this._v3);
    this._tf.setRotation(this._q4);
  }

  /** Advance the sim and write physics bodies back to the VRM bones. */
  step(dt: number) {
    if (!this.enabled || !this.world) return;
    // 1) kinematic bodies follow their bones (body colliders + chain anchors)
    for (const sb of this.bodies) {
      if (!sb || sb.mode !== 0) continue;
      this._m1.copy(this.Ainv).multiply(sb.bone.matrixWorld).multiply(sb.X).multiply(sb.Om);
      this.setScratchBt(this._m1);
      sb.body.getMotionState().setWorldTransform(this._tf);
      sb.body.setCenterOfMassTransform(this._tf);
    }
    // 2) step at 120Hz fixed substeps — raising the physics rate is THE standard MMD fix
    // for skirt penetration (contacts get resolved before the leg moves through a panel).
    this.world.stepSimulation(Math.min(dt, 0.1), 8, 1 / 120);
    // 3) physics bodies drive their bones (root -> tip order = baked order)
    for (const sb of this.bodies) {
      if (!sb || sb.mode === 0) continue;
      sb.body.getMotionState().getWorldTransform(this._tf);
      btToMatrix(this.Ammo, this._tf, this._q, this._m1);
      this._m1.multiply(sb.Oinvm); // boneForm_pmx = bodyForm * O^-1
      // target world (VRM) = A * boneForm_pmx * X^-1
      this._m2.copy(this.A).multiply(this._m1).multiply(sb.Xinv);
      const parent = sb.bone.parent as THREE.Object3D;
      this._m0.copy(parent.matrixWorld).invert().multiply(this._m2); // -> local
      this._m0.decompose(this._p, this._q, this._s);
      sb.bone.quaternion.copy(this._q);
      if (sb.mode === 1) sb.bone.position.copy(this._p);
      // refresh matrices manually (chain order) so children convert against fresh parents
      sb.bone.updateMatrix();
      sb.bone.matrixWorld.multiplyMatrices(parent.matrixWorld, sb.bone.matrix);
    }
  }

  dispose() {
    if (!this.Ammo || !this.world) return;
    for (const c of this.constraints) this.world.removeConstraint(c);
    for (const sb of this.bodies) if (sb) this.world.removeRigidBody(sb.body);
    this.constraints = [];
    this.bodies = [];
    this.enabled = false;
  }
}

export const mmdHairPhysics = new MMDHairPhysics();
