import * as THREE from "three";
import type { VRM } from "@pixiv/three-vrm";
import { SPACE } from "./space-config";

/**
 * Phase 4 — turn her head and upper body toward the viewer.
 *
 * three-vrm's `VRMLookAtBoneApplier` only rotates the EYE bones. Measured on the
 * real rig: moving the viewpoint 0.9 m sideways swung the eyes 4.75 degrees and
 * the head 0.01 — i.e. not at all. So "eyes strongly, head moderately, upper
 * body weakly" needs this layer; the eyes half is already handled and this adds
 * the rest.
 *
 * Composing with the gesture system is the whole difficulty. The motion player
 * poses the NORMALIZED humanoid and then pushes it to the raw bones with
 * `humanoid.update()`, so this has to (a) write normalized bones too and (b) run
 * after the motion player and before `vrm.update()`. It is additive: it reads
 * whatever pose the gesture produced and rotates on top of it, so a gesture that
 * already turns her head simply needs less correction rather than being fought.
 */

/**
 * How the turn is shared along the spine. Sums to 1, so the total is set by
 * `SPACE.lookAt` alone and the weights only decide the shape of the curve —
 * mostly head and neck, a little chest, so it reads as attention rather than as
 * the whole torso swivelling.
 */
const CHAIN: [string, number][] = [
  ["spine", 0.06],
  ["chest", 0.10],
  ["upperChest", 0.12],
  ["neck", 0.30],
  ["head", 0.42],
];

const _headPos = new THREE.Vector3();
const _headQ = new THREE.Quaternion();
const _bodyQ = new THREE.Quaternion();
const _fwd = new THREE.Vector3();
const _want = new THREE.Vector3();
const _local = new THREE.Vector3();
const _parentQ = new THREE.Quaternion();
const _tmpQ = new THREE.Quaternion();
const _part = new THREE.Quaternion();
const _delta = new THREE.Quaternion();
const _apply = new THREE.Quaternion();

/** Smoothed rotation currently being applied, so a state change eases in. */
const _current = new THREE.Quaternion();

export function resetViewerLookAt() {
  _current.identity();
}

/**
 * Rotate the spine chain so she looks toward `targetWorld`.
 * Call once per frame, after the motion player and before `vrm.update()`.
 *
 * `weight` (0..1) scales the turn on top of `SPACE.lookAt` — the camera-follow
 * hand-over, so she turns to a person only while one is tracked and eases back
 * to facing front when they leave, instead of snapping.
 */
export function applyViewerLookAt(vrm: VRM, targetWorld: THREE.Vector3, dt: number, weight = 1) {
  if (SPACE.lookAt <= 0.001 || !vrm.humanoid) return;

  const head = vrm.humanoid.getNormalizedBoneNode("head" as never) as THREE.Object3D | null;
  const hips = vrm.humanoid.getNormalizedBoneNode("hips" as never) as THREE.Object3D | null;
  if (!head || !hips) return;

  // Bones present on THIS rig, with the weights renormalised over them. Not
  // every VRM has an upperChest — this model doesn't — and without this the
  // missing share would silently shrink the whole movement.
  const chain: { node: THREE.Object3D; weight: number }[] = [];
  let total = 0;
  CHAIN.forEach(([name, w]) => {
    const node = vrm.humanoid.getNormalizedBoneNode(name as never) as THREE.Object3D | null;
    if (node) {
      chain.push({ node, weight: w });
      total += w;
    }
  });
  if (!chain.length || total <= 0) return;
  chain.forEach((c) => { c.weight /= total; });

  head.getWorldPosition(_headPos);
  head.getWorldQuaternion(_headQ);
  hips.getWorldQuaternion(_bodyQ);

  // Which way "facing" points in this rig. three-vrm knows, because its own
  // look-at applier needs the same answer; falling back to +Z matches the VRM
  // normalized-space convention.
  const faceFront = (vrm.lookAt?.faceFront as THREE.Vector3 | undefined) ?? undefined;
  _fwd.copy(faceFront ?? new THREE.Vector3(0, 0, 1)).applyQuaternion(_headQ).normalize();

  _want.subVectors(targetWorld, _headPos);
  if (_want.lengthSq() < 1e-6) return;
  _want.normalize();

  // Clamp in the BODY frame, not the head's: limits have to mean "relative to
  // where her torso is facing", or a gesture that already turned her would let
  // the clamp allow another full turn on top.
  _local.copy(_want).applyQuaternion(_tmpQ.copy(_bodyQ).invert());
  const front = faceFront ?? new THREE.Vector3(0, 0, 1);
  // Express the direction as yaw/pitch about the body's own axes.
  const zSign = front.z >= 0 ? 1 : -1;
  let yaw = Math.atan2(_local.x * zSign, _local.z * zSign);
  let pitch = Math.asin(THREE.MathUtils.clamp(_local.y, -1, 1));

  const maxYaw = THREE.MathUtils.degToRad(SPACE.lookAtMaxYaw);
  const maxPitch = THREE.MathUtils.degToRad(SPACE.lookAtMaxPitch);
  yaw = THREE.MathUtils.clamp(yaw, -maxYaw, maxYaw);
  pitch = THREE.MathUtils.clamp(pitch, -maxPitch, maxPitch);

  // Rebuild the clamped direction and take it back to world space.
  const cp = Math.cos(pitch);
  _local.set(Math.sin(yaw) * cp * zSign, Math.sin(pitch), Math.cos(yaw) * cp * zSign);
  _local.applyQuaternion(_bodyQ).normalize();

  // Rotation from where the head points now to where we want it, scaled by the
  // master strength: the eyes are meant to carry most of the tracking, the head
  // only to agree with them.
  _delta.setFromUnitVectors(_fwd, _local);
  _apply.identity().slerp(_delta, THREE.MathUtils.clamp(SPACE.lookAt * weight, 0, 1));

  // Ease so a jump in the tracked position (a re-lock after losing the face)
  // doesn't snap her neck.
  const k = 1 - Math.exp(-SPACE.lookAtSpeed * dt);
  _current.slerp(_apply, k);

  // Distribute down the chain. World matrices are refreshed per bone via
  // getWorldQuaternion(), which updates ancestors — so each bone sees the
  // rotation its parents just received.
  for (let i = 0; i < chain.length; i += 1) {
    const { node, weight } = chain[i];
    const parent = node.parent;
    if (!parent) continue;
    parent.getWorldQuaternion(_parentQ);
    _part.identity().slerp(_current, weight);
    // Convert the world-space delta into this bone's parent space:
    //   newLocal = inv(parentWorld) * delta * parentWorld * local
    _tmpQ.copy(_parentQ).invert().multiply(_part).multiply(_parentQ);
    node.quaternion.premultiply(_tmpQ);
  }
}
