import * as THREE from "three";
import { VRMHumanoid, VRMLookAt, VRMLookAtBoneApplier, VRMLookAtRangeMap } from "@pixiv/three-vrm";
import type { VRM } from "@pixiv/three-vrm";

/**
 * Make a plain rigged .glb usable everywhere a .vrm is.
 *
 * Everything in this app that moves a character - the co-speech retarget, the
 * idle clips, gaze, the pose clamps - talks to `vrm.humanoid` and the VRM
 * humanoid bone NAMES. That is the whole reason VRM was adopted: the names are
 * normative, so motion authored once lands on any VRM without a retarget.
 *
 * A glTF has no such guarantee. What it usually does have, because almost every
 * avatar pipeline in existence descends from the same rig, is the Mixamo
 * skeleton: Hips / Spine / Spine1 / LeftArm / LeftForeArm / LeftUpLeg, with or
 * without a `mixamorig:` prefix. Ready Player Me, Avaturn and Character Creator
 * all ship it. That is a de-facto standard rather than a specified one, but it
 * is consistent enough to map from - and mapping names is a table, not a
 * retarget: no bone is re-parented, no rotation is converted, nothing is
 * resampled.
 *
 * So instead of a second animation path for glb models, this builds a real
 * `VRMHumanoid` over the glTF's own nodes and returns an object shaped like a
 * VRM. `VRMHumanoid` then does the part that actually matters: it derives the
 * NORMALIZED rig, whose rest pose is identity, so an A-pose export and a T-pose
 * export both receive the same motion correctly.
 *
 * What such a model does NOT get, because the file cannot describe them: spring
 * bones (hair/cloth physics), VRM expression presets, first-person culling, and
 * the meta/licence block. Faces are handled separately by name - see the morph
 * mapping in vrm-expression.
 */

/** VRM humanoid bone name -> candidate node names, best first. */
const BONE_MAP: Record<string, string[]> = {
  hips: ["Hips"],
  spine: ["Spine"],
  chest: ["Spine1"],
  upperChest: ["Spine2"],
  neck: ["Neck"],
  head: ["Head"],
  // c_eye.* is Auto-Rig Pro's eyeball deform bone (female_android).
  leftEye: ["LeftEye", "c_eye.l"],
  rightEye: ["RightEye", "c_eye.r"],

  leftShoulder: ["LeftShoulder"],
  leftUpperArm: ["LeftArm"],
  leftLowerArm: ["LeftForeArm"],
  leftHand: ["LeftHand"],
  rightShoulder: ["RightShoulder"],
  rightUpperArm: ["RightArm"],
  rightLowerArm: ["RightForeArm"],
  rightHand: ["RightHand"],

  leftUpperLeg: ["LeftUpLeg"],
  leftLowerLeg: ["LeftLeg"],
  leftFoot: ["LeftFoot"],
  leftToes: ["LeftToeBase"],
  rightUpperLeg: ["RightUpLeg"],
  rightLowerLeg: ["RightLeg"],
  rightFoot: ["RightFoot"],
  rightToes: ["RightToeBase"],

  // Fingers. Mixamo numbers them 1/2/3 from the palm out; VRM names the thumb's
  // first joint Metacarpal and the other fingers' first joint Proximal, so the
  // thumb is offset by one against the rest. Getting that wrong bends the thumb
  // at the wrong knuckle, which is subtle in a still and obvious in motion.
  leftThumbMetacarpal: ["LeftHandThumb1"],
  leftThumbProximal: ["LeftHandThumb2"],
  leftThumbDistal: ["LeftHandThumb3"],
  leftIndexProximal: ["LeftHandIndex1"],
  leftIndexIntermediate: ["LeftHandIndex2"],
  leftIndexDistal: ["LeftHandIndex3"],
  leftMiddleProximal: ["LeftHandMiddle1"],
  leftMiddleIntermediate: ["LeftHandMiddle2"],
  leftMiddleDistal: ["LeftHandMiddle3"],
  leftRingProximal: ["LeftHandRing1"],
  leftRingIntermediate: ["LeftHandRing2"],
  leftRingDistal: ["LeftHandRing3"],
  leftLittleProximal: ["LeftHandPinky1"],
  leftLittleIntermediate: ["LeftHandPinky2"],
  leftLittleDistal: ["LeftHandPinky3"],
  rightThumbMetacarpal: ["RightHandThumb1"],
  rightThumbProximal: ["RightHandThumb2"],
  rightThumbDistal: ["RightHandThumb3"],
  rightIndexProximal: ["RightHandIndex1"],
  rightIndexIntermediate: ["RightHandIndex2"],
  rightIndexDistal: ["RightHandIndex3"],
  rightMiddleProximal: ["RightHandMiddle1"],
  rightMiddleIntermediate: ["RightHandMiddle2"],
  rightMiddleDistal: ["RightHandMiddle3"],
  rightRingProximal: ["RightHandRing1"],
  rightRingIntermediate: ["RightHandRing2"],
  rightRingDistal: ["RightHandRing3"],
  rightLittleProximal: ["RightHandPinky1"],
  rightLittleIntermediate: ["RightHandPinky2"],
  rightLittleDistal: ["RightHandPinky3"],
};

/** VRM refuses to build without these; everything else is optional. */
const REQUIRED = [
  "hips", "spine", "head",
  "leftUpperArm", "leftLowerArm", "leftHand",
  "rightUpperArm", "rightLowerArm", "rightHand",
  "leftUpperLeg", "leftLowerLeg", "leftFoot",
  "rightUpperLeg", "rightLowerLeg", "rightFoot",
];

/** `mixamorig:LeftArm` / `mixamorig1:LeftArm` / `Armature|LeftArm` -> `leftarm`. */
function normalizeName(raw: string): string {
  return raw
    .replace(/^mixamorig\d*:/i, "")
    .replace(/^.*\|/, "")
    .replace(/[\s_.-]/g, "")
    .toLowerCase();
}

export interface GlbHumanoidResult {
  vrm: VRM;
  /** Bones that could not be found — empty when the rig mapped cleanly. */
  missing: string[];
  /** Every skinned bone name in the file, for diagnosing a rig that did not map. */
  boneNames: string[];
}

/**
 * Build a VRM-shaped object over a plain glTF. Returns null when the file has no
 * recognisable humanoid, which is the honest answer for a static prop or a rig
 * from some other lineage — better than a half-mapped skeleton that moves wrongly.
 */
export function buildVrmFromGltf(gltf: { scene: THREE.Object3D }): GlbHumanoidResult | null {
  const byName = new Map<string, THREE.Object3D>();
  const boneNames: string[] = [];
  gltf.scene.traverse((o: THREE.Object3D) => {
    if (!o.name) return;
    boneNames.push(o.name);
    const k = normalizeName(o.name);
    // First writer wins: a duplicate name deeper in the tree is a helper/twist
    // bone, and the shallower one is the joint the rig is actually built on.
    if (!byName.has(k)) byName.set(k, o);
  });

  const humanBones: Record<string, { node: THREE.Object3D }> = {};
  const missing: string[] = [];
  for (const [vrmName, candidates] of Object.entries(BONE_MAP)) {
    let node: THREE.Object3D | undefined;
    for (const c of candidates) {
      node = byName.get(normalizeName(c));
      if (node) break;
    }
    if (node) humanBones[vrmName] = { node };
    else missing.push(vrmName);
  }

  const missingRequired = REQUIRED.filter((b) => !humanBones[b]);
  if (missingRequired.length) {
    console.warn(
      "[glb-humanoid] not a recognisable humanoid — missing required bones:",
      missingRequired.join(", "),
      "\n  bones found in file:", boneNames.slice(0, 40),
    );
    return null;
  }

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const humanoid = new VRMHumanoid(humanBones as any);

  // Eyes, when the rig has them. The applier captures the eyes' rest pose here,
  // while the file is still in its bind pose. Its ranges are placeholders: the
  // viewer's setupGaze installs the tuned ones for every model it drives.
  let lookAt: VRMLookAt | undefined;
  if (humanBones.leftEye && humanBones.rightEye) {
    gltf.scene.updateMatrixWorld(true);
    const unit = () => new VRMLookAtRangeMap(90, 90);
    lookAt = new VRMLookAt(humanoid, new VRMLookAtBoneApplier(humanoid, unit(), unit(), unit(), unit()));
    // Aim from between the eyes, not from the head joint at the top of the neck:
    // ~8cm lower, which tilts the gaze a few degrees off a viewer at 1.5m.
    const head = humanBones.head.node;
    const l = humanBones.leftEye.node.getWorldPosition(new THREE.Vector3());
    const r = humanBones.rightEye.node.getWorldPosition(new THREE.Vector3());
    lookAt.offsetFromHeadBone.copy(head.worldToLocal(l.add(r).multiplyScalar(0.5)));
  }

  // Shaped like a VRM, carrying only what this app actually reads from one:
  // scene, humanoid, lookAt, meta. springBoneManager / expressionManager stay
  // undefined and every call site already guards them.
  const vrm = {
    scene: gltf.scene,
    humanoid,
    lookAt,
    meta: { name: "glb-humanoid", metaVersion: "1" },
    update(delta: number) {
      humanoid.update();
      lookAt?.update(delta);
    },
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
  } as any as VRM;

  return { vrm, missing, boneNames };
}
