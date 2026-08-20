#!/usr/bin/env node
/**
 * FBX/GLB (Mixamo-rigged humanoid) -> SMPL position clip JSON, the format
 * `motion-lib-catalog/` holds and `/motion/idle` serves.
 *
 * This replaces the lost `scratchpad/fbx_to_smpl_clip.py` referenced from
 * motion_lib.py. It samples BONE WORLD POSITIONS off three.js' own
 * AnimationMixer rather than re-deriving FK, so what lands in the JSON is
 * exactly what the runtime's retarget would have seen had it played the source
 * clip directly.
 *
 * OUTPUT CONVENTIONS — all four measured off the existing KAWAII clips, not
 * guessed. Break any of them and the clip plays but sits wrong:
 *   - frame: X-left, Y-up, Z-front, METRES. Same frame the VRM retarget uses
 *     (vrm-motion-player.ts:7-11), so no axis conversion happens at playback.
 *   - floor: min Y over the WHOLE clip is exactly 0.
 *   - origin: the ankle midpoint of FRAME 0 sits at x=0, z=0. (Not the pelvis —
 *     the pelvis lands at z=+0.019 in every shipped clip because of it.)
 *   - 64 joints, KAWAII layout: 22 body, 40 finger (incl. the `*4` fingertips
 *     the runtime needs for the distal knuckles), 2 eyes.
 *
 * THE EYES ARE NOT DECORATION. vrm-motion-player.ts:152 builds the HEAD's
 * orientation basis from the R_Eye->L_Eye vector, so a clip without them gets a
 * head that cannot roll or yaw. Mixamo rigs have no eye bones, so they are
 * synthesised from the head bone's rotation relative to its own rest pose —
 * exact, and it reproduces the shipped clips' ±0.045 m eye offset at rest.
 *
 * Usage:
 *   node scripts/fbx_to_smpl_clip.mjs <in.fbx|in.glb> [...] [options]
 *     --out DIR      output directory (default: motion-lib-catalog/)
 *     --name KEY     output key; only valid with a single input
 *     --prefix STR   prepended to every auto-derived key (e.g. --prefix chan_)
 *     --fps N        resample rate (default 30)
 *     --scale N      source units per metre; default auto-detects cm vs m
 *     --keep-last    keep the final frame (default drops it, so the clip loops)
 *     --dry          report only, write nothing
 */
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const REPO = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const NM = path.join(REPO, "frontend-src", "node_modules");

// three and its loaders are written for a browser. We never render or decode a
// pixel here — only the skeleton and the animation tracks are read — so the
// browser globals they touch on the way in are stubbed rather than emulated.
// Textures resolve to empty objects and are simply never looked at.
globalThis.self ??= globalThis;
globalThis.window ??= globalThis;
globalThis.createImageBitmap ??= async () => ({ width: 1, height: 1, close() {} });
globalThis.document ??= {
  createElementNS: () => ({ getContext: () => null, style: {}, setAttribute() {} }),
  createElement: () => ({ getContext: () => null, style: {}, setAttribute() {} }),
};
globalThis.URL.createObjectURL ??= () => "blob:stub";
globalThis.URL.revokeObjectURL ??= () => {};

// three is only installed under frontend-src/. Import it by absolute path; the
// loaders' own bare `three` specifier resolves to this same build/three.module.js
// (three's exports map points "." there), so both share one module instance.
const THREE = await import(pathToFileURL(path.join(NM, "three/build/three.module.js")).href);
const { FBXLoader } = await import(
  pathToFileURL(path.join(NM, "three/examples/jsm/loaders/FBXLoader.js")).href
);
const { GLTFLoader } = await import(
  pathToFileURL(path.join(NM, "three/examples/jsm/loaders/GLTFLoader.js")).href
);

/** The 64 joints, in the order the runtime expects to find them. */
const SMPL_JOINTS = [
  "Pelvis", "L_Hip", "R_Hip", "Spine1", "L_Knee", "R_Knee", "Spine2",
  "L_Ankle", "R_Ankle", "Spine3", "L_Foot", "R_Foot", "Neck", "L_Collar",
  "R_Collar", "Head", "L_Shoulder", "R_Shoulder", "L_Elbow", "R_Elbow",
  "L_Wrist", "R_Wrist",
  ...["L", "R"].flatMap((s) =>
    ["Index", "Middle", "Ring", "Pinky", "Thumb"].flatMap((f) =>
      [1, 2, 3, 4].map((i) => `${s}_${f}${i}`))),
  "L_Eye", "R_Eye",
];

/**
 * SMPL joint -> rig node, in the normalised naming glb-humanoid.ts uses.
 * The SPINE OFFSET is the one people get wrong: SMPL's Spine1/2/3 are Mixamo's
 * Spine/Spine1/Spine2, so every entry is shifted by one.
 */
const SMPL_TO_RIG = {
  Pelvis: "hips",
  L_Hip: "leftupleg", R_Hip: "rightupleg",
  L_Knee: "leftleg", R_Knee: "rightleg",
  L_Ankle: "leftfoot", R_Ankle: "rightfoot",
  L_Foot: "lefttoebase", R_Foot: "righttoebase",
  Spine1: "spine", Spine2: "spine1", Spine3: "spine2",
  Neck: "neck", Head: "head",
  L_Collar: "leftshoulder", R_Collar: "rightshoulder",
  L_Shoulder: "leftarm", R_Shoulder: "rightarm",
  L_Elbow: "leftforearm", R_Elbow: "rightforearm",
  L_Wrist: "lefthand", R_Wrist: "righthand",
  L_Eye: "lefteye", R_Eye: "righteye", // absent on Mixamo rigs -> synthesised
};
for (const [s, S] of [["L", "Left"], ["R", "Right"]]) {
  for (const f of ["Index", "Middle", "Ring", "Pinky", "Thumb"]) {
    for (const i of [1, 2, 3, 4]) {
      SMPL_TO_RIG[`${s}_${f}${i}`] = `${S}Hand${f}${i}`.toLowerCase();
    }
  }
}

/** Joints without which the retarget cannot build a body at all. */
const REQUIRED = [
  "Pelvis", "Spine1", "Spine2", "Spine3", "Neck", "Head",
  "L_Hip", "L_Knee", "L_Ankle", "R_Hip", "R_Knee", "R_Ankle",
  "L_Collar", "L_Shoulder", "L_Elbow", "L_Wrist",
  "R_Collar", "R_Shoulder", "R_Elbow", "R_Wrist",
];

/**
 * `mixamorig:LeftArm` / `mixamorigLeftArm` / `Armature|LeftArm` -> `leftarm`.
 * The COLON IS OPTIONAL, which glb-humanoid.ts's otherwise-identical version does not
 * allow: FBX node names cannot contain ':', so a .fbx straight from Mixamo arrives as
 * `mixamorigHips` and a colon-anchored prefix strips nothing. Measured: all 65 bones
 * present, zero matched.
 */
const normalizeName = (raw) =>
  raw.replace(/^mixamorig\d*:?/i, "").replace(/^.*\|/, "").replace(/[\s_.-]/g, "").toLowerCase();

function parseArgs(argv) {
  const o = { inputs: [], out: path.join(REPO, "motion-lib-catalog"), fps: 30, prefix: "" };
  for (let i = 0; i < argv.length; i += 1) {
    const a = argv[i];
    if (a === "--out") o.out = path.resolve(argv[++i]);
    else if (a === "--name") o.name = argv[++i];
    else if (a === "--prefix") o.prefix = argv[++i];
    else if (a === "--fps") o.fps = Number(argv[++i]);
    else if (a === "--scale") o.scale = Number(argv[++i]);
    else if (a === "--keep-last") o.keepLast = true;
    else if (a === "--dry") o.dry = true;
    else if (a.startsWith("--")) throw new Error(`unknown option ${a}`);
    else o.inputs.push(a);
  }
  if (!o.inputs.length) throw new Error("no input files");
  if (o.name && o.inputs.length > 1) throw new Error("--name needs exactly one input");
  return o;
}

async function loadSource(file) {
  const buf = fs.readFileSync(file);
  const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
  const base = path.dirname(path.resolve(file)) + path.sep;
  if (/\.fbx$/i.test(file)) {
    const root = new FBXLoader().parse(ab, base);
    return { root, animations: root.animations || [] };
  }
  if (/\.(glb|gltf)$/i.test(file)) {
    const gltf = await new Promise((res, rej) =>
      new GLTFLoader().parse(ab, base, res, rej));
    return { root: gltf.scene, animations: gltf.animations || [] };
  }
  throw new Error(`unsupported input ${path.extname(file)} (want .fbx/.glb)`);
}

function collectBones(root) {
  const byName = new Map();
  root.updateMatrixWorld(true);
  root.traverse((o) => {
    const n = normalizeName(o.name || "");
    if (!n) return;
    // A bone always wins over a mesh/empty that happens to share the name.
    if (!byName.has(n) || (o.isBone && !byName.get(n).isBone)) byName.set(n, o);
  });
  return byName;
}

function convert(file, opt) {
  return loadSource(file).then(({ root, animations }) => {
    if (!animations.length) throw new Error("no animation track in file");
    const clip = animations[0];
    const bones = collectBones(root);

    const nodes = {};
    const missing = [];
    for (const j of SMPL_JOINTS) {
      const n = bones.get(SMPL_TO_RIG[j]);
      if (n) nodes[j] = n;
      else missing.push(j);
    }
    const hardMissing = REQUIRED.filter((j) => !nodes[j]);
    if (hardMissing.length) {
      throw new Error(
        `rig is missing required bones: ${hardMissing.join(", ")}\n` +
        `        (found ${bones.size} nodes; is this a Mixamo-named humanoid?)`);
    }

    // REST pose, captured before the mixer touches anything: the eye synthesis
    // works off the head's rotation *relative to rest*, which is what makes it
    // independent of whatever local axis convention the rig's head bone uses.
    root.updateMatrixWorld(true);
    const headRestInv = nodes.Head.getWorldQuaternion(new THREE.Quaternion()).invert();

    const mixer = new THREE.AnimationMixer(root);
    mixer.clipAction(clip).play();

    const fps = opt.fps;
    // Drop the final frame by default: Mixamo's last frame duplicates the first,
    // and keeping it stutters every loop of an idle.
    const nFrames = Math.max(1, Math.round(clip.duration * fps) + (opt.keepLast ? 1 : 0));

    const iHead = SMPL_JOINTS.indexOf("Head");
    const raw = [];
    const eyeAxes = []; // per frame, unit; the eyes are placed AFTER scaling (below)
    const v = new THREE.Vector3();
    const q = new THREE.Quaternion();
    const eyeAxis = new THREE.Vector3();
    for (let f = 0; f < nFrames; f += 1) {
      mixer.setTime(f / fps);
      root.updateMatrixWorld(true);
      const frame = new Array(SMPL_JOINTS.length);
      // Head rotation since rest, applied to world +X (= the character's left at
      // rest), gives the interocular axis for this frame.
      nodes.Head.getWorldQuaternion(q).multiply(headRestInv);
      eyeAxis.set(1, 0, 0).applyQuaternion(q);
      eyeAxes.push([eyeAxis.x, eyeAxis.y, eyeAxis.z]);
      SMPL_JOINTS.forEach((j, i) => {
        const n = nodes[j];
        if (n) { n.getWorldPosition(v); frame[i] = [v.x, v.y, v.z]; return; }
        // No such bone on this rig. The eyes get synthesised after scaling; any
        // other absentee collapses onto the head so the array stays dense.
        frame[i] = null;
      });
      for (let i = 0; i < frame.length; i += 1) {
        if (!frame[i] && SMPL_JOINTS[i] !== "L_Eye" && SMPL_JOINTS[i] !== "R_Eye") {
          frame[i] = frame[iHead].slice();
        }
      }
      raw.push(frame);
    }

    // ---- normalise ------------------------------------------------------
    // Scale: Mixamo FBX ships in centimetres, GLB usually in metres. Decide from
    // the figure's own height rather than trusting the extension.
    let hi = -Infinity; let lo = Infinity;
    for (const fr of raw) for (const p of fr) { if (p && p[1] > hi) hi = p[1]; if (p && p[1] < lo) lo = p[1]; }
    const srcHeight = hi - lo;
    const scale = opt.scale || (srcHeight > 10 ? 0.01 : 1);

    const joints = raw.map((fr) => fr.map((p) => (p ? [p[0] * scale, p[1] * scale, p[2] * scale] : null)));

    // Eyes, synthesised in FINAL METRES. Doing this before the scale multiply put a
    // 0.045 *metre* offset into centimetre-space source data, so a Mixamo .fbx came
    // out with the eyes 0.9 mm apart instead of 90 — a head basis built from a vector
    // that short is pure numerical noise. Measured: 0.0009 m separation before, 0.09 after.
    const EYE_HALF = 0.045; // half the 0.09 m interocular distance the shipped clips use
    const iLE = SMPL_JOINTS.indexOf("L_Eye");
    const iRE = SMPL_JOINTS.indexOf("R_Eye");
    joints.forEach((fr, f) => {
      const h = fr[iHead];
      const a = eyeAxes[f];
      for (const [i, s] of [[iLE, 1], [iRE, -1]]) {
        if (!fr[i]) {
          fr[i] = [h[0] + a[0] * EYE_HALF * s, h[1] + a[1] * EYE_HALF * s, h[2] + a[2] * EYE_HALF * s];
        }
      }
    });

    let minY = Infinity;
    for (const fr of joints) for (const p of fr) if (p[1] < minY) minY = p[1];
    const iLA = SMPL_JOINTS.indexOf("L_Ankle");
    const iRA = SMPL_JOINTS.indexOf("R_Ankle");
    const ox = (joints[0][iLA][0] + joints[0][iRA][0]) / 2;
    const oz = (joints[0][iLA][2] + joints[0][iRA][2]) / 2;
    for (const fr of joints) {
      for (const p of fr) { p[0] -= ox; p[1] -= minY; p[2] -= oz; }
    }
    const r4 = (n) => Math.round(n * 1e4) / 1e4;

    return {
      clipName: clip.name,
      srcFps: Math.round(clip.tracks[0]?.times?.length / clip.duration) || null,
      scale,
      height: r4(srcHeight * scale),
      missing: missing.filter((j) => j !== "L_Eye" && j !== "R_Eye"),
      synthEyes: !nodes.L_Eye,
      // Reported so a scale mistake can never silently collapse the head basis again.
      eyeSep: r4(Math.hypot(
        joints[0][iLE][0] - joints[0][iRE][0],
        joints[0][iLE][1] - joints[0][iRE][1],
        joints[0][iLE][2] - joints[0][iRE][2],
      )),
      data: {
        joint_names: SMPL_JOINTS,
        joints: joints.map((fr) => fr.map((p) => [r4(p[0]), r4(p[1]), r4(p[2])])),
        fps,
      },
    };
  });
}

/** The planted-feet audit motion_lib._measure() runs, reproduced so a clip can be
 *  judged BEFORE it is written into the catalog. Same thresholds as the picker. */
function audit(d) {
  const jn = d.joint_names; const J = d.joints;
  const ix = Object.fromEntries(jn.map((n, i) => [n, i]));
  const pv = ix.Pelvis;
  let lift = 0; let travel = 0;
  for (const a of ["L_Ankle", "R_Ankle"]) {
    const j = ix[a];
    const ys = J.map((f) => f[j][1] - f[pv][1]);
    const xs = J.map((f) => f[j][0] - f[pv][0]);
    const zs = J.map((f) => f[j][2] - f[pv][2]);
    lift = Math.max(lift, Math.max(...ys) - Math.min(...ys));
    travel = Math.max(travel, Math.hypot(
      Math.max(...xs) - Math.min(...xs), Math.max(...zs) - Math.min(...zs)));
  }
  let knee = 0;
  for (const [hip, kn, an] of [["L_Hip", "L_Knee", "L_Ankle"], ["R_Hip", "R_Knee", "R_Ankle"]]) {
    const h = ix[hip]; const k = ix[kn]; const a = ix[an];
    for (const f of J) {
      const u = [0, 1, 2].map((i) => f[h][i] - f[k][i]);
      const w = [0, 1, 2].map((i) => f[a][i] - f[k][i]);
      const du = Math.hypot(...u) || 1; const dw = Math.hypot(...w) || 1;
      const dot = (u[0] * w[0] + u[1] * w[1] + u[2] * w[2]) / (du * dw);
      knee = Math.max(knee, 180 - (Math.acos(Math.max(-1, Math.min(1, dot))) * 180) / Math.PI);
    }
  }
  const hipDrop = Math.max(...J.map((f) => f[pv][1])) - Math.min(...J.map((f) => f[pv][1]));

  // STANCE WIDTH. The original audit measured knees and foot lift but nothing
  // LATERAL, so a whole batch of clips shipped with the legs visibly splayed and the
  // numbers all looked fine. thighOut is the mean outward tilt of the thigh (hip->ankle,
  // sideways component); foreAft separates a splayed stance from a staggered one,
  // because a step forward inflates ankle separation without opening the legs at all.
  // Reference: this avatar's own embedded idle is +0.5 deg, the KAWAII clips -1.5..-0.9.
  // Mixamo standing idles run +3 to +5 and read as a wrestler's stance on a chancellor.
  const hw = Math.hypot(
    J[0][ix.L_Hip][0] - J[0][ix.R_Hip][0], J[0][ix.L_Hip][2] - J[0][ix.R_Hip][2],
  ) || 1;
  let splaySum = 0; let splayN = 0; let lat = 0; let foreAft = 0;
  for (const f of J) {
    lat += Math.abs(f[ix.L_Ankle][0] - f[ix.R_Ankle][0]);
    foreAft = Math.max(foreAft, Math.abs(f[ix.L_Ankle][2] - f[ix.R_Ankle][2]));
    for (const [hip, ank, sgn] of [["L_Hip", "L_Ankle", 1], ["R_Hip", "R_Ankle", -1]]) {
      const h = f[ix[hip]]; const a = f[ix[ank]];
      const d = [a[0] - h[0], a[1] - h[1], a[2] - h[2]];
      const n = Math.hypot(...d) || 1;
      splaySum += (Math.asin(Math.max(-1, Math.min(1, (d[0] * sgn) / n))) * 180) / Math.PI;
      splayN += 1;
    }
  }
  const thighOut = splaySum / splayN;

  // TOE-OUT is what actually reaches the render. Measured on this avatar: a source
  // clip with the left foot at +28.3 deg drew at +25.2, while swapping a +3.1 deg
  // thigh splay for -1.1 moved the drawn foot separation by 0.03 — i.e. the retarget
  // re-solves the stance WIDTH but takes the ankle ORIENTATION straight from the clip.
  // ~10-15 deg of toe-out is normal human stance; what reads as "legs splayed" is one
  // foot much further out than the other, so both the magnitude and the asymmetry are
  // scored. Reference: this avatar's embedded idle is +3.9/+14.9.
  const toe = { L: 0, R: 0 };
  for (const f of J) {
    for (const [ank, toeJ, side, sgn] of [
      ["L_Ankle", "L_Foot", "L", 1], ["R_Ankle", "R_Foot", "R", -1]]) {
      const a = f[ix[ank]]; const t = f[ix[toeJ]];
      toe[side] += (Math.atan2((t[0] - a[0]) * sgn, t[2] - a[2]) * 180) / Math.PI;
    }
  }
  const toeL = toe.L / J.length; const toeR = toe.R / J.length;
  const maxToe = Math.max(Math.abs(toeL), Math.abs(toeR));
  const toeAsym = Math.abs(toeL - toeR);
  return {
    toeL: Math.round(toeL * 10) / 10,
    toeR: Math.round(toeR * 10) / 10,
    splayFeet: maxToe > 15 || toeAsym > 8,
    footLift: Math.round(lift * 1e3) / 1e3,
    footTravel: Math.round(travel * 1e3) / 1e3,
    kneeMax: Math.round(knee),
    hipDrop: Math.round(hipDrop * 1e3) / 1e3,
    thighOut: Math.round(thighOut * 10) / 10,
    latSep: Math.round((lat / J.length / hw) * 100) / 100,
    foreAft: Math.round(foreAft * 100) / 100,
    warn: lift > 0.15 || knee > 70,
    // thighOut is PRINTED but deliberately does NOT raise a flag: the retarget
    // re-solves stance width against the avatar's own legs, so the clip's value
    // barely survives to the screen (a +3.1 and a -1.1 clip drew 0.03 apart). It is
    // kept only because it explains the source pose. splayFeet is the one that bites.
    wide: false,
    stagger: foreAft > 0.25,
  };
}

const opt = parseArgs(process.argv.slice(2));
if (!opt.dry) fs.mkdirSync(opt.out, { recursive: true });

let bad = 0;
for (const file of opt.inputs) {
  const key = (opt.name
    || opt.prefix + path.basename(file).replace(/\.(fbx|glb|gltf)$/i, "")
  ).toLowerCase().replace(/[^a-z0-9_]+/g, "_");
  try {
    const r = await convert(file, opt);
    const a = audit(r.data);
    const secs = (r.data.joints.length / r.data.fps).toFixed(1);
    const flags = [
      a.warn ? "moves the feet — the ankle IK will fight this" : "",
      a.wide ? `LEGS SPLAYED ${a.thighOut}deg — reads as a wide stance` : "",
      a.stagger ? `STAGGERED ${a.foreAft}m fore-aft` : "",
      a.splayFeet ? `FEET SPLAYED L${a.toeL >= 0 ? "+" : ""}${a.toeL}/R${a.toeR >= 0 ? "+" : ""}${a.toeR}deg` : "",
    ].filter(Boolean);
    console.log(
      `${key}\n` +
      `   ${r.data.joints.length} frames (${secs}s @${r.data.fps}) ` +
      `scale=${r.scale} height=${r.height}m ` +
      `eyes=${r.synthEyes ? "synth" : "rig"}/${(r.eyeSep * 1000).toFixed(0)}mm\n` +
      `   footLift ${a.footLift}m  footTravel ${a.footTravel}m  ` +
      `knee ${a.kneeMax}deg  hipDrop ${a.hipDrop}m\n` +
      `   thighOut ${a.thighOut >= 0 ? "+" : ""}${a.thighOut}deg  latSep ${a.latSep}  ` +
      `foreAft ${a.foreAft}m  toes L${a.toeL >= 0 ? "+" : ""}${a.toeL}/R${a.toeR >= 0 ? "+" : ""}${a.toeR}deg` +
      (flags.length ? `\n   ** ${flags.join(" / ")} **` : "  OK"));
    if (r.missing.length) console.log(`   missing (optional): ${r.missing.join(", ")}`);
    if (!opt.dry) {
      fs.writeFileSync(path.join(opt.out, `${key}.json`), JSON.stringify(r.data));
    }
  } catch (e) {
    bad += 1;
    console.error(`${key}\n   FAILED: ${e.message}`);
  }
}
process.exit(bad ? 1 : 0);
