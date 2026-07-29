/* eslint-disable no-underscore-dangle */
import * as THREE from "three";
import { aiDebug } from "./debug/ai-debug";
import type { VRM } from "@pixiv/three-vrm";
import { clampHandsOutsideTorso, newTorsoClampState, SKIRT } from "./torso-clamp";

// SMPL-X joint -> VRM humanoid bone AIM map: the bone's rest child-direction is
// rotated onto the SMPL (from -> to) direction. SMPL and VRM share the frame
// (X-left, Y-up, Z-front), so no axis conversion is needed. This is the same
// retarget validated in the offline harness — far simpler than the MMD pipeline
// because VRM is a standard humanoid (no twist/helper bones, clean T-pose).
const AIM: Record<string, [string, string]> = {
  spine: ["Spine1", "Spine2"], chest: ["Spine2", "Spine3"], upperChest: ["Spine3", "Neck"],
  neck: ["Neck", "Head"],
  leftShoulder: ["L_Collar", "L_Shoulder"], leftUpperArm: ["L_Shoulder", "L_Elbow"],
  leftLowerArm: ["L_Elbow", "L_Wrist"], leftHand: ["L_Wrist", "L_Middle1"],
  rightShoulder: ["R_Collar", "R_Shoulder"], rightUpperArm: ["R_Shoulder", "R_Elbow"],
  rightLowerArm: ["R_Elbow", "R_Wrist"], rightHand: ["R_Wrist", "R_Middle1"],
  leftUpperLeg: ["L_Hip", "L_Knee"], leftLowerLeg: ["L_Knee", "L_Ankle"], leftFoot: ["L_Ankle", "L_Foot"],
  rightUpperLeg: ["R_Hip", "R_Knee"], rightLowerLeg: ["R_Knee", "R_Ankle"], rightFoot: ["R_Ankle", "R_Foot"],
};
/** ROLL RECOVERY. A bone's orientation is 3 DOF but the direction to its child pins only 2 —
 *  the roll about the bone's own axis is NOT recoverable from a single segment, and
 *  setFromUnitVectors (what the aim path uses) is the MINIMAL rotation, i.e. it sets that roll
 *  to zero for every bone. Only the four FRAME bones (hips/chest/head/hands) build a real
 *  basis from two landmark pairs, so everything else throws its twist away — which is exactly
 *  why twist-heavy clips fall apart while calm ones look fine.
 *
 *  Three CONSECUTIVE joints do pin it: they span a plane, and that plane is the joint's bend
 *  plane. A knee that flexes in a given plane fully determines the thigh's roll. So for each
 *  bone here, [a, b, c] = its own segment a->b plus the next joint c; the bend axis
 *  (a->b) x (b->c) supplies the missing third axis.
 *
 *  The FOOT stays aim-only on purpose: ankle->toe is the last segment, there is no fourth
 *  joint to make a plane with, and a real ankle barely rolls — so it should inherit the
 *  shin's roll, which is what it now gets. */
const PLANE: Record<string, [string, string, string]> = {
  leftUpperLeg: ["L_Hip", "L_Knee", "L_Ankle"],
  leftLowerLeg: ["L_Knee", "L_Ankle", "L_Foot"],
  rightUpperLeg: ["R_Hip", "R_Knee", "R_Ankle"],
  rightLowerLeg: ["R_Knee", "R_Ankle", "R_Foot"],
  // Elbow hinge: pins the UPPER ARM's roll (humerus axial rotation). An earlier attempt
  // applied the plane as a FULL orientation for the arms and blew them into a rigid T —
  // the clip's rest frame has the arms nearly straight, so the rest plane was noise. The
  // hinge-roll path below avoids that failure mode by construction: it only ever corrects
  // the ROLL component, and it fades with the bend of BOTH the current frame and the rest
  // frame (planeRestSin), so a straight-at-rest arm simply keeps its inherited roll.
  leftUpperArm: ["L_Shoulder", "L_Elbow", "L_Wrist"],
  rightUpperArm: ["R_Shoulder", "R_Elbow", "R_Wrist"],
};

/** Bones whose roll is pinned by the NEXT joint being a hinge: the knee pins the thigh,
 *  the ankle pins the shin (an ankle cannot yaw — foot direction changes are tibial
 *  rotation), and the elbow pins the upper arm. Legs gate on kneeRoll, arms on elbowRoll. */
const HINGE_ROLL = new Set([
  "leftUpperLeg", "rightUpperLeg", "leftLowerLeg", "rightLowerLeg",
  "leftUpperArm", "rightUpperArm",
]);

// parents before children
const ORDER = [
  "spine", "chest", "upperChest", "neck",
  "leftShoulder", "leftUpperArm", "leftLowerArm", "leftHand",
  "rightShoulder", "rightUpperArm", "rightLowerArm", "rightHand",
  "leftUpperLeg", "leftLowerLeg", "leftFoot", "rightUpperLeg", "rightLowerLeg", "rightFoot",
];
const CHILD: Record<string, string> = {
  spine: "chest", chest: "upperChest", upperChest: "neck", neck: "head",
  leftShoulder: "leftUpperArm", leftUpperArm: "leftLowerArm", leftLowerArm: "leftHand",
  rightShoulder: "rightUpperArm", rightUpperArm: "rightLowerArm", rightLowerArm: "rightHand",
  leftUpperLeg: "leftLowerLeg", leftLowerLeg: "leftFoot", leftFoot: "leftToes",
  rightUpperLeg: "rightLowerLeg", rightLowerLeg: "rightFoot", rightFoot: "rightToes",
};

// --- Hands + fingers ------------------------------------------------------------
// SMPL-X carries full hand articulation (5 fingers × 3 joints/side) and the VRM has
// the matching finger bones, but they were never retargeted — so the hands stayed in
// the bind pose while the sample's fingers gesture ("手の動きがちがう"). Also the WRIST
// itself wasn't oriented (CHILD["leftHand"] was undefined, so its aim silently
// no-op'd). Aim the hand toward the middle-finger base (so the palm follows the
// sample), then aim each finger phalanx onto its SMPL segment. SMPL "Pinky" -> VRM
// "Little"; the VRM thumb is Metacarpal/Proximal/Distal (no Intermediate).
(["left", "right"] as const).forEach((side) => {
  const S = side === "left" ? "L" : "R";
  CHILD[`${side}Hand`] = `${side}MiddleProximal`; // orient the wrist toward the palm/fingers
  ([["Index", "Index"], ["Middle", "Middle"], ["Ring", "Ring"], ["Little", "Pinky"]] as const)
    .forEach(([vrmF, smplF]) => {
      AIM[`${side}${vrmF}Proximal`] = [`${S}_${smplF}1`, `${S}_${smplF}2`];
      CHILD[`${side}${vrmF}Proximal`] = `${side}${vrmF}Intermediate`;
      AIM[`${side}${vrmF}Intermediate`] = [`${S}_${smplF}2`, `${S}_${smplF}3`];
      CHILD[`${side}${vrmF}Intermediate`] = `${side}${vrmF}Distal`;
      // The DISTAL knuckle needs a 4th landmark (the fingertip) to aim at. Without it it
      // was never retargeted at all — measured 0 deg of motion across a whole session on
      // bones carrying ~190 skin weight each, i.e. a fist that never closed. The KAWAII
      // converter now emits `*4` tips; GestureLSM (SMPL-X) has none, so its clips simply
      // skip these bones as before.
      AIM[`${side}${vrmF}Distal`] = [`${S}_${smplF}3`, `${S}_${smplF}4`];
      ORDER.push(`${side}${vrmF}Proximal`, `${side}${vrmF}Intermediate`, `${side}${vrmF}Distal`);
    });
  AIM[`${side}ThumbMetacarpal`] = [`${S}_Thumb1`, `${S}_Thumb2`];
  CHILD[`${side}ThumbMetacarpal`] = `${side}ThumbProximal`;
  AIM[`${side}ThumbProximal`] = [`${S}_Thumb2`, `${S}_Thumb3`];
  CHILD[`${side}ThumbProximal`] = `${side}ThumbDistal`;
  AIM[`${side}ThumbDistal`] = [`${S}_Thumb3`, `${S}_Thumb4`];
  ORDER.push(`${side}ThumbMetacarpal`, `${side}ThumbProximal`, `${side}ThumbDistal`);
});

// --- Full-orientation (frame) bones -------------------------------------------
// The aim-by-child-direction method sets a bone's DIRECTION but can't retarget a
// full 3-axis orientation, so two big motions were being dropped entirely:
//   • the PELVIS turn/lean (~26°): the hips have no single "child" to aim, and the
//     spine/arms being aimed to ABSOLUTE world directions means they compensate for
//     a fixed pelvis — so the lower body never turns with the sample.
//   • the HEAD look (~26°): the head bone has no "forward" child; only the neck was
//     positioning it, so nods/turns of the face were lost.
// Build each bone's frame from anatomical landmarks: `up` = along the bone, `right`
// = the left/right landmark line (hips: the hip line; head: the eye line). Retarget
// is RELATIVE to the clip's first (rest) frame so it stacks cleanly on the absolute
// aim chain and preserves each rig's own rest pose.
/** Minimum sin(angle) between a FRAME bone's two landmark axes for the basis to be usable
 *  (see buildFrameQuat). ~11.5 deg: the anatomical pairs below sit near 90 deg in any sane
 *  pose, so this only ever fires on a degenerate hand. */
// DISABLED (0) pending a look-check: measured to fire on 1/3622 frames, i.e. it was doing
// nothing, so it is not worth carrying risk for during a regression hunt.
const FRAME_MIN_SIN = 0;

/** A FRAME bone's landmark axes must also keep at least this fraction of their REST length
 *  to be trusted. Retargeting a crossfaded frame interpolates each landmark independently,
 *  so an axis can shrink toward zero even though neither clip is degenerate — the
 *  interpolated wrist walking into the interpolated middle knuckle. A collapsed axis
 *  normalizes to a noise direction, which is how the wrist snapped 60-130 deg for one or
 *  two frames in the middle of a fade. */
// DISABLED (0) pending a look-check — same reason as FRAME_MIN_SIN (never observed firing).
const FRAME_MIN_LEN_FRAC = 0;

/** Slew limit (rad/s) on a FRAME bone's retargeted target. ~860 deg/s — several times the
 *  fastest hand in any of the clips, so real motion passes untouched, but the landmark
 *  basis FLIPPING (measured 140-185 deg between two consecutive frames, oscillating for a
 *  few frames mid-crossfade) cannot. The per-bone low-pass does not help here: it only
 *  spreads a 180 deg target step over a few frames, which is still a visible snap. */
// DISABLED (Infinity) pending a look-check. It never saturated on the clips measured
// (max step 12.58 deg vs a 14.3 deg cap), but the WRISTS are FRAME bones and the reported
// symptoms include the wrists, so it must not be in the picture while bisecting.
const FRAME_MAX_SLEW = Infinity;

const FRAME: Record<string, { up: [string, string]; right: [string, string] }> = {
  hips: { up: ["Pelvis", "Spine1"], right: ["R_Hip", "L_Hip"] },
  // chest: full orientation from the SHOULDER LINE (right axis) so its roll matches the
  // sample — the aim-only version left the roll uncontrolled and tilted the torso ~4°.
  // Only used when the VRM has no upperChest (Miku); see adaptChains/chestFrame.
  chest: { up: ["Spine2", "Neck"], right: ["R_Shoulder", "L_Shoulder"] },
  head: { up: ["Neck", "Head"], right: ["R_Eye", "L_Eye"] },
  // hands: aim-only left the wrist TWIST uncontrolled so the palm faced the wrong way
  // ("掌の向きが逆"). Drive the full hand orientation: up = wrist->middle knuckle, right
  // = the knuckle line (index->pinky) — together they fix the palm-facing direction.
  leftHand: { up: ["L_Wrist", "L_Middle1"], right: ["L_Index1", "L_Pinky1"] },
  rightHand: { up: ["R_Wrist", "R_Middle1"], right: ["R_Index1", "R_Pinky1"] },
};

type Clip = {
  names: string[];
  idx: Record<string, number>;
  joints: number[][][];
  fps: number;
  /** start-pose-matched playhead offset (see _matchStart); also the loop-wrap point */
  pad?: number;
  /** generated with the PREVIOUS clip's latents as seed: frames [0, overlap) replay
   *  the prior clip's tail, so starting at overlap-1 lands EXACTLY on the prior end
   *  pose and the motion genuinely continues (Plan A: gateway-side chaining) */
  chained?: boolean;
  /** KAWAII talk-loop: a seamless (seam≈0) Speak clip that WRAPS its playhead so it
   *  keeps talking for the whole sentence instead of freezing at its last frame. */
  loop?: boolean;
  /** BETWEEN-turNS idle animation (KAWAII Idle*): plays ONCE, then the player asks the
   *  backend for the next random one instead of holding/looping its last frame. */
  idle?: boolean;
  /** Generated by GestureLSM (vs a baked KAWAII clip). Tagged by the backend — it used
   *  to be inferred from the joint count, which stopped working once the KAWAII clips
   *  also carried fingers. */
  lsm?: boolean;
};

/**
 * Drives a VRM avatar with audio-generated co-speech gestures (GestureLSM). The
 * backend serves the raw SMPL joints (/motion/clip-smpl/{id}); this retargets them
 * to the VRM humanoid every frame. Singleton (mirrors mmdMotionPlayer) so the
 * always-mounted audio/websocket layer can drive it without prop drilling.
 */
class VRMMotionPlayer {
  private vrm: VRM | null = null;

  private restWQ: Record<string, THREE.Quaternion> = {};

  private restDir: Record<string, THREE.Vector3> = {};

  // Frame (hips/head) retarget: Miku's bone world-quat at rest, and the SMPL
  // anatomical frame at the clip's first frame (the "rest" the delta is measured
  // from). `_restClip` tracks which clip the SMPL rest was captured for.
  private frameRestMiku: Record<string, THREE.Quaternion> = {};

  private frameRestSmpl: Record<string, THREE.Quaternion> = {};
  /** min(|up|, |right|) of each FRAME bone's landmark axes at the clip's rest frame — the
   *  scale reference for rejecting a collapsed axis (see FRAME_MIN_LEN_FRAC). */
  private frameRestLen: Record<string, number> = {};
  /** Rest basis of each PLANE bone (see PLANE) in the clip's own space. */
  /** PARENT-RELATIVE aim (the root fix for twist-heavy clips breaking).
   *
   *  setFromUnitVectors is the MINIMAL rotation from the rest direction to the current one,
   *  so a bone whose direction a body-turn does not change — the near-vertical spine, thighs
   *  and shins — receives NO rotation at all and stays facing the REST front while the pelvis
   *  (FRAME, full orientation) turns 150 deg under it. The belly skin is then torn between
   *  the turned 下半身 and the un-turned 上半身 ("おなかがくしゃくしゃ"), and the foot (whose
   *  ankle->toe direction DOES turn) yaws while the shin does not — the ankle folds sideways
   *  ("足首が曲がってはいけない方向"). Direction-only fidelity metrics scored all of this as
   *  perfect, which is how it survived every earlier audit: the error is pure ROLL.
   *
   *  Fix: inherit the PARENT's world rotation delta first, then apply only the small swing
   *  that corrects the child direction. Twist propagates down the chain from the bones that
   *  genuinely know it (hips/chest via their two-landmark FRAME bases); the bone still points
   *  exactly where the clip says. No tuned parameters. Live A/B: __vrmMotion.parentAim. */
  parentAim = true;
  private restPWInv: Record<string, THREE.Quaternion> = {};
  private _paDelta = new THREE.Quaternion();
  private _paPred = new THREE.Vector3();

  /** Roll recovery from the bend plane (see PLANE). Live: __vrmMotion.planeRoll = true. */
  planeRoll = false;

  private planeRestSmpl: Record<string, THREE.Quaternion> = {};
  private _qPlane = new THREE.Quaternion();
  private _planeSin = 0;
  private _krA = new THREE.Quaternion();
  private _krB = new THREE.Quaternion();
  /** Recover the THIGH's roll from the knee's bend plane. The knee is a hinge, so the plane
   *  hip-knee-ankle pins the femur's twist exactly; the shin then inherits it (a hinge adds
   *  no roll) and so does the foot. Without this, parentAim gives the thigh the PELVIS's roll,
   *  and any real leg rotation is missing all the way down — which is the residual "足首が
   *  ちょっとおかしい" left after parentAim. Direction is untouched: only the component about
   *  the bone's own axis is changed, faded out as the knee straightens (no data there).
   *  Live: __vrmMotion.kneeRoll = false. */
  //
  // DEFAULT OFF: it makes clip SEAMS worse. The weight below is a function of how bent the
  // hinge is RIGHT NOW, and a crossfade blends two poses — the knee passes through straight,
  // the weight collapses to 0, and the correction switches off over a frame or two. Measured
  // across four real clip switches: at-seam deviation p95 1.08 -> 2.92 deg, max 1.71 -> 5.66,
  // with fresh 22-24 deg spikes on the shins. Its steady-state benefit was only ~2 deg, so it
  // is not worth a visible seam. To re-enable it properly the weight has to be carried through
  // the fade (e.g. hold the pre-fade weight, or derive the bend from the unblended new frame).
  kneeRoll = false;
  /** Same mechanism for the ARMS (elbow hinge -> upper-arm roll). Separate switch so the
   *  two can be A/B'd independently. Live: __vrmMotion.elbowRoll = false. */
  elbowRoll = false;   // same seam problem as kneeRoll (see above)
  /** Hinge bend (sin) at which the roll correction reaches full strength. */
  kneeRollFull = 0.45;
  /** Bend of each hinge at the clip's REST frame — a straight-at-rest joint has a noise
   *  plane there, so its correction is faded out by this as well as by the current bend. */
  private planeRestSin: Record<string, number> = {};
  /** Previous frame's retargeted WORLD target per FRAME bone, for the slew limit. */
  private _qtPrev: Record<string, THREE.Quaternion> = {};
  /** Last update()'s dt, so applyFrameBone can rate-limit in real time units. */
  private _dt = 1 / 60;
  /** Carries the torso-guard push between frames so it can be rate-limited (see
   *  GUARD_RATE_MPS in torso-clamp) — an unlimited hand-off pops the elbow. */
  private _torsoGuard = newTorsoClampState();

  // The PREVIOUS clip's SMPL rest, kept so the FRAME bones' (hips/chest/head) relative
  // retarget can BLEND its rest baseline from old->new across a crossfade. Without this,
  // the rest re-captures instantly on a clip change and the hips rotation baseline jumps
  // — the body "swings left->right" at the seam even though the joint positions blend.
  private _frameRestSmplPrev: Record<string, THREE.Quaternion> = {};
  private _restBlend = new THREE.Quaternion();

  // Hands use ABSOLUTE (not relative) frames so the palm matches the sample exactly
  // (the relative delta left Miku's rest-hand offset). handK maps the SMPL hand frame
  // onto Miku's hand bone: target_world = SMPL_frame_now * handK.
  private handK: Record<string, THREE.Quaternion> = {};

  private _restClip: Clip | null = null;

  private clip: Clip | null = null;

  // Two-clip crossfade: to switch/loop clips WITHOUT the body ever losing velocity (the
  // synchronized whole-body pose-ease at a transition is the brief "止まる" at the seam),
  // we keep the OUTGOING clip advancing too and blend its joints with the incoming clip's
  // over `_fadeDur`. Both playheads are moving, so the blended frame is always in motion.
  private _prevClip: Clip | null = null;
  private _prevElapsed = 0;
  private _fadeT = 1; // 1 = no crossfade in progress
  private _fadeDur = 0.6;  // gentler clip reorient -> less twintail whip at seams
  /** The outgoing side is a single-frame SNAPSHOT (a seam landed mid-fade), so it has no
   *  velocity of its own. Smoothstep's flat head would read as a stall there — shorten
   *  the fade for that case only, so the quiet part is 2-3 frames instead of ~0.15s.
   *  Read as an EFFECTIVE duration when advancing _fadeT: the call sites assign _fadeDur
   *  after _beginFade() runs, so clamping the field itself would be overwritten. */
  private _fadeStatic = false;
  //
  // The 0.25s shortening this used to apply is GONE. It was added to hide a stall I never
  // actually saw on screen, and it is what made a mid-fade interruption read as a jolt: a
  // clip landing part-way through a 1.0s idle crossfade restarted the fade at 0.25s, i.e. the
  // pose then travelled FOUR TIMES faster. Traced on a real second switch: _fadeT stepped
  // 0.462 -> 0.446 and then ran 0.52, 0.59, 0.66 per frame. Something does apply clips
  // uninvited (the backend keeps a pre-generated buffer warm and fetches land every 2-3s even
  // with no conversation), so this path is reached in normal use — it has to stay smooth.
  private get _fadeDurEff() {
    void this._fadeStatic; // kept for diagnostics (__vrmMotion._fadeStatic)
    return this._fadeDur;
  }

  /** Crossfade weight. Smoothstep: w'(0) = w'(1) = 0, so the blend adds no velocity step
   *  at either end of the fade (see the position blend for why that matters). */
  private _fadeW() { const x = this._fadeT; return x * x * (3 - 2 * x); }
  private _frame2: number[][] = [];
  /** _frame2 REMAPPED into the current clip's joint order, so the retarget can read it
   *  with the current clip's `idx`. Clips do NOT share a layout: a KAWAII clip is
   *  [22 body, fingers, fingertips, eyes] (64) while GestureLSM emits SMPL-X (55) with
   *  jaw+eyes at 22-24 — everything from index 22 on is shifted by 3 between them. Both
   *  the position blend and the prev-clip rotation used to index straight into _frame2,
   *  so across a KAWAII<->LSM seam every finger landmark came from a different joint and
   *  the wrist orientation was built from garbage (853-2396 deg/s^2, the biggest jerk
   *  measured anywhere). Remapping by NAME is layout-agnostic. */
  private _frame2m: number[][] = [];
  /** _frameNew / _frame2m after the torso clamp — what the retarget must actually read.
   *  The clamp pushes the WRIST out of the torso but leaves the knuckles alone, and the
   *  hand basis is wrist->middle-knuckle, so a clamped frame gives a very different wrist
   *  orientation. Reading the unclamped arrays in the crossfade path made the hand jump by
   *  the clamp's entire effect on the frame the fade ended: measured BOTH wrists snapping
   *  113-134 deg together at fade->1, with the hands crossed in front of the chest. */
  private _frameNewC: number[][] = [];
  private _frame2mC: number[][] = [];
  private _fadeMap: number[] = [];
  private _fadeMapFor: Clip | null = null;
  private _fadeMapFrom: Clip | null = null;

  // The NEW clip's frame (pre-position-blend), kept so FRAME bones can blend the prev/new
  // clips' ROTATION directly across a crossfade (see applyFrameBone).
  private _frameNew: number[][] = [];

  // The audio element the CURRENT gesture belongs to. The gesture is generated from
  // this sentence's audio, so gesture-time == audio-time: we drive `elapsed` from
  // audio.currentTime, so the gesture stays locked to the RIGHT speech (it arrives
  // ~2s late and simply catches up to the current audio position) and ends with it —
  // no "one sentence behind" lag, no bleed into the next line, no after-speech motion.
  private _gestAudio: HTMLAudioElement | null = null;

  private elapsed = 0; // seconds into the current clip

  private speed = 0.8; // calmer than 1.0 (co-speech peaks can be jerky)

  // Per-frame low-pass on the retargeted bones (0..1; lower = smoother, more lag).
  // The 30fps clip is linearly interpolated to 60fps and the co-speech shoulders/
  // arms carry high-frequency noise — at 0.35 the shoulders visibly shake and the
  // neck jitter drives the head so the hair over-reacts. 0.12 cuts the neck jitter
  // ~70% and the shoulder shake ~45% with no measurable loss of gesture reach (a
  // FIXED factor, not dt-scaled: a dt-scaled coeff wobbles with frame time and adds
  // its own jitter, and low fps interpolates the 30fps clip less so needs no boost).
  // Tunable live: __vrmMotion.smooth = 0.08..0.2.
  smooth = 0.12;

  // Extra smoothing multiplier for the HEAD + NECK specifically (lower = heavier low-pass).
  // The long twintails hang off the head, so a rough head/neck input drives their sideways
  // "プルプル"; these two are already gain-reduced, so smoothing them ~2x harder calms the
  // hair without making the arms/hands sluggish. Tunable: __vrmMotion.headSmoothMul.
  //
  // PER SOURCE now (authored 1.0 / lsm 0.5), because this multiplier is what stopped the
  // chest from following fast authored motion. One low-pass stage at 0.3*0.5 lags ~95ms, and
  // the chest sits TWO heavy-smoothed stages down (spine -> chest), so it trails by ~190ms.
  // Measured consequences: on the cartwheel the clip demands an inverted torso (up.y -1.0)
  // and the chest only reaches +0.27 — ~105 deg short at the ~500 deg/s the clip turns; and
  // at ordinary co-speech rates the same lag is the systematic 5.6-10.1 deg chest error seen
  // on every clip. The twintail "プルプル" this guards against comes from GestureLSM's noisy
  // output, not from hand-authored KAWAII animation, so only the lsm set needs it.
  headSmoothMul = 0.5;

  /** Per-bone smoothing slerp factor — the whole UPPER CHAIN gets the heavier low-pass.
   *  The twintails hang off the head's WORLD transform, which is the product of the whole
   *  chain spine->chest->neck->head. Speech-synced ~4Hz beat motion in the chest/spine
   *  (the head's PARENTS) moves the head world and the hair amplifies it into "プルプル" —
   *  so smoothing only head/neck can't help (the parents dominate). Heavy-smooth spine,
   *  chest, neck AND head; leave the arms/hands at the normal factor so the gesture reads. */
  private smoothFor(bone: string): number {
    return bone === "head" || bone === "neck" || bone === "chest" || bone === "spine"
      ? this.smooth * this.headSmoothMul
      : this.smooth;
  }

  // When a gesture starts FROM IDLE, seed the low-pass from the CURRENT pose (not a
  // hard reset) so the body eases from wherever it is into the new clip instead of
  // snapping to the new clip's frame 0 (the "フレーム飛び" at gesture transitions).
  // NEVER re-seed on a clip->clip handover: the rendered pose is already damped
  // toward restLocal (gainFor), so seeding from it and damping again DOUBLE-damps —
  // the neck/head/pelvis visibly collapsed toward idle for one frame at every
  // sentence seam (the "1フレームだけidleになる" hitch). Mid-gesture the smoothing
  // buffer is fresh every frame, so continuity needs no seed.
  private _seedSmooth = false;

  // Gesture damping toward the rest/idle pose (co-speech motion is full-body, so on a
  // planted upright avatar the raw gesture over-shrugs the shoulders and over-moves
  // the torso — "上半身だけ動いて変なポーズ"). Blend each bone's target back toward its
  // idle local by these gains: shoulders (clavicles) heavily (they cause the shrug),
  // torso/arms by the overall gain, fingers/head kept full. Live: __vrmMotion.*Gain.
  // NOTE: every gain below is now FULL (1 = play the clip as animated). They used to be
  // damped because the avatar was planted with locked legs, so the clip's weight shifts
  // had nowhere to go and came out as shrugs/twists; and because the KAWAII clips had no
  // finger data, so the "hand" landmarks were noise. Both reasons are gone (legs free,
  // fingers exported). `damp()` restores the old conservative set for A/B.
  shoulderGain = 1.0;   // was 0.55 — clavicle shrug

  gestureGain = 1.0;    // was 0.92 — overall upper body (spine/chest/neck/arms)

  fingerGain = 1.0;     // was 0.12 — finger curl

  handGain = 1.0;       // was 0.25 — wrist / palm facing

  private chestFrame = false; // drive the chest by full frame (set in adaptChains)

  private restLocal: Record<string, THREE.Quaternion> = {};

  private gestureSeq = 0;

  // Procedural ambient motion (no gesture clip): a gentle IDLE (breath + sway) when
  // silent, and a livelier volume-driven TALKING motion (arms/torso/head gesturing,
  // scaled by the speech energy) while speaking — so the body starts moving the
  // instant speech starts and never freezes in the gaps, with the GestureLSM clip
  // crossfading OVER it when it arrives (~1-2s later). Everything blends (never snaps).
  private idleT = 0;

  private idleBase: Record<string, THREE.Quaternion> = {};

  /** STANDING pose offset (no-clip only; idle clips normally own the body). Zeroed on
   *  request — plain relaxed standing like the Live2D Rem; the hand/skirt separation is the
   *  wrist clamp in torso-clamp.ts, which works on clip-driven poses too. */
  stand = { armZ: 0, elbowY: 0 };

  /** Live handle for the skirt band of the hand clamp (see torso-clamp.ts SKIRT). */
  skirtClamp = SKIRT;

  private static IDLE_BONES = [
    "spine", "chest", "neck", "head",
    "leftUpperArm", "rightUpperArm", "leftLowerArm", "rightLowerArm",
  ];

  // Talking state: the audio element + its per-slice volume envelope (same signal the
  // lip-sync uses). `_talkE` is the low-passed speech energy driving the gesture size.
  private _talking = false;

  private _talkAudio: HTMLAudioElement | null = null;

  private _talkVol: number[] = [];

  private _talkSlice = 20;

  private _talkE = 0;

  // Streaming (true-TTS-streaming) drive: a live speech-energy provider (0..1)
  // wired by the websocket handler from the gapless WebAudio player's analyser.
  // When set it takes precedence in currentVolume(). Cleared by stopTalking().
  private _talkLive: (() => number) | null = null;

  private _ambTarget = new THREE.Quaternion();

  private _ambIdentity = new THREE.Quaternion();

  // After a pose discontinuity (idle->gesture teleport, gesture->idle), re-seat the
  // spring bones for a couple of frames so the hair/skirt don't whip/explode.
  private pendingSpringInit = 0;

  private _holdT = 0; // time spent holding at the clip end waiting for the chained successor

  // scratch interpolated frame (avoid per-frame allocation)
  private _frame: number[][] = [];

  // smoothed per-bone local quaternion (low-passes the jerky retarget so the spring
  // bones get smooth input and can sway without being flung)
  private _smoothQ: Record<string, THREE.Quaternion> = {};

  // scratch objects (avoid per-frame allocation)
  private _e = new THREE.Euler();

  private _dq = new THREE.Quaternion();

  private _pq = new THREE.Quaternion();

  private _v0 = new THREE.Vector3();

  private _v1 = new THREE.Vector3();

  private _dir = new THREE.Vector3();

  private _delta = new THREE.Quaternion();

  // MMD "append/grant" (付与親) chains that the VRM export dropped. This model skins its
  // legs to a duplicate D chain (足DL/ひざDL/足首DL) that MMD drove by copying the base
  // bones' rotation; glTF has no such link, so the humanoid leg bones rotate a skeleton
  // that carries ZERO vertex weight and the legs render as sticks. Discovered in
  // captureRest, replayed every frame by syncAppendBones.
  private appendPairs: {
    src: THREE.Object3D; dst: THREE.Object3D;
    srcRestWInv: THREE.Quaternion; dstRestW: THREE.Quaternion;
  }[] = [];

  private _aq0 = new THREE.Quaternion();

  private _aq1 = new THREE.Quaternion();

  // scratch for the frame (hips/head) retarget
  private _up = new THREE.Vector3();

  private _right = new THREE.Vector3();

  private _fwd = new THREE.Vector3();

  private _m4 = new THREE.Matrix4();

  private _qf = new THREE.Quaternion();

  private _qt = new THREE.Quaternion();

  private _qA = new THREE.Quaternion();

  private _qtN = new THREE.Quaternion(); // FRAME-bone crossfade: new/prev clip targets

  private _qtP = new THREE.Quaternion();

  // --- Foot IK -------------------------------------------------------------------
  // OFF: the legs now follow the clip like every other limb (aim retarget), and the
  // pelvis translates with it, so crouches/weight-shifts/foot-lifts play as animated.
  // footIK=true instead PINS both ankles to fixed floor targets — no slide, but any
  // clip that actually moves its feet gets its legs fought by the IK (bent-over knees,
  // thighs through the skirt), which is why so many idle clips had to be rejected.
  // Read-only mirror of _ikW (the real, continuously blended weight) for diagnostics —
  // set __vrmMotion.authored.footIK / .lsm.footIK to change the behaviour per source.
  footIK = false;

  // Lower body FOLLOWS the motion now (footIK plants the ankles). lockLegs=true forced
  // the pelvis to its front REST while the upper body aimed off the CLIP's frame — the
  // mismatch inserted a spurious relative yaw ("上半身だけ正面を向かない"). With the legs
  // free, pelvis + torso share one frame, so no twist. Live: __vrmMotion.lockLegs = true.
  lockLegs = false;

  // genOnly mode fills the DEAD gaps (no generated clip, or a clip holding at its
  // end) with a GENTLE talking sway so the body doesn't freeze between the sparse
  // (~4s) generated gestures. Small on purpose. 0 = fully static gaps (old genOnly).
  // Live: __vrmMotion.genOnlyTalk.
  genOnlyTalk = 0.3;

  // Push the hands/elbows out of the torso so co-speech gestures don't sink into the
  // (clothed, thicker) body. Bigger than the SMPL body because Miku's arm FK lands the
  // hand short of the clamp target. 0 disables. Live: __vrmMotion.torsoRadius.
  torsoRadius = 0.21;

  pelvisRotGain = 1.0; // was 0.5 — pelvis lean/turn (the feet can move with it now)

  // Head + neck. Damping these kept the FACE toward the camera and stopped the long
  // twintails whipping at clip seams; at 1.0 the head turns/nods exactly as animated.
  // Tunable live: __vrmMotion.headGain / .neckGain (or damp() for the old set).
  headGain = 1.0;      // was 0.2

  neckGain = 1.0;      // was 0.18

  pelvisTransGain = 1.0; // was 0.5 — full weight-shift / crouch travel of the pelvis

  // --- PER-SOURCE gains -----------------------------------------------------------
  // The two clip sources need opposite treatment, and using one setting for both is
  // what made "unlock everything" break GestureLSM:
  //  • KAWAII/idle clips are HAND-ANIMATED — every channel is clean, so play them at
  //    1.0 and let the feet move as authored (footIK off).
  //  • GestureLSM is MODEL OUTPUT — its finger/palm joints are noisy (full follow
  //    twists the wrist and jitters the fingers), it shrugs the clavicles, turns the
  //    head off-camera and spreads/steps the legs. Those are exactly the artefacts the
  //    damping below was tuned to remove; the retarget itself is accurate (measured:
  //    0.9-3.9 deg per-bone agreement with the clip), so this is denoising, not a fix
  //    for a mis-mapping.
  // `smooth` (per-bone low-pass), `torso` (push hands out of the body) and `beat`
  // (scale the gesture by speech loudness) are ALSO motion-altering — they were built
  // for the noisy source, so authored clips get them off/loose too.
  // torso 0.21, was 0. At 0 clampHandsOutsideTorso returns immediately, so the hand/skirt
  // guard never ran on the AUTHORED (KAWAII) clips at all — which is every idle motion, and
  // the reason the hands went through the skirt and nothing I changed in the clamp had any
  // effect. Measured on idle43_handonhip: frames with a hand inside the cloth 35% -> 16%.
  authored = { shoulder: 1, gesture: 1, finger: 1, hand: 1, head: 1, neck: 1, pelvisRot: 1, pelvisTrans: 1, footIK: false, smooth: 0.3, headMul: 0.5, torso: 0.21, skirtOnly: true, beat: false };

  // beat=false on BOTH sources: RMS-driven amplitude scaling is a procedural layer riding
  // on top of a playing clip, and the body should be driven by the clip alone.
  // Live: __vrmMotion.lsm.beat = true to bring the loudness punch back.
  lsm = { shoulder: 0.55, gesture: 0.92, finger: 0.12, hand: 0.25, head: 0.2, neck: 0.18, pelvisRot: 0.5, pelvisTrans: 0.5, footIK: true, smooth: 0.12, headMul: 0.5, torso: 0.21, skirtOnly: false, beat: false };

  perSource = true; // false = use ONE set for both (see freeAll/dampAll)

  beatSync = true;  // scale the gesture with speech loudness (set per source)

  private _gainsFor: string | null = null;

  // Authored (KAWAII) clips ran for months with NO torso clamp (torso was 0) and their arms
  // looked right; enabling the full clamp for the skirt fix also started pushing their
  // ELBOWS (base profile ~0.208m at elbow height vs authored elbows at ~0.19m), which bent
  // the arms outward — the reported reversed-looking elbows. skirtOnly restricts the clamp
  // to the WRISTS + skirt band for authored clips, restoring the untouched elbows; LSM
  // keeps the full chest-crossing guard it always had.
  private _torsoSkirtOnly = true;

  // The two sets differ a LOT (finger 1 -> 0.12, hand 1 -> 0.25, footIK off -> on). Applying
  // that as a step the instant the clip changes put a hard discontinuity in the middle of
  // the crossfade: measured a 0.21m one-frame hand jump at the KAWAII->GestureLSM seam and a
  // 0.15m foot jump coming back. So the set RAMPS over the same window as the frame blend.
  private _gainFrom: Record<string, number> | null = null;

  private _gainMix = 1; // 1 = fully on _gainsFor

  private _gainRampDur = 0.4;

  private _ikW = 0; // continuous foot-IK weight: 0 = legs follow the clip, 1 = ankles planted

  private _beatW = 0;

  /** Retarget the gain set toward `src`, starting from whatever is live right now. */
  private _applyGains(src: "authored" | "lsm") {
    if (this._gainsFor === src) return;
    const first = this._gainsFor === null;
    this._gainsFor = src;
    const g = src === "lsm" ? this.lsm : this.authored;
    if (first) { // nothing to blend from on the very first clip
      this.shoulderGain = g.shoulder; this.gestureGain = g.gesture;
      this.fingerGain = g.finger; this.handGain = g.hand;
      this.headGain = g.head; this.neckGain = g.neck;
      this.pelvisRotGain = g.pelvisRot; this.pelvisTransGain = g.pelvisTrans;
      this.smooth = g.smooth; this.headSmoothMul = g.headMul; this.torsoRadius = g.torso;
      this._torsoSkirtOnly = g.skirtOnly;
      this._ikW = g.footIK ? 1 : 0; this._beatW = g.beat ? 1 : 0;
      this.footIK = g.footIK; this.beatSync = g.beat;
      this._gainMix = 1;
      return;
    }
    this._gainFrom = {
      shoulder: this.shoulderGain, gesture: this.gestureGain, finger: this.fingerGain,
      hand: this.handGain, head: this.headGain, neck: this.neckGain,
      pelvisRot: this.pelvisRotGain, pelvisTrans: this.pelvisTransGain,
      smooth: this.smooth, headMul: this.headSmoothMul,
      torso: this.torsoRadius, ik: this._ikW, beat: this._beatW,
    };
    this._torsoSkirtOnly = g.skirtOnly; // boolean: flips at the set change (radius ramps)
    this._gainMix = 0;
    this._gainRampDur = Math.max(0.2, this._fadeDur); // finish with the frame crossfade
  }

  /** Advance the gain ramp. Same smoothstep as the frame blend so pose and damping
   *  arrive together instead of the damping landing a whole crossfade early — and so a
   *  gain SET change (authored <-> lsm) cannot step the hand/finger scale at full rate
   *  on frame 1: that was the largest jerk measured anywhere (2396 deg/s^2 on
   *  rightHand at the first idle->GestureLSM handover). */
  private _tickGains(dt: number) {
    if (this._gainMix >= 1 || !this._gainFrom || !this._gainsFor) return;
    this._gainMix = Math.min(1, this._gainMix + dt / this._gainRampDur);
    const t = this._gainMix * this._gainMix * (3 - 2 * this._gainMix);
    const g = this._gainsFor === "lsm" ? this.lsm : this.authored;
    const f = this._gainFrom;
    const L = (a: number, b: number) => a + (b - a) * t;
    this.shoulderGain = L(f.shoulder, g.shoulder); this.gestureGain = L(f.gesture, g.gesture);
    this.fingerGain = L(f.finger, g.finger); this.handGain = L(f.hand, g.hand);
    this.headGain = L(f.head, g.head); this.neckGain = L(f.neck, g.neck);
    this.pelvisRotGain = L(f.pelvisRot, g.pelvisRot); this.pelvisTransGain = L(f.pelvisTrans, g.pelvisTrans);
    this.smooth = L(f.smooth, g.smooth); this.headSmoothMul = L(f.headMul, g.headMul);
    this.torsoRadius = L(f.torso, g.torso);
    this._ikW = L(f.ik, g.footIK ? 1 : 0);
    this._beatW = L(f.beat, g.beat ? 1 : 0);
    this.footIK = this._ikW > 0.5;   // for callers that still want a yes/no
    this.beatSync = this._beatW > 1e-3;
  }

  /** Force EVERYTHING (both sources) to play exactly as animated — no damping, free
   *  feet. Faithful, but GestureLSM's noisy channels show. Live: __vrmMotion.freeAll() */
  freeAll() {
    this.perSource = false; this._gainsFor = null;
    Object.assign(this.lsm, this.authored);
    this._applyGains("authored");
  }

  /** Force the conservative set on both (the planted-feet era). */
  dampAll() {
    this.perSource = false; this._gainsFor = null;
    Object.assign(this.authored, this.lsm);
    this._applyGains("lsm");
  }

  private footTarget: Record<string, THREE.Vector3> = {}; // planted ankle world targets

  private legLen: Record<string, number> = {}; // rest hip->knee / knee->ankle lengths

  private hipsRestPos = new THREE.Vector3(); // hips local position at rest

  // Floor contact. With the ankles no longer IK-pinned, nothing stopped a clip that
  // crouches or lifts a foot from pushing the foot THROUGH the ground (measured -5cm on
  // idle04_lookatfeet). Captured at rest = the lowest contact point when standing;
  // applyFrame lifts the hips by any shortfall. Live: __vrmMotion.floorClamp = false.
  private groundY = Infinity;

  floorClamp = true;

  private _lift = 0; // current smoothed lift applied to the hips (m)

  private smplPelvisRel = new THREE.Vector3(); // SMPL pelvis - feet-centroid at clip rest

  private smplPelvis0 = new THREE.Vector3(); // clip frame-0 pelvis (floor-normalized)

  private smplHipMid0 = new THREE.Vector3(); // clip frame-0 L/R hip-socket midpoint

  private _socketLocal = new THREE.Vector3(); // avatar hip-socket midpoint, hips-local

  private _hipsRestQ = new THREE.Quaternion(); // avatar hips world orientation at rest

  private _hasSocket = false;

  private _pvA = new THREE.Vector3(); // applyPelvisShift scratch

  private _pvB = new THREE.Vector3();

  private _pvC = new THREE.Vector3(); // planted-feet pelvis offset, blended against _ik0

  private _aimLeg: Record<string, THREE.Quaternion> = {}; // pre-IK leg pose, for the IK blend

  // Keep her on her spot: high-pass the clip's horizontal root travel (see applyPelvisShift).
  // OFF by default — clips play as authored. This was added while mis-diagnosing the stick
  // legs (the real cause was the dropped MMD append chain, see discoverAppendBones) and it
  // suppresses real authored travel, so it is opt-in: __vrmMotion.inPlace = true.
  inPlace = false;

  inPlaceRate = 0.02; // per frame; ~0.8s time constant at 60fps

  private _xzBase = new THREE.Vector3();

  private _ik0 = new THREE.Vector3();

  private _ik1 = new THREE.Vector3();

  private _ik2 = new THREE.Vector3();

  private _ik3 = new THREE.Vector3();

  setVRM(vrm: VRM) {
    this.vrm = vrm;
    this.clip = null;
    this.adaptChains();
    this.captureRest();
    this.applyIdlePose(); // arms-down rest instead of the T-pose
    // Capture every retargeted bone's idle local quat — the pose the gesture is
    // damped back toward (shoulders/torso) so it doesn't look contorted upright.
    this.restLocal = {};
    for (const b of ORDER) { const nb = this.node(b); if (nb) this.restLocal[b] = nb.quaternion.clone(); }
    for (const b of Object.keys(FRAME)) { const nb = this.node(b); if (nb) this.restLocal[b] = nb.quaternion.clone(); }
    // capture the resting local quaternions the idle sway oscillates around
    this.idleBase = {};
    for (const b of VRMMotionPlayer.IDLE_BONES) {
      const nb = this.node(b);
      if (nb) this.idleBase[b] = nb.quaternion.clone();
    }
  }

  clearVRM(vrm: VRM) {
    if (this.vrm === vrm) { this.vrm = null; this.clip = null; }
  }

  private node(bone: string): THREE.Object3D | null {
    return this.vrm ? this.vrm.humanoid.getNormalizedBoneNode(bone as never) : null;
  }

  /** How much of the retargeted rotation to keep for a bone (rest .. full). Damps
   *  the clavicle shrug hardest, the torso/arms by the overall gain, keeps fingers
   *  and legs full (the legs are already uprighted upstream). */
  private gainFor(bone: string): number {
    if (/Shoulder$/.test(bone)) return this.shoulderGain; // clavicles
    if (/(Proximal|Intermediate|Metacarpal)$/.test(bone)) return this.fingerGain; // fingers
    if (/(UpperLeg|LowerLeg|Foot)$/.test(bone)) return 1; // legs
    if (bone === "neck") return this.neckGain; // the neck swings the HEAD-world, which whips the twintails at seams
    return this.gestureGain; // spine, chest, upper/lower arm, hand
  }

  /** Adapt the retarget chains to THIS VRM's actual bones. Many stylized VRMs (Miku
   *  included) omit the optional `upperChest` bone. Without adapting, `CHILD.chest`
   *  points at the missing upperChest, so captureRest can't get the chest's rest
   *  direction and the chest is NEVER retargeted — the upper torso stays rigid while
   *  the sample's chest bends/turns with speech ("胴/肩がサンプルと違う"). Re-point the
   *  chest straight at the neck so it absorbs the whole Spine2->Neck bend. Same idea
   *  for a missing toes bone (then the foot is the leaf). */
  private adaptChains() {
    const has = (b: string) => !!this.node(b);
    if (has("upperChest")) { AIM.chest = ["Spine2", "Spine3"]; CHILD.chest = "upperChest"; }
    else { AIM.chest = ["Spine2", "Neck"]; CHILD.chest = "neck"; }
    // No upperChest -> the single chest bone spans the whole upper torso; drive its
    // FULL orientation (roll from the shoulder line) so it doesn't tilt. VRMs WITH
    // upperChest keep the plain aim (the frame's Spine2->Neck up would over-reach).
    this.chestFrame = !has("upperChest");
  }

  /** Capture the VRM T-pose per-bone world quaternion + rest child direction. */
  private captureRest() {
    if (!this.vrm) return;
    this.vrm.scene.updateMatrixWorld(true);
    this.restWQ = {}; this.restDir = {}; this.restPWInv = {};
    for (const b of ORDER) {
      const nb = this.node(b); const c = CHILD[b];
      // A chain LEAF (the finger distals) has no humanoid child to measure its rest
      // direction against, so fall back to the raw rig's own tip bone (人指先L etc.).
      // restDir is a WORLD direction and both rigs coincide at rest, so this is exact.
      const nc = (c ? this.node(c) : null) ?? this.rawTipChild(b);
      if (!nb || !nc) continue;
      const q = new THREE.Quaternion(); nb.getWorldQuaternion(q); this.restWQ[b] = q;
      const pw = new THREE.Quaternion();
      nb.parent!.getWorldQuaternion(pw);
      this.restPWInv[b] = pw.invert();
      const pb = new THREE.Vector3(); nb.getWorldPosition(pb);
      const pc = new THREE.Vector3(); nc.getWorldPosition(pc);
      this.restDir[b] = pc.sub(pb).normalize();
    }
    // Frame bones (hips/head): capture Miku's rest world orientation.
    this.frameRestMiku = {};
    for (const b of Object.keys(FRAME)) {
      const nb = this.node(b);
      if (nb) { const q = new THREE.Quaternion(); nb.getWorldQuaternion(q); this.frameRestMiku[b] = q; }
    }
    // Foot-IK rest data: planted ankle targets, leg segment lengths, hips rest pos.
    const hipsN = this.node("hips");
    if (hipsN) this.hipsRestPos.copy(hipsN.position);
    // Hip-socket midpoint in hips-LOCAL space + the rest hips orientation. The pelvis
    // rotates about the HIP AXIS in real bodies, so pelvis translation must keep the
    // SOCKETS (not the pelvis centre) on the clip's track — otherwise a bow swings the
    // sockets, the ankles dig in, and the floor clamp bobs the whole body upward.
    if (hipsN) {
      const ul = this.node("leftUpperLeg"); const ur = this.node("rightUpperLeg");
      if (ul && ur) {
        const a = new THREE.Vector3(); ul.getWorldPosition(a);
        const c = new THREE.Vector3(); ur.getWorldPosition(c);
        const hw = new THREE.Vector3(); hipsN.getWorldPosition(hw);
        const hq = new THREE.Quaternion(); hipsN.getWorldQuaternion(hq);
        this._socketLocal.copy(a).add(c).multiplyScalar(0.5).sub(hw)
          .applyQuaternion(hq.clone().invert());
        this._hipsRestQ.copy(hq);
        this._hasSocket = true;
      }
    }
    (["left", "right"] as const).forEach((side) => {
      const up = this.node(`${side}UpperLeg`); const kn = this.node(`${side}LowerLeg`); const ft = this.node(`${side}Foot`);
      if (!up || !kn || !ft) return;
      const pu = new THREE.Vector3(); up.getWorldPosition(pu);
      const pk = new THREE.Vector3(); kn.getWorldPosition(pk);
      const pf = new THREE.Vector3(); ft.getWorldPosition(pf);
      this.legLen[`${side}Upper`] = pu.distanceTo(pk);
      this.legLen[`${side}Lower`] = pk.distanceTo(pf);
      this.footTarget[side] = pf.clone(); // planted ankle target (rest world pos)
      const fq = new THREE.Quaternion(); ft.getWorldQuaternion(fq);
      this.restWQ[`${side}Foot`] = fq; // keep the foot flat regardless of missing toes
      // lowest contact point at rest = the floor for this rig (toe tip if it has toes)
      const to = this.node(`${side}Toes`);
      const pt = new THREE.Vector3();
      (to ?? ft).getWorldPosition(pt);
      this.groundY = Math.min(this.groundY, pt.y, pf.y);
    });
    // Hand ABSOLUTE-frame correction: K = R_miku_rest_anat^-1 * Q_miku_rest, so
    // target = SMPL_frame * K puts the SMPL hand orientation onto Miku's hand bone.
    this.handK = {};
    (["left", "right"] as const).forEach((side) => {
      const hand = `${side}Hand`; const qmr = this.frameRestMiku[hand];
      const w = this.node(hand); const mid = this.node(`${side}MiddleProximal`);
      const ix = this.node(`${side}IndexProximal`); const li = this.node(`${side}LittleProximal`);
      if (!qmr || !w || !mid || !ix || !li) return;
      const wp = new THREE.Vector3(); w.getWorldPosition(wp);
      const mp = new THREE.Vector3(); mid.getWorldPosition(mp);
      const ip = new THREE.Vector3(); ix.getWorldPosition(ip);
      const lp = new THREE.Vector3(); li.getWorldPosition(lp);
      const rAnat = new THREE.Quaternion();
      if (this.buildFrameQuat(mp.sub(wp), lp.sub(ip), rAnat)) {
        this.handK[hand] = rAnat.invert().multiply(qmr);
      }
    });
    this.discoverAppendBones();
    this.discoverTwistBones();
    this.discoverRatioAppends();
    this._restClip = null; // force re-capture of the SMPL rest frame for the next clip
  }

  /** The raw rig's first child bone of `bone` — used as a rest-direction reference for
   *  chain leaves that the VRM humanoid doesn't extend (finger distals -> 人指先L etc.). */
  private rawTipChild(bone: string): THREE.Object3D | null {
    const rb = this.vrm?.humanoid.getRawBoneNode(bone as never) as THREE.Object3D | undefined;
    if (!rb) return null;
    for (const c of rb.children) {
      const cb = c as unknown as { isBone?: boolean };
      if (cb.isBone || c.type === "Bone") return c;
    }
    return null;
  }

  /** MMD twist-distribution bones (腕捩1/2/3, 手捩1/2/3). Same dropped-付与親 story as the
   *  D chain, but these sit on a SIDE branch off 腕/ひじ that only carries skin weight, so
   *  the mesh still follows the limb — what was lost is the twist being SPREAD along it.
   *  Measured: 腕捩3L carries 528 weight (the biggest static weight after 下半身) right at
   *  the elbow and never rotated, so a hard forearm twist sheared the skin there.
   *  master = the bone whose local rotation holds the twist, axis = the limb's own
   *  direction in the helper's parent space, ratio = how far along the limb it sits. */
  private twistPairs: {
    node: THREE.Object3D; master: THREE.Object3D;
    axisH: THREE.Vector3;  // limb axis in the HELPER's parent space (where we apply)
    axisM: THREE.Vector3;  // limb axis in the MASTER's parent space (where we measure)
    ratio: number; rest: THREE.Quaternion; masterRestInv: THREE.Quaternion;
  }[] = [];

  private _tq0 = new THREE.Quaternion();

  /** How long a finished clip may HOLD its end pose before it loop-crossfades back into
   *  itself. Holding keeps the handover seamless (the next chained clip blends from the
   *  held pose) but the body is visually still. Measured mid-speech stalls are 0.03-0.7s,
   *  all of which sailed under the old 1.8s threshold and so were never covered.
   *  Paired A/B on the same prompt (2 runs each, normalised by speech time):
   *    1.8s (old) -> never fired
   *    0.5s       -> frozen 30.7% / 31.0% of speech
   *    0.12s      -> frozen  5.2% /  5.5%,  seam p95 unchanged (0.034-0.041m either way)
   *  The occasional 0.2-0.6m single-frame spikes show up at BOTH settings, so they are
   *  clip switches, not the wrap. Live: __vrmMotion.holdWrapAfter. */
  // 0.8s: was 0.12 originally, briefly 2.5. The wrap is the fallback for "the clip ran
  // out and no successor is here yet". At 0.12 it fired inside EVERY GestureLSM tail gap
  // (clips cap ~4.3s, sentences run ~5s) and snapped toward the clip's near-neutral opening
  // = the mid-speech snap toward the standing pose. At 2.5 the tails froze instead (no
  // motion through the end of the reply). 0.8 reads as a natural gesture hold, then motion
  // resumes — and the wrap goes to the _matchStart frame now, so resuming is a soft
  // crossfade, not a snap.
  holdWrapAfter = 0.8;

  /** Procedural sway around a clip's HELD end pose, for inter-clip gaps. OFF: the body is
   *  driven by the clips alone. Measured hold time is 0s on both pipelines now that clips
   *  chain, so this costs nothing today — but if long gaps come back the body will simply
   *  freeze on the held pose. Live: __vrmMotion.holdBreath = true. */
  holdBreath = false;

  /** Max twist (rad) a helper may distribute. Live: __vrmMotion.twistMax = 0. */
  twistMax = Math.PI / 2;

  /** How much of the chain's TWIST is handed to the aim-driven parent instead of piling up
   *  in the absolutely-oriented joint below it (see twistShareInto).
   *
   *  forearmTwist: pronation happens in the FOREARM in a real arm — the wrist joint itself
   *  barely rotates axially — yet the retarget left the elbow's roll free and dumped the
   *  lot on the wrist (measured up to 204 deg: "手首もねじれすぎる").
   *  spineTwist: lumbar rotation is real but small next to thoracic (~5-13 deg vs ~35),
   *  and the spine was getting exactly none of it, so the belly stayed square to the front
   *  while the single chest bone cranked all the way around — very visible on a look-back
   *  ("おなかのねじれも壊れてる").
   *  Live: __vrmMotion.forearmTwistShare / .spineTwistShare (0 = previous behaviour). */
  //
  // DEFAULT 0 = the pre-existing behaviour. Both were validated only in quaternion
  // algebra (the child's world orientation provably does not move), never on the avatar,
  // and turning them on regressed the look: the belly over-twisted, the upper body stopped
  // facing front and the wrists read wrong. Re-enable ONE at a time and judge on screen.
  //
  // forearmTwistShare RE-ENABLED (0.85), now that the baseline roll is sane. This is MMD
  // parity: smpl_to_mmd.py's ARM_TWIST puts forearm roll (pronation/supination) into ひじ,
  // and THIS rig is built for that split — 腕捩1-3 spread the FOREARM's twist along the
  // upper-arm skin and 手捩1-3 spread the hand's residual. Measured before the change on
  // lookathands: the hand's local twist hit 110 deg (the wrist candy-wrap the user zoomed
  // in on), past the helpers' ±90 clamp. 0.85 leaves a natural ~15% in the wrist.
  // The earlier attempt at 0.7 looked wrong only because the pre-parentAim baseline fed it
  // garbage twist. spineTwistShare stays 0: parentAim already carries pelvis roll up the
  // spine, and the belly reads correctly since.
  forearmTwistShare = 0.85;
  spineTwistShare = 0;
  private _twChild = new THREE.Quaternion();
  private _twRel = new THREE.Quaternion();
  private _twQ = new THREE.Quaternion();
  private _twAxis = new THREE.Vector3();

  /** Ratio-scaled FULL-rotation append links (as opposed to twist-only). Read off the
   *  source PMX: ひじ補助 <- ひじ at 0.58, and it shares ひじ's parent and position, so the
   *  local rotation transfers directly. It fills the inner elbow crease; without it the
   *  bone sat frozen with ~81 skin weight on it. */
  private ratioPairs: {
    node: THREE.Object3D; master: THREE.Object3D; ratio: number;
    rest: THREE.Quaternion; masterRestInv: THREE.Quaternion;
  }[] = [];

  private discoverRatioAppends() {
    this.ratioPairs = [];
    if (!this.vrm) return;
    // name -> [master name, ratio]. Only links whose master this pipeline actually drives
    // are listed: the PMX also grants 肩C<-肩P, 腰キャンセル<-腰 and 目<-両目, but 肩P/腰 are
    // never driven here (no-ops) and the eyes are driven directly by the lookAt applier.
    // PMX spelling (side as a PREFIX) — resolveBone maps it onto whatever the export
    // renamed it to. ひじ補助 itself carries no skin weight; its CHILD ひじ補助+ does (81),
    // and that rides along through ordinary parenting, so driving the parent is correct.
    const SPEC: Record<string, [string, number]> = {
      左ひじ補助: ["左ひじ", 0.58], 右ひじ補助: ["右ひじ", 0.58],
    };
    const byName: Record<string, THREE.Object3D> = {};
    this.vrm.scene.traverse((o) => { if (o.name) byName[o.name] = o; });
    for (const [dst, [src, ratio]] of Object.entries(SPEC)) {
      const node = this.resolveBone(byName, dst); const master = this.resolveBone(byName, src);
      // the local-rotation transfer is only valid while they share a parent frame
      if (!node || !master || node.parent !== master.parent) continue;
      this.ratioPairs.push({
        node, master, ratio,
        rest: node.quaternion.clone(),
        masterRestInv: master.quaternion.clone().invert(),
      });
    }
    if (this.ratioPairs.length) {
      console.info("[vrm-motion] restored ratio appends:",
        this.ratioPairs.map((r) => `${r.node.name}=${r.ratio}x${r.master.name}`).join(" "));
    } else {
      console.warn("[vrm-motion] no ratio appends found (ひじ補助) — the inner elbow crease "
        + "will not fill");
    }
  }

  private syncRatioAppends() {
    for (const r of this.ratioPairs) {
      this._tq0.copy(r.master.quaternion).multiply(r.masterRestInv); // delta since rest
      this._aq0.identity().slerp(this._tq0, r.ratio);                // scale the whole delta
      r.node.quaternion.copy(this._aq0).multiply(r.rest);
      r.node.updateWorldMatrix(false, false);
    }
  }

  /** Find a rig bone by its ORIGINAL PMX name.
   *
   *  The export renames every sided bone: the PMX carries the side as a PREFIX (右腕捩1,
   *  左ひじ補助) and the VRM ends up with it as a suffix (腕捩1.R, ひじ補助.L). Every lookup
   *  here used to match the PMX spelling verbatim, so it found NOTHING — the twist helpers
   *  and the elbow-assist append were silently never driven even though they carry real
   *  skin weight (腕捩/手捩 ~1000 across the two arms, ひじ補助+ 81). That is why a hard
   *  forearm twist still sheared at one joint instead of spreading down the arm. */
  private resolveBone(byName: Record<string, THREE.Object3D>, pmxName: string) {
    if (byName[pmxName]) return byName[pmxName];
    const side = pmxName[0] === "右" ? "R" : pmxName[0] === "左" ? "L" : "";
    if (!side) return null;
    const base = pmxName.slice(1);
    for (const sep of [".", "_", " ", ""]) {
      const n = byName[`${base}${sep}${side}`];
      if (n) return n;
    }
    return null;
  }

  private discoverTwistBones() {
    this.twistPairs = [];
    if (!this.vrm) return;
    const raw = (b: string) => this.vrm!.humanoid.getRawBoneNode(b as never) as THREE.Object3D | null;
    const byName: Record<string, THREE.Object3D> = {};
    this.vrm.scene.traverse((o) => { if (o.name) byName[o.name] = o; });
    // Straight from the PMX: 腕捩1/2/3 <- 腕捩 and 手捩1/2/3 <- 手捩, ratios 0.25/0.50/0.75.
    // Neither MASTER is driven by this pipeline (the retarget puts the forearm twist in ひじ
    // and the hand twist in 手首), so each group is sourced from the joint that actually
    // holds that twist instead.
    const groups = [
      { jp: "腕捩", limb: (s: string) => `${s}UpperArm`, master: (s: string) => `${s}LowerArm` },
      { jp: "手捩", limb: (s: string) => `${s}LowerArm`, master: (s: string) => `${s}Hand` },
    ];
    const wanted: { node: THREE.Object3D; limb: string; master: string; ratio: number }[] = [];
    for (const g of groups) {
      for (const [jpSide, side] of [["左", "left"], ["右", "right"]] as const) {
        for (let n = 1; n <= 3; n += 1) {
          const node = this.resolveBone(byName, `${jpSide}${g.jp}${n}`);
          if (node) wanted.push({ node, limb: g.limb(side), master: g.master(side), ratio: n / 4 });
        }
      }
    }
    const pos = new THREE.Vector3(); const cpos = new THREE.Vector3();
    this.vrm.scene.updateMatrixWorld(true);
    wanted.forEach(({ node: o, limb: limbName, master: masterName, ratio }) => {
      const limb = raw(limbName); const master = raw(masterName);
      if (!limb || !master || !o.parent || !master.parent) return;
      limb.getWorldPosition(pos); master.getWorldPosition(cpos);
      const dir = cpos.sub(pos);
      if (dir.lengthSq() < 1e-8) return;
      dir.normalize();
      // The helper and the master can sit in DIFFERENT parent frames (腕捩N hangs off
      // 腕 while ひじ hangs off 腕捩), so express the limb axis separately in each.
      const hq = new THREE.Quaternion(); o.parent.getWorldQuaternion(hq);
      const mq = new THREE.Quaternion(); master.parent.getWorldQuaternion(mq);
      this.twistPairs.push({
        node: o,
        master,
        axisH: dir.clone().applyQuaternion(hq.invert()).normalize(),
        axisM: dir.clone().applyQuaternion(mq.invert()).normalize(),
        ratio,
        rest: o.quaternion.clone(),
        masterRestInv: master.quaternion.clone().invert(),
      });
    });
    if (this.twistPairs.length) {
      console.info("[vrm-motion] restored MMD twist distribution:",
        this.twistPairs.map((t) => `${t.node.name}=${t.ratio}x${t.master.name}`).join(" "));
    } else {
      // Silence here is what hid the naming mismatch for so long.
      console.warn("[vrm-motion] no MMD twist helpers found (腕捩N/手捩N) — a forearm twist "
        + "will shear at one joint instead of spreading along the arm");
    }
  }

  /** Give each twist helper `ratio` of its master's twist about the limb axis. Only the
   *  TWIST component is taken — the bend must not leak in or the helper would swing the
   *  skin sideways. */
  /** Hand part of a chain's TWIST to its aim-driven PARENT instead of letting it all pile
   *  up in the absolutely-oriented joint below.
   *
   *  An aim is setFromUnitVectors — the MINIMAL rotation between two directions — so it
   *  carries no roll about the limb axis whatsoever. Every bit of the chain's twist
   *  therefore lands on the one joint that IS given a full orientation: the wrist (from the
   *  palm frame) and, on a VRM without upperChest, the single chest bone (from the shoulder
   *  line). The result is a wrist twisted past anything anatomical, and an abdomen that
   *  stays square to the front while the chest turns.
   *
   *  Because the child is oriented ABSOLUTELY, anything the parent takes is subtracted from
   *  the child's LOCAL twist for free: the palm and the shoulder line still land exactly
   *  where the clip put them, and rotating the parent about its own axis does not move the
   *  child's position either. Only the skin between them changes — which is the whole point.
   *
   *  `parentDesired` is the parent's world target, modified in place. `this._dir` must still
   *  hold the parent's (normalized) aim direction — that is the twist axis. */
  private twistShareInto(
    parentDesired: THREE.Quaternion,
    childBone: string,
    frame: number[][],
    idx: Record<string, number>,
    share: number,
  ): void {
    if (!(share > 0)) return;
    const cfg = FRAME[childBone];
    if (!cfg) return;
    if (!this.quatFromFrame(frame, idx, cfg, this._twChild, childBone)) return;
    const k = this.handK[childBone];
    if (k) {
      this._twChild.multiply(k); // hands: absolute palm
    } else {
      const rs = this.frameRestSmpl[childBone]; const rm = this.frameRestMiku[childBone];
      if (!rs || !rm) return;
      this._twChild.multiply(this._qA.copy(rs).invert()).multiply(rm);
    }
    this._twAxis.copy(this._dir); // limb axis, world (SMPL and VRM share axes here)
    // World rotation taking the parent's orientation to the child's, then its component
    // about the limb axis (swing-twist: the rotation vector's projection on the axis).
    this._twRel.copy(this._twChild).multiply(this._qA.copy(parentDesired).invert());
    const d = this._twRel.x * this._twAxis.x + this._twRel.y * this._twAxis.y
      + this._twRel.z * this._twAxis.z;
    let ang = 2 * Math.atan2(d, this._twRel.w);
    if (ang > Math.PI) ang -= 2 * Math.PI;
    else if (ang < -Math.PI) ang += 2 * Math.PI;
    ang = Math.max(-Math.PI, Math.min(Math.PI, ang)) * share;
    this._twQ.setFromAxisAngle(this._twAxis, ang);
    parentDesired.premultiply(this._twQ);
  }

  private syncTwistBones() {
    for (const t of this.twistPairs) {
      // Rotation the master has picked up SINCE rest, in its parent's frame.
      this._tq0.copy(t.master.quaternion).multiply(t.masterRestInv);
      // swing-twist decomposition: the twist half-angle is the rotation vector's
      // component along the limb axis. Taking only this keeps the bend out.
      const d = this._tq0.x * t.axisM.x + this._tq0.y * t.axisM.y + this._tq0.z * t.axisM.z;
      let ang = 2 * Math.atan2(d, this._tq0.w);
      // Wrap to (-pi, pi] then clamp. The aim retarget leaves the elbow's ROLL free, so
      // the hand can absorb a huge apparent twist (measured 204 deg) — distributing 0.75
      // of that put 153 deg on 手捩3 and would shear the forearm worse than not having it.
      // A real forearm tops out near a quarter turn, so bound it there.
      if (ang > Math.PI) ang -= 2 * Math.PI;
      else if (ang < -Math.PI) ang += 2 * Math.PI;
      ang = Math.max(-this.twistMax, Math.min(this.twistMax, ang));
      this._tq0.setFromAxisAngle(t.axisH, ang * t.ratio);
      t.node.quaternion.copy(this._tq0).multiply(t.rest);
      t.node.updateWorldMatrix(false, false);
    }
  }

  /** Rebuild the MMD append (付与親) links the VRM export dropped.
   *
   *  Miku's rig carries TWO parallel leg chains: 足/ひざ/足首 (what the VRM humanoid maps,
   *  vertex weight 0) and 足D/ひざD/足首D (where every leg vertex actually is). In MMD the D
   *  chain copies the base chain's rotation via 付与親; glTF/VRM has no equivalent, so the
   *  link vanished on export and the legs became rigid no matter what the retarget did —
   *  measured: knee bone 5..98 deg, mesh dead straight.
   *
   *  Detection is name-agnostic: any humanoid bone whose whole subtree carries no skin
   *  weight is dead, and its live twin is the weighted non-humanoid bone sitting at the
   *  same rest position. Rest orientations of the two chains differ (up to 171 deg here),
   *  so syncAppendBones transplants the WORLD rotation delta, not the local quaternion. */
  private discoverAppendBones() {
    this.appendPairs = [];
    if (!this.vrm) return;
    const weight = new Map<THREE.Object3D, number>();
    this.vrm.scene.traverse((o: THREE.Object3D) => {
      const sm = o as THREE.SkinnedMesh;
      if (!sm.isSkinnedMesh) return;
      const gi = sm.geometry.attributes.skinIndex; const gw = sm.geometry.attributes.skinWeight;
      if (!gi || !gw) return;
      for (let v = 0; v < gi.count; v += 1) {
        for (let k = 0; k < 4; k += 1) {
          const w = gw.getComponent(v, k);
          if (w <= 1e-4) continue;
          const bn = sm.skeleton.bones[gi.getComponent(v, k)];
          if (bn) weight.set(bn, (weight.get(bn) ?? 0) + w);
        }
      }
    });
    if (!weight.size) return;
    const raw = (b: string) => this.vrm!.humanoid.getRawBoneNode(b as never) as THREE.Object3D | null;
    const humanoidNodes = new Set<THREE.Object3D>();
    const names: string[] = Object.keys((this.vrm.humanoid as unknown as { humanBones: object }).humanBones ?? {});
    for (const n of names) { const r = raw(n); if (r) humanoidNodes.add(r); }
    // Candidate hosts: weighted bones the humanoid does not own.
    const hosts: { node: THREE.Object3D; pos: THREE.Vector3 }[] = [];
    for (const [bn, w] of weight) {
      if (w <= 1 || humanoidNodes.has(bn)) continue;
      const p = new THREE.Vector3(); bn.getWorldPosition(p);
      hosts.push({ node: bn, pos: p });
    }
    const wp = new THREE.Vector3();
    for (const n of names) {
      const src = raw(n);
      if (!src) continue;
      let sub = 0;
      src.traverse((o) => { sub += weight.get(o) ?? 0; });
      if (sub > 1) continue; // this bone (or its subtree) really does deform the mesh
      src.getWorldPosition(wp);
      let best: THREE.Object3D | null = null; let bestD = 5e-3; // 5mm: a duplicate, not a neighbour
      for (const h of hosts) {
        const d = h.pos.distanceTo(wp);
        if (d < bestD) { bestD = d; best = h.node; }
      }
      if (!best || this.appendPairs.some((p) => p.dst === best)) continue;
      const sq = new THREE.Quaternion(); src.getWorldQuaternion(sq);
      const dq = new THREE.Quaternion(); best.getWorldQuaternion(dq);
      this.appendPairs.push({ src, dst: best, srcRestWInv: sq.invert(), dstRestW: dq });
    }
    // Parent-first: each destination's local rotation is derived from its parent's world
    // orientation, so an ancestor in the same chain has to be solved before its child.
    const depth = (o: THREE.Object3D) => { let d = 0; let c: THREE.Object3D | null = o; while (c) { d += 1; c = c.parent; } return d; };
    this.appendPairs.sort((a, b) => depth(a.dst) - depth(b.dst));
    if (this.appendPairs.length) {
      console.info("[vrm-motion] restored MMD append bones:",
        this.appendPairs.map((p) => `${p.src.name}->${p.dst.name}`).join(" "));
    }
  }

  /** Replay every MMD rig link the glTF export dropped: the append (付与親) duplicates and
   *  the twist distribution. Call once per frame, after posing, before the springs. */
  syncAppendBones() {
    if (!this.vrm || (!this.appendPairs.length && !this.twistPairs.length && !this.ratioPairs.length)) return;
    this.vrm.humanoid.update(); // make sure the raw source bones hold this frame's pose
    this.syncTwistBones();
    this.syncRatioAppends();
    for (const p of this.appendPairs) {
      p.src.updateWorldMatrix(true, false);
      p.src.getWorldQuaternion(this._aq0);
      this._aq0.multiply(p.srcRestWInv).multiply(p.dstRestW); // (srcW * srcRest^-1) * dstRest
      if (p.dst.parent) {
        p.dst.parent.getWorldQuaternion(this._aq1);
        p.dst.quaternion.copy(this._aq1.invert()).multiply(this._aq0);
      } else {
        p.dst.quaternion.copy(this._aq0);
      }
      p.dst.updateWorldMatrix(false, false);
    }
    this.vrm.scene.updateMatrixWorld(true);
  }

  /** Build the world orientation of an anatomical frame (up + right landmark axes)
   *  from one SMPL frame, into `out`. Returns false if the landmarks are missing. */
  /** Full orientation of a bone from THREE consecutive joints: its own segment plus the next
   *  one, whose cross product is the bend axis that pins the otherwise-free roll. */
  private quatFromPlane(
    frame: number[][],
    idx: Record<string, number>,
    tri: [string, string, string],
    out: THREE.Quaternion,
  ): boolean {
    const a = idx[tri[0]]; const b = idx[tri[1]]; const c = idx[tri[2]];
    if (a === undefined || b === undefined || c === undefined) return false;
    const n = frame.length;
    if (a >= n || b >= n || c >= n) return false;
    this._up.fromArray(frame[b]).sub(this._v0.fromArray(frame[a]));
    this._right.fromArray(frame[c]).sub(this._v1.fromArray(frame[b]));
    this._fwd.crossVectors(this._up, this._right);   // bend axis: perpendicular to both
    // How BENT the joint is: sin of the angle between the two segments. A straight limb
    // carries no roll information at all, so the caller fades its correction out with this.
    const lu = this._up.length(); const lr = this._right.length();
    this._planeSin = (lu > 1e-6 && lr > 1e-6) ? this._fwd.length() / (lu * lr) : 0;
    // Straight limb -> no plane -> no roll information. Keep the aim result instead of
    // inventing one from noise.
    if (this._fwd.lengthSq() < 1e-9) return false;
    this._right.copy(this._fwd);
    return this.buildFrameQuat(this._up, this._right, out);
  }

  private quatFromFrame(
    frame: number[][],
    idx: Record<string, number>,
    cfg: { up: [string, string]; right: [string, string] },
    out: THREE.Quaternion,
    /** Pass the FRAME bone to also reject a frame whose landmark axes have COLLAPSED
     *  relative to this bone's rest lengths (see FRAME_MIN_LEN_FRAC). Omitted while
     *  capturing the rest frame itself, which is the reference. */
    bone?: string,
  ): boolean {
    const ua = idx[cfg.up[0]]; const ub = idx[cfg.up[1]];
    const ra = idx[cfg.right[0]]; const rb = idx[cfg.right[1]];
    if (ua === undefined || ub === undefined || ra === undefined || rb === undefined) return false;
    // The resolved indices come from `idx` (the CURRENT clip's joint layout) but `frame`
    // may be the OUTGOING clip's (crossfade): KAWAII has 22 joints, GestureLSM 55, so a
    // finger index into the 22-row frame is out of range -> undefined. Skip (caller then
    // uses the new clip's rotation alone) instead of fromArray(undefined) crashing.
    const flen = frame.length;
    if (ua >= flen || ub >= flen || ra >= flen || rb >= flen) return false;
    this._up.fromArray(frame[ub]).sub(this._v0.fromArray(frame[ua]));
    this._right.fromArray(frame[rb]).sub(this._v1.fromArray(frame[ra]));
    if (bone !== undefined) {
      const rest = this.frameRestLen[bone];
      if (rest) {
        const min = rest * FRAME_MIN_LEN_FRAC;
        if (this._up.length() < min || this._right.length() < min) return false;
      }
    }
    return this.buildFrameQuat(this._up, this._right, out);
  }

  /** Orthonormal world orientation from an up + right vector (both get mutated). */
  private buildFrameQuat(up: THREE.Vector3, right: THREE.Vector3, out: THREE.Quaternion): boolean {
    if (up.lengthSq() < 1e-8 || right.lengthSq() < 1e-8) return false;
    up.normalize();
    right.normalize();
    this._fwd.crossVectors(right, up);
    // |right x up| = sin(angle between the two landmark axes) now that BOTH are unit, so
    // this is a real conditioning test. It used to run on an un-normalized `right`, making
    // the threshold |right|^2 sin^2 < 1e-8 — for a ~6cm knuckle span that only rejects
    // below sin 0.0017 (0.1 deg), i.e. never. Any near-parallel pair still produced a
    // basis, and since `right` is then re-orthogonalized against `up` out of what little
    // is left, it FLIPS sign as the axes cross: measured the wrist snapping 83-91 deg for
    // two frames and back, mid-clip, whenever a KAWAII hand pose curled the fingers enough
    // to line L_Index1->L_Pinky1 up with L_Wrist->L_Middle1. Reject instead — the caller
    // keeps the previous orientation, and holding a wrist for two frames is invisible
    // where a 90 deg flip is not.
    if (this._fwd.lengthSq() < FRAME_MIN_SIN * FRAME_MIN_SIN) return false;
    this._fwd.normalize();
    right.crossVectors(up, this._fwd).normalize(); // re-orthogonalize
    this._m4.makeBasis(right, up, this._fwd);
    out.setFromRotationMatrix(this._m4);
    return true;
  }

  /** Capture the SMPL anatomical frames at the clip's first (rest) frame — the
   *  reference the per-frame delta is measured against. */
  private captureSmplRest() {
    if (!this.clip) return;
    const f0 = this.clip.joints[0];
    this._frameRestSmplPrev = this.frameRestSmpl; // keep old rest to blend across the crossfade
    this.frameRestSmpl = {};
    this.frameRestLen = {};
    this.planeRestSmpl = {};
    this.planeRestSin = {};
    for (const b of Object.keys(PLANE)) {
      const q = new THREE.Quaternion();
      if (this.quatFromPlane(f0, this.clip.idx, PLANE[b], q)) {
        this.planeRestSmpl[b] = q;
        this.planeRestSin[b] = this._planeSin;
      }
    }
    for (const b of Object.keys(FRAME)) {
      const q = new THREE.Quaternion();
      if (this.quatFromFrame(f0, this.clip.idx, FRAME[b], q)) this.frameRestSmpl[b] = q;
      // Reference scale for the collapsed-axis test (FRAME_MIN_LEN_FRAC).
      const cfg = FRAME[b]; const n = this.clip.idx;
      const ua = n[cfg.up[0]]; const ub = n[cfg.up[1]];
      const ra = n[cfg.right[0]]; const rb = n[cfg.right[1]];
      if (ua !== undefined && ub !== undefined && ra !== undefined && rb !== undefined
          && ua < f0.length && ub < f0.length && ra < f0.length && rb < f0.length) {
        const uL = Math.hypot(f0[ub][0] - f0[ua][0], f0[ub][1] - f0[ua][1], f0[ub][2] - f0[ua][2]);
        const rL = Math.hypot(f0[rb][0] - f0[ra][0], f0[rb][1] - f0[ra][1], f0[rb][2] - f0[ra][2]);
        const m = Math.min(uL, rL);
        if (m > 1e-4) this.frameRestLen[b] = m;
      }
    }
    // Foot-IK weight-shift reference: SMPL pelvis relative to the feet centroid at rest.
    const n = this.clip.idx;
    if (n.Pelvis !== undefined && n.L_Foot !== undefined && n.R_Foot !== undefined) {
      const p = f0[n.Pelvis]; const lf = f0[n.L_Foot]; const rf = f0[n.R_Foot];
      this.smplPelvisRel.set(p[0] - (lf[0] + rf[0]) / 2, p[1] - (lf[1] + rf[1]) / 2, p[2] - (lf[2] + rf[2]) / 2);
    }
    // Free-leg pelvis reference: the clip's own frame-0 pelvis + hip-socket midpoint
    // (clips are floor-normalized at conversion, so these are absolute over y=0).
    if (n.Pelvis !== undefined) {
      const p = f0[n.Pelvis];
      this.smplPelvis0.set(p[0], p[1], p[2]);
    } else {
      this.smplPelvis0.set(0, 0, 0);
    }
    if (n.L_Hip !== undefined && n.R_Hip !== undefined) {
      const a = f0[n.L_Hip]; const c = f0[n.R_Hip];
      this.smplHipMid0.set((a[0] + c[0]) / 2, (a[1] + c[1]) / 2, (a[2] + c[2]) / 2);
    } else {
      this.smplHipMid0.copy(this.smplPelvis0);
    }
    this._restClip = this.clip;
  }

  /** Retarget one frame bone (hips/head): apply the SMPL frame's rotation-from-rest
   *  onto Miku's rest orientation, smoothed. */
  private applyFrameBone(bone: string, frame: number[][], idx: Record<string, number>) {
    const nb = this.node(bone); const cfg = FRAME[bone];
    if (!nb || !cfg) return;
    if (!this.quatFromFrame(frame, idx, cfg, this._qf, bone)) return; // _qf = SMPL frame now (world)
    const k = this.handK[bone];
    if (k) {
      // hands: ABSOLUTE — match the sample's palm, retargeted from the position-BLENDED
      // frame. Giving the hands the same prev/new rotation SLERP the other frame bones use
      // was tried and reverted: it made the wrist pop 70-130 deg for a frame or two in the
      // MIDDLE of a fade. The two clip families disagree about the palm by a very large
      // angle, so any momentary disagreement about which side of that arc to take lands as
      // a huge step, whereas the blended-landmark route can only ever move as far as the
      // landmarks did. (The jerk figure that motivated the slerp — 2396 "deg/s^2" — was an
      // artifact of dividing by a short frame time; measured in degrees off a
      // constant-velocity extrapolation, this path is the quieter of the two.)
      this._qt.copy(this._qf).multiply(k);
    } else {
      const rm = this.frameRestMiku[bone];
      if (!rm) return;
      if (this._fadeT < 1 && this._prevClip && this._frameNewC.length) {
        // CROSSFADE: blend the retargeted ROTATION of the prev + new clips. Blending the
        // joint POSITIONS then retargeting is NON-LINEAR (quatFromFrame builds the rotation
        // from landmark vectors), so it produced a head-rotation SPIKE at the seam that
        // whipped the twintails sideways. Retarget each clip against ITS OWN rest, slerp.
        const t = this._fadeW(); // same curve as the position blend
        const rsNew = this.frameRestSmpl[bone];
        if (!rsNew || !this.quatFromFrame(this._frameNewC, idx, cfg, this._qf, bone)) return;
        this._qtN.copy(this._qf).multiply(this._qA.copy(rsNew).invert()).multiply(rm);
        const rsPrev = this._frameRestSmplPrev[bone] || rsNew;
        if (this.quatFromFrame(this._frame2mC, idx, cfg, this._qf, bone)) {
          this._qtP.copy(this._qf).multiply(this._qA.copy(rsPrev).invert()).multiply(rm);
          this._qt.copy(this._qtP).slerp(this._qtN, t);
        } else {
          this._qt.copy(this._qtN);
        }
      } else {
        const rs = this.frameRestSmpl[bone];
        if (!rs) return;
        this._qt.copy(this._qf).multiply(this._qA.copy(rs).invert()).multiply(rm); // relative delta
      }
    }
    // Slew-limit the WORLD target (see FRAME_MAX_SLEW). Done on the world target, before
    // the parent-relative conversion, so a fast-moving parent cannot be mistaken for a
    // flip in this bone.
    const prevT = this._qtPrev[bone];
    if (prevT) {
      const ang = 2 * Math.acos(Math.min(1, Math.abs(prevT.dot(this._qt))));
      const maxStep = FRAME_MAX_SLEW * this._dt;
      if (ang > maxStep && ang > 1e-6) this._qt.copy(prevT).slerp(this._qt, maxStep / ang);
    }
    (this._qtPrev[bone] ??= new THREE.Quaternion()).copy(this._qt);
    nb.parent!.getWorldQuaternion(this._pq);
    this._pq.invert().multiply(this._qt); // -> local target
    let sq = this._smoothQ[bone];
    if (!sq) { sq = this._pq.clone(); this._smoothQ[bone] = sq; }
    else sq.slerp(this._pq, this.smoothFor(bone));
    // Damp toward rest: pelvis by its own gain, chest by the gesture gain, and the WRIST
    // by handGain (keeps the hand relaxed instead of twisting to the noisy finger-palm).
    const g = bone === "hips" ? this.pelvisRotGain : bone === "chest" ? this.gestureGain
      : bone === "head" ? this.headGain
      : (bone === "leftHand" || bone === "rightHand") ? this.handGain : 1;
    const rl = this.restLocal[bone];
    if (g >= 0.999 || !rl) nb.quaternion.copy(sq);
    else nb.quaternion.copy(rl).slerp(sq, g);
    nb.updateMatrixWorld(true);
  }

  /** Shift the hips by the SMPL weight-shift (pelvis relative to the feet centroid),
   *  damped — the visible "load one leg / lean" motion that IK turns into knee bend. */
  private applyPelvisShift(frame: number[][], idx: Record<string, number>) {
    const hipsN = this.node("hips");
    if (!hipsN || idx.Pelvis === undefined) return;
    const p = frame[idx.Pelvis];
    // The two anchors disagree by several cm, so switching between them on a boolean put a
    // step in the hips at every source change. Compute both and cross-fade by _ikW.
    const w = this._ikW;
    let planted = false;
    if (w > 1e-3 && idx.L_Foot !== undefined && idx.R_Foot !== undefined) {
      // Planted feet (GestureLSM): the feet midpoint is a STABLE anchor, so measure the
      // weight shift relative to it. (Absolute pelvis would fight the pinned ankles.)
      const lf = frame[idx.L_Foot]; const rf = frame[idx.R_Foot];
      this._pvC.set(p[0] - (lf[0] + rf[0]) / 2, p[1] - (lf[1] + rf[1]) / 2, p[2] - (lf[2] + rf[2]) / 2);
      this._pvC.sub(this.smplPelvisRel).multiplyScalar(this.pelvisTransGain);
      planted = true;
    }
    let free = false;
    if (w < 0.999) {
      // FREE legs (authored clips): the feet MOVE, so a feet-relative anchor injects
      // garbage the moment a foot lifts — in a handstand it flips sign (+0.86 -> -0.87)
      // and shoved the hips ~1.7m underground ("handstandがおかしい/おなかより下が
      // おかしい"). Clips are floor-normalized at conversion (y=0), so track them
      // ABSOLUTELY — but track the HIP-SOCKET midpoint, not the pelvis centre: the
      // pelvis rotates about the hip axis, so with centre-tracking a mere bow swings
      // the sockets, digs the ankles into the floor and the clamp bobs the body (+11cm
      // measured). Socket-tracking keeps the legs rooted while the pelvis pitches.
      if (this.smplPelvis0.y > 1e-3 && this.smplHipMid0.y > 1e-3) {
      // Scale by SOCKET height, not the hips-node height: on MMD-converted rigs the
      // hips node sits well BELOW the anatomical hip line (this Miku: node 0.68, real
      // sockets 0.94 — the 下半身 tail), and 0.68/0.936 under-scaled every pelvis move.
      const sockY = this.hipsRestPos.y + (this._hasSocket ? this._socketLocal.y : 0);
      const s = (sockY / this.smplHipMid0.y) * this.pelvisTransGain;
      const hm = this.smplHipMid0;
      const cl = idx.L_Hip !== undefined && idx.R_Hip !== undefined
        ? this._pvA.set(
          ((frame[idx.L_Hip][0] + frame[idx.R_Hip][0]) / 2 - hm.x) * s,
          ((frame[idx.L_Hip][1] + frame[idx.R_Hip][1]) / 2 - hm.y) * s,
          ((frame[idx.L_Hip][2] + frame[idx.R_Hip][2]) / 2 - hm.z) * s)
        : this._pvA.set((p[0] - this.smplPelvis0.x) * s, (p[1] - this.smplPelvis0.y) * s, (p[2] - this.smplPelvis0.z) * s);
      this._ik0.copy(cl);
      if (this.inPlace) {
        // DANCE IN PLACE. A dance clip walks the root around (dance04 travels 0.66m):
        // on a fixed camera she slides out of shot and her feet skate along the floor,
        // which reads as "legs dragged like sticks" even though they articulate exactly
        // as authored (measured: thigh swing 71 deg vs the clip's 72). High-pass the
        // horizontal component — quick weight shifts survive, sustained travel decays —
        // and the steps register as steps because she stays centred. Vertical is kept
        // in full so crouches/bows/handstands still work. Live: __vrmMotion.inPlace.
        this._xzBase.x += (this._ik0.x - this._xzBase.x) * this.inPlaceRate;
        this._xzBase.z += (this._ik0.z - this._xzBase.z) * this.inPlaceRate;
        this._ik0.x -= this._xzBase.x;
        this._ik0.z -= this._xzBase.z;
      }
      if (this._hasSocket) {
        // rotation compensation: hipsPos = rest + Δclip + (R_rest·m − R_now·m), so the
        // avatar's socket midpoint (hipsPos + R_now·m) lands exactly on the clip track
        this._pvB.copy(this._socketLocal).applyQuaternion(this._hipsRestQ);
        this._ik0.add(this._pvB);
        this._pvB.copy(this._socketLocal).applyQuaternion(hipsN.quaternion);
        this._ik0.sub(this._pvB);
      }
      free = true;
      }
    }
    if (planted && free) this._ik0.lerp(this._pvC, w);
    else if (planted) this._ik0.copy(this._pvC);
    else if (!free) return;
    this._ik1.copy(this.hipsRestPos).add(this._ik0); // target hips position
    // Ease toward the target so the weight-shift doesn't jump at gesture transitions.
    hipsN.position.lerp(this._ik1, this.smooth);
    hipsN.updateMatrixWorld(true);
  }

  /** Aim a bone's rest child-direction at a world direction (reuses restDir/restWQ). */
  private aimBoneTo(bone: string, worldDir: THREE.Vector3) {
    const nb = this.node(bone); const rd = this.restDir[bone]; const rq = this.restWQ[bone];
    if (!nb || !rd || !rq) return;
    this._delta.setFromUnitVectors(rd, worldDir);
    this._qt.copy(this._delta).multiply(rq); // desired world quat
    nb.parent!.getWorldQuaternion(this._pq);
    this._pq.invert().multiply(this._qt); // -> local
    nb.quaternion.copy(this._pq);
  }

  /** 2-bone leg IK: bend hip+knee so the ankle reaches its planted target, knee
   *  pointing forward. Foot kept flat. Call after the pelvis + body are posed. */
  private solveLegIK(side: "left" | "right") {
    const hipN = this.node(`${side}UpperLeg`); const kneeN = this.node(`${side}LowerLeg`);
    const footN = this.node(`${side}Foot`); const target = this.footTarget[side];
    const l1 = this.legLen[`${side}Upper`]; const l2 = this.legLen[`${side}Lower`];
    if (!hipN || !kneeN || !footN || !target || !l1 || !l2) return;
    const h = this._ik0; hipN.getWorldPosition(h);
    const toT = this._ik1.subVectors(target, h);
    let d = toT.length();
    d = Math.min(l1 + l2 - 1e-3, Math.max(Math.abs(l1 - l2) + 1e-3, d));
    toT.normalize();
    const a = (l1 * l1 - l2 * l2 + d * d) / (2 * d);
    const hgt = Math.sqrt(Math.max(0, l1 * l1 - a * a));
    const perp = this._ik2.set(0, 0, 1); // knee bends forward (+Z)
    perp.addScaledVector(toT, -perp.dot(toT));
    if (perp.lengthSq() < 1e-6) perp.set(0, 0, 1);
    perp.normalize();
    const knee = this._ik3.copy(h).addScaledVector(toT, a).addScaledVector(perp, hgt);
    this.aimBoneTo(`${side}UpperLeg`, this._ik1.subVectors(knee, h).normalize());
    hipN.updateMatrixWorld(true);
    this.aimBoneTo(`${side}LowerLeg`, this._ik2.subVectors(target, knee).normalize());
    kneeN.updateMatrixWorld(true);
    footN.parent!.getWorldQuaternion(this._pq); // keep the foot flat (rest world orient)
    this._pq.invert().multiply(this.restWQ[`${side}Foot`]);
    footN.quaternion.copy(this._pq);
    footN.updateMatrixWorld(true);
  }

  /** Rotate the upper arms down so idle isn't a T-pose (looks natural at rest). */
  private applyIdlePose() {
    const down = (bone: string, sign: number) => {
      const nb = this.node(bone);
      if (nb) nb.quaternion.setFromAxisAngle(new THREE.Vector3(0, 0, 1), sign * 1.25);
    };
    down("leftUpperArm", -1); // +Z rotation brings the left arm down
    down("rightUpperArm", 1);
    if (this.vrm) { this.vrm.humanoid.update(); }
  }

  cancelGesture() {
    this.gestureSeq += 1;
    // Growth loops key off their OWN epoch, not gestureSeq — see _extendPartial.
    this._growEpoch += 1;
    this._growing.clear();
  }

  /**
   * Poll /motion/clip-smpl/{id} until the SMPL joints are ready (200), then play
   * them forward once (0.8x) as a co-speech gesture, holding the last frame.
   */
  playGesture(url: string, opts: { tries?: number; intervalMs?: number; audio?: HTMLAudioElement } = {}) {
    if (!this.vrm) return;
    const tries = opts.tries ?? 8;
    const intervalMs = opts.intervalMs ?? 1000;
    const audio = opts.audio ?? null; // sentence audio to lock the gesture to
    const gseq = ++this.gestureSeq;
    let attempt = 0;
    const poll = () => {
      if (gseq !== this.gestureSeq || !this.vrm) return;
      const ctrl = new AbortController();
      const to = setTimeout(() => ctrl.abort(), 4000);
      fetch(url, { signal: ctrl.signal })
        .then((r) => {
          if (r.status === 200) return r.json();
          if (r.status === 202) return null;
          throw new Error(String(r.status));
        })
        .then((data) => {
          if (gseq !== this.gestureSeq) return;
          // Drop a gesture whose sentence audio already ENDED: it would otherwise
          // play entirely during the NEXT sentence (a visible content/timing
          // mismatch). The procedural talking layer covers the gap. (B + C)
          if (audio && (audio.ended || (audio.duration && audio.currentTime > audio.duration - 0.25))) return;
          if (data && data.joints && data.joint_names) {
            const idx: Record<string, number> = {};
            data.joint_names.forEach((n: string, i: number) => { idx[n] = i; });
            const clipObj: Clip = { names: data.joint_names, idx, joints: data.joints, fps: data.fps || 30, loop: !!data.loop, lsm: !!data.gesturelsm };
            const fromIdle = !this.clip; // seed only when idle was driving the pose
            this._stats.applies += 1;
            this._prepareClip(clipObj, data);
            this._beginFade(); // crossfade from the outgoing clip (no seam stop)
            this.clip = clipObj;
            this._gestAudio = audio; // lock the gesture to this sentence's audio time
            // Land at the frame _prepareClip CHOSE (continuity frame for a chained clip,
            // pose-matched frame otherwise) — NOT at currentTime past it. Adding the audio
            // clock here defeated the choice: LSM clips land 0.8-1.4s after their sentence
            // starts, so playback began ~1s past the frame picked to match the body, and
            // the fade had to cover that pose gap in 0.2-0.45s = the jolt at every
            // GestureLSM seam. The rate-slew in update() re-syncs to the clock gradually.
            this.elapsed = clipObj.pad ?? 0;
            // NO spring reset here: sb.reset() snaps the hair to its REST direction (this
            // model's rest floats outward = the "跳ね上がる" jump at motion start). Heavy
            // drag + center-space sim now damp the transition instead.
            this._seedSmooth = fromIdle; // ease from the current pose (no snap to frame 0)
            // TEXT-STREAMING (ActionPlan): the clip arrived PARTIAL (~0.35s after
            // the sentence text) — keep extending it in place while it plays.
            if (data.partial) this._extendPartial(url, data);
          } else if (++attempt < tries) {
            setTimeout(poll, intervalMs);
          }
        })
        .catch(() => { /* give up quietly; idle pose holds */ })
        .finally(() => clearTimeout(to));
    };
    poll();
  }

  /** GestureLSM co-speech UPGRADE: poll `url` and crossfade it OVER the currently
   *  playing (KAWAII) clip via setClip the moment it lands. Unlike playGesture this
   *  does NOT claim gestureSeq, so it can never cancel the primary's in-flight poll
   *  (that race left a sentence with no motion at all). It aborts itself if a NEWER
   *  gesture claims the body (seq changed: next sentence / interrupt) or its own
   *  sentence audio already ended (stale). Fast poll = minimal catch-up latency. */
  upgradeGesture(url: string, opts: { tries?: number; intervalMs?: number; audio?: HTMLAudioElement } = {}) {
    if (!this.vrm) return;
    const tries = opts.tries ?? 160;
    const intervalMs = opts.intervalMs ?? 150;
    const audio = opts.audio ?? null;
    const seq = this.gestureSeq; // passive snapshot — do NOT bump
    let attempt = 0;
    const poll = () => {
      if (seq !== this.gestureSeq || !this.vrm) return; // superseded
      if (audio && (audio.ended || (audio.duration && audio.currentTime > audio.duration - 0.25))) return; // stale
      const ctrl = new AbortController();
      const to = setTimeout(() => ctrl.abort(), 4000);
      fetch(url, { signal: ctrl.signal })
        .then((r) => (r.status === 200 ? r.json() : null))
        .then((data) => {
          if (seq !== this.gestureSeq) return;
          if (data && data.joints && data.joint_names) this.setClip(data, audio);
          else if (++attempt < tries) setTimeout(poll, intervalMs);
        })
        .catch(() => { if (++attempt < tries) setTimeout(poll, intervalMs); })
        .finally(() => clearTimeout(to));
    };
    poll();
  }

  /** TEXT-STREAMING clips arrive partial (first latents ~0.35s in) and GROW while
   *  the sentence plays. Poll the same endpoint and append the NEW frames IN PLACE:
   *  every consumer (the prefetch map, the playing clip) holds a reference to
   *  payload.joints, so the live gesture extends seamlessly until the backend marks
   *  it complete (partial:false) — generation runs ~4x faster than playback, so the
   *  playhead never catches the writer. */
  private _extendPartial(url: string, payload: any, maxPolls = 40, intervalMs = 250) {
    if (!payload?.partial || !Array.isArray(payload.joints)) return;
    // Two guards this loop used to lack, both of which cost the NEXT turn its gesture:
    //
    // 1. A cancel epoch. This loop had no abort check at all, so an interrupt could not stop
    //    it — it kept polling for up to maxPolls*intervalMs (40 x 250ms = 10s) per sentence,
    //    for sentences already cut. Measured on a barge-in: 516 x 202 against
    //    /motion/clip-smpl in ~50s, with FOUR dead ids polled in the same frame; the gateway
    //    is a single slot, so the fresh turn's clip starved behind that traffic
    //    ("話しかけられたときモーションが止まってしまう").
    //    It must NOT key off gestureSeq: fetchClip bumps that for every SENTENCE, so doing so
    //    killed the growth of the clip that was still playing the moment the next sentence
    //    started fetching — a text-streamed GestureLSM clip then froze at its first ~0.35s of
    //    frames, which reads as "GestureLSM stopped working". _growEpoch moves only on an
    //    actual cancel (interrupt / turn end), which is exactly the condition that should
    //    abandon a growing clip.
    // 2. one loop per url. _extendPartial is called from BOTH fetchClip and playGesture's
    //    poll, so the same clip could get two growth loops appending to the same array
    //    (the measurement shows the same id fetched twice in one frame).
    if (this._growing.has(url)) return;
    this._growing.add(url);
    const epoch = this._growEpoch; // only a cancel moves this
    let polls = 0;
    const done = () => { this._growing.delete(url); };
    const step = () => {
      if (polls++ >= maxPolls || payload.partial === false) return done();
      if (epoch !== this._growEpoch || !this.vrm) return done(); // cancelled / interrupted
      const ctrl = new AbortController();
      const to = setTimeout(() => ctrl.abort(), 4000);
      fetch(url, { signal: ctrl.signal })
        .then((r) => (r.status === 200 ? r.json() : null))
        .then((d) => {
          if (epoch !== this._growEpoch) return done();
          if (d?.joints && d.joints.length > payload.joints.length) {
            for (let i = payload.joints.length; i < d.joints.length; i++) payload.joints.push(d.joints[i]);
          }
          if (d && !d.partial) { payload.partial = false; return done(); } // complete
          if (d === null) return done(); // entry consumed elsewhere / gone — stop
          setTimeout(step, intervalMs);
        })
        .catch(() => { done(); /* network blip: stop extending; the clip holds */ })
        .finally(() => clearTimeout(to));
    };
    setTimeout(step, intervalMs);
  }

  /**
   * BUFFER for full sync: await /motion/clip-smpl/{id} until it's ready (200) or the
   * timeout, returning the clip data (or null on timeout/interrupt). The caller waits
   * for this BEFORE starting the audio, so the gesture plays synced from frame 0. Only
   * the reply's first sentence actually waits — later sentences generate ahead (during
   * the earlier ones) and this returns immediately.
   */
  async fetchClip(url: string, timeoutMs = 4000): Promise<any | null> {
    if (!this.vrm) return null;
    const gseq = ++this.gestureSeq; // claim the slot; interrupt/newer call invalidates
    const deadline = performance.now() + timeoutMs;
    while (performance.now() < deadline) {
      if (gseq !== this.gestureSeq || !this.vrm) return null; // superseded / interrupted
      const pollStart = performance.now();
      try {
        const ctrl = new AbortController();
        const to = setTimeout(() => ctrl.abort(), 4000);
        const r = await fetch(url, { signal: ctrl.signal });
        clearTimeout(to);
        if (r.status === 200) {
          const d = await r.json();
          if (gseq !== this.gestureSeq) return null;
          if (d && d.joints && d.joint_names) {
            if (d.partial) this._extendPartial(url, d); // grows while it plays
            return d;
          }
        }
      } catch { /* transient; retry until the deadline */ }
      // Pace the polls. /motion/clip-smpl long-polls (~1s) once the id is registered,
      // but 202s INSTANTLY for an id it hasn't seen yet — so without a sleep we flood
      // the same event loop that is generating the gesture. Measured: dropping this to
      // 15ms made TTFA WORSE (3.9s -> 4.8-7.4s). Keep it.
      const held = performance.now() - pollStart;
      if (held < 250) await new Promise((res) => setTimeout(res, 250 - held));
    }
    return null; // timed out — caller plays anyway and the gesture catches up
  }

  /** Play an already-fetched clip locked to its audio (synced from frame 0). */
  setClip(data: any, audio: HTMLAudioElement | null) {
    this._stats.applies += 1;
    if (!this.vrm || !data?.joints || !data?.joint_names) return;
    if (audio && audio.ended) return; // stale: its sentence already finished (B + C)
    const idx: Record<string, number> = {};
    data.joint_names.forEach((n: string, i: number) => { idx[n] = i; });
    const clipObj: Clip = { names: data.joint_names, idx, joints: data.joints, fps: data.fps || 30, loop: !!data.loop, idle: !!data.idle, lsm: !!data.gesturelsm };
    const fromIdle = !this.clip; // seed only when idle was driving the pose
    this._prepareClip(clipObj, data);
    this._beginFade(); // crossfade from the outgoing clip (no seam stop)
    this.clip = clipObj;
    this._gestAudio = audio;
    // Same as the poll path: start at the chosen frame, let the rate-slew catch up.
    this.elapsed = clipObj.pad ?? 0;
    // no spring reset (it snaps hair to the floating rest pose = jump-up); see poll() note
    this._seedSmooth = fromIdle;
    this._xzBase.set(0, 0, 0); // new clip: restart the in-place high-pass at its origin
  }

  /** Decide how a fresh clip joins the ongoing motion. CHAINED clips (gateway seeded
   *  them with the previous clip's latents) start at seed_overlap-1: that frame IS the
   *  previous end pose, so the motion genuinely continues — no pose search needed and
   *  the fade is short. Unchained clips (chain broken: first line of a reply, gateway
   *  fallback, interrupt) fall back to start-pose matching + adaptive fade.
   *  Runs BEFORE this.clip is swapped (matching compares via the OLD clip's indices). */
  private _prepareClip(clipObj: Clip, data: any) {
    this._holdT = 0;
    const overlap = Number(data.seed_overlap) || 0;
    if (data.chained && this.clip && this._frame.length) {
      // Seeded from the previous clip's latents -> the motion CONTINUES it. Start at the
      // replay frame (overlap-1) if the gateway gave an explicit overlap, else at frame 0
      // (seed-continuity form, overlap=0). Either way DON'T _matchStart: that offsets the
      // playhead (desyncs the audio) and re-introduces the seam we're avoiding. A short
      // fade covers the tiny residual so the seam is smooth AND stays voice-synced.
      clipObj.chained = true;
      clipObj.pad = overlap > 0 ? Math.max(0, overlap - 1) / clipObj.fps : 0;
      // overlap=0 (seed-continuity): frame 0 is CLOSE to the prior end but not exact, so
      // a slightly longer fade smooths the residual. overlap>0 is an exact replay -> snappy.
      // (A "verify the chain by pose distance" experiment lived here briefly — it compared
      // the wrong frame, because setClip used to add audio.currentTime on top of the pad it
      // validated. With the landing fixed to the pad itself, plain trust is correct again.)
      this._fadeDur = overlap > 0 ? 0.2 : 0.45;
      return;
    }
    const match = this._matchStart(clipObj);
    clipObj.pad = match.pad;
    if (clipObj.idle) {
      // Idle clips are unrelated poses with NO audio to stay in sync with, so take the
      // long way round: a slow fade is what makes one idle animation melt into the next
      // instead of stepping. (Speech clips must stay tight — see below.)
      this._fadeDur = 1.0;
      return;
    }
    // GestureLSM clips DON'T chain on the public gateway (no seed_out), so every
    // sentence seam is a genuine pose CHANGE — _matchStart picks the closest start
    // frame, and the fade must be LONG ENOUGH to blend the residual travel or the jump
    // reads as "継ぎ目でがくがく". Both playheads keep advancing through the fade, so a
    // longer fade smooths the seam WITHOUT lagging/desyncing the motion vs the voice.
    this._fadeDur = Math.min(0.7, Math.max(0.35, 0.25 + match.dist * 2.2));
  }

  // Clips pre-fetched by id the moment their payload arrives (while the PRIOR line is
  // still playing), so they're in hand when this line plays — no 500ms-cap miss -> loop.
  /** urls with a live partial-growth loop, so the same clip is never grown twice. */
  private _growing = new Set<string>();
  /** Bumped ONLY by cancelGesture (interrupt / turn end). Growth loops abort on it. */
  private _growEpoch = 0;

  private _prefetched = new Map<string, unknown>();

  /** Start pre-fetching a co-speech clip (poll /motion/clip-smpl until ready) and cache
   *  it by id. Called from the websocket handler on payload arrival — the ~1.8s the clip
   *  takes overlaps the prior line's playback, so it's ready by the time this line plays. */
  prefetch(url: string, id: string, tries = 16, intervalMs = 500) {
    if (!id || this._prefetched.has(id)) return;
    if (this._prefetched.size > 8) this._prefetched.clear(); // bound (stale ids)
    this._prefetched.set(id, null); // in-flight marker
    let attempt = 0;
    const poll = () => {
      const ctrl = new AbortController();
      const to = setTimeout(() => ctrl.abort(), 4000);
      fetch(url, { signal: ctrl.signal })
        .then((r) => { if (r.status === 200) return r.json(); if (r.status === 202) return null; throw new Error(String(r.status)); })
        .then((d) => {
          if (d && d.joints && d.joint_names) {
            this._prefetched.set(id, d);
            // Partial (text-streaming): keep growing the stored object — takePrefetched
            // hands out the same reference, so a clip applied mid-growth extends live.
            if (d.partial) this._extendPartial(url, d);
          }
          else if (++attempt < tries) setTimeout(poll, intervalMs);
          else this._prefetched.delete(id);
        })
        .catch(() => { this._prefetched.delete(id); })
        .finally(() => clearTimeout(to));
    };
    poll();
  }

  /** Take (and remove) a ready pre-fetched clip for `id`, or null if not ready yet. */
  takePrefetched(id: string): unknown | null {
    const d = this._prefetched.get(id);
    if (d) { this._prefetched.delete(id); return d; }
    return null;
  }

  /** End the gesture. NO hard reset — updateIdle blends every bone (and the legs/hips)
   *  back toward the ambient pose, so gesture->talk/idle eases instead of snapping. */
  stop() {
    this.clip = null; this._prevClip = null; this._fadeT = 1; this._gestAudio = null;
    this._turnOver = false;
    // Let the body settle for a beat before the between-turns idle cycle picks up, so
    // the turn doesn't end straight into an idle animation.
    this._idleNextAt = performance.now() + 900;
    this._idleSuspended = false;
  }

  /** BARGE-IN. Like endTurn, not like stop(): the clip STAYS on the body.
   *
   *  stop() clears it, and measured on a real barge-in that costs two artifacts. The clear
   *  itself lurches (344 deg/s in one frame as the body snaps toward the ambient pose), and
   *  then nothing drives the body until the answer's own gesture lands — 1.83s of clip=none
   *  during which she STARTS SPEAKING at 1.2s and the procedural sway decays to 1.7 deg/s.
   *  That dead stretch is the "返答する瞬間一瞬モーションがなくなる".
   *
   *  Keeping the clip means she free-runs the gesture she was already making (natural: she
   *  was cut off mid-gesture), holds it with the hold-breathing, and the new sentence's clip
   *  crossfades over it. Idle is allowed back in quickly too, so a slow answer still has
   *  something moving instead of a mannequin. */
  interrupt() {
    this._gestAudio = null;   // the audio it was chasing is gone
    this._turnOver = true;    // idle may take over even though a clip is still set
    this._idleSuspended = false;
    this._idleNextAt = performance.now() + 250; // much shorter than stop()'s 900ms settle
  }

  /** Normal end of a reply. Unlike stop() this KEEPS the last co-speech clip on the body
   *  (holding, with the hold-breathing) instead of dropping it — clearing it snapped the
   *  pose toward the rest pose in ~0.3s, which is the ugly talk->idle seam. The first
   *  idle clip then CROSSFADES out of the pose she actually ended on. */
  endTurn() {
    this._gestAudio = null;   // free-run: no audio clock to chase
    this._turnOver = true;    // idle may take over even though a clip is still set
    this._idleSuspended = false;
    this._idleNextAt = performance.now() + 700;
  }

  isActive() { return !!this.clip; }

  // --- BETWEEN-TURNS IDLE ---------------------------------------------------------
  // Standing still with only the procedural breath/sway reads as a mannequin. When
  // nothing else is driving the body, play a random KAWAII Idle clip, then another
  // after a short beat. Any speech clip simply crossfades over it (setClip/playGesture),
  // and idle never starts while a clip is active — so it can't fight co-speech motion.
  idleClips = true;           // master switch (live: __vrmMotion.idleClips = false)
  idleGapMin = 1.2;           // seconds of plain breathing between idle clips
  idleGapMax = 4.0;
  private _idleNextAt = 0;    // performance.now() ms; 0 = "as soon as we're idle"
  private _idleFetching = false;
  private _idleSuspended = false;
  private _turnOver = false;  // reply finished: the held clip may be replaced by idle

  /** Stop the idle cycle (a turn started); resumes on its own once the body is free. */
  suspendIdle(sec = 1.5) {
    this._idleSuspended = true;
    this._turnOver = false; // a reply is live again
    this._idleNextAt = performance.now() + sec * 1000;
  }

  private _idleTick() {
    if (!this.idleClips || !this.vrm || this._talking || this._idleFetching) return;
    // A clip is still on the body only blocks idle while the reply is LIVE; once the
    // turn is over the idle clip crossfades out of that held pose (smooth handover).
    if (this.clip && !this._turnOver) return;
    const now = performance.now();
    if (this._idleSuspended) { // wait out the post-turn settle before resuming
      if (now < this._idleNextAt) return;
      this._idleSuspended = false;
    }
    if (now < this._idleNextAt) return;
    this._idleFetching = true;
    const seq = this.gestureSeq;
    fetch('/motion/idle')
      .then((r) => (r.status === 200 ? r.json() : null))
      .then((d) => {
        // a turn may have started while we were fetching — never stomp a LIVE clip
        if (!d || !d.joints || !d.joint_names || this._talking || seq !== this.gestureSeq) return;
        if (this.clip && !this._turnOver) return;
        // ...and the gestureSeq test above is NOT sufficient. The turn OPENER is applied
        // through upgradeGesture, which deliberately does not claim gestureSeq (so it can
        // never cancel the primary poll), so an idle fetch in flight when a turn starts is
        // not invalidated by it. Traced in normal operation: idle clip at t=36.4, opener
        // bridge (81f/55j) at t=37.5, then this idle landing at t=39.9 with _fadeT=0.02 and
        // the STATIC snapshot path — a re-fade on top of a speech clip that had just started,
        // which is the jolt at the seam. An idle animation may only ever take a body that is
        // empty or already idling; a speech clip outranks it, full stop.
        // ...but ONLY while the turn is live. endTurn() deliberately leaves the last speech
        // clip on the body so the first idle can crossfade out of the pose she finished on;
        // without the _turnOver term this blocked that handover forever and she never went
        // back to idling after speaking.
        if (this.clip && !this.clip.idle && !this._turnOver) return;
        this._turnOver = false;
        this.setClip(d, null); // crossfades out of the held pose
      })
      .catch(() => {})
      .finally(() => { this._idleFetching = false; });
  }

  /** True while a GestureLSM co-speech clip (SMPL-X, ~55 joints — KAWAII is 22) owns the
   *  body, INCLUDING the short hold after it runs out. The model caps clips at ~4.3s but
   *  sentences run ~5s, so every sentence has a tail gap; letting the next sentence's
   *  KAWAII talk-loop fill it made the body flip between two motion STYLES several times
   *  per turn ("めっちゃバグってる"). Blocking through the hold keeps one style for the
   *  whole turn — the hold-breathing covers the gap and the next GestureLSM (0.8-1.4s)
   *  takes over. The +1.5s cap is the safety valve: if generation stops arriving
   *  entirely, KAWAII is allowed back in rather than freezing on a stale pose. */
  gestureLSMActive() {
    const c = this.clip;
    // The backend TAGS GestureLSM clips (`gesturelsm: true`) and every Clip is built with
    // `lsm: !!data.gesturelsm`, so the flag is always a boolean. The old joint-count
    // fallback (>= 40) is not just dead, it is now WRONG — KAWAII clips carry 64 joints
    // since fingertips were added — so it is gone rather than left as a trap.
    if (!c || !c.lsm) return false;
    const clipLen = (c.joints.length - 1) / c.fps;
    return this.elapsed < clipLen + 1.5;
  }

  /** The RAW SMPL joint positions currently driving the VRM this tick — the
   *  interpolated + crossfade-blended `_frame` the retarget reads — plus the
   *  name->index map. A side-by-side reference figure (the motion model's own
   *  default body) renders from THIS so it plays the identical clip in perfect
   *  sync with the retargeted avatar. Null when no gesture clip is active. */
  getCurrentFrame(): { frame: number[][]; idx: Record<string, number> } | null {
    if (!this.clip || !this._frame.length) return null;
    return { frame: this._frame, idx: this.clip.idx };
  }

  /** Start/stop the volume-driven talking motion. Fed the same audio + volume envelope
   *  the lip-sync uses, so the body's gesture energy tracks the speech in real time. */
  startTalking(audio: HTMLAudioElement, volumes: number[], sliceMs: number) {
    this._talkAudio = audio; this._talkVol = volumes || []; this._talkSlice = sliceMs || 20;
    this._talking = true;
    // No warm loop: replaying the PREVIOUS sentence's/reply's cached gesture is exactly
    // the "motion is one sentence behind" desync the user saw. Each sentence gets ONLY
    // its own gesture (playGesture), bound to its own audio (stopped at speech end).
  }

  /** Streaming variant of startTalking: no HTMLAudioElement/volumes[] exist —
   *  a live RMS provider (the gapless WebAudio player's analyser, same signal
   *  the streaming lip-sync uses) drives the identical talking energy, so the
   *  body gestures from the FIRST streamed chunk with zero added latency. */
  startTalkingLive(levelFn: () => number) {
    this._talkLive = levelFn;
    this._talkAudio = null; this._talkVol = [];
    this._talking = true;
    // Kick: seed the energy so the body starts gesturing the INSTANT the turn
    // begins (humans gesture slightly before speaking) instead of waiting for
    // the RMS low-pass to ramp (~1s of visible lag).
    this._talkE = Math.max(this._talkE, 0.45);
  }

  stopTalking() { this._talking = false; this._talkAudio = null; this._talkVol = []; this._talkLive = null; }

  /** Speech energy (0..1) at the current playback position, or 0 when not speaking. */
  private currentVolume(): number {
    if (this._talking && this._talkLive) {
      const v = this._talkLive();
      return Number.isFinite(v) ? Math.min(1, Math.max(0, v)) : 0;
    }
    const a = this._talkAudio;
    if (!this._talking || !a || a.paused || a.ended || !this._talkVol.length) return 0;
    const i = Math.floor((a.currentTime * 1000) / this._talkSlice);
    return this._talkVol[Math.min(i, this._talkVol.length - 1)] || 0;
  }

  /** Re-seat the spring bones to the CURRENT pose (no velocity) so a pose teleport
   *  doesn't fling the hair/skirt. three-vrm: setInitState (fallback reset). */
  private springReset() {
    const sb = (this.vrm as unknown as { springBoneManager?: any })?.springBoneManager;
    if (!sb) return;
    // reset() zeros the spring velocity (settles each joint to rest relative to its
    // CURRENT parent) — the surest way to stop a wild spring. setInitState is a fallback.
    if (typeof sb.reset === "function") sb.reset();
    else if (typeof sb.setInitState === "function") sb.setInitState();
  }

  /** Interpolate a clip's joints at `elapsed` seconds into `out` (30fps clip -> render fps). */
  /** START-POSE MATCHING: pick the playhead in the NEW clip whose pose is closest to
   *  the CURRENTLY RENDERED frame. Seams were visible because the fade had to carry the
   *  arms from wherever they ended to the new clip's fixed start pose ("終了位置と開始
   *  位置が若干違う") — starting the new clip at its best-matching frame shrinks that
   *  travel to near zero. Distance = wrist/elbow/head positions in SMPL space (both
   *  frames already live there; no retarget needed). Search window: first ~1.2s (bounded
   *  desync vs speech — co-speech beats are loose at that scale). */
  private _matchStart(clip: Clip): { pad: number; dist: number } {
    const last = clip.joints.length - 1;
    const defPad = Math.min(0.3, (last / clip.fps) * 0.15);
    const cur = this._frame;
    const oldIdx = this.clip?.idx;
    if (!cur.length || !oldIdx) return { pad: defPad, dist: 0 };
    const KEYS = ["L_Wrist", "R_Wrist", "L_Elbow", "R_Elbow", "Head"];
    const pairs: [number, number][] = [];
    for (const k of KEYS) {
      const a = oldIdx[k]; const b = clip.idx[k];
      if (a !== undefined && b !== undefined && cur[a]) pairs.push([a, b]);
    }
    if (!pairs.length) return { pad: defPad, dist: 0 };
    const maxFi = Math.max(2, Math.min(Math.round(clip.fps * 1.2), Math.round(last * 0.6)));
    let bestFi = Math.max(1, Math.round(defPad * clip.fps)); let bestD = Infinity;
    for (let fi = 1; fi <= maxFi; fi += 1) {
      const f = clip.joints[fi]; let s = 0;
      for (const [a, b] of pairs) {
        const p = cur[a]; const q = f[b];
        const dx = p[0] - q[0]; const dy = p[1] - q[1]; const dz = p[2] - q[2];
        s += dx * dx + dy * dy + dz * dz;
      }
      if (s < bestD) { bestD = s; bestFi = fi; }
    }
    return { pad: bestFi / clip.fps, dist: Math.sqrt(bestD / pairs.length) };
  }

  private frameInto(clip: Clip, elapsed: number, out: number[][]) {
    const last = clip.joints.length - 1;
    const f = elapsed * clip.fps;
    const fi = Math.min(last, Math.max(0, Math.floor(f)));
    const fj = Math.min(last, fi + 1);
    const frac = Math.min(1, Math.max(0, f - fi));
    const a = clip.joints[fi];
    const c = clip.joints[fj];
    if (out.length !== a.length) { out.length = 0; for (let k = 0; k < a.length; k += 1) out.push([0, 0, 0]); }
    for (let k = 0; k < a.length; k += 1) {
      out[k][0] = a[k][0] + (c[k][0] - a[k][0]) * frac;
      out[k][1] = a[k][1] + (c[k][1] - a[k][1]) * frac;
      out[k][2] = a[k][2] + (c[k][2] - a[k][2]) * frac;
    }
  }

  /** Rewrite the outgoing clip's frame (_frame2, in the OUTGOING clip's joint order) into
   *  _frame2m, laid out in the CURRENT clip's order, so every downstream read can use
   *  `this.clip.idx`. A joint the outgoing clip does not have (jaw/eyes vs fingertips)
   *  takes the incoming clip's own value, which makes the blend a no-op for it rather
   *  than mixing in an unrelated joint. */
  private _remapPrevFrame() {
    const cl = this.clip; const pv = this._prevClip;
    if (!cl || !pv) return;
    if (this._fadeMapFor !== cl || this._fadeMapFrom !== pv) {
      this._fadeMap = cl.names.map((n) => { const j = pv.idx[n]; return j === undefined ? -1 : j; });
      this._fadeMapFor = cl; this._fadeMapFrom = pv;
    }
    const M = this._fadeMap; const P = this._frame2; const F = this._frame;
    if (this._frame2m.length !== M.length) this._frame2m = M.map(() => [0, 0, 0]);
    for (let k = 0; k < M.length; k += 1) {
      const j = M[k];
      const src = j >= 0 && j < P.length ? P[j] : F[k];
      if (!src) continue;
      const o = this._frame2m[k];
      o[0] = src[0]; o[1] = src[1]; o[2] = src[2];
    }
  }

  /** Begin a crossfade from the current clip (if any) into `elapsed` of the new one. */
  private _beginFade() {
    if (!this.clip) return;
    if (this._prevClip && this._fadeT < 1 && this._frame.length) {
      // A fade is STILL in progress (short sentences: 1-2s lines vs 0.35-0.9s fades).
      // Re-pointing prev at the current clip would drop the old outgoing component
      // (weight 1-t) in ONE frame — a visible pop at the seam. Fade from a snapshot
      // of the currently RENDERED blend instead: static, but the ease-out fade gives
      // the incoming clip velocity from frame 1 and the bone low-pass hides the hold.
      this._prevClip = {
        names: this.clip.names,
        idx: this.clip.idx,
        joints: [this._frame.map((p) => p.slice())],
        fps: this.clip.fps,
      };
      this._prevElapsed = 0;
      this._fadeStatic = true;
    } else {
      this._prevClip = this.clip;
      this._prevElapsed = this.elapsed;
      this._fadeStatic = false;
    }
    this._fadeT = 0;
  }

  /** Advance + apply the current gesture frame. Call once per frame BEFORE vrm.update(). */
  // --- usage ledger: which layer actually drove the body, per turn ---------
  private _stats = { clip: 0, hold: 0, proc: 0, idle: 0, applies: 0 };

  statsReset() { this._stats = { clip: 0, hold: 0, proc: 0, idle: 0, applies: 0 }; }

  statsSnapshot() { return { ...this._stats }; }

  update(dt: number) {
    if (!this.vrm || !this.clip) return;
    this._dt = dt > 0 ? dt : 1 / 60;
    // authored clip vs model output need different damping — see the gain sets above
    if (this.perSource) this._applyGains(this.clip.lsm ? "lsm" : "authored");
    this._tickGains(dt);
    // ledger: generated clip actively playing vs holding at its end
    if (this._holdT < 0.25) this._stats.clip += dt; else this._stats.hold += dt;
    if (this._restClip !== this.clip) {
      // CHAINED clips inherit the reply's rest reference: their frame 0 is a REPLAY of
      // the previous clip's tail (mid-gesture), not a neutral pose — capturing it as
      // "rest" would re-zero the FRAME bones onto a mid-gesture stance (posture snap).
      if (this.clip.chained && Object.keys(this.frameRestSmpl).length) this._restClip = this.clip;
      else this.captureSmplRest(); // (re)capture the SMPL rest frame
    }
    const last = this.clip.joints.length - 1;
    const clipLen = last / this.clip.fps;
    // Skip the clip's near-neutral START ramp (pad): a co-speech clip begins from a
    // rest-ish pose, so crossfading INTO frame 0 dips the whole body to neutral each
    // time and whips the hair (the "びくっと"). Offsetting past the ramp makes every
    // crossfade go gesture->gesture, not gesture->neutral. The gesture then leads speech
    // by ~pad, which is natural for co-speech. When start-pose matching picked an offset
    // (clip.pad), keep syncing/looping around THAT point.
    const pad = this.clip.pad ?? Math.min(0.3, clipLen * 0.15);
    const loop = !!this.clip.loop && clipLen > 0;
    const ga = this._gestAudio;
    // A KAWAII talk-loop stays audio-synced for the WHOLE sentence (however long) and
    // WRAPS its playhead; other clips sync only while the playhead fits the clip, then
    // free-run + hold. This is what keeps the body talking instead of freezing.
    const synced = !!ga && !ga.paused && !ga.ended && (loop || (ga.currentTime + pad) <= clipLen);
    if (synced && ga) {
      // DON'T hard-set the playhead to audio.currentTime — that clock updates in ~16ms
      // CHUNKS (many frames report the same value, then it jumps), so a hard set makes
      // `elapsed` STEP, the retargeted head jitters, and the long twintails amplify it
      // into the fast "震える" tremble (the pose INPUT is what shakes, not the springs).
      // Instead advance smoothly by real dt and only gently pull drift toward the audio
      // clock — a ~0.17s time-constant follow that's frame-rate independent. Hard-resync
      // only on a big gap (seek / first sync) so we still lock on quickly.
      const target = ga.currentTime + pad;
      const drift = target - this.elapsed;
      // NEVER teleport the playhead (the old |drift|>0.25 -> elapsed = target): a clip
      // lands 0.8-1.4s after its sentence starts (queued sentences: seconds), so the first
      // sync tick always saw drift ~ that latency and the teleport threw the body a
      // second of motion forward in ONE frame — the surviving speech-time jolt. The
      // playhead now only ADVANCES, and only its RATE bends toward the audio clock:
      // +35% catch-up / -25% slow-down at most, both invisible in a gesture, closing a
      // 1s landing lag over ~3s of playback.
      // +-12% at most: a stronger catch-up (0.35 was tried) consumes the clip's content
      // faster than real time, so the gesture ran out even further before the sentence end
      // — the user's exact complaint. At 12% the clip plays essentially 1:1 to its end and
      // a 1s landing lag shrinks slowly across the turn instead of being chased.
      this.elapsed += dt * this.speed * (1 + Math.max(-0.12, Math.min(0.12, drift * 0.6)));
      this._holdT = 0;
    } else {
      // Free-running (audio ended / gap between sentences). Decelerate into the clip
      // END and HOLD there (a natural gesture "hold" phase) instead of instantly
      // loop-wrapping: the next sentence's clip is CHAINED from this exact end pose,
      // so waiting here makes the handover truly seamless. Only when the gap drags on
      // (late clip / long pause) do we wrap and keep gesturing.
      const remainEnd = clipLen - this.elapsed;
      if (remainEnd < 0.15) {
        this.elapsed += Math.max(0, remainEnd) * Math.min(1, dt * 6); // brief settle
        this._holdT += dt;
      } else {
        this.elapsed += dt * this.speed;
      }
    }
    // Build this frame's joints (30fps clip lerped to ~60fps render — nearest-frame
    // stepping jolts the spring bones / "hair goes wild"). Loop clips WRAP the playhead
    // (seam≈0 -> seamless) so a 5s talk clip covers an arbitrarily long sentence.
    const playhead = loop ? ((this.elapsed % clipLen) + clipLen) % clipLen : this.elapsed;
    this.frameInto(this.clip, playhead, this._frame);
    // If a crossfade is in progress, keep the OUTGOING clip advancing too and blend it
    // in (smoothstep) — both playheads move, so the body never loses velocity at the
    // seam (this is what kills the brief "止まる" between lines/loops).
    if (this._prevClip && this._fadeT < 1) {
      // Decelerate the OUTGOING clip smoothly as it nears its end instead of letting it
      // slam into the last frame and FREEZE mid-fade — that hard stop of the outgoing
      // component is the "継ぎ目がスムーズじゃない" hitch (blend result decelerates then
      // re-accelerates). Ease its playhead rate to 0 over the last 0.25s.
      const prevLen = (this._prevClip.joints.length - 1) / this._prevClip.fps;
      const remain = Math.max(0, prevLen - this._prevElapsed);
      this._prevElapsed += dt * this.speed * Math.min(1, remain / 0.25);
      this.frameInto(this._prevClip, this._prevElapsed, this._frame2);
      this._fadeT = Math.min(1, this._fadeT + dt / this._fadeDurEff);
      this._remapPrevFrame(); // _frame2 -> _frame2m, in THIS clip's joint order
      // SMOOTHSTEP, not ease-out. The blended pose moves at
      //     v = (1-w)*v_prev + w*v_new + w'*(p_new - p_prev)
      // so the weight's SLOPE is a velocity term of its own. Ease-out x(2-x) starts at
      // its maximum slope (2/_fadeDur), which steps that term from 0 to full in one
      // frame — an acceleration spike on every seam, worst on `hips` because it is the
      // root and carries the whole body ("ビックってなる"). Measured 75-2400 deg/s^2 at
      // seams vs a 24 deg/s^2 baseline. Smoothstep has w'(0)=w'(1)=0, so velocity is
      // continuous at BOTH ends of the fade.
      // The flat start used to read as a stall when the outgoing clip had already ended
      // and sat still; that is handled at the source now — holdWrapAfter re-loops a
      // finished clip so it keeps moving, and the static-snapshot path in _beginFade
      // shortens the fade so its quiet head is only ~2-3 frames.
      const t = this._fadeW();
      const F = this._frame; const P = this._frame2m;
      // Keep the NEW clip's frame (pre-blend) so FRAME bones can blend rotations, not
      // positions (position-blend gives a non-linear head-rotation spike -> hair whip).
      if (this._frameNew.length !== F.length) this._frameNew = F.map((a) => a.slice());
      else for (let k = 0; k < F.length; k += 1) { const g = this._frameNew[k]; const f = F[k]; g[0] = f[0]; g[1] = f[1]; g[2] = f[2]; }
      const n = Math.min(F.length, P.length);
      for (let k = 0; k < n; k += 1) {
        F[k][0] = P[k][0] + (F[k][0] - P[k][0]) * t;
        F[k][1] = P[k][1] + (F[k][1] - P[k][1]) * t;
        F[k][2] = P[k][2] + (F[k][2] - P[k][2]) * t;
      }
      if (this._fadeT >= 1) this._prevClip = null;
    }
    this.applyFrame(this._frame, this.clip.idx);
    this.applyBeatSync(dt);
    // HOLD BREATHING: while a finished clip HOLDS its end pose in an inter-clip gap,
    // add a gentle sway AROUND that pose (multiplied on top, so it never DRIFTS toward
    // idle — that drift was the "がくがく") so the ~0.2s gaps read as continued talking,
    // not a freeze and not a jump. Ramps in with holdT; the next chained clip (holdT=0)
    // turns it off. Keeps the body alive between GestureLSM clips without a seam.
    if (this.holdBreath && this._holdT > 0.04) {
      const s = Math.min(1, (this._holdT - 0.04) * 3);
      const t2 = performance.now() / 1000;
      const add = (bone: string, rx: number, ry: number, rz: number) => {
        const nb = this.node(bone); if (!nb) return;
        this._e.set(rx * s, ry * s, rz * s); this._dq.setFromEuler(this._e);
        nb.quaternion.multiply(this._dq); nb.updateMatrixWorld(true);
      };
      add('spine', Math.sin(t2 * 1.6) * 0.012, Math.sin(t2 * 1.05) * 0.016, 0);
      add('chest', Math.sin(t2 * 1.6 + 0.5) * 0.01, Math.sin(t2 * 1.25) * 0.013, 0);
      add('leftUpperArm', Math.sin(t2 * 1.4) * 0.022, 0, Math.sin(t2 * 1.15) * 0.02);
      add('rightUpperArm', Math.sin(t2 * 1.4 + Math.PI) * 0.022, 0, Math.sin(t2 * 1.15 + Math.PI) * 0.02);
      add('neck', Math.sin(t2 * 1.5) * 0.01, Math.sin(t2 * 1.1) * 0.012, 0);
    }
    // Re-seat springs only across the idle<->gesture transition (teleport). The
    // ongoing jerks that flung the hair are now killed by SMOOTHING the retargeted
    // bones in applyFrame, so the springs get smooth input and can sway for real.
    if (this.pendingSpringInit > 0) { this.springReset(); this.pendingSpringInit -= 1; }
    // Loop via a crossfade (outgoing = this clip held at its end, incoming = frame 0) so
    // the loop keeps velocity instead of stopping. `elapsed > clipLen` only happens when
    // free-running (the audio-sync caps at clipLen). A fresh line's clip crossfades over
    // this; reply-end/interrupt clears the clip to settle back to idle. (Loops only fire
    // in gaps with no fresh clip; back-to-back lines just crossfade.)
    // An IDLE clip plays ONCE: let it finish, then hand the body back to the breathing
    // idle and schedule the next random one (a looped idle would read as a tic).
    // NB: test the HOLD, not `elapsed >= clipLen` — the free-run settle above eases the
    // playhead into the end ASYMPTOTICALLY, so it never actually reaches clipLen and the
    // cycle stalled on its first clip forever ("モーションがランダムに生成されない").
    // 0.05, not 0.15: holdWrapAfter is 0.12, and the loop-crossfade below used to fire FIRST
    // and reset _holdT — so this test never passed, _turnOver never flipped, and the idle
    // rotation never advanced. The clip just wrapped end->start forever ("選ばれたモーションが
    // ずっと再生される", and that wrap IS the "毎回がくっとなる").
    if (this.clip.idle && !this._turnOver && (this._holdT > 0.05 || this.elapsed >= clipLen)) {
      // Finished: mark it replaceable but KEEP it on the body, holding (with the
      // hold-breathing) until the next one CROSSFADES out of this exact pose. Clearing
      // it here instead made the body drift to the rest pose during the gap and then
      // pop into the next clip — the visible idle-to-idle seam.
      this._turnOver = true;
      this._idleNextAt = performance.now()
        + (this.idleGapMin + aiDebug.random() * (this.idleGapMax - this.idleGapMin)) * 1000;
    }
    // NEVER wrap an IDLE clip. The wrap exists so a SPEECH clip that ran out keeps gesturing
    // while it waits for the next sentence; an idle animation has nothing to wait for — the
    // next random idle crossfades out of its held end pose, which is smooth, whereas the wrap
    // jumps end->start on a clip that was never authored to loop.
    // ...and never a GestureLSM clip either: its successor is SEEDED from this clip's end
    // pose, so the smooth path is hold-at-end (with hold-breathing) -> crossfade into the
    // next clip's frame 0 = the very same pose. The wrap instead yanked the body to the
    // clip's near-neutral opening ~0.8s before every landing — the "clip stops early, then
    // びくっ, then the next motion" the user described. KAWAII speech clips keep the wrap
    // (their loops are authored to cycle).
    // A GestureLSM clip caps at ~4.3s, so a LONG sentence outlives its clip. Two very
    // different situations look identical from here (clip exhausted, free-running):
    //   * the sentence is STILL SPEAKING -> she must keep gesturing, so wrap;
    //   * the sentence is OVER (tail gap) -> the successor is seeded from this end pose,
    //     so hold and let it crossfade out of exactly that pose.
    // Blanket-banning the wrap for lsm fixed the second case and broke the first: on a long
    // line the body froze at the clip's last frame until the next sentence arrived. Tell
    // them apart by the audio clock, which is still running only in the first case.
    const speaking = !!ga && !ga.paused && !ga.ended;
    const mayWrap = this.clip.lsm ? speaking : true;
    if (!loop && !this.clip.idle && mayWrap && (this.elapsed > clipLen || this._holdT > this.holdWrapAfter)) {
      // A finished clip HOLDS at its end pose waiting for the NEXT (chained) clip to
      // crossfade from that exact pose — a SEAMLESS handover (no jump). The gentle idle
      // sway (updateIdle) keeps it from a dead freeze meanwhile. We only loop-crossfade
      // (which JUMPS end->start = "がくがく") as a last resort if no successor arrives
      // for a long time, so seam-jerk is gone at the cost of a short hold in the pause.
      // Wrap to the frame that best MATCHES the pose on the body, not a fixed `pad`. The
      // crossfade already kept position continuous, but end-pose and start-pose are far
      // apart, so the body had to cover that whole gap inside 0.4s — that traversal IS the
      // "がくっ" that survived while speaking (idle clips no longer wrap at all). _matchStart
      // scans the clip's opening for the pose closest to what is rendered right now, which
      // is the same trick the sentence seam uses, so there is almost no gap left to cover.
      // Must be read BEFORE _prevElapsed/elapsed change: it compares against _frame.
      const wrapPad = this._matchStart(this.clip).pad;
      this._prevClip = this.clip;
      this._prevElapsed = this.elapsed;
      this._fadeT = 0;
      this._fadeDur = 0.4;
      this._holdT = 0;
      this.elapsed = wrapPad;
      // no spring reset at the loop seam — it snapped the hair to the floating rest
      // pose = the mid-motion "跳ね上がる" jump. The crossfade + heavy drag handle it.
    }
  }

  // --- BEAT SYNC: modulate the GENERATED clip's amplitude with the live speech
  // RMS (same signal as the lip-sync) so gesture strokes punch on loud syllables.
  // Text-driven ActionPlan clips know nothing about the audio; scaling each
  // retargeted bone's rotation from its aim-rest (identity) by a fast-attack /
  // slow-release energy envelope makes the motion FEEL beat-locked: quiet
  // passages soften to 0.80x, stressed syllables overdrive to 1.15x. Hands and
  // hips/legs are excluded (palm facing + stance stability). Runs after
  // applyFrame, before the springs read the pose. Disable live with:
  // localStorage.motionBeatSync = '0' (no rebuild).
  private _beatE = 0;

  private _beatQ = new THREE.Quaternion();

  private _beatCache = { ts: -1e9, val: true };

  private applyBeatSync(_dt: number) {
    if (!this.vrm) return;
    const now = performance.now();
    if (now - this._beatCache.ts > 2000) {
      this._beatCache.val = localStorage.getItem('motionBeatSync') !== '0';
      this._beatCache.ts = now;
    }
    if (!this._beatCache.val || this._beatW <= 1e-3) return; // off for authored clips
    const cv = this.currentVolume(); // live RMS (0..1)
    // fast attack so the stroke lands ON the syllable, slower musical release
    this._beatE += (cv - this._beatE) * (cv > this._beatE ? 0.55 : 0.1);
    // Centre at 1.0: quiet passages keep the gesture at FULL reach (faithful to the
    // model — the old 0.8 floor shrank the arms ~20% whenever it was quiet, so Miku
    // under-reached vs the reference = the "少し違う動き"); loud syllables still
    // overdrive to 1.15 for the beat punch.
    // faded in/out with the gain ramp so the amplitude doesn't step at a source change
    const g = 1 + Math.min(0.15, 0.15 * this._beatE) * this._beatW;
    const gHalf = 1 + (g - 1) * 0.5; // subtler on the torso/head chain
    for (const b of ["leftUpperArm", "rightUpperArm", "leftLowerArm", "rightLowerArm"]) {
      const nb = this.node(b);
      if (nb) { this._beatQ.identity().slerp(nb.quaternion, g); nb.quaternion.copy(this._beatQ); }
    }
    // NB: NOT "head" — it isn't reset per frame (no eye landmarks in the 22-joint
    // stream), so scaling it here accumulated into a continuous neck spin.
    for (const b of ["spine", "chest", "neck"]) {
      const nb = this.node(b);
      if (nb) { this._beatQ.identity().slerp(nb.quaternion, gHalf); nb.quaternion.copy(this._beatQ); }
    }
  }

  /** Blend a bone TOWARD (idle-base * euler-offset) by alpha — never snaps, so idle,
   *  talking, and gesture->ambient transitions are all smooth. */
  private ambient(bone: string, rx: number, ry: number, rz: number, alpha: number) {
    const nb = this.node(bone); const base = this.idleBase[bone];
    if (!nb || !base) return;
    this._e.set(rx, ry, rz); this._dq.setFromEuler(this._e);
    this._ambTarget.copy(base).multiply(this._dq);
    nb.quaternion.slerp(this._ambTarget, alpha);
  }

  /** Ambient (no gesture clip): livelier TALKING motion while speaking, gentle idle
   *  when silent. Everything is BLENDED, so it eases in from the gesture's last pose
   *  (no frame-skip). Call per frame after update(), before vrm.update(). */
  // GENERATED-ONLY mode (default ON): the body is driven exclusively by generated
  // clips — no procedural talking layer, no hold-takeover blend. Evaluation mode
  // for the ActionPlan text-streaming motion; the breath idle still runs when no
  // clip exists so the avatar isn't a statue between turns. Re-enable the
  // procedural layers live with: localStorage.motionGenOnly = '0'; reload.
  private _genOnlyCache = { ts: -1e9, val: true };

  private _genOnly(): boolean {
    const now = performance.now();
    if (now - this._genOnlyCache.ts > 2000) {
      this._genOnlyCache.val = localStorage.getItem('motionGenOnly') !== '0';
      this._genOnlyCache.ts = now;
    }
    return this._genOnlyCache.val;
  }

  updateIdle(dt: number) {
    if (!this.vrm) return;
    this._idleTick(); // between turns: keep a random KAWAII idle clip cycling
    const genOnly = this._genOnly();
    // While a gesture clip is actively PLAYING it owns the body. But once the
    // clip has run out and is HOLDING its last pose (long sentence, late/no
    // successor clip), ramp the ambient/talking layer back in so the body never
    // freezes mid-speech — the freeze WAS this early-return firing during hold.
    const a = 0.14; // blend rate (smoothness vs responsiveness)
    // While a clip exists (playing OR holding its end pose in a gap), update()'s HOLD
    // BREATHING owns the body — it sways AROUND the held pose without drifting. This
    // ambient layer only runs BETWEEN turns (no clip), easing back to the idle pose.
    // (Previously this ran during holds and its pull toward idle read as がくがく.)
    if (this.clip) return;
    this.idleT += dt;
    const t = this.idleT;
    // Asymmetric low-pass on the speech energy: FAST attack so gestures are
    // visible the moment the voice starts, slow release so brief pauses don't
    // still the body (the old symmetric 0.12 made motion lag speech by ~1s).
    const cv = this.currentVolume();
    this._talkE += (cv - this._talkE) * (cv > this._talkE ? 0.2 : 0.06);
    // generated-only: a GENTLE talking layer (capped genOnlyTalk) fills the gaps so
    // the body sways/gestures softly between sparse generated clips instead of
    // freezing. Full mode keeps the stronger layer.
    const e = genOnly
      ? Math.min(this.genOnlyTalk, this._talkE)
      : Math.min(1, this._talkE * 1.6); // 0 = silent, ~1 = talking
    if (!this.clip) {
      if (this._talking) this._stats.proc += dt; else this._stats.idle += dt;
    }
    // Always-on gentle idle (breath + slow sway), present even at e=0.
    const breath = Math.sin(t * 1.5) * 0.014;
    const sway = Math.sin(t * 0.6) * 0.012;
    const hy = Math.sin(t * 0.45) * 0.06 + Math.sin(t * 0.23) * 0.03;
    const hx = Math.sin(t * 0.37) * 0.025;
    // Talking layer (scaled by e): torso sways more, head nods, arms/elbows gesture.
    this.ambient("spine", breath + Math.sin(t * 2.1) * 0.02 * e, Math.sin(t * 1.7) * 0.03 * e, sway + Math.sin(t * 1.3) * 0.02 * e, a);
    this.ambient("chest", breath * 0.5 + 0.03 * e, Math.sin(t * 1.7 + 0.6) * 0.025 * e, sway * 0.5, a);
    this.ambient("neck", hx * 0.4 + Math.sin(t * 3.3) * 0.05 * e, hy * 0.4 + Math.sin(t * 2.0) * 0.04 * e, Math.sin(t * 1.1) * 0.03 * e, a);
    this.ambient("head", hx + Math.sin(t * 3.3 + 0.4) * 0.07 * e, hy + Math.sin(t * 1.8) * 0.05 * e, Math.sin(t * 0.9) * 0.04 * e, a);
    // Arms: mirror-phased gesturing, only when talking (e>0). rz lifts, rx forward.
    // The DC terms (this.stand) hold the hands clear of the skirt; the sines gesture around them.
    const sZ = this.stand.armZ; const sE = this.stand.elbowY;
    this.ambient("leftUpperArm", Math.sin(t * 2.6) * 0.10 * e, 0, sZ + Math.sin(t * 2.0) * 0.22 * e, a);
    this.ambient("rightUpperArm", Math.sin(t * 2.6 + Math.PI) * 0.10 * e, 0, -sZ + Math.sin(t * 2.0 + Math.PI) * 0.22 * e, a);
    this.ambient("leftLowerArm", 0, sE + Math.sin(t * 3.0) * 0.28 * e, 0, a);
    this.ambient("rightLowerArm", 0, -sE + Math.sin(t * 3.0 + Math.PI) * 0.28 * e, 0, a);
    // Ease the pelvis rotation + hands + IK legs + hips shift back to rest (chest and
    // head keep doing the idle sway above, so don't zero them).
    // During a hold-takeover, pull hips/hands/legs back only GENTLY — a strong
    // pull shifts the pelvis and whips the spring bones (the "ぶるっ" shudder).
    const aRest = this.clip ? a * 0.25 : a;
    for (const b of ["hips", "leftHand", "rightHand"]) { const nb = this.node(b); if (nb) nb.quaternion.slerp(this._ambIdentity, aRest); }
    for (const b of ["leftUpperLeg", "leftLowerLeg", "leftFoot", "rightUpperLeg", "rightLowerLeg", "rightFoot"]) {
      const nb = this.node(b); if (nb) nb.quaternion.slerp(this._ambIdentity, aRest);
    }
    const hipsN = this.node("hips"); if (hipsN) hipsN.position.lerp(this.hipsRestPos, a);
    if (this.pendingSpringInit > 0) { this.springReset(); this.pendingSpringInit -= 1; }
  }

  private applyFrame(frame: number[][], idx: Record<string, number>) {
    if (!this.vrm) return;
    // Push hands/elbows out of the (thicker, clothed) torso so co-speech gestures
    // don't sink into the body. Returns the same ref when nothing needed clamping.
    frame = clampHandsOutsideTorso(frame, idx, this.torsoRadius, this._torsoGuard, this._dt, this._torsoSkirtOnly);
    // Same clamp for the two frames the crossfade path retargets directly, so the hand
    // orientation is built from consistent landmarks whether or not a fade is running
    // (see _frameNewC). Both are already in THIS clip's layout, so `idx` applies.
    this._frameNewC = clampHandsOutsideTorso(this._frameNew, idx, this.torsoRadius, undefined, 1 / 60, this._torsoSkirtOnly);
    this._frame2mC = clampHandsOutsideTorso(this._frame2m, idx, this.torsoRadius, undefined, 1 / 60, this._torsoSkirtOnly);
    // (those two are read only for hips/head/chest, whose landmarks the clamp never touches,
    //  so they stay stateless — sharing the rate-limiter would fight the rendered frame.)
    // Pelvis (root) full orientation FIRST — the aim chain below hangs off it and,
    // being absolute, re-aligns correctly regardless of the pelvis rotation.
    // Seed the low-pass from the CURRENT pose on a new gesture so it eases in from
    // wherever the body is (kills the frame-0 snap / "フレーム飛び" at transitions).
    if (this._seedSmooth) {
      for (const b of ORDER) { const nb = this.node(b); if (nb) this._smoothQ[b] = nb.quaternion.clone(); }
      for (const b of Object.keys(FRAME)) { const nb = this.node(b); if (nb) this._smoothQ[b] = nb.quaternion.clone(); }
      this._seedSmooth = false;
    }
    // Undo the previous frame's floor lift FIRST: the pelvis below is written by a lerp
    // from the current value, so leaving the lift in would feed it back into the next
    // measurement (lift -> looks grounded -> drop -> penetrate = a vibrating body).
    if (this._lift) {
      const h0 = this.node("hips");
      if (h0) h0.position.y -= this._lift;
    }
    if (this.lockLegs) {
      // Lower body frozen: pelvis held at its rest pose/position so the feet never
      // move; the upper-body aim chain is absolute, so it re-aligns off the fixed
      // pelvis and the gesture still reads fully.
      const hipsN = this.node("hips");
      if (hipsN) {
        if (this.restLocal.hips) hipsN.quaternion.copy(this.restLocal.hips);
        hipsN.position.copy(this.hipsRestPos);
        hipsN.updateMatrixWorld(true);
      }
    } else {
      this.applyFrameBone("hips", frame, idx);
      // Pelvis translation (weight shift / crouch) whenever the lower body is free —
      // with footIK the legs are IK'd to it, without it they aim-retarget from it. Skip
      // it and a crouching clip would keep the hips at standing height while the legs
      // bend, floating the feet.
      this.applyPelvisShift(frame, idx);
    }
    // reset the retargeted bones, then aim each from the SMPL directions.
    for (const b of ORDER) { const nb = this.node(b); if (nb) nb.quaternion.identity(); }
    // The head is NOT in ORDER and its FRAME retarget NO-OPs on the 22-joint stream
    // (no R_Eye/L_Eye landmarks), so it is never reset — without this, applyBeatSync's
    // per-frame scale ACCUMULATED on the head and the neck "spun" (どんどん回る). Reset
    // it every frame so it simply follows the neck (its only real orientation source).
    { const hn = this.node("head"); if (hn) hn.quaternion.identity(); }
    this.vrm.scene.updateMatrixWorld(true);
    for (const b of ORDER) {
      // Legs are ALWAYS aim-retargeted now; the IK (if any) blends over that result below,
      // so switching sources ramps instead of snapping the feet to their planted targets.
      if (this.lockLegs && /(UpperLeg|LowerLeg|Foot)$/.test(b)) continue;
      if (b === "chest") { if (this.chestFrame) { this.applyFrameBone("chest", frame, idx); continue; } }
      else if (FRAME[b]) { this.applyFrameBone(b, frame, idx); continue; } // hands (palm-facing)
      const nb = this.node(b); const aim = AIM[b]; const rd = this.restDir[b]; const rq = this.restWQ[b];
      if (!nb || !aim || !rd || !rq) continue;
      const fi = idx[aim[0]]; const ti = idx[aim[1]];
      if (fi === undefined || ti === undefined) continue;
      this._v0.fromArray(frame[fi]); this._v1.fromArray(frame[ti]);
      this._dir.copy(this._v1).sub(this._v0);
      if (this._dir.lengthSq() < 1e-8) continue;
      this._dir.normalize();
      if (this.parentAim && this.restPWInv[b]) {
        // parent world delta from rest -> where the rest direction went with the parent ->
        // swing the remainder. desired = swing * parentDelta * restWorld; the bone's world
        // direction still lands EXACTLY on the clip's (swing closes the gap by construction).
        nb.parent!.getWorldQuaternion(this._paDelta);
        this._paDelta.multiply(this.restPWInv[b]);
        this._paPred.copy(rd).applyQuaternion(this._paDelta).normalize();
        this._delta.setFromUnitVectors(this._paPred, this._dir);
        this._paDelta.premultiply(this._delta);
      } else {
        this._paDelta.copy(this._delta.setFromUnitVectors(rd, this._dir));
      }
      let desired = this._paDelta.multiply(rq); // world target
      // Whole-orientation plane retarget (superseded by kneeRoll below; kept switchable).
      if (this.planeRoll && PLANE[b] && this.planeRestSmpl[b]
          && this.quatFromPlane(frame, idx, PLANE[b], this._qPlane)) {
        desired = this._qPlane
          .multiply(this._qA.copy(this.planeRestSmpl[b]).invert())
          .multiply(rq);
      } else if ((b.includes("Leg") ? this.kneeRoll : this.elbowRoll)
          && HINGE_ROLL.has(b) && this.planeRestSmpl[b]
          && this.quatFromPlane(frame, idx, PLANE[b], this._qPlane)) {
        // Plane-derived target: same bone DIRECTION as `desired` (both aim down hip->knee),
        // so the two can only differ by a rotation about that direction = pure roll.
        this._krA.copy(this._qPlane)
          .multiply(this._krB.copy(this.planeRestSmpl[b]).invert())
          .multiply(rq);
        this._krB.copy(this._krA).multiply(this._qA.copy(desired).invert());
        const dp = this._krB.x * this._dir.x + this._krB.y * this._dir.y + this._krB.z * this._dir.z;
        let ang = 2 * Math.atan2(dp, this._krB.w);
        if (ang > Math.PI) ang -= 2 * Math.PI; else if (ang < -Math.PI) ang += 2 * Math.PI;
        const wNow = Math.max(0, Math.min(1, (this._planeSin - 0.10) / (this.kneeRollFull - 0.10)));
        const wRest = Math.max(0, Math.min(1, ((this.planeRestSin[b] ?? 0) - 0.10) / (this.kneeRollFull - 0.10)));
        const w = wNow * wRest;
        if (w > 0) {
          this._krB.setFromAxisAngle(this._dir, ang * w);
          desired.premultiply(this._krB);
        }
      }
      // Take a share of the twist that would otherwise all land on the absolutely-oriented
      // joint below this one (wrist / chest). Must run while _dir still holds this bone's
      // aim direction, and BEFORE the parent-relative conversion below.
      if (b === "spine" && this.chestFrame) {
        this.twistShareInto(desired, "chest", frame, idx, this.spineTwistShare);
      } else if (b === "leftLowerArm" || b === "rightLowerArm") {
        this.twistShareInto(desired, `${b.slice(0, -8)}Hand`, frame, idx, this.forearmTwistShare);
      }
      nb.parent!.getWorldQuaternion(this._pq);
      this._pq.invert().multiply(desired); // _pq = raw target LOCAL quat
      // Low-pass the bone toward its target: a jerky retarget frame (SMPL noise /
      // aim near singularities) would otherwise snap the bone and fling the springs.
      // Smoothing gives the spring bones smooth input so they sway instead of exploding.
      let sq = this._smoothQ[b];
      if (!sq) { sq = this._pq.clone(); this._smoothQ[b] = sq; }
      else sq.slerp(this._pq, this.smoothFor(b));
      // Damp toward the idle rest pose (kills the clavicle shrug / tames the torso).
      const g = this.gainFor(b); const rl = this.restLocal[b];
      if (g >= 0.999 || !rl) nb.quaternion.copy(sq);
      else nb.quaternion.copy(rl).slerp(sq, g);
      nb.updateMatrixWorld(true);
    }
    // Head look direction LAST — the neck (its parent) is now posed by the aim loop.
    this.applyFrameBone("head", frame, idx);
    if (this.lockLegs) {
      // Hold both legs straight at their rest pose (feet flat, planted) — no IK,
      // no bend, no slide. The frozen pelvis keeps the whole lower body still.
      for (const b of ["leftUpperLeg", "leftLowerLeg", "leftFoot", "rightUpperLeg", "rightLowerLeg", "rightFoot"]) {
        const nb = this.node(b);
        if (nb) { if (this.restLocal[b]) nb.quaternion.copy(this.restLocal[b]); else nb.quaternion.identity(); }
      }
    } else if (this._ikW > 1e-3) {
      // Leg IK: plant the ankles on their targets from the now-posed hips — blended over
      // the aim-retargeted legs by _ikW so a source change eases the feet into/out of
      // being planted (a hard switch jumped the foot 0.15m and kicked the floor clamp).
      const LEGS = ["leftUpperLeg", "leftLowerLeg", "leftFoot", "rightUpperLeg", "rightLowerLeg", "rightFoot"];
      const blend = this._ikW < 0.999;
      if (blend) {
        for (const bn of LEGS) {
          const nb = this.node(bn); if (!nb) continue;
          (this._aimLeg[bn] ??= new THREE.Quaternion()).copy(nb.quaternion);
        }
      }
      this.vrm.scene.updateMatrixWorld(true);
      this.solveLegIK("left");
      this.solveLegIK("right");
      if (blend) {
        for (const bn of LEGS) {
          const nb = this.node(bn); const a = this._aimLeg[bn];
          if (!nb || !a) continue;
          this._qt.copy(nb.quaternion);            // IK result
          nb.quaternion.copy(a).slerp(this._qt, this._ikW);
        }
        this.vrm.scene.updateMatrixWorld(true);
      }
    }
    this.applyFloorClamp();
    this.vrm.humanoid.update();
  }

  /** Keep her ON the floor. Without the ankle IK nothing stopped a crouch or a foot-lift
   *  from driving the foot THROUGH the ground (measured -5cm). Lift the hips by whatever
   *  the lowest contact point is short of its rest height — never push DOWN, so a clip
   *  that legitimately lifts both feet still reads as a lift. Smoothed, because a hard
   *  per-frame correction would jitter the whole body. */
  private applyFloorClamp() {
    if (!this.floorClamp || !this.vrm || !isFinite(this.groundY)) return;
    const hipsN = this.node("hips");
    if (!hipsN) return;
    this.vrm.scene.updateMatrixWorld(true);
    let low = Infinity;
    for (const s of ["left", "right"] as const) {
      const n = this.node(`${s}Toes`) ?? this.node(`${s}Foot`);
      if (!n) continue;
      n.getWorldPosition(this._ik2);
      low = Math.min(low, this._ik2.y);
    }
    // Inverted/crouched poses put the HEAD or HANDS lowest (handstand, deep bow) —
    // clamp on whichever body part is actually closest to the ground.
    for (const bn of ["head", "leftHand", "rightHand"]) {
      const n = this.node(bn);
      if (!n) continue;
      n.getWorldPosition(this._ik2);
      low = Math.min(low, this._ik2.y);
    }
    if (!isFinite(low)) return;
    // `low` is measured on the UNLIFTED pose (applyFrame removes the previous frame's
    // lift before posing), so `want` is the true penetration. Measuring the lifted pose
    // instead made this a feedback loop — lift, "no longer penetrating", drop, penetrate
    // again — i.e. the whole body vibrating ("ブルってなる").
    const want = Math.max(0, this.groundY - low);
    this._lift += (want - this._lift) * 0.2;
    if (this._lift > 1e-4) {
      hipsN.position.y += this._lift;
      hipsN.updateMatrixWorld(true);
    } else {
      this._lift = 0;
    }
  }
}

export const vrmMotionPlayer = new VRMMotionPlayer();
