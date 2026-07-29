/* eslint-disable no-underscore-dangle */
/**
 * Hand/elbow torso-penetration guard (radial depth clamp).
 *
 * Co-speech gestures often bring the hands/elbows INTO the torso, and neither the
 * VRM aim-retarget nor the SMPL LBS has any body collision — so the hands sink into
 * the body on both avatars. This pushes the four arm joints OUT so their horizontal
 * (xz) distance from the spine axis is at least `radius`, with a mild vertical taper
 * (wider at the belly, narrower near the neck). The arm direction then points outward
 * and the hand clears the body.
 *
 * A pushed WRIST carries its whole hand: the retarget reads the hand's orientation from
 * wrist->middle-knuckle and index-knuckle->pinky-knuckle, so moving the wrist alone
 * silently ROTATES the hand. Worse, as the push grows the wrist crosses the knuckles,
 * wrist->middle-knuckle collapses to zero length and the basis FLIPS — measured as both
 * wrists snapping 80-134 deg for one or two frames whenever a gesture crossed both hands
 * in front of the chest. Translating the finger joints by the same delta keeps the hand's
 * internal geometry (and therefore its orientation) exactly as animated, which is also
 * what pushing a hand out of a body should do physically.
 *
 * Returns the SAME frame reference when nothing needed clamping (no allocation);
 * otherwise a shallow copy with only the pushed joints replaced.
 * `radius`: ~0.19 for Miku (clothed body is thicker), ~0.14 for the raw SMPL body.
 */

const ARM = ["L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist"];

/** Finger joints that ride along with each wrist (whatever of them the layout carries:
 *  GestureLSM has 3 per finger, the KAWAII clips 4 including a fingertip). */
/** Three changes ported from the MMD retarget sit behind these flags. All are OFF while the
 *  reported wrist / finger / ankle trouble is bisected — with every flag false this file
 *  behaves exactly as it did before this session. Flip one at a time and judge on screen.
 *    CARRY_HAND   the pushed wrist takes its finger joints with it (no silent hand rotation)
 *    CAPSULE      test against the pelvis..neck SEGMENT, not an infinite column
 *    GUARD_RATE   limit how fast the push may change (MMD caps it to stop elbow pop)  */
// ON. With it off only the WRIST was pushed out while the finger joints stayed put, so the
// retarget (which reads the hand basis from wrist->middle-knuckle and index->pinky knuckle)
// silently ROTATED the hand: the wrist cleared the cloth and the palm/fingers turned into it
// instead — "手の位置がスカートにめり込む形になってる". Translating the finger joints by the
// same delta keeps the hand's internal geometry, so it moves out as one rigid piece.
const CARRY_HAND = true;
const CAPSULE = false;
// ON. The skirt springs carry NO hand collider (COLLIDE_FOR.skirt is the two hair roots
// only, matching the PMX mask), so a hand entering the skirt cannot push it — the "びくびく"
// at the penetration boundary is the HAND chattering, not the cloth. This clamp is the only
// thing acting there, and without a rate limit the push appears and vanishes the instant the
// joint crosses the capsule, so a hand hovering on the boundary flips in/out every frame.
// The limiter makes the hand-off continuous (see GUARD_RATE_MPS: 1.87 -> 0.91 units/frame
// on the MMD side).
const GUARD_RATE_ON = true;

const HAND_CHILDREN: Record<string, RegExp> = {
  L_Wrist: /^L_(Index|Middle|Ring|Pinky|Thumb)\d$/,
  R_Wrist: /^R_(Index|Middle|Ring|Pinky|Thumb)\d$/,
};

/** Max rate of CHANGE of the push, m/s. Ported from the MMD retarget, which hit exactly this
 *  problem: the correction itself is continuous, but it hands over between constraints and can
 *  reverse within a frame, and an unlimited hand-off pops the elbow. smpl_to_mmd.py records the
 *  measurement — a 0.10 m/FRAME cap (3 m/s at its 30fps) "keeps residual penetration ~5mm while
 *  cutting the worst elbow pop 1.87 -> 0.91 units/frame". This side had no cap at all, which is
 *  why fast clips snapped the elbow/wrist as a hand crossed the torso boundary. */
const GUARD_RATE_MPS = 3.0;

/** SKIRT band. The body profile below narrows toward the hips (0.667x at t=0) because it
 *  models the TORSO, so a hand dropped to the hip lands INSIDE the skirt and drags the cloth
 *  — which is the whole reason the skirt shudders. Measured on the model (skirt bone radii,
 *  normalised by the pelvis->neck span): the cloth is 0.153 at the waist and flares to 0.267
 *  at its widest going down, i.e. it gets WIDER exactly where the torso profile gets thinner.
 *  `top`/`hem` are heights in fractions of the pelvis->neck span measured from the pelvis
 *  (negative = below it); `rTop`/`rHem` are the cloth radius as a fraction of `radius`.
 *  Pushing only to the surface is deliberate: the user wants the hand to TOUCH the skirt,
 *  just not go through it. Tune live with __vrmMotion.skirtClamp. */
export const SKIRT = { on: true, top: 0.60, hem: -0.15, rTop: 1.10, rHem: 1.60 };

/** Per-joint push carried between frames so it can be rate-limited. Caller owns it. */
export type TorsoClampState = { push: Record<number, [number, number]> };

export const newTorsoClampState = (): TorsoClampState => ({ push: {} });

export function clampHandsOutsideTorso(
  frame: number[][],
  idx: Record<string, number>,
  radius: number,
  /** Omit for a stateless clamp (reference figure); pass both to rate-limit the push. */
  state?: TorsoClampState,
  dt = 1 / 60,
  /** Authored (KAWAII) clips: clamp ONLY the wrists, ONLY against the skirt band. Their
   *  arms were authored clean and ran unclamped for months; the full profile pushed their
   *  elbows and visibly bent the arms. LSM passes false and keeps the whole guard. */
  skirtOnly = false,
): number[][] {
  if (!(radius > 0)) return frame;
  const pi = idx.Pelvis; const ni = idx.Neck;
  if (pi === undefined || ni === undefined || !frame[pi] || !frame[ni]) return frame;
  const pel = frame[pi]; const nek = frame[ni];
  const dy = (nek[1] - pel[1]) || 1;
  let out: number[][] | null = null;
  const lim = GUARD_RATE_MPS * Math.max(1e-4, dt);
  for (const name of ARM) {
    const isWrist = name.endsWith("Wrist");
    if (skirtOnly && !isWrist) continue;
    const i = idx[name];
    if (i === undefined || !frame[i]) continue;
    const j = frame[i];
    // CAPSULE, not an infinite cylinder: outside the pelvis..neck span the nearest point is
    // an END of the segment, so a hand raised above the head or dropped below the hips is
    // measured against the shoulder/hip cap instead of being shoved sideways by a column
    // that reaches to infinity. (The MMD side has always used a capsule; clamping `t` here
    // was the cylinder equivalent, and it pushed raised hands outward for no reason —
    // exactly the kind of thing only a violent clip reaches.)
    const tRaw = (j[1] - pel[1]) / dy;
    const t = Math.max(0, Math.min(1, tRaw));
    const cx = pel[0] + (nek[0] - pel[0]) * t; // closest point on the spine segment
    const cy = pel[1] + (nek[1] - pel[1]) * t;
    const cz = pel[2] + (nek[2] - pel[2]) * t;
    // Bump profile: widest across the CHEST/belly (t~0.55) where gesturing hands cross
    // in, narrower at the HIPS (so resting arms-at-side aren't flared out) and the neck.
    let r = skirtOnly ? 0 : radius * (1 - 1.1 * (t - 0.55) * (t - 0.55));
    // Widen to the cloth radius inside the skirt band (see SKIRT). tRaw, not t: the band
    // reaches BELOW the pelvis, where t is clamped to 0 and carries no height information.
    // WRISTS only: the band exists so a resting hand stays on the cloth surface; applying
    // it to the elbows pushed them outward for no reason (the elbow never reaches the
    // skirt), which read as broken arms.
    if (SKIRT.on && isWrist && tRaw <= SKIRT.top && tRaw >= SKIRT.hem) {
      const f = (SKIRT.top - tRaw) / (SKIRT.top - SKIRT.hem); // 0 at the waist, 1 at the hem
      r = Math.max(r, radius * (SKIRT.rTop + (SKIRT.rHem - SKIRT.rTop) * f));
    }
    const dx = j[0] - cx; const dy3 = j[1] - cy; const dz = j[2] - cz;
    const d3 = Math.hypot(dx, dy3, dz);      // true capsule distance
    const dxz = Math.hypot(dx, dz);
    // Raw push, horizontal (keep the joint's height) but gated on the capsule test.
    let ox = 0; let oz = 0;
    if (CAPSULE ? d3 < r : dxz < r) {
      if (dxz > 1e-6) {
        const s = r / dxz;
        ox = cx + dx * s - j[0]; oz = cz + dz * s - j[2];
      } else {
        // Dead on the axis: the direction is undefined. Push FORWARD rather than skip —
        // skipping left the joint buried in the chest (the MMD side does the same).
        ox = 0; oz = r;
      }
    }
    // Rate-limit the CHANGE of the push (see GUARD_RATE_MPS). Applies while easing back to
    // zero too, which is the half that actually stops the pop: without it the correction
    // vanishes the instant the joint clears the capsule.
    if (state && GUARD_RATE_ON) {
      const prev = state.push[i] || (state.push[i] = [0, 0]);
      let ddx = ox - prev[0]; let ddz = oz - prev[1];
      const dl = Math.hypot(ddx, ddz);
      if (dl > lim) { const k = lim / dl; ddx *= k; ddz *= k; }
      ox = prev[0] + ddx; oz = prev[1] + ddz;
      prev[0] = ox; prev[1] = oz;
    }
    if (ox === 0 && oz === 0) continue;
    if (!out) out = frame.slice(); // shallow copy; other joints stay shared
    out[i] = [j[0] + ox, j[1], j[2] + oz]; // push out horizontally, keep height
    // Carry the hand rigidly (see the note above): same delta, no rotation.
    const rx = CARRY_HAND ? HAND_CHILDREN[name] : undefined;
    if (!rx) continue;
    for (const key in idx) {
      if (!rx.test(key)) continue;
      const k = idx[key];
      const f = frame[k];
      if (f === undefined || !f) continue;
      out[k] = [f[0] + ox, f[1], f[2] + oz];
    }
  }
  return out || frame;
}
