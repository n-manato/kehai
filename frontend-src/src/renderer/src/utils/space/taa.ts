import * as THREE from "three";
import { FullScreenQuad } from "three/examples/jsm/postprocessing/Pass.js";
import { CopyShader } from "three/examples/jsm/shaders/CopyShader.js";
import { SPACE } from "./space-config";

/**
 * Temporal antialiasing: sub-pixel camera jitter, plus the matrices a velocity
 * buffer and a reprojection resolve need.
 *
 * WHY THIS EXISTS
 * ---------------
 * The pipeline had no temporal component at all. MSAA averages coverage WITHIN
 * a frame; SMAA infers edges FROM a finished frame. Both treat every frame
 * independently, so when she moves, every high-contrast edge lands on a
 * different sub-pixel position each frame and each frame independently computes
 * its own best answer — the values jump between neighbours and the eye reads it
 * as crawling.
 *
 * Measured, per 1/60 s frame, share of pixels whose luminance moved:
 *
 *     outline pixels   5.4%      body's internal edges   8.6%      flat  0.2%
 *
 * The outline was the complaint, but it is STEADIER than the internal edges —
 * it is simply the most visible victim. This is scene-wide, so the fix has to be
 * scene-wide.
 *
 * WHY NOT three.js's own passes
 * -----------------------------
 * `TAARenderPass` says in its own header "when there is no motion in the scene",
 * and has no history invalidation whatsoever — any camera move blends the old
 * image in at 1/32 per frame, for ever. `SSAARenderPass` is not temporal at all;
 * it renders the scene 16x per frame. Neither is usable here.
 *
 * WHY THE JITTER IS A HAND-WRITTEN MATRIX SKEW
 * --------------------------------------------
 * Both three.js passes jitter with `camera.setViewOffset`, which routes through
 * `updateProjectionMatrix()`. This app does not: utils/space/off-axis.ts calls
 * `projectionMatrix.makePerspective(...)` directly every frame to build an
 * asymmetric frustum, and `makePerspective` ignores the view offset entirely. So
 * `setViewOffset` would be silently discarded — and only when head tracking is
 * live, which headless tests never exercise. Skewing the matrix afterwards is
 * the only mechanism that survives both branches.
 */

/** Radical-inverse van der Corput / Halton, the standard TAA sample sequence. */
function halton(index: number, base: number): number {
  let f = 1;
  let r = 0;
  let i = index;
  while (i > 0) {
    f /= base;
    r += f * (i % base);
    i = Math.floor(i / base);
  }
  return r;
}

/** 8 samples: enough to converge quickly, short enough that a rejected history
 *  is re-earned within an eighth of a second. */
const SAMPLES = 8;
const OFFSETS: [number, number][] = Array.from({ length: SAMPLES }, (_, i) => [
  halton(i + 1, 2) - 0.5,
  halton(i + 1, 3) - 0.5,
]);

export class Taa {
  /** Frame counter driving the jitter sequence. Deliberately NOT dt-driven: the
   *  debug clock freezes dt to 0 while the render loop keeps running, and the
   *  history still needs to converge in that state. */
  private frame = 0;

  /** Jitter applied to the CURRENT frame, in NDC. Kept so it can be subtracted
   *  back out — velocity must be computed from unjittered matrices, or the
   *  jitter itself reads as scene motion and the reprojection chases it. */
  readonly jitter = new THREE.Vector2();

  /** Unjittered view-projection, this frame and last. The velocity pass needs
   *  both; the resolve needs neither, but keeping them here keeps the "who owns
   *  the previous frame" question in one place. */
  readonly viewProj = new THREE.Matrix4();

  readonly prevViewProj = new THREE.Matrix4();

  private hasPrev = false;

  /** The projection exactly as it was BEFORE this class skewed it, and exactly as
   *  it was AFTER. Both are needed because the jitter is applied by mutation:
   *  see the un-skew at the top of `apply()`. */
  private clean = new THREE.Matrix4();

  private wrote = new THREE.Matrix4();

  private hasWrote = false;

  /**
   * True once the scene has held still long enough for the history to be as good
   * as it is going to get, at which point the resolve stops accumulating and just
   * passes the history through.
   *
   * This exists for DETERMINISM, not for speed. `freeze()` only forces `dt` to 0;
   * it does not stop `composer.render()`, so a frozen scene keeps drawing — and
   * the jitter keeps walking the sequence, so every frozen frame is a slightly
   * different image. An exponential history never reaches a fixed point, and
   * tests/visual/scene.spec.ts compares captures with ZERO amplitude tolerance
   * (1 LSB counts as a difference), so `waitUntilStill` would spin for ever.
   * Latching the history stops that dead: once held, consecutive captures are
   * byte-identical.
   */
  converged = false;

  /** Consecutive frames with dt == 0 and an unmoved camera. */
  private stillFrames = 0;

  enabled(): boolean {
    return (SPACE.taa ?? 0) > 0.5;
  }

  /** Whether the resolve should hold the history instead of accumulating into it.
   *  `taaSettle: 0` disables the latch — which is what a MEASUREMENT run wants,
   *  because a latched history reports a perfect 0.000% frame-to-frame difference
   *  whether or not the accumulation underneath it actually works. */
  holding(): boolean {
    return this.converged;
  }

  /**
   * Skew the projection by a sub-pixel offset for this frame.
   *
   * MUST be called AFTER `applyOffAxis()`, and EVERY frame — `resize()` runs
   * from a ResizeObserver and a devicePixelRatio watcher, outside the animation
   * loop, and calls `updateProjectionMatrix()`, which would wipe a latched
   * jitter without the loop ever knowing.
   */
  apply(
    camera: THREE.PerspectiveCamera,
    drawW: number,
    drawH: number,
    dt: number,
  ): void {
    // UN-SKEW FIRST. The jitter is applied by adding to the matrix in place, so
    // it is only correct if the matrix arrives clean each frame — and it does
    // NOT. applyOffAxis() rewrites the projection only while a head is actually
    // being tracked; with no webcam (headless, and every user who never enables
    // it) it returns false and leaves the matrix exactly as it found it, which
    // means last frame's skew is still in it. Adding to that integrates the
    // sequence into an ever-growing frame offset instead of a sub-pixel one — a
    // drift that a frame-to-frame difference measurement cannot see, because
    // each individual step is still sub-pixel.
    //
    // Restoring by COPY rather than by subtracting the offset back off: (a+j)-j
    // is not exactly a in floating point, and a matrix that wobbles in its last
    // bit never compares equal, which would keep the stillness latch below from
    // ever engaging.
    if (this.hasWrote && camera.projectionMatrix.equals(this.wrote)) {
      camera.projectionMatrix.copy(this.clean);
    }
    // matrixWorldInverse is only refreshed by updateMatrixWorld(), which the
    // renderer calls later in the frame — read it now and it is one frame stale.
    // Harmless for the stillness test, fatal for velocity, so fix it here once.
    camera.updateMatrixWorld();
    // Snapshot the clean matrix BEFORE skewing: this is what velocity uses.
    this.clean.copy(camera.projectionMatrix);
    this.prevViewProj.copy(this.viewProj);
    this.viewProj.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
    if (!this.hasPrev) {
      this.prevViewProj.copy(this.viewProj);
      this.hasPrev = true;
    }

    // Stillness, for the latch above. dt == 0 means the debug clock is frozen, so
    // no integrator (pose, springs, particles, grain phase) advances; an unchanged
    // view-projection means the camera did not move either. Everything that could
    // still change the image after that is the jitter, which is exactly what the
    // history is averaging.
    const still = dt === 0 && this.viewProj.equals(this.prevViewProj);
    this.stillFrames = still ? this.stillFrames + 1 : 0;
    const settle = SPACE.taaSettle ?? 0;
    this.converged = settle > 0 && this.stillFrames >= settle;

    // STOP JITTERING ONCE LATCHED, not just accumulating. The resolve is handing
    // the held history straight through, so a further sub-pixel offset cannot
    // improve it — but BokehPass sits DOWNSTREAM of the resolve and renders the
    // scene itself to get depth, so a still-walking jitter kept changing its blur
    // and the final image with it. Measured: frozen captures differed on 0.19% of
    // outline pixels with the jitter still running, and are byte-identical without
    // it. Anything else added after the resolve that reads the camera inherits
    // this for free.
    if (!this.enabled() || this.converged) {
      this.jitter.set(0, 0);
      // Record what is now in the camera even when standing down, so that
      // switching taa off (or latching) mid-run cannot leave the last skew in.
      this.wrote.copy(camera.projectionMatrix);
      this.hasWrote = true;
      return;
    }

    const [ox, oy] = OFFSETS[this.frame % SAMPLES];
    this.frame += 1;
    // NDC spans 2 units across the buffer, hence the factor of 2.
    this.jitter.set((ox * 2) / Math.max(drawW, 1), (oy * 2) / Math.max(drawH, 1));
    // m02 / m12 — the skew terms. Adding here shifts the whole frustum by a
    // fraction of a pixel without touching fov, aspect or the off-axis
    // asymmetry that off-axis.ts just wrote.
    camera.projectionMatrix.elements[8] += this.jitter.x;
    camera.projectionMatrix.elements[9] += this.jitter.y;
    camera.projectionMatrixInverse.copy(camera.projectionMatrix).invert();
    this.wrote.copy(camera.projectionMatrix);
    this.hasWrote = true;
  }

  /** Called when the history is no longer valid for the incoming frame:
   *  a camera teleport, a scene rebuild, a resize. Cheaper to be wrong in this
   *  direction — a discarded history costs one aliased frame, a kept one costs
   *  a smear that never leaves. */
  reset(): void {
    this.converged = false;
    this.stillFrames = 0;
    this.invalidate += 1;
  }

  /** Bumped by `reset()`. The pass watches it rather than being called directly,
   *  so anything that invalidates the history (a camera teleport, a scene
   *  rebuild, a resize) only has to know about `taa`, not about the composer. */
  invalidate = 0;

  /** How many skeletons the velocity pre-pass is tracking — a probe's proof
   *  that the motion vectors actually cover her, not just the environment. */
  velocitySkins = 0;

  state(): {
    enabled: boolean
    converged: boolean
    frame: number
    stillFrames: number
    feedback: number
    clamp: number
    velocitySkins: number
  } {
    return {
      enabled: this.enabled(),
      converged: this.converged,
      frame: this.frame,
      stillFrames: this.stillFrames,
      feedback: SPACE.taaFeedback ?? 0,
      clamp: SPACE.taaClamp ?? 0,
      velocitySkins: this.velocitySkins,
    };
  }
}

export const taa = new Taa();

/* -------------------------------------------------------------------------- */
/* The resolve                                                                */
/* -------------------------------------------------------------------------- */

/**
 * Colour is compared in YCoCg, not RGB. The neighbourhood box has to be tight to
 * kill ghosts, and in RGB the three axes are correlated — a bright edge widens
 * all three at once, so the box swells and lets stale colour through. YCoCg
 * separates luma from chroma, and it is a lossless integer-friendly rotation, so
 * nothing is given up for it.
 */
const YCOCG = /* glsl */`
vec3 toYCoCg(vec3 c) {
  return vec3(
     0.25 * c.r + 0.5 * c.g + 0.25 * c.b,
     0.5  * c.r             - 0.5  * c.b,
    -0.25 * c.r + 0.5 * c.g - 0.25 * c.b);
}
vec3 fromYCoCg(vec3 c) {
  return vec3(c.x + c.y - c.z, c.x + c.z, c.x - c.y - c.z);
}
`;

const RESOLVE_FRAG = /* glsl */`
uniform sampler2D tDiffuse;   // this frame, straight out of GTAO (INPUT res)
uniform sampler2D tHistory;   // what the resolve produced last frame (OUTPUT res)
uniform sampler2D tVelocity;  // UV motion per pixel, from the velocity pre-pass
uniform sampler2D tVelDepth;  // that pass's depth — dilation and disocclusion
uniform sampler2D tPrevDepth; // last frame's depth, LINEARISED, stored below
// Two texel sizes because input and output may differ (taaUpscale < 1): the
// current frame, its velocity and both depths live at INPUT res; the history
// and this very fragment live at OUTPUT res. UVs are shared — normalised
// coordinates are resolution-independent — only pixel-unit maths splits.
uniform vec2 uTexelIn;
uniform vec2 uTexelOut;
uniform vec2 uCamDepth;       // near, far — to linearise tVelDepth
uniform float uFeedback;      // share of history to keep when it is trusted
uniform float uClamp;         // box half-width, in std deviations of the 3x3
uniform float uReject;        // clip distance past which the history is binned
uniform float uMaxVel;        // px/frame beyond which the fetch is a guess
uniform float uFastFeedback;  // feedback ceiling once a pixel is moving fast
uniform float uLock;          // thin-feature lock strength, 0 disables
uniform float uDisocc;        // how much a depth disocclusion kills history
uniform float uValid;         // 0 on the first frame after an invalidation
uniform float uHold;          // 1 = latched: pass the history through untouched
uniform float uDebug;         // 1 = alpha carries the reject amount, not the lock
varying vec2 vUv;

${YCOCG}

/** Push a colour into an axis-aligned box, along the line toward its centre.
 *  Clipping toward the centre rather than clamping per channel keeps the result
 *  on the line between the two colours, so a rejected history desaturates toward
 *  the local mean instead of skewing hue one channel at a time. */
vec3 clipToBox(vec3 p, vec3 bmin, vec3 bmax) {
  vec3 c = 0.5 * (bmax + bmin);
  vec3 e = 0.5 * (bmax - bmin) + 1e-5;
  vec3 v = p - c;
  vec3 a = abs(v / e);
  float m = max(a.x, max(a.y, a.z));
  return m > 1.0 ? c + v / m : p;
}

/**
 * Catmull-Rom history fetch (Jimenez's 5-tap arrangement). A BILINEAR fetch of
 * a moving history is a box filter applied again every frame — the image
 * measurably melts (character sharpness -31% mid-motion). Catmull-Rom's
 * negative lobes undo the box blur each round instead of compounding it; the
 * ringing they can introduce is caught by the neighbourhood clamp downstream,
 * which is exactly why shipped TAAs (UE, CoD, Uncharted) pair these two.
 * Five bilinear taps approximate the full 9-tap bicubic by dropping the
 * corners and renormalising.
 */
vec3 historyCatmullRom(vec2 uv) {
  vec2 samplePos = uv / uTexelOut;
  vec2 tc1 = floor(samplePos - 0.5) + 0.5;
  vec2 f = samplePos - tc1;
  vec2 w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
  vec2 w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
  vec2 w2 = f * (0.5 + f * (2.0 - 1.5 * f));
  vec2 w3 = f * f * (-0.5 + 0.5 * f);
  vec2 w12 = w1 + w2;
  vec2 tc0 = (tc1 - 1.0) * uTexelOut;
  vec2 tc3 = (tc1 + 2.0) * uTexelOut;
  vec2 tc12 = (tc1 + w2 / w12) * uTexelOut;
  vec3 c = texture2D(tHistory, vec2(tc12.x, tc0.y)).rgb * (w12.x * w0.y)
         + texture2D(tHistory, vec2(tc0.x, tc12.y)).rgb * (w0.x * w12.y)
         + texture2D(tHistory, vec2(tc12.x, tc12.y)).rgb * (w12.x * w12.y)
         + texture2D(tHistory, vec2(tc3.x, tc12.y)).rgb * (w3.x * w12.y)
         + texture2D(tHistory, vec2(tc12.x, tc3.y)).rgb * (w12.x * w3.y);
  float wsum = w12.x * w0.y + w0.x * w12.y + w12.x * w12.y + w3.x * w12.y + w12.x * w3.y;
  return max(c / wsum, 0.0);
}

void main() {
  if (uHold > 0.5) {
    // Latched: byte-for-byte passthrough, so frozen captures stay identical.
    gl_FragColor = texelFetch(tHistory, ivec2(gl_FragCoord.xy), 0);
    return;
  }

  // STAGE 4: reprojection. The velocity buffer says how far this pixel's
  // surface moved since last frame (bones + camera for her, camera for the
  // environment), so the history is fetched where the surface WAS. This is what
  // stage 2 could not do — without it, ordinary motion became either a rejected
  // history (the old flicker) or an admitted stale one (the measured
  // regression: internal edges +76% at t=20). With it, an aligned history is
  // the RIGHT history, and the clamp below is demoted from doing all the work
  // to catching what velocity cannot describe: disocclusions, morph-driven
  // motion (zero velocity by design), and movers that teleport.
  //
  // DILATED by closest depth. The colour buffer is 8x MSAA, so her rim is
  // blended over the background for a pixel or two — but the velocity buffer
  // is single-sample, and right at the silhouette the centre tap lands on the
  // BACKGROUND and reprojects rim colour with the wrong motion. Measured
  // before dilation: outline pixels +46% at t=20 while internal edges were
  // already -35%. The nearest surface in the 3x3 is the one the rim colour
  // belongs to, so its velocity wins. Same rescue for sub-pixel hair strands
  // the single-sample raster misses entirely.
  vec2 bestUv = vUv;
  float bestD = texture2D(tVelDepth, vUv).x;
  for (int y = -1; y <= 1; y++) {
    for (int x = -1; x <= 1; x++) {
      vec2 o = vUv + vec2(float(x), float(y)) * uTexelIn;
      float d = texture2D(tVelDepth, o).x;
      if (d < bestD) { bestD = d; bestUv = o; }
    }
  }
  vec2 vel = texture2D(tVelocity, bestUv).xy;
  vec2 histUv = vUv - vel;
  float speedPx = length(vel / uTexelOut);

  float valid = uValid;
  // Off-screen last frame, or moving implausibly fast (a gesture snap, a
  // wrapped traffic mover): there is nothing true to fetch. Ghost-averse
  // brief: drop it and pay one aliased frame.
  if (any(lessThan(histUv, vec2(0.0))) || any(greaterThan(histUv, vec2(1.0)))
      || speedPx > uMaxVel) valid = 0.0;

  // DEPTH DISOCCLUSION (what FSR2 calls its depth-based rejection). The clamp
  // rejects by COLOUR, so a newly-uncovered surface that happens to resemble
  // its old occluder slips through as a ghost. Geometry does not lie — but it
  // does WOBBLE: the jitter re-rasters every depth edge each frame, so a
  // point-vs-point compare fires on every hair strand every frame and was
  // measured killing the history on 7.7% of internal-edge pixels (vs 0.34%
  // with it off). Two guards make it a disocclusion test instead of an edge
  // detector: only pixels that actually MOVED can be disocclusions (a still
  // pixel's history is trivially the same surface), and the current depth is
  // compared against the RANGE of last frame's 3x3 around the fetch — the
  // sub-pixel wobble stays inside that range, a genuinely different surface
  // does not. Thin occluders slip past the range test, but a thin occluder's
  // background was visible in the neighbourhood anyway, which is exactly the
  // case the colour clamp already handles. This is what makes the
  // thin-feature lock below safe to hold as hard as it does.
  float disocc = 0.0;
  if (uValid > 0.5 && speedPx > 0.5) {
    float zRaw = texture2D(tVelDepth, vUv).x * 2.0 - 1.0;
    float ldCur = (2.0 * uCamDepth.x * uCamDepth.y)
                / (uCamDepth.y + uCamDepth.x - zRaw * (uCamDepth.y - uCamDepth.x));
    float dLo = 1e9;
    float dHi = -1e9;
    for (int y = -1; y <= 1; y++) {
      for (int x = -1; x <= 1; x++) {
        float d = texture2D(tPrevDepth, histUv + vec2(float(x), float(y)) * uTexelIn).x;
        dLo = min(dLo, d);
        dHi = max(dHi, d);
      }
    }
    if (ldCur < dLo * 0.9 || ldCur > dHi * 1.1) disocc = uDisocc;
    valid *= 1.0 - disocc;
  }

  // texelFetch when still, Catmull-Rom when moving. A varying is not
  // bit-exactly the texel centre, so any filtered tap at "the same place"
  // folds a few thousandths of each neighbour in every frame and melts a
  // static image; a MOVING fetch needs filtering — but bicubic, not bilinear,
  // or the filtering itself becomes the blur (see historyCatmullRom).
  vec4 hist = speedPx < 0.01
    ? texelFetch(tHistory, ivec2(gl_FragCoord.xy), 0)
    : vec4(historyCatmullRom(histUv), 1.0);

  vec3 cur = texture2D(tDiffuse, vUv).rgb;
  vec3 curY = toYCoCg(cur);

  // 3x3 moments AND hard extremes. The variance box (mean +- k*sigma) is the
  // useful one — it ignores a single outlier texel that would blow a min/max box
  // wide open — but intersecting it with the true extremes stops it claiming a
  // range the neighbourhood does not actually contain.
  vec3 m1 = vec3(0.0);
  vec3 m2 = vec3(0.0);
  vec3 lo = vec3(1e6);
  vec3 hi = vec3(-1e6);
  float crossN = 0.0;
  float crossS = 0.0;
  float crossE = 0.0;
  float crossW = 0.0;
  for (int y = -1; y <= 1; y++) {
    for (int x = -1; x <= 1; x++) {
      vec3 s = toYCoCg(texture2D(tDiffuse, vUv + vec2(float(x), float(y)) * uTexelIn).rgb);
      m1 += s;
      m2 += s * s;
      lo = min(lo, s);
      hi = max(hi, s);
      // The four edge-neighbour lumas, for the thin-feature test below. The
      // literal index comparisons vanish when the driver unrolls the loop.
      if (x == 0 && y == -1) crossS = s.x;
      if (x == 0 && y ==  1) crossN = s.x;
      if (x == -1 && y == 0) crossW = s.x;
      if (x ==  1 && y == 0) crossE = s.x;
    }
  }
  vec3 mean = m1 / 9.0;
  vec3 sigma = sqrt(max(vec3(0.0), m2 / 9.0 - mean * mean));
  vec3 bmin = max(lo, mean - uClamp * sigma);
  vec3 bmax = min(hi, mean + uClamp * sigma);

  // THIN-FEATURE LOCK (FSR2's "lock" idea, simplified). A one-pixel ink line or
  // hair strand only rasterises on the frames the jitter happens to catch it;
  // on the misses, the accumulated line in the history is exactly what the
  // clamp is built to reject, so the feature blinks at jitter frequency — the
  // residual outline oscillation at low motion. Detection: this pixel's luma
  // pokes out past ALL four edge neighbours by a margin scaled to the local
  // contrast, i.e. the raster caught a sub-pixel feature THIS frame. The lock
  // rides in the history's alpha, reprojected with it, refreshed on every
  // re-detection and decaying over 8 frames (one jitter cycle) otherwise.
  // While locked: keep more history and widen the clamp box, so the misses do
  // not erase what the hits accumulated. Broken immediately by invalidation,
  // disocclusion, or real motion — that is what keeps it from ghosting.
  float band = max(bmax.x - bmin.x, 1e-3);
  float thin = uLock > 0.001
    && (curY.x > max(max(crossN, crossS), max(crossE, crossW)) + 0.1 * band
     || curY.x < min(min(crossN, crossS), min(crossE, crossW)) - 0.1 * band)
    ? 1.0 : 0.0;
  float histLock = speedPx < 0.01
    ? texelFetch(tHistory, ivec2(gl_FragCoord.xy), 0).a
    : texture2D(tHistory, histUv).a;
  // While the reject-debug view is up, alpha carried the diagnostic, not locks.
  if (uDebug > 0.5) histLock = 0.0;
  if (valid < 0.5 || speedPx > 4.0) histLock = 0.0;
  float effLock = histLock * uLock;
  float newLock = max(thin, histLock - 0.125);

  // Widen the box for locked pixels — around its centre, so the widening does
  // not drag the admissible range toward either extreme.
  vec3 mid = 0.5 * (bmin + bmax);
  vec3 halfw = 0.5 * (bmax - bmin) * (1.0 + 1.5 * effLock);
  bmin = mid - halfw;
  bmax = mid + halfw;
  // The box must contain THIS pixel, or the blend below pulls toward a colour
  // that appears nowhere in the neighbourhood. A tight sigma can otherwise
  // exclude the very sample being resolved.
  bmin = min(bmin, curY);
  bmax = max(bmax, curY);

  vec3 histY = toYCoCg(hist.rgb);
  vec3 clipped = clipToBox(histY, bmin, bmax);

  // How far the history had to travel to become admissible, measured against the
  // local contrast so a flat area and a hard edge are judged on the same scale.
  // Far == this pixel is showing something the previous frame never saw, i.e. a
  // disocclusion. Ghost-averse brief: bin it rather than clip it and hope.
  float moved = length(clipped - histY);
  float reject = clamp(moved / (length(bmax - bmin) + 1e-4), 0.0, 1.0);
  float trust = 1.0 - smoothstep(0.0, max(uReject, 1e-4), reject);
  // VELOCITY-DEPENDENT FEEDBACK, the standard companion to reprojection. A
  // long history is what a nearly-still edge needs and exactly what a fast one
  // must not have: reprojection carries colour to the right place, but it
  // cannot invent the surface's new SILHOUETTE, so on something swinging hard
  // (her tie, the skirt hem) a 20-frame history keeps drawing the old shape.
  // Measured under a hard body swing: TAA's frame differed from the true frame
  // on up to 9.1% of the region, with the tie visibly narrower and a navy
  // panel on it missing entirely — that is the "the front flashes past"
  // artefact. Fading the ceiling out between 2 and 20 px/frame leaves the
  // still-and-slow case (the original crawling-edge complaint) untouched.
  float fast = smoothstep(2.0, 20.0, speedPx);
  float feedback = mix(uFeedback, min(uFeedback, uFastFeedback), fast);
  // Locked pixels hold almost all of their history: 0.985 means the alternating
  // raster hits/misses of a sub-pixel feature move the output by ~1.5% per
  // frame instead of the normal ingest — steady to the eye, and the lock's
  // 8-frame expiry bounds how long a stale feature can survive being wrong.
  // The lock is already broken above 4 px/frame, so it never fights the fade.
  // (No backticks in here: these shaders are JS template literals, and one in a
  // comment silently ends the string — the build failed with "Expected ;".)
  float w = mix(feedback, 0.985, effLock) * trust * valid;

  // ALPHA CARRIES THE LOCK (or, in the reject-debug view, the diagnostic), and
  // the colour is always the real result. An earlier version painted the debug
  // view here instead — which fed the debug colour straight back in as next
  // frame's history, so every reading after the first was of the readout, not
  // of the image. The copy-out writes 1.0 downstream, so nothing else sees it.
  gl_FragColor = vec4(fromYCoCg(mix(curY, clipped, w)), uDebug > 0.5 ? reject : newLock);
}
`;

/** Store this frame's depth, linearised, so next frame's resolve can compare
 *  geometry instead of guessing from colour. Same near/far mapping as the
 *  line pass's ld(). */
const DEPTH_STORE_FRAG = /* glsl */`
uniform sampler2D tDepth;
uniform vec2 uCamDepth; // near, far
varying vec2 vUv;
void main() {
  float z = texture2D(tDepth, vUv).x * 2.0 - 1.0;
  float lin = (2.0 * uCamDepth.x * uCamDepth.y)
            / (uCamDepth.y + uCamDepth.x - z * (uCamDepth.y - uCamDepth.x));
  gl_FragColor = vec4(lin, 0.0, 0.0, 1.0);
}
`;

/**
 * Hand the resolved frame down the chain — and, when asked, the resolve's own
 * decision instead of the image. Alpha holds how far the history had to be
 * clipped to be admissible; red shows that, green shows what survived of it.
 * Reading it: black = the history was taken as-is, red = this pixel is showing
 * something last frame never had.
 */
const COPY_FRAG = /* glsl */`
uniform sampler2D tDiffuse;
uniform sampler2D tVelocity;
uniform vec2 uTexel;
uniform float uReject;
uniform float uDebug;
uniform float uSharpen;
varying vec2 vUv;
void main() {
  vec4 c = texture2D(tDiffuse, vUv);
  if (uDebug > 1.5) {
    // Velocity view. 0.5 grey = still; R/G offset = screen motion at 0.1 per
    // px/frame (saturates at ±5); blue marks skinned (her) pixels. The stage-3
    // gate is read here: a FROZEN clock must show flat grey — vrm.update(0)
    // keeps prev == curr bones — and a STEPPED clock must light her up while
    // the backdrop stays near-grey (camera drift is sub-pixel per frame).
    vec4 v = texture2D(tVelocity, vUv);
    gl_FragColor = vec4(0.5 + (v.xy / uTexel) * 0.1, v.z * 0.25, 1.0);
    return;
  }
  if (uDebug > 0.5) {
    float trust = 1.0 - smoothstep(0.0, max(uReject, 1e-4), c.a);
    gl_FragColor = vec4(c.a, trust, 0.0, 1.0);
    return;
  }
  vec3 col = c.rgb;
  // Contrast-adaptive sharpen, the industry companion to a temporal resolve
  // (AMD CAS after TAA, RCAS after DLSS): an unsharp mask against the cross
  // average, CLAMPED into the local extremes — the clamp is what separates
  // this from a naive unsharp mask, because it stops halos before they form.
  // Outside the history loop on purpose: the resolve's feedback would compound
  // any sharpening that entered the accumulator, frame after frame.
  if (uSharpen > 0.001) {
    vec3 tN = texture2D(tDiffuse, vUv + vec2(0.0, uTexel.y)).rgb;
    vec3 tS = texture2D(tDiffuse, vUv - vec2(0.0, uTexel.y)).rgb;
    vec3 tE = texture2D(tDiffuse, vUv + vec2(uTexel.x, 0.0)).rgb;
    vec3 tW = texture2D(tDiffuse, vUv - vec2(uTexel.x, 0.0)).rgb;
    vec3 lo = min(col, min(min(tN, tS), min(tE, tW)));
    vec3 hi = max(col, max(max(tN, tS), max(tE, tW)));
    col = clamp(col + (col - (tN + tS + tE + tW) * 0.25) * uSharpen, lo, hi);
  }
  gl_FragColor = vec4(col, 1.0);
}
`;

/**
 * Where this sits: sceneComposer(Render -> linePass -> GTAO) -> **THIS** ->
 * postComposer(copy/sharpen -> grade -> DOF -> bloom -> diffusion -> Output ->
 * SMAA).
 *
 * NOT a composer pass any more, and that is the point (TAAU): an EffectComposer
 * ping-pongs two same-sized buffers, so a mid-chain pass can never write more
 * pixels than it reads. Splitting the chain at exactly the old TaaPass seam
 * lets the scene side render at `taaUpscale` scale while the history — and
 * everything after it — lives at full output resolution. The jitter makes the
 * low-res samples land on different output pixels each frame, so the history
 * genuinely RECONSTRUCTS the missing resolution rather than stretching it.
 *
 * The old ordering constraints all survive the split: linePass is still the
 * first needsSwap pass of the scene side, and the resolve still sees a sharp,
 * ungraded, grain-free frame — grain lives in gradePass on the post side.
 */
export class TaaResolve {
  private prevRT: THREE.WebGLRenderTarget;

  private curRT: THREE.WebGLRenderTarget;

  /** Last frame's linearised depth, at input res — the disocclusion referee. */
  private prevDepthRT: THREE.WebGLRenderTarget;

  private resolveMat: THREE.ShaderMaterial;

  private depthMat: THREE.ShaderMaterial;

  /** The bridge into the post chain — handed to a ShaderPass by the viewer.
   *  Public because the composer owns its lifecycle from there. */
  readonly copyMat: THREE.ShaderMaterial;

  private quad: FullScreenQuad;

  /** What the post chain should read this frame. Ping-pong alternates the
   *  underlying texture, so the viewer re-reads this getter every frame. */
  get output(): THREE.Texture {
    return this.lastWritten;
  }

  private lastWritten: THREE.Texture;

  /** No history worth sampling yet — first frame, or just invalidated. */
  private valid = false;

  private seenInvalidate = 0;

  constructor(outW: number, outH: number, inW: number, inH: number) {
    this.prevRT = TaaResolve.makeRT(outW, outH);
    this.curRT = TaaResolve.makeRT(outW, outH);
    this.lastWritten = this.prevRT.texture;
    this.prevDepthRT = new THREE.WebGLRenderTarget(Math.max(1, inW), Math.max(1, inH), {
      type: THREE.HalfFloatType,
      format: THREE.RedFormat,
      // Nearest is load-bearing: bilinear across a depth EDGE invents
      // intermediate depths that exist on neither surface, and every one of
      // them would read as a disocclusion.
      minFilter: THREE.NearestFilter,
      magFilter: THREE.NearestFilter,
      depthBuffer: false,
      stencilBuffer: false,
    });
    this.prevDepthRT.texture.name = "Taa.prevDepth";
    this.resolveMat = new THREE.ShaderMaterial({
      uniforms: {
        tDiffuse: { value: null },
        tHistory: { value: null },
        tVelocity: { value: null },
        tVelDepth: { value: null },
        tPrevDepth: { value: this.prevDepthRT.texture },
        uTexelIn: { value: new THREE.Vector2(1 / Math.max(inW, 1), 1 / Math.max(inH, 1)) },
        uTexelOut: { value: new THREE.Vector2(1 / Math.max(outW, 1), 1 / Math.max(outH, 1)) },
        uCamDepth: { value: new THREE.Vector2(0.1, 100) },
        uFeedback: { value: 0.85 },
        uClamp: { value: 1.0 },
        uReject: { value: 0.5 },
        uMaxVel: { value: 80 },
        uFastFeedback: { value: 0.6 },
        uLock: { value: 1 },
        uDisocc: { value: 1 },
        uValid: { value: 0 },
        uHold: { value: 0 },
        uDebug: { value: 0 },
      },
      vertexShader: CopyShader.vertexShader,
      fragmentShader: RESOLVE_FRAG,
      depthTest: false,
      depthWrite: false,
      blending: THREE.NoBlending,
    });
    this.depthMat = new THREE.ShaderMaterial({
      uniforms: {
        tDepth: { value: null },
        uCamDepth: { value: this.resolveMat.uniforms.uCamDepth.value },
      },
      vertexShader: CopyShader.vertexShader,
      fragmentShader: DEPTH_STORE_FRAG,
      depthTest: false,
      depthWrite: false,
      blending: THREE.NoBlending,
    });
    this.copyMat = new THREE.ShaderMaterial({
      uniforms: {
        tDiffuse: { value: null },
        tVelocity: { value: null },
        uTexel: { value: new THREE.Vector2(1 / Math.max(outW, 1), 1 / Math.max(outH, 1)) },
        uReject: { value: 0.5 },
        uDebug: { value: 0 },
        uSharpen: { value: 0 },
      },
      vertexShader: CopyShader.vertexShader,
      fragmentShader: COPY_FRAG,
      depthTest: false,
      depthWrite: false,
      blending: THREE.NoBlending,
    });
    this.quad = new FullScreenQuad(this.resolveMat);
  }

  /** Linear, because a REPROJECTED fetch lands between texels and needs the
   *  filter. The still case — where linear-at-the-centre would fold a few
   *  thousandths of each neighbour in per frame and melt a static image — is
   *  handled in the shader with texelFetch, which bypasses filtering entirely. */
  private static makeRT(width: number, height: number): THREE.WebGLRenderTarget {
    const rt = new THREE.WebGLRenderTarget(Math.max(1, width), Math.max(1, height), {
      type: THREE.HalfFloatType, // the composer chain is linear HDR, not 0..1
      minFilter: THREE.LinearFilter,
      magFilter: THREE.LinearFilter,
      depthBuffer: false,
      stencilBuffer: false,
    });
    rt.texture.name = "Taa.history";
    return rt;
  }

  /** The velocity pre-pass output. Wired once at construction time by the
   *  viewer; kept as a setter because the pre-pass is a sibling (it renders
   *  geometry in the animate loop), not something this pass owns. */
  setVelocity(texture: THREE.Texture, depth: THREE.Texture | null = null): void {
    this.resolveMat.uniforms.tVelocity.value = texture;
    this.resolveMat.uniforms.tVelDepth.value = depth;
    this.depthMat.uniforms.tDepth.value = depth;
    this.copyMat.uniforms.tVelocity.value = texture;
  }

  setSize(outW: number, outH: number, inW: number, inH: number): void {
    const ow = Math.max(1, Math.floor(outW));
    const oh = Math.max(1, Math.floor(outH));
    const iw = Math.max(1, Math.floor(inW));
    const ih = Math.max(1, Math.floor(inH));
    // Only invalidate when a size ACTUALLY changed. Every `__space()` call ends
    // in resize(), which lands here with identical dimensions — so without this
    // test, tuning any unrelated knob threw the history away and TAA looked worse
    // than it is for the next few frames, right when someone is judging it.
    if (ow === this.prevRT.width && oh === this.prevRT.height
      && iw === this.prevDepthRT.width && ih === this.prevDepthRT.height) return;
    this.prevRT.setSize(ow, oh);
    this.curRT.setSize(ow, oh);
    this.prevDepthRT.setSize(iw, ih);
    this.resolveMat.uniforms.uTexelOut.value.set(1 / ow, 1 / oh);
    this.resolveMat.uniforms.uTexelIn.value.set(1 / iw, 1 / ih);
    this.copyMat.uniforms.uTexel.value.set(1 / ow, 1 / oh);
    // A resized history is a stretched one. Cheaper to redraw one aliased frame.
    taa.reset();
  }

  /**
   * Resolve this frame. Reads the scene composer's output (any resolution),
   * writes the full-res result reachable via `output`. When the latch holds,
   * nothing is rendered at all — the previous output IS the frame, which is
   * both the determinism guarantee and a free idle-time saving.
   */
  render(
    renderer: THREE.WebGLRenderer,
    camera: THREE.PerspectiveCamera,
    input: THREE.Texture,
  ): void {
    if (taa.invalidate !== this.seenInvalidate) {
      this.seenInvalidate = taa.invalidate;
      this.valid = false;
    }
    if (this.valid && taa.holding()) return; // latched: last output stands

    const u = this.resolveMat.uniforms;
    u.tDiffuse.value = input;
    u.tHistory.value = this.prevRT.texture;
    u.uCamDepth.value.set(camera.near, camera.far);
    // Capped below 1: at exactly 1 the current frame stops contributing and the
    // history becomes immortal, which is the TAARenderPass failure mode.
    u.uFeedback.value = Math.min(0.98, Math.max(0, SPACE.taaFeedback ?? 0.85));
    u.uClamp.value = Math.max(0, SPACE.taaClamp ?? 1);
    u.uReject.value = Math.max(0.01, SPACE.taaReject ?? 0.5);
    u.uMaxVel.value = Math.max(1, SPACE.taaMaxVel ?? 80);
    u.uFastFeedback.value = Math.min(0.98, Math.max(0, SPACE.taaFastFeedback ?? 0.6));
    u.uLock.value = Math.min(1, Math.max(0, SPACE.taaLock ?? 1));
    u.uDisocc.value = Math.min(1, Math.max(0, SPACE.taaDisocc ?? 1));
    u.uValid.value = this.valid ? 1 : 0;
    u.uHold.value = 0;
    u.uDebug.value = (SPACE.taaDebug ?? 0) === 1 ? 1 : 0;

    // Ping-pong rather than resolve-then-copy-back: the resolve samples the
    // history, so it cannot also be writing it.
    const restore = renderer.getRenderTarget();
    this.quad.material = this.resolveMat;
    renderer.setRenderTarget(this.curRT);
    this.quad.render(renderer);

    // Stash this frame's depth for next frame's disocclusion test.
    this.quad.material = this.depthMat;
    renderer.setRenderTarget(this.prevDepthRT);
    this.quad.render(renderer);
    renderer.setRenderTarget(restore);

    this.lastWritten = this.curRT.texture;
    const swap = this.prevRT;
    this.prevRT = this.curRT;
    this.curRT = swap;
    this.valid = true;
  }

  /** Per-frame wiring for the post chain's input pass — runs whether TAA is on
   *  or off, because the bridge pass is now the only road into the post chain. */
  updateCopy(input: THREE.Texture, taaOn: boolean): void {
    this.copyMat.uniforms.tDiffuse.value = input;
    this.copyMat.uniforms.uReject.value = Math.max(0.01, SPACE.taaReject ?? 0.5);
    this.copyMat.uniforms.uDebug.value = taaOn ? Math.max(0, SPACE.taaDebug ?? 0) : 0;
    // Sharpen exists to buy back what the temporal accumulation costs; with TAA
    // off there is nothing to buy back, and `taa: 0` must stay the untouched
    // pre-TAA image.
    this.copyMat.uniforms.uSharpen.value = taaOn ? Math.max(0, SPACE.taaSharpen ?? 0) : 0;
  }

  dispose(): void {
    this.prevRT.dispose();
    this.curRT.dispose();
    this.prevDepthRT.dispose();
    this.resolveMat.dispose();
    this.depthMat.dispose();
    this.copyMat.dispose();
    this.quad.dispose();
  }
}

/* -------------------------------------------------------------------------- */
/* The velocity buffer                                                        */
/* -------------------------------------------------------------------------- */

/**
 * Vertex shader for the motion-vector pass. The three-supplied chunks handle
 * the CURRENT position (morphs, skinning, instancing all included, with the
 * renderer feeding boneTexture / morph uniforms per object as usual); the
 * PREVIOUS position is hand-rolled from a snapshot texture.
 *
 * THE ALGEBRA THAT MAKES THIS SMALL: Skeleton.update() writes WORLD-space skin
 * matrices (bone.matrixWorld * boneInverse), and in the default "attached" bind
 * mode SkinnedMesh refreshes bindMatrixInverse to inverse(matrixWorld) every
 * frame — so in `modelMatrix * bindMatrixInverse * Σ w·bone·bindMatrix·pos` the
 * outer two matrices cancel, and the bone matrices ALONE carry every motion
 * source: the gesture pose, the spring bones, the Bullet twintails, and the
 * root travel bodyDepth applies. One texture snapshot per skeleton is the whole
 * previous-frame state; no per-mesh matrices, no root bookkeeping.
 *
 * Everything NOT skinned is treated as static in object space (prev = curr), so
 * its velocity is pure camera motion — exact for the environment. The traffic
 * movers and particles that translate themselves come out wrong and fall to the
 * resolve's clamp, as designed (flat-area variation measured at 0.2%).
 *
 * MORPHS ARE INCLUDED, and on this model that is not optional: EVERY mesh —
 * face, body AND all the skirt/coat panels — carries the same 83 morph targets,
 * with ~11 active at full weight. An earlier version used the current
 * influences for both positions, so all morph-driven motion had zero velocity,
 * the resolve fetched history from where the surface used to be ON SCREEN
 * rather than where the surface was, and the previous shape lingered — visible
 * as the skirt's inside flickering with a glimpse of its other side. The
 * previous influences therefore get snapshotted per mesh alongside the bones,
 * and the previous position re-runs the morph accumulation with them.
 */
const VELOCITY_VERT = /* glsl */`
#include <common>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
uniform mat4 uCurrVP;   // unjittered — the raster below uses the JITTERED
uniform mat4 uPrevVP;   // camera, but velocity must not see the jitter walk
#ifdef USE_MORPHTARGETS
// Same layout as three's own morphTargetInfluences, one frame behind. Sized by
// MORPHTARGETS_COUNT, so the program variant three compiles per morph count
// gets a correctly sized array and the per-draw upload matches it.
uniform float uPrevMorph[ MORPHTARGETS_COUNT ];
uniform float uPrevMorphBase;
#endif
#ifdef USE_SKINNING
uniform highp sampler2D uPrevBoneTexture;
mat4 getPrevBoneMatrix( const in float i ) {
  int size = textureSize( uPrevBoneTexture, 0 ).x;
  int j = int( i ) * 4;
  int x = j % size;
  int y = j / size;
  return mat4(
    texelFetch( uPrevBoneTexture, ivec2( x, y ), 0 ),
    texelFetch( uPrevBoneTexture, ivec2( x + 1, y ), 0 ),
    texelFetch( uPrevBoneTexture, ivec2( x + 2, y ), 0 ),
    texelFetch( uPrevBoneTexture, ivec2( x + 3, y ), 0 ) );
}
#endif
varying vec4 vCurr;
varying vec4 vPrev;
varying float vSkinned;
void main() {
  #include <begin_vertex>
  // PREVIOUS object-space position: the same morph accumulation three's
  // <morphtarget_vertex> does, driven by last frame's weights.
  vec3 prevObj = vec3( position );
  #ifdef USE_MORPHTARGETS
    prevObj *= uPrevMorphBase;
    for ( int i = 0; i < MORPHTARGETS_COUNT; i ++ ) {
      if ( uPrevMorph[ i ] != 0.0 ) prevObj += getMorph( gl_VertexID, i, 0 ).xyz * uPrevMorph[ i ];
    }
  #endif
  #include <morphtarget_vertex>
  #include <skinbase_vertex>
  vec4 prevWorld;
  vSkinned = 0.0;
  #ifdef USE_SKINNING
    // Previous world position: last frame's morphs through last frame's bones.
    vec4 pv = bindMatrix * vec4( prevObj, 1.0 );
    prevWorld =
        skinWeight.x * getPrevBoneMatrix( skinIndex.x ) * pv
      + skinWeight.y * getPrevBoneMatrix( skinIndex.y ) * pv
      + skinWeight.z * getPrevBoneMatrix( skinIndex.z ) * pv
      + skinWeight.w * getPrevBoneMatrix( skinIndex.w ) * pv;
    vSkinned = 1.0;
  #endif
  #include <skinning_vertex>
  vec4 obj = vec4( transformed, 1.0 );
  vec4 prevLocal = vec4( prevObj, 1.0 );
  #ifdef USE_INSTANCING
    obj = instanceMatrix * obj;
    prevLocal = instanceMatrix * prevLocal;
  #endif
  vec4 currWorld = modelMatrix * obj;
  #ifndef USE_SKINNING
    // Not skinned: still in object space, so only the morphs and the camera can
    // have moved it. modelMatrix is this frame's — a self-animating object that
    // moves its own transform reads as static and falls to the resolve's clamp.
    prevWorld = modelMatrix * prevLocal;
  #endif
  vCurr = uCurrVP * currWorld;
  vPrev = uPrevVP * prevWorld;
  gl_Position = projectionMatrix * viewMatrix * currWorld;
  gl_PointSize = 1.0; // Points render through the override too; keep them legal
}
`;

const VELOCITY_FRAG = /* glsl */`
varying vec4 vCurr;
varying vec4 vPrev;
varying float vSkinned;
void main() {
  // Clip -> NDC per fragment (perspective-correct), NDC delta -> UV delta.
  vec2 vel = (vCurr.xy / vCurr.w - vPrev.xy / vPrev.w) * 0.5;
  gl_FragColor = vec4(vel, vSkinned, 1.0);
}
`;

/**
 * The motion-vector pre-pass. A sibling of the line pass's normalRT render in
 * the animate loop — NOT a composer pass, because it renders geometry, not a
 * fullscreen quad. MRT was ruled out (composer RT is count:1 and every material
 * in the scene speaks GLSL1 gl_FragColor), so this is one extra scene render;
 * the frame already contains 3-5 of those.
 *
 * Call `render()` before composer.render() each frame TAA is accumulating, and
 * `snapshot()` after composer.render() — after, because Skeleton.update() runs
 * several times inside the composer (normal pass / RenderPass / GTAO / Bokeh)
 * with identical values, and the snapshot must be of the pose this frame was
 * actually drawn with, before the next vrm.update() moves the bones.
 */
export class TaaVelocity {
  readonly rt: THREE.WebGLRenderTarget;

  private mat: THREE.ShaderMaterial;

  private skins = new Map<THREE.Skeleton, { arr: Float32Array; tex: THREE.DataTexture }>();

  /** Previous morph weights, PER MESH (each mesh owns its own influences array,
   *  even when several share one skeleton and one morph set). */
  private morphs = new Map<THREE.Mesh, { prev: Float32Array; relative: boolean }>();

  /** True while the snapshot describes the frame before this one. Cleared by
   *  render(); if it is stale (TAA was just toggled on), the snapshot is taken
   *  NOW, so the first frame reads zero velocity instead of garbage. */
  private fresh = false;

  private clearColor = new THREE.Color();

  constructor(width: number, height: number) {
    this.rt = new THREE.WebGLRenderTarget(Math.max(1, width), Math.max(1, height), {
      type: THREE.HalfFloatType, // velocities are signed and sub-texel
      minFilter: THREE.NearestFilter,
      magFilter: THREE.NearestFilter,
      depthBuffer: true, // self-occlusion: the near hand's motion, not the torso's
      stencilBuffer: false,
    });
    this.rt.texture.name = "Taa.velocity";
    // Depth as a TEXTURE, not just a buffer: the resolve dilates velocity by
    // closest depth (see tVelDepth there), which needs to read this back.
    this.rt.depthTexture = new THREE.DepthTexture(Math.max(1, width), Math.max(1, height));
    this.mat = new THREE.ShaderMaterial({
      uniforms: {
        uCurrVP: { value: new THREE.Matrix4() },
        uPrevVP: { value: new THREE.Matrix4() },
        uPrevBoneTexture: { value: null },
        uPrevMorph: { value: null },
        uPrevMorphBase: { value: 1 },
      },
      vertexShader: VELOCITY_VERT,
      fragmentShader: VELOCITY_FRAG,
      // `side` is set PER DRAW from the real material — see the hook in
      // register(). It must never be forced here: this buffer's whole job is to
      // describe the surface the COLOUR pass actually drew, and culling is part
      // of that. Forcing DoubleSide was a real bug — looking up under the skirt,
      // the single-sided panels are culled in colour (so the legs show) but were
      // still rasterised here, so the legs inherited the SKIRT's depth and
      // motion and the resolve dragged skirt-coloured history across them. That
      // is the "the other side of the skirt flashes past" artefact.
      blending: THREE.NoBlending,
    });
  }

  get texture(): THREE.Texture {
    return this.rt.texture;
  }

  /** Per-skeleton previous-bone snapshot, plus the per-mesh hook that injects it.
   *  The override material is SHARED by every mesh in the pass, so per-mesh data
   *  can only travel through onBeforeRender — three calls it before uploading
   *  uniforms, and uniformsNeedUpdate forces the re-upload per draw. */
  private register(mesh: THREE.Mesh): void {
    const skinned = mesh as THREE.SkinnedMesh;
    let tex: THREE.DataTexture | null = null;
    if (skinned.isSkinnedMesh) {
      const { skeleton } = skinned;
      let rec = this.skins.get(skeleton);
      if (!rec) {
        // computeBoneTexture() pads boneMatrices to the texture's square layout,
        // so after it, boneMatrices.length IS the texel count — copy verbatim.
        if (!skeleton.boneTexture) skeleton.computeBoneTexture();
        const size = (skeleton.boneTexture as THREE.DataTexture).image.width;
        const arr = new Float32Array(size * size * 4);
        arr.set(skeleton.boneMatrices);
        const t = new THREE.DataTexture(arr, size, size, THREE.RGBAFormat, THREE.FloatType);
        t.needsUpdate = true;
        rec = { arr, tex: t };
        this.skins.set(skeleton, rec);
        taa.velocitySkins = this.skins.size;
      }
      tex = rec.tex;
    }
    let mrec: { prev: Float32Array; relative: boolean } | null = null;
    const infl = mesh.morphTargetInfluences;
    if (infl && infl.length) {
      mrec = {
        prev: new Float32Array(infl),
        relative: !!(mesh.geometry as THREE.BufferGeometry).morphTargetsRelative,
      };
      this.morphs.set(mesh, mrec);
    }
    const prevHook = mesh.onBeforeRender;
    const { mat } = this;
    // eslint-disable-next-line no-param-reassign
    mesh.onBeforeRender = function hook(renderer, scene, camera, geometry, material, group) {
      // The override material is SHARED by every mesh in the pass, so per-mesh
      // data can only travel through here — three calls it before uploading
      // uniforms, and uniformsNeedUpdate forces the re-upload per draw.
      if (material === mat) {
        // Match the REAL material's culling for this draw. `side` is pure cull
        // state for a ShaderMaterial (no program variant), and setProgram reads
        // it after this hook, so a per-draw assignment lands. Per group, so a
        // mesh's [base, outline] pair gets FrontSide then BackSide exactly as
        // the colour pass does — which also gives the silhouette the outline
        // hull's velocity, the surface actually visible there.
        const own = Array.isArray(this.material)
          ? this.material[group?.materialIndex ?? 0]
          : this.material;
        mat.side = own?.side ?? THREE.FrontSide;
        if (tex) mat.uniforms.uPrevBoneTexture.value = tex;
        if (mrec) {
          mat.uniforms.uPrevMorph.value = mrec.prev;
          // Mirrors WebGLMorphtargets: relative (all glTF/VRM morphs) keeps the
          // base at 1 and adds deltas; absolute needs 1 - sum(weights).
          let base = 1;
          if (!mrec.relative) {
            let sum = 0;
            for (let i = 0; i < mrec.prev.length; i += 1) sum += mrec.prev[i];
            base = 1 - sum;
          }
          mat.uniforms.uPrevMorphBase.value = base;
        }
        mat.uniformsNeedUpdate = true;
      }
      prevHook.call(this, renderer, scene, camera, geometry, material, group);
    };
    // eslint-disable-next-line no-param-reassign
    mesh.userData.__taaVel = true;
  }

  render(
    renderer: THREE.WebGLRenderer,
    scene: THREE.Scene,
    camera: THREE.PerspectiveCamera,
  ): void {
    // One walk: adopt new skinned meshes (a model swap re-creates them), and
    // hide what the override material cannot legally render — Sprites (the
    // debug figures' joint labels) and fat lines use their own vertex layouts
    // and would rasterise garbage velocity over the frame.
    const hidden: THREE.Object3D[] = [];
    scene.traverse((o) => {
      const m = o as THREE.SkinnedMesh;
      if ((m.isSkinnedMesh || (m.isMesh && m.morphTargetInfluences?.length))
        && !m.userData.__taaVel) this.register(m);
      const f = o as unknown as { isSprite?: boolean; isLine2?: boolean; isLineSegments2?: boolean };
      if ((f.isSprite || f.isLine2 || f.isLineSegments2) && o.visible) {
        // eslint-disable-next-line no-param-reassign
        o.visible = false;
        hidden.push(o);
      }
    });
    if (!this.fresh) this.snapshot();

    this.mat.uniforms.uCurrVP.value.copy(taa.viewProj);
    this.mat.uniforms.uPrevVP.value.copy(taa.prevViewProj);

    const bg = scene.background;
    const ov = scene.overrideMaterial;
    const rt = renderer.getRenderTarget();
    const autoClear = renderer.autoClear;
    renderer.getClearColor(this.clearColor);
    const clearAlpha = renderer.getClearAlpha();
    scene.background = null; // the background painter would write COLOUR here
    scene.overrideMaterial = this.mat;
    renderer.setClearColor(0x000000, 0); // zero velocity where nothing renders
    renderer.autoClear = true;
    renderer.setRenderTarget(this.rt);
    renderer.render(scene, camera);
    renderer.setRenderTarget(rt);
    renderer.setClearColor(this.clearColor, clearAlpha);
    renderer.autoClear = autoClear;
    scene.overrideMaterial = ov;
    scene.background = bg;
    hidden.forEach((o) => {
      // eslint-disable-next-line no-param-reassign
      o.visible = true;
    });
    this.fresh = false;
  }

  /** Copy this frame's final bones AND morph weights; they become next frame's
   *  "previous". Must run after the scene render — vrmExpression / vrmLipSync /
   *  faceMotion all write morphs late, and Skeleton.update() runs several times
   *  inside the composer with identical values. */
  snapshot(): void {
    this.skins.forEach((rec, skeleton) => {
      rec.arr.set(skeleton.boneMatrices);
      // eslint-disable-next-line no-param-reassign
      rec.tex.needsUpdate = true;
    });
    this.morphs.forEach((rec, mesh) => {
      const infl = mesh.morphTargetInfluences;
      if (infl && infl.length === rec.prev.length) rec.prev.set(infl);
    });
    this.fresh = true;
  }

  setSize(width: number, height: number): void {
    const w = Math.max(1, width);
    const h = Math.max(1, height);
    this.rt.setSize(w, h);
    // setSize does not resize an attached DepthTexture (same trap as normalRT
    // in the viewer) — update it by hand or the dilation samples a stale size.
    if (this.rt.depthTexture) {
      const img = (this.rt.depthTexture as unknown as {
        image: { width: number; height: number }
      }).image;
      img.width = w;
      img.height = h;
    }
  }

  dispose(): void {
    this.rt.dispose();
    this.mat.dispose();
    this.skins.forEach((rec) => rec.tex.dispose());
    this.skins.clear();
    this.morphs.clear();
  }
}
