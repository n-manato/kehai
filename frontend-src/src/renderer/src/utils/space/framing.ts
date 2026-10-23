import * as THREE from "three";
import { SPACE } from "./space-config";
import { isProtoMode } from "./proto-mode";
import { viewerTransform, viewerWorld } from "./viewer-transform";

/**
 * Camera placement for the PROTO unit: life-size framing, plus a slow drift.
 *
 * FRAMING IS SOLVED, NOT DIALLED. The first version derived the distance from
 * "one world metre = one panel metre at her depth", which is a clean definition
 * of life-size and cropped her feet: it fixes the visible height at her depth to
 * the panel height, so the frame is eyeY ± panelH/2 and a standing 1.5 m eye
 * cuts the bottom off at mid-shin. Widening the FOV does not help, because
 * staying life-size pulls the camera closer by the same factor.
 *
 * Instead the caller states the world Y range that must stay in frame and how
 * much of the frame it should fill; distance and pitch fall out of that. Her
 * feet are then in shot by construction rather than by a lucky setting.
 *
 * DRIFT. With no head tracking there is no motion parallax at all, and a static
 * viewpoint reduces every depth cue to the pictorial ones — which is exactly why
 * the first attempt "全然立体に見えない". A few centimetres of continuous camera
 * movement restores real parallax between the near and far structure. It is
 * deliberately slow and small: enough that the layers visibly separate, not
 * enough to read as a moving shot.
 */

const _target = new THREE.Vector3();

/**
 * How far the drift has currently pushed the camera off its base, in metres.
 *
 * Published because the exaggerated layer parallax (`applyParallax`) was written
 * against the head TRACKER's pose, and with no head tracked that pose is a
 * constant — so the whole effect was inert and its strength dial did nothing.
 * The drift is the only viewpoint motion this installation actually has, so it
 * is what the layer separation has to be driven by.
 */
export const driftOffset = new THREE.Vector3();

/**
 * The camera's orientation with the drift removed.
 *
 * Needed because the drift is now an ORBIT: it swings the camera around her by
 * up to fifteen degrees, so anything that wants to reason about the "resting"
 * view has to undo a rotation as well as a translation. While the drift was a
 * few centimetres of translation the rotation was under half a degree and could
 * be ignored; at fifteen degrees it cannot.
 */
export const driftBaseQuat = new THREE.Quaternion();

/** Resting camera position and the point it orbits, published for the same reason. */
export const driftBase = new THREE.Vector3();
export const driftTarget = new THREE.Vector3();

/** Camera position at `yawDeg` around `driftTarget`, on the resting orbit. */
export function orbitPose(out: THREE.Vector3, yawDeg: number): THREE.Vector3 {
  const yaw = THREE.MathUtils.degToRad(yawDeg);
  const dx = driftBase.x - driftTarget.x;
  const dz = driftBase.z - driftTarget.z;
  const c = Math.cos(yaw);
  const s = Math.sin(yaw);
  return out.set(
    driftTarget.x + dx * c + dz * s,
    driftBase.y,
    driftTarget.z - dx * s + dz * c,
  );
}

const _orbit = new THREE.Vector3();
const _viewer = new THREE.Vector3();

/**
 * The camera following the attended person (see `headOrbitGain`).
 *
 * `weight` is the hand-over between the idle sway (0) and following (1), eased
 * over `headOrbitBlendS` so a face appearing or leaving never cuts the shot.
 * Published because her head turn uses the same hand-over: the turn toward a
 * person and the camera swinging to them have to arrive together.
 */
export const headFollow = { weight: 0, yawDeg: 0, targetDeg: 0 };

export interface Framing {
  /** Re-apply the base framing (after a settings change). */
  apply(): void;
  /** Per-frame drift. Call after controls.update(), before rendering. */
  update(dt: number): void;
  /** Base camera position, without drift — what off-axis should build on. */
  readonly base: THREE.Vector3;
  /** Adopt the camera's current position as the new drift centre. */
  rebase(): void;
  /** Suspend the drift while the user is dragging the camera. */
  setInteracting(v: boolean): void;
}

export function createFraming(
  camera: THREE.PerspectiveCamera,
  controls: { target: THREE.Vector3; enabled: boolean; update: () => void },
): Framing {
  const base = new THREE.Vector3();
  let t = 0;
  /**
   * True while the user is dragging.
   *
   * The drift writes camera.position every frame from `base`, which ran straight
   * over whatever OrbitControls had just computed — so dragging fought a spring
   * pulling the camera back and the controls felt broken. It has to stand down
   * for the duration of the interaction, then adopt wherever the user left it.
   */
  let interacting = false;
  // Remembered so a user orbit is not fought: if they move the camera, the
  // drift rides on wherever they left it rather than yanking it back.
  let ownsCamera = false;

  // Drift is worth having in every mode: without head tracking it is the only
  // source of motion parallax, and parallax is what stops a correctly drawn
  // perspective from still reading as a flat picture.
  const rebase = () => {
    base.copy(camera.position);
    ownsCamera = true;
  };

  const apply = () => {
    // Applied in EVERY mode. Gating this on PROTO mode meant an ordinary load
    // kept the viewer's original bust-up camera and none of the framing work
    // showed up at all — "変わってない". The backdrop was already unconditional;
    // the framing has to match it or the two describe different scenes.
    if (!SPACE.fitBody && !isProtoMode()) {
      rebase();
      return;
    }
    if (Math.abs(camera.fov - SPACE.fovDeg) > 0.01) {
      camera.fov = SPACE.fovDeg;
      camera.updateProjectionMatrix();
    }
    const halfFov = (camera.fov * Math.PI) / 360;

    /**
     * Solve the framing instead of dialling it.
     *
     * "Life-size with the camera at standing height" cannot show her feet, and
     * that is geometry, not tuning: life-size means the visible height at her
     * depth is exactly the panel height, so the frame covers eyeY ± panelH/2.
     * With a 1.9 m panel that is eyeY ± 0.95, and a 1.5 m eye puts the bottom of
     * frame at 0.55 m — mid-shin. No FOV fixes it, because widening the FOV pulls
     * the camera closer by the same factor to stay life-size.
     *
     * So: pick how much of the frame her body should fill, derive the distance
     * from that, then pitch the camera so her feet-to-head range sits inside the
     * frame with the margin left over. Feet are then visible by construction.
     */
    const bodyTop = SPACE.fitTopY;
    const bodyBottom = SPACE.fitBottomY;
    const bodySpan = Math.max(bodyTop - bodyBottom, 0.1);

    // Visible height at her depth, and the distance that produces it.
    const visible = bodySpan / THREE.MathUtils.clamp(SPACE.fillFrac, 0.2, 1);
    const dist = SPACE.lifeSize ? visible / (2 * Math.tan(halfFov)) : SPACE.protoCamZ;

    /**
     * Eye height is the control; the horizon falls out of it.
     *
     * These two cannot both be set. With her whole body in frame the pitch is
     * determined, and the horizon is simply where the camera axis is level — so
     * driving the horizon (as the previous version did) drives the eye down with
     * it, and a low horizon means literally looking up at her. Eye height is the
     * one of the pair that describes what the shot feels like, so that is the
     * knob; `__space().horizonAtM` reports where the horizon ended up.
     */
    const eyeY = SPACE.viewerEyeY;
    base.set(0, eyeY, dist);

    // Where the frame centre must land. NOT simply her mid-height: centring
    // splits the leftover margin evenly, and what is wanted is her feet almost
    // touching the bottom edge with the slack pushed above her head. Solve for
    // the centre that puts bodyBottom exactly `bottomMargin` of a frame height
    // up from the bottom edge.
    const wantCentre = SPACE.fitBody
      ? bodyBottom + visible * (0.5 - THREE.MathUtils.clamp(SPACE.bottomMargin, 0, 0.4))
      // Manual mode: horizonFrac says where the horizon sits in the frame, and
      // the horizon is wherever the camera axis is level.
      : eyeY - Math.tan((SPACE.horizonFrac - 0.5) * camera.fov * Math.PI / 180) * dist;

    _target.set(0, wantCentre, 0);
    camera.position.copy(base);
    controls.target.copy(_target);
    camera.lookAt(_target);
    controls.enabled = !SPACE.protoLock;
    controls.update();
    ownsCamera = true;
  };

  return {
    base,
    apply,
    rebase,
    setInteracting(v: boolean) {
      interacting = v;
      if (!v) rebase();
    },
    update(dt: number) {
      if (interacting || !ownsCamera) return;

      // Follow hand-over and the bearing to follow. Updated even when the drift
      // is switched off: following a person is not the ambient sway.
      const tracking = viewerTransform.isTracking();
      const blend = Math.max(SPACE.headOrbitBlendS, 0.05);
      headFollow.weight = THREE.MathUtils.clamp(headFollow.weight + (tracking ? dt : -dt) / blend, 0, 1);
      if (tracking) {
        viewerWorld(_viewer);
        // Bearing of the viewer from the point the camera orbits, in the same
        // sense as the orbit's yaw: positive = to the right of the screen.
        const bearing = THREE.MathUtils.radToDeg(
          Math.atan2(_viewer.x - controls.target.x, _viewer.z - controls.target.z),
        );
        const max = Math.max(SPACE.headOrbitMaxDeg, 0);
        headFollow.targetDeg = THREE.MathUtils.clamp(bearing * SPACE.headOrbitGain, -max, max);
      }
      // Eased rather than set: changing who she attends to is a jump of the
      // bearing, and the shot should swing to the new person, not cut.
      headFollow.yawDeg += (headFollow.targetDeg - headFollow.yawDeg)
        * (1 - Math.exp(-Math.max(SPACE.headOrbitSpeed, 0.01) * dt));
      const fw = headFollow.weight;

      const driftOn = SPACE.cameraDriftEnabled && SPACE.cameraDrift > 0.0001;
      if (!driftOn && fw <= 0) {
        // Snap back rather than freezing wherever the drift happened to be, or
        // turning it off leaves the camera at a random offset for the session.
        if (driftOffset.lengthSq() > 0) {
          driftOffset.set(0, 0, 0);
          camera.position.copy(base);
          camera.lookAt(controls.target);
        }
        return;
      }
      t += dt;
      const w = (Math.PI * 2) / Math.max(SPACE.cameraDriftPeriod, 1);
      // The sway yields to following: a camera that keeps wandering while it
      // is meant to be looking from your side reads as not looking at all.
      const dw = driftOn ? 1 - fw : 0;

      /**
       * TWO low frequencies per axis, not one.
       *
       * A single sine is a pendulum: it has one period, it reaches the same two
       * extremes every cycle, and the eye locks onto it within a couple of
       * passes — at which point it stops reading as "the viewpoint is alive" and
       * starts reading as "the camera is oscillating", which is the one thing
       * this must not look like. Summing a second, incommensurate frequency
       * means the path never repeats and has no recognisable turning points.
       * Weights sum to 1 so the amplitudes below stay the true metre bounds.
       */
      const mix = (f1: number, f2: number, phase: number) =>
        Math.sin(t * w * f1) * 0.62 + Math.sin(t * w * f2 + phase) * 0.38;

      // Remember the resting orientation before the orbit moves it, so the fit
      // solvers can put the camera back exactly.
      camera.position.copy(base);
      camera.lookAt(controls.target);
      driftBaseQuat.copy(camera.quaternion);
      driftBase.copy(base);
      driftTarget.copy(controls.target);

      const yBob = mix(0.63, 0.271, 0.4) * SPACE.cameraDriftY * dw;
      // Height relative to the virtual eye, so a viewer at that height moves nothing.
      const yFollow = THREE.MathUtils.clamp((_viewer.y - SPACE.viewerEyeY) * SPACE.headOrbitY, -0.3, 0.3) * fw;

      /**
       * ORBIT around her, not a translation past her.
       *
       * A lateral slide and a swing around the subject look similar over a few
       * centimetres, but they are different motions and only one of them scales:
       * at fifteen degrees a translation would carry the camera nearly a metre
       * sideways and out of the framing entirely, while an orbit keeps the
       * distance — and therefore her size and her position in frame — fixed by
       * construction, and spends the whole movement on showing her from a
       * different angle. That is also what makes the background sweep, which is
       * where the depth reads from. Following a person is the same orbit with
       * its angle taken from where they stand.
       */
      let yaw = THREE.MathUtils.degToRad(headFollow.yawDeg) * fw;
      // Legacy pure-translation drift, kept for cameraOrbitDeg = 0.
      let slideX = 0;
      let slideZ = 0;
      if (SPACE.cameraOrbitDeg > 0.001) {
        yaw += mix(1.00, 0.412, 1.7) * THREE.MathUtils.degToRad(SPACE.cameraOrbitDeg) * dw;
      } else {
        slideX = mix(1.00, 0.412, 1.7) * SPACE.cameraDrift * dw;
        slideZ = mix(0.37, 0.183, 2.9) * SPACE.cameraDriftZ * dw;
      }
      const dx = base.x - controls.target.x;
      const dz = base.z - controls.target.z;
      const c = Math.cos(yaw);
      const s = Math.sin(yaw);
      _orbit.set(
        controls.target.x + dx * c + dz * s + slideX,
        base.y + yBob + yFollow,
        controls.target.z - dx * s + dz * c + slideZ,
      );
      camera.position.copy(_orbit);

      driftOffset.subVectors(camera.position, base);
      // Re-aim at the solved framing target every frame. This is what makes the
      // drift read as depth rather than as a wobble: she stays put in the frame
      // and the background slides behind her, which is parallax. Moving the
      // camera without re-aiming would slide HER across the frame instead.
      camera.lookAt(controls.target);
    },
  };
}
