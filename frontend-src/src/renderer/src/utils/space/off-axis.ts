import * as THREE from "three";
import { GLASS_Z, PARALLAX } from "./layers";
import { SPACE } from "./space-config";
import { driftOffset } from "./framing";
import { viewerTransform } from "./viewer-transform";

/**
 * Phase 3 — head-coupled off-axis projection ("fish tank VR").
 *
 * The panel stops being a camera viewfinder and becomes a fixed WINDOW into the
 * scene. The window's four corners are nailed to world space; the eye moves; the
 * frustum is rebuilt each frame as the (asymmetric) pyramid from the eye through
 * that fixed rectangle. Step left and you look into the box from the left rather
 * than panning the box left — which is the whole difference between "a screen
 * showing a character" and "a hole in the wall".
 *
 * This is Kooima's generalised perspective projection. Deliberately NOT done by
 * nudging the camera and keeping a symmetric frustum: that produces a rotation,
 * not a parallax shift, and reads as a wobbling camera rather than a still room.
 *
 * The window rectangle is in metres and MUST match the physical panel, because
 * the illusion is a claim about real-world geometry. Those numbers can't be
 * discovered from software — `__protoCalibrate()` walks through measuring them.
 */

export interface Panel {
  /** Width of the WebGL viewport, in metres on the glass. */
  width: number;
  /** Height of the WebGL viewport, in metres on the glass. */
  height: number;
  /** World X of the centre of that viewport (0 = centred on the panel). */
  centerX: number;
  /** World Y of the centre of that viewport (floor is y=0). */
  centerY: number;
  /** World Z of the glass. */
  z: number;
}

/**
 * The window is the CANVAS, not the panel.
 *
 * Off-axis projection maps a world-space rectangle onto the whole viewport, so
 * the rectangle has to be the physical patch of glass the viewport actually
 * covers. Feeding it the full panel while the canvas is inset (the sidebar takes
 * a chunk, and it isn't centred) does two things at once: the image is stretched
 * by the aspect difference, and the whole scene is sheared because the frustum is
 * built around the wrong centre. Measured on the real unit, that reads as the
 * illusion simply not working.
 *
 * So: convert the canvas rect into metres using the measured panel size, and
 * offset it by how far the canvas centre sits from the window centre. This
 * assumes the browser window fills the panel — i.e. fullscreen/kiosk, which is
 * the intended deployment and what `__protoCalibrate()` checks.
 */
export function currentPanel(container: HTMLElement): Panel {
  const vw = Math.max(window.innerWidth, 1);
  const vh = Math.max(window.innerHeight, 1);
  const r = container.getBoundingClientRect();

  // ONE scale for both axes. Display pixels are physically square, so deriving
  // metres-per-pixel separately per axis invents non-square pixels: the moment
  // the browser window's aspect differs from panelWidthM/panelHeightM — which it
  // does whenever the window isn't exactly the panel's shape — the whole scene
  // gets stretched along one axis and the illusion dies. Taking the scale from
  // the width alone keeps the projected rectangle exactly the canvas's shape,
  // whatever the window is doing. `__offAxis()` reports the height the
  // measurement implies so a bad measurement is visible rather than silent.
  const mpp = SPACE.panelWidthM / vw;

  return {
    width: Math.max(r.width * mpp, 1e-3),
    height: Math.max(r.height * mpp, 1e-3),
    centerX: (r.left + r.width / 2 - vw / 2) * mpp,
    // Screen Y grows downward, world Y grows up.
    centerY: SPACE.panelCenterY - (r.top + r.height / 2 - vh / 2) * mpp,
    z: GLASS_Z,
  };
}

/** Panel height the width measurement implies — compare against the measured one. */
export function impliedPanelHeightM(): number {
  const vw = Math.max(window.innerWidth, 1);
  const vh = Math.max(window.innerHeight, 1);
  return (SPACE.panelWidthM / vw) * vh;
}

const pa = new THREE.Vector3(); // lower-left of the window
const pb = new THREE.Vector3(); // lower-right
const pc = new THREE.Vector3(); // upper-left
const va = new THREE.Vector3();
const vb = new THREE.Vector3();
const vc = new THREE.Vector3();
const vr = new THREE.Vector3();
const vu = new THREE.Vector3();
const vn = new THREE.Vector3();
const eye = new THREE.Vector3();

/**
 * Rebuild `camera`'s projection so the window stays fixed and the eye moves.
 * Returns false and leaves the camera alone when off-axis is disabled.
 */
let wasActive = false;

export function applyOffAxis(camera: THREE.PerspectiveCamera, panel: Panel): boolean {
  /**
   * Only take the camera over when there is a real head to couple to.
   *
   * Off-axis writes camera.position/quaternion/projectionMatrix every frame, so
   * while it is on, OrbitControls has no effect — zoom, pan and orbit all appear
   * dead. With no tracking there is nothing to gain from that: the projection
   * would just be locked to a guessed resting pose, computed from panel
   * dimensions that are placeholders until someone measures them. So when no
   * head is tracked (and no pose is pinned) it stands down completely and the
   * ordinary, fully interactive camera is left alone.
   */
  if (!SPACE.offAxis || !viewerTransform.isTracking()) {
    if (wasActive) {
      // Restore the symmetric frustum we replaced, or the last off-axis matrix
      // would persist as a frozen skew.
      camera.updateProjectionMatrix();
      wasActive = false;
    }
    return false;
  }

  const pose = viewerTransform.get();
  // `strength` lets the effect be dialled from 0 (plain perspective) to beyond 1
  // (exaggerated). Physically 1 is "correct", but a viewer standing further back
  // than the tracker thinks will read 1 as too weak, so it stays adjustable.
  const s = SPACE.offAxisStrength;

  // The eye is positioned relative to the PANEL centre (that is what the tracker
  // measures against), not the viewport centre — so an inset canvas correctly
  // shows an off-centre view rather than re-centring itself on the viewer.
  eye.set(
    pose.x * s,
    SPACE.panelCenterY + pose.y * s,
    panel.z + Math.max(pose.z, 0.15),
  );

  const halfW = panel.width / 2;
  const halfH = panel.height / 2;
  pa.set(panel.centerX - halfW, panel.centerY - halfH, panel.z);
  pb.set(panel.centerX + halfW, panel.centerY - halfH, panel.z);
  pc.set(panel.centerX - halfW, panel.centerY + halfH, panel.z);

  // Orthonormal basis of the window. Computed rather than hard-coded so a tilted
  // or rotated panel needs only different corners, not different maths.
  vr.subVectors(pb, pa).normalize();
  vu.subVectors(pc, pa).normalize();
  vn.crossVectors(vr, vu).normalize();

  va.subVectors(pa, eye);
  vb.subVectors(pb, eye);
  vc.subVectors(pc, eye);

  // Perpendicular distance from the eye to the window plane.
  const d = -va.dot(vn);
  if (!(d > 1e-4)) return false; // eye is on/behind the glass — nothing sensible to draw

  const { near, far } = camera;
  const k = near / d;
  const left = vr.dot(va) * k;
  const right = vr.dot(vb) * k;
  const bottom = vu.dot(va) * k;
  const top = vu.dot(vc) * k;

  camera.projectionMatrix.makePerspective(left, right, top, bottom, near, far, camera.coordinateSystem);
  camera.projectionMatrixInverse.copy(camera.projectionMatrix).invert();

  // The eye moves; the view direction does NOT rotate. All the parallax lives in
  // the asymmetric frustum above — rotating here would undo it.
  camera.position.copy(eye);
  camera.quaternion.identity();
  camera.updateMatrixWorld(true);
  wasActive = true;
  return true;
}

/**
 * Exaggerated motion parallax on top of the physically-correct projection.
 *
 * Off-axis already gives real parallax for free, so this is a stylistic dial,
 * not a correction: on a panel this shallow the physically-correct shift can be
 * too subtle to read at a glance, and pushing the near layers a little further
 * than truth makes the depth legible without breaking the window.
 *
 * Ground-anchored objects opt out via `userData.noParallax` — shearing the floor
 * away from her feet would destroy exactly what Phase 1 built.
 */
export function applyParallax(bands: Map<string, THREE.Group>) {
  const extra = SPACE.parallaxBoost;
  if (extra <= 0.001) {
    bands.forEach((g) => g.position.set(0, 0, 0));
    return;
  }
  /**
   * Driven by whichever viewpoint motion actually exists.
   *
   * With a head tracked, the tracker's pose is the honest input. Without one it
   * is a CONSTANT — which meant this whole function was inert on the installed
   * unit and its strength dial silently did nothing. The camera drift is the
   * only viewpoint movement there, so it stands in: same sign convention, same
   * units (metres of eye movement), just a different source.
   */
  const tracking = viewerTransform.isTracking();
  const pose = viewerTransform.get();
  const dx = tracking ? pose.x : driftOffset.x;
  const dy = tracking ? pose.y : driftOffset.y;
  // The drift is ~30x smaller than a head movement, so it needs its own scale
  // to reach the same visible separation from a dial calibrated for tracking.
  const k = tracking ? 1 : 30;
  bands.forEach((group, band) => {
    const w = (PARALLAX as Record<string, number>)[band] ?? 0;
    // Shift AGAINST the viewer: moving right should reveal what was hidden
    // behind the right edge, which means the contents slide left.
    group.position.set(-dx * k * w * extra, -dy * k * w * extra * 0.5, 0);
  });
}

/** Restore the stock symmetric projection (off-axis off / tracking lost). */
export function clearOffAxis(camera: THREE.PerspectiveCamera) {
  camera.updateProjectionMatrix();
}
