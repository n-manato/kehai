import type * as THREE from "three";
import { GLASS_Z } from "./layers";
import { SPACE } from "./space-config";

/**
 * Where the viewer's head is, in the SAME world coordinates as the scene
 * (metres, panel-centre-ish origin, +Z toward the viewer — see layers.ts).
 *
 * This is the single junction the whole spatial illusion hangs off. Phase 2
 * (head tracking) writes it; Phase 3 (off-axis projection), Phase 4 (Miku's
 * gaze / distance reactions) and Phase 5 (audio listener) all read it. Keeping
 * one shared transform rather than passing camera data around is what stops the
 * eyes, the perspective and the sound from disagreeing about where you are.
 *
 * It is deliberately a plain singleton with no React involvement: it updates at
 * camera frame rate and is read inside the render loop, so routing it through
 * state would add a re-render per frame for no benefit.
 */

/**
 * One Euro filter — the standard low-latency smoother for head tracking.
 *
 * A fixed lerp forces a bad trade: smooth enough to kill jitter means visibly
 * laggy when you move, and the illusion breaks on lag far more than on jitter.
 * One Euro adapts its cutoff to speed: heavy smoothing while you hold still,
 * almost none while you move. (Casiez, Roussel & Vogel, CHI 2012.)
 */
class OneEuro {
  private xPrev: number | null = null;

  private dxPrev = 0;

  private tPrev = 0;

  constructor(
    private minCutoff = 1.0,
    private beta = 0.35,
    private dCutoff = 1.0,
  ) {}

  private static alpha(cutoff: number, dt: number) {
    const tau = 1 / (2 * Math.PI * cutoff);
    return 1 / (1 + tau / dt);
  }

  reset() {
    this.xPrev = null;
    this.dxPrev = 0;
  }

  filter(x: number, t: number): number {
    if (this.xPrev === null || t <= this.tPrev) {
      this.xPrev = x;
      this.tPrev = t;
      return x;
    }
    const dt = Math.max(t - this.tPrev, 1e-3);
    this.tPrev = t;

    const dx = (x - this.xPrev) / dt;
    const aD = OneEuro.alpha(this.dCutoff, dt);
    this.dxPrev = aD * dx + (1 - aD) * this.dxPrev;

    const cutoff = this.minCutoff + this.beta * Math.abs(this.dxPrev);
    const a = OneEuro.alpha(cutoff, dt);
    const filtered = a * x + (1 - a) * this.xPrev;
    this.xPrev = filtered;
    return filtered;
  }
}

export interface ViewerPose {
  /** Metres, right-positive, from the panel centre. */
  x: number;
  /** Metres, up-positive, from the panel centre. */
  y: number;
  /** Metres in front of the panel (always positive when tracking). */
  z: number;
  /** Degrees. Head orientation — optional extra, not needed for off-axis. */
  yaw: number;
  pitch: number;
  roll: number;
  /** 0 = no face found and the pose is the resting default. */
  confidence: number;
}

/** Where the viewer is assumed to be when nothing is tracking. */
export const DEFAULT_POSE: ViewerPose = {
  x: 0,
  y: 0,
  z: 1.2,
  yaw: 0,
  pitch: 0,
  roll: 0,
  confidence: 0,
};

class ViewerTransform {
  private pose: ViewerPose = { ...DEFAULT_POSE };

  private fx = new OneEuro();

  private fy = new OneEuro();

  private fz = new OneEuro(0.8, 0.25);

  private lastSeen = 0;

  /** Seconds of no detection before we ease back to the resting pose. */
  private holdSeconds = 0.8;

  private pinned = false;

  /**
   * Hold a fixed pose, ignoring the tracker.
   *
   * Not just a debug affordance: it is the only way to see what the off-axis
   * projection does at a given eye position without a person physically standing
   * there, which is how the effect gets tuned and how it can be regression-shot.
   */
  pin(x: number, y: number, z: number) {
    this.pinned = true;
    this.pose.x = x;
    this.pose.y = y;
    this.pose.z = z;
    this.pose.confidence = 1;
  }

  unpin() {
    this.pinned = false;
    this.fx.reset();
    this.fy.reset();
    this.fz.reset();
    this.pose.confidence = 0;
  }

  isPinned() {
    return this.pinned;
  }

  /** Raw measurement from the tracker, in panel-centre metres. */
  setMeasurement(x: number, y: number, z: number, yaw: number, pitch: number, roll: number, tSeconds: number) {
    if (this.pinned) return;
    this.pose.x = this.fx.filter(x, tSeconds);
    this.pose.y = this.fy.filter(y, tSeconds);
    this.pose.z = this.fz.filter(z, tSeconds);
    this.pose.yaw = yaw;
    this.pose.pitch = pitch;
    this.pose.roll = roll;
    this.pose.confidence = 1;
    this.lastSeen = tSeconds;
  }

  /** No face this frame. Kept separate so a blink or a turn doesn't snap. */
  markLost(tSeconds: number) {
    if (this.pinned) return;
    if (tSeconds - this.lastSeen > this.holdSeconds) this.pose.confidence = 0;
  }

  /**
   * Ease back toward the resting pose when tracking is lost, so the perspective
   * settles to the neutral view instead of freezing at whatever it saw last.
   */
  update(dt: number) {
    if (this.pinned || this.pose.confidence > 0) return;
    const k = 1 - Math.pow(0.05, dt / 1.2);
    this.pose.x += (DEFAULT_POSE.x - this.pose.x) * k;
    this.pose.y += (DEFAULT_POSE.y - this.pose.y) * k;
    this.pose.z += (DEFAULT_POSE.z - this.pose.z) * k;
    this.pose.yaw += (0 - this.pose.yaw) * k;
    this.pose.pitch += (0 - this.pose.pitch) * k;
    this.pose.roll += (0 - this.pose.roll) * k;
  }

  get(): Readonly<ViewerPose> {
    return this.pose;
  }

  isTracking(): boolean {
    return this.pose.confidence > 0;
  }

  reset() {
    this.pinned = false;
    this.pose = { ...DEFAULT_POSE };
    this.fx.reset();
    this.fy.reset();
    this.fz.reset();
    this.lastSeen = 0;
  }
}

export const viewerTransform = new ViewerTransform();

/**
 * The tracked viewer's head as a point in the scene: panel-centre metres moved
 * onto the glass (GLASS_Z) at the panel's height. The same placement off-axis
 * uses for its eye, without the strength dial — this is where the person IS, for
 * her gaze and for the follow orbit's bearing.
 */
export function viewerWorld(out: THREE.Vector3): THREE.Vector3 {
  const p = viewerTransform.get();
  return out.set(p.x, SPACE.panelCenterY + p.y, GLASS_Z + Math.max(p.z, 0.15));
}
