import type * as THREE from "three";
import { SPACE } from "./space-config";

/**
 * PROTO mode — the framing used when the app is running on the PROTO Luma unit
 * rather than on a desktop browser.
 *
 * The desktop framing is a bust-up shot: the camera sits at (0, 1.05, 2.4) with a
 * 30° FOV, so the visible slab is only ~1.29 m tall centred on y=0.98 and her feet
 * are off-screen. That is fine for a talking-head window, but the whole point of
 * the spatial layer is that she is STANDING somewhere — contact shadow, floor grid
 * and reflection all need the ground to be visible. So PROTO mode pulls the camera
 * back and drops it to frame the whole body plus a stretch of floor in front.
 *
 * Enabled with `?proto=1`, and sticky afterwards: the unit can be pointed at the
 * URL once and keep the mode across reloads/power cycles without anyone re-typing
 * a query string. `?proto=0` clears it.
 */

const LS_PROTO = "protoMode";

let cached: boolean | null = null;

/** True when the app should present itself as a window in a physical PROTO unit. */
export function isProtoMode(): boolean {
  if (cached !== null) return cached;
  let on = false;
  try {
    const q = new URLSearchParams(window.location.search).get("proto");
    if (q === "1" || q === "true") {
      on = true;
      localStorage.setItem(LS_PROTO, "1");
    } else if (q === "0" || q === "false") {
      localStorage.removeItem(LS_PROTO);
    } else {
      on = localStorage.getItem(LS_PROTO) === "1";
    }
  } catch {
    /* no window/localStorage (SSR, sandboxed iframe) — stay off */
  }
  cached = on;
  return on;
}

/** Force the mode on/off at runtime (debug panel / console). */
export function setProtoMode(on: boolean) {
  cached = on;
  try {
    if (on) localStorage.setItem(LS_PROTO, "1");
    else localStorage.removeItem(LS_PROTO);
  } catch {
    /* ignore */
  }
}

/**
 * Apply the PROTO framing to an existing camera + OrbitControls pair.
 * A no-op outside PROTO mode, so the desktop path is byte-for-byte unchanged.
 *
 * Safe to call repeatedly — `window.__space({ protoCamZ: 4 })` re-runs it so the
 * framing can be dialled in against the real unit without a rebuild.
 */
export function applyProtoFraming(
  camera: THREE.PerspectiveCamera,
  controls: { target: THREE.Vector3; enabled: boolean; update: () => void },
): boolean {
  if (!isProtoMode()) return false;
  camera.position.set(0, SPACE.protoCamY, SPACE.protoCamZ);
  controls.target.set(0, SPACE.protoTargetY, 0);
  camera.lookAt(controls.target);
  // Nobody drives a mouse in front of a PROTO unit, and a stray drag would leave
  // the framing wrong with no way to reset it. Phase 3 takes the camera over
  // completely (head-coupled off-axis projection), so this is where that starts.
  controls.enabled = !SPACE.protoLock;
  controls.update();
  return true;
}
