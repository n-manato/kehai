/**
 * WHAT KIND OF MACHINE IS THIS, AND WHAT SHOULD IT NOT ATTEMPT.
 *
 * The renderer had exactly one hardware check before this file — `detectQuality()`
 * in space-config.ts, which reads `hardwareConcurrency` and `devicePixelRatio` and
 * whose only consumer is the particle count. An iPhone 15 resolved to "medium" and
 * got 1000 fewer motes, while still being asked for a 3.0 Mpx drawing buffer, six
 * scene passes, ~38 fullscreen passes and a 296-body Bullet simulation.
 *
 * The split here is deliberately COARSE — phone-ish or not — because the thing it
 * gates is not a slider. Profiling the desktop showed this frame is CPU-bound, not
 * fill-bound: at 1.23 Mpx and at 2.77 Mpx it cost the same 26 ms. Lowering the
 * resolution on a phone therefore buys far less than it looks like it should, and
 * what actually has to go is per-frame CPU work.
 *
 * Override for testing with `?mobile=1` / `?mobile=0`, which also persists — the
 * desktop box has no phone attached to check against.
 */

const LS_KEY = "kehaiDevice1";

function readOverride(): boolean | null {
  try {
    const q = new URLSearchParams(window.location.search).get("mobile");
    if (q === "1" || q === "0") {
      window.localStorage.setItem(LS_KEY, q);
      return q === "1";
    }
    const saved = window.localStorage.getItem(LS_KEY);
    if (saved === "1" || saved === "0") return saved === "1";
  } catch { /* private mode */ }
  return null;
}

/**
 * Touch-first device?
 *
 * `pointer: coarse` is the honest question — "is the primary input a finger" —
 * and it is what separates a phone from a laptop with a touchscreen. iPadOS is
 * the exception that needs a UA test: it reports itself as a Macintosh, so the
 * media query alone lets a tablet through as a desktop.
 */
export function isMobileDevice(): boolean {
  const forced = readOverride();
  if (forced !== null) return forced;
  if (typeof window === "undefined" || typeof navigator === "undefined") return false;
  const ua = navigator.userAgent || "";
  const iOS = /iPhone|iPad|iPod/.test(ua)
    || (/Macintosh/.test(ua) && (navigator.maxTouchPoints || 0) > 1);
  if (iOS) return true;
  const coarse = window.matchMedia?.("(pointer: coarse)")?.matches ?? false;
  return coarse && (navigator.maxTouchPoints || 0) > 0;
}

export interface DeviceProfile {
  mobile: boolean;
  /**
   * Run the Bullet hair simulation?
   *
   * OFF on mobile. Measured at 6.3 ms per frame on a desktop CPU for 296 rigid
   * bodies and 381 constraints — more than a third of the entire frame budget at
   * 60 fps, before an A16 is asked to do it. The hair does not freeze without it:
   * the VRM spring bones that Bullet displaces are left in place, so twintails and
   * skirt still swing. What is lost is the MMD-accurate part — collision against
   * the body capsules and the inertia of the four-chain twintail rig.
   */
  hairPhysics: boolean;
  /** Total drawing-buffer pixels. iPhone 15 is 393x852 css at dpr 3 = 3.01 Mpx. */
  maxDrawPixels: number;
  pixelRatioCap: number;
  /** iOS reports MAX_SAMPLES 4, so 8 was never achievable there anyway. */
  msaaSamples: number;
}

const DESKTOP: DeviceProfile = {
  mobile: false,
  hairPhysics: true,
  maxDrawPixels: 8_300_000,
  pixelRatioCap: 3.0,
  msaaSamples: 8,
};

const MOBILE: DeviceProfile = {
  mobile: true,
  hairPhysics: false,
  maxDrawPixels: 1_200_000,
  pixelRatioCap: 1.5,
  msaaSamples: 4,
};

let cached: DeviceProfile | null = null;

export function deviceProfile(): DeviceProfile {
  if (!cached) cached = isMobileDevice() ? { ...MOBILE } : { ...DESKTOP };
  return cached;
}
