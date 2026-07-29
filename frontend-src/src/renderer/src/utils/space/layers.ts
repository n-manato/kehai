/**
 * Coordinate convention for the Spatial Presence Layer ("PROTO as a window").
 *
 * Everything the spatial work depends on — the environment bands, the glass
 * plane, the parallax weights — is defined ONCE here so Phase 2 (head tracking),
 * Phase 3 (off-axis projection) and Phase 4 (Miku's Z movement / glass collider)
 * all agree without re-deriving numbers.
 *
 * Units are metres. Miku stands at the origin facing +Z (toward the viewer),
 * just INSIDE the window rather than deep in the box. A real window only shows
 * things at the glass at 1:1 — anything further back is smaller by the ratio of
 * the distances — so "life-size" and "deep inside" cannot both hold. She sits
 * near the front and the space extends behind her.
 *
 *   z = +2.4   viewer's eye (the camera)      <- Phase 3 derives this from the real panel
 *   z = +0.45  PROTO front glass  (GLASS_Z)   <- Phase 4 collider / glass-touch ripple
 *   z = +0.2 … +0.8   Foreground
 *   z =  0            Character (Miku)
 *   z = -1  … -8      Near environment
 *   z = -8  … -40     Far environment
 *   z = -40 … -95     "Infinite" (dissolves into haze)
 *   the CSS radial-gradient backdrop sits behind all of it, effectively at infinity.
 *
 * WHY the far band stops at -95 instead of -2000: the camera is
 * `PerspectiveCamera(30, aspect, 0.01, 100)` and we deliberately do NOT widen its
 * far plane. Apparent depth is driven by ANGULAR size, not absolute distance — a
 * 30 m ring at z=-80 is pixel-identical to a 120 m ring at z=-320. Raising `far`
 * would buy nothing and would break two tuned things: the ink-line pass
 * linearises depth with `uCamNear/uCamFar` (so `lineZInt` would need retuning, and
 * Miku's outline is the most carefully tuned part of the look), and the bokeh
 * pass's circle-of-confusion distribution would shift. So: compress the world,
 * keep the camera untouched.
 */

/**
 * Layer the decorative environment renders on. Kept off layer 0 so the ink-line
 * pre-pass can exclude it with a single `camera.layers.disable(SPACE_LAYER)` —
 * otherwise the toon outline detector draws ink lines around every neon ring.
 *
 */
export const SPACE_LAYER = 3;

/** World Z of the PROTO front surface — the "window" the viewer looks through. */
export const GLASS_Z = 0.45;

/** Floor height. Miku's feet are planted here. */
export const FLOOR_Y = 0;

/** Depth bands. `[nearZ, farZ]`, both negative-going away from the viewer. */
export const BANDS = {
  foreground: [0.8, 0.2],
  character: [0.5, -0.5],
  near: [-1, -8],
  far: [-8, -40],
  infinite: [-40, -95],
} as const;

export type BandName = keyof typeof BANDS;

/**
 * How much each band should shift when the viewer's head moves sideways
 * (Phase 3 motion parallax). Foreground swings fully with the viewer; the
 * "infinite" band barely moves, which is what sells the depth on a flat panel.
 *
 * Declared now, applied in Phase 3 — the environment builder tags each group with
 * its band so the parallax pass has something to grab.
 */
export const PARALLAX = {
  foreground: 1.0,
  character: 0.75,
  near: 0.35,
  far: 0.1,
  infinite: 0.02,
} as const;

/** Distance from the camera at which haze is fully saturated (see haze.ts). */
export const HAZE_FAR = 95;
