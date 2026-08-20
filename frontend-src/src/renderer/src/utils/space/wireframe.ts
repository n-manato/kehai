import { LineBuilder, type LineSet } from "./lines";
import { SPACE } from "./space-config";
import type { SpaceModule, SpaceMood } from "./types";

/**
 * The static space: a life-size wireframe volume she stands in.
 *
 * Built at real scale on purpose. The grid is 1 m and she is 1.6 m, so the eye
 * gets a ruler it already knows how to read — that is worth more for "she is a
 * real size, standing over there" than any amount of invented far-field
 * structure. The previous attempt built a 95 m corridor of glowing rings; at a
 * fixed viewpoint on a flat panel that is just a picture of a corridor, because
 * nothing about it changes with where you are.
 *
 * Depth here comes from three things that survive a static viewpoint:
 * convergence (parallel lines meeting), density gradient (rows crowding toward
 * the horizon), and the fade to nothing. The cues that actually sell it —
 * occlusion and parallax — are in traffic.ts and the camera drift.
 */

/**
 * How far the LONGITUDINAL lines run toward the viewer. They point away from the
 * camera, so they never cross her — they only stop the floor from ending in a
 * hard edge across the frame.
 */
const FRONT_Z = 1.5;

/**
 * Where the cross rows start. Behind her feet: a row passing in front of her was
 * tried and cut — it reads as a line drawn over her rather than as floor, and
 * the occlusion cue it was meant to provide is already carried by the traffic.
 */
const ROWS_FRONT_Z = -0.15;

export function createWireframe(): SpaceModule {
  const half = SPACE.gridHalfWidth;
  const depth = SPACE.gridDepth;

  // --- Floor ---------------------------------------------------------------
  const floor = new LineBuilder();

  // Lines running away from the viewer: these are what converge.
  const cols = Math.round((half * 2) / SPACE.gridSpacing);
  for (let i = 0; i <= cols; i += 1) {
    const x = -half + i * SPACE.gridSpacing;
    // The line through her feet is dropped and the two beside her are lifted:
    // a bright line running out of the soles of her feet reads as a seam, and
    // the pair flanking her instead reads as ground she is standing between.
    if (Math.abs(x) < 0.01) continue;
    const near = Math.abs(x) < SPACE.gridSpacing * 1.5;
    floor.add(x, 0, FRONT_Z, x, 0, -depth, near ? 1.0 : 0.55, LineBuilder.colourFor(i));
  }

  // Rows running across. Spaced on a power curve so they crowd toward the
  // vanishing point the way evenly-spaced real rows do — the density gradient
  // is a stronger depth cue on a still frame than the convergence is.
  for (let i = 0; i <= SPACE.gridRows; i += 1) {
    const t = i / SPACE.gridRows;
    const z = ROWS_FRONT_Z - (ROWS_FRONT_Z + depth) * t ** 1.6;
    // Every fourth row heavier, so there is a readable rhythm rather than mush.
    floor.add(-half, 0, z, half, 0, z, i % 4 === 0 ? 1.0 : 0.6);
  }

  // No ring on the ground where she stands. Two were tried and cut: from the
  // low viewpoint this framing uses, a circle on the floor is seen at such a
  // grazing angle that it flattens into a wide arc sweeping the bottom of the
  // frame — it stops reading as a mark on the ground and becomes two stray
  // curves in front of her. The contact shadow does the grounding on its own.

  // --- Portals -------------------------------------------------------------
  // Upright wireframe rectangles receding into the space, sized so their
  // angular size shrinks monotonically: they read as a corridor of gates.
  const portals = new LineBuilder();
  for (let i = 0; i < SPACE.portalCount; i += 1) {
    const t = i / Math.max(SPACE.portalCount - 1, 1);
    // Starts just behind her, so the space reads as being behind her rather
    // than as a corridor she happens to be standing in the middle of.
    const z = -1.2 - 30 * t ** 1.45;
    // Narrow. The visible half-width at depth d is only about 0.15*d here, so
    // the previous 1.5 m+ uprights sat outside the frame at every depth they
    // were bright enough to see.
    const w = 0.75 + 0.055 * Math.abs(z);
    const h = 1.9 + 0.09 * Math.abs(z);
    const weight = 0.85 - 0.5 * t;
    const c = LineBuilder.colourFor(i * 7);
    // Corners only, not closed rectangles: a complete frame at every depth
    // becomes a stack of bright boxes competing with her silhouette.
    const bracket = Math.min(w, h) * 0.32;
    const y0 = 0;
    const y1 = h;
    [-w, w].forEach((x) => {
      portals.add(x, y0, z, x, y0 + bracket, z, weight, c);
      portals.add(x, y1, z, x, y1 - bracket, z, weight, c);
      portals.add(x, y0, z, x - Math.sign(x) * bracket, y0, z, weight, c);
      portals.add(x, y1, z, x - Math.sign(x) * bracket, y1, z, weight, c);
    });
  }

  // --- Rails ---------------------------------------------------------------
  // A pair of lines overhead and one to each side. Enough to imply a volume
  // with a ceiling without drawing one.
  const rails = new LineBuilder();
  ([[-0.95, 2.4], [0.95, 2.4], [-1.5, 1.2], [1.5, 1.2]] as [number, number][])
    .forEach(([x, y], i) => {
      rails.add(x, y, FRONT_Z, x, y, -depth, 0.5, LineBuilder.colourFor(i * 13));
    });

  const sets: LineSet[] = [
    floor.build("space-floor", 1, 2, -20),
    portals.build("space-portals", 1, 2, -18),
    rails.build("space-rails", 1, 2, -18),
  ];

  return {
    objects: [
      // Ground-anchored; must not be shifted by any parallax exaggeration.
      { band: "character", object: sets[0].object },
      { band: "near", object: sets[1].object },
      { band: "near", object: sets[2].object },
    ],

    update(dt: number, elapsed: number, mood: SpaceMood) {
      const gain = SPACE.emissiveGain;
      sets[0].uniforms.uInt.value = SPACE.grid * gain;
      sets[1].uniforms.uInt.value = SPACE.portals * gain;
      sets[2].uniforms.uInt.value = SPACE.rails * gain;
      // An intensity of 0 does NOT mean "not drawn": the shader discards at
      // a <= 0.002 (lines.ts), so the whole set was still being submitted,
      // transformed and rasterised every frame just to kill every fragment —
      // and in FOUR passes, since the space layer is also in the TAA velocity
      // pass and the panels' backdrop render. grid/portals/rails all ship at 0
      // (see space-config.ts), so this is three draw calls x four passes of
      // pure waste on the shipped configuration. Purely a visibility gate: any
      // non-zero intensity puts the set straight back.
      for (const set of sets) set.object.visible = set.uniforms.uInt.value > 1e-4;
      // Fade distances live rather than baked, so the ground boundary can be
      // dialled against the real panel without rebuilding the geometry.
      const far = Math.max(SPACE.floorFadeM, 0.5);
      sets.forEach((set, i) => {
        set.uniforms.uTime.value = elapsed;
        set.uniforms.uPulse.value = mood.pulse;
        // The structures stand a little deeper than the floor so they do not all
        // stop on the same line, which would read as a wall.
        const reach = i === 0 ? far : far * 1.6;
        // Hold FULL strength until close to the boundary, then drop over the
        // last quarter. A gentle ramp from 0.35 of the reach meant the floor was
        // already half gone at her feet and the whole grid read as missing —
        // this is a boundary, not a global dimmer.
        set.uniforms.uFadeNear.value = reach * 0.75;
        set.uniforms.uFadeFar.value = reach;
      });
    },

    dispose() {
      sets.forEach((s) => s.dispose());
    },
  };
}
