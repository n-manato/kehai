import * as THREE from "three";
import { LineBuilder } from "./lines";
import { SPACE } from "./space-config";
import type { SpaceModule, SpaceMood } from "./types";

/**
 * The far field: thin translucent light columns ("data towers") and small
 * floating hexagonal fragments, thinning and fading with distance.
 *
 * This is where the actual sense of DEPTH behind her comes from. The panels and
 * rings sit close in and establish that objects float at various distances; the
 * towers extend that reading a long way back, because they are the same KIND of
 * object repeated at many depths. Identical objects at different apparent sizes
 * is the oldest depth cue there is, and unlike perspective on a floor it does
 * not require drawing a surface for the eye to run along.
 *
 * Columns rather than buildings, and no ground line: they simply fade out at
 * both ends. A tower that meets a floor states where the floor is, which is the
 * one thing this space must not do — the real world behind the glass supplies
 * the ground, and anything drawn there collides with it.
 *
 * Deliberately dim. Their job is to be *just* readable as structure; brighten
 * them and the far field becomes a lit backdrop and the aperture stops looking
 * transparent.
 *
 * "Dim" is not the same as "darker than the field", and they had drifted into
 * the second. That note was written when the backdrop was WHITE, where dark ink
 * was the only mark that could be seen at all; s30 moved the field to a light
 * lavender and nothing here was re-checked. Measured on the frozen t=6 frame,
 * the towers came out at a mean -3.6/255 against their surround with only 5.3%
 * of their pixels brighter than it — which does not read as distant structure,
 * it reads as grime on the glass. Same root cause as the HUD rings, which were
 * ink for the same reason (see glow-lines.ts).
 *
 * They are now tuned to sit a shade ABOVE the field and no more. They get their
 * own colours rather than the shared `lineColor`/`accentB` precisely so that
 * hitting that target cannot brighten the traffic, wireframe and near-field
 * marks, which are read against a different part of the frame. No glow layer:
 * the note's actual point stands, and the far field must not become a light
 * source.
 */

interface Drifter {
  group: THREE.Group;
  uniforms: Record<string, { value: number }>;
  spin: number;
  bob: number;
  phase: number;
  baseY: number;
  baseInt: number;
}

export function createTowers(): SpaceModule {
  const root = new THREE.Group();
  root.name = "space-towers";
  const disposers: (() => void)[] = [];
  const drifters: Drifter[] = [];

  let seed = 4242;
  const rnd = () => {
    seed = (seed * 1664525 + 1013904223) >>> 0;
    return seed / 4294967296;
  };

  // --- Towers: one line set per depth slab, so each slab can carry its own
  // intensity. Intensity by distance IS the aerial perspective here.
  const SLABS = 4;
  for (let s = 0; s < SLABS; s += 1) {
    const t = s / (SLABS - 1); // 0 = nearest slab, 1 = furthest
    const zNear = -4 - s * 7;
    const zFar = zNear - 7;
    const b = new LineBuilder();
    const perSlab = Math.max(1, Math.round(SPACE.towerCount / SLABS));
    for (let i = 0; i < perSlab; i += 1) {
      const z = zNear + (zFar - zNear) * rnd();
      // Keep a corridor clear behind her so the towers frame her instead of
      // growing out of her head.
      const side = rnd() < 0.5 ? -1 : 1;
      const x = side * (1.1 + rnd() * (3.5 + s * 2.2));
      const h = 2.6 + rnd() * (5 + s * 3);
      const y0 = -0.4 + rnd() * 1.2;
      const w = 0.05 + rnd() * 0.09;
      const colour = rnd() < 0.35 ? SPACE.towerAccent : SPACE.towerColor;
      const weight = 0.85 - t * 0.35;
      // Two verticals plus a few rungs: enough to read as a structure, few
      // enough that a hundred of them stay cheap.
      b.add(x - w, y0, z, x - w, y0 + h, z, weight, colour);
      b.add(x + w, y0, z, x + w, y0 + h, z, weight, colour);
      const rungs = 2 + Math.floor(rnd() * 3);
      for (let r = 1; r <= rungs; r += 1) {
        const yy = y0 + (h * r) / (rungs + 1);
        b.add(x - w, yy, z, x + w, yy, z, weight * 0.7, colour);
      }
    }
    const set = b.build(`space-towers-${s}`, 900, 1000, -19);
    const g = new THREE.Group();
    g.add(set.object);
    root.add(g);
    drifters.push({
      group: g,
      uniforms: set.uniforms,
      spin: 0,
      bob: 0,
      phase: 0,
      baseY: 0,
      // Falls off steeply with depth: the far slab is a suggestion, not a wall.
      baseInt: 0.85 * (1 - t * 0.72),
    });
    disposers.push(set.dispose);
  }

  // --- Floating hex fragments, mid depth, slowly turning.
  for (let i = 0; i < SPACE.hexCount; i += 1) {
    const b = new LineBuilder();
    const size = 0.10 + rnd() * 0.22;
    b.addPolygon(0, 0, 0, size, 6, 0.8, rnd() < 0.4 ? SPACE.towerAccent : SPACE.towerColor, rnd() * Math.PI);
    const set = b.build(`space-hex-${i}`, 900, 1000, -19);
    const g = new THREE.Group();
    g.add(set.object);
    const z = -2.5 - rnd() * 12;
    const x = (rnd() < 0.5 ? -1 : 1) * (0.9 + rnd() * 3.4);
    const y = 0.3 + rnd() * 2.6;
    g.position.set(x, y, z);
    g.rotation.set(rnd() * Math.PI, rnd() * Math.PI, rnd() * Math.PI);
    root.add(g);
    drifters.push({
      group: g,
      uniforms: set.uniforms,
      spin: (0.05 + rnd() * 0.12) * (rnd() < 0.5 ? -1 : 1),
      bob: 0.10 + rnd() * 0.14,
      phase: rnd() * Math.PI * 2,
      baseY: y,
      baseInt: 0.75 - Math.min(0.5, (-z - 2.5) / 24),
    });
    disposers.push(set.dispose);
  }

  return {
    objects: [{ band: "far", object: root }],
    update(_dt, elapsed, mood: SpaceMood) {
      root.visible = SPACE.towers > 0.001;
      if (!root.visible) return;
      const gain = SPACE.towers * (mood?.energy != null ? 0.85 + mood.energy * 0.3 : 1);
      for (const d of drifters) {
        if (d.spin !== 0) {
          d.group.rotation.y += d.spin * 0.016;
          d.group.position.y = d.baseY + Math.sin(elapsed * d.bob + d.phase) * 0.05;
        }
        if (d.uniforms.uInt) d.uniforms.uInt.value = gain * d.baseInt;
      }
    },
    dispose() {
      disposers.forEach((d) => d());
      root.clear();
    },
  };
}
