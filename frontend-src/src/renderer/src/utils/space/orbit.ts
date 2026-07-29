import * as THREE from "three";
import { LineBuilder } from "./lines";
import { SPACE } from "./space-config";
import type { SpaceModule, SpaceMood } from "./types";

/**
 * Rings that ORBIT her — passing in front of her body, then swinging round
 * behind it.
 *
 * This is the answer to "a drawn room keeps reading as a picture". Floor, walls
 * and portal rings all tried to state depth by perspective, and perspective is
 * exactly what a flat image reproduces perfectly, so none of it survived a
 * fixed viewpoint. An orbit states depth a different way: for half of every
 * revolution the ring crosses her silhouette and hides part of her, and for the
 * other half she hides the ring. Depth ORDER is asserted twice per cycle, and a
 * still image cannot assert it even once.
 *
 * It is also not architecture. Nothing here is a surface, a boundary or a room:
 * it is equipment moving around a person, which is why it survives the panel's
 * "no bright planes" rule and does not collide with the real objects behind
 * the glass.
 *
 * The near-future read comes from the construction rather than from colour:
 * gyroscope rings on tilted, incommensurate axes, each broken into arc segments
 * with gaps, so they scan rather than spin. Solid closed circles look like a
 * logo; interrupted arcs look like instrumentation.
 */

interface Ring {
  pivot: THREE.Group; // carries the tilt
  spin: THREE.Group; // carries the rotation
  uniforms: Record<string, { value: number }>;
  axis: THREE.Vector3;
  rate: number; // radians per second
  phase: number;
}

/**
 * One ring as a set of arcs with gaps. `segments` arcs spread evenly, each
 * covering `duty` of its slot.
 */
function makeRing(
  radius: number,
  segments: number,
  duty: number,
  weight: number,
  colour: number,
  ticks: boolean,
) {
  const b = new LineBuilder();
  const STEPS = 14;
  for (let s = 0; s < segments; s += 1) {
    const a0 = (s / segments) * Math.PI * 2;
    const span = ((Math.PI * 2) / segments) * duty;
    for (let i = 0; i < STEPS; i += 1) {
      const t0 = a0 + (i / STEPS) * span;
      const t1 = a0 + ((i + 1) / STEPS) * span;
      b.add(
        Math.cos(t0) * radius, Math.sin(t0) * radius, 0,
        Math.cos(t1) * radius, Math.sin(t1) * radius, 0,
        weight, colour,
      );
    }
    if (ticks) {
      // Short radial marks at each arc start — the detail that makes it read as
      // a graduated instrument rather than a decorative hoop.
      const c = Math.cos(a0);
      const sn = Math.sin(a0);
      b.add(c * radius, sn * radius, 0, c * radius * 1.06, sn * radius * 1.06, 0, weight * 0.8, colour);
    }
  }
  return b.build(`space-orbit-${radius.toFixed(2)}`, 900, 1000, -14);
}

export function createOrbit(): SpaceModule {
  const root = new THREE.Group();
  root.name = "space-orbit";

  const count = Math.max(0, Math.round(SPACE.orbitCount));
  const rings: Ring[] = [];
  const disposers: (() => void)[] = [];

  for (let i = 0; i < count; i += 1) {
    // Radii straddle her: big enough to clear her body at the sides, so the
    // crossing happens across her torso rather than clipping her outline.
    const radius = SPACE.orbitRadius * (0.82 + i * 0.26);
    const set = makeRing(
      radius,
      3 + (i % 3), // 3..5 arcs
      0.52 + (i % 2) * 0.14, // alternating gap ratio
      i === 0 ? 1.15 : 0.85, // the innermost ring reads as the primary
      i % 2 === 0 ? SPACE.lineColor : SPACE.accentA,
      i % 2 === 0,
    );
    const spin = new THREE.Group();
    spin.add(set.object);
    const pivot = new THREE.Group();
    pivot.add(spin);
    // Tilt each ring differently and never to a round number of degrees: two
    // rings sharing a plane read as one wobbling object.
    pivot.rotation.set(
      THREE.MathUtils.degToRad(18 + i * 27),
      THREE.MathUtils.degToRad(11 + i * 41),
      THREE.MathUtils.degToRad(7 + i * 13),
    );
    // Centred on her chest so the crossing lands on her torso.
    pivot.position.set(0, SPACE.orbitHeightM, 0);
    root.add(pivot);

    rings.push({
      pivot,
      spin,
      uniforms: set.uniforms,
      axis: new THREE.Vector3(0, 0, 1),
      // Incommensurate rates: the set never returns to the same arrangement, so
      // it cannot settle into a pattern the eye starts predicting.
      rate: (0.20 + i * 0.085) * (i % 2 === 0 ? 1 : -1),
      phase: i * 1.7,
    });
    disposers.push(set.dispose);
  }

  return {
    objects: [{ band: "near", object: root }],
    update(_dt, elapsed, mood: SpaceMood) {
      const gain = SPACE.orbit * (mood?.energy != null ? 0.75 + mood.energy * 0.5 : 1);
      root.visible = SPACE.orbit > 0.001;
      if (!root.visible) return;
      for (const r of rings) {
        r.spin.rotation.z = r.phase + elapsed * r.rate;
        if (r.uniforms.uInt) r.uniforms.uInt.value = gain;
      }
    },
    dispose() {
      disposers.forEach((d) => d());
      root.clear();
    },
  };
}
