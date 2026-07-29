import * as THREE from "three";
import { LineBuilder } from "./lines";
import { SPACE } from "./space-config";
import type { SpaceModule, SpaceMood } from "./types";

/**
 * Wireframe shapes travelling from the far end, past her, and out through the
 * front of the box.
 *
 * This is the load-bearing part of the whole redesign. On a static flat panel,
 * perspective alone produces a *picture* of a deep space — the previous version
 * had convergence, atmospheric fade and a receding ring corridor, and still
 * read flat, because none of it changed. Two cues do not care that the viewpoint
 * is fixed:
 *
 *   - OCCLUSION. A shape that passes in front of her crosses her body and hides
 *     part of it, then a moment later is behind her and she hides it. Nothing
 *     else states depth order as unambiguously, and it costs one line crossing
 *     a silhouette.
 *   - LOOMING. Something growing in angular size is read as approaching, and
 *     the rate of growth reads as distance. A ring that swells and sweeps past
 *     the viewpoint gives the eye a continuous depth track through the scene.
 *
 * The shapes are deliberately open and few. They are traffic through the space,
 * not decoration in it; a stream of them would become texture and stop reading
 * as individual objects moving through depth.
 */

interface Mover {
  group: THREE.Group;
  uniforms: Record<string, { value: number }>;
  z: number;
  speed: number;
  spin: number;
  radius: number;
  x: number;
  /** -1..1 offset within the height band; the metres are resolved per frame. */
  yOffset: number;
}

/** Where things enter, and where they are recycled after passing the viewer. */
const START_Z = -32;
const END_Z = 3.2;

function makeShape(index: number) {
  const b = new LineBuilder();
  const colour = LineBuilder.colourFor(index * 31);
  const kind = index % 3;
  if (kind === 0) {
    b.addPolygon(0, 0, 0, 1, 40, 1, colour); // ring
  } else if (kind === 1) {
    b.addPolygon(0, 0, 0, 1, 6, 1, colour); // hexagon
  } else {
    // Open bracket pair — reads as a frame passing by without closing a box
    // around her.
    for (let s = -1; s <= 1; s += 2) {
      b.add(s, -1, 0, s, -0.45, 0, 1, colour);
      b.add(s, 1, 0, s, 0.45, 0, 1, colour);
      b.add(s, -1, 0, s * 0.5, -1, 0, 1, colour);
      b.add(s, 1, 0, s * 0.5, 1, 0, 1, colour);
    }
  }
  // Fades over the full travel range; these must be gone before they clip the
  // near plane, or they flash out.
  // Distance fade is driven per-mover in update() instead of by the shader's
  // depth ramp: what matters here is the fade as it PASSES the viewer, and the
  // shader only knows how to fade things that are far away.
  return b.build(`space-traffic-${index}`, 900, 1000, -16);
}

export function createTraffic(): SpaceModule {
  const root = new THREE.Group();
  root.name = "space-traffic";

  const count = Math.max(0, Math.round(SPACE.trafficCount));
  const movers: Mover[] = [];
  let seed = 90210;
  const rnd = () => {
    seed = (seed * 1664525 + 1013904223) >>> 0;
    return seed / 4294967296;
  };

  for (let i = 0; i < count; i += 1) {
    const g = new THREE.Group();
    const set = makeShape(i);
    g.add(set.object);
    // Spread along the travel so they arrive staggered rather than in a pulse.
    const z = START_Z + ((END_Z - START_Z) * i) / Math.max(count, 1);
    /**
     * Small — a shape has to be smaller than she is to read as an OBJECT going
     * past her. At the previous 0.7-2.0 m they grew to over two metres across on
     * approach, ran off all four edges and appeared as a pair of giant arcs
     * sweeping the frame: no longer something at shoulder height, just curved
     * lines in the background.
     */
    const radius = 0.22 + rnd() * 0.28;
    // Close to her, not scattered wide: at this size a shape 1.3 m off-axis
    // sails past in empty space and never touches her silhouette, and crossing
    // the silhouette is the entire point.
    const x = (rnd() - 0.5) * 1.2;
    const yOffset = rnd() * 2 - 1;
    g.position.set(x, SPACE.trafficHeightM + yOffset * SPACE.trafficSpreadM, z);
    g.scale.setScalar(radius);
    root.add(g);
    movers.push({
      group: g, uniforms: set.uniforms, z,
      speed: 0.7 + rnd() * 0.7, spin: (rnd() - 0.5) * 0.25, radius, x, yOffset,
    });
  }

  return {
    // Spans the whole depth by design, so it is parented near the character
    // rather than to a band that a parallax pass would shift.
    objects: [{ band: "character", object: root }],

    update(dt: number, elapsed: number, mood: SpaceMood) {
      root.visible = SPACE.traffic > 0.001;
      if (!root.visible) return;
      // Thinking/speaking speeds the flow up — the same "data streaming" read as
      // before, but now carried by objects whose motion also creates depth.
      const speed = SPACE.trafficSpeed * (1 + 1.6 * mood.stream);
      for (let i = 0; i < movers.length; i += 1) {
        const m = movers[i];
        m.z += speed * m.speed * dt;
        if (m.z > END_Z) {
          m.z = START_Z;
        }
        m.group.position.z = m.z;
        // Height resolved every frame rather than baked at build time, so
        // trafficHeightM can be dialled against the real panel without a rebuild.
        m.group.position.y = SPACE.trafficHeightM + m.yOffset * SPACE.trafficSpreadM;
        m.group.rotation.z += m.spin * dt;
        // Grow with proximity so the angular size climbs faster than the
        // distance shrinks — that acceleration is what reads as "coming at you".
        const t = (m.z - START_Z) / (END_Z - START_Z);
        // Modest growth. Enough to read as approaching, not so much that a shape
        // ends up larger than the frame while it is passing her.
        m.group.scale.setScalar(m.radius * (0.55 + 0.5 * t));
        // Fade in out of the far dark, and fade back out while sweeping past the
        // viewpoint. Without the second half a shape ends up metres across and
        // full brightness right in front of the lens, which stops reading as
        // "something went by" and becomes a white bar across the frame.
        const fadeIn = Math.min(1, t / 0.18);
        const fadeOut = 1 - Math.max(0, (t - 0.72) / 0.28);
        m.uniforms.uInt.value = SPACE.traffic * SPACE.emissiveGain
          * Math.max(0, fadeIn * fadeOut);
      }
    },

    dispose() {
      root.traverse((o) => {
        const mesh = o as THREE.LineSegments;
        mesh.geometry?.dispose?.();
        (mesh.material as THREE.Material | undefined)?.dispose?.();
      });
    },
  };
}
