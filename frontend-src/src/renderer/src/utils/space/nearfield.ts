import * as THREE from "three";
import { LineBuilder } from "./lines";
import { SPACE } from "./space-config";
import type { SpaceModule, SpaceMood } from "./types";

/**
 * A very small number of marks that live BETWEEN the viewer and her, drifting
 * slowly sideways across the frame.
 *
 * The orbit rings assert depth order around her body. These assert the other
 * half of the statement: that the frame has a volume in FRONT of her too. One
 * mark passing over her shoulder and hiding it for a second establishes "she is
 * further away than this" more firmly than any amount of background structure,
 * because occlusion is the one cue that cannot be faked by a flat image and
 * cannot be argued with by the viewer.
 *
 * Kept to a handful of small marks on purpose. Anything dense in the near field
 * stops being individual objects and becomes a texture laid over the picture —
 * at which point it is a dirty screen, not a deep space. Near-future look comes
 * from the shapes being instrument-like (short bars, small brackets, a few
 * points) rather than dust or petals.
 */

interface Mote {
  group: THREE.Group;
  uniforms: Record<string, { value: number }>;
  speed: number; // metres/sec across the frame
  y: number;
  z: number;
  drift: number; // vertical bob rate
  phase: number;
}

/** Sideways travel limits, metres. Wide enough to leave the portrait frame. */
const X_MIN = -1.5;
const X_MAX = 1.5;

function makeMote(index: number, colour: number) {
  const b = new LineBuilder();
  const kind = index % 3;
  if (kind === 0) {
    // Short horizontal bar with end caps — reads as a machined part.
    b.add(-0.05, 0, 0, 0.05, 0, 0, 1.1, colour);
    b.add(-0.05, -0.012, 0, -0.05, 0.012, 0, 1.1, colour);
    b.add(0.05, -0.012, 0, 0.05, 0.012, 0, 1.1, colour);
  } else if (kind === 1) {
    // Open bracket.
    b.add(-0.03, 0.035, 0, -0.05, 0.035, 0, 1.0, colour);
    b.add(-0.05, 0.035, 0, -0.05, -0.035, 0, 1.0, colour);
    b.add(-0.05, -0.035, 0, -0.03, -0.035, 0, 1.0, colour);
  } else {
    // Tiny cross — the smallest thing that still reads as an object, not dirt.
    b.add(-0.018, 0, 0, 0.018, 0, 0, 1.0, colour);
    b.add(0, -0.018, 0, 0, 0.018, 0, 1.0, colour);
  }
  return b.build(`space-near-${index}`, 900, 1000, -10);
}

export function createNearfield(): SpaceModule {
  const root = new THREE.Group();
  root.name = "space-nearfield";

  const count = Math.max(0, Math.round(SPACE.nearfieldCount));
  const motes: Mote[] = [];
  const disposers: (() => void)[] = [];
  let seed = 7717;
  const rnd = () => {
    seed = (seed * 1664525 + 1013904223) >>> 0;
    return seed / 4294967296;
  };

  for (let i = 0; i < count; i += 1) {
    const set = makeMote(i, i % 2 === 0 ? SPACE.accentA : SPACE.lineColor);
    const g = new THREE.Group();
    g.add(set.object);
    // Well in front of her (she stands around z = 0) but behind the near plane,
    // and spread in depth so they do not read as one pane of glass.
    const z = SPACE.nearfieldZ * (0.6 + rnd() * 0.8);
    // Heights biased to her upper body — that is where a crossing actually
    // covers something and is read, rather than passing through empty frame.
    const y = 0.75 + rnd() * 0.9;
    g.position.set(X_MIN + (X_MAX - X_MIN) * rnd(), y, z);
    root.add(g);
    motes.push({
      group: g,
      uniforms: set.uniforms,
      // Nearer motes move FASTER across the frame: that speed difference is
      // motion parallax, and it is doing as much work here as the occlusion.
      speed: (0.045 + rnd() * 0.05) * (z / SPACE.nearfieldZ) * (rnd() < 0.5 ? -1 : 1),
      y,
      z,
      drift: 0.12 + rnd() * 0.15,
      phase: rnd() * Math.PI * 2,
    });
    disposers.push(set.dispose);
  }

  return {
    objects: [{ band: "near", object: root }],
    update(dt, elapsed, mood: SpaceMood) {
      root.visible = SPACE.nearfield > 0.001;
      if (!root.visible) return;
      const gain = SPACE.nearfield * (mood?.energy != null ? 0.8 + mood.energy * 0.4 : 1);
      for (const m of motes) {
        m.group.position.x += m.speed * dt;
        if (m.group.position.x > X_MAX) m.group.position.x = X_MIN;
        if (m.group.position.x < X_MIN) m.group.position.x = X_MAX;
        // Slight vertical bob so they never track a perfectly straight line,
        // which would read as a scanline artefact.
        m.group.position.y = m.y + Math.sin(elapsed * m.drift + m.phase) * 0.035;
        if (m.uniforms.uInt) m.uniforms.uInt.value = gain;
      }
    },
    dispose() {
      disposers.forEach((d) => d());
      root.clear();
    },
  };
}
