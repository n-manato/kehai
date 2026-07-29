import * as THREE from "three";
import { GlowRibbonBuilder } from "./glow-lines";
import { SPACE } from "./space-config";
import type { SpaceModule, SpaceMood } from "./types";

/**
 * The big concentric ring system directly behind her, plus hexagonal frames.
 *
 * This is the "AI control centre" backplate: one large ring centred on her, a
 * few concentric arc bands at slightly different radii and rotation rates, and
 * scattered hex frames further out. Symmetric and centred, so she reads as the
 * subject the whole apparatus is built around rather than something standing in
 * front of scenery.
 *
 * Why rings work here where the earlier orbiting ones did not: those were thin
 * arcs at her own depth, so they read as scratches ON the picture. These sit
 * clearly BEHIND her at several distinct depths, and she occludes them — the
 * occlusion runs the right way round, which is what states that she is in front
 * of a space rather than pasted onto a pattern.
 *
 * Drawn with `glow-lines.ts` ribbons, NOT with the `lines.ts` GL lines the rest
 * of the environment uses. A GL line is one pixel wide and composites as ink: on
 * the frozen t=6 frame the old rings measured 82.5% of their pixels DARKER than
 * the field they sat on, at L=157 against a bloom threshold of 217 — so they
 * could not glow, and no amount of intensity on a one-pixel dark mark was going
 * to make them. See glow-lines.ts for the rest of that measurement.
 *
 * Still no filled disc: a bright filled circle behind her would turn the
 * aperture into a lit panel, which is the one thing this display must never look
 * like. A ribbon with a core and a falloff is a line with width, not an area.
 */

interface Band {
  group: THREE.Group;
  uniforms: Record<string, { value: number }>;
  rate: number;
  phase: number;
  baseInt: number;
}

export function createHudRings(): SpaceModule {
  const root = new THREE.Group();
  root.name = "space-hud-rings";
  const bands: Band[] = [];
  const disposers: (() => void)[] = [];

  const R = SPACE.hudRadius;
  const Z = SPACE.hudZ;
  const Y = SPACE.hudHeightM;
  // Bake the ribbon a little wider than the halo actually needs, so the width
  // can be pushed live with __space({ hudHaloW }) without the glow being cut
  // off square at the edge of the geometry.
  const HALO_MARGIN = 1.25;

  // Each band is its own object so it can rotate at its own rate — a single
  // static ring set is a graphic, several turning slowly against each other is
  // a machine.
  const specs: {
    radius: number; segments: number; duty: number; scale: number;
    colour: number; rate: number; ticks?: number; z: number; int: number;
  }[] = [
    // The dominant outer ring, nearly closed.
    { radius: R, segments: 4, duty: 0.94, scale: 1.0, colour: SPACE.hudColor, rate: 0.035, ticks: 72, z: 0, int: 1.0 },
    // Inner bands, more broken, counter-rotating.
    { radius: R * 0.86, segments: 6, duty: 0.62, scale: 0.72, colour: SPACE.hudAccent, rate: -0.06, z: -0.25, int: 0.8 },
    { radius: R * 0.71, segments: 3, duty: 0.40, scale: 0.6, colour: SPACE.hudColor, rate: 0.09, z: -0.45, int: 0.62 },
    // A wide faint halo well behind, to give the system a far edge.
    { radius: R * 1.35, segments: 8, duty: 0.30, scale: 0.5, colour: SPACE.hudAccent, rate: -0.02, z: -1.1, int: 0.4 },
  ];

  specs.forEach((sp, i) => {
    const b = new GlowRibbonBuilder();
    const half = SPACE.hudHaloW * sp.scale * HALO_MARGIN;
    for (let s = 0; s < sp.segments; s += 1) {
      const a0 = (s / sp.segments) * Math.PI * 2;
      const span = ((Math.PI * 2) / sp.segments) * sp.duty;
      b.addArc(sp.radius, a0, span, 0, half, sp.colour, 1, sp.scale, 44);
    }
    if (sp.ticks) {
      // Graduation ticks — the detail that reads as instrumentation. Thinner
      // than the ring they hang off, or they read as a second ring.
      const tScale = sp.scale * 0.55;
      const tHalf = SPACE.hudHaloW * tScale * HALO_MARGIN;
      const len = R * 0.035;
      for (let t = 0; t < sp.ticks; t += 1) {
        const a = (t / sp.ticks) * Math.PI * 2;
        const c = Math.cos(a);
        const s2 = Math.sin(a);
        b.addSegment(
          c * sp.radius, s2 * sp.radius,
          c * (sp.radius + len), s2 * (sp.radius + len),
          0, tHalf, sp.colour, 0.75, tScale,
        );
      }
    }
    const set = b.build(`space-hud-band-${i}`, {
      fadeNear: 900, fadeFar: 1000, renderOrder: -18,
      core: SPACE.hudCore, halo: SPACE.hudHalo,
      coreW: SPACE.hudCoreW, haloW: SPACE.hudHaloW,
    });
    const g = new THREE.Group();
    g.add(set.object);
    g.position.set(0, Y, Z + sp.z);
    root.add(g);
    bands.push({ group: g, uniforms: set.uniforms, rate: sp.rate, phase: i * 0.9, baseInt: sp.int });
    disposers.push(set.dispose);
  });

  // Hex frames scattered around the rings — static, at mixed depths, so the
  // turning bands have something fixed to turn against.
  {
    const b = new GlowRibbonBuilder();
    let seed = 20260729;
    const rnd = () => {
      seed = (seed * 1664525 + 1013904223) >>> 0;
      return seed / 4294967296;
    };
    const hScale = 0.55;
    const hHalf = SPACE.hudHaloW * hScale * HALO_MARGIN;
    for (let i = 0; i < SPACE.hudHexCount; i += 1) {
      const a = rnd() * Math.PI * 2;
      const rr = R * (0.95 + rnd() * 0.9);
      const size = R * (0.07 + rnd() * 0.09);
      b.addPolygon(
        Math.cos(a) * rr, Math.sin(a) * rr, -rnd() * 2.2,
        size, 6, hHalf,
        rnd() < 0.4 ? SPACE.hudAccent : SPACE.hudColor,
        1, hScale, rnd() * Math.PI,
      );
    }
    const set = b.build("space-hud-hex", {
      fadeNear: 900, fadeFar: 1000, renderOrder: -18,
      core: SPACE.hudCore, halo: SPACE.hudHalo,
      coreW: SPACE.hudCoreW, haloW: SPACE.hudHaloW,
    });
    const g = new THREE.Group();
    g.add(set.object);
    g.position.set(0, Y, Z - 0.6);
    root.add(g);
    bands.push({ group: g, uniforms: set.uniforms, rate: 0, phase: 0, baseInt: 0.55 });
    disposers.push(set.dispose);
  }

  return {
    objects: [{ band: "near", object: root }],
    update(_dt, elapsed, mood: SpaceMood) {
      root.visible = SPACE.hud > 0.001;
      if (!root.visible) return;
      const gain = SPACE.hud * (mood?.energy != null ? 0.8 + mood.energy * 0.35 : 1);
      for (const bd of bands) {
        if (bd.rate !== 0) bd.group.rotation.z = bd.phase + elapsed * bd.rate;
        const u = bd.uniforms;
        if (u.uInt) u.uInt.value = gain * bd.baseInt;
        // Re-read the shape uniforms every frame so __space({ hudCore, ... })
        // takes effect without a rebuild, the same contract every other module
        // in this layer honours.
        if (u.uCore) u.uCore.value = SPACE.hudCore;
        if (u.uHalo) u.uHalo.value = SPACE.hudHalo;
        if (u.uCoreW) u.uCoreW.value = SPACE.hudCoreW;
        if (u.uHaloW) u.uHaloW.value = SPACE.hudHaloW;
        if (u.uTime) u.uTime.value = elapsed;
        if (u.uPulse) u.uPulse.value = mood?.pulse ?? 1;
      }
    },
    dispose() {
      disposers.forEach((d) => d());
      root.clear();
    },
  };
}
