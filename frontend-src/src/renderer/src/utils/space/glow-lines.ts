import * as THREE from "three";

/**
 * Ribbon lines that read as LIGHT rather than as ink.
 *
 * `lines.ts` draws everything with GL LINES and normal alpha blending, which is
 * right for the structure that should read as a technical drawing (traffic,
 * towers, wireframe). It cannot make the HUD rings look like a lit instrument,
 * and the reason is measurable rather than aesthetic:
 *
 *   - A GL line is ONE pixel wide whatever `weight` says — in `lines.ts` that
 *     parameter is alpha, not width. On the PROTO panel the pixel ratio is capped
 *     at 3, so the ring is a third of a CSS pixel. There is no room across the
 *     mark for a core and a falloff, and a glow IS a falloff.
 *   - Composited normally, a 0x1fa8d8 line over the 0xc4c2cc field lands BELOW
 *     the field in luminance. Measured on the frozen t=6 frame: 82.5% of ring
 *     pixels were darker than their surround, mean delta -9.6/255. That is the
 *     definition of ink.
 *   - Nothing in the ring came within reach of the bloom threshold (0.85, i.e.
 *     217/255) — ring pixels measured L=157 — so UnrealBloom contributed exactly
 *     nothing and there was no halo to see.
 *
 * So this module does the three things that follow from those measurements: give
 * the mark WIDTH, composite it ADDITIVELY, and drive the core bright enough to
 * cross the bloom threshold so the post chain adds an outer halo on top of the
 * one drawn here.
 *
 * Additive is only available because the backdrop moved off white (s30). The
 * warning in `lines.ts` — "additive means add light, and on a white field there
 * is no headroom" — was correct for the white field it was written against. The
 * field is now 0xc4c2cc, measured mean luminance 167/255, which leaves ~88
 * levels of headroom. That headroom is the whole reason a glow can exist here.
 *
 * Geometry is a triangle strip expanded IN THE PATH'S OWN PLANE rather than in
 * screen space. The rings are flat, fronto-parallel and all within z -1.7..-2.8
 * against a camera at +3.1, so the depth spread is under 20% and a world-space
 * width is within a pixel of a screen-space one — for a fraction of the vertex
 * shader. It also means the far halo ring is genuinely thinner than the near
 * one, which is the correct perspective rather than a uniform sticker.
 */

const GLOW_VERT = /* glsl */ `
attribute vec3 aColor;
attribute float aAcross;
attribute float aInt;
attribute float aScale;
varying vec3 vColor;
varying float vAcross;
varying float vInt;
varying float vScale;
varying float vDepth;
varying vec3 vWorld;

void main() {
  vec4 world = modelMatrix * vec4(position, 1.0);
  vWorld = world.xyz;
  vec4 mv = viewMatrix * world;
  vDepth = -mv.z;
  vColor = aColor;
  vAcross = aAcross;
  vInt = aInt;
  vScale = aScale;
  gl_Position = projectionMatrix * mv;
}
`;

/**
 * `vAcross` arrives in METRES from the centre of the mark, not normalised, so
 * the core and halo widths below are real distances and can be reasoned about
 * against the ring radius instead of against an arbitrary 0..1.
 */
const GLOW_FRAG = /* glsl */ `
varying vec3 vColor;
varying float vAcross;
varying float vInt;
varying float vScale;
varying float vDepth;
varying vec3 vWorld;

uniform float uInt;
uniform float uCore;
uniform float uHalo;
uniform float uCoreW;
uniform float uHaloW;
uniform float uFadeNear;
uniform float uFadeFar;
uniform float uTime;
uniform float uPulse;

void main() {
  float fade = 1.0 - smoothstep(uFadeNear, uFadeFar, vDepth);
  if (fade <= 0.001) discard;

  float d = abs(vAcross);
  // Gaussian core: the filament itself, small and driven hard so it clips to
  // white and crosses the bloom threshold.
  float cw = max(uCoreW * vScale, 1e-5);
  float core = exp(-(d * d) / (cw * cw) * 2.5);
  // Exponential halo: the light around the filament. Exponential rather than
  // gaussian because a real bloom skirt has a long tail, and a gaussian's falls
  // off too abruptly to read as glow.
  float hw = max(uHaloW * vScale, 1e-5);
  float halo = exp(-d / hw * 3.0);

  // Same breathing as lines.ts, so the console never reads as a frozen still.
  float breathe = 1.0 + 0.06 * uPulse * sin(uTime * 0.6 + vWorld.z * 0.15);

  float g = (uCore * core + uHalo * halo) * fade * uInt * vInt * breathe;
  if (g <= 0.002) discard;
  // Additive: alpha is the blend factor, so the whole quantity goes in RGB and
  // alpha stays at 1. Writing the gain into alpha instead would square it.
  gl_FragColor = vec4(vColor * g, 1.0);
}
`;

export interface GlowSet {
  object: THREE.Mesh;
  uniforms: Record<string, { value: number }>;
  dispose(): void;
}

type Pt = [number, number, number];

/**
 * Accumulates ribbon paths, then bakes them into one draw call.
 *
 * Paths are polylines in a plane of constant z; the ribbon is expanded along the
 * in-plane perpendicular. Corners are mitred, because the hex frames turn 60
 * degrees at each vertex and per-segment normals would leave a visible notch on
 * the outside of every corner.
 */
export class GlowRibbonBuilder {
  private pos: number[] = [];

  private col: number[] = [];

  private across: number[] = [];

  private inten: number[] = [];

  private scale: number[] = [];

  private idx: number[] = [];

  private vcount = 0;

  private c = new THREE.Color();

  /**
   * @param pts     polyline, in a plane of constant z
   * @param half    ribbon half-width in metres — the extent the halo may reach
   * @param scale   per-path width multiplier the shader applies to core/halo
   * @param closed  join the last point back to the first (polygons)
   */
  addPath(pts: Pt[], half: number, colour: number, intensity = 1, scale = 1, closed = false) {
    const n = pts.length;
    if (n < 2) return;

    // In-plane normal of each segment, in the XY plane.
    const segN: [number, number][] = [];
    const last = closed ? n : n - 1;
    for (let i = 0; i < last; i += 1) {
      const a = pts[i];
      const b = pts[(i + 1) % n];
      const dx = b[0] - a[0];
      const dy = b[1] - a[1];
      const len = Math.hypot(dx, dy) || 1;
      segN.push([-dy / len, dx / len]);
    }

    const c = this.c.set(colour);
    const base = this.vcount;

    for (let i = 0; i < n; i += 1) {
      // Mitre: average the normals either side of the vertex and lengthen by
      // 1/cos(theta/2) so the ribbon edge stays parallel through the corner.
      const prev = closed ? segN[(i - 1 + segN.length) % segN.length] : segN[Math.max(0, i - 1)];
      const next = closed ? segN[i % segN.length] : segN[Math.min(segN.length - 1, i)];
      let nx = prev[0] + next[0];
      let ny = prev[1] + next[1];
      const nl = Math.hypot(nx, ny) || 1;
      nx /= nl;
      ny /= nl;
      // dot with either segment normal gives cos(theta/2); clamp so a hairpin
      // cannot produce an enormous spike.
      const cosHalf = Math.max(0.35, nx * next[0] + ny * next[1]);
      const ext = half / cosHalf;

      const p = pts[i];
      for (let s = -1; s <= 1; s += 2) {
        this.pos.push(p[0] + nx * ext * s, p[1] + ny * ext * s, p[2]);
        this.col.push(c.r, c.g, c.b);
        this.across.push(half * s);
        this.inten.push(intensity);
        this.scale.push(scale);
      }
      this.vcount += 2;
    }

    const quads = closed ? n : n - 1;
    for (let i = 0; i < quads; i += 1) {
      const a = base + i * 2;
      const b = base + ((i + 1) % n) * 2;
      this.idx.push(a, a + 1, b + 1, a, b + 1, b);
    }
  }

  /** Arc of `radius` from `a0` spanning `span` radians, at `z`. */
  addArc(
    radius: number, a0: number, span: number, z: number,
    half: number, colour: number, intensity = 1, scale = 1, steps = 40,
  ) {
    const pts: Pt[] = [];
    for (let i = 0; i <= steps; i += 1) {
      const t = a0 + (i / steps) * span;
      pts.push([Math.cos(t) * radius, Math.sin(t) * radius, z]);
    }
    this.addPath(pts, half, colour, intensity, scale, false);
  }

  /** Closed regular polygon in the XY plane at `z`. */
  addPolygon(
    cx: number, cy: number, cz: number,
    radius: number, sides: number,
    half: number, colour: number, intensity = 1, scale = 1, rotate = 0,
  ) {
    const pts: Pt[] = [];
    for (let i = 0; i < sides; i += 1) {
      const a = rotate + (i / sides) * Math.PI * 2;
      pts.push([cx + Math.cos(a) * radius, cy + Math.sin(a) * radius, cz]);
    }
    this.addPath(pts, half, colour, intensity, scale, true);
  }

  /** Straight segment — graduation ticks and the like. */
  addSegment(
    x1: number, y1: number, x2: number, y2: number, z: number,
    half: number, colour: number, intensity = 1, scale = 1,
  ) {
    this.addPath([[x1, y1, z], [x2, y2, z]], half, colour, intensity, scale, false);
  }

  get empty() {
    return this.idx.length === 0;
  }

  build(
    name: string,
    opts: {
      fadeNear: number; fadeFar: number; renderOrder?: number;
      core: number; halo: number; coreW: number; haloW: number;
    },
  ): GlowSet {
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.Float32BufferAttribute(this.pos, 3));
    geo.setAttribute("aColor", new THREE.Float32BufferAttribute(this.col, 3));
    geo.setAttribute("aAcross", new THREE.Float32BufferAttribute(this.across, 1));
    geo.setAttribute("aInt", new THREE.Float32BufferAttribute(this.inten, 1));
    geo.setAttribute("aScale", new THREE.Float32BufferAttribute(this.scale, 1));
    geo.setIndex(this.idx);

    const uniforms = {
      uInt: { value: 1 },
      uCore: { value: opts.core },
      uHalo: { value: opts.halo },
      uCoreW: { value: opts.coreW },
      uHaloW: { value: opts.haloW },
      uFadeNear: { value: opts.fadeNear },
      uFadeFar: { value: opts.fadeFar },
      uTime: { value: 0 },
      uPulse: { value: 1 },
    };

    const mat = new THREE.ShaderMaterial({
      uniforms,
      vertexShader: GLOW_VERT,
      fragmentShader: GLOW_FRAG,
      transparent: true,
      blending: THREE.AdditiveBlending,
      // Depth TEST stays on so the character still occludes the console — that
      // occlusion is the whole reason the rings sit behind her. Depth WRITE is
      // off so overlapping ribbons sum instead of the nearest one winning, which
      // is what makes crossings brighten like real light.
      depthWrite: false,
      side: THREE.DoubleSide,
    });

    const object = new THREE.Mesh(geo, mat);
    object.frustumCulled = false;
    object.renderOrder = opts.renderOrder ?? -18;
    object.name = name;

    return {
      object,
      uniforms,
      dispose() {
        geo.dispose();
        mat.dispose();
      },
    };
  }
}
