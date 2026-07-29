import * as THREE from "three";
import { SPACE } from "./space-config";

/**
 * Every mark in this environment is a GL line or a point. Nothing is a filled
 * surface.
 *
 * That is a hardware constraint, not a style choice. On the PROTO panel a filled
 * AREA of colour reads as "there is a display here" and collapses the illusion
 * that the character is in the box; a white field with thin coloured lines on it
 * does not. So the whole space is drawn with one-pixel lines and points, which
 * also happens to be the right look for a digital/neural space and keeps it
 * simple.
 */

const LINE_VERT = /* glsl */ `
attribute vec3 aColor;
attribute float aWeight;
varying vec3 vColor;
varying float vWeight;
varying float vDepth;
varying vec3 vWorld;

void main() {
  vec4 world = modelMatrix * vec4(position, 1.0);
  vWorld = world.xyz;
  vec4 mv = viewMatrix * world;
  vDepth = -mv.z;
  vColor = aColor;
  vWeight = aWeight;
  gl_Position = projectionMatrix * mv;
}
`;

const LINE_FRAG = /* glsl */ `
varying vec3 vColor;
varying float vWeight;
varying float vDepth;
varying vec3 vWorld;

uniform float uInt;
uniform float uFadeNear;
uniform float uFadeFar;
uniform float uTime;
uniform float uPulse;

void main() {
  // Atmospheric perspective: fade the line's OPACITY out with distance so it
  // dissolves into the field, whatever colour the field happens to be.
  float fade = 1.0 - smoothstep(uFadeNear, uFadeFar, vDepth);
  if (fade <= 0.001) discard;

  // Very slight breathing so the space is never a frozen still frame; a static
  // image is read as a picture no matter how good its perspective is.
  float breathe = 1.0 + 0.06 * uPulse * sin(uTime * 0.6 + vWorld.z * 0.15);

  float a = uInt * vWeight * fade * breathe;
  if (a <= 0.002) discard;
  gl_FragColor = vec4(vColor, a);
}
`;

export interface LineSet {
  object: THREE.LineSegments;
  uniforms: Record<string, { value: number }>;
  dispose(): void;
}

/** Accumulates line segments, then bakes them into one draw call. */
export class LineBuilder {
  private pos: number[] = [];

  private col: number[] = [];

  private wgt: number[] = [];

  private cA = new THREE.Color();

  private cB = new THREE.Color();

  add(
    x1: number, y1: number, z1: number,
    x2: number, y2: number, z2: number,
    weight = 1,
    colour?: number,
  ) {
    this.pos.push(x1, y1, z1, x2, y2, z2);
    const c = this.cA.set(colour ?? SPACE.lineColor);
    for (let i = 0; i < 2; i += 1) {
      this.col.push(c.r, c.g, c.b);
      this.wgt.push(weight);
    }
  }

  /** Closed polygon in the XY plane at `z`, rotated/scaled by the caller. */
  addPolygon(
    cx: number, cy: number, cz: number,
    radius: number, sides: number, weight = 1, colour?: number, rotate = 0,
  ) {
    for (let i = 0; i < sides; i += 1) {
      const a0 = rotate + (i / sides) * Math.PI * 2;
      const a1 = rotate + ((i + 1) / sides) * Math.PI * 2;
      this.add(
        cx + Math.cos(a0) * radius, cy + Math.sin(a0) * radius, cz,
        cx + Math.cos(a1) * radius, cy + Math.sin(a1) * radius, cz,
        weight, colour,
      );
    }
  }

  get empty() {
    return this.pos.length === 0;
  }

  build(name: string, fadeNear: number, fadeFar: number, renderOrder = -20): LineSet {
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.Float32BufferAttribute(this.pos, 3));
    geo.setAttribute("aColor", new THREE.Float32BufferAttribute(this.col, 3));
    geo.setAttribute("aWeight", new THREE.Float32BufferAttribute(this.wgt, 1));

    const uniforms = {
      uInt: { value: 1 },
      uFadeNear: { value: fadeNear },
      uFadeFar: { value: fadeFar },
      uTime: { value: 0 },
      uPulse: { value: 1 },
    };
    const mat = new THREE.ShaderMaterial({
      uniforms,
      vertexShader: LINE_VERT,
      fragmentShader: LINE_FRAG,
      transparent: true,
      // NOT additive. Additive means "add light", and on a white field there is
      // no headroom left to add any — every line would vanish. Normal alpha
      // blending composites the ink over the field, which is what a drawing on
      // white needs.
      blending: THREE.NormalBlending,
      // The character is opaque and writes depth, so she still occludes the
      // parts of the space behind her — and anything placed in FRONT of her
      // draws over her, which is where the occlusion depth cue comes from.
      depthWrite: false,
    });

    const object = new THREE.LineSegments(geo, mat);
    object.frustumCulled = false;
    object.renderOrder = renderOrder;
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

  /** Pick white or an accent, deterministically, for a given index. */
  static colourFor(i: number): number {
    const r = ((i * 2654435761) % 1000) / 1000;
    if (r > SPACE.accentMix) return SPACE.lineColor;
    return r > SPACE.accentMix * 0.5 ? SPACE.accentA : SPACE.accentB;
  }
}
