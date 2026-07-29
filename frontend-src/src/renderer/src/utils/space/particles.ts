import * as THREE from "three";
import { HAZE_FRAGMENT, HAZE_PARS, HAZE_VERTEX, hazeUniforms, syncHaze } from "./haze";
import { SPACE, getBudget } from "./space-config";
import type { HazeUniforms } from "./haze";
import type { BandName } from "./layers";
import type { SpaceModule, SpaceMood } from "./types";

/**
 * Particle fields, one per depth band.
 *
 * They do two jobs the geometry can't. First, they populate the volume BETWEEN
 * the structures, so the space has a measurable thickness rather than being a
 * set of surfaces at discrete distances. Second — the foreground layer, the few
 * motes drifting between the glass and Miku — is the only thing in Phase 1 that
 * is unambiguously in FRONT of her, which is what makes her sit inside the box
 * instead of on its front face.
 *
 * Drift is done entirely in the vertex shader against a wrapped clock, so the
 * CPU never touches the buffers: a few thousand points cost one draw call and no
 * per-frame work, which matters because Phase 3's head tracking needs the frame
 * budget.
 */

const PARTICLE_VERT = /* glsl */ `
${HAZE_PARS}
attribute float aSize;
attribute float aSpeed;
attribute float aPhase;
varying float vFade;

uniform float uTime;
uniform float uDrift;
uniform float uZNear;
uniform float uZFar;
uniform float uSize;
uniform float uPixelRatio;

void main() {
  vec3 p = position;

  // Wrap the band so the field is endless without ever regenerating it.
  float len = uZNear - uZFar;
  float z = p.z - uDrift * aSpeed;
  p.z = uZFar + mod(z - uZFar, len);

  // A touch of lateral sway so it reads as suspended dust, not a moving wall.
  p.x += sin(uTime * 0.25 + aPhase) * 0.25;
  p.y += cos(uTime * 0.19 + aPhase * 1.7) * 0.18;

  vec4 mvPosition = viewMatrix * modelMatrix * vec4(p, 1.0);
  ${HAZE_VERTEX}

  // Fade in at both ends of the band so wrapping never pops.
  float edge = min(uZNear - p.z, p.z - uZFar) / max(len * 0.15, 0.001);
  vFade = clamp(edge, 0.0, 1.0);

  gl_PointSize = uSize * aSize * uPixelRatio * (12.0 / max(-mvPosition.z, 0.1));
  gl_Position = projectionMatrix * mvPosition;
}
`;

const PARTICLE_FRAG = /* glsl */ `
${HAZE_PARS}
${HAZE_FRAGMENT}
varying float vFade;
uniform vec3 uColor;
uniform float uOpacity;

void main() {
  // Soft round mote. Squaring the ramp keeps the core tight so bloom has a
  // point to grab instead of smearing a fuzzy disc.
  float d = length(gl_PointCoord - 0.5);
  float m = smoothstep(0.5, 0.0, d);
  float a = m * m * uOpacity * vFade * (1.0 - hazeFactor());
  if (a < 0.004) discard;
  gl_FragColor = vec4(uColor, a);
}
`;

interface Field {
  points: THREE.Points;
  uniforms: Record<string, { value: unknown }> & HazeUniforms;
  geo: THREE.BufferGeometry;
  mat: THREE.ShaderMaterial;
  band: BandName;
  baseOpacity: number;
  driftRate: number;
}

function makeField(opts: {
  band: BandName;
  count: number;
  zNear: number;
  zFar: number;
  spread: number;
  yMin: number;
  yMax: number;
  color: number;
  size: number;
  opacity: number;
  driftRate: number;
  seed: number;
  name: string;
}): Field {
  let s = opts.seed >>> 0;
  const rnd = () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };

  const n = Math.max(opts.count, 1);
  const position = new Float32Array(n * 3);
  const aSize = new Float32Array(n);
  const aSpeed = new Float32Array(n);
  const aPhase = new Float32Array(n);
  for (let i = 0; i < n; i += 1) {
    position[i * 3] = (rnd() - 0.5) * opts.spread;
    position[i * 3 + 1] = opts.yMin + rnd() * (opts.yMax - opts.yMin);
    position[i * 3 + 2] = opts.zFar + rnd() * (opts.zNear - opts.zFar);
    aSize[i] = 0.5 + rnd() * 1.1;
    aSpeed[i] = 0.55 + rnd() * 0.9;
    aPhase[i] = rnd() * Math.PI * 2;
  }

  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(position, 3));
  geo.setAttribute("aSize", new THREE.BufferAttribute(aSize, 1));
  geo.setAttribute("aSpeed", new THREE.BufferAttribute(aSpeed, 1));
  geo.setAttribute("aPhase", new THREE.BufferAttribute(aPhase, 1));

  const uniforms = {
    uTime: { value: 0 },
    uDrift: { value: 0 },
    uZNear: { value: opts.zNear },
    uZFar: { value: opts.zFar },
    uSize: { value: opts.size },
    uPixelRatio: { value: Math.min(window.devicePixelRatio || 1, 1.5) },
    uColor: { value: new THREE.Color(opts.color) },
    uOpacity: { value: opts.opacity },
    ...hazeUniforms(),
  };

  const mat = new THREE.ShaderMaterial({
    uniforms,
    vertexShader: PARTICLE_VERT,
    fragmentShader: PARTICLE_FRAG,
    transparent: true,
    depthWrite: false,
    blending: THREE.NormalBlending,
  });

  const points = new THREE.Points(geo, mat);
  points.frustumCulled = false; // the shader moves them; the CPU bounds are stale
  points.renderOrder = -8;
  points.name = opts.name;

  return {
    points,
    uniforms,
    geo,
    mat,
    band: opts.band,
    baseOpacity: opts.opacity,
    driftRate: opts.driftRate,
  };
}

export function createParticles(): SpaceModule {
  const budget = getBudget();

  const fields: Field[] = [
    // Between the glass and Miku. Deliberately sparse — this layer is a depth
    // cue, and more than a handful of motes reads as dirt on the screen.
    makeField({
      band: "foreground",
      count: budget.particlesFore,
      zNear: 0.8,
      zFar: 0.15,
      spread: 3.2,
      yMin: 0.1,
      yMax: 2.4,
      color: 0x8fa3c8,
      size: 2.2,
      opacity: 0.30,
      driftRate: 0.10,
      seed: 11,
      name: "space-particles-fore",
    }),
    // The volume she is standing in.
    makeField({
      band: "near",
      count: budget.particlesNear,
      zNear: -0.5,
      zFar: -26,
      spread: 22,
      yMin: 0.05,
      yMax: 7,
      color: 0x8fa3c8,
      size: 2.4,
      opacity: 0.30, // near volume: the darkest/largest of the three
      driftRate: 0.55,
      seed: 22,
      name: "space-particles-near",
    }),
    makeField({
      band: "far",
      count: budget.particlesFar,
      zNear: -26,
      zFar: -62,
      spread: 60,
      yMin: 0.05,
      yMax: 16,
      color: 0x8fa3c8,
      size: 1.2,
      opacity: 0.11, // far volume: markedly smaller AND fainter — aerial perspective
      driftRate: 1.1,
      seed: 33,
      name: "space-particles-far",
    }),
    // Distant lights: the far wall of the space is not a wall, it's a scatter of
    // points too small to resolve. Nearly static — motion at this depth would
    // give the distance away.
    makeField({
      band: "infinite",
      count: budget.distantLights,
      zNear: -62,
      zFar: -94,
      spread: 130,
      yMin: -2,
      yMax: 34,
      color: 0x8fa3c8,
      size: 1.5,
      opacity: 0.5,
      driftRate: 0.12,
      seed: 44,
      name: "space-particles-deep",
    }),
  ];

  let drift = 0;

  return {
    objects: fields.map((f) => ({ band: f.band, object: f.points })),

    update(dt: number, elapsed: number, mood: SpaceMood) {
      // Thinking/speaking accelerates the flow toward the vanishing point — the
      // "data streaming away" read, done with a clock rather than new geometry.
      drift += dt * (1 + 5.5 * mood.stream);
      const gain = SPACE.particles * SPACE.emissiveGain * (0.7 + 0.5 * mood.energy);
      fields.forEach((f) => {
        f.uniforms.uTime.value = elapsed;
        f.uniforms.uDrift.value = drift * f.driftRate;
        f.uniforms.uOpacity.value = f.baseOpacity * gain;
        syncHaze(f.uniforms);
      });
    },

    dispose() {
      fields.forEach((f) => {
        f.geo.dispose();
        f.mat.dispose();
      });
    },
  };
}
