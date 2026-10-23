import * as THREE from "three";
import { SPACE } from "./space-config";
import type { FlowState } from "./flow-state";
import type { SpaceModule } from "./types";

/**
 * The data flow: a cloud of small square "bits" around her that moves with what
 * she is doing — the AI shown as process, with no text.
 *
 *   idle      — the cloud sways slowly around her, each bit bobbing on its own
 *   listening — the sway quickens and the cloud tightens a little
 *   thinking  — bits stream INTO her chest and vanish there
 *   speaking  — bits stream OUT of her chest, deeper in colour with the voice
 *
 * Written for the PROTO panel, a transparent LCD on a light box: WHITE is
 * see-through and only dark or saturated colour holds. So the bits are saturated
 * blue on a white field, and depth comes from the panel itself — the farther a bit
 * is, the smaller and paler it gets, until it dissolves into the real box behind
 * the glass. That is aerial perspective for free, and unlike a drawn room it never
 * contradicts the depth the viewer can actually see.
 *
 * All motion is in the vertex shader (three blended paths per bit, weighted by
 * the eased state), so the CPU never touches the buffers.
 */

const VERT = /* glsl */ `
attribute float aPhase;
attribute float aSpeed;
attribute float aSize;
uniform float uTime;
uniform float uIdle;
uniform float uListen;
uniform float uThink;
uniform float uSpeak;
uniform float uVoice;
uniform vec3 uCore;
uniform float uSize;
uniform float uPixelRatio;
varying float vAlpha;
varying float vDepth;

vec3 rotY(vec3 p, float a) {
  float c = cos(a), s = sin(a);
  return vec3(c * p.x + s * p.z, p.y, -s * p.x + c * p.z);
}

void main() {
  // Home: the cloud swaying around her (livelier while listening), each bit bobbing
  // on its own phase so it never reads as one rigid object. A SWAY, not a turn: a
  // full rotation carried the bits placed behind her round to the front, where
  // they sat on her body like specks on the screen.
  float turn = sin(uTime * (0.08 + 0.1 * uListen) * aSpeed + aPhase) * 0.3;
  vec3 home = rotY(position, turn);
  home.y += sin(uTime * 0.4 * aSpeed + aPhase) * 0.04;
  home = mix(home, vec3(uCore.x, home.y, uCore.z), 0.12 * uListen);

  // The far end of a stream: out to HER side of the bit's home, in front of her.
  // Streams between the chest core and the homes themselves (mostly behind her)
  // ran straight through her torso and were hidden for nearly their whole length.
  float side = home.x < 0.0 ? -1.0 : 1.0;
  vec3 fan = uCore + vec3(
    side * (0.7 + abs(home.x) * 0.6),
    (home.y - uCore.y) * 0.7,
    0.15 + 0.3 * fract(aPhase * 3.7));

  // Only ~30% of the bits take part in a stream; the rest keep drifting at home.
  // Calm on purpose: every bit streaming at 0.3-0.7 laps/s, sped up further by
  // the voice, read as a busy flicker ("動きが速すぎてうざい"). The voice now
  // only deepens the colour; the pace stays fixed (~6-10 s a lap).
  float part = step(fract(aPhase * 5.3), 0.3);

  // Thinking: drawn in from the sides, accelerating, gone at the core.
  float fi = fract(uTime * 0.11 * aSpeed + aPhase * 0.159);
  vec3 pIn = mix(fan, uCore, fi * fi);
  float aIn = smoothstep(0.0, 0.12, fi) * (1.0 - smoothstep(0.75, 1.0, fi));

  // Speaking: sent out of the core to the sides, decelerating.
  float fo = fract(uTime * 0.11 * aSpeed + aPhase * 0.271);
  vec3 pOut = mix(uCore, fan, sqrt(fo));
  float aOut = smoothstep(0.0, 0.06, fo) * (1.0 - smoothstep(0.7, 1.0, fo));

  float wThink = uThink * part;
  float wSpeak = uSpeak * part;
  float wHome = 1.0 - wThink - wSpeak;
  vec3 p = home * wHome + pIn * wThink + pOut * wSpeak;
  vAlpha = wHome + aIn * wThink + aOut * wSpeak * (0.8 + 0.6 * uVoice);

  vec4 mv = modelViewMatrix * vec4(p, 1.0);
  vDepth = -mv.z;
  gl_PointSize = uSize * aSize * uPixelRatio * (3.0 / max(vDepth, 0.3));
  gl_Position = projectionMatrix * mv;
}
`;

const FRAG = /* glsl */ `
uniform vec3 uColor;
uniform vec3 uFarColor;
uniform float uOpacity;
uniform float uCamDist;
varying float vAlpha;
varying float vDepth;

void main() {
  // A square bit, soft-edged: digital, not dust.
  vec2 c = abs(gl_PointCoord - 0.5);
  float d = max(c.x, c.y);
  float m = 1.0 - smoothstep(0.32, 0.5, d);
  // Behind her it pales toward the field (= see-through on the panel).
  float far = clamp((vDepth - uCamDist) / 6.0, 0.0, 1.0);
  float a = m * vAlpha * uOpacity * (1.0 - 0.55 * far);
  if (a < 0.01) discard;
  gl_FragColor = vec4(mix(uColor, uFarColor, far), a);
}
`;

export function createDataFlow(state: FlowState, camera: THREE.Camera): SpaceModule {
  let s = 20261022 >>> 0;
  const rnd = () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };
  const n = Math.max(1, Math.round(SPACE.flowCount));
  const position = new Float32Array(n * 3);
  const aPhase = new Float32Array(n);
  const aSpeed = new Float32Array(n);
  const aSize = new Float32Array(n);
  for (let i = 0; i < n; i += 1) {
    // A thick shell around her: never inside the body column (r < 0.45), mostly
    // beside and behind her so the bits frame her rather than cover her face.
    const r = 0.45 + Math.pow(rnd(), 0.7) * 2.2;
    const a = (rnd() * 1.3 - 0.65) * Math.PI + Math.PI; // centred BEHIND (-Z), +-117 deg
    let x = Math.sin(a) * r;
    const z = Math.cos(a) * r * 1.3;
    // In front of her only off to the side: a bit between the viewer and her body
    // reads as a speck on the screen, one beside her as something in the room.
    if (z > 0 && Math.abs(x) < 0.6) x = Math.sign(x || 1) * (0.6 + rnd() * 0.4);
    position[i * 3] = x;
    position[i * 3 + 1] = 0.05 + rnd() * 2.5;
    position[i * 3 + 2] = z;
    aPhase[i] = rnd() * Math.PI * 2;
    aSpeed[i] = 0.6 + rnd() * 0.8;
    aSize[i] = 0.5 + rnd() * rnd() * 1.6;
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(position, 3));
  geo.setAttribute("aPhase", new THREE.BufferAttribute(aPhase, 1));
  geo.setAttribute("aSpeed", new THREE.BufferAttribute(aSpeed, 1));
  geo.setAttribute("aSize", new THREE.BufferAttribute(aSize, 1));

  const uniforms = {
    uTime: { value: 0 },
    uIdle: { value: 1 },
    uListen: { value: 0 },
    uThink: { value: 0 },
    uSpeak: { value: 0 },
    uVoice: { value: 0 },
    uCore: { value: new THREE.Vector3(0, SPACE.flowCoreY, SPACE.flowCoreZ) },
    uSize: { value: SPACE.flowSize },
    uPixelRatio: { value: Math.min(window.devicePixelRatio || 1, 2) },
    uColor: { value: new THREE.Color(SPACE.flowColor) },
    uFarColor: { value: new THREE.Color(SPACE.flowFarColor) },
    uOpacity: { value: SPACE.flow },
    uCamDist: { value: 3 },
  };
  const mat = new THREE.ShaderMaterial({
    uniforms,
    vertexShader: VERT,
    fragmentShader: FRAG,
    transparent: true,
    depthWrite: false,
    blending: THREE.NormalBlending,
    toneMapped: false,
  });
  const points = new THREE.Points(geo, mat);
  points.frustumCulled = false; // moved in the shader
  points.name = "space-data-flow";

  return {
    objects: [{ band: "character", object: points }],

    update(_dt, elapsed) {
      const w = state.w;
      uniforms.uTime.value = elapsed;
      uniforms.uIdle.value = w.idle;
      uniforms.uListen.value = w.listen;
      uniforms.uThink.value = w.think;
      uniforms.uSpeak.value = w.speak;
      uniforms.uVoice.value = Math.min(1, w.voice * 3);
      uniforms.uCore.value.set(0, SPACE.flowCoreY, SPACE.flowCoreZ);
      uniforms.uSize.value = SPACE.flowSize;
      uniforms.uOpacity.value = SPACE.flow;
      uniforms.uColor.value.set(SPACE.flowColor);
      uniforms.uFarColor.value.set(SPACE.flowFarColor);
      uniforms.uCamDist.value = camera.position.length();
      points.visible = SPACE.flow > 0;
    },

    dispose() {
      geo.dispose();
      mat.dispose();
    },
  };
}
