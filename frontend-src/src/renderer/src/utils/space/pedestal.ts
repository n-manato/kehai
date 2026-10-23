import * as THREE from "three";
import { FLOOR_Y } from "./layers";
import { SPACE } from "./space-config";
import type { FlowState } from "./flow-state";
import type { SpaceModule } from "./types";

/**
 * The projector base she stands on: concentric rings laid flat on the floor.
 *
 * A ring on the ground is drawn in perspective — an ellipse — so it states the
 * floor plane and her position in depth without drawing a room. It sits in the
 * contact shadow (shadow.ts), which on the PROTO panel is the strongest anchor
 * there is: white is see-through there, so a dark patch under her feet is what
 * makes her stand ON the box floor rather than float on the glass.
 *
 *   outer ring — solid, breathes slowly; pulses with the voice while she talks
 *   dash ring  — segmented, turns; fast while thinking, slow otherwise
 *   pool       — a faint saturated tint on the floor inside the rings
 */

const RING_VERT = /* glsl */ `
varying vec2 vLocal;
void main() {
  vLocal = position.xy;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;

const DASH_FRAG = /* glsl */ `
uniform vec3 uColor;
uniform float uOpacity;
uniform float uSpin;
uniform float uDashes;
varying vec2 vLocal;
void main() {
  float ang = atan(vLocal.y, vLocal.x) / 6.2831853 + 0.5;
  float f = fract(ang * uDashes - uSpin);
  float on = smoothstep(0.0, 0.04, f) * (1.0 - smoothstep(0.52, 0.56, f));
  float a = on * uOpacity;
  if (a < 0.01) discard;
  gl_FragColor = vec4(uColor, a);
}
`;

const POOL_FRAG = /* glsl */ `
uniform vec3 uColor;
uniform float uOpacity;
uniform float uRadius;
varying vec2 vLocal;
void main() {
  float r = length(vLocal) / uRadius;
  float a = pow(clamp(1.0 - r, 0.0, 1.0), 1.6) * uOpacity;
  if (a < 0.004) discard;
  gl_FragColor = vec4(uColor, a);
}
`;

export function createPedestal(state: FlowState): SpaceModule {
  const group = new THREE.Group();
  group.name = "space-pedestal";
  group.rotation.x = -Math.PI / 2; // lay the XY-plane rings flat on the floor
  group.position.y = FLOOR_Y + 0.004;

  const color = new THREE.Color(SPACE.pedestalColor);
  const outerGeo = new THREE.RingGeometry(0.52, 0.535, 160);
  const outerMat = new THREE.MeshBasicMaterial({
    color, transparent: true, opacity: 0.8, depthWrite: false, toneMapped: false,
  });
  const outer = new THREE.Mesh(outerGeo, outerMat);

  const dashGeo = new THREE.RingGeometry(0.59, 0.61, 160);
  const dashU = {
    uColor: { value: color.clone() },
    uOpacity: { value: 0.6 },
    uSpin: { value: 0 },
    uDashes: { value: 36 },
  };
  const dashMat = new THREE.ShaderMaterial({
    uniforms: dashU, vertexShader: RING_VERT, fragmentShader: DASH_FRAG,
    transparent: true, depthWrite: false, toneMapped: false,
  });
  const dash = new THREE.Mesh(dashGeo, dashMat);

  const poolGeo = new THREE.CircleGeometry(0.62, 96);
  const poolU = {
    uColor: { value: color.clone() },
    uOpacity: { value: 0.16 },
    uRadius: { value: 0.62 },
  };
  const poolMat = new THREE.ShaderMaterial({
    uniforms: poolU, vertexShader: RING_VERT, fragmentShader: POOL_FRAG,
    transparent: true, depthWrite: false, toneMapped: false,
  });
  const pool = new THREE.Mesh(poolGeo, poolMat);
  pool.position.z = -0.001; // under the rings

  group.add(pool, outer, dash);
  [pool, outer, dash].forEach((m) => { m.renderOrder = -5; });

  let spin = 0;
  return {
    objects: [{ band: "character", object: group }],

    update(dt, elapsed) {
      const w = state.w;
      const k = SPACE.pedestal;
      group.visible = k > 0;
      const voice = Math.min(1, w.voice * 3);
      // Breathing at rest, a beat with the voice while talking.
      const breathe = 0.5 + 0.5 * Math.sin(elapsed * 0.9);
      outerMat.opacity = k * (0.55 + 0.2 * breathe * w.idle + 0.35 * voice * w.speak + 0.15 * w.listen);
      outer.scale.setScalar(1 + 0.04 * voice * w.speak);
      spin += dt * (0.02 + 0.12 * w.listen + 0.35 * w.think + 0.06 * w.speak);
      dashU.uSpin.value = spin;
      dashU.uOpacity.value = k * (0.45 + 0.3 * w.think + 0.2 * w.listen);
      poolU.uOpacity.value = k * (0.12 + 0.1 * voice * w.speak + 0.05 * w.think);
      color.set(SPACE.pedestalColor);
      outerMat.color.copy(color);
      dashU.uColor.value.copy(color);
      poolU.uColor.value.copy(color);
    },

    dispose() {
      outerGeo.dispose(); outerMat.dispose();
      dashGeo.dispose(); dashMat.dispose();
      poolGeo.dispose(); poolMat.dispose();
    },
  };
}
