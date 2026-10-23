import * as THREE from "three";
import { SPACE } from "./space-config";
import type { FlowState } from "./flow-state";
import type { SpaceModule } from "./types";

/**
 * A soft veil of cyan just behind her, between her and the 0/1 field.
 *
 * The photographer's backlight: a lighter zone right behind the subject separates
 * her from the background and puts a gap of air between the two planes. On the
 * PROTO panel (white = see-through) a colour cannot be brighter than the box, so
 * it reads as a faint blue haze behind her rather than as light — still enough to
 * set her apart from the far field.
 */

const VERT = /* glsl */ `
varying vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;

const FRAG = /* glsl */ `
uniform vec3 uColor;
uniform float uOpacity;
varying vec2 vUv;
void main() {
  // Elliptical falloff, taller than wide, densest at her upper body.
  vec2 d = (vUv - vec2(0.5, 0.56)) * vec2(2.0, 1.55);
  float r = length(d);
  float a = pow(clamp(1.0 - r, 0.0, 1.0), 2.2) * uOpacity;
  if (a < 0.003) discard;
  gl_FragColor = vec4(uColor, a);
}
`;

export function createBacklight(state: FlowState): SpaceModule {
  const geo = new THREE.PlaneGeometry(2.2, 2.8);
  const uniforms = {
    uColor: { value: new THREE.Color(SPACE.backlightColor) },
    uOpacity: { value: SPACE.backlight },
  };
  const mat = new THREE.ShaderMaterial({
    uniforms, vertexShader: VERT, fragmentShader: FRAG,
    transparent: true, depthWrite: false, toneMapped: false,
  });
  const mesh = new THREE.Mesh(geo, mat);
  mesh.position.set(0, 1.3, -0.55);
  mesh.renderOrder = -4; // over the far 0/1 field, under her
  mesh.name = "space-backlight";
  let glow = 0;

  return {
    objects: [{ band: "character", object: mesh }],

    update(dt, elapsed) {
      const w = state.w;
      // A slow breath at rest; a touch fuller while she thinks or talks. Eased, so
      // it never pulses with the syllables.
      const target = 0.15 * w.think + 0.2 * Math.min(1, w.voice * 3) * w.speak;
      glow += (target - glow) * Math.min(1, dt * 0.7);
      const breathe = 0.9 + 0.1 * Math.sin(elapsed * 0.5);
      uniforms.uOpacity.value = SPACE.backlight * breathe * (1 + glow);
      uniforms.uColor.value.set(SPACE.backlightColor);
      mesh.visible = SPACE.backlight > 0;
    },

    dispose() {
      geo.dispose();
      mat.dispose();
    },
  };
}
