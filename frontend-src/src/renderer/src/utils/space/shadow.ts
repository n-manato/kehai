import * as THREE from "three";
import { SPACE } from "./space-config";
import type { SpaceModule } from "./types";

/**
 * Soft contact shadow, stretching from under her feet toward the viewer and off
 * the bottom of the frame.
 *
 * Worth noting why this is possible at all now: on the black-field version
 * darkness was undrawable — a shadow is an absence of light, and on black an
 * absence of light is nothing. Against white, dark ink is the most visible mark
 * available, so the strongest grounding cue in the book came back.
 *
 * Running it off the bottom edge is deliberate. A shadow that stops inside the
 * frame draws its own outline and says "this is where the floor ends"; one that
 * leaves the frame says the floor carries on past the panel, toward the viewer.
 *
 * Written as a shader rather than a gradient texture: the falloff has to be
 * ANISOTROPIC — tight behind her, long toward the viewer — and getting that from
 * a texture means fighting UV orientation on a rotated plane for a result that
 * is then fixed. Here the shape is computed from world position, so `reach` and
 * `softness` stay live knobs.
 */

const SHADOW_VERT = /* glsl */ `
varying vec3 vWorld;
void main() {
  vec4 w = modelMatrix * vec4(position, 1.0);
  vWorld = w.xyz;
  gl_Position = projectionMatrix * viewMatrix * w;
}
`;

const SHADOW_FRAG = /* glsl */ `
varying vec3 vWorld;
uniform vec3 uColor;
uniform float uOpacity;
uniform float uRadius;
uniform float uReach;
uniform float uSoft;

void main() {
  float x = vWorld.x / uRadius;
  // +Z is toward the viewer: stretched that way, tight the other, so it reads
  // as a shadow thrown forward rather than a symmetrical pool.
  float z = vWorld.z >= 0.0 ? vWorld.z / uReach : vWorld.z / (uRadius * 0.9);
  float d = sqrt(x * x + z * z);
  // A high exponent is what makes it soft: the falloff spends most of its range
  // near zero, so there is no readable edge anywhere.
  float a = pow(clamp(1.0 - d, 0.0, 1.0), uSoft) * uOpacity;
  if (a <= 0.002) discard;
  gl_FragColor = vec4(uColor, a);
}
`;

export function createShadow(): SpaceModule {
  const uniforms = {
    uColor: { value: new THREE.Color(0x1d2a3f) },
    uOpacity: { value: SPACE.contactShadow },
    uRadius: { value: SPACE.shadowRadius },
    uReach: { value: SPACE.shadowReach },
    uSoft: { value: SPACE.shadowSoft },
  };
  const mat = new THREE.ShaderMaterial({
    uniforms,
    vertexShader: SHADOW_VERT,
    fragmentShader: SHADOW_FRAG,
    transparent: true,
    depthWrite: false,
    side: THREE.DoubleSide,
  });

  // Big enough to hold the longest reach; the shader decides what is actually
  // inked, so an oversized plane costs nothing but two triangles.
  const geo = new THREE.PlaneGeometry(6, 12);
  const mesh = new THREE.Mesh(geo, mat);
  mesh.rotation.x = -Math.PI / 2;
  mesh.position.set(0, 0.004, 2.4); // centred forward, so it runs off the frame
  mesh.renderOrder = -30;
  mesh.frustumCulled = false;
  mesh.name = "space-contact-shadow";

  return {
    objects: [{ band: "character", object: mesh }],

    update() {
      mesh.visible = SPACE.contactShadow > 0.001;
      uniforms.uOpacity.value = SPACE.contactShadow;
      uniforms.uRadius.value = SPACE.shadowRadius;
      uniforms.uReach.value = SPACE.shadowReach;
      uniforms.uSoft.value = SPACE.shadowSoft;
    },

    dispose() {
      geo.dispose();
      mat.dispose();
    },
  };
}
