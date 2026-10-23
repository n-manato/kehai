import * as THREE from "three";
import { Reflector } from "three/examples/jsm/objects/Reflector.js";
import { FLOOR_Y } from "./layers";
import { SPACE } from "./space-config";
import type { SpaceModule } from "./types";

/**
 * Her faint reflection in a polished floor, fading out away from her feet.
 *
 * A reflection is what makes a floor read as a SURFACE you could walk back into,
 * rather than the bottom edge of a picture: it states the ground plane and her
 * place on it at once. Kept faint — on the PROTO panel (white = see-through) only
 * the darker parts of the reflection hold anyway.
 *
 * Cost control: the mirror renders at a third of the drawing buffer, and only
 * once per frame. Reflector re-renders the scene in onBeforeRender, which fires
 * on EVERY render that draws it — including the passes that redraw the scene with
 * one override material (TAA velocity, GTAO, depth of field). Those would both
 * waste a full scene render each and overwrite the mirror with velocity colours,
 * so the hook is skipped while an override material is set or once already done
 * this frame. The mirror's camera sees layer 0 only (her and the floor), so the
 * 0/1 field is not reflected.
 */

const VERT = /* glsl */ `
uniform mat4 textureMatrix;
varying vec4 vUv;
varying vec3 vWorld;
void main() {
  vUv = textureMatrix * vec4(position, 1.0);
  vec4 w = modelMatrix * vec4(position, 1.0);
  vWorld = w.xyz;
  gl_Position = projectionMatrix * viewMatrix * w;
}
`;

const FRAG = /* glsl */ `
uniform vec3 color;
uniform sampler2D tDiffuse;
uniform float uOpacity;
uniform float uRadius;
varying vec4 vUv;
varying vec3 vWorld;
void main() {
  vec4 base = texture2DProj(tDiffuse, vUv);
  float r = length(vWorld.xz) / uRadius;
  // In FRONT of her feet only. Behind her, the floor mirrors the backs of her legs,
  // which read as a grey shadow standing behind her ("後ろの影いらない").
  float fade = (1.0 - smoothstep(0.15, 1.0, r)) * smoothstep(-0.12, 0.08, vWorld.z);
  float a = base.a * fade * uOpacity;
  if (a < 0.004) discard;
  gl_FragColor = vec4(base.rgb * color, a);
}
`;

export function createFloorReflection(renderer: THREE.WebGLRenderer): SpaceModule {
  const size = new THREE.Vector2();
  renderer.getDrawingBufferSize(size);
  const geo = new THREE.PlaneGeometry(6, 6);
  const mirror = new Reflector(geo, {
    textureWidth: Math.max(256, Math.round(size.x / 3)),
    textureHeight: Math.max(256, Math.round(size.y / 3)),
    clipBias: 0.003,
    multisample: 0,
    shader: {
      name: "KehaiFloorReflection",
      uniforms: {
        color: { value: new THREE.Color(0xffffff) },
        tDiffuse: { value: null },
        textureMatrix: { value: new THREE.Matrix4() },
        uOpacity: { value: SPACE.reflection },
        uRadius: { value: 1.6 },
      },
      vertexShader: VERT,
      fragmentShader: FRAG,
    },
  });
  mirror.rotation.x = -Math.PI / 2;
  mirror.position.y = FLOOR_Y + 0.002;
  mirror.renderOrder = -7;
  mirror.name = "space-floor-reflection";
  // Not a floor to anything else: kept out of GTAO's depth (vrm-viewer.tsx), which
  // otherwise shaded her shins as if standing against a real surface.
  mirror.userData.hideFromOverrides = true;
  const mat = mirror.material as THREE.ShaderMaterial;
  mat.transparent = true;
  mat.depthWrite = false;

  const render = mirror.onBeforeRender;
  let frame = -1;
  let done = -1;
  mirror.onBeforeRender = function guarded(r, scene, camera, geometry, material, group) {
    if ((scene as THREE.Scene).overrideMaterial || done === frame) return;
    done = frame;
    render.call(this, r, scene, camera, geometry, material, group);
  };

  return {
    objects: [{ band: "character", object: mirror }],

    update() {
      frame += 1;
      mat.uniforms.uOpacity.value = SPACE.reflection;
      mirror.visible = SPACE.reflection > 0;
    },

    dispose() {
      geo.dispose();
      mirror.dispose();
    },
  };
}
