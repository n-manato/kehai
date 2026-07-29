import * as THREE from "three";
import { HAZE_FAR } from "./layers";
import { SPACE } from "./space-config";

/**
 * Atmospheric perspective for the environment — the far end of the space loses
 * contrast and dissolves into the backdrop colour, which is what makes a 95 m
 * corridor read as "infinite" instead of "95 m".
 *
 * Deliberately NOT `scene.fog`. Scene fog applies to every material that opts in,
 * and three-vrm's MToon does — so it would wash out and re-tint Miku, whose
 * shading is the most carefully tuned thing in the viewer. Instead each
 * environment material carries its own depth fade, which cannot touch the
 * character by construction.
 */

export interface HazeUniforms {
  uHazeColor: { value: THREE.Color };
  uHazeFar: { value: number };
  uHazeAmt: { value: number };
  [k: string]: { value: unknown };
}

export function hazeUniforms(): HazeUniforms {
  return {
    uHazeColor: { value: new THREE.Color(SPACE.hazeColor) },
    uHazeFar: { value: HAZE_FAR },
    uHazeAmt: { value: SPACE.haze },
  };
}

/** Push current SPACE settings into a material's haze uniforms. */
export function syncHaze(u: HazeUniforms) {
  u.uHazeColor.value.set(SPACE.hazeColor);
  u.uHazeAmt.value = SPACE.haze;
}

/**
 * Inject the depth fade into a STOCK three material (MeshBasicMaterial, …) so
 * instancing, instance colours and vertex colours keep working — hand-writing a
 * ShaderMaterial would mean re-implementing all of that.
 *
 * Returns the uniforms so the caller can keep them in sync with SPACE.
 */
export function applyHazeToMaterial(mat: THREE.Material, additive = false): HazeUniforms {
  const u = hazeUniforms();
  mat.onBeforeCompile = (shader: { uniforms: Record<string, unknown>; vertexShader: string; fragmentShader: string }) => {
    Object.assign(shader.uniforms, u);
    shader.vertexShader = shader.vertexShader
      .replace("#include <common>", `#include <common>\n${HAZE_PARS}`)
      // project_vertex is where `mvPosition` gets its final value (after
      // instancing/skinning), so the depth has to be read after it, not before.
      .replace("#include <project_vertex>", `#include <project_vertex>\n${HAZE_VERTEX}`);
    shader.fragmentShader = shader.fragmentShader
      .replace("#include <common>", `#include <common>\n${HAZE_PARS}\n${HAZE_FRAGMENT}`)
      // Last chunk in the chain. Inside the composer three skips tone mapping
      // (it only injects it when drawing straight to the screen), so what we are
      // blending here is still linear — same space the floor shader writes in.
      .replace(
        "#include <dithering_fragment>",
        additive
          // Additive blending is `src.rgb * src.a + dst`, so scaling alpha alone
          // already fades the contribution to nothing. Tinting toward the
          // backdrop colour as well would ADD navy to the frame rather than
          // remove light, and fade it quadratically on top.
          ? `#include <dithering_fragment>
        gl_FragColor.a *= 1.0 - hazeFactor();`
          : `#include <dithering_fragment>
        float spaceHaze = hazeFactor();
        gl_FragColor.rgb = mix(gl_FragColor.rgb, uHazeColor, spaceHaze);
        gl_FragColor.a *= 1.0 - spaceHaze;`,
      );
  };
  // Materials sharing an injection compile to identical source, so they can
  // share a program — without a stable key three treats each as a fresh variant.
  (mat as unknown as { customProgramCacheKey: () => string }).customProgramCacheKey = () =>
    (additive ? "space-haze-add" : "space-haze");
  return u;
}

/** Declarations — put in both shader stages. */
export const HAZE_PARS = /* glsl */ `
varying float vHazeDepth;
`;

/** Vertex stage: record the view-space depth. Needs `mvPosition` in scope. */
export const HAZE_VERTEX = /* glsl */ `
vHazeDepth = -mvPosition.z;
`;

/**
 * Fragment stage helper. Returns 0 near the camera, 1 at uHazeFar.
 * Squared falloff so the near/mid field stays crisp and only the deep end melts —
 * a linear ramp greys out the whole scene and kills the near-vs-far contrast that
 * the depth illusion depends on.
 */
export const HAZE_FRAGMENT = /* glsl */ `
uniform vec3 uHazeColor;
uniform float uHazeFar;
uniform float uHazeAmt;

float hazeFactor() {
  float t = clamp(vHazeDepth / uHazeFar, 0.0, 1.0);
  return clamp(t * t * uHazeAmt, 0.0, 1.0);
}
`;
