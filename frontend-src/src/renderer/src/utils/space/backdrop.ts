import * as THREE from "three";
import { SPACE } from "./space-config";
import type { SpaceModule } from "./types";

/**
 * The white field, as scene geometry rather than as a CSS colour.
 *
 * Setting `container.style.background` looked right and did nothing: the navy
 * was never coming from CSS. It is produced inside the composer by the grade
 * pass, which lifts blacks toward blue-violet as part of the character's look —
 * so an empty scene grades to navy no matter what colour sits behind the canvas.
 * The character's effects are off-limits, so the fix is to give the grade
 * something white to act on instead of nothing.
 *
 * A sphere rather than a plane because the camera drifts; a plane sized for one
 * viewpoint develops edges at another. `BackSide` so we are inside it, and it is
 * on the space layer, which the viewer's ink-line pre-pass already excludes —
 * otherwise the outline detector would draw a border around the whole world.
 *
 * It writes depth on purpose. The viewer's notes record a screen-space GTAO
 * artefact — a straight horizontal colour step — on pixels that were pure
 * background, caused by position reconstruction failing where there is no depth.
 * Real geometry gives every pixel a valid depth, which removes the condition
 * that caused it.
 */
export function createBackdrop(): SpaceModule {
  const geo = new THREE.SphereGeometry(94, 24, 16);
  const mat = new THREE.MeshBasicMaterial({
    color: new THREE.Color(SPACE.backdropColor),
    side: THREE.BackSide,
    // Tone mapping would pull a pure white down toward grey before the grade
    // ever sees it; skipping it keeps the field as bright as it was authored.
    toneMapped: false,
    fog: false,
  });
  const mesh = new THREE.Mesh(geo, mat);
  mesh.frustumCulled = false;
  // First thing drawn, so everything else composites over it.
  mesh.renderOrder = -1000;
  mesh.name = "space-backdrop";

  return {
    objects: [{ band: "infinite", object: mesh }],

    update() {
      mesh.visible = !!SPACE.backdrop;
      mat.color.set(SPACE.scene === "dataflow" ? SPACE.flowBackdrop : SPACE.backdropColor);
    },

    dispose() {
      geo.dispose();
      mat.dispose();
    },
  };
}
