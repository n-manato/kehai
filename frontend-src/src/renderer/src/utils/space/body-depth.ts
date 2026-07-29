import * as THREE from "three";
import { SPACE } from "./space-config";

/**
 * Slow depth motion of the CHARACTER herself.
 *
 * Every previous attempt at depth on this panel built a space around her —
 * floor, rails, portal rings — and each one made things worse, because a drawn
 * room is a flat pattern that contradicts the real depth visible behind the
 * glass. With all of that gone the room can no longer state the depth, so she
 * has to.
 *
 * Why this works where drawn geometry does not: a still photograph can imitate
 * perspective, haze and even a shadow, but it cannot change a subject's angular
 * size. When she drifts a hand-span toward the viewer and back, her silhouette
 * grows and shrinks by a couple of percent, her near arm grows faster than her
 * far shoulder, and the viewer's visual system resolves that as one object
 * moving through depth. It needs nothing else in the frame to compare against
 * — which is exactly the situation here.
 *
 * It also stacks with the camera drift instead of cancelling it: the camera
 * moves the viewpoint, this moves the subject, and the two run on incommensurate
 * periods so the relative motion never settles into a loop the eye can predict.
 *
 * Kept deliberately small. Past roughly 0.2 m she reads as walking at the
 * viewer, which is a different (and much more demanding) piece of staging.
 */
export interface BodyDepth {
  /** Call once per frame AFTER the motion player has written the pose. */
  update(dt: number, root: THREE.Object3D): void;
}

export function createBodyDepth(): BodyDepth {
  let t = 0;
  let baseZ = Number.NaN; // NaN until the first frame adopts the authored Z

  return {
    update(dt, root) {
      if (!root) return;
      // Adopt whatever the framing/motion code placed her at as the centre, so
      // this never fights those systems — it only ever adds an offset.
      if (Number.isNaN(baseZ)) baseZ = root.position.z;
      const centre = baseZ;
      const amp = SPACE.bodyDepth;
      if (amp <= 0.0001) {
        root.position.z = centre; // parked back at the authored position
        return;
      }
      t += dt;
      const w = (Math.PI * 2) / Math.max(SPACE.bodyDepthPeriod, 1);
      // Two incommensurate components: a slow travel plus a much slower, much
      // smaller sway. A single sine is periodic enough that the eye locks onto
      // the turnaround points and starts reading it as a mechanism.
      root.position.z =
        centre + Math.sin(t * w) * amp + Math.sin(t * w * 0.43) * amp * 0.35;
    },
  };
}
